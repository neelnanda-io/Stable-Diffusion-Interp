# %%
"""
# W3D5 - Part 2 - Stable Diffusion

## Setting up your instance

In order to download the stable-diffusion models from HuggingFace, there is some setup required:

1. Make a [HuggingFace account](https://huggingface.co/join) and confirm your email address.
1. Visit [here](https://huggingface.co/CompVis/stable-diffusion-v1-4) and click `yes` to the terms and conditions (after thoroughly reading them, of course) and then click `access repository`.
1. Generate a [HuggingFace token](https://huggingface.co/settings/tokens) with a `read` role.
1. Run `huggingface-cli login` in your VSCode terminal and paste the token you generated above (ignore the warning text). This will allow the Python module to download the pretrained models we will be using.

You should now be able to load the pretrained models in this notebook.

## Introducing the model
git 
Before moving on to integrate CLIP into the Stable Diffusion (SD) model, it's worth briefly reviewing what we've built in Part 1. CLIP provides a text encoder and image encoder that are trained together to minimize contrastive loss, and therefore allows for embedding arbitrary text sequences in a latent space that has some relevance to images.

Now, we will introduce the Stable Diffusion model, a state-of-the-art architecture that integrates the text encoder from CLIP (although other text encoders can be used) into a modified diffusion model which is similar to your work from yesterday, W3D4. The primary differences between an "ordinary" diffusion model and the Stable Diffusion model:

* Text encoding is done using a frozen CLIP text encoder. By frozen, we mean the encoder was pretrained separately using the contrastive loss as described yesterday, and not modified at all during the training of SD. The vision transformer part of CLIP is not used in SD.
* U-Net operates in a latent space which has a lower spatial dimensionality than the pixel space of the input. The schematic below describes "LDM-8" which means that the spatial dimension is shrunk by a factor of 8 in width and height. More downsampling makes everything faster, but reduces perceptual quality.
* The encoding to and decoding from the latent space are done using a Variational Autoencoder (VAE), which is trained on the reconstruction error after compressing images into the latent space and then decompressing them again. At inference time, we have no need of the encoder portion because we start with random latents, not pixels. We only need to make one call to the VAE decoder at the very end to turn our latents back into pixels.

## Schematic

<p align="center">
    <img src="w3d5_stablediff_schematic.png" width="500"/><br>
    Source: <a href="https://huggingface.co/blog/stable_diffusion">https://huggingface.co/blog/stable_diffusion</a>
</p>

## A final product

Before getting into it, here's a *very rough* example of the sort of interpolation-based animations that you will (hopefully) be generating by the end of the day with only a few minutes of runtime! Ideally, yours will be smoother and more creative given longer inference time to generate more frames. :)

<p align="center">
    <img src="w3d5_stable_diffusion_animation.gif" width="500"/><br/>
</p>

## Table of Contents

- [Setting up your instance](#setting-up-your-instance)
- [Introducing the model](#introducing-the-model)
- [Schematic](#schematic)
- [A final product](#a-final-product)
- [References (optional) for Part 2](#references-optional-for-part-)
- [Preparation](#preparation)
- [Text encoder](#text-encoder)
- [Getting pretrained models](#getting-pretrained-models)
- [Tokenization](#tokenization)
- [Assembling the inference pipeline](#assembling-the-inference-pipeline)
- [Trying it out!](#trying-it-out)
- [Implementing interpolation](#implementing-interpolation)
- [Prompt Interpolation](#prompt-interpolation)
    - [Saving a GIF](#saving-a-gif)
- [Bonus](#bonus)
    - [Speeding up interpolation](#speeding-up-interpolation)
    - [Multiple prompt image generation](#multiple-prompt-image-generation)
- [Acknowledgements](#acknowledgements)

## References (optional) for Part 2

- [Stable Diffusion Paper](https://arxiv.org/abs/2112.10752)
- [HuggingFace implementation](https://huggingface.co/CompVis/stable-diffusion-v1-4)
- [HuggingFace diffusers module](https://github.com/huggingface/diffusers/tree/71ba8aec55b52a7ba5a1ff1db1265ffdd3c65ea2)
- [Classifier-Free Diffusion Guidance paper](https://arxiv.org/abs/2207.12598)

"""

# %%
"""
# Implementation

Now, we will work on implementing Stable Diffusion according to the above schematic as each of the parts have already been implemented. Furthermore, due to the significant training time of a model, we will use pretrained models from HuggingFace: [CompVis/stable-diffusion-v1-4](https://huggingface.co/CompVis/stable-diffusion-v1-4). This pretrained Stable Diffusion pipeline includes weights for the text encoder, tokenizer, variational autoencoder, and U-Net that have been trained together (again, with a fixed pretrained text encoder).

## Preparation

First, we import the necessary libraries, define a config class, and provide a helper function to assist you in your implementation. This function gets the pretrained models for the tokenizer, text encoder, VAE, and U-Net. As always, it is worth reading through this code to make sure you understand what it does.
"""

# %%
from neel.imports import *
from neel_plotly import *
import torch
torch.set_grad_enabled(False)
# %%

import os
import sys
from collections import OrderedDict
from dataclasses import dataclass
from typing import Callable, Generic, TypeVar, Union, cast

import numpy as np
import torch as t
from diffusers import AutoencoderKL, UNet2DConditionModel
from diffusers.schedulers.scheduling_lms_discrete import LMSDiscreteScheduler
from PIL import Image
from tqdm.auto import tqdm
from transformers import CLIPTextModel, CLIPTokenizer
from transformers.models.clip import modeling_clip
from transformers.tokenization_utils import PreTrainedTokenizer

from w3d5_globals import CLIPConfig, CLIPTextConfig, CLIPVisionConfig
from w3d5_part1_clip_solution import CLIPModel

import plotly.express as px

MAIN = __name__ == "__main__"
DEVICE = t.device("cuda" if t.cuda.is_available() else "cpu")
IS_CI = os.getenv("IS_CI")
if IS_CI:
    sys.exit(0)


@dataclass
class StableDiffusionConfig:  # Stable diffusion config
    """
    Default configuration for Stable Diffusion.

    guidance_scale is used for classifier-free guidance.

    The sched_ parameters are specific to LMSDiscreteScheduler.
    """

    generator: t.Generator
    height: int = 512
    width: int = 512
    num_inference_steps: int = 100  # Number of denoising steps
    guidance_scale: float = 7.5
    sched_beta_start: float = 0.00085
    sched_beta_end: float = 0.012
    sched_beta_schedule: str = "scaled_linear"
    sched_num_train_timesteps: int = 1000
    

    # def __init__(self, generator: t.Generator):
    #     self.generator = generator


T = TypeVar("T", CLIPTokenizer, CLIPTextModel, AutoencoderKL, UNet2DConditionModel)


def load_model(cls: type[T], subfolder: str) -> T:
    model = cls.from_pretrained(
        "CompVis/stable-diffusion-v1-4", subfolder=subfolder, use_auth_token=True
    )
    return cast(T, model)


def load_tokenizer() -> CLIPTokenizer:
    return load_model(CLIPTokenizer, "tokenizer")


def load_text_encoder() -> CLIPTextModel:
    return load_model(CLIPTextModel, "text_encoder").to(DEVICE)


def load_vae() -> AutoencoderKL:
    return load_model(AutoencoderKL, "vae").to(DEVICE)


def load_unet() -> UNet2DConditionModel:
    return load_model(UNet2DConditionModel, "unet").to(DEVICE)


# %%
"""
Now that the pretrained models we will be using are in an accessible format, try printing one of them out to examine its architecture. For example:
"""
if MAIN:
    vae = load_vae()
    print(vae)
    del vae
# %%
"""
## Text encoder

Next, we provide a function to initialize a text encoder model from our implementation of CLIP and load the weights from the pretrained CLIP text encoder. This uses the `load_state_dict` function, which loads the variables from an `OrderedDict` that maps parameter names to their values (typically tensors) into another model with identically named parameters.

In this case, the pretrained `state_dict` of the `CLIPTextModel` instance contains keys prepended with `text_model.`, as the `CLIPTextModel` encapsulates the `CLIPTextTransformer` model, i.e. `type(CLIPTextModel.text_model) == CLIPTextTransformer`. Therefore, to match the input dictionary keys to the parameter names in our `CLIPModel.text_model` class, we need to modify the dictionary from `pretrained.state_dict()` to remove the `text_model.` from each key.

**Note:** You may have noticed that by using the `text_model` member of our `CLIPModel` class from Part 1, we depend on the implementation of `CLIPTextTransformer` imported from `modeling_clip`. This is the same class as that used by the pretrained text model in `CLIPTextModel`. Therefore, this function effectively initializes a `CLIPTextTransformer` class with the pretrained weights just to copy its weights to a second `CLIPTextTransformer` class. However, if we later choose to modify or re-implement the `text_model` in our `CLIPModel` class, maintaining the same parameter names, this function will serve to initialize its weights using the pretrained model weights.
"""
# %%


def clip_text_encoder(pretrained: CLIPTextModel) -> modeling_clip.CLIPTextTransformer:
    pretrained_text_state_dict = OrderedDict(
        [(k[11:], v) for (k, v) in pretrained.state_dict().items()]
    )
    clip_config = CLIPConfig(CLIPVisionConfig(), CLIPTextConfig())
    clip_text_encoder = CLIPModel(clip_config).text_model
    clip_text_encoder.to(DEVICE)
    clip_text_encoder.load_state_dict(pretrained_text_state_dict)
    return clip_text_encoder


# %%
"""
## Getting pretrained models

Now, we're ready to start building the model. There are only a few parts to instantiate and connect into a pipeline that can transform our text prompt into an image. First, we initialize the pretrained models as well as our `CLIPModel.text_model` with pretrained text encoder weights.
"""
# %%



class Pretrained(nn.Module):
    def __init__(self):
        super().__init__()
        self.tokenizer = load_tokenizer()
        self.vae = load_vae()
        self.unet = load_unet()
        pretrained_text_encoder = load_text_encoder()
        self.text_encoder = clip_text_encoder(pretrained_text_encoder)  # type: ignore


if MAIN:
    pretrained = Pretrained()
# %%
"""
## Tokenization

We provide part of a helper function that uses our `PreTrainedTokenizer` to tokenize prompt strings, embed the tokens, and concatenate embeddings for the empty padding token for "classifier-free guidance" ([see paper for details](https://arxiv.org/abs/2207.12598)).

Please implement the `uncond_embeddings` used for classifier-free guidance below, based on the format of `text_embeddings`. Note that `uncond_embeddings` should be of the same shape as `text_embeddings`, and `max_length` has already been assigned for you. Return the concatenated tensor with `uncond_embeddings` and `text_embeddings`, in that order.
"""


def embed_text(pretrained: Pretrained, prompt: list[str]) -> t.Tensor:
    text_input = pretrained.tokenizer(
        prompt,
        padding="max_length",
        max_length=pretrained.tokenizer.model_max_length,
        truncation=True,
        return_tensors="pt",
    )
    # TBD: do with one call to text_encoder for speed?
    # TBD: check if we need attention mask
    text_embeddings = pretrained.text_encoder(text_input.input_ids.to(DEVICE))[0]
    max_length = text_input.input_ids.shape[-1]
    "SOLUTION"
    uncond_input = pretrained.tokenizer(
        [""] * len(prompt),
        padding="max_length",
        max_length=max_length,
        return_tensors="pt",
    )
    uncond_embeddings = pretrained.text_encoder(uncond_input.input_ids.to(DEVICE))[0]
    return t.cat([uncond_embeddings, text_embeddings])


# %%
"""
## Assembling the inference pipeline

Using the scheduler parameters defined in the config at the beginning (`sched_`), instantiate and return the `LMSDiscreteScheduler` in `get_scheduler()`. The scheduler defines the noise schedule during training and/or inference, and will be used later in our inference process.


"""
# %%
def get_scheduler(config: StableDiffusionConfig) -> LMSDiscreteScheduler:
    "SOLUTION"
    return LMSDiscreteScheduler(
        beta_start=config.sched_beta_start,
        beta_end=config.sched_beta_end,
        beta_schedule=config.sched_beta_schedule,
        num_train_timesteps=config.sched_num_train_timesteps,
    )


# %%
r"""
Now, we will implement the missing parts of the inference pipeline in `stable_diffusion_inference()` below. The intended behavior of this function is as follows:

1. Initialize the scheduler, batch_size, multiply latent random Gaussian noise by initial scheduler noise term $\sigma_0$
2. If prompt strings are provided, compute text embeddings
3. In the inference loop, for each timestep defined by the scheduler:
    1. Expand/repeat latent embeddings by 2 for classifier-free guidance, divide the result by $\sqrt{\sigma_i^2 + 1}$ using $\sigma_i$ from the scheduler
    2. Compute concatenated noise prediction using U-Net, feeding in latent input, timestep, and text embeddings
    3. Split concatenated noise prediction $N_c = [N_u, N_t]$ into the unconditional $N_u$ and text $N_t$ portion. You can use the `torch.Tensor.chunk()` function for this.
    4. Compute the total noise prediction $N$ with respect to the guidance scale factor $g$: $N = N_u + g * (N_t - N_u)$
    5. Step to the previous timestep using the scheduler to get the next latent input
4. Rescale latent embedding and decode into image space using VAE decoder
5. Rescale resulting image into RGB space
6. Permute dimensions and convert to `PIL.Image.Image` objects for viewing/saving

Examine the existing implementation, identify which parts are missing, and implement these by referring to the surrounding code and module implementations as necessary.
"""
# %%


def stable_diffusion_inference(
    pretrained: Pretrained,
    config: StableDiffusionConfig,
    prompt: Union[list[str], t.Tensor],
    latents: t.Tensor,
) -> t.Tensor:
    scheduler = get_scheduler(config)
    if isinstance(prompt, list):
        text_embeddings = None
        # [dim=0 idx 0 to n/2 are for uncond_embedding "", n/2 to n are for tokens]
        text_embeddings = embed_text(pretrained, prompt)
    elif isinstance(prompt, t.Tensor):
        text_embeddings = prompt
    scheduler.set_timesteps(config.num_inference_steps)
    latents = latents * scheduler.sigmas[0]
    with t.autocast("cuda"):
        for i, ts in enumerate(tqdm(scheduler.timesteps)):
            # Expand latent input for classifier-free guidance (corresponding to uncond_embeddings and original text_embeddings)
            latent_input = None
            if "SOLUTION":
                latent_input = (
                    latents.repeat((2, 1, 1, 1)) / (scheduler.sigmas[i] ** 2 + 1).sqrt()
                )
            # Predict noise in latent input
            with t.no_grad():
                if "SOLUTION":
                    noise_pred_concat = pretrained.unet(
                        latent_input, ts, encoder_hidden_states=text_embeddings
                    )["sample"]
            if "SOLUTION":
                # Split tensor into uncond and text for classifier-free guidance
                noise_pred_uncond, noise_pred_text = noise_pred_concat.chunk(2)
                # Compute noise using guidance factor to scale influence of text prompt
                noise_pred = noise_pred_uncond + config.guidance_scale * (
                    noise_pred_text - noise_pred_uncond
                )
            # Step to previous timestep (denoising one step)
            latents = scheduler.step(noise_pred, ts, latents)["prev_sample"]  # type: ignore - this is guaranteed by diffusers
    # After inference loop, scale latent output and decode images using VAE
    images = pretrained.vae.decode(latents / 0.18215).sample
    # print(images)
    images = (images * 255 / 2 + 255 / 2).clamp(0, 255)
    images = einops.rearrange(images, "batch rgb height width -> batch height width rgb")
    # images = images.detach().cpu().permute(0, 2, 3, 1).numpy().round().astype("uint8")
    # pil_images = ([Image.fromarray(image) for image in images])
    return images


# %%
"""
## Trying it out!

Finally, after implementing a function to compute our latent noise (provided for you below), we can use our Stable Diffusion inference pipeline by passing in the pretrained models, config, and a prompt of strings.
"""


def latent_sample(config: StableDiffusionConfig, batch_size: int) -> t.Tensor:
    latents = t.randn(
        (
            batch_size,
            cast(int, pretrained.unet.in_channels),
            config.height // 8,
            config.width // 8,
        ),
        generator=config.generator,
    ).to(DEVICE)
    return latents


# [dim=0 idx 0 to n/2 are for uncond_embedding "", n/2 to n are for tokens]

if MAIN:
    SEED = 1
    s = time.time()
    config = StableDiffusionConfig(
        generator=t.manual_seed(SEED),
        num_inference_steps=10
    )  # Pass in seed generator to create the initial latent noise
    prompt = ["A digital illustration of a medieval town", "A digital illustration of a Greek town", "A digital illustration of a modern town", "A digital illustration of a prehistoric town"]
    latents = latent_sample(config, len(prompt))
    images = stable_diffusion_inference(pretrained, config, prompt, latents)
    print(time.time() - s)
    imshow(images, facet_col=0)
    # images[0].save("./w3d5_image.png")

# %%
"""
# Fun with animations!

Finally, let's close off MLAB with some interpolation-based animation fun. The idea is relatively straightforward: the continuous text embedding space from which our Stable Diffusion pipeline generates an image means that we can interpolate between two or more text embeddings to build a set of images generated from the interpolation path. In other words, this means that we can generate a relatively sensible (to the extent that the denoising model works as we expect) image "between" any other two images.

## Implementing interpolation

Given that we've already built our Stable Diffusion inference pipeline, the only thing we need to add is interpolation. Here, we create a function to handle the interpolation of tensors, `interpolate_embeddings()`, and use this function in `run_interpolation()` to loop over each embedded prompt, feeding it into the Stable Diffusion inference pipeline.

Please complete the implementation of `interpolate_embeddings()` as described. However, as this is the last day of MLAB content, if you would prefer to play around with generating images/animations feel free to use the solution code implementation.
"""


def interpolate_embeddings(concat_embeddings: t.Tensor, scale_factor: int) -> t.Tensor:
    """
    Returns a tensor with `scale_factor`-many interpolated tensors between each pair of adjacent
    embeddings.
    concat_embeddings: t.Tensor - Contains uncond_embeddings and text_embeddings concatenated together
    scale_factor: int - Number of interpolations between pairs of points
    out: t.Tensor - shape: [2 * scale_factor * (concat_embeddings.shape[0]/2 - 1), *concat_embeddings.shape[1:]]
    """
    if "SOLUTION":
        uncond_embeddings, text_embeddings = concat_embeddings.chunk(2)
        uncond_embeddings = uncond_embeddings.to(DEVICE)
        text_embeddings = text_embeddings.to(DEVICE)
        num_prompts = text_embeddings.shape[0]
        assert num_prompts > 1
        to_concat = []
        to_concat.append(uncond_embeddings[:-1].repeat(scale_factor, 1, 1))
        for i in range(num_prompts - 1):
            start, end = text_embeddings[i], text_embeddings[i + 1]
            lin = t.linspace(0.0, 1.0, scale_factor).reshape(-1, 1, 1).to(DEVICE)
            to_concat.append(start + lin * (end - start))
        out = t.concat(to_concat)
    assert out.shape == (
        2 * scale_factor * (num_prompts - 1),
        *text_embeddings.shape[1:],
    )
    return out


def run_interpolation(
    prompts: list[str], scale_factor: int, batch_size: int, latent_fn: Callable
) -> list[Image.Image]:
    SEED = 1
    config = StableDiffusionConfig(
        t.manual_seed(SEED)
    )  # Pass in seed generator to create the initial latent noise
    concat_embeddings = embed_text(pretrained, prompts)
    uncond_interp, text_interp = interpolate_embeddings(
        concat_embeddings, scale_factor
    ).chunk(2)
    split_interp_emb = t.split(text_interp, batch_size, dim=0)
    interpolated_images = []
    for t_emb in tqdm(split_interp_emb):
        concat_split = t.concat([uncond_interp[: t_emb.shape[0]], t_emb])
        config = StableDiffusionConfig(t.manual_seed(SEED))
        latents = latent_fn(config, t_emb.shape[0])
        interpolated_images += stable_diffusion_inference(
            pretrained, config, concat_split, latents
        )
    return interpolated_images


# %%
"""
## Prompt Interpolation

Finally, if you've implemented Stable Diffusion correctly, you're ready to play with prompt interpolation. Go ahead and fiddle with the prompts and interpolation scaling factor below, and be sure to share your favorite results on Slack!

`scale_factor` indicates the number of images between each consecutive prompt.
"""
if MAIN:
    prompts = [
        "a photograph of a cat on a lawn",
        "a photograph of a dog on a lawn",
        "a photograph of a bunny on a lawn",
    ]
    interpolated_images = run_interpolation(
        prompts, scale_factor=2, batch_size=1, latent_fn=latent_sample
    )

# %%
"""
## Saving a GIF

Save your list of images as a GIF by running the following:
"""


def save_gif(images: list[Image.Image], filename):
    # Duration argument is in ms, loop=0 specifies looping infinitely
    images[0].save(
        filename, save_all=True, append_images=images[1:], duration=100, loop=0
    )


if MAIN:
    save_gif(interpolated_images, "w3d5_animation1.gif")

# %%
"""
## Speeding up interpolation

Consider how you might speed up the interpolation inference process above. Note that batching multiple prompts (making sure to concatenate their correspondings unconditional embeddings as expected) tends to speed up the per-prompt generation time. However, this also affects the random generation of Gaussian noise fed into the U-Net as the noise is different for each sample, which in practice tends to result in images that don't always "fit" together or play smoothly in an animation. Think about how you can modify the latent noise generation step to batch prompts without affecting the randomness relative to individually feeding prompts into the model, and try implementing this change.

Here, a new function `latent_sample_same()` is created which uses the same inputs as `latent_sample()` and is intended to output the same noise for a batch size of 1. For larger batches, it should use the same noise for each image in the batch. Implement this quick change, looking back at `latent_sample()` if needed, and try testing whether a larger interpolation batch size with this sampling function improves performance on your system. This will depend on your maximum batch size usually constrained by GPU memory size as well as other minor factors.
"""


def latent_sample_same(config: StableDiffusionConfig, batch_size: int) -> t.Tensor:
    if "SOLUTION":
        latents = t.randn(
            (
                1,
                cast(int, pretrained.unet.in_channels),
                config.height // 8,
                config.width // 8,
            ),
            generator=config.generator,
        ).to(DEVICE)
        latents = latents.expand(batch_size, -1, -1, -1)
    return latents


# %%
"""
For example, here is a call to `run_interpolation()` that uses a batch size of 2 and passes in your modified `latent_sample_same()` function to generate random noise.
"""
if MAIN:
    prompts = [
        "a photograph of a cat on a lawn",
        "a photograph of a dog on a lawn",
        "a photograph of a bunny on a lawn",
    ]
    interpolated_images = run_interpolation(
        prompts, scale_factor=2, batch_size=2, latent_fn=latent_sample_same
    )
    save_gif(interpolated_images, "w3d5_animation2.gif")

"""

If you've gotten this far, you're all done with today's content as well as the standard MLAB content. Congratulations!

## Bonus

Here are a few bonus tasks as inspiration. However, feel free to play with the Stable Diffusion model to find your own idea!

### Multiple prompt image generation

What does it mean to combine prompts within a single image? Can this be done by modifying the Stable Diffusion inference process to condition on two or more text embeddings, or for parts of an image to condition on different embeddings?

### Stylistic changes

Try to identify changes in prompts that induce stylistic changes in the resulting image. For example, a painting as opposed to a photograph, or a greyscale photograph as opposed to a color photograph.

## Acknowledgements

- [HuggingFace blog post on Stable Diffusion](https://huggingface.co/blog/stable_diffusion>https://huggingface.co/blog/stable_diffusion), a great resource for introducing the model and implementing an inference pipeline.
"""

# %%
def plot_image(image, **kwargs):
    if len(image.shape) == 4:
        facet_col = 0
    else:
        facet_col = None
    if image.shape[-3] == 3:
        image = einops.rearrange(image, "... rgb height width -> ... height width rgb")
    image = image.clamp(0, 255)
    imshow(image, facet_col=facet_col, **kwargs)

MAGIC_LATENT_SCALE = 0.18215
def encode_and_decode_image(image):
    image = image.squeeze()
    assert image.shape == (3, 512, 512)
    image = image[None]
    latent = pretrained.vae.encode(image).latent_dist.mean # type: ignore
    scaled_latent = latent / MAGIC_LATENT_SCALE
    output = pretrained.vae.decode(scaled_latent).sample.squeeze() # type: ignore
    output_image = to_numpy(output * 255 / 2)
    print(image.shape, output_image.shape)
    plot_image(np.stack([to_numpy(image.squeeze()), output_image]), facet_labels=["original", "recons"])
encode_and_decode_image(image)
# %%
class Toy(nn.Module):
    def __init__(self):
        super().__init__()
        self.W = nn.Parameter(torch.arange(4).float())
    
    def forward(self, x):
        return x @ self.W
toy = Toy()
x = torch.ones(4)
print(toy(x))

handle = toy.register_forward_hook(lambda module, input, output: print(module, input, output))
print(toy(x))
handle.remove()
handle = toy.register_forward_pre_hook(lambda module, input: print(module, input))
print(toy(x))
handle.remove()
# %%
try:
    reset_all_hooks()
except:
    print("Couldn't reset all")
HANDLES = []
def add_hook(module, hook):
    handle = module.register_forward_hook(hook)
    HANDLES.append(handle)
def reset_all_hooks():
    for handle in HANDLES:
        handle.remove()
    HANDLES.clear()
# %%
def show_shapes_hook(module, input, output, name):
    print(name)
    print("\tModule:", module._get_name())
    for i, inp in enumerate(input):
        if isinstance(inp, torch.Tensor):
            print(f"\tInput {i}:", inp.shape, inp.dtype)
        elif isinstance(inp, int) or isinstance(inp, float):
            print(f"\tInput {i}:", inp)
        else:    
            print(f"\tInput {i}:", type(inp))
    if isinstance(output, torch.Tensor):
        print(f"\tOutput:", output.shape, output.dtype)
    else:    
        print(f"\tOutput:", type(output))
reset_all_hooks()
# %%
reset_all_hooks()
add_hook(pretrained, partial(show_shapes_hook, name="pretrained"))
add_hook(pretrained.vae.encoder, partial(show_shapes_hook, name="pretrained.vae.encoder"))
add_hook(pretrained.vae.decoder, partial(show_shapes_hook, name="pretrained.vae.decoder"))
add_hook(pretrained.text_encoder, partial(show_shapes_hook, name="pretrained.text_encoder"))
add_hook(pretrained.unet, partial(show_shapes_hook, name="pretrained.unet"))

# %%

SEED = 1
s = time.time()
config = StableDiffusionConfig(
    generator=t.manual_seed(SEED),
    num_inference_steps=10
)  # Pass in seed generator to create the initial latent noise
prompt = ["A digital illustration of a medieval town", "A digital illustration of a Greek town", "A digital illustration of a modern town", "A digital illustration of a prehistoric town", "A cuddle"]
latents = latent_sample(config, len(prompt))
images = stable_diffusion_inference(pretrained, config, prompt, latents)
print(time.time() - s)
imshow(images, facet_col=0)
# %%
reset_all_hooks()


# %%
