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
    elif isinstance(output, tuple):
        for i, outp in enumerate(output):
            if isinstance(outp, torch.Tensor):
                print(f"\tOutput {i}:", outp.shape, outp.dtype)
            elif isinstance(outp, int) or isinstance(outp, float):
                print(f"\tOutput {i}:", outp)
            else:
                print(f"\tOutput {i}:", type(outp))
    else:
        print(f"\tOutput:", type(output))

# %%
IMAGE_PATH = Path("/workspace/Stable-Diffusion-Interp/images")


def plot_image(image, animation=False, **kwargs):
    image = to_numpy(image)
    if len(image.shape) == 4:
        if animation:
            animation_frame=0
            facet_col = None
        else:
            facet_col = 0
            animation_frame=None
    else:
        facet_col = None
        animation_frame = None
    if image.shape[-3] == 3:
        image = einops.rearrange(image, "... rgb height width -> ... height width rgb")
    image = np.clip(image, 0, 255)
    imshow(image, facet_col=facet_col, animation_frame=animation_frame, **kwargs)


def save_image(tensor, name):
    tensor = tensor.squeeze()
    if tensor.shape[-3] == 3:
        tensor = einops.rearrange(tensor, "rgb h w -> h w rgb")
    tensor = to_numpy(tensor)
    tensor = np.clip(tensor, 0, 255)
    tensor = tensor.round().astype("uint8")
    image = Image.fromarray(tensor)
    if not name.endswith(".png"):
        name = name + ".png"
    image.save(IMAGE_PATH / name)


# save_image(images[0], "test2")


def load_image(name):
    if not name.endswith(".png"):
        name = name + ".png"
    image = Image.open(IMAGE_PATH / name)
    image = np.array(image)
    image = einops.rearrange(image, "h w rgb -> rgb h w")
    image = torch.tensor(image, dtype=torch.float32, device="cuda")
    return image


image = load_image("test")
plot_image(image)

# %%
@dataclass
class StableDiffusionConfig:  # Stable diffusion config
    """
    Default configuration for Stable Diffusion.

    guidance_scale is used for classifier-free guidance.

    The sched_ parameters are specific to LMSDiscreteScheduler.
    """

    seed: int = 1
    height: int = 512
    width: int = 512
    default_num_inference_steps: int = 10  # Number of denoising steps
    guidance_scale: float = 7.5
    sched_beta_start: float = 0.00085
    sched_beta_end: float = 0.012
    sched_beta_schedule: str = "scaled_linear"
    sched_num_train_timesteps: int = 1000

# config = StableDiffusionConfig()


def clip_text_encoder(pretrained: CLIPTextModel) -> modeling_clip.CLIPTextTransformer:
    pretrained_text_state_dict = OrderedDict(
        [(k[11:], v) for (k, v) in pretrained.state_dict().items()]
    )
    clip_config = CLIPConfig(CLIPVisionConfig(), CLIPTextConfig())
    clip_text_encoder = CLIPModel(clip_config).text_model
    clip_text_encoder.to(DEVICE)
    clip_text_encoder.load_state_dict(pretrained_text_state_dict)
    return clip_text_encoder

PAD = 49407
DECODER_SCALE = 0.18215

class StableDiffusion(nn.Module):
    def __init__(self, config: StableDiffusionConfig):
        super().__init__()
        self.tokenizer = load_tokenizer()
        self.vae = load_vae()
        self.unet = load_unet()
        pretrained_text_encoder = load_text_encoder()
        self.text_encoder = clip_text_encoder(pretrained_text_encoder)  # type: ignore
        self.config = config

    def embed_text(self, prompts: str | list[str]) -> t.Tensor:
        if isinstance(prompts, str):
            prompts = [prompts]
        text_input = self.tokenizer(
            prompts,
            padding="max_length",
            max_length=self.tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        )
        # TBD: do with one call to text_encoder for speed?
        # TBD: check if we need attention mask
        text_embeddings = self.text_encoder(text_input.input_ids.to(DEVICE))[0]
        max_length = text_input.input_ids.shape[-1]

        uncond_input = self.tokenizer(
            [""] * len(prompts),
            padding="max_length",
            max_length=max_length,
            return_tensors="pt",
        )
        uncond_embeddings = self.text_encoder(uncond_input.input_ids.to(DEVICE))[0]
        return t.cat([uncond_embeddings, text_embeddings])

    def to_tokens(self, prompts: str | list[str]) -> t.Tensor:
        if isinstance(prompts, str):
            prompts = [prompts]
        text_input = self.tokenizer(
            prompts,
            padding="max_length",
            max_length=self.tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        )
        return text_input.input_ids.to(DEVICE)

    def to_str_tokens(self, prompts: str | list[str] | t.Tensor, remove_pads=True) -> list[str] | list[list[str]]:
        if isinstance(prompts, str):
            prompts = [prompts]
        if isinstance(prompts, list):
            tokens = self.to_tokens(prompts)
        else:
            tokens = prompts

        if len(tokens.shape)==2 and tokens.shape[0] == 1:
            tokens = tokens[0]

        if len(tokens.shape)==2:
            return [self.to_str_tokens(tokens[i], remove_pads) for i in range(tokens.shape[0])]
        else:
            index = (tokens == PAD).nonzero().min()
            tokens = tokens[:index]
            return self.tokenizer.batch_decode(tokens)

    def get_latent_sample(self, batch_size: int, seed: int | None = 1, identical_noise: bool = False) -> t.Tensor:
        if seed is not None:
            gen = t.manual_seed(seed)
        else:
            gen = None
        latents = t.randn(
        (
            batch_size if not identical_noise else 1,
            cast(int, stable_diffusion.unet.in_channels), # 4
            self.config.height // 8, # 64
            self.config.width // 8, # 64
        ), generator=gen).to(DEVICE)

        if identical_noise:
            latents = einops.repeat(latents, "1 ... -> b ...", b=batch_size)

        return latents

    def get_scheduler(self, num_inference_steps: int | None = None) -> LMSDiscreteScheduler:
        scheduler = LMSDiscreteScheduler(
            beta_start=self.config.sched_beta_start,
            beta_end=self.config.sched_beta_end,
            beta_schedule=self.config.sched_beta_schedule,
            num_train_timesteps=self.config.sched_num_train_timesteps,
        )
        if num_inference_steps is None:
            num_inference_steps = self.config.default_num_inference_steps
        scheduler.set_timesteps(num_inference_steps)
        return scheduler

    def denoising_step(
            self,
            latents: t.Tensor,
            text_embeddings: t.Tensor,
            ts: t.Tensor,
            sigma: t.Tensor,
        ) -> tuple[t.Tensor, t.Tensor]:
        latent_input = (
            latents.repeat((2, 1, 1, 1)) / (sigma ** 2 + 1).sqrt()
        )
        noise_pred_concat = stable_diffusion.unet(
            latent_input, ts, text_embeddings
        )["sample"]
        # Split tensor into uncond and text for classifier-free guidance
        noise_pred_uncond, noise_pred_text = noise_pred_concat.chunk(2)
        return noise_pred_uncond, noise_pred_text

    def decode_to_image(self, latents: t.Tensor):
        # After inference loop, scale latent output and decode images using VAE
        images = stable_diffusion.vae.decode(latents / DECODER_SCALE).sample # type: ignore
        images = (images * 255 / 2 + 255 / 2).clamp(0, 255)
        # images = einops.rearrange(
        #     images, "batch rgb height width -> batch height width rgb"
        # )
        return images

    def encode_and_decode_image(self, image: np.ndarray | t.Tensor) -> t.Tensor:
        image = image.squeeze()
        if image.shape != (3, 512, 512):
            assert image.shape == (512, 512, 3)
            image = einops.rearrange(image, "height width rgb -> rgb height width")
        if not isinstance(image, torch.Tensor):
            image = torch.tensor(image)
        image = image.float().cuda()
        image = image[None]
        latent = self.vae.encode(image).latent_dist.mean  # type: ignore
        scaled_latent = latent / DECODER_SCALE
        output = self.vae.decode(scaled_latent).sample.squeeze()  # type: ignore
        output_image = to_numpy(output * 255 / 2)
        print(image.shape, output_image.shape)
        plot_image(
            np.stack([to_numpy(image.squeeze()), output_image]), # type: ignore
            facet_labels=["original", "recons"],
        )
        return t.tensor(np.stack([to_numpy(image.squeeze()), output_image])).to(DEVICE) # type: ignore

    @t.no_grad()
    def generate(
        self,
        prompts: list[str] | str | t.Tensor,
        num_inference_steps: int | None = None,
        latents: t.Tensor | None = None,
        identical_noise: bool = False,
        seed: int | None = 1,
    ):
        if not isinstance(prompts, t.Tensor):
            text_embeddings = self.embed_text(prompts) # type: ignore
        else:
            text_embeddings = prompts

        scheduler = self.get_scheduler(num_inference_steps)

        if latents is None:
            latents = self.get_latent_sample(batch_size=len(text_embeddings)//2, seed=seed, identical_noise=identical_noise)

        latents = latents * scheduler.sigmas[0]
        latent_list = [latents]
        with t.autocast("cuda"):
            for i, ts in enumerate(tqdm(scheduler.timesteps)):
                noise_pred_uncond, noise_pred_text = self.denoising_step(
                    latents,
                    text_embeddings,
                    ts,
                    scheduler.sigmas[i],
                )
                # Compute noise using guidance factor to scale influence of text prompt
                noise_pred = noise_pred_uncond + config.guidance_scale * (
                    noise_pred_text - noise_pred_uncond
                )
                # Step to previous timestep (denoising one step)
                latents = scheduler.step(noise_pred, ts, latents)["prev_sample"] # type: ignore
                latent_list.append(latents)
        
        images = self.decode_to_image(latents)
        
        all_latents = t.stack(latent_list)
        return images, all_latents

if MAIN:
    config = StableDiffusionConfig()
    stable_diffusion = StableDiffusion(config)

# %%
prompts = [
    "A digital illustration of a medieval town",
    "A digital illustration of a Greek town",
    "A digital illustration of a modern town",
    "A digital illustration of a prehistoric town",
]
images, town_latents = stable_diffusion.generate(prompts, 100)
plot_image(images)
if ((images[0] - image).abs().max() < 0.6).item():
    print("Equals the saved image!")
else:
    raise ValueError("Images are not equal!")

# %%
stacked_images = stable_diffusion.encode_and_decode_image(image)

# %%

prompts = [
    "A red dress on a mannequin, stock photography",
    "A black dress on a mannequin, stock photography",
]
num_inference_steps = 25
identical_noise = True
seed = 1
dress_images_base, dress_latents_base = stable_diffusion.generate(prompts, num_inference_steps, identical_noise=identical_noise, seed=seed)

plot_image(dress_images_base)
# %%
prompts = [
    "A red dress on a mannequin, stock photography",
    "A black dress on a mannequin, stock photography",
    "A green dress on a mannequin, stock photography",
    "A blue dress on a mannequin, stock photography",
    "A yellow dress on a mannequin, stock photography",
    "A purple dress on a mannequin, stock photography",
    "A white dress on a mannequin, stock photography",
    "A beige dress on a mannequin, stock photography",
]
num_inference_steps = 25
identical_noise = True
for seed in range(1, 10):
    dress_images_base_temp, dress_latents_base_temp = stable_diffusion.generate(prompts, num_inference_steps, identical_noise=identical_noise, seed=seed)

    plot_image(dress_images_base_temp, title=f"Seed {seed}", facet_labels=[p.split(" ")[1] for p in prompts])
# %%
# Switching experiments
c1 = "red"
c2 = "black"
prompts = [
    f"A {c1} dress on a mannequin, stock photography",
    f"A {c2} dress on a mannequin, stock photography",
]
num_inference_steps = 25
identical_noise = True
seed = 2
if not isinstance(prompts, t.Tensor):
    text_embeddings = stable_diffusion.embed_text(prompts) # type: ignore
else:
    text_embeddings = prompts


latents = stable_diffusion.get_latent_sample(batch_size=len(text_embeddings)//2, seed=seed, identical_noise=identical_noise)

scheduler = stable_diffusion.get_scheduler(num_inference_steps)
latents = latents * scheduler.sigmas[0]
# latent_list = [latents]
original_latents = t.clone(latents)

switched_images = []
switched_latents = []
switch_timesteps = [0, 1, 2, 3, 4, 5, 10, 20, 24]
for switch in switch_timesteps:
    latents = original_latents
    scheduler = stable_diffusion.get_scheduler(num_inference_steps)
    text_embeddings = stable_diffusion.embed_text(prompts)
    switched_latents.append([latents])
    with t.autocast("cuda"):
        for i, ts in enumerate(tqdm(scheduler.timesteps)):
            if i==switch:
                print("Switching at time", i, ts)
                text_embeddings = t.stack([
                    text_embeddings[0],
                    text_embeddings[1],
                    text_embeddings[3],
                    text_embeddings[2],
                ])
            noise_pred_uncond, noise_pred_text = stable_diffusion.denoising_step(
                latents,
                text_embeddings,
                ts,
                scheduler.sigmas[i],
            )
            # Compute noise using guidance factor to scale influence of text prompt
            noise_pred = noise_pred_uncond + config.guidance_scale * (
                noise_pred_text - noise_pred_uncond
            )
            # Step to previous timestep (denoising one step)
            latents = scheduler.step(noise_pred, ts, latents)["prev_sample"] # type: ignore
            switched_latents[-1].append(latents)

    images = stable_diffusion.decode_to_image(latents)
    # plot_image(images, title=f"Switch at {switch}")
    switched_images.append(images)
switched_images = t.stack(switched_images)
switched_latents = t.stack([t.stack(latents) for latents in switched_latents])
plot_image(switched_images[:, 0], facet_labels=[f"Switched at {switch}" for switch in switch_timesteps], title=f"{c1.capitalize()} -> {c2.capitalize()} Seed {seed}", height=300)
plot_image(switched_images[:, 1], facet_labels=[f"Switched at {switch}" for switch in switch_timesteps], title=f"{c2.capitalize()} -> {c1.capitalize()} Seed {seed}", height=300)
# all_latents = t.stack(latent_list)

# plot_image(images)

# %%
c1 = "red"
c2 = "black"
num_inference_steps = 25
identical_noise = True
seed = 1
switch_timesteps=(0, 1, 2, 3, 4, 5, 10, 20, 24)

@cache
def switched_generation(c1, c2, seed=1, num_inference_steps=25, switch_timesteps=(0, 1, 2, 3, 4, 5, 10, 20, 24)):
    prompts = [
        f"A {c1} dress on a mannequin, stock photography",
        f"A {c2} dress on a mannequin, stock photography",
    ]
    identical_noise = True
    if not isinstance(prompts, t.Tensor):
        text_embeddings = stable_diffusion.embed_text(prompts) # type: ignore
    else:
        text_embeddings = prompts


    latents = stable_diffusion.get_latent_sample(batch_size=len(text_embeddings)//2, seed=seed, identical_noise=identical_noise)

    scheduler = stable_diffusion.get_scheduler(num_inference_steps)
    latents = latents * scheduler.sigmas[0]
    # latent_list = [latents]
    original_latents = t.clone(latents)

    switched_images = []
    switched_latents = []
    for switch in tqdm(switch_timesteps):
        latents = original_latents
        scheduler = stable_diffusion.get_scheduler(num_inference_steps)
        text_embeddings = stable_diffusion.embed_text(prompts)
        switched_latents.append([latents])
        with t.autocast("cuda"):
            for i, ts in enumerate((scheduler.timesteps)):
                if i==switch:
                    print("Switching at time", i, ts)
                    text_embeddings = t.stack([
                        text_embeddings[0],
                        text_embeddings[1],
                        text_embeddings[3],
                        text_embeddings[2],
                    ])
                noise_pred_uncond, noise_pred_text = stable_diffusion.denoising_step(
                    latents,
                    text_embeddings,
                    ts,
                    scheduler.sigmas[i],
                )
                # Compute noise using guidance factor to scale influence of text prompt
                noise_pred = noise_pred_uncond + config.guidance_scale * (
                    noise_pred_text - noise_pred_uncond
                )
                # Step to previous timestep (denoising one step)
                latents = scheduler.step(noise_pred, ts, latents)["prev_sample"] # type: ignore
                switched_latents[-1].append(latents)

        images = stable_diffusion.decode_to_image(latents)
        # plot_image(images, title=f"Switch at {switch}")
        switched_images.append(images)
    switched_images = t.stack(switched_images)
    switched_latents = t.stack([t.stack(latents) for latents in switched_latents])
    plot_image(switched_images[:, 0], facet_labels=[f"Switched at {switch}" for switch in switch_timesteps], title=f"{c1.capitalize()} -> {c2.capitalize()} Seed {seed}", height=300)
    plot_image(switched_images[:, 1], facet_labels=[f"Switched at {switch}" for switch in switch_timesteps], title=f"{c2.capitalize()} -> {c1.capitalize()} Seed {seed}", height=300)
    return switched_images, switched_latents

# for seed in tqdm(range(3, 8)):
#     switched_generation(c1, c2, seed, num_inference_steps, tuple(switch_timesteps))


for seed in tqdm(range(1, 4)):
    switched_generation("red", "blue", seed, num_inference_steps, switch_timesteps)
# %%
switched_images, switched_latents = switched_generation(
        "red", "black", 1, num_inference_steps, switch_timesteps
)
# %%
# switched_latents.shape == [switch, step, batch, 4, 64, 64]
switched_latents = switched_latents.reshape(switched_latents.shape[:3]+(-1,))
base_trajectory_1 = switched_latents[0, :, 1]
base_trajectory_2 = switched_latents[0, :, 0]
line((switched_latents - base_trajectory_1[:, None]).norm(dim=-1), facet_col=2, line_labels=switch_timesteps)
line((switched_latents - base_trajectory_2[:, None]).norm(dim=-1), facet_col=2, line_labels=switch_timesteps)
# %%
(switched_latents - base_trajectory_1[:, None]).norm(dim=-1)
# def test(steps=10, seed=1):
#     s = time.time()
#     config = StableDiffusionConfig(
#         generator=t.manual_seed(seed), num_inference_steps=steps
#     )  # Pass in seed generator to create the initial latent noise
#     prompt = [
#         "A digital illustration of a medieval town",
#         "A digital illustration of a Greek town",
#         "A digital illustration of a modern town",
#         "A digital illustration of a prehistoric town",
#         "A cuddle",
#     ]
#     latents = latent_sample(config, len(prompt))
#     images = stable_diffusion_inference(stable_diffusion, config, prompt, latents)
#     print(time.time() - s)
#     imshow(images, facet_col=0)


# # %%
# class Toy(nn.Module):
#     def __init__(self):
#         super().__init__()
#         self.W = nn.Parameter(torch.arange(4).float())

#     def forward(self, x):
#         return x @ self.W


# toy = Toy()
# x = torch.ones(4)
# print(toy(x))

# handle = toy.register_forward_hook(
#     lambda module, input, output: print(module, input, output)
# )
# print(toy(x))
# handle.remove()
# handle = toy.register_forward_pre_hook(lambda module, input: print(module, input))
# print(toy(x))
# handle.remove()
# # %%
# try:
#     reset_all_hooks()
# except:
#     print("Couldn't reset all")
# HANDLES = []


# def add_hook(module, hook):
#     handle = module.register_forward_hook(hook)
#     HANDLES.append(handle)


# def reset_all_hooks():
#     for handle in HANDLES:
#         handle.remove()
#     HANDLES.clear()


# # %%
# def show_shapes_hook(module, input, output, name):
#     print(name)
#     print("\tModule:", module._get_name())
#     for i, inp in enumerate(input):
#         if isinstance(inp, torch.Tensor):
#             print(f"\tInput {i}:", inp.shape, inp.dtype)
#         elif isinstance(inp, int) or isinstance(inp, float):
#             print(f"\tInput {i}:", inp)
#         else:
#             print(f"\tInput {i}:", type(inp))
#     if isinstance(output, torch.Tensor):
#         print(f"\tOutput:", output.shape, output.dtype)
#     elif isinstance(output, tuple):
#         for i, outp in enumerate(output):
#             if isinstance(outp, torch.Tensor):
#                 print(f"\tOutput {i}:", outp.shape, outp.dtype)
#             elif isinstance(outp, int) or isinstance(outp, float):
#                 print(f"\tOutput {i}:", outp)
#             else:
#                 print(f"\tOutput {i}:", type(outp))
#     else:
#         print(f"\tOutput:", type(output))


# reset_all_hooks()
# # %%
# reset_all_hooks()
# add_hook(stable_diffusion.unet.conv_act, partial(show_shapes_hook, name="blah"))
# test(steps=1)
# # print(1/0)
# # %%
# reset_all_hooks()
# add_hook(stable_diffusion, partial(show_shapes_hook, name="pretrained"))
# add_hook(
#     stable_diffusion.vae.encoder, partial(show_shapes_hook, name="pretrained.vae.encoder")
# )
# add_hook(
#     stable_diffusion.vae.decoder, partial(show_shapes_hook, name="pretrained.vae.decoder")
# )
# add_hook(
#     stable_diffusion.text_encoder, partial(show_shapes_hook, name="pretrained.text_encoder")
# )
# add_hook(stable_diffusion.unet, partial(show_shapes_hook, name="pretrained.unet"))

# # %%
# reset_all_hooks()
# for name, mod in stable_diffusion.unet.named_children():
#     name = "pretrained.unet." + name
#     add_hook(mod, partial(show_shapes_hook, name=name))
# test(steps=1)
# reset_all_hooks()
# for name, mod in stable_diffusion.unet.named_modules():
#     name = "pretrained.unet." + name
#     add_hook(mod, partial(show_shapes_hook, name=name))
# test(steps=1)

# # %%

# # %%
# reset_all_hooks()


# # %%
# reset_all_hooks()
# inp_cache = {}
# cache = {}
# def cache_hook(module, input, output, name):
#     inp_cache[name] = input
#     cache[name] = output
#     return output
# for name, mod in stable_diffusion.unet.named_children():
#     name = "pretrained.unet." + name
#     add_hook(mod, partial(cache_hook, name=name))
# test(steps=1)
# for c, v in cache.items():
#     try:
#         print(c, v.shape)
#     except:
#          print(c, type(v))
# for c, v in inp_cache.items():
#     for i, v2 in enumerate(v):
#         try:
#             print(c, i, v2.shape)
#         except:
#             print(c, i, type(v2))
# # %%
# reset_all_hooks()
# # inp_cache = {}
# cache = []
# def cache_hook(module, input, output):
#     cache.append(input[0])
#     # cache[name] = output
#     # return output
# add_hook(stable_diffusion.unet.time_proj, cache_hook)
# test(20)
# imshow(torch.stack(cache))

# # %%
# reset_all_hooks()
# for name, mod in stable_diffusion.unet.named_children():
#     if name != "conv_act":
#         name = "pretrained.unet." + name
#         add_hook(mod, partial(show_shapes_hook, name=name))
# for name, mod in enumerate(stable_diffusion.unet.down_blocks):
#     name = "pretrained.unet.down_blocks" + str(name)
#     add_hook(mod, partial(show_shapes_hook, name=name))
# for name, mod in enumerate(stable_diffusion.unet.up_blocks):
#     name = "pretrained.unet.up_blocks" + str(name)
#     add_hook(mod, partial(show_shapes_hook, name=name))
# test(steps=1)
# # %%
