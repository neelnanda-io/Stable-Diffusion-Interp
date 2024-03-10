# %%
"""
# W3D5 - Part 1 - Contrastive Language-Image Pre-Training (CLIP)

CLIP is a model that contains a vision model and a language model, and it is trained so that the embeddings produced by the two parts are similar when the image and the text have a similar meaning or topic. Today, we're using a vision transformer (which is exactly what it sounds like) and a GPT-style transformer, but you could also use a ResNet and a bag-of-words model or whatever you like.

For training data, the authors scraped the Internet for images that have captions under them to obtain (image, text) pairs that we assume are both "about" the same thing. For example, the image might be a guinea pig eating a cucumber and the caption says "Mr. Fluffers fueling up with some Vitamin C."

<p align="center">
    <img src="clip_images/guineapig_cucumber2.jpg" width="400" /><br>
    Mr. Fluffers fueling up with some Vitamin C.
</p>

To do traditional supervised learning, we would start with the image, feed it in, and then try to unembed to generate text and compare the predicted text to the actual caption. Or start with the text, generate an image, and compare the image to the actual.

For either of these, it's tricky to define a differentiable loss function that captures "how close is the meaning of these two pieces of text" or "how much do these two images depict the same thing".

CLIP instead uses a **contrastive loss**, which just means that the embedding of an image and the embedding of its matching caption should be similar, while the embedding of an image and all the other captions in the batch should be dissimilar. "Similar" is efficiently calculated using the cosine similarity.

It turns out that if you use large enough batch sizes (like 32,768) and a large enough dataset (400 million pairs), this works quite well. It takes thousands of GPU-days to train a CLIP, so we will just play with the pretrained weights today.

Some cool applications of this are:

- Given an image, you can embed it and see how similar that embedding is to the embedding of the string "photo of a dog" versus the string "photo of a cat". This means you can classify images, but in a much more flexible way than a traditional supervised classifier where the set of categories is fixed up front.
- Given some text, you can search a large database of image embeddings for images that are similar to the text embedding.

The outline of today's tasks are:

Part 1:

- Implement your own vision transformer
- Reuse old GPT code for the text transformer
- Assemble a CLIP and load pretrained weights
- Play with CLIP!

Part 2:

- Complete the implementation of a Stable Diffusion inference pipeline
- Run inference on your model
- Play with other things you can do with Stable Diffusion, such as animations

## Table of Contents

- [References (optional) for Part 1](#references-optional-for-part-)
- [Vision Transformers](#vision-transformers)
    - [Patch Embedding](#patch-embedding)
    - [Positional Embedding](#positional-embedding)
    - [Class Embedding (replacement for "begin" token)](#class-embedding-replacement-for-begin-token)
    - [Config Classes](#config-classes)
    - [position_ids](#positionids)
- [CLIP MLP](#clip-mlp)
- [Self-Attention](#self-attention)
- [CLIP Layer](#clip-layer)
- [CLIP Encoder](#clip-encoder)
- [CLIPVisionTransformer](#clipvisiontransformer)
- [CLIPTextTransformer](#cliptexttransformer)
- [CLIPModel](#clipmodel)
- [Data Preparation](#data-preparation)
- [Cosine Similarities](#cosine-similarities)
- [Running the Model](#running-the-model)
- [Implementing Constrastive Loss](#implementing-constrastive-loss)
- [Bonus](#bonus)
    - [Prompt Engineering and Zero Shot Classification](#prompt-engineering-and-zero-shot-classification)
    - [GELU approximations](#gelu-approximations)
    - [X-CLIP](#x-clip)

## References (optional) for Part 1

CLIP

- [Paper](https://arxiv.org/pdf/2103.00020.pdf)
- [Official OpenAI repo](https://github.com/openai/CLIP)
- [HuggingFace implementation](https://huggingface.co/sentence-transformers/clip-ViT-L-14)

X-CLIP - Includes experimental improvements from recent papers

- [Code repo](https://github.com/lucidrains/x-clip)
"""
# %%

import glob
import os
import sys
from typing import Callable, Union, cast

import pandas as pd
import torch as t
from einops import rearrange, repeat
from fancy_einsum import einsum
from IPython.display import display
from matplotlib import pyplot as plt
from PIL import Image
from torch import nn
from torch.nn import functional as F
from torchvision import transforms
from transformers.models.clip import modeling_clip


import utils
import w3d5_test
from utils import allclose
from w3d5_globals import (
    CLIPConfig,
    CLIPOutput,
    CLIPTextConfig,
    CLIPVisionConfig,
    get_reference_model,
    get_reference_clip_model,
)

MAIN = __name__ == "__main__"
IS_CI = os.getenv("IS_CI")
device = t.device("cuda" if t.cuda.is_available() else "cpu")

# %%
"""
## Vision Transformers

Our first task for today is to implement a vision transformer.

Using transformers on image data is actually easier than using it on text data. Because our input data is already continuous, we don't need a special tokenizer and we don't need a `nn.Embedding` to translate from discrete token ids to a continuous representation. (Technically the RGB pixel values are discrete in the range [0, 256), but we won't worry about this).

The only issue is that for our images (which for today we'll assume to be exactly 224 x 224px) treating each pixel as a sequence element would result in a sequence length around 50K. Since self-attention is quadratic in the sequence length, we'd prefer to decrease this sequence length to something more manageable. This is analogous to how we won't model text as individual characters, but as slightly larger chunks.

The original [Vision Transformers paper](https://arxiv.org/pdf/2010.11929.pdf) used 14x14 patches of 16x16 pixels each, but in our implementation of CLIP (matching CLIP ViT-L/14) the patch size is 14x14 pixels.

The rest of the vision transformer is going to look extremely similar to what you've seen with GPT.

### Patch Embedding

There are a couple equivalent ways to obtain an embedding vector for each patch. For example, you could use `einops.rearrange` and a `Linear(patch_pixels, hidden_size)`. Instead, we're going to follow HuggingFace and use a `nn.Conv2d` with appropriate stride and kernel size (and no bias).

### Positional Embedding

When first learning vision transformers, I expected the positional embedding would work best by indicating (x, y) coordinates for each patch so that the model can easily understand the 2D spatial relationships.

However, the Vision Transformers paper found no difference between this 2D method and just simply numbering the patches (see appendix D.4). This means that the model has to learn itself that patch 16 is correlated with patches 0 and 32 (because they are vertically adjacent), but this doesn't seem to be a problem. They speculate that there are so few patches that it's just very easy to memorize these patterns.

### Class Embedding (replacement for "begin" token)

When using text, it's common practice to have the tokenizer prepend a special placeholder token called the "begin token". When we train the model for sequence classification, we use the final layer's embedding at this sequence position for the representation of the entire sequence, so attention heads learn to copy relevant data to this position.

Since there's no separate tokenizer for vision, we're going to initialize a random normal embedding vector of `embedding_size` and prepend that to every sequence. This embedding is called the class embedding because it's used for classification.

### Config Classes

Config dataclasses for CLIP have been defined and imported from `w3d5_globals.py` These will be a helpful reference as you build the CLIP components.
"""
# %%


def print_class_attrs(cls: type) -> None:
    print(f"\n\n{cls.__name__}\n---")
    for k, v in ((k, v) for k, v in vars(cls).items() if k[0] != "_"):
        print(f"{k}: {v}")


if MAIN:
    print_class_attrs(CLIPVisionConfig)
    print_class_attrs(CLIPTextConfig)
    print_class_attrs(CLIPConfig)

# %%
"""

### position_ids

Register a buffer called `position_ids` which just contains `arange(0, (self.image_size // self.patch_size) ** 2 + 1)`. The extra index is for the class embedding in addition to the standard patches. This avoids redundantly allocating the `arange` on the target device on every forward pass.
"""
# %%
class CLIPVisionEmbeddings(nn.Module):
    config: CLIPVisionConfig
    patch_size: int
    image_size: int
    embed_dim: int
    num_patches: int
    class_embedding: nn.Parameter
    patch_embedding: nn.Conv2d
    position_embedding: nn.Embedding
    position_ids: t.Tensor

    def __init__(self, config: CLIPVisionConfig):
        """Assign values from input config to class member variables as appropriate,
        e.g. self.patch_size = config.patch_size"""
        super().__init__()
        "SOLUTION"
        self.config = config
        self.patch_size = config.patch_size
        self.image_size = config.image_size
        self.embed_dim = config.hidden_size
        self.num_patches = (self.image_size // self.patch_size) ** 2
        self.num_positions = self.num_patches

        self.class_embedding = nn.Parameter(t.randn(self.embed_dim))
        self.patch_embedding = nn.Conv2d(
            in_channels=3,
            out_channels=self.embed_dim,
            kernel_size=self.patch_size,
            stride=self.patch_size,
            bias=False,
        )
        self.position_embedding = nn.Embedding(self.num_positions + 1, self.embed_dim)
        self.register_buffer(
            "position_ids", t.arange(self.num_positions + 1).expand((1, -1))
        )

    def forward(self, x: t.Tensor) -> t.Tensor:
        """Apply the patch embeddings and the positional embeddings and return their sum.

        x: shape (batch, channels=3, height=224, width=224)
        out: shape (batch, sequence, hidden)
        """
        "SOLUTION"
        B, C, H, W = x.shape
        patch_embed = self.patch_embedding(x)
        patch_embed = rearrange(
            patch_embed, "batch hidden grid_h grid_w -> batch (grid_h grid_w) hidden"
        )
        pos_embed = self.position_embedding(self.position_ids)
        class_embedding = repeat(
            self.class_embedding, "h -> b 1 h", b=B, h=self.embed_dim
        )
        return t.cat((class_embedding, patch_embed), dim=1) + pos_embed


if MAIN and not IS_CI:
    w3d5_test.test_vision_embeddings(CLIPVisionEmbeddings)


# %%
"""
## CLIP MLP

The remaining layers of CLIP operate on embedding vectors of `hidden_size`, so they're independent of whether the input was text or images.

The MLP uses a faster approximation to the [GELU](https://arxiv.org/pdf/1606.08415.pdf) nonlinearity. Note that as of PyTorch 1.11, `nn.GELU` and `F.gelu` compute the exact equation for GELU.

Use the equation from the paper and implement the sigmoid approximation from section 2 yourself. Plot the absolute difference on the interval [-5, 5] and check how different the approximation is from the exact. Then implement the MLP using the approximation.

The MLP looks the same as in a standard transformer: a Linear layer that goes from hidden size to an intermediate size 4 times larger, a GELU, and a second Linear back down to the hidden size.
"""
# %%
def gelu_sigmoid_approximation(x: t.Tensor) -> t.Tensor:
    """Return sigmoid approximation of GELU of input tensor x with same shape."""
    "SOLUTION"
    return x * t.sigmoid(1.702 * x)


def plot_gelu_approximation(x: t.Tensor):
    fig, (ax0, ax1) = plt.subplots(nrows=2, figsize=(12, 12))  # type: ignore
    actual = F.gelu(x)
    approx = gelu_sigmoid_approximation(x)
    diff = (actual - approx).abs()
    x_cpu = x.cpu()
    ax0.plot(x_cpu, diff.cpu(), label="absolute error")
    ax0.legend()
    ax1.plot(x_cpu, actual.cpu(), label="exact", alpha=0.5)
    ax1.plot(x_cpu, approx.cpu(), label="sigmoid", alpha=0.5)
    ax1.legend()
    ax1.set(xlabel=f"x ({x.dtype})")


if MAIN and not IS_CI:
    x = t.linspace(-5, 5, 400)
    plot_gelu_approximation(x)

    if t.cuda.is_available():
        x16 = t.linspace(-5, 5, 400, dtype=t.float16, device=device)
        plot_gelu_approximation(x16)


# %%
class CLIPMLP(nn.Module):
    fc1: nn.Linear
    fc2: nn.Linear

    def __init__(self, config: Union[CLIPVisionConfig, CLIPTextConfig]):
        """Initialize parent class, then assign fully-connected layers based
        on shape in input config"""
        "SOLUTION"
        super().__init__()
        self.fc1 = nn.Linear(config.hidden_size, config.intermediate_size)
        self.fc2 = nn.Linear(config.intermediate_size, config.hidden_size)

    def forward(self, x: t.Tensor) -> t.Tensor:
        """Run forward pass of MLP, including fully-connected layers and non-linear
        activations where appropriate"""
        "SOLUTION"
        x = self.fc1(x)
        x = gelu_sigmoid_approximation(x)
        x = self.fc2(x)
        return x


if MAIN and not IS_CI:
    w3d5_test.test_mlp(CLIPMLP)


# %%
"""
## Self-Attention

For the vision transformer, the authors don't use masked attention. You should be able to copy and paste from your `BertSelfAttention` class you wrote previously and fix up the variable names. Or try writing it from memory for the practice.
"""
# %%
class CLIPAttention(nn.Module):
    num_heads: int
    head_size: int
    q_proj: nn.Linear
    k_proj: nn.Linear
    v_proj: nn.Linear
    out_proj: nn.Linear
    dropout: nn.Dropout

    def __init__(self, config: Union[CLIPVisionConfig, CLIPTextConfig]):
        """Assign values from input config to class member variables as appropriate"""
        "SOLUTION"
        super().__init__()
        self.num_heads = config.num_attention_heads
        assert config.hidden_size % config.num_attention_heads == 0
        self.head_size = config.hidden_size // config.num_attention_heads
        self.q_proj = nn.Linear(
            config.hidden_size, config.num_attention_heads * self.head_size
        )
        self.k_proj = nn.Linear(
            config.hidden_size, config.num_attention_heads * self.head_size
        )
        self.v_proj = nn.Linear(
            config.hidden_size, config.num_attention_heads * self.head_size
        )
        self.out_proj = nn.Linear(
            config.num_attention_heads * self.head_size, config.hidden_size
        )
        self.dropout = nn.Dropout(config.dropout)

    def attention_pattern_pre_softmax(self, x: t.Tensor) -> t.Tensor:
        """Return the attention pattern after scaling but before softmax.

        pattern[batch, head, q, k] should be the match between a query at sequence position q and a key at sequence position k.
        """
        "SOLUTION"
        B, S, H = x.shape
        Q = self.q_proj(x)
        Q = rearrange(
            Q, "b seq (head head_size) -> b head seq head_size", head=self.num_heads
        )
        K = self.k_proj(x)
        K = rearrange(
            K, "b seq (head head_size) -> b head seq head_size", head=self.num_heads
        )
        out = einsum(
            "b head seq_q head_size, b head seq_k head_size -> b head seq_q seq_k", Q, K
        )
        out = out / (self.head_size**0.5)
        assert out.shape == (B, self.num_heads, S, S)
        return out

    def forward(self, x: t.Tensor) -> t.Tensor:
        """Perform forward pass through attention layer, computing attention pattern and value projections
        to combine into output. Remember to apply dropout."""
        "SOLUTION"
        B, S, H = x.shape
        attention_pattern = self.attention_pattern_pre_softmax(x)
        softmaxed_attention = attention_pattern.softmax(dim=-1)
        V = self.v_proj(x)
        V = rearrange(
            V, "b seq (head head_size) -> b head seq head_size", head=self.num_heads
        )
        combined_values = einsum(
            "b head seq_k head_size, b head seq_q seq_k -> b head seq_q head_size",
            V,
            softmaxed_attention,
        )
        out = self.out_proj(
            rearrange(combined_values, "b head seq head_size -> b seq (head head_size)")
        )
        out = self.dropout(out)
        assert out.shape == (B, S, H)
        return out


if MAIN and not IS_CI:
    w3d5_test.test_vision_attention(CLIPAttention)


# %%

"""
## CLIP Layer

Identical to GPT (besides calling our slightly different MLP), so this is provided for you. Make sure to read through and understand
the operations being performed.
"""
# %%
class CLIPEncoderLayer(nn.Module):
    def __init__(self, config: Union[CLIPVisionConfig, CLIPTextConfig]):
        super().__init__()
        self.embed_dim = config.hidden_size
        self.self_attn = CLIPAttention(config)
        self.layer_norm1 = nn.LayerNorm(self.embed_dim)
        self.mlp = CLIPMLP(config)
        self.layer_norm2 = nn.LayerNorm(self.embed_dim)

    def forward(self, x):
        x = x + self.self_attn(self.layer_norm1(x))
        x = x + self.mlp(self.layer_norm2(x))
        return x


# %%
"""
## CLIP Encoder

This is also provided as it's trivial. Note that a full-fledged implementation this would have more code in it for things like checkpointing.
"""
# %%
class CLIPEncoder(nn.Module):
    layers: utils.StaticModuleList[CLIPEncoderLayer]

    def __init__(self, config: Union[CLIPVisionConfig, CLIPTextConfig]):
        super().__init__()
        self.layers = utils.StaticModuleList(
            [CLIPEncoderLayer(config) for _ in range(config.num_hidden_layers)]
        )

    def forward(self, x: t.Tensor) -> t.Tensor:
        for layer in self.layers:
            x = layer(x)
        return x


# %%
"""
## CLIPVisionTransformer

This is the last class to implement before we can load pretrained weights for the vision transformer!

The output will consist of only the first sequence position corresponding to the prepended "class embedding". Do the slice before the final layer norm to avoid unnecessary computation.

We've made all the variable names identical so far with the idea that the state dict should exactly match. However, the pretrained weights have spelled `pre_layrnorm` incorrectly. Sad! If this really bothers you, you can fix it in your version and adjust the weight loading code to adapt.
"""
# %%
class CLIPVisionTransformer(nn.Module):
    config: CLIPVisionConfig
    embeddings: CLIPVisionEmbeddings
    pre_layrnorm: nn.LayerNorm
    encoder: CLIPEncoder
    post_layernorm: nn.LayerNorm

    def __init__(self, config: CLIPVisionConfig):
        """Assign values from input config to class member variables as appropriate"""
        "SOLUTION"
        super().__init__()
        self.config = config
        embed_dim = config.hidden_size
        self.embeddings = CLIPVisionEmbeddings(config)
        self.pre_layrnorm = nn.LayerNorm(embed_dim)
        self.encoder = CLIPEncoder(config)
        self.post_layernorm = nn.LayerNorm(embed_dim)

    def forward(self, x: t.Tensor) -> t.Tensor:
        """Perform forward pass through vision transformer: embedding, layer norm, encoder, layer norm
        Return output corresponding to prepended class_embedding"""
        "SOLUTION"
        x = self.embeddings(x)
        x = self.pre_layrnorm(x)
        x = self.encoder(x)
        x = x[:, 0]
        x = self.post_layernorm(x)
        return x


if MAIN and not IS_CI:
    w3d5_test.test_vision_transformer(CLIPVisionTransformer)

# %%
"""

## CLIPTextTransformer

The text transformer looks a lot like BERT, except it does have the causal attention mask like GPT.

It supports sequences of varying lengths with padding at the end, and padding tokens are also masked out during attention. We won't bother re-implementing the code, since this is very similar to what you've done before.

We do need a tokenizer for the text stuff, and again we'll use the provided one since it works the same as you've seen previously.
"""
if MAIN and not IS_CI:
    # TODO: change text tokenizer and encoder to the one for stable diffusion
    tokenize = get_reference_model().tokenize

# %%

"""
## CLIPModel

Now we're ready to put together the full model. In general, since we allow mixing and matching any models, the embedding of the image and text aren't going to have the same dimension.

The CLIPModel has two linear projections that take the individual model outputs to a common hidden size of `config.projection_dim`.

CLIPModel also takes care of normalizing each unit vector to have a L2 norm of 1. This is because cosine similarity can be calculated as a dot product of two unit vectors. Finally, the two embeddings are packaged into a tuple.

The scalar parameter `logit_scale` is only used during training, where it's used to multiply the similarity scores before computing the contrastive loss.

```mermaid

graph TD
    subgraph CLIPModel

    Image --> ImageTransformer --> VisualProjection --> Normalize1[Normalize] --> CLIPOutput
    Text --> TextTransformer --> TextProjection --> Normalize2[Normalize] --> CLIPOutput

    end
```
"""

# %%
class CLIPModel(nn.Module):
    config: CLIPConfig
    text_config: CLIPTextConfig
    vision_config: CLIPVisionConfig
    projection_dim: int
    text_embed_dim: int
    vision_embed_dim: int
    text_model: modeling_clip.CLIPTextTransformer
    vision_model: CLIPVisionTransformer
    visual_projection: nn.Linear
    text_projection: nn.Linear
    logit_scale: nn.Parameter

    def __init__(self, config: CLIPConfig):
        """Assign values from input config to class member variables as appropriate.

        The typechecker will complain when passing our CLIPTextConfig to CLIPTextTransformer, because the latter expects type transformers.models.clip.configuration_clip.CLIPTextConfig. You can ignore this as our type is in fact compatible.
        """
        "SOLUTION"
        super().__init__()
        self.config = config
        text_config = config.text_config
        vision_config = config.vision_config

        self.projection_dim = config.projection_dim
        self.text_embed_dim = text_config.hidden_size
        self.vision_embed_dim = vision_config.hidden_size

        self.text_model = modeling_clip.CLIPTextTransformer(text_config)  # type: ignore - this is compatible
        self.vision_model = CLIPVisionTransformer(vision_config)

        self.visual_projection = nn.Linear(
            self.vision_embed_dim, self.projection_dim, bias=False
        )
        self.text_projection = nn.Linear(
            self.text_embed_dim, self.projection_dim, bias=False
        )
        self.logit_scale = nn.Parameter(t.ones([]) * self.config.logit_scale_init_value)

    def forward(self, input_ids, attention_mask, pixel_values) -> CLIPOutput:
        """
        Perform forward pass through CLIP model, applying text and vision model/projection.

        input_ids: (batch, sequence)
        attention_mask: (batch, sequence). 1 for visible, 0 for invisible.
        pixel_values: (batch, channels, height, width)
        """
        "SOLUTION"
        # TBD t.linalg.vector_norm instead
        vis = self.vision_model(pixel_values)
        vis = self.visual_projection(vis)
        vis = vis / vis.norm(p=2, dim=-1, keepdim=True)
        text = self.text_model(input_ids, attention_mask).pooler_output
        text = self.text_projection(text)
        text = text / text.norm(p=2, dim=-1, keepdim=True)
        return CLIPOutput(text, vis)


if MAIN and not IS_CI:
    w3d5_test.test_clip_model(CLIPModel)


# %%
"""
## Data Preparation

The data preparation is the same as you've seen before. The ImageNet normalization constants are used. Feel free to supply some of your own text and/or images here.
"""
# %%


def get_images(glob_fnames: str) -> tuple[list[str], list[Image.Image]]:
    filenames = glob.glob(glob_fnames)
    images = [Image.open(filename).convert("RGB") for filename in filenames]
    image_names = [
        os.path.splitext(os.path.basename(filename))[0] for filename in filenames
    ]
    for im in images:
        display(im)
    return image_names, images


if MAIN and not IS_CI:
    preprocess = cast(
        Callable[[Image.Image], t.Tensor],
        transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Resize((224, 224)),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        ),
    )
    texts = [
        "A guinea pig eating a cucumber",
        "A pencil sketch of a guinea pig",
        "A rabbit eating a carrot",
        "A paperclip maximizer",
    ]

    out = tokenize(texts)
    input_ids = out["input_ids"]
    attention_mask = out["attention_mask"]
    image_names, images = get_images("./clip_images/*")
    pixel_values = t.stack([preprocess(im) for im in images], dim=0)

# %%
"""
## Cosine Similarities

Since the model already normalizes each embedding to be a unit vector, this function becomes a one-liner.
"""
# %%
def cosine_similarities(a: t.Tensor, b: t.Tensor) -> t.Tensor:
    """Return cosine similarities between all pairs of embeddings.

    Each element of the batch should be a unit vector already.

    a: shape (batch_a, hidden_size)
    b: shape (batch_b, hidden_size)
    out: shape (batch_a, batch_b)
    """
    "SOLUTION"
    allclose(t.linalg.norm(a, dim=1), t.ones(a.shape[0], device=a.device))
    allclose(t.linalg.norm(b, dim=1), t.ones(b.shape[0], device=b.device))
    return a @ b.T


if MAIN:
    w3d5_test.test_cosine_similarity(cosine_similarities)


# %%
"""
## Running the Model

Run the model and compute the cosine similarities between each image and each piece of text. Visualize the results and see if they match what you expect.
"""
# %%


def load_trained_model(config: CLIPConfig):
    model = CLIPModel(config)
    full_state_dict = get_reference_clip_model().state_dict()
    model.load_state_dict(full_state_dict)
    return model


if MAIN and not IS_CI:
    config = CLIPConfig(CLIPVisionConfig(), CLIPTextConfig())
    model = load_trained_model(config).to(device)
    with t.inference_mode():
        out = model(
            input_ids.to(device), attention_mask.to(device), pixel_values.to(device)
        )
    similarities = cosine_similarities(out.text_embeds, out.image_embeds)
    df = pd.DataFrame(
        similarities.detach().cpu().numpy(), index=texts, columns=image_names
    ).round(3)
    display(df)


# %%
"""
## Implementing Constrastive Loss

We're not going to train today, but we'll implement the contrastive loss to make sure we understand it.

There's a nice trick to implement the contrastive loss using of the average of two `F.cross_entropy` terms. See if you can find it.

<details>

<summary>Spoiler - Contrastive Loss Calculation</summary>

First compute the matrix `similarities[text_index][image_index]`, of shape (batch, batch).

Since the ith text corresponds to the ith image in the training data, `similarities[i]` should have a value near 1 at index i, and be low otherwise. This is just like cross entropy where the target is class i.

The same holds for `similarities[:, i]`, so that's the second cross entropy term. Each value in our matrix contributed to each term, so taking the average prevents double-counting.

</details>

"""
# %%
def contrastive_loss(
    text_embeds: t.Tensor, image_embeds: t.Tensor, logit_scale: t.Tensor
) -> t.Tensor:
    """Return the contrastive loss between a batch of text and image embeddings.

    The embeddings must be in order so that text_embeds[i] corresponds with image_embeds[i].

    text_embeds: (batch, output_dim)
    image_embeds: (batch, output_dim)
    logit_scale: () - log of the scale factor to apply to each element of the similarity matrix

    Out: scalar tensor containing the loss
    """
    "SOLUTION"
    assert text_embeds.shape == image_embeds.shape
    similarity = t.matmul(text_embeds, image_embeds.t()) * logit_scale.exp()
    caption_loss = F.cross_entropy(similarity.T, t.arange(len(similarity.T)))
    image_loss = F.cross_entropy(similarity, t.arange(len(similarity)))
    return (caption_loss + image_loss) / 2.0


if MAIN:
    w3d5_test.test_contrastive_loss(contrastive_loss)

# %%
"""
# On to Part 2

In the following part of this day's exercises, we will finally get to play with the exciting and *very* state-of-the-art Stable Diffusion model, with ideas for bonus tasks after you complete the implementation of the model.

If you would like to continue working on CLIP-related models, here are some bonus tasks that you can return to after completing Part 2.

## Bonus

### Prompt Engineering and Zero Shot Classification

Thinking back to Part 1, CLIP can be used as a classifier by comparing the unknown image's embedding with the embedding of a prompt like "a photo of [class name]". Implement this idea and see how good the results are, then try to improve them by finding a better prompt. Or, use several prompts and ensemble the outputs together.

### GELU approximations

In the CLIP model, could we have "gotten away" with using PyTorch's GELU instead of the approximation the authors used? Or are the pretrained weights precisely adapted to the approximation? Try running the pretrained weights using the PyTorch exact implementation and see how different the results are.

### X-CLIP

Read through the code at the [X-CLIP](https://github.com/lucidrains/x-clip) repo and try to understand some of the modifications and improvements.
"""
