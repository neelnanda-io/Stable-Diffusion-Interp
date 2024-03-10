from dataclasses import dataclass
from typing import cast

import torch as t


# Lazy load on first use, so importing this module isn't slow
__REFERENCE_MODEL = None


def get_reference_model():
    """Return the reference CLIP."""
    from sentence_transformers import SentenceTransformer

    global __REFERENCE_MODEL
    if __REFERENCE_MODEL is None:
        __REFERENCE_MODEL = SentenceTransformer("clip-ViT-L-14")
    return __REFERENCE_MODEL


def get_reference_clip_model():
    from transformers.models.clip import modeling_clip

    return cast(modeling_clip.CLIPModel, get_reference_model()[0].model)


def get_reference_vision_model():
    return get_reference_clip_model().vision_model


@dataclass(frozen=True)
class CLIPVisionConfig:
    attention_dropout = 0.0
    dropout = 0.0
    hidden_size = 1024
    image_size = 224
    initializer_factor = 1.0
    initializer_range = 0.02
    intermediate_size = 4096
    layer_norm_eps = 1e-05
    num_attention_heads = 16
    num_hidden_layers = 24
    patch_size = 14


@dataclass(frozen=True)
class CLIPTextConfig:
    attention_dropout = 0.0
    bos_token_id = 0
    dropout = 0.0
    eos_token_id = 2
    hidden_size = 768
    initializer_factor = 1.0
    hidden_act = "quick_gelu"
    initializer_range = 0.02
    intermediate_size = 3072
    layer_norm_eps = 1e-05
    max_position_embeddings = 77
    num_attention_heads = 12
    num_hidden_layers = 12
    pad_token_id = 1
    vocab_size = 49408
    output_attentions = False
    output_hidden_states = False
    use_return_dict = True


@dataclass(frozen=True)
class CLIPConfig:
    initializer_factor = 1.0
    logit_scale_init_value = 2.6592
    projection_dim = 768
    vision_config: CLIPVisionConfig
    text_config: CLIPTextConfig


@dataclass
class CLIPOutput:
    text_embeds: t.Tensor
    image_embeds: t.Tensor
