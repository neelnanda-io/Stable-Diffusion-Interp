from typing import cast

import torch as t
import torch.nn.functional as F
from transformers.models.clip import modeling_clip

from utils import allclose, allclose_atol, assert_all_equal, report
from w3d5_globals import (
    get_reference_model,
    get_reference_clip_model,
    get_reference_vision_model,
    CLIPConfig,
    CLIPTextConfig,
    CLIPVisionConfig,
)


@report
def test_vision_embeddings(CLIPVisionEmbeddings):
    theirs = get_reference_vision_model().embeddings
    x = t.randn((1, 3, 224, 224))
    expected = theirs(x)

    mine = CLIPVisionEmbeddings(CLIPVisionConfig())
    mine.load_state_dict(theirs.state_dict())
    actual = mine(x)

    allclose(actual, expected)


@report
def test_mlp(CLIPMLP):
    theirs = cast(
        modeling_clip.CLIPMLP, get_reference_vision_model().encoder.layers[0].mlp
    )

    mine = CLIPMLP(CLIPVisionConfig())
    mine.load_state_dict(theirs.state_dict())

    x = t.randn((2, 3, 1024))
    with t.inference_mode():
        expected = theirs(x)
        actual = mine(x)
    allclose(actual, expected)


@report
def test_vision_attention(CLIPAttention):
    theirs = cast(
        modeling_clip.CLIPAttention,
        get_reference_vision_model().encoder.layers[0].self_attn,
    )

    mine = CLIPAttention(CLIPVisionConfig())
    mine.load_state_dict(theirs.state_dict())

    x = t.randn((1, 3, 1024))
    with t.inference_mode():
        expected = cast(t.Tensor, theirs(x)[0])
        actual = mine(x)
    allclose_atol(actual, expected, 1e-4)


@report
def test_vision_transformer(CLIPVisionTransformer):
    theirs = get_reference_vision_model()

    mine = CLIPVisionTransformer(CLIPVisionConfig())
    mine.load_state_dict(theirs.state_dict())

    x = t.randn((1, 3, 224, 224))
    with t.inference_mode():
        expected = theirs(x).pooler_output
        actual = mine(x)
    allclose_atol(actual, expected, 1e-4)


@report
def test_clip_model(CLIPModel):
    theirs = get_reference_clip_model()

    my_config = CLIPConfig(CLIPVisionConfig(), CLIPTextConfig())
    mine = CLIPModel(my_config)
    mine.load_state_dict(theirs.state_dict())
    x = dict(
        input_ids=t.randint(0, 100, size=(1, 50)),
        attention_mask=t.ones(1, 50),
        pixel_values=t.randn(1, 3, 224, 224),
    )
    with t.inference_mode():
        expected = theirs(**x)
        actual = mine(**x)
    allclose(actual.text_embeds, expected.text_embeds, rtol=0.01)
    allclose(actual.image_embeds, expected.image_embeds, rtol=0.01)


@report
def test_cosine_similarity(cosine_similarities):
    a = t.tensor(
        [
            [0.0, 0.0, 1.0],
            [(2**-0.5), (2**-0.5), 0.0],
        ]
    )
    b = t.tensor([[0.0, 0.0, -1], [0.0, 0.0, 1.0], [(2**-0.5), 0.0, (2**-0.5)]])
    similarities = cosine_similarities(a, b)
    expected = t.tensor([[-1.0000, 1.0000, 2 ** (-0.5)], [0.0000, 0.0000, 0.5000]])
    assert similarities.min() >= -1.0
    assert similarities.max() <= 1.0
    allclose(similarities, expected)


@report
def test_contrastive_loss(contrastive_loss):
    a = t.tensor(
        [
            [0.0, 0.0, 1.0],
            [(2**-0.5), (2**-0.5), 0.0],
        ]
    )
    b = t.tensor([[0.0, 0.0, -1], [(2**-0.5), 0.0, (2**-0.5)]])
    similarity = t.matmul(a, b.t())
    caption_loss = F.cross_entropy(similarity.T, t.arange(len(similarity.T)))
    image_loss = F.cross_entropy(similarity, t.arange(len(similarity)))
    loss = (caption_loss + image_loss) / 2.0
    assert loss.item() == contrastive_loss(a, b, logit_scale=t.tensor([0])).item()
