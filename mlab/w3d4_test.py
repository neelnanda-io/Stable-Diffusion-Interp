import torch as t

from utils import allclose, report


@t.inference_mode()
@report
def test_groupnorm(GroupNorm, affine: bool):
    if not affine:
        x = t.arange(72, dtype=t.float32).view(3, 6, 2, 2)
        ref = t.nn.GroupNorm(num_groups=3, num_channels=6, affine=False)
        expected = ref(x)
        gn = GroupNorm(num_groups=3, num_channels=6, affine=False)
        actual = gn(x)
        allclose(actual, expected)

    else:
        t.manual_seed(776)
        x = t.randn((3, 6, 8, 10), dtype=t.float32)
        ref = t.nn.GroupNorm(num_groups=3, num_channels=6, affine=True)
        expected = ref(x)
        gn = GroupNorm(num_groups=3, num_channels=6, affine=True)
        gn.weight.copy_(ref.weight)
        gn.bias.copy_(ref.bias)
        actual = gn(x)
        allclose(actual, expected)


@t.inference_mode()
@report
def test_self_attention(SelfAttention):
    channels = 16
    img = t.randn(1, channels, 64, 64)
    sa = SelfAttention(channels=channels, num_heads=4)
    out = sa(img)
    assert out.shape == img.shape


@t.inference_mode()
@report
def test_attention_block(AttentionBlock):
    ab = AttentionBlock(channels=16)
    img = t.randn(1, 16, 64, 64)
    out = ab(img)
    assert out.shape == img.shape


@t.inference_mode()
@report
def test_residual_block(ResidualBlock):
    in_channels = 6
    out_channels = 10
    step_dim = 1000
    groups = 2
    time_emb = t.randn(1, 1000)
    img = t.randn(1, in_channels, 32, 32)
    rb = ResidualBlock(in_channels, out_channels, step_dim, groups)
    out = rb(img, time_emb)
    assert out.shape == (1, out_channels, 32, 32)


@t.inference_mode()
@report
def test_downblock(DownBlock, downsample: bool):
    in_channels = 8
    out_channels = 12
    time_emb_dim = 1000
    groups = 2
    time_emb = t.randn(1, 1000)
    img = t.randn(1, in_channels, 32, 32)
    db = DownBlock(in_channels, out_channels, time_emb_dim, groups, downsample)
    out, skip = db(img, time_emb)
    assert skip.shape == (1, out_channels, 32, 32)
    if downsample:
        assert out.shape == (1, out_channels, 16, 16)
    else:
        assert out.shape == (1, out_channels, 32, 32)


@t.inference_mode()
@report
def test_midblock(MidBlock):
    mid_channels = 8
    time_emb_dim = 1000
    groups = 2
    time_emb = t.randn(1, 1000)
    img = t.randn(1, mid_channels, 32, 32)
    mid = MidBlock(mid_channels, time_emb_dim, groups)
    out = mid(img, time_emb)
    assert out.shape == (1, mid_channels, 32, 32)


@t.inference_mode()
@report
def test_upblock(UpBlock, upsample):
    in_channels = 8
    out_channels = 12
    time_emb_dim = 1000
    groups = 2
    time_emb = t.randn(1, 1000)
    img = t.randn(1, out_channels, 16, 16)
    skip = t.rand_like(img)
    up = UpBlock(in_channels, out_channels, time_emb_dim, groups, upsample)
    out = up(img, time_emb, skip)
    if upsample:
        assert out.shape == (1, in_channels, 32, 32)
    else:
        assert out.shape == (1, in_channels, 16, 16)


@t.inference_mode()
@report
def test_unet(Unet):
    # dim mults is limited by number of multiples of 2 in the image
    # 28 -> 14 -> 7 is ok but can't half again without having to deal with padding
    image_size = 28
    channels = 8
    batch_size = 8
    model = Unet(
        image_shape=(8, 28, 28),
        channels=channels,
        dim_mults=(
            1,
            2,
            4,
        ),
    )
    x = t.randn((batch_size, channels, image_size, image_size))
    num_steps = t.randint(0, 1000, (batch_size,))
    out = model(x, num_steps)
    assert out.shape == x.shape
