"""Model builders and the multispectral channel-adaptation surgery.

THE CENTRAL PROBLEM OF THIS FILE
--------------------------------
Every ImageNet-pretrained backbone takes 3 input channels. Sentinel-2 gives 13.
There are three ways out, and the choice matters:

  (a) RGB only            — throw away 10 bands, keep the pretrained stem intact.
  (b) Random new stem     — keep 13 bands, discard the pretrained stem weights.
  (c) Inflated stem       — keep 13 bands AND the pretrained stem, by mapping the
                            3 pretrained kernels onto 13 input channels.

(c) is the main path here. The reason (c) beats (b) is that a pretrained stem is
not a random projection: it holds oriented edge detectors and colour-opponent
filters that are useful for *any* image-like input, multispectral included.
Reinitialising it throws away the cheapest part of the pretraining to buy
nothing.

The implementation of (c) copies the pretrained RGB kernels into the matching
Sentinel-2 red/green/blue channels and fills the remaining channels with the
mean pretrained kernel, then rescales the whole tensor by 3/C_in. The rescale is
the part people forget: without it, summing over 13 input channels instead of 3
inflates the expected pre-activation magnitude by roughly 13/3, which pushes the
first BatchNorm far off its pretrained running statistics and undoes the
advantage you were trying to keep.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from .bands import BAND_IDS, NUM_BANDS, RGB_INDICES


# --------------------------------------------------------------------------
# Channel surgery
# --------------------------------------------------------------------------
def inflate_conv_weight(
    weight: torch.Tensor, in_channels: int, rgb_positions=RGB_INDICES
) -> torch.Tensor:
    """Map a pretrained (out, 3, kh, kw) kernel onto (out, in_channels, kh, kw).

    RGB-aligned initialisation: the pretrained red/green/blue kernels are placed
    at the Sentinel-2 channels that actually carry red/green/blue light
    (B04/B03/B02); every other channel gets the mean of the three, which is the
    best available guess for a band whose visual semantics are unknown to the
    pretrained model. Finally the tensor is scaled by 3 / in_channels so the
    expected activation magnitude entering the network is preserved.

    On that scale factor: 3/C is the right correction when the input channels
    are strongly correlated, because the pre-activation is then a sum of C
    near-identical terms and grows linearly in C. Sentinel-2 bands are indeed
    strongly correlated (they are all measuring the same surface under the same
    illumination), so 3/C is used. For genuinely independent channels the sum
    would grow as sqrt(C) and sqrt(3/C) would be the correct factor instead —
    worth knowing, but it is the wrong model of this data.
    """
    assert weight.ndim == 4 and weight.shape[1] == 3, f"expected (O,3,kh,kw), got {tuple(weight.shape)}"
    out_ch, _, kh, kw = weight.shape
    mean_kernel = weight.mean(dim=1, keepdim=True)  # (O, 1, kh, kw)
    new = mean_kernel.repeat(1, in_channels, 1, 1).clone()
    for src, dst in enumerate(rgb_positions):
        if dst < in_channels:
            new[:, dst] = weight[:, src]
    new *= 3.0 / in_channels
    assert new.shape == (out_ch, in_channels, kh, kw)
    return new


def adapt_first_conv(conv: nn.Conv2d, in_channels: int, mode: str = "inflate") -> nn.Conv2d:
    """Replace a 3-channel Conv2d with an `in_channels` one.

    mode="inflate" -> strategy (c); mode="random" -> strategy (b), kept so the
    ablation in NB02 can measure what the pretrained stem is actually worth.
    """
    if mode not in {"inflate", "random"}:
        raise ValueError(f"mode must be 'inflate' or 'random', got {mode!r}")
    new = nn.Conv2d(
        in_channels,
        conv.out_channels,
        kernel_size=conv.kernel_size,
        stride=conv.stride,
        padding=conv.padding,
        bias=conv.bias is not None,
    )
    if mode == "inflate":
        with torch.no_grad():
            new.weight.copy_(inflate_conv_weight(conv.weight.data, in_channels))
            if conv.bias is not None:
                new.bias.copy_(conv.bias.data)
    return new


def adapt_patch_embed(proj: nn.Conv2d, in_channels: int, mode: str = "inflate") -> nn.Conv2d:
    """Same surgery for a ViT patch-embedding projection.

    A ViT's patch embedding *is* a strided Conv2d whose kernel equals the patch
    size, so the identical argument applies — only the layer's name changes.
    """
    return adapt_first_conv(proj, in_channels, mode=mode)


# --------------------------------------------------------------------------
# Arm 1 — small CNN from scratch
# --------------------------------------------------------------------------
class SmallCNN(nn.Module):
    """4 Conv-BN-ReLU-MaxPool blocks, GAP, linear head. ~0.3M parameters.

    Exists to answer one question: how much does *spatial* structure add on top
    of the per-band statistics of Arm 0? Residential and Industrial have similar
    spectra and different textures, so this is where they should separate.
    Deliberately tiny — it must train in a couple of minutes on a T4, and a big
    from-scratch model on 19k chips would only measure overfitting.
    """

    def __init__(self, in_channels: int = NUM_BANDS, num_classes: int = 10, width: int = 32):
        super().__init__()
        chans = [in_channels, width, width * 2, width * 4, width * 4]
        blocks = []
        for i in range(4):
            blocks += [
                nn.Conv2d(chans[i], chans[i + 1], 3, padding=1, bias=False),
                nn.BatchNorm2d(chans[i + 1]),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2),  # 64 -> 32 -> 16 -> 8 -> 4
            ]
        self.features = nn.Sequential(*blocks)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.feature_dim = chans[-1]
        self.head = nn.Linear(self.feature_dim, num_classes)

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        return self.pool(self.features(x)).flatten(1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.forward_features(x))


# --------------------------------------------------------------------------
# Arms 2 and 3 — pretrained backbones
# --------------------------------------------------------------------------
class BackboneClassifier(nn.Module):
    """Wraps a timm/torchvision feature extractor with a linear head.

    Exposes `forward_features` so NB04 can pull frozen penultimate embeddings
    and NB06 can hook the last conv block for Grad-CAM, without either notebook
    reaching into the backbone's internals.
    """

    def __init__(self, backbone: nn.Module, feature_dim: int, num_classes: int):
        super().__init__()
        self.backbone = backbone
        self.feature_dim = feature_dim
        self.head = nn.Linear(feature_dim, num_classes)

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        f = self.backbone(x)
        return f.flatten(1) if f.ndim > 2 else f

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.forward_features(x))


def build_resnet18(
    in_channels: int = NUM_BANDS,
    num_classes: int = 10,
    pretrained: bool = True,
    stem_mode: str = "inflate",
) -> BackboneClassifier:
    """ImageNet ResNet-18 with an adapted stem.

    Note we do NOT upsample the 64x64 input for ResNet: a stride-2 7x7 stem plus
    maxpool already reduces 64 -> 16 before layer1, leaving a 2x2 final feature
    map, which is small but workable. Upsampling to 224 would cost ~12x the
    compute for interpolated detail that was never measured by the sensor.
    """
    from torchvision.models import ResNet18_Weights, resnet18

    weights = ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
    net = resnet18(weights=weights)
    if in_channels != 3:
        net.conv1 = adapt_first_conv(net.conv1, in_channels, mode=stem_mode)
    feature_dim = net.fc.in_features
    net.fc = nn.Identity()
    return BackboneClassifier(net, feature_dim, num_classes)


def build_vit(
    in_channels: int = NUM_BANDS,
    num_classes: int = 10,
    pretrained: bool = True,
    model_name: str = "vit_small_patch16_224",
    img_size: int = 64,
    stem_mode: str = "inflate",
) -> BackboneClassifier:
    """A small pretrained ViT, reconfigured for 64x64 inputs.

    THE INPUT-SIZE TRADE-OFF. ViT-B/16 expects 224x224. Two options:
      * upsample 64 -> 224: keeps the pretrained positional embeddings exactly,
        but spends ~12x the compute on bicubic-invented pixels;
      * keep 64x64 and let timm interpolate the positional embeddings to the new
        4x4 = 16-token grid.
    We take the second. It is far cheaper on a T4 and the interpolation of
    position embeddings is a well-established, mild perturbation, whereas
    upsampling fabricates spatial detail the sensor never resolved. The cost is
    that only 16 patch tokens remain, which limits how much the attention can do.

    Expect this arm NOT to beat ResNet-18 here. Transformers are data-hungry and
    ~19k training chips is small; that is the textbook result, not a bug.
    """
    import timm

    net = timm.create_model(
        model_name, pretrained=pretrained, num_classes=0, img_size=img_size, in_chans=3
    )
    if in_channels != 3:
        proj = net.patch_embed.proj
        net.patch_embed.proj = adapt_patch_embed(proj, in_channels, mode=stem_mode)
    feature_dim = net.num_features
    return BackboneClassifier(net, feature_dim, num_classes)


def build_model(arch: str, in_channels: int = NUM_BANDS, num_classes: int = 10, **kw) -> nn.Module:
    """Single entry point so notebooks and configs name models by string."""
    builders = {
        "small_cnn": lambda: SmallCNN(in_channels, num_classes, **kw),
        "resnet18": lambda: build_resnet18(in_channels, num_classes, **kw),
        "vit": lambda: build_vit(in_channels, num_classes, **kw),
    }
    if arch not in builders:
        raise KeyError(f"unknown arch {arch!r}; known: {sorted(builders)}")
    return builders[arch]()


def count_parameters(model: nn.Module, trainable_only: bool = True) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad or not trainable_only)


@torch.no_grad()
def extract_features(model: nn.Module, loader, device: str = "cuda", amp: bool = True):
    """Frozen penultimate features for a whole loader -> (N, D) float32 numpy.

    Used by NB04. Extraction is the slow part of a few-shot study and probing is
    instant, so this is always run once and cached to disk.
    """
    import numpy as np

    model.eval().to(device)
    feats, labels = [], []
    autocast = torch.autocast(device_type="cuda", enabled=amp and device == "cuda")
    for x, y in loader:
        x = x.to(device, non_blocking=True)
        with autocast:
            f = model.forward_features(x) if hasattr(model, "forward_features") else model(x)
        feats.append(f.float().cpu().numpy())
        labels.append(np.asarray(y))
    return np.concatenate(feats), np.concatenate(labels)


def band_mask_model(model: nn.Module, keep: list[int], fill: float = 0.0):
    """Wrap a model so listed input channels survive and the rest are set to `fill`.

    Used for the band-ablation study in NB06. Zeroing after normalisation means
    an ablated band is replaced by its *training mean*, i.e. "no information",
    rather than by a physically impossible zero reflectance.
    """

    class _Masked(nn.Module):
        def __init__(self):
            super().__init__()
            self.inner = model
            self.register_buffer("mask", torch.zeros(1, NUM_BANDS, 1, 1))
            self.mask[:, keep] = 1.0

        def forward(self, x):
            return self.inner(x * self.mask + fill * (1 - self.mask))

    return _Masked()


def describe_band_order() -> str:
    return " ".join(f"{i}:{b}" for i, b in enumerate(BAND_IDS))
