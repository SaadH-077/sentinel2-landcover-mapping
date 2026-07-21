"""Augmentation for overhead multispectral imagery.

The recipe here deliberately differs from a standard ImageNet recipe, for two
domain reasons that are worth being able to defend:

1. NO CANONICAL "UP". A photograph of a dog has a correct orientation; a 64x64
   nadir view of a forest does not. The full dihedral group of the square
   (4 rotations x optional flip, 8 elements) is therefore a label-preserving
   symmetry of this data. That is 8x free augmentation that a natural-image
   recipe cannot use.

2. PIXEL VALUES ARE MEASUREMENTS. Each channel is a calibrated reflectance, and
   the *ratios between channels* are the physical signal that NDVI and friends
   are built from. Per-channel colour jitter destroys exactly that signal — it
   would be like adding noise to the units of a sensor reading. So brightness
   variation, if used at all, is applied identically across all bands, which is
   the correct model of an illumination/atmospheric change rather than an
   arbitrary distortion.
"""

from __future__ import annotations

import torch


class DihedralAugment:
    """Random element of the 8-element symmetry group of the square.

    Equivalent to random horizontal/vertical flips plus random 90-degree
    rotations, but drawn as a single uniform choice over the group, which makes
    the sampling distribution explicit.
    """

    def __init__(self, p: float = 1.0):
        self.p = p

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        if torch.rand(()) > self.p:
            return x
        k = int(torch.randint(0, 4, ()))
        x = torch.rot90(x, k, dims=(-2, -1))
        if bool(torch.randint(0, 2, ())):
            x = torch.flip(x, dims=(-1,))
        return x.contiguous()


class RandomResizedCropSmall:
    """Scale/translation jitter via a random crop resized back to the input size.

    Kept mild (scale >= 0.7): EuroSAT chips are only 64x64 = 640 m across, so an
    aggressive crop can remove the very object that defines the label — crop a
    'Highway' chip hard enough and the road leaves the frame, giving a wrong
    label rather than an augmented one.
    """

    def __init__(self, size: int = 64, scale: tuple[float, float] = (0.7, 1.0)):
        self.size, self.scale = size, scale

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        import torch.nn.functional as F

        _, h, w = x.shape
        s = float(torch.empty(()).uniform_(*self.scale))
        ch, cw = max(8, int(round(h * s))), max(8, int(round(w * s)))
        top = int(torch.randint(0, h - ch + 1, ()))
        left = int(torch.randint(0, w - cw + 1, ()))
        crop = x[:, top : top + ch, left : left + cw]
        if (ch, cw) == (self.size, self.size):
            return crop
        return F.interpolate(
            crop.unsqueeze(0).float(), size=(self.size, self.size),
            mode="bilinear", align_corners=False,
        ).squeeze(0)


class IlluminationJitter:
    """Multiply ALL bands by one shared factor near 1.

    Models a global illumination / atmospheric-transmission change, which is a
    physically plausible nuisance. Because the factor is shared, every band
    ratio — and therefore every spectral index — is preserved exactly. This is
    the safe alternative to colour jitter on reflectance data.
    """

    def __init__(self, strength: float = 0.1, p: float = 0.5):
        self.strength, self.p = strength, p

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        if torch.rand(()) > self.p:
            return x
        factor = 1.0 + float(torch.empty(()).uniform_(-self.strength, self.strength))
        return x * factor


class Compose:
    def __init__(self, transforms):
        self.transforms = list(transforms)

    def __call__(self, x):
        for t in self.transforms:
            x = t(x)
        return x

    def __repr__(self) -> str:
        inner = ", ".join(type(t).__name__ for t in self.transforms)
        return f"Compose({inner})"


def train_transform(size: int = 64, crop: bool = True, illumination: bool = True) -> Compose:
    ts: list = [DihedralAugment()]
    if crop:
        ts.append(RandomResizedCropSmall(size))
    if illumination:
        ts.append(IlluminationJitter())
    return Compose(ts)


def eval_transform() -> Compose:
    """No augmentation at evaluation. Named for symmetry and to make the
    absence of test-time augmentation an explicit choice rather than an omission."""
    return Compose([])
