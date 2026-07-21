"""Sentinel-2 band metadata, spectral indices and RGB compositing.

The band table below is the physical description of the sensor; everything in
this project that says "the red band" resolves through it rather than through a
magic integer. Band-order mistakes are the classic silent bug in multispectral
pipelines — they produce plausible-looking but wrong maps — so index lookups go
through `band_index()` and are asserted at pipeline boundaries.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class BandInfo:
    """One Sentinel-2 MSI band.

    Attributes
    ----------
    band_id      : Sentinel-2 identifier, e.g. "B08".
    name         : common name.
    wavelength   : central wavelength in nm (Sentinel-2A).
    resolution   : native ground sampling distance in metres.
    use          : what it is physically used for.
    """

    band_id: str
    name: str
    wavelength: float
    resolution: int
    use: str


# EuroSAT stores 13 bands in this exact order. It is Sentinel-2 L1C band order
# with B8A sitting between B08 and B09. Do not reorder: the tuple index IS the
# channel index of every array in this project.
BANDS: tuple[BandInfo, ...] = (
    BandInfo("B01", "Coastal aerosol", 443.0, 60, "aerosol / atmospheric correction"),
    BandInfo("B02", "Blue", 490.0, 10, "true colour; water penetration"),
    BandInfo("B03", "Green", 560.0, 10, "true colour; vegetation vigour peak"),
    BandInfo("B04", "Red", 665.0, 10, "true colour; chlorophyll absorption"),
    BandInfo("B05", "Red edge 1", 705.0, 20, "vegetation stress / species"),
    BandInfo("B06", "Red edge 2", 740.0, 20, "vegetation stress / species"),
    BandInfo("B07", "Red edge 3", 783.0, 20, "leaf area index"),
    BandInfo("B08", "NIR", 842.0, 10, "biomass, NDVI, land/water boundary"),
    BandInfo("B08A", "Narrow NIR", 865.0, 20, "vegetation, less water-vapour effect"),
    BandInfo("B09", "Water vapour", 945.0, 60, "atmospheric water vapour"),
    BandInfo("B10", "Cirrus", 1375.0, 60, "cirrus cloud detection (L1C only)"),
    BandInfo("B11", "SWIR 1", 1610.0, 20, "soil/vegetation moisture, built-up"),
    BandInfo("B12", "SWIR 2", 2190.0, 20, "burnt area, mineral, built-up"),
)

BAND_IDS: tuple[str, ...] = tuple(b.band_id for b in BANDS)
NUM_BANDS = len(BANDS)

# Groups used for the band-ablation study in NB06. They follow the physics
# (which part of the spectrum) rather than the file layout.
BAND_GROUPS: dict[str, tuple[str, ...]] = {
    "visible": ("B02", "B03", "B04"),
    "red_edge": ("B05", "B06", "B07"),
    "nir": ("B08", "B08A"),
    "swir": ("B11", "B12"),
    "atmospheric": ("B01", "B09", "B10"),
}

# Sentinel-2 L2A (what a real scene download gives you, NB05) has no B10:
# the cirrus band is consumed by the atmospheric correction and dropped.
L2A_MISSING_BANDS: tuple[str, ...] = ("B10",)


def band_index(band_id: str) -> int:
    """Channel index of a band id. Raises rather than returning -1."""
    try:
        return BAND_IDS.index(band_id)
    except ValueError as exc:
        raise KeyError(f"unknown band {band_id!r}; known: {BAND_IDS}") from exc


def band_indices(band_ids) -> list[int]:
    return [band_index(b) for b in band_ids]


RGB_BANDS: tuple[str, str, str] = ("B04", "B03", "B02")
RGB_INDICES: tuple[int, int, int] = tuple(band_indices(RGB_BANDS))  # type: ignore[assignment]


# --------------------------------------------------------------------------
# Spectral indices
# --------------------------------------------------------------------------
def normalized_difference(a: np.ndarray, b: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """(a - b) / (a + b), safe against a zero denominator.

    Both bands read zero over no-data pixels and over deep shadow, so the
    denominator genuinely does hit zero on real scenes. Returning NaN there
    poisons every downstream mean; we return 0.0, which is the neutral value of
    a normalised difference, and it is the caller's job to carry a no-data mask
    if the distinction matters.
    """
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    denom = a + b
    out = np.zeros(np.broadcast(a, b).shape, dtype=np.float64)
    valid = np.abs(denom) > eps
    np.divide(a - b, denom, out=out, where=valid)
    return out


def _channel(x: np.ndarray, band_id: str) -> np.ndarray:
    """Pull one band from a (..., C, H, W) or (..., C) array."""
    x = np.asarray(x)
    idx = band_index(band_id)
    if x.shape[-1] == NUM_BANDS and (x.ndim == 1 or x.shape[-1] != x.shape[0]):
        # feature-vector layout (..., C)
        return x[..., idx]
    return x[..., idx, :, :]


def ndvi(x: np.ndarray) -> np.ndarray:
    """Normalised Difference Vegetation Index, (NIR - Red) / (NIR + Red).

    High for green biomass: chlorophyll absorbs red, leaf mesophyll scatters NIR.
    """
    return normalized_difference(_channel(x, "B08"), _channel(x, "B04"))


def ndwi(x: np.ndarray) -> np.ndarray:
    """McFeeters NDWI, (Green - NIR) / (Green + NIR). High over open water."""
    return normalized_difference(_channel(x, "B03"), _channel(x, "B08"))


def ndbi(x: np.ndarray) -> np.ndarray:
    """Normalised Difference Built-up Index, (SWIR1 - NIR) / (SWIR1 + NIR)."""
    return normalized_difference(_channel(x, "B11"), _channel(x, "B08"))


INDEX_FUNCS = {"NDVI": ndvi, "NDWI": ndwi, "NDBI": ndbi}


def spectral_index_features(x: np.ndarray) -> np.ndarray:
    """Per-chip mean NDVI/NDWI/NDBI for a batch (N, C, H, W) -> (N, 3)."""
    x = np.asarray(x, dtype=np.float64)
    assert x.ndim == 4 and x.shape[1] == NUM_BANDS, f"expected (N,13,H,W), got {x.shape}"
    feats = [INDEX_FUNCS[k](x).reshape(x.shape[0], -1).mean(axis=1) for k in INDEX_FUNCS]
    return np.stack(feats, axis=1)


def band_statistics_features(x: np.ndarray) -> np.ndarray:
    """Per-chip mean and std of every band for a batch (N, C, H, W) -> (N, 2C)."""
    x = np.asarray(x, dtype=np.float64)
    assert x.ndim == 4, f"expected (N,C,H,W), got {x.shape}"
    flat = x.reshape(x.shape[0], x.shape[1], -1)
    return np.concatenate([flat.mean(axis=2), flat.std(axis=2)], axis=1)


def classical_features(x: np.ndarray) -> np.ndarray:
    """Arm 0 feature vector: 13 means + 13 stds + 3 index means = 29 features."""
    feats = np.concatenate([band_statistics_features(x), spectral_index_features(x)], axis=1)
    assert feats.shape[1] == 2 * NUM_BANDS + 3
    return feats


CLASSICAL_FEATURE_NAMES: tuple[str, ...] = tuple(
    [f"{b}_mean" for b in BAND_IDS] + [f"{b}_std" for b in BAND_IDS] + ["NDVI", "NDWI", "NDBI"]
)


# --------------------------------------------------------------------------
# Rendering
# --------------------------------------------------------------------------
def percentile_stretch(
    x: np.ndarray, low: float = 2.0, high: float = 98.0, per_channel: bool = True
) -> np.ndarray:
    """Linear contrast stretch between the given percentiles, clipped to [0, 1].

    Why this exists: surface reflectance over land occupies a small, low part of
    the sensor's dynamic range, so raw values rendered directly look almost
    black. A 2nd-98th percentile stretch maps the populated part of the
    histogram onto the display range while discarding specular and shadow
    outliers. Every satellite RGB you have ever seen has had this done to it.

    Applied per channel by default, which also removes the blue-ish cast caused
    by Rayleigh scattering being strongest in the blue band. Pass
    `per_channel=False` to preserve relative band magnitudes.
    """
    x = np.asarray(x, dtype=np.float64)
    if per_channel and x.ndim >= 3:
        axes = tuple(range(1, x.ndim))
        lo = np.percentile(x, low, axis=axes, keepdims=True)
        hi = np.percentile(x, high, axis=axes, keepdims=True)
    else:
        lo = np.percentile(x, low)
        hi = np.percentile(x, high)
    scale = np.where(np.asarray(hi - lo) > 1e-12, hi - lo, 1.0)
    return np.clip((x - lo) / scale, 0.0, 1.0)


def _composite(x: np.ndarray, band_ids, stretch: bool, low: float, high: float) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    assert x.ndim == 3 and x.shape[0] == NUM_BANDS, f"expected (13,H,W), got {x.shape}"
    comp = x[band_indices(band_ids)]
    if stretch:
        comp = percentile_stretch(comp, low, high)
    out = np.transpose(comp, (1, 2, 0))  # -> (H, W, 3) for matplotlib
    assert out.shape[-1] == 3
    return out


def true_color(x: np.ndarray, stretch: bool = True, low: float = 2.0, high: float = 98.0):
    """B04/B03/B02 — what a human eye in orbit would roughly see."""
    return _composite(x, ("B04", "B03", "B02"), stretch, low, high)


def false_color_nir(x: np.ndarray, stretch: bool = True, low: float = 2.0, high: float = 98.0):
    """B08/B04/B03 — healthy vegetation glows red (high NIR into the red gun)."""
    return _composite(x, ("B08", "B04", "B03"), stretch, low, high)


def false_color_swir(x: np.ndarray, stretch: bool = True, low: float = 2.0, high: float = 98.0):
    """B12/B08/B04 — moisture and burnt/bare ground; water is near black."""
    return _composite(x, ("B12", "B08", "B04"), stretch, low, high)


COMPOSITES = {
    "true_color": (true_color, "B04/B03/B02 (true colour)"),
    "false_color_nir": (false_color_nir, "B08/B04/B03 (NIR false colour)"),
    "false_color_swir": (false_color_swir, "B12/B08/B04 (SWIR false colour)"),
}


def to_uint8_rgb(x: np.ndarray, low: float = 2.0, high: float = 98.0) -> np.ndarray:
    """(13, H, W) reflectance -> (H, W, 3) uint8, ready for a photo-domain model.

    This is the exact adapter used to feed CLIP and SAM in NB03/NB05, and it is
    where ten of the thirteen bands are thrown away.
    """
    return (true_color(x, stretch=True, low=low, high=high) * 255.0).round().astype(np.uint8)


def band_table_markdown() -> str:
    """Markdown table of the sensor. Rendered in NB01 instead of being typed."""
    lines = [
        "| Band | Name | Central wavelength (nm) | Native GSD (m) | Physical use |",
        "|---|---|---|---|---|",
    ]
    lines += [
        f"| {b.band_id} | {b.name} | {b.wavelength:.0f} | {b.resolution} | {b.use} |"
        for b in BANDS
    ]
    return "\n".join(lines)
