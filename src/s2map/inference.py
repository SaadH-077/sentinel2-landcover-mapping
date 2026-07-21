"""Sliding-window inference over a full scene, stitching, and GeoTIFF output.

THE SHAPE OF THE PROBLEM. The model is a *chip* classifier: 64x64 in, one class
out. A scene is ~1500x1500 pixels. So the scene is cut into overlapping 64x64
tiles, each tile gets one probability vector, and that vector is painted across
the tile's footprint. Overlapping tiles are averaged.

WHY OVERLAP-AND-AVERAGE RATHER THAN A DISJOINT GRID. A tile at the edge of an
object sees truncated context and often predicts differently from the tile that
contains the object centred. With a disjoint grid those disagreements land on
tile boundaries and produce a visibly blocky, gridded map. With stride 32 every
interior pixel is covered by four tiles whose contexts are shifted, and
averaging their probabilities smooths the seam away. It costs 4x the forward
passes, which on a T4 is seconds.

WHAT IT CANNOT FIX. The output resolution is still the tile, not the pixel: a
64x64 chip classifier can never produce a boundary sharper than ~32 pixels
(320 m). That is exactly the limitation NB05 addresses by combining these
predictions with SAM's segment boundaries.

rasterio is imported inside the functions that need it, so the tiling logic —
the part with the shape bugs — is testable without a geospatial install.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np


# --------------------------------------------------------------------------
# Tiling
# --------------------------------------------------------------------------
def pad_for_tiling(array: np.ndarray, tile: int, stride: int) -> tuple[np.ndarray, tuple[int, int]]:
    """Reflection-pad a (C, H, W) array so a whole number of tiles fits exactly.

    Reflection rather than zeros: a zero border is a physically impossible
    reflectance and would create a hard artificial edge that the model reads as
    a real feature, systematically corrupting the predictions in the outermost
    tile. Reflecting continues the local texture instead.
    """
    array = np.asarray(array)
    assert array.ndim == 3, f"expected (C,H,W), got {array.shape}"
    _, h, w = array.shape

    def pad_amount(size: int) -> int:
        if size <= tile:
            return tile - size
        extra = (size - tile) % stride
        return 0 if extra == 0 else stride - extra

    ph, pw = pad_amount(h), pad_amount(w)
    if ph == 0 and pw == 0:
        return array, (0, 0)
    padded = np.pad(array, ((0, 0), (0, ph), (0, pw)), mode="reflect")
    return padded, (ph, pw)


def tile_positions(height: int, width: int, tile: int, stride: int) -> list[tuple[int, int]]:
    """Top-left corners covering the whole array, last row/column clamped inward.

    Clamping means the final tile may overlap its neighbour by more than
    `stride`. That is deliberate: the alternative — dropping the remainder —
    leaves an unpredicted strip on the right and bottom edges of the map.
    """
    tops = list(range(0, max(height - tile, 0) + 1, stride))
    lefts = list(range(0, max(width - tile, 0) + 1, stride))
    if tops[-1] + tile < height:
        tops.append(height - tile)
    if lefts[-1] + tile < width:
        lefts.append(width - tile)
    return [(top, left) for top in tops for left in lefts]


def sliding_window_predict(
    scene: np.ndarray,
    predict_fn,
    num_classes: int,
    tile: int = 64,
    stride: int = 32,
    batch_size: int = 256,
    progress=None,
) -> np.ndarray:
    """Dense per-pixel class probabilities for a (C, H, W) scene -> (K, H, W).

    `predict_fn` takes a float32 (B, C, tile, tile) numpy batch and returns
    (B, K) probabilities. Keeping it a plain callable means this function is
    framework-agnostic and can be unit-tested with a constant dummy model, which
    is how the seam-averaging logic is verified.
    """
    scene = np.asarray(scene, dtype=np.float32)
    _, h0, w0 = scene.shape
    padded, _ = pad_for_tiling(scene, tile, stride)
    _, hp, wp = padded.shape

    prob_sum = np.zeros((num_classes, hp, wp), dtype=np.float64)
    weight = np.zeros((1, hp, wp), dtype=np.float64)

    positions = tile_positions(hp, wp, tile, stride)
    iterator = progress(positions) if progress is not None else positions
    batch, coords = [], []

    def flush():
        if not batch:
            return
        probs = np.asarray(predict_fn(np.stack(batch)), dtype=np.float64)
        assert probs.shape == (len(batch), num_classes), (
            f"predict_fn returned {probs.shape}, expected {(len(batch), num_classes)}"
        )
        for (top, left), p in zip(coords, probs):
            prob_sum[:, top : top + tile, left : left + tile] += p[:, None, None]
            weight[:, top : top + tile, left : left + tile] += 1.0
        batch.clear()
        coords.clear()

    for top, left in iterator:
        batch.append(padded[:, top : top + tile, left : left + tile])
        coords.append((top, left))
        if len(batch) >= batch_size:
            flush()
    flush()

    assert weight.min() > 0, "some pixels were never covered by a tile"
    probs = (prob_sum / weight)[:, :h0, :w0]
    assert probs.shape == (num_classes, h0, w0), f"stitched {probs.shape}, expected {(num_classes, h0, w0)}"
    return probs.astype(np.float32)


def torch_predict_fn(model, device: str = "cuda", amp: bool = True):
    """Adapt a torch classifier into the `predict_fn` callable above."""
    import torch

    model.eval().to(device)

    def fn(batch: np.ndarray) -> np.ndarray:
        with torch.no_grad():
            x = torch.from_numpy(np.ascontiguousarray(batch)).to(device)
            with torch.autocast(device_type="cuda", enabled=amp and device == "cuda"):
                logits = model(x)
            return torch.softmax(logits.float(), dim=1).cpu().numpy()

    return fn


# --------------------------------------------------------------------------
# Post-processing
# --------------------------------------------------------------------------
def apply_mask(class_map: np.ndarray, mask: np.ndarray, fill: int = 255) -> np.ndarray:
    """Set masked (e.g. cloudy) pixels to a nodata value instead of a class.

    Predicting land cover under a cloud is not a prediction, it is a guess about
    the cloud. Marking it nodata is the honest output.
    """
    out = np.asarray(class_map).copy()
    out[np.asarray(mask)] = fill
    return out


def class_areas(
    class_map: np.ndarray,
    class_names,
    pixel_size_m: float = 10.0,
    nodata: int = 255,
) -> dict[str, dict]:
    """Predicted area per class in hectares.

    This is the point of producing a georeferenced map rather than a picture:
    the output becomes a quantity someone can act on. One Sentinel-2 pixel is
    10x10 m = 0.01 ha, so the arithmetic is trivial — the georeferencing is what
    makes it meaningful.
    """
    class_map = np.asarray(class_map)
    valid = class_map != nodata
    total_valid = int(valid.sum())
    pixel_ha = (pixel_size_m**2) / 10_000.0
    out = {}
    for i, name in enumerate(class_names):
        n = int((class_map == i).sum())
        out[name] = {
            "pixels": n,
            "hectares": round(n * pixel_ha, 2),
            "fraction": round(n / total_valid, 5) if total_valid else 0.0,
        }
    out["_nodata"] = {"pixels": int((~valid).sum()), "hectares": round(int((~valid).sum()) * pixel_ha, 2)}
    return out


def majority_vote_by_segment(
    class_map: np.ndarray, segments: np.ndarray, num_classes: int, nodata: int = 255
) -> np.ndarray:
    """Assign every SAM segment the majority class predicted inside it.

    THE IDEA. SAM produces crisp, object-aligned boundaries with no semantics —
    it knows *where* things are and not *what* they are. The chip classifier
    produces semantics with boundaries quantised to the tile grid. They fail in
    exactly complementary ways, so using SAM's segments as superpixels and
    filling each one with the classifier's majority vote gives crisp boundaries
    AND labels. No training, no new model.
    """
    class_map = np.asarray(class_map)
    segments = np.asarray(segments)
    assert class_map.shape == segments.shape, f"{class_map.shape} vs {segments.shape}"
    out = class_map.copy()
    for seg_id in np.unique(segments):
        m = segments == seg_id
        values = class_map[m]
        values = values[values != nodata]
        if values.size == 0:
            continue
        out[m] = np.bincount(values, minlength=num_classes).argmax()
    return out


def agreement_matrix(a: np.ndarray, b: np.ndarray, n_a: int, n_b: int) -> np.ndarray:
    """Cross-tabulation of two categorical maps.

    Named "agreement", not "confusion": in NB05 this compares our prediction
    against ESA WorldCover, which is another *model's* product and not ground
    truth. Calling the result accuracy would silently promote one map to truth.
    """
    a, b = np.asarray(a).ravel(), np.asarray(b).ravel()
    valid = (a < n_a) & (b < n_b)
    return np.bincount(a[valid] * n_b + b[valid], minlength=n_a * n_b).reshape(n_a, n_b)


# --------------------------------------------------------------------------
# Georeferenced output
# --------------------------------------------------------------------------
def write_geotiff(
    path: Path | str,
    array: np.ndarray,
    transform,
    crs,
    dtype: str = "uint8",
    nodata: int | float | None = 255,
    class_names=None,
) -> Path:
    """Write a (H, W) or (C, H, W) array as a GeoTIFF carrying CRS and transform.

    A PNG is a picture; a GeoTIFF is a map. The affine transform and CRS are
    what let this file be dropped into QGIS on top of other layers, have areas
    computed from it in real units, and be intersected with a cadastral or
    protected-area boundary. Carrying the georeferencing through from the source
    imagery — rather than saving a screenshot — is what makes this geospatial
    work rather than image processing.
    """
    import rasterio

    array = np.asarray(array)
    if array.ndim == 2:
        array = array[None]
    count, height, width = array.shape
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        height=height,
        width=width,
        count=count,
        dtype=dtype,
        crs=crs,
        transform=transform,
        nodata=nodata,
        compress="deflate",
    ) as dst:
        dst.write(array.astype(dtype))
        if class_names is not None:
            # Embedded so the class meaning travels with the file rather than
            # living only in this repo's README.
            dst.update_tags(class_names=",".join(class_names))
            dst.write_colormap(1, _colormap(class_names))
    return path


def _colormap(class_names) -> dict[int, tuple[int, int, int, int]]:
    from . import config as cfg

    def rgb(hex_color: str) -> tuple[int, int, int, int]:
        h = hex_color.lstrip("#")
        return (int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16), 255)

    cmap = {i: rgb(cfg.CLASS_COLORS.get(n, "#888888")) for i, n in enumerate(class_names)}
    cmap[255] = (0, 0, 0, 0)  # nodata -> transparent
    return cmap
