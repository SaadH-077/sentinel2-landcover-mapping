"""Fetching a real Sentinel-2 scene: STAC search, windowed COG reads, cloud mask.

WHAT STAC IS. A SpatioTemporal Asset Catalog is a JSON specification for
describing geospatial assets, with a search API on top. It lets you ask "which
Sentinel-2 scenes cover this box, in this date range, with less than 10% cloud"
and get back metadata and asset URLs — without downloading a single pixel first.
Before STAC this meant a portal, a login, and a 700 MB SAFE archive.

WHY WINDOWED COG READS ARE THE WHOLE TRICK. A Sentinel-2 tile covers 110x110 km;
the full 10 m band set is hundreds of megabytes per scene. A Cloud-Optimised
GeoTIFF is internally tiled and laid out so that an HTTP range request can fetch
just the bytes for one window. Reading a 15x15 km AOI therefore transfers a few
megabytes instead of a few hundred. This is the single most important practical
technique for working with satellite data at scale, and it is the only reason
notebook 05 runs inside a free Colab session.

Every rasterio/pystac import here is function-local so that this module can be
imported (and the rest of the package tested) without the geospatial stack
installed.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .bands import BAND_IDS, NUM_BANDS

# Verified-at-build-time endpoint; if it is unreachable, NB05 says so and uses
# the documented fallback rather than inventing a different URL.
EARTH_SEARCH_URL = "https://earth-search.aws.element84.com/v1"
COLLECTION = "sentinel-2-l2a"

# Earth Search asset keys for L2A. Note B10 is absent: the cirrus band is
# consumed by the atmospheric correction and is not part of an L2A product.
# The model was trained on 13-band L1C, so this gap has to be handled
# explicitly — see `stack_bands`.
ASSET_KEYS: dict[str, str] = {
    "B01": "coastal",
    "B02": "blue",
    "B03": "green",
    "B04": "red",
    "B05": "rededge1",
    "B06": "rededge2",
    "B07": "rededge3",
    "B08": "nir",
    "B08A": "nir08",
    "B09": "nir09",
    "B11": "swir16",
    "B12": "swir22",
}
SCL_ASSET = "scl"

# Scene Classification Layer codes (L2A). See the Sentinel-2 L2A ATBD.
SCL_LABELS: dict[int, str] = {
    0: "no data",
    1: "saturated / defective",
    2: "dark area pixels",
    3: "cloud shadow",
    4: "vegetation",
    5: "not vegetated",
    6: "water",
    7: "unclassified",
    8: "cloud medium probability",
    9: "cloud high probability",
    10: "thin cirrus",
    11: "snow / ice",
}
# Masked by default: clouds, their shadows, cirrus, and invalid pixels. Class 2
# (dark area) and 11 (snow) are left in — they are real surfaces, not artefacts.
DEFAULT_CLOUD_CLASSES: tuple[int, ...] = (0, 1, 3, 8, 9, 10)


@dataclass
class SceneWindow:
    """A stack of bands read over one AOI window, with its georeferencing."""

    array: np.ndarray            # (C, H, W) float32 reflectance
    band_ids: tuple[str, ...]
    transform: object            # affine.Affine, 10 m grid
    crs: object                  # rasterio CRS (a UTM zone)
    scl: np.ndarray | None = None
    item_id: str = ""
    datetime: str = ""
    cloud_cover: float = float("nan")

    @property
    def shape(self) -> tuple[int, int]:
        return self.array.shape[-2:]


def search_scenes(
    bbox: tuple[float, float, float, float],
    date_range: str,
    max_cloud: float = 10.0,
    limit: int = 20,
    url: str = EARTH_SEARCH_URL,
    collection: str = COLLECTION,
) -> list:
    """STAC search over bbox (lon/lat, EPSG:4326) and an ISO date range.

    Returns items sorted by cloud cover ascending. The selection rule used in
    NB05 is "least cloudy item in the window that fully covers the AOI", which
    is stated in the notebook rather than left implicit — picking the newest
    instead would be a different and equally defensible rule, and the reader
    should know which one produced the map.
    """
    from pystac_client import Client

    client = Client.open(url)
    search = client.search(
        collections=[collection],
        bbox=list(bbox),
        datetime=date_range,
        query={"eo:cloud_cover": {"lt": max_cloud}},
        limit=limit,
    )
    items = list(search.items())
    return sorted(items, key=lambda it: it.properties.get("eo:cloud_cover", 100.0))


def summarize_items(items) -> list[dict]:
    return [
        {
            "id": it.id,
            "datetime": str(it.datetime),
            "cloud_cover": it.properties.get("eo:cloud_cover"),
            "platform": it.properties.get("platform"),
            "crs": it.properties.get("proj:epsg"),
        }
        for it in items
    ]


def read_window(
    href: str,
    bbox: tuple[float, float, float, float],
    target_shape: tuple[int, int] | None = None,
    resampling: str = "bilinear",
):
    """Read only the AOI window of one COG, reprojecting the bbox into its CRS.

    RESAMPLING CHOICE. The 20 m and 60 m bands must land on the same 10 m grid
    as B02/B03/B04/B08. Bilinear is correct for continuous reflectance: it
    interpolates a physical quantity that genuinely varies smoothly. Nearest
    neighbour is correct for *categorical* rasters — which is why the SCL mask
    below is read with nearest, and reading it bilinearly would blend cloud
    class 9 and vegetation class 4 into a meaningless class 6.5.
    """
    import rasterio
    from rasterio.enums import Resampling
    from rasterio.warp import transform_bounds
    from rasterio.windows import from_bounds

    method = {"bilinear": Resampling.bilinear, "nearest": Resampling.nearest}[resampling]
    with rasterio.open(href) as src:
        left, bottom, right, top = transform_bounds("EPSG:4326", src.crs, *bbox, densify_pts=21)
        window = from_bounds(left, bottom, right, top, transform=src.transform)
        out_shape = target_shape or (int(round(window.height)), int(round(window.width)))
        data = src.read(1, window=window, out_shape=out_shape, resampling=method)
        transform = src.window_transform(window)
        # Rescale the affine transform to the output grid when the read was resampled.
        scale_x = window.width / out_shape[1]
        scale_y = window.height / out_shape[0]
        transform = transform * rasterio.Affine.scale(scale_x, scale_y)
        return data, transform, src.crs


def load_scene(
    item,
    bbox: tuple[float, float, float, float],
    band_ids: tuple[str, ...] = BAND_IDS,
    scale: float = 1e-4,
    include_scl: bool = True,
) -> SceneWindow:
    """Read the AOI window of every requested band onto a common 10 m grid.

    BAND ORDER IS ASSERTED, not assumed. Feeding a model the same 13 arrays in
    the wrong order produces a map that looks entirely plausible and is entirely
    wrong, with no error anywhere. It is the classic silent bug of this domain.

    Bands the product does not carry (B10 on L2A) are filled with zeros AFTER
    the scene is standardised, i.e. with the "no information" value, and are
    listed in the returned band_ids so the caller cannot lose track of them.

    `scale` converts L2A integer DN to reflectance (DN * 1e-4). EuroSAT L1C is
    stored on the same 1e-4 convention, so the two are at least in the same
    units — but NOT the same distribution, which is the whole subject of NB05's
    domain-shift section.
    """
    # The 10 m reference grid comes from B04, then every other band is read to
    # exactly that shape, which guarantees perfect alignment by construction
    # rather than by hoping the windows round the same way.
    ref, transform, crs = read_window(item.assets[ASSET_KEYS["B04"]].href, bbox)
    target_shape = ref.shape
    layers, missing = [], []
    for band in band_ids:
        if band not in ASSET_KEYS or ASSET_KEYS[band] not in item.assets:
            missing.append(band)
            layers.append(np.zeros(target_shape, dtype=np.float32))
            continue
        data, _, _ = read_window(item.assets[ASSET_KEYS[band]].href, bbox, target_shape)
        layers.append(data.astype(np.float32) * scale)

    array = np.stack(layers).astype(np.float32)
    assert array.shape[0] == len(band_ids), f"{array.shape[0]} layers for {len(band_ids)} bands"
    if len(band_ids) == NUM_BANDS:
        assert tuple(band_ids) == BAND_IDS, (
            f"band order {tuple(band_ids)} != training order {BAND_IDS} — refusing to continue"
        )

    scl = None
    if include_scl and SCL_ASSET in item.assets:
        scl, _, _ = read_window(item.assets[SCL_ASSET].href, bbox, target_shape, resampling="nearest")
        scl = scl.astype(np.uint8)

    scene = SceneWindow(
        array=array,
        band_ids=tuple(band_ids),
        transform=transform,
        crs=crs,
        scl=scl,
        item_id=item.id,
        datetime=str(item.datetime),
        cloud_cover=float(item.properties.get("eo:cloud_cover", float("nan"))),
    )
    scene.missing_bands = tuple(missing)  # type: ignore[attr-defined]
    return scene


def cloud_mask(scl: np.ndarray, classes: tuple[int, ...] = DEFAULT_CLOUD_CLASSES) -> np.ndarray:
    """True where the pixel is unusable (cloud, shadow, cirrus, no-data).

    WHY THIS DOMINATES OPTICAL REMOTE SENSING. Roughly two thirds of the Earth
    is cloud-covered at any moment, and persistently so in exactly the tropical
    regions where deforestation monitoring matters most. That is why operational
    systems either composite many dates to build a cloud-free mosaic, or fall
    back on Sentinel-1 radar, which sees through cloud entirely at the cost of a
    much harder-to-interpret signal.
    """
    scl = np.asarray(scl)
    return np.isin(scl, np.asarray(classes))


def scl_class_fractions(scl: np.ndarray) -> dict[str, float]:
    scl = np.asarray(scl)
    total = scl.size
    return {
        SCL_LABELS.get(int(c), f"class {int(c)}"): float(n / total)
        for c, n in zip(*np.unique(scl, return_counts=True))
    }


def scene_band_stats(scene: np.ndarray, valid_mask: np.ndarray | None = None) -> dict:
    """Per-band mean/std of the scene itself, over valid pixels only.

    This is the ingredient for the NB05 mitigation: standardising the scene with
    its OWN statistics instead of EuroSAT's absorbs a global L1C->L2A offset,
    at the cost of assuming the AOI's land-cover mix resembles the training
    mix — an assumption worth stating, since it fails for, say, an all-water AOI.
    """
    arr = np.asarray(scene, dtype=np.float64)
    c = arr.shape[0]
    flat = arr.reshape(c, -1)
    if valid_mask is not None:
        flat = flat[:, ~np.asarray(valid_mask).ravel()]
    return {
        "mean": flat.mean(axis=1).tolist(),
        "std": flat.std(axis=1).tolist(),
        "n_pixels": int(flat.shape[1]),
    }
