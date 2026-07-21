"""EuroSAT (13-band) loading, caching, splitting and normalisation.

Loading strategy, in order, with each fallback recorded in `DatasetBundle.notes`
so the notebook can print exactly which path was taken on this machine:

  1. an already-built local cache (outputs/eurosat_ms.npy)          -- instant
  2. a local extracted EuroSAT directory of GeoTIFFs                -- offline
  3. torchgeo's EuroSAT loader with download=True                   -- primary
  4. the HuggingFace hub copies (blanchon/EuroSAT_MSI, torchgeo/eurosat)

Nothing here invents a URL. If every path fails the caller gets a
`DatasetUnavailable` with the list of what was tried, and the notebook prints it.

The cache is a plain .npy memmap of shape (N, 13, 64, 64), dtype uint16
(~2.9 GB). Memmapping rather than loading keeps peak RAM well inside Colab's
~12 GB even though the array itself is larger than a comfortable fraction of it.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from . import config as cfg
from .bands import BAND_IDS, NUM_BANDS

CHIP_SIZE = 64
CACHE_NAME = "eurosat_ms.npy"
LABELS_NAME = "eurosat_labels.npy"
MANIFEST_NAME = "eurosat_manifest.json"


class DatasetUnavailable(RuntimeError):
    """Every documented acquisition path failed. Carries what was attempted."""


@dataclass
class DatasetBundle:
    """The dataset as used by every notebook.

    X is a memmap; index it, do not `np.asarray` the whole thing.
    """

    X: np.ndarray                      # (N, 13, 64, 64) uint16, L1C DN / reflectance*1e4
    y: np.ndarray                      # (N,) int64, index into cfg.CLASS_NAMES
    class_names: tuple[str, ...]
    source: str
    notes: list[str] = field(default_factory=list)

    def __len__(self) -> int:
        return int(self.X.shape[0])

    def describe(self) -> str:
        counts = np.bincount(self.y, minlength=len(self.class_names))
        lines = [
            f"source          {self.source}",
            f"samples         {len(self):,}",
            f"chip shape      {tuple(self.X.shape[1:])}  dtype={self.X.dtype}",
            f"bands           {', '.join(BAND_IDS)}",
            "class counts:",
        ]
        lines += [f"  {n:<22} {c:>6,}" for n, c in zip(self.class_names, counts)]
        return "\n".join(lines)


# --------------------------------------------------------------------------
# Acquisition
# --------------------------------------------------------------------------
def _cache_paths(cache_dir: Path) -> tuple[Path, Path, Path]:
    return cache_dir / CACHE_NAME, cache_dir / LABELS_NAME, cache_dir / MANIFEST_NAME


def _load_cache(cache_dir: Path) -> DatasetBundle | None:
    xp, yp, mp = _cache_paths(cache_dir)
    if not (xp.exists() and yp.exists() and mp.exists()):
        return None
    manifest = json.loads(mp.read_text(encoding="utf-8"))
    n = manifest["n_samples"]
    X = np.load(xp, mmap_mode="r")
    y = np.load(yp)
    if X.shape[0] != n or y.shape[0] != n:
        return None
    return DatasetBundle(
        X=X,
        y=y,
        class_names=tuple(manifest["class_names"]),
        source=manifest["source"],
        notes=[f"loaded from cache {xp}"],
    )


def _finalize_cache(cache_dir: Path, y: np.ndarray, source: str) -> DatasetBundle:
    """Write labels + manifest for an image memmap already written at CACHE_NAME.

    The image array is built in place with `open_memmap` by the loaders above,
    so it must NOT be re-saved here — that would mean np.save-ing a file onto
    the memmap currently backing it.
    """
    cache_dir.mkdir(parents=True, exist_ok=True)
    xp, yp, mp = _cache_paths(cache_dir)
    X = np.load(xp, mmap_mode="r")
    np.save(yp, y)
    mp.write_text(
        json.dumps(
            {
                "n_samples": int(X.shape[0]),
                "shape": list(X.shape),
                "dtype": str(X.dtype),
                "band_ids": list(BAND_IDS),
                "class_names": list(cfg.CLASS_NAMES),
                "source": source,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return DatasetBundle(np.load(xp, mmap_mode="r"), y, cfg.CLASS_NAMES, source, [f"cached to {xp}"])


def _find_tif_root(root: Path) -> Path | None:
    """Locate the directory whose children are the 10 class folders of GeoTIFFs."""
    if not root.exists():
        return None
    candidates = [root, *(p for p in root.rglob("*") if p.is_dir())]
    for d in candidates:
        subdirs = {p.name for p in d.iterdir() if p.is_dir()} if d.is_dir() else set()
        if set(cfg.CLASS_NAMES).issubset(subdirs):
            sample = next((d / cfg.CLASS_NAMES[0]).glob("*.tif"), None)
            if sample is not None:
                return d
    return None


def _load_from_tif_dir(tif_root: Path, source: str, cache_dir: Path) -> DatasetBundle:
    """Read a directory tree of EuroSAT GeoTIFFs into the memmap cache."""
    import rasterio

    files: list[Path] = []
    labels: list[int] = []
    for label, name in enumerate(cfg.CLASS_NAMES):
        class_files = sorted((tif_root / name).glob("*.tif"))
        if not class_files:
            raise DatasetUnavailable(f"class directory {tif_root / name} contains no .tif files")
        files.extend(class_files)
        labels.extend([label] * len(class_files))

    n = len(files)
    cache_dir.mkdir(parents=True, exist_ok=True)
    xp, _, _ = _cache_paths(cache_dir)
    X = np.lib.format.open_memmap(
        xp, mode="w+", dtype=np.uint16, shape=(n, NUM_BANDS, CHIP_SIZE, CHIP_SIZE)
    )
    for i, path in enumerate(files):
        with rasterio.open(path) as src:
            arr = src.read()  # (13, 64, 64)
        if arr.shape != (NUM_BANDS, CHIP_SIZE, CHIP_SIZE):
            raise DatasetUnavailable(f"{path} has shape {arr.shape}, expected (13,64,64)")
        X[i] = arr.astype(np.uint16)
    X.flush()
    del X  # release the write memmap before reopening read-only
    y = np.asarray(labels, dtype=np.int64)
    bundle = _finalize_cache(cache_dir, y, source)
    bundle.notes.insert(0, f"read {n:,} GeoTIFFs from {tif_root}")
    return bundle


def _try_torchgeo(data_dir: Path, cache_dir: Path) -> DatasetBundle:
    """torchgeo's EuroSAT loader (all 13 bands), with auto-download.

    torchgeo ships its own train/val/test text files; we ignore them and rebuild
    our own stratified split so that every notebook here shares one split
    definition. The loader is used purely as a verified downloader.
    """
    from torchgeo.datasets import EuroSAT

    EuroSAT(root=str(data_dir), split="train", download=True, checksum=False)
    tif_root = _find_tif_root(data_dir)
    if tif_root is None:
        raise DatasetUnavailable(f"torchgeo download finished but no class dirs under {data_dir}")
    return _load_from_tif_dir(tif_root, "torchgeo.datasets.EuroSAT (13-band, auto-download)", cache_dir)


def _try_huggingface(cache_dir: Path, repo: str) -> DatasetBundle:
    """HuggingFace hub fallback. Handles both the MSI-tif and array layouts."""
    from datasets import load_dataset

    ds = load_dataset(repo, split="train")
    n = len(ds)
    xp, _, _ = _cache_paths(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    X = np.lib.format.open_memmap(
        xp, mode="w+", dtype=np.uint16, shape=(n, NUM_BANDS, CHIP_SIZE, CHIP_SIZE)
    )
    labels = np.zeros(n, dtype=np.int64)
    feature_names = list(ds.features)
    label_key = "label" if "label" in feature_names else "labels"
    image_key = next(k for k in ("image", "img", "tif", "bands") if k in feature_names)
    names = ds.features[label_key].names if hasattr(ds.features[label_key], "names") else None

    for i, rec in enumerate(ds):
        arr = np.asarray(rec[image_key])
        if arr.ndim == 3 and arr.shape[-1] == NUM_BANDS:  # (H, W, C) -> (C, H, W)
            arr = np.transpose(arr, (2, 0, 1))
        if arr.shape != (NUM_BANDS, CHIP_SIZE, CHIP_SIZE):
            raise DatasetUnavailable(
                f"{repo} record {i} has shape {arr.shape}; expected 13x64x64 "
                "(is this the RGB-only variant?)"
            )
        X[i] = arr.astype(np.uint16)
        raw_label = rec[label_key]
        name = names[raw_label] if names is not None else str(raw_label)
        labels[i] = cfg.CLASS_NAMES.index(name)
    X.flush()
    del X
    return _finalize_cache(cache_dir, labels, f"huggingface:{repo}")


def load_eurosat_ms(
    data_dir: Path | str | None = None,
    cache_dir: Path | str | None = None,
    allow_download: bool = True,
) -> DatasetBundle:
    """Return the 13-band EuroSAT dataset, trying each documented source in turn."""
    data_dir = Path(data_dir or cfg.DATA_DIR)
    cache_dir = Path(cache_dir or cfg.DATA_DIR / "cache")
    attempts: list[str] = []

    cached = _load_cache(cache_dir)
    if cached is not None:
        return cached

    local = _find_tif_root(data_dir)
    if local is not None:
        return _load_from_tif_dir(local, f"local GeoTIFF tree at {local}", cache_dir)
    attempts.append(f"local GeoTIFF tree under {data_dir}: not found")

    if not allow_download:
        raise DatasetUnavailable("allow_download=False and no local copy found:\n  " + "\n  ".join(attempts))

    for label, fn in (
        ("torchgeo.datasets.EuroSAT", lambda: _try_torchgeo(data_dir, cache_dir)),
        ("hf:blanchon/EuroSAT_MSI", lambda: _try_huggingface(cache_dir, "blanchon/EuroSAT_MSI")),
        ("hf:torchgeo/eurosat", lambda: _try_huggingface(cache_dir, "torchgeo/eurosat")),
    ):
        try:
            bundle = fn()
            bundle.notes = attempts + bundle.notes
            return bundle
        except Exception as exc:  # noqa: BLE001 - we genuinely want to try the next source
            attempts.append(f"{label}: {type(exc).__name__}: {exc}")

    raise DatasetUnavailable(
        "could not obtain 13-band EuroSAT. Attempted:\n  "
        + "\n  ".join(attempts)
        + "\nManual fallback: download EuroSATallBands.zip from "
        "https://github.com/phelber/EuroSAT, extract it under data/, and re-run."
    )


# --------------------------------------------------------------------------
# Splits
# --------------------------------------------------------------------------
def make_stratified_splits(
    y: np.ndarray,
    train_frac: float = 0.70,
    val_frac: float = 0.15,
    test_frac: float = 0.15,
    seed: int = cfg.SEED,
) -> dict[str, np.ndarray]:
    """Deterministic stratified 70/15/15 split.

    Implemented with an explicit per-class permutation rather than
    sklearn's train_test_split so that the split is reproducible across
    scikit-learn versions — a split that silently changes between library
    versions would invalidate every cached feature and checkpoint in the repo.

    NOTE ON VALIDITY (repeated in NB01 and the README): this is a *random*
    split. EuroSAT chips are cut from a small number of Sentinel-2 scenes, so
    chips near each other are spatially autocorrelated and a random split leaks
    information from train into test, mildly inflating every number reported
    here. The methodologically correct alternative is a spatially blocked split
    by source scene or region. We keep the random split for comparability with
    published EuroSAT numbers and state the limitation instead of hiding it.
    """
    total = train_frac + val_frac + test_frac
    if abs(total - 1.0) > 1e-9:
        raise ValueError(f"split fractions must sum to 1, got {total}")
    y = np.asarray(y)
    rng = np.random.default_rng(seed)
    out: dict[str, list[np.ndarray]] = {"train": [], "val": [], "test": []}
    for c in np.unique(y):
        idx = np.flatnonzero(y == c)
        idx = idx[rng.permutation(idx.size)]
        n_train = int(round(train_frac * idx.size))
        n_val = int(round(val_frac * idx.size))
        out["train"].append(idx[:n_train])
        out["val"].append(idx[n_train : n_train + n_val])
        out["test"].append(idx[n_train + n_val :])
    splits = {k: np.sort(np.concatenate(v)) for k, v in out.items()}

    # Leakage guard: assert here, not only in the tests, because this is the one
    # error in the whole pipeline that silently makes the results look good.
    all_idx = np.concatenate(list(splits.values()))
    assert all_idx.size == y.size, f"split covers {all_idx.size} of {y.size} samples"
    assert np.unique(all_idx).size == y.size, "splits overlap"
    return splits


def save_splits(splits: dict[str, np.ndarray], path: Path | str | None = None) -> Path:
    path = Path(path or cfg.SPLITS_NPZ)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **splits)
    return path


def load_splits(path: Path | str | None = None) -> dict[str, np.ndarray]:
    path = Path(path or cfg.SPLITS_NPZ)
    if not path.exists():
        raise FileNotFoundError(f"{path} not found — run notebook 01 first, it writes the splits.")
    with np.load(path) as f:
        return {k: f[k] for k in f.files}


def few_shot_indices(
    y: np.ndarray, train_idx: np.ndarray, k: int, seed: int
) -> np.ndarray:
    """Draw k labelled examples per class from the training split only.

    Never draws from val/test, so a k-shot probe remains evaluable on the same
    held-out test set as every other arm.
    """
    rng = np.random.default_rng(seed)
    picked = []
    for c in np.unique(y[train_idx]):
        pool = train_idx[y[train_idx] == c]
        if pool.size < k:
            raise ValueError(f"class {c} has only {pool.size} training samples, k={k} requested")
        picked.append(rng.choice(pool, size=k, replace=False))
    return np.sort(np.concatenate(picked))


# --------------------------------------------------------------------------
# Normalisation
# --------------------------------------------------------------------------
def compute_band_stats(
    X: np.ndarray, train_idx: np.ndarray, chunk: int = 512
) -> dict[str, list[float]]:
    """Per-band mean/std over the TRAINING SPLIT ONLY.

    Using the full dataset would leak test-set statistics into training-time
    preprocessing. The effect is small on EuroSAT, but it is free to do
    correctly and an interviewer will look for it.

    Streamed in chunks so this never materialises the full 2.9 GB array.
    """
    n_bands = X.shape[1]
    count = 0
    total = np.zeros(n_bands, dtype=np.float64)
    total_sq = np.zeros(n_bands, dtype=np.float64)
    for start in range(0, train_idx.size, chunk):
        batch = np.asarray(X[train_idx[start : start + chunk]], dtype=np.float64)
        flat = batch.transpose(1, 0, 2, 3).reshape(n_bands, -1)
        total += flat.sum(axis=1)
        total_sq += (flat**2).sum(axis=1)
        count += flat.shape[1]
    mean = total / count
    var = np.maximum(total_sq / count - mean**2, 0.0)
    return {
        "band_ids": list(BAND_IDS),
        "mean": mean.tolist(),
        "std": np.sqrt(var).tolist(),
        "n_pixels": int(count),
        "computed_on": "train split only",
    }


def save_band_stats(stats: dict, path: Path | str | None = None) -> Path:
    path = Path(path or cfg.BAND_STATS_JSON)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(stats, indent=2), encoding="utf-8")
    return path


def load_band_stats(path: Path | str | None = None) -> dict:
    path = Path(path or cfg.BAND_STATS_JSON)
    if not path.exists():
        raise FileNotFoundError(f"{path} not found — notebook 01 writes it.")
    return json.loads(path.read_text(encoding="utf-8"))


def normalize(x: np.ndarray, stats: dict) -> np.ndarray:
    """Standardise a (..., 13, H, W) array with the saved training statistics."""
    mean = np.asarray(stats["mean"], dtype=np.float32).reshape(-1, 1, 1)
    std = np.asarray(stats["std"], dtype=np.float32).reshape(-1, 1, 1)
    return (np.asarray(x, dtype=np.float32) - mean) / np.maximum(std, 1e-6)


# --------------------------------------------------------------------------
# torch plumbing
# --------------------------------------------------------------------------
class EuroSATChips:
    """torch Dataset over a memmap + an index array.

    Deliberately does not subclass torch.utils.data.Dataset at import time so
    that this module stays importable without torch (the unit tests run without
    a GPU stack). It satisfies the same protocol.
    """

    def __init__(
        self,
        X: np.ndarray,
        y: np.ndarray,
        indices: np.ndarray,
        stats: dict,
        transform=None,
        bands: list[int] | None = None,
    ):
        self.X, self.y, self.indices, self.stats = X, y, np.asarray(indices), stats
        self.transform = transform
        self.bands = bands  # e.g. RGB-only ablation

    def __len__(self) -> int:
        return int(self.indices.size)

    def __getitem__(self, i: int):
        import torch

        j = int(self.indices[i])
        chip = normalize(np.asarray(self.X[j]), self.stats)
        if self.bands is not None:
            chip = chip[self.bands]
        chip = torch.from_numpy(np.ascontiguousarray(chip))
        if self.transform is not None:
            chip = self.transform(chip)
        return chip, int(self.y[j])


def make_loader(dataset, batch_size: int, shuffle: bool, num_workers: int = 2, seed: int = cfg.SEED):
    """DataLoader with a seeded generator so shuffling is reproducible."""
    import torch
    from torch.utils.data import DataLoader

    generator = torch.Generator()
    generator.manual_seed(seed)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=False,
        generator=generator if shuffle else None,
        persistent_workers=num_workers > 0,
    )
