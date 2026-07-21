"""Paths, class metadata and seeding.

Single source of truth for anything a notebook would otherwise hard-code.
Every path is derived from the repository root so the same code runs from a
local checkout and from `/content/s2-chips-to-map` on Colab.
"""

from __future__ import annotations

import os
import random
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

# --------------------------------------------------------------------------
# Paths
# --------------------------------------------------------------------------
# config.py lives at <root>/src/s2map/config.py -> three parents up is <root>.
ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = Path(os.environ.get("S2MAP_DATA_DIR", ROOT / "data"))
OUTPUT_DIR = Path(os.environ.get("S2MAP_OUTPUT_DIR", ROOT / "outputs"))
FIGURE_DIR = Path(os.environ.get("S2MAP_FIGURE_DIR", ROOT / "figures"))
CONFIG_DIR = ROOT / "configs"

RESULTS_JSON = OUTPUT_DIR / "results.json"
BAND_STATS_JSON = OUTPUT_DIR / "band_stats.json"
SPLITS_NPZ = OUTPUT_DIR / "splits.npz"

# --------------------------------------------------------------------------
# Classes
# --------------------------------------------------------------------------
# EuroSAT's directory names, in the alphabetical order that every loader
# (torchgeo, torchvision.ImageFolder, HuggingFace) assigns label indices in.
# Anything that maps an integer back to a name MUST use this order.
CLASS_NAMES: tuple[str, ...] = (
    "AnnualCrop",
    "Forest",
    "HerbaceousVegetation",
    "Highway",
    "Industrial",
    "Pasture",
    "PermanentCrop",
    "Residential",
    "River",
    "SeaLake",
)
NUM_CLASSES = len(CLASS_NAMES)

# The CamelCase directory names are not English. For a vision-language model the
# class *name* is part of the model input, so rewriting these is a modelling
# decision, not cosmetics. NB03 quantifies exactly how much it is worth.
NATURAL_CLASS_NAMES: dict[str, str] = {
    "AnnualCrop": "annual cropland with seasonal crops",
    "Forest": "dense forest",
    "HerbaceousVegetation": "herbaceous vegetation and shrubland",
    "Highway": "a highway or major road",
    "Industrial": "an industrial area with factories and warehouses",
    "Pasture": "pasture and grazing grassland",
    "PermanentCrop": "permanent cropland such as orchards or vineyards",
    "Residential": "a residential neighbourhood with houses",
    "River": "a river",
    "SeaLake": "a sea or a lake",
}

# Stable, colour-blind-considerate categorical palette. Fixed here so every
# figure in every notebook gives a class the same colour.
CLASS_COLORS: dict[str, str] = {
    "AnnualCrop": "#e8c547",
    "Forest": "#1b6b3a",
    "HerbaceousVegetation": "#7fbf6a",
    "Highway": "#7a7a7a",
    "Industrial": "#a3336b",
    "Pasture": "#bfe08c",
    "PermanentCrop": "#c98a2b",
    "Residential": "#d9534f",
    "River": "#4f9bd9",
    "SeaLake": "#1f3f8f",
}

# --------------------------------------------------------------------------
# Experiment configuration
# --------------------------------------------------------------------------
SEED = 42
SEEDS: tuple[int, ...] = (0, 1, 2)  # multi-seed reporting: mean +/- std over these


@dataclass
class TrainConfig:
    """Hyperparameters shared by every trained arm (see src/s2map/train.py).

    Defaults are the ones actually used; configs/default.yaml can override them.
    """

    epochs: int = 30
    batch_size: int = 128
    lr: float = 3e-4
    weight_decay: float = 1e-4
    warmup_epochs: int = 2
    patience: int = 10           # early stopping on validation macro-F1
    label_smoothing: float = 0.0
    amp: bool = True             # mixed precision; ~2x on a T4
    num_workers: int = 2
    grad_clip: float | None = 1.0


@dataclass
class SplitConfig:
    train_frac: float = 0.70
    val_frac: float = 0.15
    test_frac: float = 0.15
    seed: int = SEED


@dataclass
class Config:
    train: TrainConfig = field(default_factory=TrainConfig)
    split: SplitConfig = field(default_factory=SplitConfig)
    seed: int = SEED
    seeds: tuple[int, ...] = SEEDS
    # Few-shot sweep (NB04). Log-spaced so the headline plot's x axis is even.
    few_shot_k: tuple[int, ...] = (1, 2, 5, 10, 20, 50, 100)
    few_shot_draws: int = 5      # random label draws per (encoder, k)
    # NB05 area of interest; see configs/default.yaml for the choice of box.
    aoi: dict[str, Any] = field(
        default_factory=lambda: {
            "name": "Bay of Cadiz, Spain",
            "bbox": [-6.30, 36.48, -6.12, 36.62],
            "date_range": "2023-06-01/2023-09-30",
            "max_cloud_cover": 10,
        }
    )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def load_config(path: str | Path | None = None) -> Config:
    """Load configs/default.yaml over the dataclass defaults.

    Kept deliberately shallow: only the two nested sections are merged, because
    a general deep-merge invites silent typos in the YAML being ignored.
    """
    path = Path(path) if path is not None else CONFIG_DIR / "default.yaml"
    cfg = Config()
    if not path.exists():
        return cfg

    import yaml  # imported lazily so `import s2map.config` has no yaml dependency

    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    for section, dc in (("train", cfg.train), ("split", cfg.split)):
        for key, value in (raw.get(section) or {}).items():
            if not hasattr(dc, key):
                raise KeyError(f"unknown key {section}.{key} in {path}")
            setattr(dc, key, value)
    for key in ("seed", "seeds", "few_shot_k", "few_shot_draws", "aoi"):
        if key in raw:
            value = raw[key]
            setattr(cfg, key, tuple(value) if isinstance(value, list) else value)
    return cfg


# --------------------------------------------------------------------------
# Determinism
# --------------------------------------------------------------------------
def set_seed(seed: int = SEED, deterministic: bool = True) -> int:
    """Seed python, numpy and torch (incl. CUDA). Returns the seed for printing.

    `deterministic=True` selects deterministic cuDNN kernels. That costs a few
    percent of throughput and is worth it here: a headline claim of
    "mean +/- std over 3 seeds" is meaningless if a rerun of the same seed
    gives a different number.
    """
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import torch

        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        if deterministic:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
    except ImportError:  # torch-free contexts (e.g. running the tests alone)
        pass
    return seed


def get_device() -> str:
    try:
        import torch

        return "cuda" if torch.cuda.is_available() else "cpu"
    except ImportError:
        return "cpu"


def ensure_dirs() -> None:
    for d in (DATA_DIR, OUTPUT_DIR, FIGURE_DIR):
        d.mkdir(parents=True, exist_ok=True)


def print_environment() -> None:
    """Print versions, GPU and seed. Called at the top of every notebook."""
    import platform
    import sys

    print(f"python           {sys.version.split()[0]} on {platform.system()}")
    print(f"numpy            {np.__version__}")
    for mod in ("torch", "torchvision", "timm", "sklearn", "rasterio", "open_clip"):
        try:
            m = __import__(mod)
            print(f"{mod:<16} {getattr(m, '__version__', 'unknown')}")
        except ImportError:
            print(f"{mod:<16} not installed")
        except Exception as exc:  # noqa: BLE001
            # Deliberately broad. A compiled extension built against a different
            # numpy ABI raises ValueError, not ImportError, and letting that
            # escape would kill the setup cell of every notebook. Report it and
            # keep going: `check_environment()` below explains the fix.
            print(f"{mod:<16} FAILED TO IMPORT — {type(exc).__name__}: {exc}")
    try:
        import torch

        if torch.cuda.is_available():
            props = torch.cuda.get_device_properties(0)
            print(f"gpu              {props.name} ({props.total_memory / 1e9:.1f} GB)")
        else:
            print("gpu              none (running on CPU)")
    except ImportError:
        pass
    print(f"repo root        {ROOT}")


def check_environment(verbose: bool = True) -> bool:
    """Detect the "installed numpy != imported numpy" trap. Returns True if sane.

    On Colab, `pip install` can replace numpy *on disk* while the running kernel
    still holds the previous version *in memory*. Compiled extensions then load
    against a numpy whose C struct sizes differ from the one they were built
    for, and raise

        ValueError: numpy.dtype size changed, may indicate binary
        incompatibility. Expected 96 from C header, got 88 from PyObject

    The failure is confusing because nothing in the traceback mentions pip. The
    only fix is to restart the runtime so the interpreter picks up one
    consistent numpy — no amount of re-running the cell will help, which is
    exactly why this check prints an instruction instead of a warning.
    """
    import importlib.metadata as md

    problems: list[str] = []
    try:
        installed = md.version("numpy")
        if installed != np.__version__:
            problems.append(
                f"numpy {installed} is installed on disk but {np.__version__} is loaded in "
                "memory — a pip install replaced it after the kernel started"
            )
    except md.PackageNotFoundError:
        pass

    try:
        import torch  # noqa: F401
    except Exception as exc:  # noqa: BLE001
        problems.append(f"torch failed to import: {type(exc).__name__}: {exc}")

    if problems and verbose:
        print("\n" + "!" * 72)
        for p in problems:
            print("!! " + p)
        print("!!")
        print("!! FIX: Runtime > Restart session, then run this cell again.")
        print("!!      Do NOT re-run the install; restarting is the whole fix.")
        print("!" * 72 + "\n")
    elif verbose:
        print("environment check  OK (numpy on disk matches numpy in memory)")
    return not problems
