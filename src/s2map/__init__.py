"""s2map — Sentinel-2: from chips to map.

All logic shared by more than one notebook lives here. Notebooks import from
this package; they do not redefine it. See README.md for the study design.
"""

__version__ = "0.1.0"

__all__ = [
    "bands",
    "clip_utils",
    "config",
    "data",
    "evaluate",
    "inference",
    "models",
    "stac",
    "train",
    "transforms",
    "viz",
]
