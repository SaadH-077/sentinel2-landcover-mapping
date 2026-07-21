"""All plotting, in one place and one style.

Figures are the deliverable of a study like this, so they are built by named
functions here rather than by ad-hoc matplotlib in six notebooks. That keeps a
class the same colour in every figure and means a style change happens once.

Every function returns the Figure; `save()` writes it to figures/ at 150 dpi
with a descriptive filename.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

from . import config as cfg
from .bands import BAND_IDS, BANDS

CLASS_COLOR_LIST = [cfg.CLASS_COLORS[c] for c in cfg.CLASS_NAMES]


def set_style() -> None:
    """One consistent look: light grid, no top/right spines, readable sizes."""
    mpl.rcParams.update(
        {
            "figure.dpi": 110,
            "savefig.dpi": 150,
            "savefig.bbox": "tight",
            "font.size": 10,
            "axes.titlesize": 11,
            "axes.titleweight": "semibold",
            "axes.labelsize": 10,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.alpha": 0.25,
            "grid.linewidth": 0.6,
            "legend.frameon": False,
            "image.interpolation": "nearest",
        }
    )


def save(fig, name: str, directory: Path | str | None = None) -> Path:
    directory = Path(directory or cfg.FIGURE_DIR)
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / (name if name.endswith(".png") else f"{name}.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    print(f"saved {path}")
    return path


# --------------------------------------------------------------------------
# NB01
# --------------------------------------------------------------------------
def plot_class_distribution(y, class_names=cfg.CLASS_NAMES):
    counts = np.bincount(np.asarray(y), minlength=len(class_names))
    fig, ax = plt.subplots(figsize=(8, 3.6))
    ax.bar(range(len(class_names)), counts, color=CLASS_COLOR_LIST)
    ax.set_xticks(range(len(class_names)))
    ax.set_xticklabels(class_names, rotation=35, ha="right")
    ax.set_ylabel("chips")
    ax.set_title(f"EuroSAT class distribution (n = {counts.sum():,})")
    for i, c in enumerate(counts):
        ax.text(i, c, f"{c:,}", ha="center", va="bottom", fontsize=8)
    ax.margins(y=0.12)
    return fig


def plot_chip_grid(chips, titles, renderer, n_cols: int = 5, title: str = ""):
    """Grid of chips rendered by a composite function from bands.py."""
    n = len(chips)
    n_rows = int(np.ceil(n / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(2.1 * n_cols, 2.25 * n_rows))
    for ax, chip, t in zip(np.atleast_1d(axes).ravel(), chips, titles):
        ax.imshow(renderer(np.asarray(chip)))
        ax.set_title(t, fontsize=9)
    for ax in np.atleast_1d(axes).ravel():
        ax.axis("off")
    if title:
        fig.suptitle(title, y=1.0)
    fig.tight_layout()
    return fig


def plot_stretch_comparison(chip, class_name: str = ""):
    """Raw vs percentile-stretched true colour, side by side.

    The single most persuasive figure for why the stretch exists: the raw panel
    is nearly black not because the data are bad but because land reflectance
    occupies a small, low slice of the sensor's range.
    """
    from .bands import true_color

    fig, axes = plt.subplots(1, 2, figsize=(6.4, 3.4))
    axes[0].imshow(np.clip(true_color(chip, stretch=False), 0, 1))
    axes[0].set_title("raw B04/B03/B02\n(no stretch)")
    axes[1].imshow(true_color(chip, stretch=True))
    axes[1].set_title("2nd–98th percentile\nlinear stretch")
    for ax in axes:
        ax.axis("off")
    if class_name:
        fig.suptitle(class_name)
    fig.tight_layout()
    return fig


def plot_composites(chip, class_name: str = ""):
    from .bands import COMPOSITES

    fig, axes = plt.subplots(1, 3, figsize=(9.6, 3.6))
    for ax, (_, (fn, label)) in zip(axes, COMPOSITES.items()):
        ax.imshow(fn(chip))
        ax.set_title(label, fontsize=9)
        ax.axis("off")
    if class_name:
        fig.suptitle(class_name)
    fig.tight_layout()
    return fig


def plot_spectral_signatures(mean_by_class, std_by_class=None, class_names=cfg.CLASS_NAMES):
    """The key figure of NB01: mean reflectance per band, one curve per class.

    x is ordered by wavelength (which is EuroSAT's band order), so the classic
    vegetation red edge — the jump from B04 to B08 — appears as a visible step
    rather than as an artefact of band numbering.
    """
    fig, ax = plt.subplots(figsize=(9.5, 5.0))
    x = np.arange(len(BAND_IDS))
    for i, name in enumerate(class_names):
        ax.plot(x, mean_by_class[i], label=name, color=cfg.CLASS_COLORS[name], lw=1.8)
        if std_by_class is not None:
            ax.fill_between(
                x,
                mean_by_class[i] - std_by_class[i],
                mean_by_class[i] + std_by_class[i],
                color=cfg.CLASS_COLORS[name],
                alpha=0.10,
                lw=0,
            )
    ax.set_xticks(x)
    ax.set_xticklabels([f"{b.band_id}\n{b.wavelength:.0f}nm" for b in BANDS], fontsize=7.5)
    ax.set_xlabel("band (ordered by central wavelength)")
    ax.set_ylabel("mean reflectance (DN)")
    ax.set_title("Mean spectral signature per class (± 1 std)")
    ax.legend(ncol=2, fontsize=8.5, loc="upper left")
    return fig


def plot_index_boxplots(index_values, y, index_names=("NDVI", "NDWI", "NDBI"), class_names=cfg.CLASS_NAMES):
    """Per-class distribution of each spectral index — training-free features."""
    fig, axes = plt.subplots(len(index_names), 1, figsize=(9.5, 3.1 * len(index_names)), sharex=True)
    for k, (ax, name) in enumerate(zip(np.atleast_1d(axes), index_names)):
        data = [index_values[np.asarray(y) == c, k] for c in range(len(class_names))]
        bp = ax.boxplot(data, patch_artist=True, showfliers=False, widths=0.6)
        for patch, cname in zip(bp["boxes"], class_names):
            patch.set_facecolor(cfg.CLASS_COLORS[cname])
            patch.set_alpha(0.75)
        for median in bp["medians"]:
            median.set_color("black")
        ax.axhline(0, color="k", lw=0.7, alpha=0.5)
        ax.set_ylabel(name)
    axes[-1].set_xticks(range(1, len(class_names) + 1))
    axes[-1].set_xticklabels(class_names, rotation=35, ha="right")
    axes[0].set_title("Spectral indices by class (physics-derived, no training)")
    fig.tight_layout()
    return fig


# --------------------------------------------------------------------------
# NB02 / NB06
# --------------------------------------------------------------------------
def plot_confusion_matrix(cm, class_names=cfg.CLASS_NAMES, normalize: bool = True, title: str = ""):
    from .evaluate import row_normalize

    m = row_normalize(cm) if normalize else np.asarray(cm, dtype=float)
    fig, ax = plt.subplots(figsize=(7.4, 6.4))
    im = ax.imshow(m, cmap="Blues", vmin=0, vmax=1 if normalize else None)
    ax.set_xticks(range(len(class_names)))
    ax.set_yticks(range(len(class_names)))
    ax.set_xticklabels(class_names, rotation=45, ha="right", fontsize=8)
    ax.set_yticklabels(class_names, fontsize=8)
    ax.set_xlabel("predicted")
    ax.set_ylabel("true")
    ax.set_title(title or ("Confusion matrix (row-normalised)" if normalize else "Confusion matrix"))
    ax.grid(False)
    threshold = m.max() / 2
    for i in range(m.shape[0]):
        for j in range(m.shape[1]):
            if m[i, j] > 0.005:
                ax.text(
                    j, i, f"{m[i, j]:.2f}" if normalize else f"{int(m[i, j])}",
                    ha="center", va="center", fontsize=7,
                    color="white" if m[i, j] > threshold else "black",
                )
    fig.colorbar(im, ax=ax, fraction=0.046, label="rate" if normalize else "count")
    fig.tight_layout()
    return fig


def plot_training_curves(histories: dict, metric: str = "val_macro_f1"):
    fig, ax = plt.subplots(figsize=(7.6, 4.2))
    for name, h in histories.items():
        values = h[metric] if isinstance(h, dict) else getattr(h, metric)
        ax.plot(range(1, len(values) + 1), values, label=name, lw=1.8)
    ax.set_xlabel("epoch")
    ax.set_ylabel(metric.replace("_", " "))
    ax.set_title("Validation macro-F1 during training")
    ax.legend(fontsize=9)
    return fig


def plot_arm_summary(rows, metric: str = "test_macro_f1"):
    """Horizontal bar chart of arms with std error bars."""
    names = [r["arm"] for r in rows]
    means = [r[metric]["mean"] if isinstance(r[metric], dict) else r[metric] for r in rows]
    stds = [r[metric].get("std", 0.0) if isinstance(r[metric], dict) else 0.0 for r in rows]
    order = np.argsort(means)
    fig, ax = plt.subplots(figsize=(8, 0.5 * len(rows) + 1.8))
    ax.barh(
        [names[i] for i in order],
        [means[i] for i in order],
        xerr=[stds[i] for i in order],
        color="#4f7ec9", alpha=0.9, capsize=3,
    )
    ax.set_xlabel(metric.replace("_", " "))
    ax.set_title("Supervised baseline ladder")
    for i, idx in enumerate(order):
        ax.text(means[idx] + 0.005, i, f"{means[idx]:.3f}", va="center", fontsize=8)
    ax.set_xlim(0, 1.05)
    return fig


def plot_reliability(curves: dict, title: str = "Reliability diagram"):
    """Confidence vs empirical accuracy, with the perfect-calibration diagonal.

    Below the diagonal = overconfident, which is the usual failure mode of a
    modern network trained to convergence with cross-entropy.
    """
    fig, ax = plt.subplots(figsize=(5.2, 5.0))
    ax.plot([0, 1], [0, 1], "k--", lw=1, label="perfect calibration")
    for name, c in curves.items():
        centres = [(lo + hi) / 2 for lo, hi in zip(c["bin_lower"], c["bin_upper"])]
        mask = np.asarray(c["count"]) > 0
        ax.plot(
            np.asarray(centres)[mask], np.asarray(c["accuracy"])[mask],
            "o-", lw=1.8, ms=4, label=name,
        )
    ax.set_xlabel("confidence")
    ax.set_ylabel("empirical accuracy")
    ax.set_title(title)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.legend(fontsize=9, loc="upper left")
    return fig


def plot_failure_gallery(chips, true_labels, pred_labels, confidences, class_names=cfg.CLASS_NAMES, n_cols: int = 5):
    from .bands import true_color

    n = len(chips)
    n_rows = int(np.ceil(n / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(2.4 * n_cols, 2.7 * n_rows))
    for ax, chip, t, p, c in zip(np.atleast_1d(axes).ravel(), chips, true_labels, pred_labels, confidences):
        ax.imshow(true_color(np.asarray(chip)))
        ax.set_title(f"true {class_names[t]}\npred {class_names[p]} ({c:.2f})", fontsize=8, color="#a3336b")
    for ax in np.atleast_1d(axes).ravel():
        ax.axis("off")
    fig.suptitle("Most confidently wrong test predictions", y=1.0)
    fig.tight_layout()
    return fig


def plot_band_ablation_heatmap(matrix, group_names, class_names=cfg.CLASS_NAMES):
    """Band group x class, coloured by macro-F1 drop when the group is removed."""
    fig, ax = plt.subplots(figsize=(9.0, 0.55 * len(group_names) + 2.2))
    im = ax.imshow(matrix, cmap="Reds", aspect="auto")
    ax.set_xticks(range(len(class_names)))
    ax.set_xticklabels(class_names, rotation=40, ha="right", fontsize=8)
    ax.set_yticks(range(len(group_names)))
    ax.set_yticklabels(group_names)
    ax.set_title("Per-class F1 drop when a band group is masked out")
    ax.grid(False)
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            ax.text(j, i, f"{matrix[i, j]:.2f}", ha="center", va="center", fontsize=7)
    fig.colorbar(im, ax=ax, fraction=0.03, label="F1 drop")
    fig.tight_layout()
    return fig


# --------------------------------------------------------------------------
# NB03 / NB04
# --------------------------------------------------------------------------
def plot_embedding(coords_by_model: dict, y, class_names=cfg.CLASS_NAMES, title: str = ""):
    """Side-by-side 2-D projections of frozen embeddings, coloured by true class."""
    n = len(coords_by_model)
    fig, axes = plt.subplots(1, n, figsize=(6.0 * n, 5.6))
    for ax, (name, coords) in zip(np.atleast_1d(axes), coords_by_model.items()):
        for c, cname in enumerate(class_names):
            m = np.asarray(y) == c
            ax.scatter(coords[m, 0], coords[m, 1], s=3, alpha=0.55, color=cfg.CLASS_COLORS[cname], label=cname)
        ax.set_title(name)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.grid(False)
    handles, labels = np.atleast_1d(axes)[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=5, markerscale=4, fontsize=9)
    if title:
        fig.suptitle(title)
    fig.tight_layout(rect=(0, 0.09, 1, 1))
    return fig


def plot_label_efficiency(
    curves: dict,
    zero_shot: dict | None = None,
    supervised_ceiling: float | None = None,
    ceiling_label: str = "full supervision (13-band ResNet-18)",
    title: str = "How many labels does it take to beat zero-shot?",
):
    """THE HEADLINE FIGURE.

    curves: {name: {"k": [...], "mean": [...], "std": [...], "reference": bool}}
    zero_shot: {name: macro_f1}
    supervised_ceiling: macro-F1 of the fully supervised model.

    An encoder marked `reference=True` (the NB02 supervised backbone, which saw
    the training labels) is drawn dashed and labelled as a reference, because it
    is an upper bound rather than a fair few-shot competitor. Quietly plotting
    it as one more line would be the most misleading thing this project could do.
    """
    fig, ax = plt.subplots(figsize=(8.6, 5.6))
    palette = ["#1f77b4", "#d95f02", "#7570b3", "#3f8f5b", "#a3336b"]

    for i, (name, c) in enumerate(curves.items()):
        k = np.asarray(c["k"], dtype=float)
        mean = np.asarray(c["mean"], dtype=float)
        std = np.asarray(c.get("std", np.zeros_like(mean)), dtype=float)
        colour = palette[i % len(palette)]
        is_ref = bool(c.get("reference", False))
        ax.plot(
            k, mean, "o-" if not is_ref else "s--", color=colour, lw=2.0, ms=5,
            label=name + (" (reference: saw all labels)" if is_ref else ""),
        )
        ax.fill_between(k, mean - std, mean + std, color=colour, alpha=0.15, lw=0)

    if supervised_ceiling is not None:
        ax.axhline(supervised_ceiling, color="black", ls="--", lw=1.4)
        ax.text(
            ax.get_xlim()[1], supervised_ceiling, f" {ceiling_label}: {supervised_ceiling:.3f}",
            va="bottom", ha="right", fontsize=8.5,
        )
    for name, value in (zero_shot or {}).items():
        ax.axhline(value, color="#666666", ls=":", lw=1.3)
        ax.text(1.0, value, f" {name} zero-shot: {value:.3f}", va="bottom", ha="left", fontsize=8.5, color="#444444")

    ax.set_xscale("log")
    ax.set_xticks(sorted({int(v) for c in curves.values() for v in c["k"]}))
    ax.get_xaxis().set_major_formatter(mpl.ticker.ScalarFormatter())
    ax.set_xlabel("labelled examples per class (log scale)")
    ax.set_ylabel("test macro-F1")
    ax.set_title(title)
    ax.legend(fontsize=9, loc="lower right")
    return fig


def annotate_crossing(ax, x: float, y: float, text: str) -> None:
    """Mark a crossover point directly on the headline figure."""
    ax.plot([x], [y], marker="*", ms=14, color="black", zorder=5)
    ax.annotate(
        text, xy=(x, y), xytext=(12, -22), textcoords="offset points", fontsize=8.5,
        arrowprops=dict(arrowstyle="->", lw=0.9, color="black"),
    )


# --------------------------------------------------------------------------
# NB05
# --------------------------------------------------------------------------
def class_colormap(class_names=cfg.CLASS_NAMES):
    """Discrete colormap + norm for a categorical land-cover map."""
    from matplotlib.colors import BoundaryNorm, ListedColormap

    cmap = ListedColormap([cfg.CLASS_COLORS[c] for c in class_names])
    cmap.set_bad("#ffffff")
    norm = BoundaryNorm(np.arange(-0.5, len(class_names) + 0.5), cmap.N)
    return cmap, norm


def plot_map_panels(panels: dict, class_names=cfg.CLASS_NAMES, title: str = ""):
    """Row of panels: RGB rendering, class maps, masks — with one shared legend.

    panels: {label: ("rgb", array) | ("classes", array) | ("mask", array)}
    """
    from matplotlib.patches import Patch

    cmap, norm = class_colormap(class_names)
    n = len(panels)
    fig, axes = plt.subplots(1, n, figsize=(5.0 * n, 5.4))
    has_classes = False
    for ax, (label, (kind, array)) in zip(np.atleast_1d(axes), panels.items()):
        array = np.asarray(array)
        if kind == "rgb":
            ax.imshow(array)
        elif kind == "classes":
            has_classes = True
            ax.imshow(np.ma.masked_where(array > len(class_names) - 1, array), cmap=cmap, norm=norm)
        else:
            ax.imshow(array, cmap="gray")
        ax.set_title(label)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.grid(False)
    if has_classes:
        handles = [Patch(facecolor=cfg.CLASS_COLORS[c], label=c) for c in class_names]
        fig.legend(handles=handles, loc="lower center", ncol=5, fontsize=9)
    if title:
        fig.suptitle(title)
    fig.tight_layout(rect=(0, 0.10 if has_classes else 0, 1, 1))
    return fig


def plot_domain_shift_histograms(train_samples, scene_samples, band_ids=BAND_IDS, n_cols: int = 5):
    """Per-band histograms: EuroSAT L1C training chips vs the fetched L2A scene.

    This is the quantitative form of the domain-shift argument in NB05 — the
    gap between the two histograms is the covariate shift the model is being
    asked to absorb.
    """
    n = len(band_ids)
    n_rows = int(np.ceil(n / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(3.0 * n_cols, 2.4 * n_rows))
    for ax, i in zip(np.atleast_1d(axes).ravel(), range(n)):
        ax.hist(np.asarray(train_samples[i]).ravel(), bins=60, density=True, alpha=0.55, label="EuroSAT L1C", color="#4f7ec9")
        ax.hist(np.asarray(scene_samples[i]).ravel(), bins=60, density=True, alpha=0.55, label="scene L2A", color="#d95f02")
        ax.set_title(band_ids[i], fontsize=9)
        ax.set_yticks([])
    for ax in np.atleast_1d(axes).ravel()[n:]:
        ax.axis("off")
    np.atleast_1d(axes).ravel()[0].legend(fontsize=8)
    fig.suptitle("Per-band value distributions: training data vs real scene")
    fig.tight_layout()
    return fig
