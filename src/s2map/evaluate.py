"""Metrics, confusion structure, calibration, and the results ledger.

The metrics are implemented in numpy rather than imported from sklearn. Two
reasons: they are eight lines each and unit-testable against hand computation,
and calibration (ECE, temperature scaling) is not in sklearn anyway, so having
one consistent metrics module beats splitting them across two.

WHY MACRO-F1 EVERYWHERE. Accuracy is dominated by whichever classes are common,
and it hides a model that has quietly given up on one class. Macro-F1 averages
per-class F1 with equal weight, so ignoring a single class costs a tenth of the
score. On EuroSAT the two are close because the dataset is near-balanced, and
that closeness is itself worth reporting — it says the errors are spread, not
concentrated.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from . import config as cfg


# --------------------------------------------------------------------------
# Classification metrics
# --------------------------------------------------------------------------
def confusion_matrix(y_true, y_pred, num_classes: int | None = None) -> np.ndarray:
    """Rows = true class, columns = predicted class."""
    y_true = np.asarray(y_true, dtype=np.int64).ravel()
    y_pred = np.asarray(y_pred, dtype=np.int64).ravel()
    assert y_true.shape == y_pred.shape, f"{y_true.shape} vs {y_pred.shape}"
    k = num_classes or int(max(y_true.max(), y_pred.max()) + 1)
    flat = np.bincount(y_true * k + y_pred, minlength=k * k)
    return flat.reshape(k, k)


def per_class_prf(y_true, y_pred, num_classes: int | None = None) -> dict[str, np.ndarray]:
    """Per-class precision, recall, F1 and support.

    A class that is never predicted gets precision 0 by convention (0/0 -> 0),
    which is the conservative choice: it must not be rewarded for abstaining.
    """
    cm = confusion_matrix(y_true, y_pred, num_classes)
    tp = np.diag(cm).astype(np.float64)
    predicted = cm.sum(axis=0).astype(np.float64)
    actual = cm.sum(axis=1).astype(np.float64)
    precision = np.divide(tp, predicted, out=np.zeros_like(tp), where=predicted > 0)
    recall = np.divide(tp, actual, out=np.zeros_like(tp), where=actual > 0)
    denom = precision + recall
    f1 = np.divide(2 * precision * recall, denom, out=np.zeros_like(tp), where=denom > 0)
    return {"precision": precision, "recall": recall, "f1": f1, "support": actual.astype(int)}


def macro_f1(y_true, y_pred, num_classes: int | None = None) -> float:
    return float(per_class_prf(y_true, y_pred, num_classes)["f1"].mean())


def accuracy(y_true, y_pred) -> float:
    y_true, y_pred = np.asarray(y_true).ravel(), np.asarray(y_pred).ravel()
    return float((y_true == y_pred).mean())


def classification_summary(y_true, y_pred, class_names=cfg.CLASS_NAMES) -> dict:
    prf = per_class_prf(y_true, y_pred, len(class_names))
    return {
        "accuracy": accuracy(y_true, y_pred),
        "macro_f1": float(prf["f1"].mean()),
        "per_class": {
            name: {
                "precision": float(prf["precision"][i]),
                "recall": float(prf["recall"][i]),
                "f1": float(prf["f1"][i]),
                "support": int(prf["support"][i]),
            }
            for i, name in enumerate(class_names)
        },
        "confusion_matrix": confusion_matrix(y_true, y_pred, len(class_names)).tolist(),
    }


def row_normalize(cm: np.ndarray) -> np.ndarray:
    """Confusion matrix as per-true-class rates. Rows sum to 1 (empty rows to 0)."""
    cm = np.asarray(cm, dtype=np.float64)
    totals = cm.sum(axis=1, keepdims=True)
    return np.divide(cm, totals, out=np.zeros_like(cm), where=totals > 0)


def top_confusions(cm: np.ndarray, class_names=cfg.CLASS_NAMES, top_k: int = 10) -> list[dict]:
    """Most frequent off-diagonal (true -> predicted) pairs, normalised by support.

    Ranked by rate rather than raw count so a rare class's systematic failure is
    not buried under a common class's occasional one.
    """
    rates = row_normalize(cm)
    entries = [
        {
            "true": class_names[i],
            "predicted": class_names[j],
            "count": int(cm[i, j]),
            "rate": float(rates[i, j]),
        }
        for i in range(len(class_names))
        for j in range(len(class_names))
        if i != j and cm[i, j] > 0
    ]
    return sorted(entries, key=lambda e: e["rate"], reverse=True)[:top_k]


# --------------------------------------------------------------------------
# Calibration
# --------------------------------------------------------------------------
def softmax(logits: np.ndarray, temperature: float = 1.0) -> np.ndarray:
    z = np.asarray(logits, dtype=np.float64) / temperature
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def expected_calibration_error(
    probs: np.ndarray, labels: np.ndarray, n_bins: int = 15
) -> tuple[float, dict]:
    """ECE with equal-width confidence bins, plus the reliability-diagram data.

    ECE = sum_b (n_b / N) * |accuracy(b) - confidence(b)|, over bins of the
    predicted top-class probability.

    WHY IT MATTERS OPERATIONALLY: any deployed system routes low-confidence
    predictions to a human and auto-accepts high-confidence ones. That threshold
    is only meaningful if 0.9 confidence really means 90% correct. A model that
    is 95% accurate but reports 99% confidence will silently under-flag exactly
    the cases a reviewer needed to see.
    """
    probs = np.asarray(probs, dtype=np.float64)
    labels = np.asarray(labels).ravel()
    assert probs.ndim == 2 and probs.shape[0] == labels.shape[0]
    confidence = probs.max(axis=1)
    correct = (probs.argmax(axis=1) == labels).astype(np.float64)

    edges = np.linspace(0.0, 1.0, n_bins + 1)
    # Bin by upper edge so confidence exactly 1.0 lands in the last bin.
    bin_id = np.clip(np.digitize(confidence, edges[1:-1], right=True), 0, n_bins - 1)

    ece, curve = 0.0, {"bin_lower": [], "bin_upper": [], "count": [], "accuracy": [], "confidence": []}
    for b in range(n_bins):
        m = bin_id == b
        n_b = int(m.sum())
        acc_b = float(correct[m].mean()) if n_b else 0.0
        conf_b = float(confidence[m].mean()) if n_b else 0.0
        if n_b:
            ece += (n_b / labels.size) * abs(acc_b - conf_b)
        curve["bin_lower"].append(float(edges[b]))
        curve["bin_upper"].append(float(edges[b + 1]))
        curve["count"].append(n_b)
        curve["accuracy"].append(acc_b)
        curve["confidence"].append(conf_b)
    return float(ece), curve


def fit_temperature(
    logits: np.ndarray, labels: np.ndarray, max_iter: int = 200
) -> float:
    """Fit a single temperature T minimising validation NLL (Guo et al., 2017).

    FITTED ON VALIDATION, applied to test. One scalar, so it cannot change any
    argmax and therefore cannot change accuracy or macro-F1 — it only reshapes
    the confidences. That property is what makes it safe to apply after the fact.
    """
    import torch

    z = torch.as_tensor(np.asarray(logits), dtype=torch.float32)
    y = torch.as_tensor(np.asarray(labels), dtype=torch.long)
    log_t = torch.zeros(1, requires_grad=True)  # optimise log T to keep T > 0
    optimiser = torch.optim.LBFGS([log_t], lr=0.1, max_iter=max_iter)
    loss_fn = torch.nn.CrossEntropyLoss()

    def closure():
        optimiser.zero_grad()
        loss = loss_fn(z / torch.exp(log_t), y)
        loss.backward()
        return loss

    optimiser.step(closure)
    temperature = float(torch.exp(log_t).item())

    # Guard against the degenerate case: if the validation set is perfectly
    # classified, NLL is minimised by T -> 0 (infinitely sharp), which is not a
    # calibration result, it is a symptom of a validation set too easy or too
    # small to calibrate against. Clamp and let the caller see the bound.
    clamped = min(max(temperature, 0.05), 10.0)
    if clamped != temperature:
        print(f"warning: fitted temperature {temperature:.4g} is degenerate; clamped to {clamped}")
    return clamped


def calibration_report(logits, labels, temperature: float = 1.0, n_bins: int = 15) -> dict:
    probs = softmax(logits, temperature)
    ece, curve = expected_calibration_error(probs, labels, n_bins)
    return {
        "temperature": float(temperature),
        "ece": ece,
        "accuracy": accuracy(labels, probs.argmax(axis=1)),
        "mean_confidence": float(probs.max(axis=1).mean()),
        "reliability": curve,
    }


# --------------------------------------------------------------------------
# Results ledger
# --------------------------------------------------------------------------
def append_result(entry: dict, path: Path | str | None = None) -> Path:
    """Append one measurement to outputs/results.json.

    Every number in the README is read back out of this file. Nothing in the
    README is typed by hand, so the README cannot drift from what was actually
    run, and a rerun regenerates it.

    Entries are keyed by (notebook, arm) — re-running a notebook replaces its
    own previous entry rather than accumulating duplicates.
    """
    path = Path(path or cfg.RESULTS_JSON)
    path.parent.mkdir(parents=True, exist_ok=True)
    results = json.loads(path.read_text(encoding="utf-8")) if path.exists() else []
    if not isinstance(results, list):
        raise ValueError(f"{path} is not a JSON list; refusing to overwrite")

    entry = {**entry, "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds")}
    key = (entry.get("notebook"), entry.get("arm"))
    if key != (None, None):
        results = [r for r in results if (r.get("notebook"), r.get("arm")) != key]
    results.append(entry)
    path.write_text(json.dumps(results, indent=2, default=_json_default), encoding="utf-8")
    return path


def _json_default(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    raise TypeError(f"not JSON-serialisable: {type(o)}")


def load_results(path: Path | str | None = None) -> list[dict]:
    path = Path(path or cfg.RESULTS_JSON)
    if not path.exists():
        return []
    return json.loads(path.read_text(encoding="utf-8"))


def results_table(results: list[dict] | None = None, notebooks=None) -> str:
    """Markdown table of every recorded arm. Used to regenerate the README."""
    results = results if results is not None else load_results()
    if notebooks:
        results = [r for r in results if r.get("notebook") in notebooks]
    if not results:
        return "_No results recorded yet — run the notebooks; they write outputs/results.json._"

    def fmt(entry: dict, key: str) -> str:
        v = entry.get(key)
        if v is None:
            return "—"
        if isinstance(v, dict) and "mean" in v:
            return f"{v['mean']:.3f} ± {v['std']:.3f}"
        return f"{v:.3f}" if isinstance(v, float) else str(v)

    lines = [
        "| Notebook | Arm | Input | Test accuracy | Test macro-F1 | Params | Notes |",
        "|---|---|---|---|---|---|---|",
    ]
    for r in sorted(results, key=lambda r: (str(r.get("notebook")), str(r.get("arm")))):
        params = r.get("params")
        lines.append(
            f"| {r.get('notebook', '—')} | {r.get('arm', '—')} | {r.get('input', '—')} | "
            f"{fmt(r, 'test_accuracy')} | {fmt(r, 'test_macro_f1')} | "
            f"{f'{params / 1e6:.2f}M' if isinstance(params, (int, float)) else '—'} | "
            f"{r.get('notes', '')} |"
        )
    return "\n".join(lines)
