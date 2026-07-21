"""Shared training loop: AdamW, cosine schedule with warmup, AMP, early stopping.

One loop for every arm. If Arm 1 and Arm 2 were trained by two slightly
different loops, the comparison between them would measure the loops as much as
the models.

Model selection is on validation macro-F1, never on the test set. The test set
is touched exactly once per chapter, at the end.
"""

from __future__ import annotations

import copy
import math
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from . import config as cfg
from .evaluate import macro_f1


@dataclass
class TrainHistory:
    train_loss: list[float] = field(default_factory=list)
    val_loss: list[float] = field(default_factory=list)
    val_acc: list[float] = field(default_factory=list)
    val_macro_f1: list[float] = field(default_factory=list)
    lr: list[float] = field(default_factory=list)
    best_epoch: int = -1
    best_val_macro_f1: float = -1.0
    epochs_run: int = 0
    train_seconds: float = 0.0

    def to_dict(self) -> dict:
        return {
            "train_loss": self.train_loss,
            "val_loss": self.val_loss,
            "val_acc": self.val_acc,
            "val_macro_f1": self.val_macro_f1,
            "lr": self.lr,
            "best_epoch": self.best_epoch,
            "best_val_macro_f1": self.best_val_macro_f1,
            "epochs_run": self.epochs_run,
            "train_seconds": self.train_seconds,
        }


def cosine_warmup_lambda(total_epochs: int, warmup_epochs: int, min_factor: float = 0.01):
    """LR multiplier: linear warmup then cosine decay to `min_factor`.

    Warmup matters here specifically because of the adapted stem: on the first
    steps the freshly-scaled 13-channel conv produces activations the pretrained
    BatchNorm statistics have never seen, and a full-size LR at that moment can
    wreck the pretrained weights before they adapt.
    """

    def fn(epoch: int) -> float:
        if warmup_epochs > 0 and epoch < warmup_epochs:
            return (epoch + 1) / warmup_epochs
        progress = (epoch - warmup_epochs) / max(1, total_epochs - warmup_epochs)
        progress = min(max(progress, 0.0), 1.0)
        return min_factor + (1 - min_factor) * 0.5 * (1 + math.cos(math.pi * progress))

    return fn


@torch.no_grad()
def predict(model: nn.Module, loader, device: str = "cuda", amp: bool = True):
    """Return (logits, labels) as float32/int64 numpy arrays."""
    model.eval().to(device)
    logits_all, labels_all = [], []
    for x, y in loader:
        x = x.to(device, non_blocking=True)
        with torch.autocast(device_type="cuda", enabled=amp and device == "cuda"):
            logits = model(x)
        logits_all.append(logits.float().cpu().numpy())
        labels_all.append(np.asarray(y))
    return np.concatenate(logits_all), np.concatenate(labels_all)


def evaluate_loader(model, loader, criterion, device="cuda", amp=True) -> dict:
    logits, labels = predict(model, loader, device=device, amp=amp)
    loss = float(
        criterion(torch.from_numpy(logits), torch.from_numpy(labels).long()).item()
    )
    preds = logits.argmax(1)
    return {
        "loss": loss,
        "accuracy": float((preds == labels).mean()),
        "macro_f1": float(macro_f1(labels, preds)),
        "logits": logits,
        "labels": labels,
    }


def fit(
    model: nn.Module,
    train_loader,
    val_loader,
    train_cfg: cfg.TrainConfig | None = None,
    device: str | None = None,
    checkpoint_path: Path | str | None = None,
    verbose: bool = True,
) -> tuple[nn.Module, TrainHistory]:
    """Train with early stopping on validation macro-F1; return the BEST model.

    Returns the best-epoch weights, not the last-epoch weights. With early
    stopping patience 10 the last epoch is by construction worse than the best
    one, so reporting the final weights would systematically understate every
    arm.
    """
    train_cfg = train_cfg or cfg.TrainConfig()
    device = device or cfg.get_device()
    model = model.to(device)

    criterion = nn.CrossEntropyLoss(label_smoothing=train_cfg.label_smoothing)
    optimiser = torch.optim.AdamW(
        model.parameters(), lr=train_cfg.lr, weight_decay=train_cfg.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimiser, cosine_warmup_lambda(train_cfg.epochs, train_cfg.warmup_epochs)
    )
    scaler = torch.amp.GradScaler("cuda", enabled=train_cfg.amp and device == "cuda")

    history = TrainHistory()
    best_state = copy.deepcopy(model.state_dict())
    epochs_without_improvement = 0
    t0 = time.time()

    # Explicitly enable autograd for the training loop rather than inheriting
    # whatever ambient grad state the notebook happens to be in. A frozen
    # feature extractor loaded earlier in the same session must not be able to
    # silently disable training here.
    for epoch in range(train_cfg.epochs):
        model.train()
        running, n_seen = 0.0, 0
        with torch.enable_grad():
            for x, y in train_loader:
                x = x.to(device, non_blocking=True)
                y = y.to(device, non_blocking=True)
                optimiser.zero_grad(set_to_none=True)
                with torch.autocast(device_type="cuda", enabled=train_cfg.amp and device == "cuda"):
                    loss = criterion(model(x), y)
                scaler.scale(loss).backward()
                if train_cfg.grad_clip is not None:
                    scaler.unscale_(optimiser)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), train_cfg.grad_clip)
                scaler.step(optimiser)
                scaler.update()
                running += float(loss.item()) * x.size(0)
                n_seen += x.size(0)

        history.lr.append(optimiser.param_groups[0]["lr"])
        scheduler.step()
        train_loss = running / max(n_seen, 1)
        val = evaluate_loader(model, val_loader, criterion, device, train_cfg.amp)

        history.train_loss.append(train_loss)
        history.val_loss.append(val["loss"])
        history.val_acc.append(val["accuracy"])
        history.val_macro_f1.append(val["macro_f1"])
        history.epochs_run = epoch + 1

        improved = val["macro_f1"] > history.best_val_macro_f1 + 1e-5
        if improved:
            history.best_val_macro_f1 = val["macro_f1"]
            history.best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        if verbose:
            print(
                f"epoch {epoch + 1:>3}/{train_cfg.epochs}  "
                f"train_loss {train_loss:.4f}  val_loss {val['loss']:.4f}  "
                f"val_acc {val['accuracy']:.4f}  val_macroF1 {val['macro_f1']:.4f}"
                + ("  <- best" if improved else "")
            )
        if epochs_without_improvement >= train_cfg.patience:
            if verbose:
                print(f"early stopping at epoch {epoch + 1} (patience {train_cfg.patience})")
            break

    history.train_seconds = time.time() - t0
    model.load_state_dict(best_state)
    if checkpoint_path is not None:
        checkpoint_path = Path(checkpoint_path)
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"state_dict": best_state, "history": history.to_dict()}, checkpoint_path)
    return model, history


def fit_multi_seed(
    model_fn,
    loader_fn,
    seeds=cfg.SEEDS,
    train_cfg: cfg.TrainConfig | None = None,
    device: str | None = None,
    checkpoint_prefix: Path | str | None = None,
    verbose: bool = True,
) -> dict:
    """Train the same arm under several seeds and summarise mean +/- std.

    `model_fn(seed) -> nn.Module` and `loader_fn(seed) -> (train, val, test)`.
    Both take the seed so that weight init AND data order vary together, which
    is what the reported std should be measuring.

    A single-seed number on a 19k-sample dataset is not a result; the spread
    across seeds is routinely larger than the gap between two of these arms.
    """
    per_seed, best = [], None
    for seed in seeds:
        cfg.set_seed(seed)
        if verbose:
            print(f"\n=== seed {seed} ===")
        train_loader, val_loader, test_loader = loader_fn(seed)
        ckpt = f"{checkpoint_prefix}_seed{seed}.pt" if checkpoint_prefix else None
        model, history = fit(
            model_fn(seed), train_loader, val_loader, train_cfg, device, ckpt, verbose
        )
        criterion = nn.CrossEntropyLoss()
        test = evaluate_loader(model, test_loader, criterion, device or cfg.get_device())
        per_seed.append(
            {
                "seed": seed,
                "val_macro_f1": history.best_val_macro_f1,
                "test_accuracy": test["accuracy"],
                "test_macro_f1": test["macro_f1"],
                "epochs_run": history.epochs_run,
                "train_seconds": history.train_seconds,
            }
        )
        if best is None or history.best_val_macro_f1 > best[1]:
            best = (model, history.best_val_macro_f1, test)

    def agg(key: str) -> dict:
        vals = np.array([r[key] for r in per_seed], dtype=float)
        return {"mean": float(vals.mean()), "std": float(vals.std(ddof=0)), "values": vals.tolist()}

    return {
        "per_seed": per_seed,
        "test_accuracy": agg("test_accuracy"),
        "test_macro_f1": agg("test_macro_f1"),
        "val_macro_f1": agg("val_macro_f1"),
        "train_seconds": agg("train_seconds"),
        "best_model": best[0] if best else None,
        "best_test": best[2] if best else None,
    }
