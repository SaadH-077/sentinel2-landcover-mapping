"""Training-loop robustness.

These exist because of a real failure: loading a frozen CLIP encoder in an
earlier notebook cell disabled autograd globally, and the fine-tuning arm three
sections later died with

    RuntimeError: element 0 of tensors does not require grad and does not
    have a grad_fn

with nothing in the traceback pointing back at the cause.
"""

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from s2map import config as cfg  # noqa: E402
from s2map import data as D  # noqa: E402
from s2map import models as M  # noqa: E402
from s2map import train as T  # noqa: E402


@pytest.fixture
def tiny_problem():
    """40 synthetic chips with a class-dependent band offset, so it is learnable."""
    cfg.set_seed(0)
    rng = np.random.default_rng(0)
    y = np.repeat(np.arange(4), 10)
    X = (rng.normal(size=(40, 13, 16, 16)) * 100 + 1000).astype(np.uint16)
    for c in range(4):
        X[y == c, c] += 1500
    stats = D.compute_band_stats(X, np.arange(40))
    ds = D.EuroSATChips(X, y, np.arange(40), stats)
    loader = D.make_loader(ds, 8, False, 0, 0)
    return loader


def _one_epoch_config():
    return cfg.TrainConfig(epochs=1, batch_size=8, patience=1, amp=False, num_workers=0,
                           warmup_epochs=0)


def test_fit_trains_when_grad_is_globally_disabled(tiny_problem):
    """The exact NB04 failure: CLIP was loaded, so grad was off session-wide."""
    previous = torch.is_grad_enabled()
    try:
        torch.set_grad_enabled(False)
        model = M.SmallCNN(13, 4, width=8)
        before = [p.detach().clone() for p in model.parameters()]
        model, history = T.fit(model, tiny_problem, tiny_problem, _one_epoch_config(),
                               device="cpu", verbose=False)
        after = list(model.parameters())
        assert any(not torch.equal(b, a) for b, a in zip(before, after)), (
            "no parameter changed — the model did not actually train"
        )
        assert history.epochs_run == 1
    finally:
        torch.set_grad_enabled(previous)


def test_fit_restores_nothing_it_should_not_and_still_evaluates(tiny_problem):
    model = M.SmallCNN(13, 4, width=8)
    model, _ = T.fit(model, tiny_problem, tiny_problem, _one_epoch_config(),
                     device="cpu", verbose=False)
    logits, labels = T.predict(model, tiny_problem, device="cpu", amp=False)
    assert logits.shape == (40, 4)
    assert labels.shape == (40,)


def test_predict_does_not_leave_grad_enabled_or_disabled(tiny_problem):
    """predict() uses @torch.no_grad(); it must not leak that state to callers."""
    model = M.SmallCNN(13, 4, width=8)
    assert torch.is_grad_enabled()
    T.predict(model, tiny_problem, device="cpu", amp=False)
    assert torch.is_grad_enabled(), "predict() left autograd globally disabled"


def test_cosine_warmup_schedule_shape():
    fn = T.cosine_warmup_lambda(total_epochs=10, warmup_epochs=2)
    assert fn(0) == pytest.approx(0.5)   # linear warmup, first of two steps
    assert fn(1) == pytest.approx(1.0)   # warmup complete
    assert fn(2) > fn(5) > fn(9)         # monotonically decaying thereafter
    assert fn(9) > 0                     # never reaches zero


# --------------------------------------------------------------------------
# Grad-CAM hooks
# --------------------------------------------------------------------------
def test_grad_cam_runs_on_a_batch_and_returns_normalised_maps():
    """Regression: hooks written as `dict.setdefault` lambdas RETURN a value,
    which torch reads as replacement grad_input, giving
    "Backward hook returned an invalid number of grad_input, got 8, expected 1"
    where 8 was simply the batch size."""
    model = M.build_resnet18(13, 10, pretrained=False).eval()
    x = torch.randn(8, 13, 64, 64)
    cams, idx = M.grad_cam(model, x)

    assert cams.shape == (8, 64, 64), cams.shape
    assert idx.shape == (8,)
    assert cams.min() >= 0.0 and cams.max() <= 1.0
    assert np.all(np.isfinite(cams))


def test_grad_cam_removes_its_hooks_and_leaves_the_model_unchanged():
    """The forward-hook version of the same bug does NOT raise: it silently
    replaces the layer's output with a stale tensor. Check the model still
    computes exactly what it did before being hooked."""
    model = M.build_resnet18(13, 10, pretrained=False).eval()
    x = torch.randn(4, 13, 64, 64)
    with torch.no_grad():
        before = model(x).clone()

    M.grad_cam(model, x)
    M.grad_cam(model, x)  # twice: setdefault would now serve the first batch's activations

    layer = model.backbone.layer4
    assert not layer._forward_hooks, "forward hook was left attached"
    assert not layer._backward_hooks and not getattr(layer, "_backward_pre_hooks", {}), \
        "backward hook was left attached"
    with torch.no_grad():
        after = model(x)
    torch.testing.assert_close(before, after)


def test_grad_cam_accepts_an_explicit_target_class():
    model = M.build_resnet18(13, 10, pretrained=False).eval()
    x = torch.randn(3, 13, 64, 64)
    target = torch.tensor([1, 2, 3])
    _, idx = M.grad_cam(model, x, target=target)
    np.testing.assert_array_equal(idx, [1, 2, 3])
