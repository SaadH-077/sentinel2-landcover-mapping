"""Split determinism, leakage and stratification.

The leakage test is the important one. A train/test overlap does not raise an
error, does not look wrong in any plot, and inflates every number in the
project. It is the failure mode that is worth a permanent test.
"""

import numpy as np
import pytest

from s2map import data


@pytest.fixture
def labels():
    # deliberately imbalanced so stratification has something to preserve
    rng = np.random.default_rng(0)
    return np.repeat(np.arange(10), rng.integers(200, 400, size=10))


def test_splits_are_deterministic(labels):
    a = data.make_stratified_splits(labels, seed=42)
    b = data.make_stratified_splits(labels, seed=42)
    for key in ("train", "val", "test"):
        np.testing.assert_array_equal(a[key], b[key])


def test_a_different_seed_gives_a_different_split(labels):
    a = data.make_stratified_splits(labels, seed=42)
    b = data.make_stratified_splits(labels, seed=43)
    assert not np.array_equal(a["train"], b["train"])


def test_splits_are_disjoint_and_complete(labels):
    splits = data.make_stratified_splits(labels)
    train, val, test = (set(splits[k].tolist()) for k in ("train", "val", "test"))
    assert train.isdisjoint(val), "train/val leakage"
    assert train.isdisjoint(test), "train/test leakage"
    assert val.isdisjoint(test), "val/test leakage"
    assert len(train | val | test) == labels.size, "some samples are in no split"


def test_split_sizes_match_the_requested_fractions(labels):
    splits = data.make_stratified_splits(labels, 0.7, 0.15, 0.15)
    n = labels.size
    assert abs(splits["train"].size / n - 0.70) < 0.01
    assert abs(splits["val"].size / n - 0.15) < 0.01
    assert abs(splits["test"].size / n - 0.15) < 0.01


def test_stratification_preserves_class_proportions(labels):
    splits = data.make_stratified_splits(labels)
    overall = np.bincount(labels, minlength=10) / labels.size
    for key in ("train", "val", "test"):
        got = np.bincount(labels[splits[key]], minlength=10) / splits[key].size
        np.testing.assert_allclose(got, overall, atol=0.02)


def test_fractions_must_sum_to_one(labels):
    with pytest.raises(ValueError):
        data.make_stratified_splits(labels, 0.8, 0.15, 0.15)


def test_splits_round_trip_through_disk(labels, tmp_path):
    splits = data.make_stratified_splits(labels)
    path = data.save_splits(splits, tmp_path / "splits.npz")
    loaded = data.load_splits(path)
    for key in splits:
        np.testing.assert_array_equal(splits[key], loaded[key])


def test_few_shot_draws_only_from_train_and_is_balanced(labels):
    splits = data.make_stratified_splits(labels)
    idx = data.few_shot_indices(labels, splits["train"], k=5, seed=0)
    assert set(idx.tolist()).issubset(set(splits["train"].tolist())), "few-shot leaked outside train"
    counts = np.bincount(labels[idx], minlength=10)
    np.testing.assert_array_equal(counts, np.full(10, 5))


def test_few_shot_draws_vary_with_seed(labels):
    splits = data.make_stratified_splits(labels)
    a = data.few_shot_indices(labels, splits["train"], k=5, seed=0)
    b = data.few_shot_indices(labels, splits["train"], k=5, seed=1)
    assert not np.array_equal(a, b), "different draws must differ, or k=1 variance is unmeasurable"


def test_band_stats_are_computed_on_train_only():
    rng = np.random.default_rng(0)
    X = rng.random((100, 13, 8, 8)).astype(np.float32)
    # make the non-train half wildly different; correct stats must ignore it
    X[50:] += 100.0
    train_idx = np.arange(50)
    stats = data.compute_band_stats(X, train_idx, chunk=16)
    expected_mean = X[:50].transpose(1, 0, 2, 3).reshape(13, -1).mean(axis=1)
    np.testing.assert_allclose(stats["mean"], expected_mean, rtol=1e-5)
    assert max(stats["mean"]) < 10.0, "test-split values leaked into the statistics"


def test_normalize_produces_zero_mean_unit_std_on_train():
    rng = np.random.default_rng(1)
    X = (rng.normal(size=(64, 13, 8, 8)) * 7 + 3).astype(np.float32)
    stats = data.compute_band_stats(X, np.arange(64))
    z = data.normalize(X, stats)
    assert abs(float(z.mean())) < 1e-3
    assert abs(float(z.std()) - 1.0) < 1e-2
