"""Metrics against hand computation.

Everything here is checked against a number worked out by hand in the comment,
not against another library's output — otherwise the test only proves the two
implementations are wrong in the same way.
"""

import numpy as np
import pytest

from s2map import evaluate


def test_confusion_matrix_rows_are_true_classes():
    y_true = np.array([0, 0, 1, 1, 2])
    y_pred = np.array([0, 1, 1, 1, 0])
    cm = evaluate.confusion_matrix(y_true, y_pred, 3)
    np.testing.assert_array_equal(cm, [[1, 1, 0], [0, 2, 0], [1, 0, 0]])


def test_macro_f1_matches_hand_computation():
    # Class 0: TP=1, FP=1 (from class 2), FN=1  -> P=1/2, R=1/2, F1=0.5
    # Class 1: TP=2, FP=1 (from class 0), FN=0  -> P=2/3, R=1,   F1=0.8
    # Class 2: TP=0                             -> F1=0
    # macro-F1 = (0.5 + 0.8 + 0.0) / 3 = 0.43333...
    y_true = np.array([0, 0, 1, 1, 2])
    y_pred = np.array([0, 1, 1, 1, 0])
    assert evaluate.macro_f1(y_true, y_pred, 3) == pytest.approx(1.3 / 3)


def test_macro_f1_punishes_ignoring_a_class_more_than_accuracy_does():
    # 90 samples of class 0, 10 of class 1; predict class 0 always.
    y_true = np.array([0] * 90 + [1] * 10)
    y_pred = np.zeros(100, dtype=int)
    assert evaluate.accuracy(y_true, y_pred) == pytest.approx(0.90)
    # class 0 F1 = 2*0.9*1/1.9 = 0.947 ; class 1 F1 = 0 -> macro = 0.474
    assert evaluate.macro_f1(y_true, y_pred, 2) == pytest.approx(0.4737, abs=1e-4)


def test_perfect_prediction_scores_one():
    y = np.array([0, 1, 2, 3, 4])
    assert evaluate.macro_f1(y, y, 5) == pytest.approx(1.0)
    assert evaluate.accuracy(y, y) == pytest.approx(1.0)


def test_precision_of_a_never_predicted_class_is_zero_not_nan():
    prf = evaluate.per_class_prf(np.array([0, 1]), np.array([0, 0]), 2)
    assert np.all(np.isfinite(prf["precision"]))
    assert prf["precision"][1] == 0.0


def test_row_normalize_rows_sum_to_one():
    cm = np.array([[3, 1], [0, 0]])
    r = evaluate.row_normalize(cm)
    np.testing.assert_allclose(r[0], [0.75, 0.25])
    np.testing.assert_allclose(r[1], [0.0, 0.0])  # empty row must not be NaN


def test_softmax_rows_are_distributions():
    p = evaluate.softmax(np.array([[1.0, 2.0, 3.0], [0.0, 0.0, 0.0]]))
    np.testing.assert_allclose(p.sum(axis=1), 1.0)
    assert p[1].std() == pytest.approx(0.0)


def test_ece_is_zero_for_a_perfectly_calibrated_case():
    # 100 predictions at exactly 0.8 confidence, of which exactly 80 are right.
    n, conf = 100, 0.8
    probs = np.zeros((n, 2))
    probs[:, 0] = conf
    probs[:, 1] = 1 - conf
    labels = np.array([0] * 80 + [1] * 20)
    ece, curve = evaluate.expected_calibration_error(probs, labels, n_bins=10)
    assert ece == pytest.approx(0.0, abs=1e-9)
    assert sum(curve["count"]) == n


def test_ece_detects_overconfidence():
    # 100 predictions at 0.99 confidence, only 50 correct -> ECE ~ 0.49
    probs = np.full((100, 2), 0.01)
    probs[:, 0] = 0.99
    labels = np.array([0] * 50 + [1] * 50)
    ece, _ = evaluate.expected_calibration_error(probs, labels, n_bins=10)
    assert ece == pytest.approx(0.49, abs=1e-6)


def test_temperature_scaling_leaves_the_argmax_untouched():
    rng = np.random.default_rng(0)
    logits = rng.normal(size=(200, 5)) * 3
    labels = logits.argmax(1)
    labels[:40] = (labels[:40] + 1) % 5  # inject errors so T is not degenerate
    t = evaluate.fit_temperature(logits, labels)
    assert t > 0
    before = evaluate.softmax(logits).argmax(1)
    after = evaluate.softmax(logits, t).argmax(1)
    np.testing.assert_array_equal(before, after)


def test_temperature_scaling_reduces_ece_on_overconfident_logits():
    rng = np.random.default_rng(1)
    logits = rng.normal(size=(500, 4)) * 8  # very peaked -> overconfident
    labels = logits.argmax(1)
    flip = rng.random(500) < 0.3
    labels[flip] = (labels[flip] + 1) % 4
    t = evaluate.fit_temperature(logits, labels)
    ece_before, _ = evaluate.expected_calibration_error(evaluate.softmax(logits), labels)
    ece_after, _ = evaluate.expected_calibration_error(evaluate.softmax(logits, t), labels)
    assert ece_after < ece_before


def test_results_ledger_replaces_an_entry_for_the_same_arm(tmp_path):
    path = tmp_path / "results.json"
    evaluate.append_result({"notebook": "02", "arm": "resnet18", "test_macro_f1": 0.1}, path)
    evaluate.append_result({"notebook": "02", "arm": "resnet18", "test_macro_f1": 0.9}, path)
    evaluate.append_result({"notebook": "02", "arm": "small_cnn", "test_macro_f1": 0.5}, path)
    results = evaluate.load_results(path)
    assert len(results) == 2, "re-running a notebook must overwrite, not duplicate"
    assert [r for r in results if r["arm"] == "resnet18"][0]["test_macro_f1"] == 0.9


def test_results_table_renders_mean_and_std(tmp_path):
    path = tmp_path / "results.json"
    evaluate.append_result(
        {
            "notebook": "02",
            "arm": "resnet18",
            "input": "13-band",
            "test_macro_f1": {"mean": 0.912, "std": 0.004},
            "params": 11_200_000,
        },
        path,
    )
    table = evaluate.results_table(evaluate.load_results(path))
    assert "0.912 ± 0.004" in table
    assert "11.20M" in table
