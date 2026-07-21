"""Index arithmetic and rendering. These are the functions whose bugs are
invisible: a wrong NDVI still produces a plausible-looking number."""

import numpy as np
import pytest

from s2map import bands


def test_band_table_is_the_eurosat_order():
    # If this ever changes, every cached feature and checkpoint in the repo is
    # invalid, so it is pinned by test rather than by comment.
    assert bands.BAND_IDS == (
        "B01", "B02", "B03", "B04", "B05", "B06", "B07",
        "B08", "B08A", "B09", "B10", "B11", "B12",
    )
    assert bands.NUM_BANDS == 13
    assert bands.RGB_INDICES == (3, 2, 1)  # B04, B03, B02


def test_normalized_difference_matches_hand_computation():
    a = np.array([0.30, 0.50, 0.10])
    b = np.array([0.10, 0.50, 0.30])
    # (0.3-0.1)/(0.4)=0.5 ; (0.5-0.5)/1.0=0.0 ; (0.1-0.3)/0.4=-0.5
    np.testing.assert_allclose(bands.normalized_difference(a, b), [0.5, 0.0, -0.5])


def test_normalized_difference_handles_zero_denominator():
    out = bands.normalized_difference(np.zeros(4), np.zeros(4))
    assert np.all(np.isfinite(out)), "division by zero must not produce NaN/inf"
    np.testing.assert_array_equal(out, np.zeros(4))


def test_ndvi_on_a_synthetic_chip():
    chip = np.zeros((13, 4, 4), dtype=np.float64)
    chip[bands.band_index("B08")] = 0.40  # NIR
    chip[bands.band_index("B04")] = 0.10  # red
    # (0.40 - 0.10) / 0.50 = 0.6 everywhere
    np.testing.assert_allclose(bands.ndvi(chip), np.full((4, 4), 0.6))


def test_ndwi_and_ndbi_use_the_right_bands():
    chip = np.zeros((13, 2, 2))
    chip[bands.band_index("B03")] = 0.30
    chip[bands.band_index("B08")] = 0.10
    chip[bands.band_index("B11")] = 0.30
    np.testing.assert_allclose(bands.ndwi(chip), np.full((2, 2), 0.5))
    np.testing.assert_allclose(bands.ndbi(chip), np.full((2, 2), 0.5))


def test_classical_features_shape_and_names():
    rng = np.random.default_rng(0)
    x = rng.random((7, 13, 8, 8))
    feats = bands.classical_features(x)
    assert feats.shape == (7, 29)
    assert len(bands.CLASSICAL_FEATURE_NAMES) == 29
    # column 0 is B01 mean
    np.testing.assert_allclose(feats[:, 0], x[:, 0].reshape(7, -1).mean(axis=1))


def test_percentile_stretch_is_bounded_and_monotonic():
    rng = np.random.default_rng(1)
    x = rng.random((3, 16, 16)) * 5000 + 500
    out = bands.percentile_stretch(x)
    assert out.min() >= 0.0 and out.max() <= 1.0

    # monotonic: sorting order within a channel is preserved by a linear stretch
    flat_in = x[0].ravel()
    flat_out = out[0].ravel()
    order = np.argsort(flat_in)
    stretched = flat_out[order]
    assert np.all(np.diff(stretched) >= -1e-12)


def test_percentile_stretch_handles_a_constant_band():
    out = bands.percentile_stretch(np.full((2, 4, 4), 1234.0))
    assert np.all(np.isfinite(out)), "a constant band must not divide by zero"


def test_composites_return_hwc_and_correct_bands():
    chip = np.zeros((13, 5, 5))
    chip[bands.band_index("B04")] = 1.0  # red only
    tc = bands.true_color(chip, stretch=False)
    assert tc.shape == (5, 5, 3)
    np.testing.assert_allclose(tc[..., 0], 1.0)   # red channel
    np.testing.assert_allclose(tc[..., 1:], 0.0)


def test_to_uint8_rgb_is_display_ready():
    rng = np.random.default_rng(2)
    chip = rng.random((13, 64, 64)) * 3000
    rgb = bands.to_uint8_rgb(chip)
    assert rgb.dtype == np.uint8 and rgb.shape == (64, 64, 3)


def test_band_index_rejects_unknown_bands():
    with pytest.raises(KeyError):
        bands.band_index("B99")
