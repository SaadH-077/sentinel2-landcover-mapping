"""Tiling and stitching.

The constant-model test is the one that earns its keep: an off-by-one in the
overlap-averaging weights produces a map that is subtly darker along every tile
seam, which is easy to miss by eye on a 1500x1500 map and immediately visible
here.
"""

import numpy as np
import pytest

from s2map import inference


@pytest.mark.parametrize("shape", [(64, 64), (100, 100), (137, 91), (300, 250), (33, 200)])
def test_stitched_output_has_exactly_the_input_shape(shape):
    h, w = shape
    scene = np.random.default_rng(0).random((13, h, w)).astype(np.float32)

    def constant_model(batch):
        return np.tile(np.array([0.1, 0.9]), (len(batch), 1))

    probs = inference.sliding_window_predict(scene, constant_model, num_classes=2, tile=64, stride=32)
    assert probs.shape == (2, h, w), "stitching must reconstruct the input dimensions exactly"


def test_constant_model_gives_a_constant_map_including_at_seams():
    scene = np.random.default_rng(1).random((13, 200, 173)).astype(np.float32)
    target = np.array([0.25, 0.7, 0.05])

    def constant_model(batch):
        return np.tile(target, (len(batch), 1))

    probs = inference.sliding_window_predict(scene, constant_model, num_classes=3, tile=64, stride=32)
    for c in range(3):
        np.testing.assert_allclose(probs[c], target[c], atol=1e-5)
    # and therefore no visible seam structure
    assert probs.std(axis=(1, 2)).max() < 1e-6


def test_probabilities_still_sum_to_one_after_averaging():
    scene = np.random.default_rng(2).random((13, 150, 150)).astype(np.float32)
    rng = np.random.default_rng(3)

    def random_model(batch):
        p = rng.random((len(batch), 4))
        return p / p.sum(axis=1, keepdims=True)

    probs = inference.sliding_window_predict(scene, random_model, num_classes=4, tile=64, stride=32)
    np.testing.assert_allclose(probs.sum(axis=0), 1.0, atol=1e-5)


def test_every_pixel_is_covered_by_at_least_one_tile():
    positions = inference.tile_positions(137, 91, tile=64, stride=32)
    covered = np.zeros((137, 91), dtype=int)
    for top, left in positions:
        covered[top : top + 64, left : left + 64] += 1
    assert covered.min() >= 1


def test_padding_is_reflective_not_zero():
    scene = np.ones((2, 70, 70), dtype=np.float32) * 5.0
    padded, (ph, pw) = inference.pad_for_tiling(scene, tile=64, stride=32)
    assert (padded.shape[1] - 64) % 32 == 0
    assert padded.min() == 5.0, "reflection padding must not introduce zeros"


def test_padding_grows_an_undersized_scene_to_the_tile_size():
    padded, _ = inference.pad_for_tiling(np.ones((3, 20, 40), dtype=np.float32), tile=64, stride=32)
    assert padded.shape == (3, 64, 64)


def test_class_areas_convert_pixels_to_hectares():
    class_map = np.zeros((100, 100), dtype=np.uint8)
    class_map[:50] = 1  # 5000 pixels of class 1
    areas = inference.class_areas(class_map, ["a", "b"], pixel_size_m=10.0)
    assert areas["b"]["pixels"] == 5000
    assert areas["b"]["hectares"] == pytest.approx(50.0)  # 5000 * 100 m^2 = 50 ha
    assert areas["a"]["fraction"] == pytest.approx(0.5)


def test_nodata_is_excluded_from_area_fractions():
    class_map = np.full((10, 10), 255, dtype=np.uint8)
    class_map[:5] = 0
    areas = inference.class_areas(class_map, ["a", "b"], nodata=255)
    assert areas["a"]["fraction"] == pytest.approx(1.0)
    assert areas["_nodata"]["pixels"] == 50


def test_segment_majority_vote_cleans_up_a_noisy_segment():
    class_map = np.array([[0, 0, 1], [0, 1, 1], [2, 2, 2]], dtype=np.uint8)
    segments = np.array([[1, 1, 1], [1, 1, 1], [2, 2, 2]], dtype=np.int32)
    out = inference.majority_vote_by_segment(class_map, segments, num_classes=3)
    # segment 1 holds three 0s and three 1s -> argmax of the tie goes to 0
    assert set(np.unique(out[:2])) == {0}
    assert set(np.unique(out[2])) == {2}


def test_agreement_matrix_counts_pairs():
    a = np.array([0, 0, 1, 1])
    b = np.array([0, 1, 1, 1])
    m = inference.agreement_matrix(a, b, 2, 2)
    np.testing.assert_array_equal(m, [[1, 1], [0, 2]])
