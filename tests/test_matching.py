import numpy as np

from sfm_reconstruction.matching import (
    FeatureSet,
    _feature_mask,
    _feature_methods,
    _match_feature_sets,
    circular_image_pairs,
    match_descriptors,
)


def test_custom_matcher_applies_ratio_and_mutual_checks():
    first = np.array(
        [[0.0, 0.0], [1.0, 1.0], [10.0, 10.0]],
        dtype=np.float32,
    )
    second = np.array(
        [[0.01, 0.0], [1.01, 1.0], [20.0, 20.0], [11.0, 11.0]],
        dtype=np.float32,
    )

    matches = match_descriptors(first, second, ratio_threshold=0.8)

    assert matches.tolist() == [[0, 0], [1, 1], [2, 3]]


def test_binary_matcher_applies_ratio_and_mutual_checks():
    first = np.array([[0], [255], [15]], dtype=np.uint8)
    second = np.array([[0], [240], [255], [128]], dtype=np.uint8)

    matches = match_descriptors(
        first,
        second,
        ratio_threshold=0.8,
        descriptor_type="binary",
    )

    assert matches.tolist() == [[0, 0], [1, 2]]


def test_feature_mode_can_combine_sift_and_akaze_sources():
    assert _feature_methods("sift+akaze") == ("sift", "akaze")

    first = (
        FeatureSet(
            points=np.array([[1.0, 2.0]]),
            descriptors=np.array([[0.0, 0.0]], dtype=np.float32),
            descriptor_type="float",
            source="sift",
        ),
        FeatureSet(
            points=np.array([[3.0, 4.0]]),
            descriptors=np.array([[0]], dtype=np.uint8),
            descriptor_type="binary",
            source="akaze",
        ),
    )
    second = (
        FeatureSet(
            points=np.array([[1.5, 2.5], [9.0, 9.0]]),
            descriptors=np.array([[0.0, 0.0], [10.0, 10.0]], dtype=np.float32),
            descriptor_type="float",
            source="sift",
        ),
        FeatureSet(
            points=np.array([[3.5, 4.5], [8.0, 8.0]]),
            descriptors=np.array([[0], [255]], dtype=np.uint8),
            descriptor_type="binary",
            source="akaze",
        ),
    )

    correspondences = _match_feature_sets(first, second, ratio_threshold=0.8)

    assert correspondences.tolist() == [
        [1.0, 2.0, 1.5, 2.5],
        [3.0, 4.0, 3.5, 4.5],
    ]


def test_circular_pairs_include_loop_closure_neighbors():
    pairs = circular_image_pairs([0, 10, 20, 30, 40], pair_window=1)

    assert pairs == [(0, 10), (0, 40), (10, 20), (20, 30), (30, 40)]


def test_feature_mask_is_disabled_by_default():
    image = np.zeros((20, 30, 3), dtype=np.uint8)

    assert _feature_mask(image, False, 0.12) is None
