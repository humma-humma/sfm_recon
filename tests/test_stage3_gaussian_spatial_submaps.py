import numpy as np

from sfm_reconstruction.stage3_gaussian_spatial_submaps import (
    deterministic_spatial_labels,
    expanded_cluster_indices,
)


def test_deterministic_spatial_labels_groups_nearby_cameras():
    centers = np.asarray(
        [[0.0, 0.0, 0.0], [0.2, 0.0, 0.0], [10.0, 0.0, 0.0], [10.2, 0.0, 0.0]]
    )

    labels = deterministic_spatial_labels(centers, 2)

    np.testing.assert_array_equal(labels, [0, 0, 1, 1])


def test_expanded_cluster_indices_adds_temporal_overlap():
    labels = np.asarray([0, 0, 1, 1, 1, 2, 2])

    indices = expanded_cluster_indices(labels, 1, overlap_cameras=1)

    np.testing.assert_array_equal(indices, [1, 2, 3, 4, 5])


def test_spatial_helpers_reject_invalid_inputs():
    try:
        deterministic_spatial_labels(np.zeros((2, 2)), 2)
    except ValueError:
        pass
    else:
        raise AssertionError("expected invalid camera centers to fail")

    try:
        expanded_cluster_indices(np.asarray([0, 1]), 0, overlap_cameras=-1)
    except ValueError:
        pass
    else:
        raise AssertionError("expected negative overlap to fail")
