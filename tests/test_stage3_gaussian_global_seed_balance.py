import numpy as np

from sfm_reconstruction.stage3_gaussian_global_seed_balance import (
    region_visibility_counts,
    select_region_balanced_indices,
)


def test_region_visibility_counts_splits_cameras_evenly():
    points = np.asarray([[0.0, 0.0, 2.0], [5.0, 0.0, 1.0]])
    poses = [(np.eye(3), np.zeros(3)) for _ in range(4)]

    counts, regions = region_visibility_counts(
        points,
        poses,
        (10.0, 10.0, 5.0, 5.0, 10, 10),
        2,
    )

    np.testing.assert_array_equal(counts, [[2, 0], [2, 0]])
    np.testing.assert_array_equal(regions[0], [0, 1])
    np.testing.assert_array_equal(regions[1], [2, 3])


def test_region_balanced_selection_reserves_each_region_budget():
    counts = np.asarray(
        [
            [2, 3, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 2, 3, 0, 0],
        ],
        dtype=np.uint16,
    )

    selected, per_region = select_region_balanced_indices(
        counts,
        total_points=4,
        min_region_views=2,
    )

    np.testing.assert_array_equal(selected, [0, 1, 4, 5])
    assert per_region == [2, 2]


def test_region_balanced_selection_fills_missing_region_budget():
    counts = np.asarray([[2, 2, 0, 0, 0], [0, 0, 0, 0, 0]], dtype=np.uint16)

    selected, per_region = select_region_balanced_indices(
        counts,
        total_points=4,
        min_region_views=2,
    )

    assert len(selected) == 4
    assert per_region == [2, 0]
    assert set((0, 1)).issubset(selected)


def test_region_balanced_selection_preserves_baseline_indices():
    counts = np.asarray(
        [[2, 2, 0, 0, 0, 0], [0, 0, 0, 2, 2, 0]], dtype=np.uint16
    )

    selected, per_region = select_region_balanced_indices(
        counts,
        total_points=4,
        min_region_views=2,
        baseline_indices=np.asarray([0, 5]),
    )

    assert set((0, 5)).issubset(selected)
    assert len(selected) == 4
    assert per_region == [1, 1]
