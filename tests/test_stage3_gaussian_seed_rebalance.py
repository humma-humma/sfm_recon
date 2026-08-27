import numpy as np

from sfm_reconstruction.stage3_gaussian_seed_rebalance import (
    focus_visibility_counts,
    select_rebalanced_indices,
)


def test_focus_visibility_counts_accumulates_camera_support():
    points = np.asarray([[0.0, 0.0, 2.0], [0.5, 0.0, 2.0], [5.0, 0.0, 1.0]])
    poses = [(np.eye(3), np.zeros(3)), (np.eye(3), np.zeros(3))]

    counts = focus_visibility_counts(
        points,
        poses,
        (10.0, 10.0, 5.0, 5.0, 10, 10),
    )

    np.testing.assert_array_equal(counts, [2, 2, 0])


def test_select_rebalanced_indices_reserves_focus_budget():
    counts = np.asarray([0, 2, 0, 3, 0, 2, 0, 0], dtype=np.uint16)

    selected = select_rebalanced_indices(
        counts,
        total_points=5,
        focus_points=3,
        min_focus_views=2,
    )

    assert len(selected) == 5
    assert set((1, 3, 5)).issubset(selected)
    assert np.all(np.diff(selected) > 0)


def test_select_rebalanced_indices_rejects_invalid_budget():
    counts = np.zeros(4, dtype=np.uint16)
    for kwargs in (
        {"total_points": 0, "focus_points": 0, "min_focus_views": 1},
        {"total_points": 3, "focus_points": 4, "min_focus_views": 1},
        {"total_points": 3, "focus_points": 1, "min_focus_views": 0},
        {"total_points": 5, "focus_points": 1, "min_focus_views": 1},
    ):
        try:
            select_rebalanced_indices(counts, **kwargs)
        except ValueError:
            pass
        else:
            raise AssertionError("expected invalid rebalance budget to fail")
