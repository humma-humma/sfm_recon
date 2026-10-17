import copy

import numpy as np
import pytest

from sfm_reconstruction.splat_cleanup_evaluation import compare_candidate
from sfm_reconstruction.stage3_gaussian_depth_cleanup import depth_labels, depth_keep_mask


def report():
    return {"views": [{"image": n, "psnr": 20., "ssim": .8, "lpips": .2,
                        "opacity_coverage": .8} for n in ("a.jpg", "b.jpg")]}


def test_cleanup_gate_rejects_local_hole_despite_improved_average():
    base = report()
    candidate = copy.deepcopy(base)
    candidate["views"][0]["psnr"] = 19.
    candidate["views"][1]["psnr"] = 22.
    result = compare_candidate(base, candidate)
    assert result["mean_deltas"]["psnr"] > 0
    assert result["status"] == "rejected"
    assert not result["checks"]["worst_view_psnr"]


def test_cleanup_gate_requires_matching_views_and_visual_review():
    base = report()
    assert compare_candidate(base, base)["status"] == "passes_numeric_gate_visual_review_pending"
    candidate = report()
    candidate["views"][0]["image"] = "different.jpg"
    with pytest.raises(ValueError):
        compare_candidate(base, candidate)
    candidate = report()
    candidate["views"][0]["lpips"] = float("nan")
    with pytest.raises(ValueError):
        compare_candidate(base, candidate)


def test_depth_rule_keeps_occluded_unknown_and_supported_centers():
    points = np.array([[0., 0., z] for z in [2., 1., 3., -1.]])
    labels = depth_labels(points, np.eye(3), np.zeros(3), np.array([1., 1., 1., 1.]),
                          np.full((3, 3), 2.), .1, .05)
    np.testing.assert_array_equal(labels, [1, 2, 3, 0])
    np.testing.assert_array_equal(depth_keep_mask(np.ones(4, bool),
                                 np.array([1, 0, 0, 0]), np.array([5, 3, 0, 0])),
                                 [True, False, True, True])
    with pytest.raises(ValueError):
        depth_keep_mask(np.ones(4, bool), np.zeros(4), np.zeros(3))
