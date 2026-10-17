import numpy as np
import pytest

pytest.importorskip("sfm_reconstruction._native")

from sfm_reconstruction.native_backend import load_native
from sfm_reconstruction.stage3_gaussian_seed_rebalance import focus_visibility_counts
from sfm_reconstruction.stage3_gaussian_global_seed_balance import region_visibility_counts


@pytest.mark.parametrize("threads", [1, 4])
def test_support_matches_reference(threads):
    native = load_native()
    rng = np.random.default_rng(31)
    points = rng.normal(size=(3017, 3))
    poses = [(np.eye(3), rng.normal(size=3)) for _ in range(13)]
    intrinsics = (40., 50., 32., 24., 64, 48)
    expected = focus_visibility_counts(points, poses, intrinsics)
    actual = focus_visibility_counts(points, poses, intrinsics, backend="cpp", num_threads=threads)
    np.testing.assert_array_equal(actual, expected)
    assert native.camera_support(np.empty((0, 3)), np.empty((0, 3, 3)),
                                 np.empty((0, 3)), np.ones(4), 64, 48, threads).size == 0
    a, regions = region_visibility_counts(points, poses, intrinsics, 4)
    b, native_regions = region_visibility_counts(points, poses, intrinsics, 4,
                                                 backend="cpp", num_threads=threads)
    np.testing.assert_array_equal(a, b)
    for expected_region, actual_region in zip(regions, native_regions):
        np.testing.assert_array_equal(expected_region, actual_region)


def test_support_handles_boundaries_and_rejects_invalid_shapes():
    points = np.array([[0., 0., 1.], [2., 0., 1.], [0., 2., 1.],
                       [0., 0., 0.], [0., 0., -1.], [np.nan, 0., 1.]])
    k = (1., 1., 0., 0., 2, 2)
    np.testing.assert_array_equal(focus_visibility_counts(points, [(np.eye(3), np.zeros(3))],
                                  k, backend="cpp"), [1, 0, 0, 0, 0, 0])
    with pytest.raises(ValueError):
        focus_visibility_counts(np.zeros((3, 2)), [(np.eye(3), np.zeros(3))], k, backend="cpp")
    with pytest.raises(ValueError):
        focus_visibility_counts(points, [], k, backend="cpp", num_threads=0)
    np.testing.assert_array_equal(focus_visibility_counts(points, [], k, backend="cpp"), np.zeros(6))


def test_depth_evidence_distinguishes_occlusion_and_missing_depth():
    # Identity camera, a plane at 2 m: nearer points contradict free space;
    # farther points are occluded and must not be treated as contradictions.
    points = np.array([[0., 0., z] for z in [2., 1., 3., -1., np.nan]])
    args = (points, np.eye(3), np.zeros(3), np.array([1., 1., 1., 1.]))
    native = load_native()
    np.testing.assert_array_equal(native.depth_support(*args, np.full((3, 3), 2.), .05, .02),
                                  [1, 2, 3, 0, 0])
    np.testing.assert_array_equal(native.depth_support(*args, np.zeros((3, 3)), .05, .02),
                                  np.zeros(5))
    with pytest.raises(ValueError):
        native.depth_support(*args, np.ones((3, 3)), -.1, .02)
