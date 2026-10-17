"""Acceptance tests for the optional Ceres solver."""
import importlib.util

import cv2
import numpy as np
import pytest

from sfm_reconstruction.bundle_adjustment import bundle_adjust
from sfm_reconstruction.geometry import project_points
from sfm_reconstruction.models import Pose, Track
from sfm_reconstruction.native_backend import load_ceres, validate_backend

pytestmark = pytest.mark.skipif(
    importlib.util.find_spec("sfm_reconstruction._ceres") is None,
    reason="Optional Ceres extension is not built",
)


def fixture():
    rng = np.random.default_rng(16)
    k = np.array([[700., 3., 320.], [0., 710., 240.], [0., 0., 1.]])
    xyz = np.column_stack((rng.uniform(-1, 1, (60, 2)), rng.uniform(4, 7, 60)))
    true = {i: Pose(cv2.Rodrigues(np.array([.01*i, -.02*i, .01*i]))[0], [-.3*i, 0., 0.]) for i in (2, 5, 9, 12)}
    poses = {i: p if i in (2, 5) else Pose(p.rotation, p.translation.ravel()+rng.normal(0, .01, 3)) for i, p in true.items()}
    pixels = {i: project_points(xyz, p, k)+rng.normal(0, .1, (len(xyz), 2)) for i, p in true.items()}
    tracks = [Track({i: pixels[i][j] for i in poses}) for j in range(len(xyz))]
    points = {i: p+rng.normal(0, .005, 3) for i, p in enumerate(xyz)}
    return k, tracks, poses, points


def cost(k, tracks, poses, points, scale=2.0):
    residuals = np.array([project_points(points[p].reshape(1, 3), poses[c], k)[0]-xy
                          for p, track in enumerate(tracks) for c, xy in track.observations.items()])
    return np.sum(scale**2*(np.sqrt(1+(residuals/scale)**2)-1))


@pytest.mark.parametrize("scale", [0.5, 2.0, 4.0])
def test_ceres_matches_scalar_robust_objective_and_preserves_inputs(scale):
    k, tracks, poses, points = fixture()
    # Anisotropic residual makes joint 2D and scalar robustification differ.
    tracks[0].observations[2] += np.array([3., .1])
    original_points = {i: p.copy() for i, p in points.items()}
    initial_cost = cost(k, tracks, poses, points, scale)
    report = {}
    result = bundle_adjust(k, tracks, poses, points, {2, 5}, backend="ceres", solver_report=report,
                           max_reprojection_error=100., loss_scale=scale)
    np.testing.assert_allclose(report["initial_cost"], initial_cost, rtol=1e-10)
    np.testing.assert_allclose(report["final_cost"], cost(k, tracks, *result, scale=scale), rtol=1e-8)
    assert report["final_cost"] < initial_cost
    assert report["converged"]
    for i in (2, 5):
        assert result[0][i] is poses[i]
    for i in points:
        np.testing.assert_array_equal(points[i], original_points[i])
    reference = bundle_adjust(k, tracks, poses, points, {2, 5}, backend="cpp", max_nfev=100,
                              max_reprojection_error=100., loss_scale=scale)
    np.testing.assert_allclose(cost(k, tracks, *result, scale=scale), cost(k, tracks, *reference, scale=scale), rtol=1e-4)


def test_ceres_retains_unobserved_points_and_empty_problem():
    k, tracks, poses, points = fixture()
    tracks.append(Track({}))
    points[len(tracks)-1] = np.array([1., 2., 9.])
    result = bundle_adjust(k, tracks, poses, points, {2, 5}, backend="ceres")
    np.testing.assert_array_equal(result[1][len(tracks)-1], points[len(tracks)-1])
    empty = bundle_adjust(k, [], poses, {}, {2, 5}, backend="ceres")
    assert empty[0] is poses and empty[1] == {}
    with pytest.raises(ValueError):
        validate_backend("ceres")  # Not a geometry or pose-graph backend.


def test_ceres_rejects_invalid_indices():
    native = load_ceres()
    with pytest.raises(ValueError, match="Observation index"):
        native.bundle_adjust(np.zeros((1, 6)), np.ones((1, 3)), np.array([[2, 0]]),
                             np.zeros((1, 2)), np.eye(3), np.array([0]), 10)


@pytest.mark.parametrize("solver", ["dense_schur", "sparse_schur", "iterative_schur"])
@pytest.mark.parametrize("threads", [1, 2])
def test_ceres_solver_options_preserve_solution_and_fixed_cameras(solver, threads):
    k, tracks, poses, points = fixture()
    report = {}
    actual = bundle_adjust(k, tracks, poses, points, {2, 5}, backend="ceres",
        max_reprojection_error=100., ceres_linear_solver=solver,
        ceres_num_threads=threads, solver_report=report)
    reference = bundle_adjust(k, tracks, poses, points, {2, 5}, backend="ceres",
        max_reprojection_error=100.)
    assert report["converged"]
    assert report["num_threads_requested"] == threads
    assert report["linear_solver_requested"] == solver
    assert report["linear_solver"] == solver.upper()
    np.testing.assert_allclose(cost(k, tracks, *actual), cost(k, tracks, *reference), rtol=2e-5)
    for i in (2, 5):
        assert actual[0][i] is poses[i]


@pytest.mark.parametrize("options", [{"ceres_num_threads": 0}, {"ceres_num_threads": 1.5},
    {"ceres_linear_solver": "unknown"}])
def test_invalid_ceres_options_fail_even_for_empty_problem(options):
    with pytest.raises(ValueError):
        bundle_adjust(np.eye(3), [], {}, {}, set(), backend="ceres", **options)
