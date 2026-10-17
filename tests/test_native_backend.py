"""Numerical acceptance tests for the optional compiled extension."""
import cv2
import numpy as np
import pytest
from scipy.optimize._numdiff import approx_derivative

pytest.importorskip("sfm_reconstruction._native")

from sfm_reconstruction.bundle_adjustment import bundle_adjust
from sfm_reconstruction.geometry import project_points, reprojection_errors, points_in_front, triangulation_angles
from sfm_reconstruction.models import Pose, Track
from sfm_reconstruction.native_backend import bundle_adjustment_callbacks
from sfm_reconstruction.native_backend import pose_graph_callbacks
from sfm_reconstruction.native_backend import sparse_callbacks
from sfm_reconstruction.pose_graph import (
    PoseGraph, PoseGraphNode, PoseGraphEdge, PoseGraphConfig,
    _vec6_from_matrix, optimize_pose_graph,
)
from sfm_reconstruction.stage3 import _transform_from_trajectory_pose


def make_bundle_problem(camera_count=5, point_count=100, seed=42):
    rng = np.random.default_rng(seed)
    k = np.array([[700., 0., 320.], [0., 710., 240.], [0., 0., 1.]])
    points = np.column_stack((rng.uniform(-1, 1, (point_count, 2)), rng.uniform(4, 7, point_count)))
    true = {i: Pose(cv2.Rodrigues(np.array([.02*i, -.03*i, .01*i]))[0], [-.3*i, .01*i, .02*i]) for i in range(camera_count)}
    poses = {
        i: p if i < 2 else Pose(
            cv2.Rodrigues(cv2.Rodrigues(p.rotation)[0].ravel() + rng.normal(0, .002, 3))[0],
            p.translation.ravel() + rng.normal(0, .01, 3),
        ) for i, p in true.items()
    }
    pixels = {i: project_points(points, p, k) + rng.normal(0, .1, (point_count, 2)) for i, p in true.items()}
    tracks = [Track({i: pixels[i][j] for i in true}) for j in range(point_count)]
    initial_points = {i: p + rng.normal(0, .01, 3) for i, p in enumerate(points)}
    return k, tracks, poses, initial_points


def test_ba_native_jacobian_matches_independent_finite_differences():
    k, tracks, poses, points = make_bundle_problem(4, 10)
    camera_offsets = {2: 0, 3: 6}
    point_offsets = {p: 12 + 3*p for p in points}
    observations = [(c, p, xy) for p, t in enumerate(tracks) for c, xy in t.observations.items()]
    values = np.concatenate([np.r_[cv2.Rodrigues(poses[i].rotation)[0].ravel(), poses[i].translation.ravel()] for i in (2, 3)] + list(points.values()))
    fun, jac = bundle_adjustment_callbacks(k, observations, poses, camera_offsets, point_offsets, len(values))
    np.testing.assert_allclose(jac(values).toarray(), approx_derivative(fun, values, method="3-point"), rtol=1e-6, atol=2e-6)


def test_ba_backends_preserve_fixed_cameras_and_recover_same_solution():
    k, tracks, poses, points = make_bundle_problem()
    # Unobserved point must survive unchanged, as in the existing implementation.
    tracks.append(Track({}))
    points[len(tracks)-1] = np.array([1., 2., 9.])
    python = bundle_adjust(k, tracks, poses, points, {0, 1}, max_nfev=60)
    cpp = bundle_adjust(k, tracks, poses, points, {0, 1}, max_nfev=60, backend="cpp")
    for i in (0, 1):
        np.testing.assert_array_equal(cpp[0][i].matrix(), poses[i].matrix())
    for i in poses:
        np.testing.assert_allclose(cpp[0][i].matrix(), python[0][i].matrix(), atol=2e-5)
    for i in points:
        np.testing.assert_allclose(cpp[1][i], python[1][i], atol=2e-4)


def make_graph(seed=9):
    rng = np.random.default_rng(seed)
    true = [Pose(cv2.Rodrigues(rng.normal(0, .3, 3))[0], rng.normal(size=3)).matrix() for _ in range(5)]
    # Native graph poses are camera-to-world; numerical fixture uses arbitrary SE(3).
    nodes = [PoseGraphNode(7*i, float(i), transform.copy()) for i, transform in enumerate(true)]
    for i in range(1, len(nodes)):
        nodes[i].camera_to_world[:3, 3] += rng.normal(0, .03, 3)
        nodes[i].camera_to_world[:3, :3] = cv2.Rodrigues(rng.normal(0,.02,3))[0] @ nodes[i].camera_to_world[:3, :3]
    edges = [PoseGraphEdge(7*p, 7*q, np.linalg.inv(true[q]) @ true[p], .7, 1.3, 50, "odometry") for p, q in [(0,1),(1,2),(2,3),(3,4),(0,4)]]
    return PoseGraph(nodes, edges), true


def test_graph_native_jacobian_matches_independent_finite_differences():
    graph, _ = make_graph()
    offsets = {node.index: 6*i for i, node in enumerate(graph.nodes[1:])}
    values = np.concatenate([_vec6_from_matrix(node.camera_to_world) for node in graph.nodes[1:]])
    fun, jac = pose_graph_callbacks(graph, offsets, len(values))
    np.testing.assert_allclose(jac(values).toarray(), approx_derivative(fun, values, method="3-point"), rtol=2e-6, atol=2e-6)


@pytest.mark.parametrize("angle", [0., 1e-7, .8, np.pi-1e-4])
def test_graph_rotation_residuals_match_principal_angle_axis(angle):
    from sfm_reconstruction.pose_graph import _matrix_from_vec6
    first = np.eye(4)
    values = np.array([angle, 0., 0., .3, -.2, .1])
    second = _matrix_from_vec6(values)
    graph = PoseGraph(
        [PoseGraphNode(0, 0., first), PoseGraphNode(1, 1., second)],
        [PoseGraphEdge(0, 1, np.eye(4), .7, 1.3, 30, "loop")],
    )
    fun, _ = pose_graph_callbacks(graph, {1: 0}, 6)
    expected = _vec6_from_matrix(np.linalg.inv(second))
    expected[:3] *= 1.3
    expected[3:] *= .7
    # OpenCV zeroes very small angle-axis logs; compare within that precision.
    np.testing.assert_allclose(fun(values), expected, atol=2e-7)


def test_graph_backends_recover_geometry_and_fix_first_nonzero_index():
    graph, true = make_graph()
    for backend in ("python", "cpp"):
        result = optimize_pose_graph(graph, PoseGraphConfig(optimizer_backend=backend))
        assert result.success
        np.testing.assert_allclose(_transform_from_trajectory_pose(result.poses[0]), graph.nodes[0].camera_to_world, atol=1e-12)
        for pose, expected in zip(result.poses, true):
            np.testing.assert_allclose(_transform_from_trajectory_pose(pose), expected, atol=3e-5)


@pytest.mark.parametrize("backend", ["python", "cpp"])
def test_stable_graph_log_recovers_rotation_below_opencv_dead_zone(backend):
    from scipy.spatial.transform import Rotation
    transform = np.eye(4)
    transform[:3, :3] = Rotation.from_rotvec([1e-7, -2e-7, 3e-7]).as_matrix()
    graph = PoseGraph([PoseGraphNode(0, 0., np.eye(4)), PoseGraphNode(9, 1., transform)],
        [PoseGraphEdge(9, 0, np.eye(4), 1., 1., 30, "loop")])
    # Check the initial cost too: zeroing the initial angle would make this test vacuous.
    result = optimize_pose_graph(graph, PoseGraphConfig(optimizer_backend=backend, python_rotation_log="stable"))
    recovered = _transform_from_trajectory_pose(result.poses[1])
    assert result.initial_cost > 1e-14
    assert np.linalg.norm(Rotation.from_matrix(recovered[:3,:3]).as_rotvec()) < 1e-9
    np.testing.assert_allclose(_transform_from_trajectory_pose(result.poses[0]), np.eye(4), atol=1e-12)


def test_cached_callbacks_do_not_expose_arrays_mutated_by_scipy_robust_loss():
    calls = []
    def evaluate(values):
        calls.append(values.copy())
        return np.array([[2.]]), np.array([[[3.]]])
    fun, jac = sparse_callbacks(evaluate, np.array([[[0]]]), 1)
    values = np.array([1.])
    fun(values)[:] = 100
    jac(values).data[:] = 200
    np.testing.assert_array_equal(fun(values), [2.])
    np.testing.assert_array_equal(jac(values).toarray(), [[3.]])
    assert len(calls) == 1


@pytest.mark.parametrize("count", [0, 1, 1000])
def test_geometry_backends_match_on_strided_input_and_full_intrinsics(count):
    rng = np.random.default_rng(5)
    storage = rng.normal(size=(2*count, 3))
    storage[:, 2] += 5
    points = storage[::2]
    k = np.array([[730., 11., 320.], [0., 690., 240.], [0., 0., 1.]])
    first = Pose(cv2.Rodrigues(np.array([.1, -.2, .3]))[0], [.1, -.2, .05])
    second = Pose(np.eye(3), [-1., 0., 0.])
    observations = project_points(points, first, k) + .3
    for kernel, arguments in (
        (project_points, (points, first, k)),
        (reprojection_errors, (points, observations, first, k)),
        (points_in_front, (points, first)),
        (triangulation_angles, (points, first, second)),
    ):
        np.testing.assert_allclose(kernel(*arguments, backend="cpp"), kernel(*arguments), rtol=1e-10, atol=1e-10)


def test_geometry_degenerate_inputs_preserve_nan_and_depth_decisions():
    first = Pose.identity()
    second = Pose(np.eye(3), [-1., 0., 0.])
    points = np.array([[0., 0., 0.], [1., 0., 0.], [0., 0., -1.], [0., 0., 1.]])
    with np.errstate(divide="ignore", invalid="ignore"):
        np.testing.assert_allclose(triangulation_angles(points, first, second, backend="cpp"), triangulation_angles(points, first, second), equal_nan=True)
        np.testing.assert_allclose(project_points(points, first, np.eye(3), backend="cpp"), project_points(points, first, np.eye(3)), equal_nan=True)
    np.testing.assert_array_equal(points_in_front(points, first, backend="cpp"), points_in_front(points, first))


def test_geometry_backend_preserves_reconstruction_filter_decisions():
    from sfm_reconstruction.reconstruction import ReconstructionConfig, _filter_points
    k, tracks, poses, points = make_bundle_problem()
    config = ReconstructionConfig(min_track_observations=3)
    python = _filter_points(k, tracks, poses, points, config)
    from dataclasses import replace
    cpp = _filter_points(k, tracks, poses, points, replace(config, geometry_backend="cpp"))
    assert set(cpp) == set(python)


def test_reconstruction_accepts_combined_cpp_backends(tmp_path):
    from sfm_reconstruction.dataset import Stage1Dataset
    from sfm_reconstruction.reconstruction import ReconstructionConfig, reconstruct
    k, tracks, _, _ = make_bundle_problem(4, 80)
    paths = {}
    for pair in ((0, 1), (1, 2), (2, 3)):
        path = tmp_path / f"{pair[0]}_{pair[1]}.txt"
        np.savetxt(path, [np.r_[t.observations[pair[0]], t.observations[pair[1]]] for t in tracks])
        paths[pair] = path
    dataset = Stage1Dataset(
        tmp_path, k, {i: tmp_path/f"{i}.jpg" for i in range(4)},
        {i: f"{i}.jpg" for i in range(4)}, paths, {},
    )
    results = []
    for backend in ("python", "cpp"):
        cv2.setRNGSeed(0)
        results.append(reconstruct(dataset, ReconstructionConfig(
            bundle_adjustment_backend=backend, geometry_backend=backend,
            bundle_adjustment_max_nfev=40,
        )))
    python, cpp = results
    assert set(cpp.poses) == set(python.poses) == {0, 1, 2, 3}
    assert set(cpp.points) == set(python.points)
    for i in cpp.poses:
        np.testing.assert_allclose(cpp.poses[i].matrix(), python.poses[i].matrix(), atol=1e-3)
