"""Loading and packing for optional C++ kernels and Ceres bundle adjustment."""
from __future__ import annotations

import importlib
import os
from functools import lru_cache
from pathlib import Path

import numpy as np
from scipy.sparse import coo_matrix


@lru_cache(maxsize=1)
def load_native():
    try:
        return importlib.import_module("sfm_reconstruction._native")
    except ImportError as exc:
        raise RuntimeError(
            "C++ backend unavailable. Build native/CMakeLists.txt and install the "
            "extension into sfm_reconstruction (see docs/CPP_INTEGRATION.md)."
        ) from exc


_dll_handles = []


@lru_cache(maxsize=1)
def load_ceres():
    if os.name == "nt":
        directory = Path(os.environ.get("SFM_CERES_DLL_DIR", str(
            Path(__file__).resolve().parents[2] / "outputs/ceres_deps/Library/bin"
        )))
        if directory.is_dir():
            _dll_handles.append(os.add_dll_directory(str(directory)))
    try:
        return importlib.import_module("sfm_reconstruction._ceres")
    except ImportError as exc:
        raise RuntimeError("Ceres backend unavailable. Build with -CeresPrefix; "
                           "see docs/CPP_INTEGRATION.md. Set SFM_CERES_DLL_DIR "
                           "if the dependency DLLs are outside outputs/ceres_deps/Library/bin.") from exc


def validate_backend(backend: str, *, allow_ceres: bool = False) -> None:
    allowed = {"python", "cpp", "ceres"} if allow_ceres else {"python", "cpp"}
    if backend not in allowed:
        raise ValueError(f"backend must be one of {sorted(allowed)}")
    if backend == "cpp":
        load_native()
    if backend == "ceres":
        load_ceres()


def ceres_bundle_adjust(intrinsics, observations, poses, points, camera_offsets,
                       point_offsets, initial, max_iterations, loss_scale,
                       num_threads=1, linear_solver="dense_schur"):
    import cv2

    camera_ids = sorted(poses)
    point_ids = sorted(point_offsets)
    camera_index = {c: i for i, c in enumerate(camera_ids)}
    point_index = {p: i for i, p in enumerate(point_ids)}
    cameras = np.asarray([
        np.r_[cv2.Rodrigues(poses[c].rotation)[0].ravel(), poses[c].translation.ravel()]
        for c in camera_ids
    ])
    xyz = np.asarray([points[p] for p in point_ids]).reshape(-1, 3)
    indices = np.asarray([(camera_index[c], point_index[p]) for c, p, _ in observations], dtype=np.int64)
    pixels = np.asarray([xy for _, _, xy in observations], dtype=np.float64)
    fixed = np.asarray([camera_index[c] for c in camera_ids if c not in camera_offsets], dtype=np.int64)
    cameras, xyz, report = load_ceres().bundle_adjust(
        cameras, xyz, indices, pixels, intrinsics, fixed, max_iterations, loss_scale,
        num_threads, linear_solver
    )
    values = initial.copy()
    for c, offset in camera_offsets.items():
        values[offset:offset+6] = cameras[camera_index[c]]
    for p, offset in point_offsets.items():
        values[offset:offset+3] = xyz[point_index[p]]
    return values, dict(report)


def sparse_callbacks(evaluate, columns: np.ndarray, parameter_count: int):
    """Cache one native evaluation shared by SciPy's fun/jac callbacks."""
    rows = np.broadcast_to(
        np.arange(columns.shape[0] * columns.shape[1]).reshape(
            columns.shape[0], columns.shape[1], 1
        ), columns.shape,
    )
    mask = columns >= 0
    rows, cols = rows[mask], columns[mask]
    cached_values = cached_residuals = cached_jacobian = None

    def update(values):
        nonlocal cached_values, cached_residuals, cached_jacobian
        if cached_values is None or not np.array_equal(values, cached_values):
            residuals, derivatives = evaluate(np.ascontiguousarray(values))
            cached_values = values.copy()
            cached_residuals = residuals.ravel()
            cached_jacobian = coo_matrix(
                (derivatives[mask], (rows, cols)),
                shape=(residuals.size, parameter_count),
            ).tocsr()

    def fun(values):
        update(values)
        # SciPy's robust-loss scaling mutates residual/Jacobian arrays in place.
        return cached_residuals.copy()

    def jac(values):
        update(values)
        return cached_jacobian.copy()

    return fun, jac


def bundle_adjustment_callbacks(intrinsics, observations, poses, camera_offsets,
                                point_offsets, parameter_count):
    import cv2

    native = load_native()
    camera_ids = sorted(poses)
    camera_index = {camera_id: i for i, camera_id in enumerate(camera_ids)}
    cameras = np.asarray([
        np.r_[cv2.Rodrigues(poses[i].rotation)[0].ravel(), poses[i].translation.ravel()]
        for i in camera_ids
    ])
    offsets = np.asarray([camera_offsets.get(i, -1) for i in camera_ids], dtype=np.int64)
    indices = np.asarray([
        (camera_index[c], point_offsets[p]) for c, p, _ in observations
    ], dtype=np.int64)
    pixels = np.asarray([xy for _, _, xy in observations], dtype=np.float64)
    columns = np.empty((len(observations), 2, 9), dtype=np.int64)
    for row, (c, p, _) in enumerate(observations):
        columns[row, :, :6] = (
            np.arange(6) + camera_offsets[c] if c in camera_offsets else -1
        )
        columns[row, :, 6:] = np.arange(3) + point_offsets[p]
    return sparse_callbacks(
        lambda values: native.ba_evaluate(
            values, cameras, offsets, indices, pixels, intrinsics, True
        ), columns, parameter_count,
    )


def pose_graph_callbacks(graph, offset_of, parameter_count):
    from .pose_graph import _vec6_from_matrix

    native = load_native()
    node_index = {node.index: i for i, node in enumerate(graph.nodes)}
    initial = np.asarray([_vec6_from_matrix(node.camera_to_world, rotation_log="stable") for node in graph.nodes])
    offsets = np.asarray([offset_of.get(node.index, -1) for node in graph.nodes], dtype=np.int64)
    edges = np.asarray([
        (node_index[e.source_index], node_index[e.target_index]) for e in graph.edges
    ], dtype=np.int64)
    measured = np.asarray([e.measured_target_from_source for e in graph.edges])
    weights = np.asarray([(e.rotation_weight, e.translation_weight) for e in graph.edges])
    columns = np.empty((len(graph.edges), 6, 12), dtype=np.int64)
    for row, edge in enumerate(graph.edges):
        for j, endpoint in enumerate((edge.source_index, edge.target_index)):
            columns[row, :, 6*j:6*j+6] = (
                np.arange(6) + offset_of[endpoint] if endpoint in offset_of else -1
            )
    return sparse_callbacks(
        lambda values: native.graph_evaluate(
            values, initial, offsets, edges, measured, weights, True
        ), columns, parameter_count,
    )
