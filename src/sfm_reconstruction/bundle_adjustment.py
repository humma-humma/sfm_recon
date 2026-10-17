from __future__ import annotations

import cv2
import numpy as np
from scipy.optimize import least_squares
from scipy.sparse import lil_matrix

from .models import Pose, Track


def bundle_adjust(
    intrinsics: np.ndarray,
    tracks: list[Track],
    poses: dict[int, Pose],
    points: dict[int, np.ndarray],
    fixed_camera_ids: set[int],
    max_nfev: int = 40,
    max_reprojection_error: float = 5.0,
    backend: str = "python",
    ceres_max_iterations: int = 100,
    solver_report: dict | None = None,
    loss_scale: float = 2.0,
    ceres_num_threads: int = 1,
    ceres_linear_solver: str = "dense_schur",
) -> tuple[dict[int, Pose], dict[int, np.ndarray]]:
    from .native_backend import validate_backend

    validate_backend(backend, allow_ceres=True)
    if ceres_num_threads < 1 or not isinstance(ceres_num_threads, int):
        raise ValueError("ceres_num_threads must be a positive integer")
    if ceres_linear_solver not in {"dense_schur", "sparse_schur", "iterative_schur"}:
        raise ValueError("Unknown Ceres linear solver")
    if not np.isfinite(loss_scale) or loss_scale <= 0:
        raise ValueError("loss_scale must be finite and positive")
    variable_camera_ids = sorted(set(poses) - fixed_camera_ids)
    camera_offsets = {
        camera_id: index * 6 for index, camera_id in enumerate(variable_camera_ids)
    }
    observations_by_track: dict[int, list[tuple[int, int, np.ndarray]]] = {}
    for track_id, point in points.items():
        accepted = []
        for camera_id, observed in tracks[track_id].observations.items():
            if camera_id not in poses:
                continue
            pose = poses[camera_id]
            point_camera = pose.rotation @ point.reshape(3, 1) + pose.translation
            if point_camera[2, 0] <= 0:
                continue
            projected_h = intrinsics @ point_camera
            projected = projected_h[:2, 0] / projected_h[2, 0]
            if np.linalg.norm(projected - observed) <= max_reprojection_error:
                accepted.append((camera_id, track_id, observed))
        if len(accepted) >= 2:
            observations_by_track[track_id] = accepted

    point_ids = sorted(observations_by_track)
    point_offsets = {
        track_id: 6 * len(variable_camera_ids) + index * 3
        for index, track_id in enumerate(point_ids)
    }

    observations = [
        observation
        for track_id in point_ids
        for observation in observations_by_track[track_id]
    ]
    if not observations or not point_ids:
        return poses, points

    parameters: list[float] = []
    for camera_id in variable_camera_ids:
        rotation_vector, _ = cv2.Rodrigues(poses[camera_id].rotation)
        parameters.extend(rotation_vector.ravel())
        parameters.extend(poses[camera_id].translation.ravel())
    for track_id in point_ids:
        parameters.extend(points[track_id].ravel())
    initial = np.asarray(parameters, dtype=np.float64)

    sparsity = lil_matrix((2 * len(observations), len(initial)), dtype=np.int8)
    for observation_index, (camera_id, track_id, _) in enumerate(observations):
        rows = (2 * observation_index, 2 * observation_index + 1)
        point_offset = point_offsets[track_id]
        sparsity[rows, point_offset : point_offset + 3] = 1
        if camera_id in camera_offsets:
            camera_offset = camera_offsets[camera_id]
            sparsity[rows, camera_offset : camera_offset + 6] = 1

    def unpack(values: np.ndarray) -> tuple[dict[int, Pose], dict[int, np.ndarray]]:
        current_poses = {
            camera_id: poses[camera_id]
            for camera_id in fixed_camera_ids
            if camera_id in poses
        }
        for camera_id, offset in camera_offsets.items():
            rotation, _ = cv2.Rodrigues(values[offset : offset + 3])
            current_poses[camera_id] = Pose(rotation, values[offset + 3 : offset + 6])
        current_points = {
            track_id: values[offset : offset + 3]
            for track_id, offset in point_offsets.items()
        }
        return current_poses, current_points

    def residuals(values: np.ndarray) -> np.ndarray:
        current_poses, current_points = unpack(values)
        errors = np.empty((len(observations), 2), dtype=np.float64)
        for index, (camera_id, track_id, observed) in enumerate(observations):
            pose = current_poses[camera_id]
            point_camera = (
                pose.rotation @ current_points[track_id].reshape(3, 1)
                + pose.translation
            )
            projected_h = intrinsics @ point_camera
            projected = projected_h[:2, 0] / projected_h[2, 0]
            errors[index] = projected - observed
        return errors.ravel()

    if backend == "ceres":
        from .native_backend import ceres_bundle_adjust

        values, report = ceres_bundle_adjust(
            intrinsics, observations, poses, points, camera_offsets,
            point_offsets, initial, ceres_max_iterations, loss_scale,
            ceres_num_threads, ceres_linear_solver,
        )
        if solver_report is not None:
            solver_report.update(report)
        optimized_poses, optimized_points = unpack(values)
        return optimized_poses, {**points, **optimized_points}

    solver_kwargs = {"jac_sparsity": sparsity}
    if backend == "cpp":
        from .native_backend import bundle_adjustment_callbacks

        residuals, jacobian = bundle_adjustment_callbacks(
            intrinsics, observations, poses, camera_offsets, point_offsets, len(initial)
        )
        solver_kwargs = {"jac": jacobian}
    result = least_squares(
        residuals,
        initial,
        **solver_kwargs,
        method="trf",
        loss="soft_l1",
        f_scale=loss_scale,
        x_scale="jac",
        max_nfev=max_nfev,
        verbose=0,
    )
    optimized_poses, optimized_points = unpack(result.x)
    if solver_report is not None:
        solver_report.update(final_cost=float(result.cost), nfev=result.nfev,
                             converged=bool(result.success), message=result.message)
    merged_points = dict(points)
    merged_points.update(optimized_points)
    return optimized_poses, merged_points
