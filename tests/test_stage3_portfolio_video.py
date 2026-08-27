import numpy as np

from sfm_reconstruction.stage3_portfolio_video import (
    look_at_camera_to_world,
    orbit_camera_path,
    trajectory_camera_path,
    transform_points,
)
from sfm_reconstruction.stage3 import TrajectoryPose


def test_transform_points_applies_rotation_translation_and_scale():
    transform = np.array(
        [[0.0, -1.0, 0.0, 2.0], [1.0, 0.0, 0.0, -1.0], [0.0, 0.0, 1.0, 0.5]]
    )
    actual = transform_points(np.array([[3.0, 4.0, 5.0]]), transform, 0.5)
    np.testing.assert_allclose(actual, [[-1.0, 1.0, 2.75]])


def test_look_at_uses_nerfstudio_backward_axis():
    camera = look_at_camera_to_world(
        np.array([0.0, 0.0, 2.0]), np.zeros(3), np.array([0.0, 1.0, 0.0])
    )
    np.testing.assert_allclose(camera[:3, 2], [0.0, 0.0, 1.0])
    np.testing.assert_allclose(camera[:3, 3], [0.0, 0.0, 2.0])
    np.testing.assert_allclose(camera[:3, :3].T @ camera[:3, :3], np.eye(3), atol=1e-12)


def test_orbit_camera_path_keeps_constant_target_distance():
    points = np.array([[-1.0, -1.0, -1.0], [1.0, 1.0, 1.0]])
    cameras = orbit_camera_path(points, [0.0, 1.0, 0.0], [0.0, 0.0, 1.0], 12)
    distances = [np.linalg.norm(camera[:3, 3]) for camera in cameras]
    assert len(cameras) == 12
    np.testing.assert_allclose(distances, distances[0], rtol=1e-12)


def test_trajectory_camera_path_samples_final_pose_and_flips_camera_axes():
    poses = [
        TrajectoryPose(float(index), [index, 0.0, 0.0], [0.0, 0.0, 0.0, 1.0])
        for index in range(5)
    ]
    transform = np.column_stack((np.eye(3), np.zeros(3)))
    cameras = trajectory_camera_path(poses, transform, 0.5, frame_stride=3)
    assert len(cameras) == 3
    np.testing.assert_allclose([camera[0, 3] for camera in cameras], [0.0, 1.5, 2.0])
    np.testing.assert_allclose(cameras[0][:3, :3], np.diag([1.0, -1.0, -1.0]))
