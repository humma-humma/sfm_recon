import cv2
import numpy as np

from sfm_reconstruction.stage3_splat_temporal_diagnostics import (
    image_metrics,
    pose_increments,
    projected_point_support,
)


def test_image_metrics_identical_is_perfect():
    image = np.full((20, 20, 3), 128, dtype=np.uint8)

    psnr, ssim = image_metrics(image, image.copy())

    assert np.isinf(psnr)
    assert ssim == 1.0


def test_image_metrics_detect_degradation():
    image = np.zeros((20, 20, 3), dtype=np.uint8)
    degraded = image.copy()
    degraded[5:15, 5:15] = 255

    psnr, ssim = image_metrics(image, degraded)

    assert psnr < 10.0
    assert ssim < 1.0


def test_pose_increments_use_camera_centers_and_rotation_angle():
    rotations = [np.eye(3), cv2.Rodrigues(np.asarray([0.0, 0.0, np.pi / 2.0]))[0]]
    centers = [np.zeros(3), np.asarray([1.0, 0.0, 0.0])]
    translations = [-rotation @ center for rotation, center in zip(rotations, centers)]

    translation_steps, rotation_steps = pose_increments(rotations, translations)

    np.testing.assert_allclose(translation_steps, [0.0, 1.0])
    np.testing.assert_allclose(rotation_steps, [0.0, 90.0])


def test_projected_point_support_counts_in_frame_points():
    points = np.asarray([[0.0, 0.0, 2.0], [0.5, 0.0, 2.0], [5.0, 0.0, 1.0], [0.0, 0.0, -1.0]])

    count, median_depth = projected_point_support(
        points,
        np.eye(3),
        np.zeros(3),
        (10.0, 10.0, 5.0, 5.0, 10, 10),
    )

    assert count == 2
    assert median_depth == 2.0
