import json

import numpy as np

from sfm_reconstruction.dataset import load_stage1_dataset
from sfm_reconstruction.geometry import project_points
from sfm_reconstruction.models import Pose, Track
from sfm_reconstruction.reconstruction import (
    ReconstructionConfig,
    _filter_points,
    reconstruct,
)


def test_incremental_reconstruction_registers_synthetic_sequence(tmp_path):
    rng = np.random.default_rng(11)
    intrinsics = np.array(
        [[700.0, 0.0, 320.0], [0.0, 700.0, 240.0], [0.0, 0.0, 1.0]]
    )
    points = np.column_stack(
        (
            rng.uniform(-1.2, 1.2, 70),
            rng.uniform(-0.8, 0.8, 70),
            rng.uniform(4.0, 8.0, 70),
        )
    )
    poses = {
        image_id: Pose(np.eye(3), [-0.45 * image_id, 0.0, 0.0])
        for image_id in range(4)
    }

    images = tmp_path / "images"
    correspondences = tmp_path / "correspondences"
    images.mkdir()
    correspondences.mkdir()
    projections = {}
    for image_id, pose in poses.items():
        (images / f"{image_id:05d}.jpg").touch()
        projections[image_id] = project_points(points, pose, intrinsics)
    for first_id, second_id in ((0, 1), (1, 2), (2, 3)):
        np.savetxt(
            correspondences / f"{first_id}_{second_id}.txt",
            np.hstack((projections[first_id], projections[second_id])),
        )
    (tmp_path / "camera_parameters.json").write_text(
        json.dumps(
            {
                "intrinsics": intrinsics.tolist(),
                "extrinsics": {
                    f"{image_id:05d}.jpg": pose.matrix().tolist()
                    for image_id, pose in poses.items()
                },
            }
        )
    )

    result = reconstruct(
        load_stage1_dataset(tmp_path),
        ReconstructionConfig(
            bundle_adjustment=False,
            min_pnp_points=8,
            min_pnp_inliers=8,
            max_reprojection_error=1.0,
            pnp_reprojection_error=1.0,
            min_triangulation_angle=0.25,
        ),
    )

    assert set(result.poses) == {0, 1, 2, 3}
    assert len(result.points) == len(points)


def test_two_view_filter_rejects_low_angle_points() -> None:
    intrinsics = np.eye(3)
    poses = {
        0: Pose.identity(),
        1: Pose(np.eye(3), np.array([-1.0, 0.0, 0.0])),
    }
    points = {
        0: np.array([0.0, 0.0, 50.0]),
        1: np.array([0.0, 0.0, 5.0]),
    }
    tracks = [
        Track(
            observations={
                image_id: project_points(point.reshape(1, 3), pose, intrinsics)[0]
                for image_id, pose in poses.items()
            }
        )
        for point in points.values()
    ]

    filtered = _filter_points(
        intrinsics,
        tracks,
        poses,
        points,
        ReconstructionConfig(
            min_track_observations=2,
            two_view_min_triangulation_angle=5.0,
        ),
    )

    assert set(filtered) == {1}


def test_two_view_filter_rejects_high_residual_points() -> None:
    intrinsics = np.eye(3)
    poses = {
        0: Pose.identity(),
        1: Pose(np.eye(3), np.array([-1.0, 0.0, 0.0])),
    }
    point = np.array([0.0, 0.0, 5.0])
    observations = {
        image_id: project_points(point.reshape(1, 3), pose, intrinsics)[0]
        for image_id, pose in poses.items()
    }
    observations[1] = observations[1] + np.array([1.0, 0.0])

    filtered = _filter_points(
        intrinsics,
        [Track(observations=observations)],
        poses,
        {0: point},
        ReconstructionConfig(
            min_track_observations=2,
            max_reprojection_error=5.0,
            two_view_max_reprojection_error=0.5,
        ),
    )

    assert filtered == {}
