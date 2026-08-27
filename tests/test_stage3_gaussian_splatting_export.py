import json
from pathlib import Path

import cv2
import numpy as np

from sfm_reconstruction.stage3_gaussian_splatting_export import (
    _select_keyframe_indices,
    export_stage3_colmap_dataset,
)


def _write_fixture(tmp_path: Path) -> tuple[Path, Path, Path]:
    dataset = tmp_path / "stage3"
    rgb_dir = dataset / "rgb"
    depth_dir = dataset / "depth"
    rgb_dir.mkdir(parents=True)
    depth_dir.mkdir()
    timestamps = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]
    for index, timestamp in enumerate(timestamps):
        name = f"{timestamp:.8f}.png"
        image = np.full((40, 60, 3), index, dtype=np.uint8)
        cv2.imwrite(str(rgb_dir / name), image)
        cv2.imwrite(str(depth_dir / name), np.full((40, 60), 1000, dtype=np.uint16))
    (dataset / "camera_parameters.json").write_text(
        json.dumps({"intrinsics": [[50, 0, 30], [0, 50, 20], [0, 0, 1]]}),
        encoding="utf-8",
    )

    trajectory = tmp_path / "trajectory.txt"
    trajectory.write_text(
        "\n".join(
            f"{timestamp} {index} 0 0 0 0 0 1"
            for index, timestamp in enumerate(timestamps)
        )
        + "\n",
        encoding="utf-8",
    )
    cloud = tmp_path / "scene.ply"
    cloud.write_text(
        "\n".join(
            (
                "ply",
                "format ascii 1.0",
                "element vertex 6",
                "property float x",
                "property float y",
                "property float z",
                "property uchar red",
                "property uchar green",
                "property uchar blue",
                "end_header",
                "0 0 1 10 20 30",
                "1 0 1 11 21 31",
                "2 0 1 12 22 32",
                "3 0 1 13 23 33",
                "4 0 1 14 24 34",
                "5 0 1 15 25 35",
            )
        ),
        encoding="ascii",
    )
    return dataset, trajectory, cloud


def test_select_keyframes_preserves_final_frame() -> None:
    assert _select_keyframe_indices(6, 2) == [0, 2, 4, 5]
    assert _select_keyframe_indices(5, 2) == [0, 2, 4]


def test_export_stage3_colmap_dataset_inverts_poses_and_samples_seed(
    tmp_path: Path,
) -> None:
    dataset, trajectory, cloud = _write_fixture(tmp_path)
    output = tmp_path / "export"

    summary = export_stage3_colmap_dataset(
        dataset,
        trajectory,
        cloud,
        output,
        frame_stride=2,
        image_scale=0.5,
        max_points=3,
    )

    assert summary["images"] == 4
    assert summary["points"] == 3
    assert summary["includes_final_frame"] is True
    assert summary["uses_ground_truth"] is False
    assert summary["export_resolution"] == [30, 20]
    assert len(list((output / "images").glob("*.jpg"))) == 4
    cameras = (output / "sparse" / "0" / "cameras.txt").read_text()
    assert "1 PINHOLE 30 20 25 25 15 10" in cameras
    images = (output / "sparse" / "0" / "images.txt").read_text()
    assert "1 1 0 0 0 0 0 0 1 0.00000000.jpg" in images
    assert "2 1 0 0 0 -2 0 0 1 0.20000000.jpg" in images
    assert "4 1 0 0 0 -5 0 0 1 0.50000000.jpg" in images
    points = (output / "sparse" / "0" / "points3D.txt").read_text()
    assert "1 0 0 1 10 20 30 0" in points
    assert "2 2 0 1 12 22 32 0" in points
    assert "3 4 0 1 14 24 34 0" in points


def test_export_stage3_colmap_dataset_rejects_invalid_limits(tmp_path: Path) -> None:
    dataset, trajectory, cloud = _write_fixture(tmp_path)
    for kwargs, message in (
        ({"frame_stride": 0}, "frame_stride"),
        ({"image_scale": 0.0}, "image_scale"),
        ({"max_points": 0}, "max_points"),
    ):
        try:
            export_stage3_colmap_dataset(
                dataset,
                trajectory,
                cloud,
                tmp_path / message,
                **kwargs,
            )
        except ValueError as error:
            assert message in str(error)
        else:
            raise AssertionError(f"Expected invalid {message} to fail")
