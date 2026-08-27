import json
from pathlib import Path

import cv2
import numpy as np

from sfm_reconstruction.gaussian_splatting_export import export_colmap_dataset


def _write_fixture(tmp_path: Path) -> tuple[Path, Path]:
    dataset = tmp_path / "dataset"
    images = dataset / "images"
    images.mkdir(parents=True)
    cv2.imwrite(str(images / "00000.jpg"), np.zeros((40, 60, 3), dtype=np.uint8))
    cv2.imwrite(str(images / "00001.jpg"), np.zeros((40, 60, 3), dtype=np.uint8))
    (dataset / "camera_parameters.json").write_text(
        json.dumps({"intrinsics": [[50, 0, 30], [0, 50, 20], [0, 0, 1]]}),
        encoding="utf-8",
    )

    result = tmp_path / "result"
    result.mkdir()
    second_pose = np.eye(4)
    second_pose[0, 3] = -1.0
    (result / "estimated_camera_parameters.json").write_text(
        json.dumps(
            {
                "intrinsics": [[50, 0, 30], [0, 50, 20], [0, 0, 1]],
                "extrinsics": {
                    "00000.jpg": np.eye(4).tolist(),
                    "00001.jpg": second_pose.tolist(),
                },
            }
        ),
        encoding="utf-8",
    )
    (result / "estimated_points_rich.ply").write_text(
        "\n".join(
            (
                "ply",
                "format ascii 1.0",
                "element vertex 1",
                "property float x",
                "property float y",
                "property float z",
                "property uchar red",
                "property uchar green",
                "property uchar blue",
                "property float mean_reprojection_error",
                "end_header",
                "1 2 3 10 20 30 0.25",
            )
        ),
        encoding="ascii",
    )
    return dataset, result


def test_export_colmap_dataset_scales_images_intrinsics_and_writes_poses(tmp_path) -> None:
    dataset, result = _write_fixture(tmp_path)
    output = tmp_path / "export"

    summary = export_colmap_dataset(dataset, result, output, image_scale=0.5)

    assert summary["images"] == 2
    assert summary["points"] == 1
    assert summary["export_resolution"] == [30, 20]
    assert cv2.imread(str(output / "images" / "00000.jpg")).shape[:2] == (20, 30)
    cameras = (output / "sparse" / "0" / "cameras.txt").read_text()
    assert "1 PINHOLE 30 20 25 25 15 10" in cameras
    images = (output / "sparse" / "0" / "images.txt").read_text()
    assert "1 1 0 0 0 0 0 0 1 00000.jpg" in images
    assert "2 1 0 0 0 -1 0 0 1 00001.jpg" in images
    points = (output / "sparse" / "0" / "points3D.txt").read_text()
    assert "1 1 2 3 10 20 30 0.25" in points


def test_export_colmap_dataset_rejects_invalid_scale(tmp_path) -> None:
    dataset, result = _write_fixture(tmp_path)

    try:
        export_colmap_dataset(dataset, result, tmp_path / "export", image_scale=0.0)
    except ValueError as error:
        assert "image_scale" in str(error)
    else:
        raise AssertionError("Expected invalid image scale to fail")
