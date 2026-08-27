from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil

import cv2
import numpy as np
from scipy.spatial.transform import Rotation

from .dataset import load_image_dataset


def _load_camera_parameters(path: Path) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    intrinsics = np.asarray(data["intrinsics"], dtype=np.float64)
    if intrinsics.shape != (3, 3):
        raise ValueError("Camera intrinsics must have shape (3, 3)")
    extrinsics = {
        name: np.asarray(matrix, dtype=np.float64)
        for name, matrix in data["extrinsics"].items()
    }
    if any(matrix.shape != (4, 4) for matrix in extrinsics.values()):
        raise ValueError("Camera extrinsics must have shape (4, 4)")
    return intrinsics, extrinsics


def _load_rich_points(path: Path) -> list[tuple[np.ndarray, tuple[int, int, int], float]]:
    lines = path.read_text(encoding="ascii").splitlines()
    try:
        header_end = lines.index("end_header")
    except ValueError as error:
        raise ValueError(f"Invalid PLY header: {path}") from error

    properties = [
        line.split()[-1]
        for line in lines[:header_end]
        if line.startswith("property ")
    ]
    required = ("x", "y", "z", "red", "green", "blue")
    if any(name not in properties for name in required):
        raise ValueError(f"PLY must contain properties {required}")
    indices = {name: properties.index(name) for name in properties}
    error_index = indices.get("mean_reprojection_error")

    points = []
    for line in lines[header_end + 1 :]:
        if not line.strip():
            continue
        values = line.split()
        xyz = np.asarray([float(values[indices[name]]) for name in required[:3]])
        rgb = tuple(int(values[indices[name]]) for name in required[3:])
        error = float(values[error_index]) if error_index is not None else 0.0
        points.append((xyz, rgb, error))
    return points


def _colmap_qvec(rotation: np.ndarray) -> np.ndarray:
    quaternion_xyzw = Rotation.from_matrix(rotation).as_quat()
    quaternion_wxyz = quaternion_xyzw[[3, 0, 1, 2]]
    if quaternion_wxyz[0] < 0.0:
        quaternion_wxyz *= -1.0
    return quaternion_wxyz


def export_colmap_dataset(
    dataset_root: str | Path,
    result_dir: str | Path,
    output_dir: str | Path,
    image_scale: float = 1.0,
) -> dict[str, object]:
    if not 0.0 < image_scale <= 1.0:
        raise ValueError("image_scale must be in (0, 1]")

    dataset = load_image_dataset(dataset_root)
    result_dir = Path(result_dir).resolve()
    output_dir = Path(output_dir).resolve()
    intrinsics, extrinsics = _load_camera_parameters(
        result_dir / "estimated_camera_parameters.json"
    )
    points = _load_rich_points(result_dir / "estimated_points_rich.ply")

    registered = [
        image_id
        for image_id in dataset.image_ids
        if dataset.image_names[image_id] in extrinsics
    ]
    if not registered:
        raise ValueError("No dataset images have estimated camera poses")

    first_image = cv2.imread(str(dataset.image_paths[registered[0]]), cv2.IMREAD_COLOR)
    if first_image is None:
        raise ValueError(f"Could not read {dataset.image_paths[registered[0]]}")
    source_height, source_width = first_image.shape[:2]
    width = int(round(source_width * image_scale))
    height = int(round(source_height * image_scale))
    scaled_intrinsics = intrinsics.copy()
    scaled_intrinsics[0, :] *= width / source_width
    scaled_intrinsics[1, :] *= height / source_height

    images_dir = output_dir / "images"
    sparse_dir = output_dir / "sparse" / "0"
    images_dir.mkdir(parents=True, exist_ok=True)
    sparse_dir.mkdir(parents=True, exist_ok=True)

    for image_id in registered:
        source = dataset.image_paths[image_id]
        destination = images_dir / dataset.image_names[image_id]
        if image_scale == 1.0:
            shutil.copy2(source, destination)
            continue
        image = cv2.imread(str(source), cv2.IMREAD_COLOR)
        if image is None or image.shape[:2] != (source_height, source_width):
            raise ValueError(f"Image has an unexpected size or is unreadable: {source}")
        resized = cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)
        if not cv2.imwrite(str(destination), resized, [cv2.IMWRITE_JPEG_QUALITY, 95]):
            raise OSError(f"Could not write {destination}")

    camera_line = (
        f"1 PINHOLE {width} {height} "
        f"{scaled_intrinsics[0, 0]:.17g} {scaled_intrinsics[1, 1]:.17g} "
        f"{scaled_intrinsics[0, 2]:.17g} {scaled_intrinsics[1, 2]:.17g}"
    )
    (sparse_dir / "cameras.txt").write_text(
        "# CAMERA_ID, MODEL, WIDTH, HEIGHT, PARAMS[]\n" + camera_line + "\n",
        encoding="ascii",
    )

    image_lines = [
        "# IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME",
        "# POINTS2D[] as (X, Y, POINT3D_ID)",
    ]
    for colmap_id, image_id in enumerate(registered, start=1):
        name = dataset.image_names[image_id]
        world_to_camera = extrinsics[name]
        qvec = _colmap_qvec(world_to_camera[:3, :3])
        translation = world_to_camera[:3, 3]
        pose_values = " ".join(f"{value:.17g}" for value in (*qvec, *translation))
        image_lines.extend((f"{colmap_id} {pose_values} 1 {name}", ""))
    (sparse_dir / "images.txt").write_text(
        "\n".join(image_lines) + "\n", encoding="ascii"
    )

    point_lines = [
        "# POINT3D_ID, X, Y, Z, R, G, B, ERROR, TRACK[] as (IMAGE_ID, POINT2D_IDX)"
    ]
    for point_id, (xyz, rgb, error) in enumerate(points, start=1):
        values = " ".join(f"{value:.17g}" for value in xyz)
        point_lines.append(
            f"{point_id} {values} {rgb[0]} {rgb[1]} {rgb[2]} {error:.17g}"
        )
    (sparse_dir / "points3D.txt").write_text(
        "\n".join(point_lines) + "\n", encoding="ascii"
    )

    summary = {
        "format": "COLMAP text",
        "source_dataset": str(dataset.root),
        "source_result": str(result_dir),
        "images": len(registered),
        "points": len(points),
        "source_resolution": [source_width, source_height],
        "export_resolution": [width, height],
        "image_scale": image_scale,
        "pose_convention": "world_to_camera",
        "initialization": "estimated_points_rich.ply",
    }
    (output_dir / "gaussian_splatting_export.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Export an SfM result as a COLMAP dataset for Gaussian Splatting."
    )
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--result-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--image-scale", type=float, default=1.0)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    summary = export_colmap_dataset(
        args.dataset, args.result_dir, args.output_dir, args.image_scale
    )
    print(
        f"Exported {summary['images']} cameras and {summary['points']} points "
        f"to {args.output_dir.resolve()}"
    )


if __name__ == "__main__":
    main()
