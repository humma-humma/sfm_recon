from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from .gaussian_splatting_export import _colmap_qvec
from .stage3 import (
    Stage3Frame,
    TrajectoryPose,
    _transform_from_trajectory_pose,
    load_stage3_dataset,
    load_trajectory,
)


def _select_keyframe_indices(frame_count: int, frame_stride: int) -> list[int]:
    if frame_stride < 1:
        raise ValueError("frame_stride must be at least 1")
    if frame_count < 1:
        raise ValueError("frame_count must be at least 1")
    indices = list(range(0, frame_count, frame_stride))
    if indices[-1] != frame_count - 1:
        indices.append(frame_count - 1)
    return indices


def _load_ascii_seed_points(
    path: Path, max_points: int
) -> list[tuple[np.ndarray, tuple[int, int, int]]]:
    if max_points < 1:
        raise ValueError("max_points must be at least 1")

    with path.open("r", encoding="ascii") as handle:
        header: list[str] = []
        for raw_line in handle:
            line = raw_line.strip()
            header.append(line)
            if line == "end_header":
                break
        else:
            raise ValueError(f"Invalid PLY header: {path}")

        if "format ascii 1.0" not in header:
            raise ValueError(f"Seed cloud must be an ASCII PLY: {path}")
        vertex_lines = [line for line in header if line.startswith("element vertex ")]
        if len(vertex_lines) != 1:
            raise ValueError(f"PLY must declare one vertex element: {path}")
        vertex_count = int(vertex_lines[0].split()[-1])
        properties = [
            line.split()[-1] for line in header if line.startswith("property ")
        ]
        required = ("x", "y", "z", "red", "green", "blue")
        if any(name not in properties for name in required):
            raise ValueError(f"PLY must contain properties {required}")
        indices = {name: properties.index(name) for name in required}
        sample_stride = max(1, int(np.ceil(vertex_count / max_points)))

        points: list[tuple[np.ndarray, tuple[int, int, int]]] = []
        for vertex_index, raw_line in enumerate(handle):
            if vertex_index >= vertex_count or len(points) >= max_points:
                break
            if vertex_index % sample_stride != 0:
                continue
            values = raw_line.split()
            xyz = np.asarray(
                [float(values[indices[name]]) for name in required[:3]],
                dtype=np.float64,
            )
            rgb = tuple(int(values[indices[name]]) for name in required[3:])
            if np.isfinite(xyz).all():
                points.append((xyz, rgb))
    if not points:
        raise ValueError(f"No finite seed points found in {path}")
    return points


def _frame_lookup(frames: list[Stage3Frame]) -> dict[str, Stage3Frame]:
    return {f"{frame.timestamp:.9f}": frame for frame in frames}


def export_stage3_colmap_dataset(
    dataset_root: str | Path,
    trajectory_path: str | Path,
    point_cloud_path: str | Path,
    output_dir: str | Path,
    *,
    frame_stride: int = 16,
    image_scale: float = 0.5,
    max_points: int = 200_000,
) -> dict[str, object]:
    if not 0.0 < image_scale <= 1.0:
        raise ValueError("image_scale must be in (0, 1]")

    dataset = load_stage3_dataset(dataset_root)
    trajectory_path = Path(trajectory_path).resolve()
    point_cloud_path = Path(point_cloud_path).resolve()
    output_dir = Path(output_dir).resolve()
    trajectory = load_trajectory(trajectory_path)
    selected_indices = _select_keyframe_indices(len(trajectory), frame_stride)
    frame_by_timestamp = _frame_lookup(dataset.frames)

    selected: list[tuple[TrajectoryPose, Stage3Frame]] = []
    for index in selected_indices:
        pose = trajectory[index]
        frame = frame_by_timestamp.get(f"{pose.timestamp:.9f}")
        if frame is None:
            raise ValueError(
                f"Trajectory timestamp {pose.timestamp:.9f} has no matching RGB frame"
            )
        selected.append((pose, frame))

    first_image = cv2.imread(str(selected[0][1].rgb_path), cv2.IMREAD_COLOR)
    if first_image is None:
        raise ValueError(f"Could not read {selected[0][1].rgb_path}")
    source_height, source_width = first_image.shape[:2]
    width = int(round(source_width * image_scale))
    height = int(round(source_height * image_scale))
    scaled_intrinsics = dataset.intrinsics.copy()
    scaled_intrinsics[0, :] *= width / source_width
    scaled_intrinsics[1, :] *= height / source_height

    images_dir = output_dir / "images"
    sparse_dir = output_dir / "sparse" / "0"
    images_dir.mkdir(parents=True, exist_ok=True)
    sparse_dir.mkdir(parents=True, exist_ok=True)

    image_names: list[str] = []
    for _, frame in selected:
        image = cv2.imread(str(frame.rgb_path), cv2.IMREAD_COLOR)
        if image is None or image.shape[:2] != (source_height, source_width):
            raise ValueError(f"Image has an unexpected size or is unreadable: {frame.rgb_path}")
        if image_scale != 1.0:
            image = cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)
        name = f"{frame.rgb_path.stem}.jpg"
        destination = images_dir / name
        if not cv2.imwrite(str(destination), image, [cv2.IMWRITE_JPEG_QUALITY, 95]):
            raise OSError(f"Could not write {destination}")
        image_names.append(name)

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
    for colmap_id, ((pose, _), name) in enumerate(
        zip(selected, image_names), start=1
    ):
        camera_to_world = _transform_from_trajectory_pose(pose)
        rotation = camera_to_world[:3, :3].T
        translation = -rotation @ camera_to_world[:3, 3]
        qvec = _colmap_qvec(rotation)
        pose_values = " ".join(f"{value:.17g}" for value in (*qvec, *translation))
        image_lines.extend((f"{colmap_id} {pose_values} 1 {name}", ""))
    (sparse_dir / "images.txt").write_text(
        "\n".join(image_lines) + "\n", encoding="ascii"
    )

    points = _load_ascii_seed_points(point_cloud_path, max_points)
    point_lines = [
        "# POINT3D_ID, X, Y, Z, R, G, B, ERROR, TRACK[] as (IMAGE_ID, POINT2D_IDX)"
    ]
    for point_id, (xyz, rgb) in enumerate(points, start=1):
        values = " ".join(f"{value:.17g}" for value in xyz)
        point_lines.append(
            f"{point_id} {values} {rgb[0]} {rgb[1]} {rgb[2]} 0"
        )
    (sparse_dir / "points3D.txt").write_text(
        "\n".join(point_lines) + "\n", encoding="ascii"
    )

    summary = {
        "format": "COLMAP text",
        "source_dataset": str(dataset.root),
        "source_trajectory": str(trajectory_path),
        "source_point_cloud": str(point_cloud_path),
        "trajectory_poses": len(trajectory),
        "images": len(selected),
        "points": len(points),
        "frame_stride": frame_stride,
        "includes_final_frame": selected_indices[-1] == len(trajectory) - 1,
        "source_resolution": [source_width, source_height],
        "export_resolution": [width, height],
        "image_scale": image_scale,
        "pose_input_convention": "camera_to_world",
        "pose_export_convention": "world_to_camera",
        "initialization": point_cloud_path.name,
        "uses_ground_truth": False,
    }
    (output_dir / "gaussian_splatting_export.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Export Stage 3 SLAM keyframes as a COLMAP Gaussian Splatting dataset."
    )
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--trajectory", type=Path, required=True)
    parser.add_argument("--point-cloud", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--frame-stride", type=int, default=16)
    parser.add_argument("--image-scale", type=float, default=0.5)
    parser.add_argument("--max-points", type=int, default=200_000)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    summary = export_stage3_colmap_dataset(
        args.dataset,
        args.trajectory,
        args.point_cloud,
        args.output_dir,
        frame_stride=args.frame_stride,
        image_scale=args.image_scale,
        max_points=args.max_points,
    )
    print(
        f"Exported {summary['images']} Stage 3 keyframes and "
        f"{summary['points']} points to {args.output_dir.resolve()}"
    )


if __name__ == "__main__":
    main()
