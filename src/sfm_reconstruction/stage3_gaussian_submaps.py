from __future__ import annotations

import argparse
import json
import shutil
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .gaussian_splatting_export import _colmap_qvec
from .stage3_gaussian_seed_rebalance import (
    _even_sample,
    _load_ascii_ply,
    focus_visibility_counts,
)
from .stage3_splat_temporal_diagnostics import _read_colmap


@dataclass(frozen=True)
class SubmapSpec:
    name: str
    start: float
    end: float | None


def parse_submap_spec(value: str) -> SubmapSpec:
    fields = value.split(":")
    if len(fields) != 3 or not fields[0]:
        raise ValueError("submap must use NAME:START:END")
    start = float(fields[1])
    end = None if fields[2].lower() in {"end", "none"} else float(fields[2])
    if start < 0.0 or (end is not None and end <= start):
        raise ValueError("submap times must satisfy 0 <= START < END")
    return SubmapSpec(fields[0], start, end)


def select_local_seed_indices(
    visibility_counts: np.ndarray,
    *,
    max_points: int,
    min_views: int,
) -> np.ndarray:
    if visibility_counts.ndim != 1:
        raise ValueError("visibility_counts must be one-dimensional")
    if max_points < 1 or min_views < 1:
        raise ValueError("max_points and min_views must be positive")
    candidates = np.flatnonzero(visibility_counts >= min_views)
    return _even_sample(candidates, min(max_points, len(candidates)))


def _write_pose_subset(
    source_export: Path,
    output_dir: Path,
    name: str,
    intrinsics: tuple[float, float, float, float, int, int],
    selected_poses: list[tuple[str, np.ndarray, np.ndarray]],
    points: np.ndarray,
    colors: np.ndarray,
    *,
    max_points: int,
    min_views: int,
    selection: dict[str, object],
) -> dict[str, object]:
    if len(selected_poses) < 2:
        raise ValueError(f"submap {name!r} contains fewer than two cameras")

    pose_values = [(rotation, translation) for _, rotation, translation in selected_poses]
    visibility = focus_visibility_counts(points, pose_values, intrinsics)
    selected_indices = select_local_seed_indices(
        visibility, max_points=max_points, min_views=min_views
    )
    if len(selected_indices) == 0:
        raise ValueError(f"submap {name!r} has no multi-view seed points")

    images_dir = output_dir / "images"
    sparse_dir = output_dir / "sparse" / "0"
    images_dir.mkdir(parents=True)
    sparse_dir.mkdir(parents=True)
    for image_name, _, _ in selected_poses:
        shutil.copy2(
            source_export / "images" / image_name, images_dir / image_name
        )
    shutil.copy2(source_export / "sparse" / "0" / "cameras.txt", sparse_dir / "cameras.txt")

    image_lines = [
        "# IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME",
        "# POINTS2D[] as (X, Y, POINT3D_ID)",
    ]
    for image_id, (image_name, rotation, translation) in enumerate(
        selected_poses, start=1
    ):
        qvec = _colmap_qvec(rotation)
        pose = " ".join(f"{value:.17g}" for value in (*qvec, *translation))
        image_lines.extend((f"{image_id} {pose} 1 {image_name}", ""))
    (sparse_dir / "images.txt").write_text(
        "\n".join(image_lines) + "\n", encoding="ascii"
    )

    point_lines = [
        "# POINT3D_ID, X, Y, Z, R, G, B, ERROR, TRACK[] as (IMAGE_ID, POINT2D_IDX)"
    ]
    for point_id, source_index in enumerate(selected_indices, start=1):
        xyz = " ".join(f"{value:.17g}" for value in points[source_index])
        red, green, blue = colors[source_index]
        point_lines.append(f"{point_id} {xyz} {red} {green} {blue} 0")
    (sparse_dir / "points3D.txt").write_text(
        "\n".join(point_lines) + "\n", encoding="ascii"
    )

    selected_visibility = visibility[selected_indices]
    summary = {
        "name": name,
        **selection,
        "cameras": len(selected_poses),
        "timestamp_first": float(Path(selected_poses[0][0]).stem),
        "timestamp_last": float(Path(selected_poses[-1][0]).stem),
        "multi_view_candidates": int(np.count_nonzero(visibility >= min_views)),
        "selected_points": len(selected_indices),
        "min_views": min_views,
        "mean_selected_views": float(np.mean(selected_visibility)),
        "uses_ground_truth": False,
    }
    (output_dir / "gaussian_submap.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    return summary


def _write_submap(
    source_export: Path,
    output_dir: Path,
    spec: SubmapSpec,
    intrinsics: tuple[float, float, float, float, int, int],
    pose_records: list[tuple[str, np.ndarray, np.ndarray]],
    points: np.ndarray,
    colors: np.ndarray,
    *,
    max_points: int,
    min_views: int,
) -> dict[str, object]:
    selected_poses = [
        record
        for record in pose_records
        if float(Path(record[0]).stem) >= spec.start
        and (spec.end is None or float(Path(record[0]).stem) < spec.end)
    ]
    return _write_pose_subset(
        source_export,
        output_dir,
        spec.name,
        intrinsics,
        selected_poses,
        points,
        colors,
        max_points=max_points,
        min_views=min_views,
        selection={"start": spec.start, "end": spec.end},
    )


def export_stage3_gaussian_submaps(
    source_export: Path,
    source_cloud: Path,
    output_root: Path,
    specs: list[SubmapSpec],
    *,
    max_points: int = 200_000,
    min_views: int = 2,
) -> dict[str, object]:
    if not specs:
        raise ValueError("at least one submap is required")
    if len({spec.name for spec in specs}) != len(specs):
        raise ValueError("submap names must be unique")
    if output_root.exists():
        raise FileExistsError(f"output directory already exists: {output_root}")

    intrinsics, pose_records, _ = _read_colmap(source_export / "sparse" / "0")
    points, colors = _load_ascii_ply(source_cloud)
    output_root.mkdir(parents=True)
    summaries = [
        _write_submap(
            source_export,
            output_root / spec.name,
            spec,
            intrinsics,
            pose_records,
            points,
            colors,
            max_points=max_points,
            min_views=min_views,
        )
        for spec in specs
    ]
    summary = {
        "source_export": str(source_export.resolve()),
        "source_cloud": str(source_cloud.resolve()),
        "source_points": len(points),
        "max_points_per_submap": max_points,
        "min_views": min_views,
        "uses_ground_truth": False,
        "submaps": summaries,
    }
    (output_root / "gaussian_submaps.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Export overlapping local Stage 3 Gaussian Splatting submaps."
    )
    parser.add_argument("--source-export", required=True, type=Path)
    parser.add_argument("--source-cloud", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument(
        "--submap",
        action="append",
        required=True,
        help="NAME:START:END, where END may be 'end'",
    )
    parser.add_argument("--max-points", type=int, default=200_000)
    parser.add_argument("--min-views", type=int, default=2)
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    summary = export_stage3_gaussian_submaps(
        args.source_export,
        args.source_cloud,
        args.output_root,
        [parse_submap_spec(value) for value in args.submap],
        max_points=args.max_points,
        min_views=args.min_views,
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
