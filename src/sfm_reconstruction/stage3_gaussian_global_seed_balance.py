from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import numpy as np

from .stage3_gaussian_seed_rebalance import (
    _even_sample,
    _load_ascii_ply,
    _visible_mask,
)
from .stage3_splat_temporal_diagnostics import _read_colmap


def region_visibility_counts(
    points: np.ndarray,
    poses: list[tuple[np.ndarray, np.ndarray]],
    intrinsics: tuple[float, float, float, float, int, int],
    region_count: int,
) -> tuple[np.ndarray, list[np.ndarray]]:
    if region_count < 1 or region_count > len(poses):
        raise ValueError("region_count must be between one and the camera count")
    regions = [indices for indices in np.array_split(np.arange(len(poses)), region_count)]
    counts = np.zeros((region_count, len(points)), dtype=np.uint16)
    for region_index, camera_indices in enumerate(regions):
        for camera_index in camera_indices:
            rotation, translation = poses[int(camera_index)]
            counts[region_index] += _visible_mask(
                points, rotation, translation, intrinsics
            )
    return counts, regions


def select_region_balanced_indices(
    counts: np.ndarray,
    *,
    total_points: int,
    min_region_views: int,
    baseline_indices: np.ndarray | None = None,
) -> tuple[np.ndarray, list[int]]:
    if counts.ndim != 2 or counts.shape[0] < 1:
        raise ValueError("counts must have shape (regions, points)")
    if total_points < 1 or total_points > counts.shape[1]:
        raise ValueError("total_points must be within the source point count")
    if min_region_views < 1:
        raise ValueError("min_region_views must be positive")

    selected_mask = np.zeros(counts.shape[1], dtype=bool)
    selected_parts: list[np.ndarray] = []
    if baseline_indices is not None:
        baseline_indices = np.asarray(baseline_indices, dtype=np.int64)
        if (
            len(np.unique(baseline_indices)) != len(baseline_indices)
            or np.any(baseline_indices < 0)
            or np.any(baseline_indices >= counts.shape[1])
            or len(baseline_indices) > total_points
        ):
            raise ValueError("baseline_indices must be unique valid point indices")
        selected_mask[baseline_indices] = True
        selected_parts.append(baseline_indices)

    region_count = counts.shape[0]
    regional_budget = total_points - int(np.count_nonzero(selected_mask))
    budgets = np.full(region_count, regional_budget // region_count, dtype=np.int64)
    budgets[: regional_budget % region_count] += 1
    selected_per_region: list[int] = []
    for region_index, budget in enumerate(budgets):
        candidates = np.flatnonzero(
            (counts[region_index] >= min_region_views) & ~selected_mask
        )
        chosen = _even_sample(candidates, min(int(budget), len(candidates)))
        selected_mask[chosen] = True
        selected_parts.append(chosen)
        selected_per_region.append(len(chosen))

    selected_count = int(np.count_nonzero(selected_mask))
    remaining = total_points - selected_count
    if remaining:
        global_counts = counts.sum(axis=0, dtype=np.uint32)
        candidates = np.flatnonzero(
            (global_counts >= min_region_views) & ~selected_mask
        )
        chosen = _even_sample(candidates, min(remaining, len(candidates)))
        selected_mask[chosen] = True
        selected_parts.append(chosen)
        remaining -= len(chosen)
    if remaining:
        candidates = np.flatnonzero(~selected_mask)
        selected_parts.append(_even_sample(candidates, remaining))
    return np.sort(np.concatenate(selected_parts)), selected_per_region


def _camera_support(
    points: np.ndarray,
    poses: list[tuple[np.ndarray, np.ndarray]],
    intrinsics: tuple[float, float, float, float, int, int],
) -> np.ndarray:
    return np.asarray(
        [
            np.count_nonzero(_visible_mask(points, rotation, translation, intrinsics))
            for rotation, translation in poses
        ],
        dtype=np.int64,
    )


def _support_summary(values: np.ndarray) -> dict[str, float | int]:
    return {
        "minimum": int(np.min(values)),
        "p10": float(np.percentile(values, 10)),
        "median": float(np.median(values)),
        "mean": float(np.mean(values)),
        "maximum": int(np.max(values)),
    }


def balance_colmap_seed_cloud_globally(
    source_export: Path,
    source_cloud: Path,
    output_dir: Path,
    *,
    total_points: int = 200_000,
    baseline_points: int = 100_000,
    region_count: int = 12,
    min_region_views: int = 2,
) -> dict[str, object]:
    if output_dir.exists():
        raise FileExistsError(f"output directory already exists: {output_dir}")

    colmap_dir = source_export / "sparse" / "0"
    intrinsics, pose_records, original_points = _read_colmap(colmap_dir)
    poses = [(rotation, translation) for _, rotation, translation in pose_records]
    points, colors = _load_ascii_ply(source_cloud)
    counts, regions = region_visibility_counts(
        points, poses, intrinsics, region_count
    )
    if not 0 <= baseline_points <= total_points:
        raise ValueError("baseline_points must be between zero and total_points")
    baseline_stride = max(1, int(np.ceil(len(points) / max(1, baseline_points))))
    baseline_indices = np.arange(0, len(points), baseline_stride, dtype=np.int64)[
        :baseline_points
    ]
    selected, selected_per_region = select_region_balanced_indices(
        counts,
        total_points=total_points,
        min_region_views=min_region_views,
        baseline_indices=baseline_indices,
    )

    shutil.copytree(source_export / "images", output_dir / "images")
    output_sparse = output_dir / "sparse" / "0"
    output_sparse.mkdir(parents=True)
    shutil.copy2(colmap_dir / "cameras.txt", output_sparse / "cameras.txt")
    shutil.copy2(colmap_dir / "images.txt", output_sparse / "images.txt")
    point_lines = [
        "# POINT3D_ID, X, Y, Z, R, G, B, ERROR, TRACK[] as (IMAGE_ID, POINT2D_IDX)"
    ]
    for point_id, source_index in enumerate(selected, start=1):
        xyz = " ".join(f"{value:.17g}" for value in points[source_index])
        red, green, blue = colors[source_index]
        point_lines.append(f"{point_id} {xyz} {red} {green} {blue} 0")
    (output_sparse / "points3D.txt").write_text(
        "\n".join(point_lines) + "\n", encoding="ascii"
    )

    selected_points = points[selected]
    original_support = _camera_support(original_points, poses, intrinsics)
    balanced_support = _camera_support(selected_points, poses, intrinsics)
    region_rows = []
    for region_index, camera_indices in enumerate(regions):
        first = int(camera_indices[0])
        last = int(camera_indices[-1])
        region_rows.append(
            {
                "region": region_index,
                "camera_start": first,
                "camera_end": last,
                "timestamp_start": float(Path(pose_records[first][0]).stem),
                "timestamp_end": float(Path(pose_records[last][0]).stem),
                "candidate_points": int(
                    np.count_nonzero(counts[region_index] >= min_region_views)
                ),
                "selected_primary_points": selected_per_region[region_index],
                "original_mean_support": float(np.mean(original_support[camera_indices])),
                "balanced_mean_support": float(np.mean(balanced_support[camera_indices])),
            }
        )

    summary = {
        "source_export": str(source_export.resolve()),
        "source_cloud": str(source_cloud.resolve()),
        "uses_ground_truth": False,
        "source_points": len(points),
        "selected_points": len(selected),
        "baseline_points": len(baseline_indices),
        "region_count": region_count,
        "min_region_views": min_region_views,
        "global_multi_view_candidates": int(
            np.count_nonzero(counts.sum(axis=0, dtype=np.uint32) >= min_region_views)
        ),
        "original_camera_support": _support_summary(original_support),
        "balanced_camera_support": _support_summary(balanced_support),
        "regions": region_rows,
    }
    (output_dir / "gaussian_global_seed_balance.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Balance a Stage 3 COLMAP seed cloud across trajectory regions."
    )
    parser.add_argument("--source-export", required=True, type=Path)
    parser.add_argument("--source-cloud", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--total-points", type=int, default=200_000)
    parser.add_argument("--baseline-points", type=int, default=100_000)
    parser.add_argument("--region-count", type=int, default=12)
    parser.add_argument("--min-region-views", type=int, default=2)
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    summary = balance_colmap_seed_cloud_globally(
        args.source_export,
        args.source_cloud,
        args.output_dir,
        total_points=args.total_points,
        baseline_points=args.baseline_points,
        region_count=args.region_count,
        min_region_views=args.min_region_views,
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
