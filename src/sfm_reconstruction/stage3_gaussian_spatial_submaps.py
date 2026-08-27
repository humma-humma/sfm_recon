from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .stage3_gaussian_seed_rebalance import _load_ascii_ply
from .stage3_gaussian_submaps import _write_pose_subset
from .stage3_splat_temporal_diagnostics import _read_colmap


def deterministic_spatial_labels(
    camera_centers: np.ndarray,
    cluster_count: int,
    *,
    max_iterations: int = 50,
) -> np.ndarray:
    if camera_centers.ndim != 2 or camera_centers.shape[1] != 3:
        raise ValueError("camera_centers must have shape (cameras, 3)")
    if not 1 <= cluster_count <= len(camera_centers):
        raise ValueError("cluster_count must be between one and the camera count")

    centers = [camera_centers[0]]
    for _ in range(1, cluster_count):
        distances = np.linalg.norm(
            camera_centers[:, None, :] - np.asarray(centers)[None, :, :], axis=2
        )
        centers.append(camera_centers[np.argmax(np.min(distances, axis=1))])
    centers_array = np.asarray(centers)
    labels = np.zeros(len(camera_centers), dtype=np.int64)
    for _ in range(max_iterations):
        distances = np.linalg.norm(
            camera_centers[:, None, :] - centers_array[None, :, :], axis=2
        )
        new_labels = np.argmin(distances, axis=1)
        if np.array_equal(labels, new_labels):
            break
        labels = new_labels
        centers_array = np.asarray(
            [camera_centers[labels == index].mean(axis=0) for index in range(cluster_count)]
        )

    ordered = sorted(range(cluster_count), key=lambda index: int(np.flatnonzero(labels == index)[0]))
    remap = np.empty(cluster_count, dtype=np.int64)
    remap[ordered] = np.arange(cluster_count)
    return remap[labels]


def expanded_cluster_indices(
    labels: np.ndarray, cluster_index: int, overlap_cameras: int
) -> np.ndarray:
    if labels.ndim != 1 or overlap_cameras < 0:
        raise ValueError("labels must be one-dimensional and overlap_cameras non-negative")
    core = np.flatnonzero(labels == cluster_index)
    if len(core) == 0:
        raise ValueError("cluster has no cameras")
    selected = np.zeros(len(labels), dtype=bool)
    for camera_index in core:
        start = max(0, int(camera_index) - overlap_cameras)
        end = min(len(labels), int(camera_index) + overlap_cameras + 1)
        selected[start:end] = True
    return np.flatnonzero(selected)


def export_stage3_spatial_submaps(
    source_export: Path,
    source_cloud: Path,
    output_root: Path,
    *,
    cluster_count: int = 3,
    overlap_cameras: int = 8,
    max_points: int = 200_000,
    min_views: int = 2,
) -> dict[str, object]:
    if output_root.exists():
        raise FileExistsError(f"output directory already exists: {output_root}")
    intrinsics, pose_records, _ = _read_colmap(source_export / "sparse" / "0")
    camera_centers = np.asarray(
        [-rotation.T @ translation for _, rotation, translation in pose_records]
    )
    labels = deterministic_spatial_labels(camera_centers, cluster_count)
    points, colors = _load_ascii_ply(source_cloud)
    output_root.mkdir(parents=True)

    summaries = []
    for cluster_index in range(cluster_count):
        indices = expanded_cluster_indices(labels, cluster_index, overlap_cameras)
        core_count = int(np.count_nonzero(labels == cluster_index))
        name = f"spatial_{cluster_index}"
        summaries.append(
            _write_pose_subset(
                source_export,
                output_root / name,
                name,
                intrinsics,
                [pose_records[int(index)] for index in indices],
                points,
                colors,
                max_points=max_points,
                min_views=min_views,
                selection={
                    "cluster_index": cluster_index,
                    "core_cameras": core_count,
                    "overlap_cameras": overlap_cameras,
                },
            )
        )

    summary = {
        "source_export": str(source_export.resolve()),
        "source_cloud": str(source_cloud.resolve()),
        "source_points": len(points),
        "cluster_count": cluster_count,
        "overlap_cameras": overlap_cameras,
        "max_points_per_submap": max_points,
        "min_views": min_views,
        "uses_ground_truth": False,
        "submaps": summaries,
    }
    (output_root / "gaussian_spatial_submaps.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Export overlapping spatial Stage 3 Gaussian Splatting submaps."
    )
    parser.add_argument("--source-export", required=True, type=Path)
    parser.add_argument("--source-cloud", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--cluster-count", type=int, default=3)
    parser.add_argument("--overlap-cameras", type=int, default=8)
    parser.add_argument("--max-points", type=int, default=200_000)
    parser.add_argument("--min-views", type=int, default=2)
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    summary = export_stage3_spatial_submaps(
        args.source_export,
        args.source_cloud,
        args.output_root,
        cluster_count=args.cluster_count,
        overlap_cameras=args.overlap_cameras,
        max_points=args.max_points,
        min_views=args.min_views,
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
