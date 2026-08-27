from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import numpy as np

from .stage3_splat_temporal_diagnostics import _read_colmap


def _load_ascii_ply(path: Path) -> tuple[np.ndarray, np.ndarray]:
    header_lines = 0
    vertex_count = None
    with path.open("r", encoding="ascii") as file:
        for raw_line in file:
            header_lines += 1
            line = raw_line.strip()
            if line.startswith("element vertex "):
                vertex_count = int(line.split()[-1])
            if line == "end_header":
                break
        else:
            raise ValueError(f"invalid PLY header: {path}")
    if vertex_count is None:
        raise ValueError(f"PLY has no vertex count: {path}")
    values = np.loadtxt(
        path,
        dtype=np.float64,
        skiprows=header_lines,
        max_rows=vertex_count,
        usecols=(0, 1, 2, 3, 4, 5),
    )
    if values.shape != (vertex_count, 6):
        raise ValueError(f"expected {vertex_count} xyzrgb vertices in {path}")
    finite = np.isfinite(values[:, :3]).all(axis=1)
    return values[finite, :3], np.clip(values[finite, 3:6], 0, 255).astype(np.uint8)


def _visible_mask(
    points: np.ndarray,
    rotation: np.ndarray,
    translation: np.ndarray,
    intrinsics: tuple[float, float, float, float, int, int],
) -> np.ndarray:
    fx, fy, cx, cy, width, height = intrinsics
    camera_points = points @ rotation.T + translation
    depth = camera_points[:, 2]
    positive = depth > 1e-6
    visible = np.zeros(len(points), dtype=bool)
    x = fx * camera_points[positive, 0] / depth[positive] + cx
    y = fy * camera_points[positive, 1] / depth[positive] + cy
    visible[positive] = (x >= 0.0) & (x < width) & (y >= 0.0) & (y < height)
    return visible


def focus_visibility_counts(
    points: np.ndarray,
    focus_poses: list[tuple[np.ndarray, np.ndarray]],
    intrinsics: tuple[float, float, float, float, int, int],
) -> np.ndarray:
    counts = np.zeros(len(points), dtype=np.uint16)
    for rotation, translation in focus_poses:
        counts += _visible_mask(points, rotation, translation, intrinsics)
    return counts


def _even_sample(indices: np.ndarray, count: int) -> np.ndarray:
    if count >= len(indices):
        return indices.copy()
    positions = np.linspace(0, len(indices) - 1, num=count, dtype=np.int64)
    return indices[positions]


def select_rebalanced_indices(
    visibility_counts: np.ndarray,
    *,
    total_points: int,
    focus_points: int,
    min_focus_views: int,
) -> np.ndarray:
    if total_points < 1:
        raise ValueError("total_points must be positive")
    if not 0 <= focus_points <= total_points:
        raise ValueError("focus_points must be between zero and total_points")
    if min_focus_views < 1:
        raise ValueError("min_focus_views must be positive")
    if total_points > len(visibility_counts):
        raise ValueError("total_points exceeds the source point count")

    focus_candidates = np.flatnonzero(visibility_counts >= min_focus_views)
    selected_focus = _even_sample(focus_candidates, min(focus_points, len(focus_candidates)))
    selected = np.zeros(len(visibility_counts), dtype=bool)
    selected[selected_focus] = True
    global_candidates = np.flatnonzero(~selected)
    selected_global = _even_sample(global_candidates, total_points - len(selected_focus))
    return np.sort(np.concatenate((selected_focus, selected_global)))


def rebalance_colmap_seed_cloud(
    source_export: Path,
    source_cloud: Path,
    output_dir: Path,
    *,
    focus_start: float,
    focus_end: float,
    total_points: int = 200_000,
    focus_points: int = 80_000,
    min_focus_views: int = 2,
) -> dict[str, object]:
    if focus_end <= focus_start:
        raise ValueError("focus_end must be greater than focus_start")
    if output_dir.exists():
        raise FileExistsError(f"output directory already exists: {output_dir}")

    colmap_dir = source_export / "sparse" / "0"
    intrinsics, poses, _ = _read_colmap(colmap_dir)
    focus_pose_records = [
        (rotation, translation)
        for name, rotation, translation in poses
        if focus_start <= float(Path(name).stem) < focus_end
    ]
    if not focus_pose_records:
        raise ValueError("no COLMAP cameras fall inside the focus interval")

    points, colors = _load_ascii_ply(source_cloud)
    visibility_counts = focus_visibility_counts(points, focus_pose_records, intrinsics)
    selected = select_rebalanced_indices(
        visibility_counts,
        total_points=total_points,
        focus_points=focus_points,
        min_focus_views=min_focus_views,
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

    selected_counts = visibility_counts[selected]
    focus_candidates = int(np.count_nonzero(visibility_counts >= min_focus_views))
    summary = {
        "source_export": str(source_export.resolve()),
        "source_cloud": str(source_cloud.resolve()),
        "uses_ground_truth": False,
        "source_points": len(points),
        "selected_points": len(selected),
        "focus_start": focus_start,
        "focus_end": focus_end,
        "focus_cameras": len(focus_pose_records),
        "min_focus_views": min_focus_views,
        "focus_candidates": focus_candidates,
        "selected_focus_supported": int(np.count_nonzero(selected_counts >= min_focus_views)),
        "selected_visible_once": int(np.count_nonzero(selected_counts >= 1)),
        "mean_focus_views_per_selected_point": float(np.mean(selected_counts)),
    }
    (output_dir / "gaussian_seed_rebalance.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Rebalance a Stage 3 COLMAP seed cloud toward a weak time interval."
    )
    parser.add_argument("--source-export", required=True, type=Path)
    parser.add_argument("--source-cloud", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--focus-start", required=True, type=float)
    parser.add_argument("--focus-end", required=True, type=float)
    parser.add_argument("--total-points", type=int, default=200_000)
    parser.add_argument("--focus-points", type=int, default=80_000)
    parser.add_argument("--min-focus-views", type=int, default=2)
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    summary = rebalance_colmap_seed_cloud(
        args.source_export,
        args.source_cloud,
        args.output_dir,
        focus_start=args.focus_start,
        focus_end=args.focus_end,
        total_points=args.total_points,
        focus_points=args.focus_points,
        min_focus_views=args.min_focus_views,
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
