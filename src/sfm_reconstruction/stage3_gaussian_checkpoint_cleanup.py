from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
from scipy.spatial import cKDTree

from .stage3_gaussian_seed_rebalance import (
    _load_ascii_ply,
    focus_visibility_counts,
)
from .stage3_splat_temporal_diagnostics import _read_colmap


def cleanup_mask(
    opacity: np.ndarray,
    max_scale: np.ndarray,
    seed_distance: np.ndarray,
    camera_support: np.ndarray,
    *,
    min_opacity: float,
    max_gaussian_scale: float,
    max_seed_distance: float,
    min_camera_support: int,
) -> np.ndarray:
    arrays = (opacity, max_scale, seed_distance, camera_support)
    if any(array.ndim != 1 for array in arrays) or len({len(array) for array in arrays}) != 1:
        raise ValueError("cleanup metrics must be equally sized one-dimensional arrays")
    if not 0.0 <= min_opacity <= 1.0:
        raise ValueError("min_opacity must be between zero and one")
    if max_gaussian_scale <= 0.0 or max_seed_distance <= 0.0:
        raise ValueError("scale and distance limits must be positive")
    if min_camera_support < 1:
        raise ValueError("min_camera_support must be positive")
    return (
        (opacity >= min_opacity)
        & (max_scale <= max_gaussian_scale)
        & (seed_distance <= max_seed_distance)
        & (camera_support >= min_camera_support)
    )


def _prune_optimizer_value(value: Any, mask: Any, gaussian_count: int) -> Any:
    import torch

    if isinstance(value, torch.Tensor) and value.ndim > 0 and value.shape[0] == gaussian_count:
        return value[mask].clone()
    if isinstance(value, dict):
        return {
            key: _prune_optimizer_value(item, mask, gaussian_count)
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [_prune_optimizer_value(item, mask, gaussian_count) for item in value]
    if isinstance(value, tuple):
        return tuple(_prune_optimizer_value(item, mask, gaussian_count) for item in value)
    return value


def prune_checkpoint(checkpoint: dict[str, Any], keep: np.ndarray) -> dict[str, Any]:
    import torch

    if keep.ndim != 1 or keep.dtype != np.bool_:
        raise ValueError("keep must be a one-dimensional boolean array")
    gaussian_count = len(keep)
    mask = torch.from_numpy(keep)
    pipeline = checkpoint["pipeline"]
    gaussian_keys = [
        key for key in pipeline if key.startswith("_model.gauss_params.")
    ]
    if not gaussian_keys:
        raise ValueError("checkpoint has no Gaussian parameter tensors")
    for key in gaussian_keys:
        value = pipeline[key]
        if value.shape[0] != gaussian_count:
            raise ValueError(f"Gaussian tensor has an unexpected shape: {key}")
        pipeline[key] = value[mask].clone()
    checkpoint["optimizers"] = _prune_optimizer_value(
        checkpoint.get("optimizers", {}), mask, gaussian_count
    )
    return checkpoint


def clean_gaussian_checkpoint(
    source_checkpoint: Path,
    source_cloud: Path,
    colmap_dir: Path,
    dataparser_transform: Path,
    output_checkpoint: Path,
    summary_path: Path,
    *,
    min_opacity: float,
    max_gaussian_scale: float,
    max_seed_distance: float,
    min_camera_support: int,
) -> dict[str, object]:
    import torch

    checkpoint = torch.load(source_checkpoint, map_location="cpu", weights_only=False)
    pipeline = checkpoint["pipeline"]
    means = pipeline["_model.gauss_params.means"].numpy()
    opacity = torch.sigmoid(pipeline["_model.gauss_params.opacities"]).numpy().ravel()
    max_scale = torch.exp(pipeline["_model.gauss_params.scales"]).numpy().max(axis=1)

    transform_record = json.loads(dataparser_transform.read_text(encoding="utf-8"))
    transform = np.asarray(transform_record["transform"], dtype=np.float64)
    scale = float(transform_record["scale"])
    source_points, _ = _load_ascii_ply(source_cloud)
    normalized_source = (
        source_points @ transform[:, :3].T + transform[:, 3]
    ) * scale
    seed_distance = cKDTree(normalized_source).query(means, k=1, workers=-1)[0]

    intrinsics, pose_records, _ = _read_colmap(colmap_dir)
    original_means = (means / scale - transform[:, 3]) @ transform[:, :3]
    camera_support = focus_visibility_counts(
        original_means,
        [(rotation, translation) for _, rotation, translation in pose_records],
        intrinsics,
    )
    keep = cleanup_mask(
        opacity,
        max_scale,
        seed_distance,
        camera_support,
        min_opacity=min_opacity,
        max_gaussian_scale=max_gaussian_scale,
        max_seed_distance=max_seed_distance,
        min_camera_support=min_camera_support,
    )
    if not np.any(keep):
        raise ValueError("cleanup thresholds removed every Gaussian")

    reason_counts = {
        "low_opacity": int(np.count_nonzero(opacity < min_opacity)),
        "oversized": int(np.count_nonzero(max_scale > max_gaussian_scale)),
        "far_from_dense_cloud": int(np.count_nonzero(seed_distance > max_seed_distance)),
        "weak_camera_support": int(np.count_nonzero(camera_support < min_camera_support)),
    }
    clean = prune_checkpoint(checkpoint, keep)
    output_checkpoint.parent.mkdir(parents=True, exist_ok=True)
    torch.save(clean, output_checkpoint)
    summary = {
        "source_checkpoint": str(source_checkpoint.resolve()),
        "source_cloud": str(source_cloud.resolve()),
        "uses_ground_truth": False,
        "source_gaussians": len(keep),
        "kept_gaussians": int(np.count_nonzero(keep)),
        "removed_gaussians": int(np.count_nonzero(~keep)),
        "removed_fraction": float(np.mean(~keep)),
        "thresholds": {
            "min_opacity": min_opacity,
            "max_gaussian_scale": max_gaussian_scale,
            "max_seed_distance": max_seed_distance,
            "min_camera_support": min_camera_support,
        },
        "individual_reason_counts": reason_counts,
        "output_checkpoint": str(output_checkpoint.resolve()),
    }
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Prune weakly supported Gaussians from a Nerfstudio checkpoint."
    )
    parser.add_argument("--source-checkpoint", required=True, type=Path)
    parser.add_argument("--source-cloud", required=True, type=Path)
    parser.add_argument("--colmap-dir", required=True, type=Path)
    parser.add_argument("--dataparser-transform", required=True, type=Path)
    parser.add_argument("--output-checkpoint", required=True, type=Path)
    parser.add_argument("--summary", required=True, type=Path)
    parser.add_argument("--min-opacity", type=float, default=0.1)
    parser.add_argument("--max-gaussian-scale", type=float, default=0.02)
    parser.add_argument("--max-seed-distance", type=float, default=0.05)
    parser.add_argument("--min-camera-support", type=int, default=2)
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    summary = clean_gaussian_checkpoint(
        args.source_checkpoint,
        args.source_cloud,
        args.colmap_dir,
        args.dataparser_transform,
        args.output_checkpoint,
        args.summary,
        min_opacity=args.min_opacity,
        max_gaussian_scale=args.max_gaussian_scale,
        max_seed_distance=args.max_seed_distance,
        min_camera_support=args.min_camera_support,
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
