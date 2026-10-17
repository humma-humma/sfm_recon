"""Create an opt-in depth-consistency candidate using training cameras only."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from time import perf_counter

import cv2
import numpy as np

from .stage3 import load_stage3_dataset
from .stage3_gaussian_checkpoint_cleanup import prune_checkpoint
from .stage3_gaussian_seed_rebalance import focus_visibility_counts
from .stage3_splat_temporal_diagnostics import _read_colmap


def depth_labels(points, rotation, translation, k, depth, absolute, relative):
    """0 unknown, 1 surface agreement, 2 free-space conflict, 3 occluded."""
    if absolute < 0 or relative < 0 or not np.isfinite([absolute, relative]).all():
        raise ValueError("Tolerances must be finite and nonnegative")
    v = points @ rotation.T + translation
    ids = np.flatnonzero(np.isfinite(v).all(axis=1) & (v[:, 2] > 1e-6))
    xy = v[ids, :2] / v[ids, 2:3] * k[:2] + k[2:]
    inside = (xy[:, 0] >= 0) & (xy[:, 0] < depth.shape[1]) & (xy[:, 1] >= 0) & (xy[:, 1] < depth.shape[0])
    ids, xy = ids[inside], xy[inside].astype(np.int64)
    measured = depth[xy[:, 1], xy[:, 0]]
    valid = np.isfinite(measured) & (measured > 0)
    ids, measured = ids[valid], measured[valid]
    delta = v[ids, 2] - measured
    labels = np.zeros(len(points), np.uint8)
    labels[ids] = np.where(np.abs(delta) <= absolute + relative * measured, 1,
                           np.where(delta < 0, 2, 3))
    return labels


def depth_keep_mask(frustum, surface, conflict, min_conflicts=3):
    if min_conflicts < 1:
        raise ValueError("min_conflicts must be positive")
    if any(a.shape != frustum.shape for a in (surface, conflict)) or frustum.ndim != 1:
        raise ValueError("Evidence must be equally sized vectors")
    return frustum & ~((conflict >= min_conflicts) & (surface == 0))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--dataparser-transform", type=Path, required=True)
    parser.add_argument("--colmap-dir", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--absolute-tolerance", type=float, default=.10)
    parser.add_argument("--relative-tolerance", type=float, default=.05)
    parser.add_argument("--backend", choices=("python", "cpp"), default="python")
    args = parser.parse_args()
    import torch

    start = perf_counter()
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    means = checkpoint["pipeline"]["_model.gauss_params.means"].numpy()
    record = json.loads(args.dataparser_transform.read_text())
    transform = np.asarray(record["transform"])
    points = np.ascontiguousarray((means/record["scale"] - transform[:, 3]) @ transform[:, :3])
    k, records, _ = _read_colmap(args.colmap_dir)
    frustum = focus_visibility_counts(points, [(r, t) for _, r, t in records], k,
                                       backend=args.backend, num_threads=8) >= 1
    dataset = load_stage3_dataset(args.dataset)
    frames = {f"{f.timestamp:.9f}": f for f in dataset.frames}
    surface = np.zeros(len(points), np.uint32)
    conflict = np.zeros_like(surface)
    training = [r for i, r in enumerate(records) if i % 8 != 0]
    classify = depth_labels
    if args.backend == "cpp":
        from .native_backend import load_native
        classify = load_native().depth_support
    for i, (name, r, t) in enumerate(training):
        frame = frames[f"{float(Path(name).stem):.9f}"]
        depth = cv2.imread(str(frame.depth_path), cv2.IMREAD_UNCHANGED)
        if depth is None:
            raise ValueError(f"Cannot read {frame.depth_path}")
        if i == 0:
            rgb = cv2.imread(str(frame.rgb_path))
            if rgb is None or rgb.shape[:2] != depth.shape:
                raise ValueError("Depth must be aligned to RGB")
        kd = np.asarray(k[:4]) * np.asarray([depth.shape[1]/k[4], depth.shape[0]/k[5]]*2)
        depth = depth.astype(np.float64)/1000.
        depth[(depth < .1) | (depth > 8.)] = 0
        labels = classify(points, r, t, kd, depth, args.absolute_tolerance, args.relative_tolerance)
        surface += labels == 1
        conflict += labels == 2
        if i % 40 == 0:
            print(f"Depth cleanup {i}/{len(training)}", flush=True)
    keep = depth_keep_mask(frustum, surface, conflict)
    if not keep.any():
        raise ValueError("Cleanup removed every Gaussian")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(prune_checkpoint(checkpoint, keep), args.output)
    summary = {"source_checkpoint": str(args.checkpoint), "uses_ground_truth": False,
               "training_cameras": len(training), "held_out_cameras_excluded": len(records)-len(training),
               "source_gaussians": len(points), "frustum_gaussians": int(frustum.sum()),
               "kept_gaussians": int(keep.sum()), "absolute_tolerance": args.absolute_tolerance,
               "relative_tolerance": args.relative_tolerance, "seconds": perf_counter()-start,
               "status": "candidate_only_pending_render_evaluation"}
    args.output.with_suffix(".json").write_text(json.dumps(summary, indent=2))
    np.savez_compressed(args.output.with_suffix(".npz"), keep=keep, surface=surface, conflict=conflict)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
