"""Small exploratory BA sweep; ground truth is used only for evaluation."""
from __future__ import annotations

import argparse
from dataclasses import asdict, replace
import json
from pathlib import Path
from time import perf_counter

import numpy as np

from benchmark_native import reprojection_metrics
from sfm_reconstruction.bundle_adjustment import bundle_adjust
from sfm_reconstruction.dataset import load_dataset
from sfm_reconstruction.evaluation import evaluate_poses
from sfm_reconstruction.io import write_camera_parameters, write_point_cloud, write_summary
from sfm_reconstruction.native_backend import load_ceres
from sfm_reconstruction.reconstruction import ReconstructionConfig, reconstruct, _filter_points
from sfm_reconstruction.stage1_evaluation import _load_vertices, evaluate_stage1_points
from sfm_reconstruction.stage2_evaluation import evaluate_mesh_proxy


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--correspondences", type=Path)
    parser.add_argument("--min-track-observations", type=int, default=2)
    parser.add_argument("--max-point-distance-factor", type=float)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    native = load_ceres()
    dataset = load_dataset(args.dataset, correspondence_dir=args.correspondences)
    config = ReconstructionConfig(bundle_adjustment=False,
        min_track_observations=args.min_track_observations,
        max_point_distance_factor=args.max_point_distance_factor)
    print("Building shared pre-BA input...", flush=True)
    initial = reconstruct(dataset, config)
    gt_points = _load_vertices(dataset.root/"gt_points.ply") if (dataset.root/"gt_points.ply").is_file() else None
    gt_mesh = _load_vertices(dataset.root/"gt_mesh.ply") if (dataset.root/"gt_mesh.ply").is_file() else None
    settings = [(s, g, 100) for g in (5., 3.) for s in (2., .5, 1., 4.)]
    settings += [(2., 8., 100)] + [(2., 5., n) for n in (3, 5, 8)]
    payload = {"dataset": str(dataset.root), "ceres_version": native.version,
        "input_cameras": len(initial.poses), "input_points": len(initial.points),
        "note": "Exploratory single-dataset tuning; default post-filter is fixed. Timings are single calls. All-observation soft_l1_cost uses scale 2 for comparison; solver costs across loss scales/gates are not comparable.",
        "runs": []}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    for scale, gate, iterations in settings:
        label = f"scale{scale:g}_gate{gate:g}_iter{iterations}"
        print(label, flush=True)
        report = {}
        start = perf_counter()
        poses, points = bundle_adjust(dataset.intrinsics, initial.tracks, initial.poses,
            initial.points, set(initial.initial_pair), backend="ceres", loss_scale=scale,
            max_reprojection_error=gate, ceres_max_iterations=iterations, solver_report=report)
        seconds = perf_counter()-start
        filtered = _filter_points(dataset.intrinsics, initial.tracks, poses, points, config)
        metrics = evaluate_poses(poses, dataset.ground_truth_extrinsics)
        record = {"label": label, "loss_scale": scale, "observation_gate_px": gate,
            "max_iterations": iterations, "seconds": seconds, "solver": report,
            "retained_points": len(filtered), "same_point_ids": set(filtered) == set(initial.points),
            "pose_metrics": metrics.to_dict() if metrics else None,
            **reprojection_metrics(dataset.intrinsics, initial.tracks, poses, points)}
        if gt_points is not None:
            record["point_metrics"] = asdict(evaluate_stage1_points(gt_points, np.asarray(list(filtered.values()))))
        if gt_mesh is not None and metrics:
            record["mesh_proxy"] = asdict(evaluate_mesh_proxy(gt_mesh, np.asarray(list(filtered.values())), metrics.scale))
            record["unfiltered_mesh_proxy"] = asdict(evaluate_mesh_proxy(gt_mesh, np.asarray(list(points.values())), metrics.scale))
        final = replace(initial, poses=poses, points=filtered)
        dest = args.output.parent/label
        dest.mkdir(exist_ok=True)
        write_camera_parameters(dest/"estimated_camera_parameters.json", dataset, final)
        write_point_cloud(dest/"estimated_points.ply", final)
        write_summary(dest/"summary.json", dataset, final)
        payload["runs"].append(record)
        args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(json.dumps({k: record[k] for k in ("label", "seconds", "retained_points", "pose_metrics", "mesh_proxy") if k in record}), flush=True)


if __name__ == "__main__":
    main()
