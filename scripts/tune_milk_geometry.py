"""Exploratory milk reconstruction sweep with Ceres and coverage evaluation."""
from __future__ import annotations

import argparse
from dataclasses import asdict, replace
import json
from pathlib import Path
from time import perf_counter

import cv2
import numpy as np
from scipy.spatial import cKDTree

from sfm_reconstruction.dataset import load_dataset
from sfm_reconstruction.evaluation import evaluate_poses
from sfm_reconstruction.io import write_camera_parameters, write_point_cloud, write_summary
from sfm_reconstruction.reconstruction import ReconstructionConfig, reconstruct
from sfm_reconstruction.stage2_evaluation import _load_vertices, evaluate_mesh_proxy


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--correspondences", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--case", help="Run only the named case; default runs all")
    args = parser.parse_args()
    dataset = load_dataset(args.dataset, correspondence_dir=args.correspondences)
    gt = _load_vertices(dataset.root/"gt_mesh.ply")
    base = ReconstructionConfig(min_track_observations=3, max_point_distance_factor=1.5,
        bundle_adjustment_backend="ceres", bundle_adjustment_max_reprojection_error=3.)
    cases = [("baseline", base),
        ("track4", replace(base, min_track_observations=4)),
        ("track5", replace(base, min_track_observations=5)),
        ("angle2", replace(base, min_triangulation_angle=2.)),
        ("angle3", replace(base, min_triangulation_angle=3.)),
        ("distance1.25", replace(base, max_point_distance_factor=1.25)),
        ("distance2", replace(base, max_point_distance_factor=2.)),
        ("pnp2", replace(base, pnp_reprojection_error=2.)),
        ("angle2_distance1.25", replace(base, min_triangulation_angle=2., max_point_distance_factor=1.25))]
    if args.case:
        cases = [(label, config) for label, config in cases if label == args.case]
        if not cases:
            parser.error("Unknown case")
    payload = {"note": "Exploratory milk tuning. Full reconstruction is repeated from the same correspondence files with OpenCV seed 0. Ground truth is evaluation-only. Single-call runtime includes track building, initialization, registration, triangulation, BA and filtering; excludes matching and evaluation. Mesh-to-points distances additionally measure coverage using the same scale-only alignment as the existing proxy.",
        "dataset": str(dataset.root), "runs": []}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    for label, config in cases:
        print(label, flush=True)
        cv2.setRNGSeed(0)
        start = perf_counter()
        result = reconstruct(dataset, config)
        seconds = perf_counter()-start
        pose = evaluate_poses(result.poses, dataset.ground_truth_extrinsics)
        if pose is None:
            raise ValueError("Milk sweep requires ground-truth camera poses")
        xyz = np.asarray(list(result.points.values()))
        proxy = evaluate_mesh_proxy(gt, xyz, pose.scale)
        distances = cKDTree(xyz/pose.scale).query(gt)[0]
        record = {"label": label, "config": asdict(config), "seconds": seconds,
            "cameras": len(result.poses), "registered_camera_ids": sorted(result.poses),
            "initial_pair": result.initial_pair, "points": len(xyz),
            "pose_metrics": pose.to_dict(), "mesh_proxy": asdict(proxy),
            "coverage": {"mesh_to_points_mean": float(np.mean(distances)),
                "mesh_to_points_p90": float(np.percentile(distances, 90)),
                "mesh_vertices_fraction_within_005": float(np.mean(distances <= .05))}}
        dest = args.output.parent/label
        dest.mkdir(exist_ok=True)
        write_camera_parameters(dest/"estimated_camera_parameters.json", dataset, result)
        write_point_cloud(dest/"estimated_points.ply", result)
        write_summary(dest/"summary.json", dataset, result)
        payload["runs"].append(record)
        args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(json.dumps({k: record[k] for k in ("label", "seconds", "cameras", "points", "pose_metrics", "mesh_proxy", "coverage")}), flush=True)


if __name__ == "__main__":
    main()
