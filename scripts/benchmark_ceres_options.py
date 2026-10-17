"""Compare Schur solvers and thread counts on identical selected pre-BA inputs."""
from dataclasses import asdict, replace
import argparse
import json
from pathlib import Path
from statistics import median
from time import perf_counter

import cv2
import numpy as np
from scipy.spatial import cKDTree

from sfm_reconstruction.bundle_adjustment import bundle_adjust
from sfm_reconstruction.dataset import load_dataset
from sfm_reconstruction.evaluation import evaluate_poses
from sfm_reconstruction.native_backend import load_ceres
from sfm_reconstruction.reconstruction import ReconstructionConfig, reconstruct, _filter_points
from sfm_reconstruction.stage1_evaluation import _load_vertices, evaluate_stage1_points
from sfm_reconstruction.stage2_evaluation import evaluate_mesh_proxy

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT/"outputs/cpp_followup/ceres_options"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--confirm-pipeline", action="store_true")
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    load_ceres()  # Exclude DLL import/warmup from all measured calls.
    for name in ("box", "boot", "milk"):
        source = json.loads((ROOT/f"outputs/hyperparameter_check/{name}/results.json").read_text())
        selected = next(r for r in source["runs"] if r["label"] == source["selected"])
        config = ReconstructionConfig(**selected["config"])
        cache = None if name == "box" else ROOT/f"outputs/stage2_{name}_{'improved' if name == 'boot' else 'masked_track3_ba'}/matching_cache/correspondences"
        dataset = load_dataset(source["dataset"], correspondence_dir=cache)
        cv2.setRNGSeed(0)
        initial = reconstruct(dataset, replace(config, bundle_adjustment=False))
        path = OUT/f"{name}.json"
        payload = json.loads(path.read_text()) if path.exists() else {"dataset": name,
            "input_config": asdict(config), "cameras": len(initial.poses), "points": len(initial.points),
            "note": "Three complete BA calls per option on identical pre-BA input; filtering and evaluation excluded. Output metrics use the same fixed post-filter and GT evaluation. Tuned on these datasets, not held out.", "runs": []}
        gt_points = _load_vertices(dataset.root/"gt_points.ply") if (dataset.root/"gt_points.ply").exists() else None
        gt_mesh = _load_vertices(dataset.root/"gt_mesh.ply") if (dataset.root/"gt_mesh.ply").exists() else None
        for solver in ("dense_schur", "sparse_schur", "iterative_schur"):
            for threads in (1, 2, 4):
                label = f"{solver}_{threads}threads"
                if any(r["label"] == label for r in payload["runs"]):
                    continue
                print(f"{name}: {label}", flush=True)
                record = {"label": label, "solver": solver, "threads": threads}
                try:
                    times, reports = [], []
                    for _ in range(3):
                        report = {}
                        start = perf_counter()
                        poses, points = bundle_adjust(dataset.intrinsics, initial.tracks,
                            initial.poses, initial.points, set(initial.initial_pair), backend="ceres",
                            max_reprojection_error=config.bundle_adjustment_max_reprojection_error or config.max_reprojection_error,
                            loss_scale=config.bundle_adjustment_loss_scale,
                            ceres_max_iterations=config.ceres_max_iterations,
                            ceres_num_threads=threads, ceres_linear_solver=solver, solver_report=report)
                        times.append(perf_counter()-start)
                        reports.append(report)
                    filtered = _filter_points(dataset.intrinsics, initial.tracks, poses, points, config)
                    xyz = np.asarray(list(filtered.values()))
                    pose = evaluate_poses(poses, dataset.ground_truth_extrinsics)
                    record.update(status="ok", seconds=times, median_seconds=median(times),
                        solver_reports=reports, cameras=len(poses), points=len(filtered),
                        pose_metrics=pose.to_dict() if pose else None)
                    if gt_points is not None:
                        record["point_metrics"] = asdict(evaluate_stage1_points(gt_points, xyz))
                    if gt_mesh is not None:
                        record["mesh_proxy"] = asdict(evaluate_mesh_proxy(gt_mesh, xyz, pose.scale))
                        distances = cKDTree(xyz/pose.scale).query(gt_mesh)[0]
                        record["mesh_coverage"] = float(np.mean(distances <= .05))
                except (RuntimeError, ValueError) as exc:
                    record.update(status="failed", error=str(exc))
                payload["runs"].append(record)
                path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
                print(json.dumps({k:v for k,v in record.items() if k not in {"solver_reports", "point_metrics", "mesh_proxy", "pose_metrics"}}), flush=True)
        baseline = next(r for r in payload["runs"] if r["label"] == "dense_schur_1threads")
        def equivalent(r):
            return r["status"] == "ok" and r["cameras"] == baseline["cameras"] and r["points"] == baseline["points"] and all(
                report["converged"] and abs(report["final_cost"]-baseline["solver_reports"][-1]["final_cost"]) <= 1e-4*max(1, baseline["solver_reports"][-1]["final_cost"])
                for report in r["solver_reports"])
        candidates = [r for r in payload["runs"] if equivalent(r)]
        payload["fastest_equivalent_objective"] = min(candidates, key=lambda r: r["median_seconds"])["label"] if candidates else None
        payload["selection_note"] = "Timing candidate requires converged near-equal robust cost and identical post-filter point counts. Camera and geometry metrics must also be reviewed; this does not automatically promote a preset."
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        if args.confirm_pipeline and "pipeline_confirmation" not in payload and candidates:
            fastest = min(candidates, key=lambda r:r["median_seconds"])
            solutions, timings = {}, {}
            for label, trial in (("baseline", config), ("selected", replace(config,
                ceres_num_threads=fastest["threads"], ceres_linear_solver=fastest["solver"]))):
                print(f"{name}: full pipeline {label}", flush=True)
                cv2.setRNGSeed(0)
                start = perf_counter()
                solutions[label] = reconstruct(dataset, trial)
                timings[label] = perf_counter()-start
            baseline_result, chosen = solutions["baseline"], solutions["selected"]
            same_ids = baseline_result.points.keys() == chosen.points.keys() and baseline_result.poses.keys() == chosen.poses.keys()
            camera_difference = max(float(np.max(np.abs(baseline_result.poses[c].matrix()-chosen.poses[c].matrix()))) for c in baseline_result.poses)
            point_difference = max(float(np.linalg.norm(baseline_result.points[p]-chosen.points[p])) for p in baseline_result.points) if same_ids else None
            payload["pipeline_confirmation"] = {"option":fastest["label"], "seconds":timings,
                "identical_ids":same_ids, "max_camera_matrix_difference":camera_difference,
                "max_point_distance":point_difference,
                "equivalent":bool(same_ids and camera_difference < 1e-5 and point_difference < 1e-4)}
            if payload["pipeline_confirmation"]["equivalent"]:
                from sfm_reconstruction.io import write_camera_parameters, write_point_cloud, write_summary
                dest = OUT/name
                dest.mkdir(exist_ok=True)
                write_camera_parameters(dest/"estimated_camera_parameters.json", dataset, chosen)
                write_point_cloud(dest/"estimated_points.ply", chosen)
                write_summary(dest/"summary.json", dataset, chosen)
            path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
            print(json.dumps(payload["pipeline_confirmation"]), flush=True)


if __name__ == "__main__":
    main()
