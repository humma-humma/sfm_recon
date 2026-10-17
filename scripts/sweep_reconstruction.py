"""Checkpointed object reconstruction screening and interaction checks."""
from __future__ import annotations

import argparse
from dataclasses import asdict, replace
import json
from pathlib import Path
from time import perf_counter

import cv2
import numpy as np
from scipy.spatial import cKDTree

from benchmark_native import reprojection_metrics
from sfm_reconstruction.dataset import load_dataset
from sfm_reconstruction.evaluation import evaluate_poses
from sfm_reconstruction.io import write_camera_parameters, write_point_cloud, write_summary
from sfm_reconstruction.reconstruction import ReconstructionConfig, reconstruct
from sfm_reconstruction.stage1_evaluation import _load_vertices, evaluate_stage1_points
from sfm_reconstruction.stage2_evaluation import evaluate_mesh_proxy


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT.parent/"Experiments"


def cases(base):
    result = [("baseline", base)]
    values = {
        "bundle_adjustment_loss_scale": [.5, 1., 4.],
        "bundle_adjustment_max_reprojection_error": [2., 3., 5., 8.],
        "min_track_observations": [2, 3, 4],
        "min_triangulation_angle": [.5, 1., 2., 3.],
        "pnp_reprojection_error": [2., 3., 6.],
        "essential_threshold": [.5, 2.],
        "max_reprojection_error": [3., 8.],
        "max_point_distance_factor": [1.25, 1.5, 2.],
        "ceres_max_iterations": [5],
    }
    for key, choices in values.items():
        for value in choices:
            if value != getattr(base, key):
                result.append((f"{key}_{value:g}", replace(base, **{key: value})))
    result.extend([
        ("two_view_quality5", replace(base, min_track_observations=2,
            two_view_min_triangulation_angle=5., two_view_max_reprojection_error=3.)),
        ("two_view_quality10", replace(base, min_track_observations=2,
            two_view_min_triangulation_angle=10., two_view_max_reprojection_error=2.)),
        ("python_control20", replace(base, bundle_adjustment_backend="python", bundle_adjustment_max_nfev=20)),
    ])
    return result


def screen_score(record, baseline):
    """Guard coverage before ranking relative errors; not a held-out objective."""
    if record.get("status") != "ok":
        return float("inf")
    if record["cameras"] < baseline["cameras"] or record["points"] < .9*baseline["points"]:
        return float("inf")
    if record["image_coverage"] < .95*baseline["image_coverage"]:
        return float("inf")
    terms = []
    if baseline.get("pose_metrics"):
        for key in ("mean_translation", "mean_rotation_degrees"):
            terms.append(record["pose_metrics"][key]/max(baseline["pose_metrics"][key], 1e-12))
    else:
        terms.append(record["reprojection_rmse_px"]/baseline["reprojection_rmse_px"])
    if baseline.get("mesh_proxy"):
        if record["mesh_coverage"] < .95*baseline["mesh_coverage"]:
            return float("inf")
        terms.append(record["mesh_proxy"]["cropped_mean_nearest_vertex_distance"]/
                     baseline["mesh_proxy"]["cropped_mean_nearest_vertex_distance"])
    if baseline.get("point_metrics"):
        terms.append(record["point_metrics"]["chamfer_distance"]/baseline["point_metrics"]["chamfer_distance"])
    return float(np.mean(terms))


def run_object(name, output, limit=None):
    root = DATA/("Stage_1_Data_ver._4/Stage_1_Data_ver_4/stage1/box" if name == "box" else f"Stage_2_Data/stage2/{name}")
    correspondence = None if name == "box" else ROOT/f"outputs/stage2_{name}_{'improved' if name == 'boot' else 'masked_track3_ba'}/matching_cache/correspondences"
    dataset = load_dataset(root, correspondence_dir=correspondence)
    base = ReconstructionConfig(bundle_adjustment_backend="ceres", geometry_backend="cpp",
        min_track_observations=2 if name == "box" else 3,
        max_point_distance_factor=None if name == "box" else 1.5,
        bundle_adjustment_max_reprojection_error=5. if name == "box" else 3.)
    configs = cases(base)
    gt_points = _load_vertices(root/"gt_points.ply") if (root/"gt_points.ply").exists() else None
    gt_mesh = _load_vertices(root/"gt_mesh.ply") if (root/"gt_mesh.ply").exists() else None
    image = cv2.imread(str(next(iter(dataset.image_paths.values()))))
    height, width = image.shape[:2]
    dest = output/name
    dest.mkdir(parents=True, exist_ok=True)
    path = dest/"results.json"
    payload = json.loads(path.read_text()) if path.exists() else {
        "dataset": str(root), "seed": 0, "runs": [],
        "note": "Single-call complete reconstruction timing; cached matching is excluded. GT is evaluation-only; selection on this dataset is exploratory. Boot has no full camera/mesh ground truth. Image coverage is mean occupied 8x8 observed-feature bins for reconstructed tracks, not surface coverage."}

    def run(label, config, source=dataset, seed=0):
        previous = next((r for r in payload["runs"] if r["label"] == label), None)
        if previous:
            return previous
        print(f"{name}: {label}", flush=True)
        record = {"label": label, "config": asdict(config), "seed": seed}
        try:
            cv2.setRNGSeed(seed)
            start = perf_counter()
            result = reconstruct(source, config)
            record["seconds"] = perf_counter()-start
            if not result.points:
                raise ValueError("No reconstructed points")
            xyz = np.asarray(list(result.points.values()))
            pose = evaluate_poses(result.poses, source.ground_truth_extrinsics)
            bins = {c: set() for c in result.poses}
            for p in result.points:
                for c, xy in result.tracks[p].observations.items():
                    if c in bins and 0 <= xy[0] < width and 0 <= xy[1] < height:
                        bins[c].add((int(8*xy[0]/width), int(8*xy[1]/height)))
            record.update(status="ok", cameras=len(result.poses), points=len(xyz),
                initial_pair=result.initial_pair,
                pose_metrics=pose.to_dict() if pose else None,
                image_coverage=float(np.mean([len(v)/64 for v in bins.values()])),
                **reprojection_metrics(source.intrinsics, result.tracks, result.poses, result.points))
            if gt_points is not None:
                record["point_metrics"] = asdict(evaluate_stage1_points(gt_points, xyz))
            if gt_mesh is not None and pose:
                record["mesh_proxy"] = asdict(evaluate_mesh_proxy(gt_mesh, xyz, pose.scale))
                distances = cKDTree(xyz/pose.scale).query(gt_mesh)[0]
                record["mesh_coverage"] = float(np.mean(distances <= .05))
                record["mesh_to_points_mean"] = float(np.mean(distances))
            folder = dest/label
            folder.mkdir(exist_ok=True)
            write_camera_parameters(folder/"estimated_camera_parameters.json", source, result)
            write_point_cloud(folder/"estimated_points.ply", result)
            write_summary(folder/"summary.json", source, result)
        except (RuntimeError, ValueError, cv2.error) as exc:
            record.update(status="failed", error=str(exc))
        payload["runs"].append(record)
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(json.dumps({k: record[k] for k in ("label", "status", "seconds", "cameras", "points", "pose_metrics", "error") if k in record}), flush=True)
        return record

    for label, config in configs[:limit]:
        run(label, config)
    if limit:
        return
    # Cached frontend families provide a bounded match-quality/coverage check.
    if name != "box":
        alternative = ROOT/f"outputs/stage2_{name}_sift_akaze_probe/matching_cache/correspondences"
        if alternative.is_dir():
            source = load_dataset(root, correspondence_dir=alternative)
            run("sift_akaze_track3", base, source)
            run("sift_akaze_two_view_quality", replace(base, min_track_observations=2,
                two_view_min_triangulation_angle=10., two_view_max_reprojection_error=2.), source)
    baseline = payload["runs"][0]
    ranked = sorted((r for r in payload["runs"] if r["label"] not in {"baseline", "python_control20"}
                     and not r["label"].startswith(("sift_akaze", "selected_", "baseline_"))), key=lambda r: screen_score(r, baseline))
    improving = [r for r in ranked if screen_score(r, baseline) < 1.]
    combined = base
    used = set()
    for candidate in improving:
        changes = {k: v for k, v in candidate["config"].items() if v != asdict(base)[k]}
        if used & changes.keys():
            continue
        combined = replace(combined, **changes)
        used.update(changes)
        if len(used) >= 3:
            break
    if used:
        run("combined_screen_winners", combined)
    ranked = sorted((r for r in payload["runs"] if not r["label"].startswith(("selected_", "baseline_"))), key=lambda r: screen_score(r, baseline))
    selected = ranked[0]
    payload["selection_rule"] = "At least baseline camera count, 90% baseline point count and 95% baseline image and mesh coverage. Rank mean normalized camera errors plus geometry proxy (boot: reprojection consistency only). This is exploratory GT tuning, not independent validation."
    payload["selected"] = selected["label"]
    payload["selected_score"] = screen_score(selected, baseline)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    # Repeat the selected configuration for runtime and deterministic recovery.
    selected_config = ReconstructionConfig(**selected["config"])
    selected_source = dataset
    if selected["label"].startswith("sift_akaze"):
        selected_source = load_dataset(root, correspondence_dir=alternative)
    for label in ("baseline_repeat1", "baseline_repeat2"):
        run(label, base)
    for label in ("selected_repeat1", "selected_repeat2"):
        run(label, selected_config, selected_source)
    run("baseline_seed1", base, seed=1)
    run("selected_seed1", selected_config, selected_source, seed=1)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--objects", nargs="+", choices=("box", "boot", "milk"), default=("box", "boot", "milk"))
    parser.add_argument("--output", type=Path, default=ROOT/"outputs/hyperparameter_check")
    parser.add_argument("--limit", type=int, help="Smoke test only: first N cases")
    args = parser.parse_args()
    for name in args.objects:
        run_object(name, args.output, args.limit)


if __name__ == "__main__":
    main()
