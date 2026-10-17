"""Screen RGB-D odometry settings on start/middle/end windows, then validate a winner."""
from __future__ import annotations

import argparse
from dataclasses import asdict, replace
from functools import lru_cache
import json
from pathlib import Path
from time import perf_counter

import cv2
import numpy as np

from sfm_reconstruction import stage3
from sfm_reconstruction.official_eval import run_official_eval

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT/"outputs/hyperparameter_check/scene_frontend")
    parser.add_argument("--window-frames", type=int, default=60)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    dataset = stage3.load_stage3_dataset(ROOT.parent/"Experiments/Stage_3_Data/stage3")
    gt = stage3.load_trajectory(dataset.ground_truth_trajectory_path)
    gt_times = np.asarray([p.timestamp for p in gt])
    # Feature extraction depends only on these three settings; thresholds reuse it.
    extract = stage3._extract_stage3_features

    @lru_cache(maxsize=600)
    def features(path, mode, count, scale):
        return extract(path, stage3.Stage3OdometryConfig(feature_mode=mode, max_features=count, image_scale=scale))

    stage3._extract_stage3_features = lambda path, c: features(path, c.feature_mode, c.max_features, c.image_scale)
    base = stage3.Stage3OdometryConfig(image_scale=.5)
    cases = [("baseline", base)]
    choices = {"ratio_threshold": [.7, .75, .85], "pnp_reprojection_error": [1., 3., 4.],
        "min_depth": [.2], "max_depth": [4., 12.], "max_translation_step": [.1, None],
        "max_rotation_step_degrees": [10., 40.], "max_features": [1500, 3500],
        "feature_mode": ["sift"], "image_scale": [1.], "min_pnp_inliers": [20]}
    for key, values in choices.items():
        cases.extend((f"{key}_{value}", replace(base, **{key: value})) for value in values)
    cases.extend([("rgbd", replace(base, motion_model="rgbd")),
        ("rgbd_ransac02", replace(base, motion_model="rgbd", ransac_3d_threshold=.02)),
        ("rgbd_ransac10", replace(base, motion_model="rgbd", ransac_3d_threshold=.1)),
        ("lk_pnp", replace(base, motion_model="lk_pnp")),
        ("lk_dense", replace(base, motion_model="lk_pnp", lk_min_distance=4., lk_quality_level=.005))])
    n = args.window_frames
    starts = [0, len(dataset.frames)//2-n//2, len(dataset.frames)-n]
    path = args.output/"results.json"
    payload = json.loads(path.read_text()) if path.exists() else {"window_starts": starts,
        "window_frames": n, "note": "Exploratory RGB-D screening on three disjoint windows. GT is evaluation-only. RPE is consecutive relative-pose error without scale fitting. Window runtimes include feature-cache misses and are not comparable cold-start full-pipeline benchmarks. Failure fraction includes held poses. Full-run validation, if performed, uses official ATE.", "runs": []}
    for label, config in cases:
        if any(r["label"] == label for r in payload["runs"]):
            continue
        print(f"scene frontend: {label}", flush=True)
        record = {"label": label, "config": asdict(config), "windows": []}
        for start in starts:
            cv2.setRNGSeed(0)
            segment = replace(dataset, frames=dataset.frames[start:start+n])
            before = features.cache_info()
            began = perf_counter()
            result = stage3.estimate_rgbd_trajectory(segment, config)
            seconds = perf_counter()-began
            times = np.asarray([p.timestamp for p in result.poses])
            indices = [int(np.argmin(np.abs(gt_times-t))) for t in times]
            if any(abs(gt_times[j]-t) > .02 for j, t in zip(indices, times)):
                raise ValueError("GT timestamp mismatch")
            estimated = [stage3._transform_from_trajectory_pose(p) for p in result.poses]
            reference = [stage3._transform_from_trajectory_pose(gt[j]) for j in indices]
            translation, rotation = [], []
            for i in range(1, len(estimated)):
                true_step = np.linalg.inv(reference[i-1])@reference[i]
                estimated_step = np.linalg.inv(estimated[i-1])@estimated[i]
                error = np.linalg.inv(true_step)@estimated_step
                translation.append(np.linalg.norm(error[:3, 3]))
                rotation.append(np.degrees(np.linalg.norm(cv2.Rodrigues(error[:3, :3])[0])))
            tracked = sum(r.status == "tracked" for r in result.reports)
            dest = args.output/label/f"window{start}"
            dest.mkdir(parents=True, exist_ok=True)
            stage3.write_trajectory(dest/"trajectory.txt", result.poses)
            record["windows"].append({"start": start, "seconds": seconds,
                "feature_cache_misses": features.cache_info().misses-before.misses,
                "tracked_fraction": tracked/(len(result.poses)-1),
                "translation_rpe_rmse": float(np.sqrt(np.mean(np.square(translation)))),
                "rotation_rpe_rmse_degrees": float(np.sqrt(np.mean(np.square(rotation))))})
        for key in ("tracked_fraction", "translation_rpe_rmse", "rotation_rpe_rmse_degrees"):
            record[key] = float(np.mean([w[key] for w in record["windows"]]))
        payload["runs"].append(record)
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(json.dumps({k: v for k, v in record.items() if k not in {"windows", "config"}}), flush=True)
    baseline = payload["runs"][0]
    def score(r):
        if r["tracked_fraction"] < baseline["tracked_fraction"] or any(
            w["tracked_fraction"] < b["tracked_fraction"] for w, b in zip(r["windows"], baseline["windows"])):
            return float("inf")
        return .5*(r["translation_rpe_rmse"]/baseline["translation_rpe_rmse"]+
            r["rotation_rpe_rmse_degrees"]/baseline["rotation_rpe_rmse_degrees"])
    selected = min(payload["runs"], key=score)
    payload["screen_selected"] = selected["label"]
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    for validation_key, candidate in (("full_baseline", baseline), ("full_validation", selected)):
        if validation_key in payload:
            continue
        if validation_key == "full_validation" and selected["label"] == baseline["label"]:
            payload[validation_key] = dict(payload["full_baseline"])
            path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
            continue
        print(f"Full-scene validation: {candidate['label']}", flush=True)
        # Bound cache memory and keep full-run timing independent of window warmup.
        features.cache_clear()
        full_count = 0
        def full_features(path, config):
            nonlocal full_count
            full_count += 1
            if full_count % 250 == 0:
                print(f"Full scene: {full_count}/{len(dataset.frames)} frames", flush=True)
            return extract(path, config)
        stage3._extract_stage3_features = full_features
        cv2.setRNGSeed(0)
        start = perf_counter()
        result = stage3.estimate_rgbd_trajectory(dataset, stage3.Stage3OdometryConfig(**candidate["config"]))
        seconds = perf_counter()-start
        dest = args.output/validation_key
        dest.mkdir(exist_ok=True)
        trajectory = dest/"trajectory.txt"
        stage3.write_trajectory(trajectory, result.poses)
        evaluation = run_official_eval(dataset.ground_truth_trajectory_path, trajectory, dest,
            label=validation_key, evaluator_dir=dataset.root)
        if evaluation.returncode or evaluation.rmse is None:
            raise RuntimeError("Official evaluation failed")
        payload[validation_key] = {"label": candidate["label"], "config": candidate["config"],
            "seconds": seconds, "official_ate": evaluation.rmse,
            "compared_pose_pairs": evaluation.compared_pose_pairs,
            "tracked_frames": sum(r.status == "tracked" for r in result.reports), "poses": len(result.poses)}
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(json.dumps(payload[validation_key]), flush=True)


if __name__ == "__main__":
    main()
