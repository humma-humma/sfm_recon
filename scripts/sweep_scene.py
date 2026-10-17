"""Checkpointed Stage 3 full-trajectory pose-graph screening and confirmation."""
from __future__ import annotations

import argparse
from dataclasses import asdict, replace
import json
from pathlib import Path
from time import perf_counter

import cv2
import numpy as np

from sfm_reconstruction.official_eval import run_official_eval
from sfm_reconstruction.pose_graph import (LoopEdgeCandidate, PoseGraphConfig,
    build_keyframe_pose_graph, optimize_pose_graph, expand_keyframe_optimization)
from sfm_reconstruction.stage3 import load_stage3_dataset, load_trajectory, write_trajectory

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT/"outputs/hyperparameter_check/scene_graph")
    parser.add_argument("--confirm-frontend", action="store_true",
        help="Also apply the selected graph settings to the fully validated frontend trajectory")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    dataset = load_stage3_dataset(ROOT.parent/"Experiments/Stage_3_Data/stage3")
    trajectory = ROOT/"outputs/stage3_pose_graph_full/estimated_camera_trajectory.txt"
    raw = load_trajectory(trajectory)
    snapshot = ROOT/"outputs/native_benchmarks/graph_stage3_promoted/graph_input.npz"
    saved = np.load(snapshot)
    target, source = map(int, saved["edge_indices"][-1])
    measured = saved["measured"][-1]
    loop = LoopEdgeCandidate(source, target, raw[source].timestamp, raw[target].timestamp,
        "pnp", 500, 492, 456, .499778414, float(np.linalg.norm(measured[:3, 3])),
        float(np.degrees(np.linalg.norm(cv2.Rodrigues(measured[:3, :3])[0]))),
        True, "", source_from_target=measured)
    base = PoseGraphConfig(keyframe_stride=8, optimizer_max_nfev=50, optimizer_backend="cpp")
    cases = [("cpp_baseline", base), ("python_baseline", replace(base, optimizer_backend="python"))]
    choices = {"keyframe_stride": [4, 16, 32], "optimizer_loss": ["linear", "huber", "cauchy"],
        "optimizer_f_scale": [.1, .5, 2.], "optimizer_max_nfev": [20, 100],
        "loop_translation_weight": [.1, .25, 1., 2.],
        "loop_rotation_weight": [.1, .25, 1., 2.],
        "odometry_rotation_weight": [.5, 2.], "odometry_translation_weight": [.5, 2.]}
    for key, values in choices.items():
        cases.extend((f"{key}_{value}", replace(base, **{key: value})) for value in values)
    path = args.output/"results.json"
    payload = json.loads(path.read_text()) if path.exists() else {"input_trajectory": str(trajectory),
        "verified_loop_snapshot": str(snapshot), "loop_source": source, "loop_target": target,
        "note": "Full trajectory evaluation on one fixed verified loop. C++ is screening only; Python confirmation required. Ground truth is used only by the official evaluator. Single-call solver timing excludes graph construction, expansion and evaluation.", "runs": []}

    def evaluate(label, poses, record):
        dest = args.output/label
        dest.mkdir(exist_ok=True)
        target_path = dest/"trajectory.txt"
        write_trajectory(target_path, poses)
        metrics = run_official_eval(dataset.ground_truth_trajectory_path, target_path, dest,
            label=label, evaluator_dir=dataset.root)
        if metrics.returncode or metrics.rmse is None:
            raise RuntimeError(f"Official evaluator failed: {metrics.stdout_path}")
        record.update(official_ate=metrics.rmse, compared_pose_pairs=metrics.compared_pose_pairs,
            trajectory_poses=len(poses), official_scale=metrics.scale)
        payload["runs"].append(record)
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(json.dumps({k: record[k] for k in ("label", "seconds", "official_ate", "nodes", "nfev", "converged") if k in record}), flush=True)
        return record

    if not any(r["label"] == "raw_no_loop" for r in payload["runs"]):
        evaluate("raw_no_loop", raw, {"label": "raw_no_loop", "status": "ok", "seconds": 0.})

    def run(label, config, input_poses=None, input_family=None):
        previous = next((r for r in payload["runs"] if r["label"] == label), None)
        if previous:
            return previous
        print(f"scene: {label}", flush=True)
        record = {"label": label, "config": asdict(config)}
        if input_family:
            record["input_family"] = input_family
        poses_in = raw if input_poses is None else input_poses
        try:
            graph = build_keyframe_pose_graph(poses_in, [loop], config)
            start = perf_counter()
            result = optimize_pose_graph(graph, config)
            seconds = perf_counter()-start
            poses = expand_keyframe_optimization(poses_in, result.poses)
            record.update(status="ok", seconds=seconds, nodes=len(graph.nodes), edges=len(graph.edges),
                nfev=result.iterations, converged=result.success, message=result.message,
                initial_cost=result.initial_cost, final_cost=result.final_cost)
            return evaluate(label, poses, record)
        except (RuntimeError, ValueError) as exc:
            record.update(status="failed", error=str(exc))
            payload["runs"].append(record)
            path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
            print(json.dumps(record), flush=True)
            return record

    for label, config in cases:
        run(label, config)
    successful = [r for r in payload["runs"] if r.get("status") == "ok" and r.get("config", {}).get("optimizer_backend") == "cpp"]
    ranked = sorted(successful, key=lambda r: r["official_ate"])
    combined = base
    used = set()
    baseline = next(r for r in ranked if r["label"] == "cpp_baseline")
    for record in ranked:
        if record["official_ate"] >= baseline["official_ate"]:
            continue
        changes = {k: v for k, v in record["config"].items() if v != asdict(base)[k]}
        if used & changes.keys():
            continue
        combined = replace(combined, **changes)
        used.update(changes)
        if len(used) >= 3:
            break
    if used:
        run("cpp_combined", combined)
    ranked = sorted((r for r in payload["runs"] if r.get("status") == "ok" and r.get("config", {}).get("optimizer_backend") == "cpp"), key=lambda r: r["official_ate"])
    for index, candidate in enumerate(ranked[:3]):
        run(f"python_confirm_{index}_{candidate['label']}", replace(PoseGraphConfig(**candidate["config"]), optimizer_backend="python"))
    # Native and Python rankings can reverse at bounded iteration budgets.
    # Screen the main objective/budget knobs directly with the production solver.
    direct_python = {"keyframe_stride": [16, 32], "loop_translation_weight": [.25, 1.],
        "loop_rotation_weight": [1., 2.], "odometry_translation_weight": [2.],
        "optimizer_loss": ["linear"], "optimizer_f_scale": [.1],
        "optimizer_max_nfev": [20, 100]}
    for key, values in direct_python.items():
        for value in values:
            run(f"python_direct_{key}_{value}", replace(base, optimizer_backend="python", **{key: value}))
    python = [r for r in payload["runs"] if r.get("status") == "ok"
        and r.get("config", {}).get("optimizer_backend") == "python" and not r.get("input_family")]
    python_base = replace(base, optimizer_backend="python")
    python_baseline = next(r for r in python if r["label"] == "python_baseline")
    python_combined = python_base
    used_python = set()
    for candidate in sorted(python, key=lambda r: r["official_ate"]):
        if candidate["official_ate"] >= python_baseline["official_ate"]:
            continue
        changes = {k: v for k, v in candidate["config"].items() if v != asdict(python_base)[k]}
        if used_python & changes.keys():
            continue
        python_combined = replace(python_combined, **changes)
        used_python.update(changes)
        if len(used_python) >= 3:
            break
    if len(used_python) > 1:
        candidate = run("python_combined", python_combined)
        if candidate.get("status") == "ok" and not any(r["label"] == candidate["label"] for r in python):
            python.append(candidate)
    selected = min(python, key=lambda r: r["official_ate"])
    payload["selected_python"] = selected["label"]
    payload["selection_note"] = "Lowest official ATE among measured full-trajectory Python runs. Tuned on this sequence; not held-out validation. C++ may follow a different bounded-budget convergence path."
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    if args.confirm_frontend:
        frontend_dir = args.output.parent/"scene_frontend"
        frontend = json.loads((frontend_dir/"results.json").read_text())
        validated = frontend["full_validation"]
        trajectory_path = frontend_dir/"full_validation/trajectory.txt"
        if not trajectory_path.exists() and validated["label"] == "baseline":
            trajectory_path = frontend_dir/"full_baseline/trajectory.txt"
        frontend_poses = load_trajectory(trajectory_path)
        if len(frontend_poses) != len(raw) or any(abs(a.timestamp-b.timestamp) > 1e-6
                for a, b in zip(frontend_poses, raw)):
            raise ValueError("Frontend frames do not match the verified loop's frame indices")
        combined_label = f"frontend_{validated['label']}_graph_{selected['label']}"
        run(combined_label, PoseGraphConfig(**selected["config"]), frontend_poses, validated["label"])
        payload["combined_frontend_graph"] = combined_label
        payload["combined_note"] = "Selected frontend plus selected graph, reusing the same image-verified loop measurement. Evaluated separately from fixed-input graph screening; no held-out claim."
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
