"""Controlled log/Jacobian investigation and native tuning on saved scene inputs."""
from dataclasses import asdict, replace
import argparse
import json
from pathlib import Path
from time import perf_counter

import numpy as np
from scipy.optimize._numdiff import approx_derivative
from scipy.spatial.transform import Rotation

from sfm_reconstruction.native_backend import pose_graph_callbacks
from sfm_reconstruction.pose_graph import (PoseGraph, PoseGraphNode, PoseGraphEdge,
    PoseGraphConfig, optimize_pose_graph, expand_keyframe_optimization, _vec6_from_matrix, _matrix_from_vec6)
from sfm_reconstruction.stage3 import load_stage3_dataset, load_trajectory, write_trajectory, _transform_from_trajectory_pose
from sfm_reconstruction.official_eval import run_official_eval

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT/"outputs/cpp_followup/graph"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--confirm-selected", action="store_true")
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    saved = np.load(ROOT/"outputs/native_benchmarks/graph_stage3_promoted/graph_input.npz")
    indices = saved["node_indices"]
    nodes = [PoseGraphNode(int(i), float(t), transform) for i,t,transform in zip(indices, saved["timestamps"], saved["transforms"])]
    edges = [PoseGraphEdge(int(p), int(q), measured, float(w[1]), float(w[0]), 456, "loop" if j == len(saved["edge_indices"])-1 else "odometry")
        for j, ((p,q), measured, w) in enumerate(zip(saved["edge_indices"], saved["measured"], saved["weights"]))]
    graph = PoseGraph(nodes, edges)
    # A small real-data subgraph permits independent dense finite differences.
    sample_edges = edges[:3]+edges[-1:]
    used = {i for e in sample_edges for i in (e.source_index,e.target_index)}
    sample_nodes = [n for n in nodes if n.index in used]
    sample = PoseGraph(sample_nodes, sample_edges)
    offsets = {n.index: 6*i for i,n in enumerate(sample_nodes[1:])}
    values = np.concatenate([_vec6_from_matrix(n.camera_to_world) for n in sample_nodes[1:]])
    fun, jac = pose_graph_callbacks(sample, offsets, len(values))
    def reference(x, stable):
        poses = {sample_nodes[0].index: sample_nodes[0].camera_to_world}
        poses.update({i:_matrix_from_vec6(x[o:o+6]) for i,o in offsets.items()})
        residuals = []
        for e in sample_edges:
            error = np.linalg.inv(e.measured_target_from_source) @ np.linalg.inv(poses[e.target_index]) @ poses[e.source_index]
            vector = Rotation.from_matrix(error[:3,:3]).as_rotvec() if stable else _vec6_from_matrix(error)[:3]
            residuals.extend(np.r_[e.rotation_weight*vector, e.translation_weight*error[:3,3]])
        return np.asarray(residuals)
    analytic = jac(values).toarray()
    stable_fd = approx_derivative(lambda x:reference(x, True), values, method="3-point", abs_step=1e-6)
    legacy_fd = approx_derivative(lambda x:reference(x, False), values, method="2-point")
    rotation_rows = np.asarray([6*i+j for i in range(len(sample_edges)) for j in range(3)])
    diagnostics = {"stable_residual_max_difference": float(np.max(np.abs(fun(values)-reference(values, True)))),
        "stable_fd_jacobian_max_difference": float(np.max(np.abs(analytic-stable_fd))),
        "opencv_fd_jacobian_max_difference": float(np.max(np.abs(analytic-legacy_fd))),
        "opencv_zero_rotation_rows": int(np.count_nonzero(np.max(np.abs(legacy_fd[rotation_rows]),axis=1) == 0)),
        "rotation_rows": len(rotation_rows)}
    print(json.dumps(diagnostics), flush=True)
    path = OUT/"results.json"
    payload = json.loads(path.read_text()) if path.exists() else {"runs": []}
    payload["derivative_diagnostics"] = diagnostics
    payload["note"] = "Identical cached verified loop and graph nodes. Stable Python log is opt-in; historical OpenCV default is preserved. Single-call solver timings. GT used only by evaluation/selection; bounded dataset-specific tuning."
    dataset = load_stage3_dataset(ROOT.parent/"Experiments/Stage_3_Data/stage3")
    for family, input_path in (("raw", ROOT/"outputs/stage3_pose_graph_full/estimated_camera_trajectory.txt"),
        ("lk_dense", ROOT/"outputs/hyperparameter_check/scene_frontend/full_validation/trajectory.txt")):
        trajectory = load_trajectory(input_path)
        family_nodes = [replace(n, camera_to_world=_transform_from_trajectory_pose(trajectory[n.index])) for n in nodes]
        family_poses = {n.index:n.camera_to_world for n in family_nodes}
        family_edges = edges if family == "raw" else [replace(e,
            measured_target_from_source=np.linalg.inv(family_poses[e.target_index])@family_poses[e.source_index])
            if e.edge_type == "odometry" else e for e in edges]
        base = PoseGraphConfig(keyframe_stride=8, optimizer_max_nfev=50, optimizer_f_scale=.1)
        cases = [("python_stable", replace(base, python_rotation_log="stable")),
                 ("cpp_analytic", replace(base, optimizer_backend="cpp"))]
        if family == "lk_dense":
            for field, options in {"optimizer_f_scale":[.01,.05,.5,1.],
                    "odometry_rotation_weight":[.1,.25,.5,2.], "optimizer_max_nfev":[100]}.items():
                cases.extend((f"cpp_{field}_{value}", replace(base, optimizer_backend="cpp", **{field:value})) for value in options)
        for label, config in cases:
            label = f"{family}_{label}"
            if any(r["label"] == label for r in payload["runs"]):
                continue
            print(label, flush=True)
            # Edge weights change the objective; all measured transforms stay fixed.
            weighted = PoseGraph(family_nodes, [replace(e,
                rotation_weight=config.loop_rotation_weight if e.edge_type == "loop" else config.odometry_rotation_weight,
                translation_weight=config.loop_translation_weight if e.edge_type == "loop" else config.odometry_translation_weight) for e in family_edges])
            start = perf_counter()
            result = optimize_pose_graph(weighted, config)
            seconds = perf_counter()-start
            dest = OUT/label
            dest.mkdir(exist_ok=True)
            export = dest/"trajectory.txt"
            write_trajectory(export, expand_keyframe_optimization(trajectory, result.poses))
            evaluation = run_official_eval(dataset.ground_truth_trajectory_path, export, dest, label=label, evaluator_dir=dataset.root)
            if evaluation.returncode or evaluation.rmse is None:
                raise RuntimeError(f"Official evaluator failed: {dest}")
            record = {"label":label, "config":asdict(config), "seconds":seconds,
                "official_ate":evaluation.rmse, "nfev":result.iterations, "converged":result.success,
                "final_cost":result.final_cost, "message":result.message}
            payload["runs"].append(record)
            path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
            print(json.dumps({k:v for k,v in record.items() if k != "config"}), flush=True)
    if args.confirm_selected:
        selected = min((r for r in payload["runs"] if r["label"].startswith("lk_dense_cpp_")), key=lambda r:r["official_ate"])
        selected_config = replace(PoseGraphConfig(**selected["config"]), python_rotation_log="stable")
        for label, backend in [(f"selected_cpp_repeat{i}", "cpp") for i in range(1,4)]+[("selected_python_stable", "python")]:
            if any(r["label"] == label for r in payload["runs"]):
                continue
            config = replace(selected_config, optimizer_backend=backend)
            weighted = PoseGraph(family_nodes, [replace(e,
                rotation_weight=config.loop_rotation_weight if e.edge_type == "loop" else config.odometry_rotation_weight,
                translation_weight=config.loop_translation_weight if e.edge_type == "loop" else config.odometry_translation_weight) for e in family_edges])
            print(label, flush=True)
            start = perf_counter()
            result = optimize_pose_graph(weighted, config)
            seconds = perf_counter()-start
            dest = OUT/label
            dest.mkdir(exist_ok=True)
            export = dest/"trajectory.txt"
            write_trajectory(export, expand_keyframe_optimization(trajectory, result.poses))
            evaluation = run_official_eval(dataset.ground_truth_trajectory_path, export, dest, label=label, evaluator_dir=dataset.root)
            if evaluation.returncode or evaluation.rmse is None:
                raise RuntimeError(f"Official evaluator failed: {dest}")
            record = {"label":label, "config":asdict(config), "seconds":seconds,
                "official_ate":evaluation.rmse, "nfev":result.iterations, "converged":result.success,
                "final_cost":result.final_cost, "message":result.message}
            payload["runs"].append(record)
            path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
            print(json.dumps({k:v for k,v in record.items() if k != "config"}), flush=True)
        confirmed = [r for r in payload["runs"] if r["label"].startswith("selected_cpp_repeat")]
        reference = next(r for r in payload["runs"] if r["label"] == "selected_python_stable")
        payload["selected_native"] = selected["label"]
        payload["stable_reference_ate_agreement"] = max(abs(r["official_ate"]-reference["official_ate"]) for r in confirmed)
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
