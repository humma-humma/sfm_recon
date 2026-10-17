"""Save reviewable dataset-specific recipes without changing global defaults."""
from dataclasses import asdict
import json
from pathlib import Path

import numpy as np

from sfm_reconstruction.reconstruction import ReconstructionConfig
from sfm_reconstruction.pose_graph import PoseGraphConfig

ROOT = Path(__file__).resolve().parents[1]


def main():
    destination = ROOT/"configs/presets"
    destination.mkdir(parents=True, exist_ok=True)
    for name in ("box", "boot", "milk"):
        results = json.loads((ROOT/f"outputs/hyperparameter_check/{name}/results.json").read_text())
        selected = next(r for r in results["runs"] if r["label"] == results["selected"])
        variants = [(f"stage1_{name}" if name == "box" else f"stage2_{name}", selected)]
        if name == "milk":
            variants.append(("stage2_milk_coverage", next(r for r in results["runs"] if r["label"] == "min_track_observations_2")))
        for label, run in variants:
            dataset = "../Experiments/Stage_1_Data_ver._4/Stage_1_Data_ver_4/stage1/box" if name == "box" else f"../Experiments/Stage_2_Data/stage2/{name}"
            cache = None if name == "box" else f"outputs/stage2_{name}_{'improved' if name == 'boot' else 'masked_track3_ba'}/matching_cache/correspondences"
            payload = {"name":label, "dataset":dataset, "correspondences":cache,
                "reconstruction":asdict(ReconstructionConfig(**run["config"])),
                "reference_case":run["label"], "expected_metrics":{k:run[k] for k in
                    ("cameras", "points", "pose_metrics", "reprojection_rmse_px", "image_coverage", "point_metrics", "mesh_proxy", "mesh_coverage") if k in run},
                "note":"Dataset-specific exploratory tuning. Paths are relative to the repository root. Cached correspondence family is part of the recipe. Boot metrics measure consistency only; no full GT. Defaults are unchanged."}
            (destination/f"{label}.json").write_text(json.dumps(payload, indent=2)+"\n", encoding="utf-8")
            options_path = ROOT/f"outputs/cpp_followup/ceres_options/{name}.json"
            if label != "stage2_milk_coverage" and options_path.exists():
                options = json.loads(options_path.read_text())
                confirmation = options.get("pipeline_confirmation", {})
                if confirmation.get("equivalent"):
                    chosen = next(r for r in options["runs"] if r["label"] == confirmation["option"])
                    fast = {**payload, "name":f"{label}_fast", "reconstruction":{**payload["reconstruction"],
                        "ceres_num_threads":chosen["threads"], "ceres_linear_solver":chosen["solver"]},
                        "performance_confirmation":confirmation}
                    (destination/f"{label}_fast.json").write_text(json.dumps(fast, indent=2)+"\n", encoding="utf-8")
    frontend = json.loads((ROOT/"outputs/hyperparameter_check/scene_frontend/results.json").read_text())
    graph = json.loads((ROOT/"outputs/hyperparameter_check/scene_graph/results.json").read_text())
    selected = min((r for r in graph["runs"] if r.get("input_family") == "lk_dense"), key=lambda r:r["official_ate"])
    saved = np.load(ROOT/"outputs/native_benchmarks/graph_stage3_promoted/graph_input.npz")
    target, source = map(int, saved["edge_indices"][-1])
    payload = {"name":"stage3_scene", "dataset":"../Experiments/Stage_3_Data/stage3",
        "odometry":frontend["full_validation"]["config"],
        "pose_graph":asdict(PoseGraphConfig(**selected["config"])),
        "verified_loop":{"source_index":source, "target_index":target,
            "source_from_target":saved["measured"][-1].tolist(), "method":"pnp", "inliers":456,
            "median_reprojection_error":.499778414},
        "expected_metrics":{"poses":4396, "tracked_transitions":frontend["full_validation"]["tracked_frames"], "official_ate":selected["official_ate"]},
        "note":"Dataset-specific joint frontend/graph recipe. Python OpenCV log is explicit to preserve measured behavior. Reuses the image-verified loop for these exact frame indices; not a generic loop-discovery preset or held-out validation."}
    (destination/"stage3_scene_python_legacy.json").write_text(json.dumps({**payload,
        "name":"stage3_scene_python_legacy"}, indent=2)+"\n", encoding="utf-8")
    followup_path = ROOT/"outputs/cpp_followup/graph/results.json"
    if followup_path.exists():
        followup = json.loads(followup_path.read_text())
        if followup.get("stable_reference_ate_agreement", float("inf")) < .001:
            confirmed = next(r for r in followup["runs"] if r["label"] == "selected_cpp_repeat1")
            stable = next(r for r in followup["runs"] if r["label"] == "selected_python_stable")
            payload = {**payload, "pose_graph":confirmed["config"],
                "expected_metrics":{**payload["expected_metrics"], "official_ate":confirmed["official_ate"]},
                "note":"Dataset-specific joint frontend/native graph recipe, validated with three native repeats and independent stable Python finite differences. Same verified loop and exact frame indices. Global defaults unchanged; no held-out claim."}
            fallback = {**payload, "name":"stage3_scene_python_stable", "pose_graph":stable["config"],
                "expected_metrics":{**payload["expected_metrics"], "official_ate":stable["official_ate"]}}
            (destination/"stage3_scene_python_stable.json").write_text(json.dumps(fallback, indent=2)+"\n", encoding="utf-8")
    (destination/"stage3_scene.json").write_text(json.dumps(payload, indent=2)+"\n", encoding="utf-8")
    print(f"Saved {len(list(destination.glob('*.json')))} presets to {destination}")


if __name__ == "__main__":
    main()
