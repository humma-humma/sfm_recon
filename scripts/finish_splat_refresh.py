"""Run cleanup evaluation and fresh-preset splat training inside Nerfstudio."""
from __future__ import annotations

import hashlib
import json
import os
import math
from pathlib import Path
import subprocess
import sys
from time import perf_counter

from sfm_reconstruction.splat_cleanup_evaluation import compare_candidate

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs/splat_refresh_v1"
BASELINES = {
    "box": "outputs/stage2_milk_splatfacto/milk_half_fixed_pose/splatfacto/run_30000/config.yml",
    "boot": "outputs/stage2_boot_splatfacto/boot_half_fixed_pose/splatfacto/run_30000/config.yml",
    "milk": "outputs/stage2_milk_splatfacto/milk_half_fixed_pose/splatfacto/run_30000/config.yml",
    "scene": "outputs/stage3_splatfacto_stair_rebalanced/stage3_stride16_stair_rebalanced_restricted/splatfacto/run_5000/config.yml",
}


def execute(label, arguments):
    logs = OUT / "logs"
    logs.mkdir(exist_ok=True)
    print(f"\n--- {label} ---", flush=True)
    start = perf_counter()
    with (logs / f"{label}.log").open("a") as log:
        process = subprocess.Popen([sys.executable, *map(str, arguments)], cwd=ROOT,
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
            env=dict(os.environ, PYTHONUNBUFFERED="1", PYTHONPATH=str(ROOT / "src")))
        for line in process.stdout:
            print(line, end="", flush=True)
            log.write(line)
        code = process.wait()
    if code:
        raise RuntimeError(f"{label} failed with exit {code}; see {logs / (label+'.log')}")
    return perf_counter()-start


def evaluate(label, config, checkpoint=None):
    target = OUT / "evaluation" / label
    if not (target / "metrics.json").exists():
        arguments = ["scripts/evaluate_splat_checkpoint.py", "--config", config, "--output", target]
        if checkpoint:
            arguments += ["--checkpoint", checkpoint]
        execute(label, arguments)
    return json.loads((target / "metrics.json").read_text())


def compare_cleanup(label, reference, candidate):
    result = compare_candidate(reference, candidate)
    result.update(reference_gaussians=reference["gaussians"], candidate_gaussians=candidate["gaussians"])
    (OUT / f"{label}_selection.json").write_text(json.dumps(result, indent=2))
    return result


def old_cleanup():
    """Evaluate all three existing masks against the promoted frustum model."""
    import numpy as np
    import torch
    from sfm_reconstruction.stage3_gaussian_checkpoint_cleanup import prune_checkpoint
    config = ROOT / "outputs/stage3_splatfacto_stair_cleanup_frustum/stage3_stride16_stair_cleanup_frustum/splatfacto/run_5000/config.yml"
    reference = evaluate("old_scene_frustum", config)
    evidence = np.load(ROOT / "outputs/splatting_cpp/depth_evidence.npz")
    raw = ROOT / BASELINES["scene"]
    raw = raw.parent / "nerfstudio_models/step-000004999.ckpt"
    results = {}
    for index, name in enumerate(("strict", "middle", "loose")):
        target = OUT / "cleanup" / f"old_scene_{name}.ckpt"
        target.parent.mkdir(exist_ok=True)
        if not target.exists():
            checkpoint = torch.load(raw, map_location="cpu", weights_only=False)
            if len(checkpoint["pipeline"]["_model.gauss_params.means"]) != len(evidence[f"keep_{index}"]):
                raise ValueError("Saved depth masks do not match source checkpoint")
            torch.save(prune_checkpoint(checkpoint, evidence[f"keep_{index}"]), target)
        candidate = evaluate(f"old_scene_{name}", config, target)
        results[name] = compare_cleanup(f"old_scene_{name}", reference, candidate)
    return results


def main():
    import torch
    import importlib.metadata
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this workflow")
    manifest = json.loads((OUT / "manifest.json").read_text())
    if set(manifest["datasets"]) != {"box", "boot", "milk", "scene"}:
        raise ValueError("Run prepare_splat_refresh.py first")
    signature = hashlib.sha256((OUT / "manifest.json").read_bytes()).hexdigest()
    report_path = OUT / "gpu_workflow.json"
    report = json.loads(report_path.read_text()) if report_path.exists() else {
        "input_manifest_sha256": signature, "status": "running", "datasets": {},
        "torch": torch.__version__, "gsplat": importlib.metadata.version("gsplat"),
        "gpu": torch.cuda.get_device_name(), "promoted_artifacts_changed": False,
    }
    if report["input_manifest_sha256"] != signature:
        raise ValueError("Prepared inputs changed since this workflow started")

    def save():
        report_path.write_text(json.dumps(report, indent=2))

    save()
    try:
        report["old_scene_cleanup"] = old_cleanup()
        save()
        for name in ("boot", "milk", "scene", "box"):
            item = manifest["datasets"][name]
            config = ROOT / BASELINES[name]
            data = ROOT / item["colmap"]
            # Fail rather than silently mixing inputs if exported data changed.
            for key, relative in [("camera_sha256", "sparse/0/images.txt"), ("seed_sha256", "sparse/0/points3D.txt")]:
                if hashlib.sha256((data / relative).read_bytes()).hexdigest() != item[key]:
                    raise ValueError(f"{name} export changed after preparation")
            steps = 5000 if name == "scene" else 30000
            args = ["scripts/train_splat_refresh.py", "--baseline-config", config,
                    "--data", data, "--name", name]
            execute(f"{name}_probe_train", args + ["--iterations", 1000, "--probe"])
            probe = OUT / f"train/{name}/splatfacto/probe_1000/config.yml"
            probe_result = evaluate(f"{name}_probe", probe)
            expected_views = json.loads((data / "split_manifest.json").read_text())["test"]
            if [v["image"] for v in probe_result["views"]] != expected_views:
                raise ValueError(f"{name} probe uses the wrong held-out split")
            if not all(math.isfinite(v[k]) for v in probe_result["views"] for k in ("psnr", "ssim", "lpips")):
                raise ValueError(f"{name} probe produced nonfinite evaluation metrics")
            execute(f"{name}_final_train", args + ["--iterations", steps])
            final = OUT / f"train/{name}/splatfacto/run_{steps}/config.yml"
            result = evaluate(f"{name}_new", final)
            result_path = OUT / "evaluation" / f"{name}_new/metrics.json"
            entry = {"status": "trained_and_evaluated", "new_evaluation": str(result_path.relative_to(ROOT)),
                     "new_mean": result["mean"], "iterations": steps,
                     "new_gaussians": result["gaussians"], "reconstruction": item["metrics"]}
            if name != "box":
                baseline_config = config
                if name == "scene":
                    baseline_config = ROOT / "outputs/stage3_splatfacto_stair_cleanup_frustum/stage3_stride16_stair_cleanup_frustum/splatfacto/run_5000/config.yml"
                old = evaluate(f"{name}_old", baseline_config)
                # View names must agree before comparing pipeline quality.
                entry["comparison"] = compare_candidate(old, result)
                entry["old_mean"] = old["mean"]
            else:
                entry["comparison"] = {"status": "first_box_splat_no_historical_baseline"}
            report["datasets"][name] = entry
            save()
            if name == "scene":
                checkpoint = final.parent / "nerfstudio_models/step-000004999.ckpt"
                depth = OUT / "cleanup/new_scene_depth.ckpt"
                if not depth.exists():
                    execute("new_scene_depth_cleanup", ["-m", "sfm_reconstruction.stage3_gaussian_depth_cleanup",
                        "--checkpoint", checkpoint, "--dataparser-transform", final.parent / "dataparser_transforms.json",
                        "--colmap-dir", data / "sparse/0", "--dataset", "/datasets/stage3", "--output", depth])
                candidate = evaluate("new_scene_depth", final, depth)
                report["new_scene_cleanup"] = compare_cleanup("new_scene_depth", result, candidate)
                save()
        report["status"] = "gpu_complete_visual_review_pending"
        save()
    except Exception as error:
        report.update(status="failed", error=str(error))
        save()
        raise


if __name__ == "__main__":
    main()
