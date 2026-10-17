"""Validate completed GPU artifacts and record conservative presentation choices."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import torch

from sfm_reconstruction.splat_cleanup_evaluation import compare_candidate

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs/splat_refresh_v1"


def read(path):
    return json.loads(path.read_text())


def main():
    report = read(OUT / "gpu_workflow.json")
    manifest = read(OUT / "manifest.json")
    if report["status"] not in {"gpu_complete_visual_review_pending", "complete"}:
        raise ValueError("GPU workflow has not completed")
    assert hashlib.sha256((OUT / "manifest.json").read_bytes()).hexdigest() == report["input_manifest_sha256"]
    decisions = {"status": "complete", "depth_cleanup": "rejected",
                 "promotion_policy": "retain existing presentations when the predeclared per-view gate fails",
                 "geometry_and_presentation_are_selected_separately": True,
                 "datasets": {}}
    historical = {
        "boot": "outputs/stage2_boot_splatfacto/boot_half_fixed_pose/splatfacto/run_30000",
        "milk": "outputs/stage2_milk_splatfacto/milk_half_fixed_pose/splatfacto/run_30000",
        "scene": "outputs/stage3_splatfacto_stair_cleanup_frustum/stage3_stride16_stair_cleanup_frustum/splatfacto/run_5000",
    }
    for name in ("box", "boot", "milk", "scene"):
        steps = report["datasets"][name]["iterations"]
        fresh = f"outputs/splat_refresh_v1/train/{name}/splatfacto/run_{steps}"
        checkpoint = ROOT / fresh / "nerfstudio_models" / f"step-{steps-1:09d}.ckpt"
        state = torch.load(checkpoint, map_location="cpu", weights_only=False)
        assert state["step"] == steps-1
        params = {k: v for k, v in state["pipeline"].items() if k.startswith("_model.gauss_params.")}
        assert len(params["_model.gauss_params.means"]) == report["datasets"][name]["new_gaussians"]
        assert all(torch.isfinite(v).all() for v in params.values())
        del state, params
        selected = historical.get(name, fresh)
        if name != "box":
            result = compare_candidate(read(OUT / f"evaluation/{name}_old/metrics.json"),
                                       read(OUT / f"evaluation/{name}_new/metrics.json"))
            assert result == report["datasets"][name]["comparison"]
            assert result["status"] == "rejected", "Review changed results before choosing a model"
        decisions["datasets"][name] = {
            "geometry_preset": manifest["datasets"][name]["preset"],
            "fresh_geometry": manifest["datasets"][name]["reconstruction"],
            "fresh_trained_model": fresh,
            "presentation_config": selected + "/config.yml",
            "presentation_checkpoint": selected + f"/nerfstudio_models/step-{steps-1:09d}.ckpt",
            "decision": "first_box_baseline" if name == "box" else "retain_existing_presentation",
            "reason": "No historical box splat; reviewed as a first baseline" if name == "box" else
                      "Refreshed model failed the predeclared gate; preserve per-view rendering quality",
        }
    for name, result in report["old_scene_cleanup"].items():
        actual = compare_candidate(read(OUT / "evaluation/old_scene_frustum/metrics.json"),
                                   read(OUT / f"evaluation/old_scene_{name}/metrics.json"))
        assert actual["status"] == result["status"] == "rejected"
    actual = compare_candidate(read(OUT / "evaluation/scene_new/metrics.json"),
                               read(OUT / "evaluation/new_scene_depth/metrics.json"))
    assert actual["status"] == report["new_scene_cleanup"]["status"] == "rejected"
    decisions["visual_review"] = {
        "scope": "sampled contact sheets plus worst-regression frames, not every novel view",
        "reviewed": ["scene_old", "scene_new", "new_scene_depth", "milk_new", "boot_new", "box_new",
                     "boot_old/000_img.png", "milk_old/001_img.png", "scene_old/012_img.png",
                     "scene_new/012_img.png", "old_scene_loose/012_img.png"],
        "findings": "Refreshed scene sharpens the lab signage but worsens the 51.501833 s staircase view; milk 00226 becomes washed out. Depth cleanup damages surface detail. Box reconstructs the carton with remaining background artifacts.",
        "opacity_coverage_limitation": "Coverage is near 100% even on visibly poor renders; it cannot establish surface quality here.",
    }
    (OUT / "final_selection.json").write_text(json.dumps(decisions, indent=2))
    config_path = ROOT / "configs/splatting/selected_models.json"
    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_text(json.dumps(decisions, indent=2))
    report["status"] = "complete"
    report["final_selection"] = "outputs/splat_refresh_v1/final_selection.json"
    report["visual_review"] = decisions["visual_review"]
    (OUT / "gpu_workflow.json").write_text(json.dumps(report, indent=2))
    print("Validated four finite final checkpoints; recorded presentation selections and rejected depth cleanup.")


if __name__ == "__main__":
    main()
