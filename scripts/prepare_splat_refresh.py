"""Prepare new splat inputs from selected presets, without using GT alignment."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
from dataclasses import replace

from sfm_reconstruction.gaussian_splatting_export import export_colmap_dataset
from sfm_reconstruction.stage3 import load_stage3_dataset, load_trajectory
from sfm_reconstruction.stage3_gaussian_splatting_export import export_stage3_colmap_dataset, _select_keyframe_indices
from sfm_reconstruction.stage3_open3d_viewer import build_stage3_scene_points, write_stage3_scene_ply
from sfm_reconstruction.stage3_gaussian_seed_rebalance import rebalance_colmap_seed_cloud
from sfm_reconstruction.stage3_splat_temporal_diagnostics import _read_colmap

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs/splat_refresh_v1"


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    manifest_path = OUT / "manifest.json"
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {
        "status": "preparing", "uses_gt_alignment": False, "datasets": {}}
    for name, preset in [("box", "stage1_box_fast"), ("boot", "stage2_boot_fast"),
                         ("milk", "stage2_milk_fast"), ("scene", "stage3_scene")]:
        preset_path = ROOT / f"configs/presets/{preset}.json"
        if name in manifest["datasets"]:
            assert manifest["datasets"][name]["preset_sha256"] == digest(preset_path)
            print(f"Already prepared: {name}", flush=True)
            continue
        recipe = json.loads(preset_path.read_text())
        reconstruction = OUT / "reconstruction" / name
        command = [sys.executable, "scripts/run_preset.py", "--preset", str(preset_path),
                   "--output", str(reconstruction)]
        if name == "scene":
            command += ["--trajectory", "outputs/hyperparameter_check/scene_frontend/full_validation/trajectory.txt"]
        else:
            command += ["--rich-cloud"]
        subprocess.run(command, cwd=ROOT, check=True, env=dict(os.environ, PYTHONPATH=str(ROOT / "src")))
        export = OUT / "colmap" / name
        metadata = {}
        if name != "scene":
            export_colmap_dataset(ROOT / recipe["dataset"], reconstruction, export, image_scale=.5)
        else:
            dataset = load_stage3_dataset(ROOT / recipe["dataset"])
            trajectory = load_trajectory(reconstruction / "estimated_camera_trajectory.txt")
            selected = _select_keyframe_indices(len(trajectory), 16)
            held_out = {f"{trajectory[i].timestamp:.9f}" for i in selected[::8]}
            # Keep the historical dense sampling budget, excluding evaluation
            # camera depths from seed construction. No GT poses are loaded.
            frames = [f for i, f in enumerate(dataset.frames)
                      if i % 10 == 0 and f"{f.timestamp:.9f}" not in held_out]
            points, colors = build_stage3_scene_points(replace(dataset, frames=frames), trajectory,
                frame_stride=1, pixel_stride=4, max_points=1_200_000)
            cloud = reconstruction / "scene_dense.ply"
            write_stage3_scene_ply(cloud, points, colors)
            base_export = OUT / "colmap" / "scene_uniform"
            export_stage3_colmap_dataset(dataset.root, reconstruction / "estimated_camera_trajectory.txt",
                cloud, base_export, frame_stride=16, image_scale=.5, max_points=200_000)
            rebalance_colmap_seed_cloud(base_export, cloud, export,
                focus_start=45., focus_end=55., backend="cpp", num_threads=8)
            metadata = {"dense_points": len(points), "dense_frames": len(frames),
                        "held_out_depths_excluded_from_seeds": len(held_out),
                        "dense_cloud_sha256": digest(cloud)}
        k, poses, seeds = _read_colmap(export / "sparse/0")
        image_names = [name for name, _, _ in poses]
        split = {"train": [n for i, n in enumerate(image_names) if i % 8 != 0],
                 "test": image_names[::8]}
        (export / "split_manifest.json").write_text(json.dumps(split, indent=2))
        manifest["datasets"][name] = {
            "preset": preset_path.relative_to(ROOT).as_posix(), "preset_sha256": digest(preset_path),
            "reconstruction": reconstruction.relative_to(ROOT).as_posix(),
            "colmap": export.relative_to(ROOT).as_posix(), "cameras": len(poses), "seeds": len(seeds),
            "resolution": list(k[4:]), "train_views": len(split["train"]), "test_views": len(split["test"]),
            "camera_sha256": digest(export / "sparse/0/images.txt"),
            "seed_sha256": digest(export / "sparse/0/points3D.txt"),
            "metrics": json.loads((reconstruction / "run.json").read_text()), **metadata,
        }
        manifest_path.write_text(json.dumps(manifest, indent=2))
        print(f"Prepared {name}: {len(poses)} cameras, {len(seeds)} seeds", flush=True)
    manifest["status"] = "cpu_inputs_ready_gpu_pending"
    manifest_path.write_text(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
