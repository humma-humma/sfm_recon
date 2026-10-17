"""Opt-in native support and depth-cleanup experiments on saved checkpoints."""
from __future__ import annotations

import json
import argparse
from pathlib import Path
from time import perf_counter

import cv2
import numpy as np
import torch

from sfm_reconstruction.native_backend import load_native
from sfm_reconstruction.stage3_gaussian_seed_rebalance import focus_visibility_counts
from sfm_reconstruction.stage3_splat_temporal_diagnostics import _read_colmap
from sfm_reconstruction.stage3_gaussian_checkpoint_cleanup import prune_checkpoint
from sfm_reconstruction.stage3 import load_stage3_dataset

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs/splatting_cpp"


def timed(call, repeats=3):
    samples = []
    for _ in range(repeats):
        start = perf_counter()
        result = call()
        samples.append(perf_counter() - start)
    return result, {"seconds": samples, "median_seconds": float(np.median(samples))}


def depth_reference(points, rotation, translation, k, depth, absolute, relative):
    v = points @ rotation.T + translation
    valid = np.isfinite(v).all(axis=1) & (v[:, 2] > 1e-6)
    ids = np.flatnonzero(valid)
    xy = v[ids, :2] / v[ids, 2:3] * k[:2] + k[2:]
    inside = (xy[:, 0] >= 0) & (xy[:, 0] < depth.shape[1]) & (xy[:, 1] >= 0) & (xy[:, 1] < depth.shape[0])
    ids, xy = ids[inside], xy[inside].astype(np.int64)
    measured = depth[xy[:, 1], xy[:, 0]]
    valid = np.isfinite(measured) & (measured > 0)
    ids, measured = ids[valid], measured[valid]
    delta = v[ids, 2] - measured
    out = np.zeros(len(points), np.uint8)
    out[ids] = np.where(np.abs(delta) <= absolute + relative * measured, 1,
                        np.where(delta < 0, 2, 3))
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--depth-only", action="store_true")
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    native = load_native()
    report = {"uses_ground_truth": False, "support": {}, "gpu": {
        "status": "blocked", "reason": "Docker Linux pipe access denied; local gsplat absent; pip reports no matching distribution",
        "torch": torch.__version__, "cuda_available": torch.cuda.is_available(),
    }}
    if args.depth_only:
        report = json.loads((OUT / "results.json").read_text())
    scene_dir = ROOT / "outputs/stage3_splatfacto_stair_rebalanced/stage3_stride16_stair_rebalanced_restricted/splatfacto/run_5000"
    cases = {
        "boot": ("stage2_boot_splatfacto/boot_half_fixed_pose/splatfacto/run_30000", "stage2_boot_gaussian_colmap_half", "step-000029999.ckpt"),
        "milk": ("stage2_milk_splatfacto/milk_half_fixed_pose/splatfacto/run_30000", "stage2_milk_gaussian_colmap_half", "step-000029999.ckpt"),
        "scene": (str(scene_dir.relative_to(ROOT / "outputs")), "stage3_gaussian_colmap_stride16_half_stair_rebalanced", "step-000004999.ckpt"),
    }
    for name, (run, export, filename) in cases.items():
        if args.depth_only and name != "scene":
            continue
        directory = ROOT / "outputs" / run
        checkpoint = torch.load(directory / "nerfstudio_models" / filename, map_location="cpu", weights_only=False)
        means = checkpoint["pipeline"]["_model.gauss_params.means"].numpy()
        record = json.loads((directory / "dataparser_transforms.json").read_text())
        transform = np.asarray(record["transform"])
        points = np.ascontiguousarray((means / record["scale"] - transform[:, 3]) @ transform[:, :3])
        k, records, _ = _read_colmap(ROOT / "outputs" / export / "sparse/0")
        poses = [(r, t) for _, r, t in records]
        if args.depth_only:
            baseline = focus_visibility_counts(points, poses, k, backend="cpp", num_threads=8)
        else:
            baseline, timing = timed(lambda: focus_visibility_counts(points, poses, k))
            item = {"gaussians": len(points), "cameras": len(poses), "python": timing}
            for threads in [1, 2, 4, 8]:
                actual, timing = timed(lambda: focus_visibility_counts(points, poses, k, backend="cpp", num_threads=threads))
                mismatch = int(np.count_nonzero(actual != baseline))
                assert mismatch == 0, (name, threads, mismatch)
                item[f"cpp_{threads}"] = dict(timing, mismatches=mismatch,
                    speedup=item["python"]["median_seconds"] / timing["median_seconds"])
            report["support"][name] = item
            (OUT / "results.json").write_text(json.dumps(report, indent=2))
            print(name, json.dumps(item), flush=True)
        if name != "scene":
            del checkpoint
            continue

        dataset = load_stage3_dataset(ROOT.parent / "Experiments/Stage_3_Data/stage3")
        frames = {f"{f.timestamp:.9f}": f for f in dataset.frames}
        tolerances = [(0.02, 0.01), (0.05, 0.02), (0.10, 0.05)]
        surface = np.zeros((3, len(points)), dtype=np.uint32)
        conflict = np.zeros_like(surface)
        occluded = np.zeros_like(surface)
        start = perf_counter()
        # Match the saved interval-8 split. Held-out depth must not participate
        # in choosing a cleanup mask that will later be evaluated on that view.
        training_records = [record for i, record in enumerate(records) if i % 8 != 0]
        for frame_index, (image, r, t) in enumerate(training_records):
            frame = frames[f"{float(Path(image).stem):.9f}"]
            raw = cv2.imread(str(frame.depth_path), cv2.IMREAD_UNCHANGED)
            if raw is None:
                raise ValueError(f"Cannot read depth: {frame.depth_path}")
            # Native RGB/depth images use the same pixel grid. Project into the
            # original depth resolution to avoid interpolation across edges.
            rgb = cv2.imread(str(frame.rgb_path)) if frame_index == 0 else None
            if rgb is not None and raw.shape != rgb.shape[:2]:
                raise ValueError("Depth is not aligned to RGB; calibration required")
            kd = np.array(k[:4]) * np.array([raw.shape[1]/k[4], raw.shape[0]/k[5]]*2)
            depth = raw.astype(np.float64) / 1000.
            depth[(depth < .1) | (depth > 8.)] = 0
            for j, (absolute, relative) in enumerate(tolerances):
                labels = native.depth_support(points, r, t, kd, depth, absolute, relative)
                if frame_index in [0, len(training_records)//2, len(training_records)-1]:
                    sample = np.arange(0, len(points), 97)
                    np.testing.assert_array_equal(labels[sample], depth_reference(points[sample], r, t, kd, depth, absolute, relative))
                surface[j] += labels == 1
                conflict[j] += labels == 2
                occluded[j] += labels == 3
            if frame_index % 40 == 0:
                print(f"depth {frame_index}/{len(training_records)}", flush=True)
        report["depth"] = {"seconds": perf_counter()-start, "depth_scale": 1000.,
            "depth_range_m": [.1, 8.], "cameras": len(training_records), "candidates": [],
            "held_out_cameras_excluded": len(records)-len(training_records),
            "rule": "keep original frustum mask; remove only >=3 free-space conflicts and zero surface agreements",
            "quality_status": "evidence proxy only; held-out renders unavailable; not promoted"}
        frustum = baseline >= 1
        masks = {}
        for j, (absolute, relative) in enumerate(tolerances):
            keep = frustum & ~((conflict[j] >= 3) & (surface[j] == 0))
            masks[f"keep_{j}"] = keep
            report["depth"]["candidates"].append({
                "absolute_tolerance_m": absolute, "relative_tolerance": relative,
                "kept": int(keep.sum()), "additional_removed": int(np.count_nonzero(frustum & ~keep)),
                "surface_supported": int(np.count_nonzero(surface[j] > 0)),
                "no_valid_evidence": int(np.count_nonzero((surface[j]+conflict[j]+occluded[j]) == 0)),
                "removed_fraction_of_frustum": float(np.count_nonzero(frustum & ~keep)/frustum.sum()),
            })
            # A support count is not rendered image coverage. Report this only
            # as a diagnostic of how aggressively the mask changes the scene.
            ratios = []
            for _, r, t in records[::8]:
                before = focus_visibility_counts(points[frustum], [(r, t)], k, backend="cpp", num_threads=8).sum()
                after = focus_visibility_counts(points[keep], [(r, t)], k, backend="cpp", num_threads=8).sum()
                ratios.append(float(after/before) if before else 1.)
            report["depth"]["candidates"][-1]["held_out_projected_count_retention"] = {
                "minimum": min(ratios), "median": float(np.median(ratios)),
                "warning": "point-count proxy, not pixel coverage or image quality",
            }
        np.savez_compressed(OUT / "depth_evidence.npz", surface=surface, conflict=conflict,
                            occluded=occluded, frustum=frustum, **masks)
        # Save the loosest candidate for later held-out rendering, preserving
        # original Gaussian order and matching optimizer tensors.
        clean = prune_checkpoint(checkpoint, masks["keep_2"])
        torch.save(clean, OUT / "scene_depth_conservative.ckpt")
        report["depth"]["checkpoint"] = "scene_depth_conservative.ckpt"
        (OUT / "results.json").write_text(json.dumps(report, indent=2))
        print(json.dumps(report["depth"], indent=2), flush=True)


if __name__ == "__main__":
    main()
