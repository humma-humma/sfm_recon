# Cleanup evaluation and reconstruction-to-splat refresh

## Current state

CPU preparation, GPU training/evaluation and final visual review are complete.
The user ran the GPU workflow from normal PowerShell because the agent session
cannot open Docker's Linux pipe. All four fresh final checkpoints were validated
for expected iteration, Gaussian count and finite parameters.

Depth cleanup is rejected. Existing boot, milk and scene presentations are
retained because refreshed runs fail the predeclared regression gate. The new
box model is recorded as its first splat baseline. The latest geometry presets
remain available independently of these presentation choices. See
[final measured results](SPLAT_REFRESH_RESULTS.md) and
[`selected_models.json`](../configs/splatting/selected_models.json).

| Target | Selected reconstruction | Cameras | New seeds | Held-out views | Training target |
|---|---|---:|---:|---:|---:|
| Stage 1 box | `stage1_box_fast` | 46 | 5,776 | 6 | 30,000 |
| Stage 2 boot | `stage2_boot_fast` | 51 | 1,257 | 7 | 30,000 |
| Stage 2 milk | `stage2_milk_fast` | 50 | 1,745 | 7 | 30,000 |
| Stage 3 scene | `stage3_scene` | 276 | 200,000 | 35 | 5,000 |

The new scene uses all 4,396 poses from the selected 0.817261 ATE solution.
The bounded graph solver still reaches its 50-evaluation limit; this is not a
claim of full convergence. The matching cached dense-LK frontend was reused.
New dense fusion uses those estimated poses, 433 sampled RGB-D frames, pixel
stride 4, depth scale 1000, depth range 0.2–5 m, and 1.2 million sampled points.
Held-out camera depths are excluded from seed construction. The exported seeds
are rebalanced toward the historical 45–55 s focus interval. Thus this is a
comparison of updated pipeline inputs, not an isolated pose-only ablation.

Object exports contain newly colored points from the selected reconstruction's
tracks. The optional `run_preset.py --rich-cloud` supplies the previously missing
colored export required by the splatting bridge.

All cameras/seeds are finite and complete. Train/test sets are disjoint. Boot,
milk and scene use the same image order and held-out image names as their earlier
exports; decoded held-out RGB pixels match exactly (maximum uint8 difference 0).
Ground-truth trajectories are used only for reported reconstruction evaluation,
not alignment, fusion or training inputs.

## Reproduce the completed GPU work

From the user's regular PowerShell:

```powershell
Set-Location 'E:\Others\Project\sfm_reconstruction'
& .\scripts\finish_splat_refresh.ps1
```

The runner mounts the repository and the Stage 3 dataset (read-only), and uses
the existing pinned Nerfstudio 1.1.5 image and cache. Work runs sequentially:

1. Evaluate the existing promoted frustum scene and all three depth candidates
   on every held-out view, recording PSNR, SSIM, LPIPS, opacity coverage, runtime
   and source/render contact sheets.
2. For boot, milk, scene and box, run independent 1,000-step fixed-pose probes,
   verify the exact held-out split and finite metrics, then train fresh final
   models. Final runs start from the seed cloud rather than the probe checkpoint.
3. Clone historical trainer settings and seed 42 for boot/milk/scene, replacing
   data/output paths and run names. Box uses the historical milk trainer recipe
   as its first splat baseline. Dense rasterization remains unchanged.
4. Evaluate each new final checkpoint. Compare boot/milk/scene with the prior
   promoted splats using matching held-out images. Box has no historical splat
   baseline and is reported accordingly.
5. Create the conservative depth candidate for the new scene using 241 training
   cameras only, then render and compare it against the uncleaned new scene.

The script preserves completed checkpoints/evaluations on rerun. It checks the
prepared input hashes and rejects a changed input manifest. Failures stop the
workflow and are recorded in `gpu_workflow.json` and individual logs. No other
project's containers are stopped or reconfigured.

## Cleanup decision

The numeric gate requires matching held-out views and limits regressions to:

- Mean PSNR: at most 0.1 dB lower; worst view: at most 0.5 dB lower.
- Mean SSIM: at most 0.002 lower; mean LPIPS: at most 0.005 higher.
- Opacity coverage retention: at least 98% on average and 95% in every view.

Opacity coverage means the fraction of pixels with accumulated opacity greater
than 0.1. It is distinct from projected Gaussian counts and mesh coverage.
Candidates failing a gate are rejected. Passing candidates are marked
`passes_numeric_gate_visual_review_pending`; they are not silently promoted.
The saved contact sheets must be reviewed for holes, floaters and lost surfaces.

## Outputs and validation

All new artifacts live under `outputs/splat_refresh_v1/`:

- `manifest.json`: presets, paths, camera/seed counts and input hashes.
- `input_validation.json`: split and held-out RGB parity checks.
- `reconstruction/`, `colmap/`: actual fresh CPU reconstruction/export artifacts.
- `cleanup/`: separately saved candidates, evidence and provenance.
- `train/`: independent probe and final checkpoints with runtime records.
- `evaluation/`: per-view metrics, rendered pairs and contact sheets.
- `*_selection.json`, `gpu_workflow.json`: cleanup gates and pipeline comparisons.
- `logs/`: complete subprocess output for diagnosing GPU failures.

The reusable cleanup module was executed locally with the native backend against
the existing scene; its saved mask is checked against the earlier loose candidate.
The full source suite passed 167 tests. The PowerShell wrapper parsed cleanly.
The newly added training/evaluation orchestration uses the pinned official
[training entry point](https://github.com/nerfstudio-project/nerfstudio/blob/v1.1.5/nerfstudio/scripts/train.py)
and [evaluation setup](https://github.com/nerfstudio-project/nerfstudio/blob/v1.1.5/nerfstudio/utils/eval_utils.py).
Its GPU execution completed successfully on the RTX 2080 with gsplat 1.4.0 and
PyTorch 2.1.2+cu118. Final decisions and visual review are recorded in
`final_selection.json`; `gpu_workflow.json` now has status `complete`.
