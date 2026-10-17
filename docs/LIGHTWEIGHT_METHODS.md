# Actual GNT and 2DGS trials

The requested two-method experiment uses external official implementations:

| Setup | Why this setup fits here | Pilot settings |
| --- | --- | --- |
| [GNT](https://github.com/VITA-Group/GNT) | One 117.5 MB checkpoint; no per-scene training or new CUDA extension | Authors' generalization model, 8-layer transformer, 8 nearest eligible source views, 64 inverse-depth samples; no fine-tuning |
| [2DGS](https://github.com/hbb1/2d-gaussian-splatting) | Existing COLMAP cameras/seed points; no downloaded foundation model | Official surfel rasterizer and optimizer; 10k steps, distortion weight 1000 for objects from step 3001 and 0 for scene, normal weight .05 from step 7001; full 960×540 RGB |

These are lightweight integration choices among the investigated approaches, not a proof that no lighter research method exists. GNT's dependency setup is small, but its ray transformer is slow at full resolution. 2DGS requires compiling two extensions; this has succeeded with CUDA 11.8 for the RTX 2080 (sm75).

The Docker image is the existing `ghcr.io/nerfstudio-project/nerfstudio:1.1.5`. No other project's containers are stopped or reconfigured. Models, downloaded code, compiler packages and lossless renders stay under `%TEMP%\sfm_lightweight_methods_v1` on C:, because the project drive is nearly full. Persistent outputs contain small JPEG comparisons and JSON reports.

## Scope and controls

- One representative held-out view for each object, plus one outdoor and one indoor scene view, at 960×540. These five views are a limited pilot.
- Additional 2DGS comparisons cover both 1× and 1.7× focal framing of each object anchor and six scene cameras (18 held-out/focal views total). They are shown separately from the common five-view comparison; GNT was not rendered for these extra views.
- Three motion positions per dataset at 480×270. These have no reference photographs and cannot establish temporal stability.
- Controls are the original splats and the user's previously preferred restoration: prior on original for objects, surface/photo plus prior for scene.
- Every eighth RGB target is excluded from source/training photographs. Camera estimation and sparse seed geometry predate this split, so this is not a strict end-to-end held-out reconstruction benchmark.
- Source camera text and seed files are mounted read-only; the adapter verifies their hashes after 2DGS training. No selected presentation model is replaced.
- Quality scores describe the delivered JPEG90 images, including their compression, at matching resolution. Higher PSNR/SSIM and lower AlexNet LPIPS indicate agreement with the held-out photograph, not guaranteed geometric correctness.
- The 2DGS trial has 10k steps rather than the standard 30k training budget. Failure here would not establish that a fully tuned/converged method cannot help. Object distortion weight 1000 follows the authors' DTU evaluation recipe; scene uses the upstream general-scene default 0. These dataset-class choices are not a claim of exact paper-default reproduction. The adapter exposes `--lambda-dist` for explicit follow-up experiments.
- Upstream densification/pruning continues until step 15k, so this short trial stops during that phase. Opacity resets and pruning can remove weakly supported surfaces before they finish recovering. A regression is evidence about this pilot setting, not a verdict on 2DGS in general.
- The initial strong-penalty boot run regressed, and milk collapsed with large black regions. A bounded follow-up reruns those two at upstream default distortion weight 0, with all other settings unchanged. Both runs are retained in separate directories and in the comparison. This is exploratory tuning after observing failures, not a preregistered benchmark. The runner reproduces these two extra checks.
- GNT is evaluated zero-shot. Depth bounds come from visible sparse points; source selection and scene/domain mismatch can limit its performance. Learned rendering cannot guarantee recovery of unseen fine detail.
- GPU times are observational and include contention from the existing stack and, during the beginning of the box trial, GNT. Peak allocated memory excludes other GPU processes and CUDA allocator reservation.
- Native 2DGS CUDA render times do not predict browser FPS. The comparison page publishes images; the traversable demo retains its selected models. Deploying surfels interactively needs a compatible viewer/export path after visual selection.
- Official 2DGS assumes a centered principal point. The scene input differs from center by approximately (0.067, 2.424) pixels, uncorrected in this pilot. This can bias scene pixel metrics; it does not account for the large visible indoor blur. Object calibration is centered. GNT and the original controls use the supplied principal points.

## Reproduce

The fixed-camera inputs in `%TEMP%\sfm_sharpness_trials_v1\render_manifest.json` and the repository's exported COLMAP datasets must exist. Run in PowerShell:

```powershell
& 'E:\Others\Project\sfm_reconstruction\scripts\run_lightweight_methods.ps1' -Stage Setup
& 'E:\Others\Project\sfm_reconstruction\scripts\run_lightweight_methods.ps1' -Stage GNT
& 'E:\Others\Project\sfm_reconstruction\scripts\run_lightweight_methods.ps1' -Stage 2DGS
& 'E:\Others\Project\sfm_reconstruction\scripts\run_lightweight_methods.ps1' -Stage Report
```

Use `-Datasets boot` to rerun a single dataset; reporting requires all four completed datasets. `-Stage All` performs setup and both methods sequentially. Setup pins the external repositories and compiler packages; it does not modify the upstream optimizer or networks. Official external code and checkpoint remain under their upstream licenses and are not vendored into this project's MIT source.

Exact commits, checkpoint SHA256, source-file hashes, settings, per-frame times, memory and scores are written to `outputs/lightweight_methods/results.json`. The visual comparison is `web_demo/lightweight.html`, with shared crop, zoom, pan and two independently selectable methods. Larger lossless renders and trained PLYs remain in C: temporary storage for follow-up work.

## Completed results

All four datasets were run through both actual implementations. The two additional 2DGS default-weight checks also completed. These are source-resolution JPEG comparisons against the original splats:

| Dataset | GNT PSNR change on common views | GNT LPIPS change (negative is better) | 2DGS PSNR change on its wider camera/zoom set | 2DGS LPIPS change | Decision |
| --- | ---: | ---: | ---: | ---: | --- |
| Box | −0.20 dB | −0.0015 | +1.90 dB | −0.0356 | Keep preferred control; 2DGS rejected after visual review |
| Boot | +0.17 dB | +0.0521 | −0.36 dB | +0.0298 | Keep preferred control; neither method gives a clear overall gain |
| Milk | −0.85 dB | +0.0705 | −0.64 dB | +0.0826 | Keep preferred control |
| Scene | +3.28 dB | −0.2605 | −1.88 dB | +0.0634 | GNT improves indoors; outdoors approximately on par with preferred control; 2DGS rejected |

GNT uses one common held-out camera per object and two for scene. Wider 2DGS scores use one object camera at two focal settings, or six scene cameras at two focal settings. The table uses 2DGS distortion weight 1000 for box and 0 for boot, milk and scene. These unequal evaluation scopes must not be combined into a single model ranking.

The initial boot/milk weight-1000 experiments are retained in the page. Boot loses table coverage; milk collapses with large black regions. Weight 0 restores coverage but does not beat the original/preferred rendering overall. There is no basis here for claiming that 2DGS cannot improve with different initialization, pruning or a longer training budget.

On scene's two common cameras, GNT also beats the previously preferred restoration: mean PSNR 20.93 versus 18.25 dB, and LPIPS 0.3985 versus 0.4925. User visual review finds the indoor structure substantially improved, though still smeared. Outdoors GNT is approximately on par with the preferred restoration and may lose some detail. This does not establish equivalent gains across the full trajectory.

| Dataset | Initial 2DGS training | Default-weight check | GNT 960×540 frame | 2DGS median native CUDA render used in wider comparison |
| --- | ---: | ---: | ---: | ---: |
| Box | 9.9 min | — | 103.1 s | 37.4 ms |
| Boot | 6.5 min | 7.2 min | 101.0 s | 40.4 ms |
| Milk | 6.0 min | 7.2 min | 103.2 s | 44.4 ms |
| Scene | 8.4 min | — | 102.4 s | 42.9 ms |

GNT peaks at 2284 MiB allocated. The initial 2DGS trials peak at 899–1250 MiB; the default-weight checks peak at 839 MiB (boot) and 925 MiB (milk). The six 2DGS training runs total about 45.1 minutes; the five full-resolution and twelve half-resolution GNT renders total about 13.7 minutes of measured inference, excluding setup and the interrupted contention test. These are shared-GPU observations.

Final visual decision: 2DGS is rejected for all four datasets; the trial renders are unusable for this project despite some favorable box metrics. No further 2DGS tuning or integration is planned. Retain its results as an experiment archive. GNT is an offline indoor scene-rendering candidate; keep the preferred restoration for outdoors and the static objects, where GNT is approximately on par. No presentation checkpoint or live demo model was promoted. Neither pilot delivers photograph-level quality for all four datasets.

Verification: 182 unit tests passed, including the new camera/split adapter checks. Final image/metric/source-split checks passed for 30 comparison cases, 18 held-out camera/zoom frames and 17 GNT frames. All 135 JPEGs and 138 HTTP resources passed. Isolated headless Chrome passed 30 views and 135 method selections, matching crops, mixed resolutions, toggle/reset and reference availability. All four original checkpoint hashes match. See `outputs/lightweight_methods/verification.json` and `browser_qa.json`.
