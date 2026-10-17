# Native splatting follow-up

Experiments use existing fixed-pose splats: boot/milk at 30k steps and the
stair-rebalanced scene at 5k. They do not retrain splats or incorporate the newer
C++ pose-graph trajectory. Defaults and promoted checkpoints are unchanged.

## Camera support: exact parity, faster

The optional native kernel fuses projection, frustum tests and counting, releases
the GIL, and partitions points across C++ threads with disjoint outputs.
The reference is vectorized NumPy, not a Python point loop.

Median of three repetitions; excludes checkpoint loading/saving:

| Splat | Gaussians | Cameras | Python | C++ 1 thread | C++ 8 threads | Speedup, 8 threads |
|---|---:|---:|---:|---:|---:|---:|
| Boot | 303,180 | 51 | 1.004 s | 0.195 s | 0.0317 s | 31.7x |
| Milk | 305,864 | 50 | 1.061 s | 0.206 s | 0.0318 s | 33.4x |
| Scene | 877,543 | 276 | 12.491 s | 1.622 s | 0.3116 s | 40.1x |

All point counts match exactly for 1/2/4/8 threads on each saved checkpoint.
This preserves support-derived masks on the tested data; it is not a training
speedup or accuracy gain. Other platforms can differ at floating-point image
boundaries. Native counts use uint32, versus the reference's uint16.

Seed rebalance, global seed balancing, temporal/spatial submap exporters and
checkpoint cleanup accept `--backend cpp --num-threads 8`. Python remains the
default. Global balancing's per-camera summary calculations remain Python;
diagnostic image metrics remain NumPy/OpenCV.

## Depth-aware cleanup: experimental, not promoted

Gaussian centers are classified as surface-supported, contradicting measured
free space, occluded, or unknown. Occlusion and missing depth are not negative
evidence. Depth uses the original aligned pixel grid, scale 1000, and 0.1–8 m
range. No ground-truth trajectory or alignment is used.

The rule retains the frustum mask and removes a Gaussian only with at least
three free-space conflicts and zero surface agreements. Decisions use 241
training cameras; all 35 interval-8 held-out cameras are excluded. Independent
NumPy classifications agree on sampled points in the first/middle/last training
views for all three tolerances.

Tolerance is `absolute + relative * measured_depth`:

| Absolute / relative | Kept | Additional removal from 863,412 frustum-supported splats | Median held-out projected-count retention |
|---|---:|---:|---:|
| 0.02 m / 1% | 660,118 | 23.55% | 77.21% |
| 0.05 m / 2% | 697,763 | 19.19% | 81.62% |
| 0.10 m / 5% | 753,617 | 12.72% | 88.63% |

The final sweep took 15.61 s including depth loading/classification with warmed
filesystem caches, excluding saving and projected-count diagnostics. This is
one run, not a repeatability benchmark. The preliminary all-camera depth pass
was replaced by this training-only pass in the saved artifacts.

Projected counts are not pixel coverage or rendering quality. The loosest mask
retains only 82.45% of counts in its worst held-out view. Drift, depth noise and
elongated splats can invalidate center-based rejection. The candidate is saved
separately, not promoted. Subsequent held-out evaluation and visual review are
complete: all three masks were rejected. Even the loosest mask reduced SSIM
by 0.00724 and increased LPIPS by 0.02062. The new scene's depth candidate also
failed. See [refresh results](SPLAT_REFRESH_RESULTS.md).

## CUDA profiling: completed from the user's PowerShell

The agent session could not access Docker, but the user ran the prepared commands
successfully from a regular PowerShell. All three saved reports are complete;
the container reports gsplat 1.4.0. The earlier local `gpu.json` remains a record
of the access/dependency failure, not the final experiment status.

`scripts/profile_splat_cuda.py` measured dense/packed/sparse-gradient
forward/backward comparisons on three interval-8 views, recording repeated
synchronized runtime, peak allocation, RGB and means-gradient parity without
training updates. Sparse gradients need a compatible optimizer for actual
training; this is an isolated rasterizer benchmark, not full training.

Median of the three per-view medians, ten timed repetitions per view:

| Splat | Dense | Packed | Packed + sparse gradients |
|---|---:|---:|---:|
| Boot | 12.77 ms | 13.95 ms | 16.02 ms |
| Milk | 14.12 ms | 16.43 ms | 17.44 ms |
| Scene | 11.01 ms | 13.44 ms | 12.88 ms |

Dense is fastest in this aggregate comparison. Packed mode is approximately
9%, 16% and 22% slower for boot, milk and scene respectively. The third dense
scene view has substantial timing variation, so fixed-order short benchmarks
should not be interpreted as definitive end-to-end comparisons.

Maximum packed-vs-dense RGB difference is 0.00526 for boot, 0.00800 for milk,
and 5.07e-7 for the scene on a 0–1 scale. Maximum means-gradient differences
are 7.87e-6, 2.13e-4 and 2.98e-7 respectively. These are absolute differences,
not relative error or image-quality scores. Boot/milk are not numerically
identical across modes; no held-out quality benefit has been established.

Recorded maximum allocated memory for dense/packed/sparse is respectively
316/323/320 MiB for boot, 322/329/326 MiB for milk, and 795/774/752 MiB for scene.
These are diagnostic measurements, not a controlled memory comparison: the
reference RGB/gradient clones are allocated after the first mode's peak capture,
and subsequent modes retain them. No memory-based promotion is justified.

Recommendation: retain renderer settings, keep the opt-in C++ support kernel,
and reject the tested depth cleanup after the subsequent rendered quality checks.

See official [gsplat profiling](https://docs.gsplat.studio/main/tests/profile.html)
and the pinned [Splatfacto 1.1.5 source](https://github.com/nerfstudio-project/nerfstudio/blob/v1.1.5/nerfstudio/models/splatfacto.py).

## Reproduce and artifacts

Build the optional extension, then run from the repository root with `PYTHONPATH=src`:

```powershell
python scripts/benchmark_splat_native.py
python scripts/benchmark_splat_native.py --depth-only
```

The user successfully ran the corresponding command for all three datasets.
The milk command, to reproduce from a terminal with Docker access, is:

```powershell
docker run --rm --gpus all `
  -v "${PWD}:/workspace" -w /workspace -e PYTHONPATH=/workspace/src `
  ghcr.io/nerfstudio-project/nerfstudio:1.1.5 `
  python scripts/profile_splat_cuda.py `
  --run-dir outputs/stage2_milk_splatfacto/milk_half_fixed_pose/splatfacto/run_30000 `
  --colmap-dir outputs/stage2_milk_gaussian_colmap_half/sparse/0 `
  --output outputs/splatting_cpp/gpu_milk.json
```

Use corresponding boot/scene paths recorded in `benchmark_splat_native.py` for
the other checkpoints. Selected settings still require end-to-end validation.

Local artifacts under `outputs/splatting_cpp/`:

- `results.json`: support repetitions and final depth results.
- `depth_evidence.npz`: evidence and three candidate masks.
- `scene_depth_conservative.ckpt`: loosest candidate, matching pruned optimizer
  state, not promoted.
- `gpu.json`: actual local profiling failure.
- `gpu_boot.json`, `gpu_milk.json`, `gpu_scene.json`: completed CUDA reports.
- `gpu_summary.json`: aggregate timing, parity, and report hashes.
- `environment.json`: environment and source hashes.

Validation: 163 full-suite tests passed before the final integration edits;
all 21 affected support/seed/submap/cleanup tests passed after those edits and
additional boundary checks. `git diff --check` passed.
