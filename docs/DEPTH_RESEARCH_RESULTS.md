# Measured-depth research results

A single controlled continuation experiment, adapting confidence-aware depth supervision to Splatfacto. This is not a paper reproduction.

RGB gate versus matched 10k control: **passes_numeric_gate_visual_review_pending**. Selected models unchanged.

| Metric | RGB-only 10k | Depth-supervised 10k |
|---|---:|---:|
| psnr | 16.889580 | 17.475343 |
| ssim | 0.679651 | 0.686143 |
| lpips | 0.583312 | 0.571673 |

Worst held-out PSNR delta: -0.383 dB.
Training: 274.8 seconds; peak allocated CUDA memory 4539.6 MiB. Shared GPU workload was left running; this is not an isolated speed benchmark.

| Diagnostic image | Control depth error (m) | Candidate depth error (m) |
|---|---:|---:|
| 4.56851100.jpg | 0.7461 | 0.6052 |
| 51.5018330.jpg | 0.4834 | 0.4682 |
| 64.3018440.jpg | 0.1144 | 0.0635 |
| 98.4351660.jpg | 0.7982 | 0.7968 |
| 128.301833.jpg | 0.7150 | 0.4776 |
| 145.368511.jpg | 0.4965 | 0.0239 |

Depth errors use identical valid sensor/interior/opacity masks for control and candidate. The JSON also compares all three methods on a common intersection mask and reports TSDF coverage.

Geometry checks: `{'median_depth_error': True, 'pavement_preserved': True}`. TSDF has 680,600 vertices and 1,189,339 triangles; the stair entrance has only 18.2% sensor-surface coverage.

## Reproduction and claims

Run `scripts/run_depth_research.ps1` with the local Open3D/PyTorch Python environment. Prepared depth uses training frames and their training neighbors only; all 35 held-out depths are evaluation-only. Both splat runs start from the same original 5k checkpoint and receive 5k further steps with identical poses, Gaussian initialization and resolution schedule. The sole training change is the measured-depth loss. The reference is the already completed 10k RGB-only continuation.

The loss is robust relative Huber with weight 0.1, using measured optical-axis depth in metres. Confidence requires agreement within 5cm + 3% in at least two nearby training cameras, and excludes depth discontinuities. These are heuristic reliability weights, not calibrated learned uncertainty. Configuration and pre-evaluation gates are in `configs/splatting/depth_research.json`.

The project drive filled during the first attempt to save an intermediate checkpoint. The incomplete output was removed and training reran from the original source. Only final inference weights are saved; this candidate checkpoint cannot resume Adam optimization. Originals were untouched.

The confidence-masked TSDF uses 2.5cm voxels and 7.5cm truncation, estimated poses, and training RGB-D only. It supplies a geometric baseline, not a complete/collision-ready scene or a BundleFusion reproduction.

Paper inspiration: [Dense Depth Priors, CVPR 2022](https://barbararoessle.github.io/dense_depth_priors_nerf/). Alignment and consistent fusion context: [BundleFusion, TOG 2017](https://graphics.stanford.edu/projects/bundlefusion/). 2DGS and learned-prior integrations remain future work; neither is claimed as implemented.

Visual results: `web_demo/research.html`. Numeric and runtime artifacts: `outputs/depth_research/results.json`. Geometry: `outputs/depth_research/scene_tsdf.ply`. Sensor alignment/units remain assumptions; expected depth is not thickness; these are exploratory dataset-specific results.

## Sampled visual review

The depth-supervised candidate reveals more staircase structure at the entrance and interior, but leaves strong fog and missing detail. Pavement remains comparable or clearer. Corridor walls remain streaky; lab lettering is still lost. Exterior-stair and initial-signage regressions remain visually imperfect. This is a useful geometric research improvement, not a sufficiently clean universal replacement for the frozen presentations.

Review scope: selected diagnostics, pavement control and worst-regression views; not every novel camera. Keep the selected presentation checkpoints unchanged. Numeric gate status retains its standard visual-review-pending label; the separate review record documents this sampled assessment.

## Split scope

The inherited 5k initialization comes from the existing full-sequence estimated dense cloud and was not rebuilt using training views only. Pose estimation also precedes the rendering split. Held-out exclusion is verified for the added loss, its confidence neighbors and the TSDF integration; it is not an end-to-end split-isolation claim. Shared initialization keeps the continuation comparison controlled, but these results must not be presented as a strictly held-out reconstruction benchmark.
