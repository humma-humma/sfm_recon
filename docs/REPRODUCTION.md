# Reproduction guide

This document separates fast code verification from dataset-dependent research
runs and GPU-dependent Gaussian rendering.

## 1. Environment

Python 3.10 and 3.11 are supported.

```bash
python -m venv .venv
python -m pip install --upgrade pip
python -m pip install -e ".[dev,evaluation,visualization]"
```

Optional extras:

```bash
python -m pip install -e ".[open3d]"
python -m pip install -e ".[learned]"
```

The learned extra does not install third-party model repositories. Kornia
provides the SuperPoint/LightGlue path; DINOv2 weights are loaded through
PyTorch Hub when the place-recognition mode is requested.

## 2. Code-only verification

No project dataset is required for the unit suite:

```bash
python -m pytest -q -p no:cacheprovider
python -m build
```

Expected result for this release: `123 passed` and successful source/wheel
builds.

On Windows, the wrapper also checks locally available result artifacts:

```powershell
.\scripts\run_verification.ps1 -Python python
```

Artifact checks are skipped only when their documented local outputs are not
present; the unit suite remains the portable acceptance gate.

## 3. Image-based SfM

```bash
sfm-reconstruct \
  --stage 2 \
  --dataset /path/to/stage2/milk \
  --output-dir outputs/stage2_milk \
  --max-features 2500 \
  --feature-mode sift \
  --mask-apriltags \
  --min-track-observations 3 \
  --max-point-distance-factor 1.5 \
  --bundle-adjustment-max-nfev 20
```

Use `--feature-mode superpoint-lightglue` after installing the `learned` extra.
The promoted object results retain fixed cameras when learned points are added;
see `LEARNING_BASED_AUGMENTATION.md` for the controlled ablations.

## 4. RGB-D SLAM

A bounded smoke run:

```bash
sfm-stage3-setup \
  --dataset /path/to/stage3 \
  --output-dir outputs/stage3_smoke \
  --max-frames 500 \
  --run-rgbd-odometry \
  --max-features 1800 \
  --min-matches 20 \
  --min-pnp-inliers 8
```

Full pose-graph and learned loop-closure options are listed by:

```bash
sfm-stage3-setup --help
```

The promoted benchmark uses guarded full-sequence RGB-D VO, keyframe-stride-8
pose-graph optimization, and automatic DINOv2 retrieval with local geometric
verification. Ground truth is used only for evaluation, never for the promoted
reconstruction or Gaussian initialization.

## 5. Fixed-pose Gaussian export

Object export:

```bash
sfm-export-gaussian \
  --dataset /path/to/stage2/milk \
  --result-dir outputs/stage2_milk \
  --output-dir outputs/stage2_milk_colmap \
  --image-scale 0.5
```

SLAM export:

```bash
sfm-export-stage3-gaussian \
  --dataset /path/to/stage3 \
  --trajectory outputs/stage3_pose_graph/estimated_camera_trajectory_pose_graph.txt \
  --point-cloud outputs/stage3_pose_graph/stage3_scene_est_pose_dense.ply \
  --output-dir outputs/stage3_colmap \
  --frame-stride 16 \
  --image-scale 0.5 \
  --max-points 200000
```

Training is performed with Nerfstudio Splatfacto in its official container.
Exact checkpoint cleanup and rendering commands are documented in
`GAUSSIAN_SPLATTING.md`. Checkpoints, container caches, and generated datasets
remain outside Git.

## 6. Portfolio rendering

`sfm-stage3-portfolio-video --help` exposes the camera-path, point-cloud,
trajectory, and composition helpers used for the published timelines. All
published transitions use matched camera-frame indices. The source clips are
kept as local outputs; only the three final MP4s and contact sheets are tracked.

## 7. Expected hardware and time

| Task | Hardware | Typical scope |
|---|---|---|
| Unit tests | CPU | Seconds |
| Stage 2 SfM | CPU | Minutes, dataset dependent |
| Stage 3 smoke VO | CPU | Minutes for 500 frames |
| Full Stage 3 | CPU | Hours, configuration dependent |
| Splat training/rendering | NVIDIA GPU | Minutes to hours |

Runtime varies substantially with image count, resolution, feature count, and
optimization limits. The repository does not claim real-time execution.
