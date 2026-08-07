# Incremental SfM Reconstruction

A clean Python implementation of a calibrated multi-view 3D reconstruction
pipeline. The project reconstructs camera poses and point clouds from object
image sequences, starting either from supplied 2D correspondences or from
detected image features.

The strongest focus is the geometric reconstruction stack: feature matching,
track building, pose estimation, triangulation, bundle adjustment, dense point
fusion, diagnostics, and visualization.

## Highlights

- Incremental Structure-from-Motion from calibrated image sets.
- Stage 1 support for supplied pairwise correspondences.
- Stage 2 support for SIFT/RootSIFT features with optional AKAZE augmentation.
- Custom mutual nearest-neighbor descriptor matching and geometric filtering.
- Conflict-aware multi-view track construction.
- Essential-matrix initialization, PnP camera registration, triangulation, and
  cheirality/reprojection/angle filtering.
- Robust sparse bundle adjustment with SciPy `least_squares`.
- AprilTag masking and scene-relative pruning to suppress calibration-board and
  distant weak-geometry artifacts.
- Dense stereo fusion from recovered SfM poses, with pair scoring and dense
  coverage diagnostics.
- Rich PLY export with RGB, track length, reprojection error, and triangulation
  angle metadata.
- Reprojection residual CSVs and measured-vs-projected overlay images.
- Open3D, Matplotlib, and Blender visualization paths.
- CLI entry points and a focused unit test suite.

## Demo Assets

Generated Open3D demos are stored under `assets/demo/`.

| Scene | Poster | Video |
|---|---|---|
| Boot dense reconstruction | ![Boot dense Open3D render](assets/demo/boot_dense_open3d.png) | [MP4](assets/demo/boot_dense_open3d.mp4) |
| Milk dense reconstruction | ![Milk dense Open3D render](assets/demo/milk_dense_open3d.png) | [MP4](assets/demo/milk_dense_open3d.mp4) |

Regenerate the boot demo:

```powershell
$env:PYTHONPATH = "src"
& $python_open3d -m sfm_reconstruction.demo_video `
  --result-dir "outputs\stage2_boot_sift_akaze_probe" `
  --point-cloud "outputs\stage2_boot_sift_akaze_probe\dense_points_allpairs_cleaned.ply" `
  --output "assets\demo\boot_dense_open3d.mp4" `
  --poster-output "assets\demo\boot_dense_open3d.png" `
  --width 960 `
  --height 540 `
  --fps 20 `
  --seconds 6 `
  --point-size 2.0 `
  --zoom 0.62 `
  --trim-percentile 2 `
  --no-cameras
```

Regenerate the milk demo:

```powershell
$env:PYTHONPATH = "src"
& $python_open3d -m sfm_reconstruction.demo_video `
  --result-dir "outputs\stage2_milk_sift_akaze_probe" `
  --point-cloud "outputs\stage2_milk_sift_akaze_probe\dense_points_allpairs_cleaned.ply" `
  --output "assets\demo\milk_dense_open3d.mp4" `
  --poster-output "assets\demo\milk_dense_open3d.png" `
  --width 960 `
  --height 540 `
  --fps 20 `
  --seconds 6 `
  --point-size 2.0 `
  --zoom 0.62 `
  --trim-percentile 2 `
  --no-cameras
```

## Verified Results

| Run | Registered cameras | Sparse points | Key metric |
|---|---:|---:|---|
| Stage 1 box regression | 46 / 46 | 5,746 | Mean rotation error `0.875 deg`, mean translation error `0.0666` |
| Stage 2 milk, balanced SIFT | 50 / 50 | 1,739 | Mean rotation error `6.779 deg`, mean translation error `0.0625` |
| Stage 2 milk, four-view tracks | 50 / 50 | 961 | Mean rotation error `6.507 deg`, mean translation error `0.0610` |
| Stage 2 boot, conservative output | 51 / 51 | 1,254 | No full boot pose ground truth available |
| Stage 2 boot, SIFT+AKAZE coverage probe | 51 / 51 | 5,291 | 79,872 cleaned dense points |
| Stage 2 milk, SIFT+AKAZE coverage probe | 50 / 50 | 8,004 | 63,938 cleaned dense points |

Additional validation notes:

- Balanced milk reduced the nearest-mesh-vertex proxy error from `0.0667` to
  `0.0204` versus the original Stage 2 run.
- Four-view milk reached a nearest-mesh-vertex proxy error of `0.0162`.
- The all-pairs SIFT+AKAZE milk dense cloud reached mean/median proxy distance
  `0.0123` / `0.0100`, with `98.25%` of points within `0.05` of the supplied
  validation mesh.
- Reprojection diagnostics for SIFT+AKAZE runs:
  - Boot: 16,643 observations, median error `0.388 px`, p90 `1.304 px`.
  - Milk: 26,522 observations, median error `0.335 px`, p90 `1.098 px`.

Boot is the submission-style sequence and does not include complete ground
truth poses, so boot quality is assessed through reconstruction consistency,
camera registration, dense coverage, reprojection diagnostics, and visual
inspection.

## Pipeline

```text
RGB images + intrinsics
  -> feature extraction or supplied correspondences
  -> descriptor matching and essential-matrix filtering
  -> conflict-aware multi-view tracks
  -> initial relative pose and triangulation
  -> incremental PnP camera registration
  -> new-point triangulation and filtering
  -> robust global bundle adjustment
  -> sparse/rich PLY, camera JSON, metrics, diagnostics
  -> dense stereo fusion and cleaned presentation clouds
```

## Repository Layout

```text
src/sfm_reconstruction/
  cli.py                         Main reconstruction CLI
  matching.py                    SIFT, AKAZE, matching, masks, pair graph
  tracks.py                      Multi-view track construction
  geometry.py                    Essential pose, triangulation, PnP, errors
  reconstruction.py              Incremental SfM backend
  bundle_adjustment.py           Robust sparse bundle adjustment
  dense_fusion.py                Stereo fusion from recovered poses
  point_cloud_cleanup.py         Open3D outlier filtering/downsampling
  reprojection_diagnostics.py    Residual CSVs and image overlays
  stage3.py                      Stage 3 RGB-D SLAM dataset setup helpers
  open3d_viewer.py               Interactive Open3D viewer
  demo_video.py                  Open3D orbit MP4 renderer
  blender_viewer.py              Blender scene generation
tests/                           Unit tests for core geometry and IO paths
```

The legacy notebooks/scripts and assignment data live outside this package and
are not imported by the implementation.

## Environment

Known working interpreters on this machine:

```powershell
$python = "C:\Users\mopu01\AppData\Local\anaconda3\envs\mardm\python.exe"
$python_open3d = "C:\Users\mopu01\AppData\Local\anaconda3\envs\mardm_open3d\python.exe"
```

For a fresh environment:

```powershell
python -m pip install -e .[dev,evaluation,visualization]
python -m pip install -e .[open3d]
```

The dataset folders are local inputs and are intentionally not part of this
package. Commands below assume the same local directory structure used during
development.

## Run Reconstruction

Stage 1 calibrated correspondence regression:

```powershell
$env:PYTHONPATH = "src"
& $python -m sfm_reconstruction `
  --dataset "..\Experiments\Stage_1_Data_ver._4\Stage_1_Data_ver_4\stage1\box" `
  --output-dir "outputs\stage1_box_filter_regression" `
  --bundle-adjustment-max-nfev 20
```

Stage 2 milk balanced validation:

```powershell
$env:PYTHONPATH = "src"
& $python -m sfm_reconstruction `
  --stage 2 `
  --dataset "..\Experiments\Stage_2_Data\stage2\milk" `
  --output-dir "outputs\stage2_milk_masked_track3_ba" `
  --max-features 2500 `
  --mask-apriltags `
  --min-track-observations 3 `
  --max-point-distance-factor 1.5 `
  --bundle-adjustment-max-nfev 20
```

Stage 2 boot conservative output:

```powershell
$env:PYTHONPATH = "src"
& $python -m sfm_reconstruction `
  --stage 2 `
  --dataset "..\Experiments\Stage_2_Data\stage2\boot" `
  --output-dir "outputs\stage2_boot_improved" `
  --max-features 2500 `
  --mask-apriltags `
  --min-track-observations 3 `
  --max-point-distance-factor 1.5 `
  --bundle-adjustment-max-nfev 20
```

Stage 2 boot coverage-focused SIFT+AKAZE run:

```powershell
$env:PYTHONPATH = "src"
& $python_open3d -m sfm_reconstruction `
  --stage 2 `
  --dataset "..\Experiments\Stage_2_Data\stage2\boot" `
  --output-dir "outputs\stage2_boot_sift_akaze_probe" `
  --max-features 3000 `
  --feature-mode "sift+akaze" `
  --sift-contrast-threshold 0.015 `
  --sift-edge-threshold 12 `
  --mask-apriltags `
  --min-track-observations 2 `
  --two-view-min-triangulation-angle 10 `
  --two-view-max-reprojection-error 2 `
  --max-point-distance-factor 1.5 `
  --bundle-adjustment-max-nfev 20 `
  --write-reprojection-diagnostics `
  --reprojection-overlay-limit 300
```

Each reconstruction output contains:

- `estimated_camera_parameters.json`
- `estimated_points.ply`
- `estimated_points_rich.ply`
- `summary.json`

## Stage 3 Setup

Stage 3 is the optional RGB-D SLAM trajectory task. The setup helper validates
the assignment layout and can run a first-pass RGB-D visual odometry baseline:

```powershell
$env:PYTHONPATH = "src"
& $python -m sfm_reconstruction.stage3 `
  --dataset "..\stage3" `
  --output-dir "outputs\stage3_rgbd_vo_subset500_fullres" `
  --max-frames 500 `
  --run-rgbd-odometry `
  --max-features 1800 `
  --min-matches 20 `
  --min-pnp-inliers 8
```

The baseline reuses the Stage 1/2 matching and geometry utilities: feature
matching, depth backprojection, PnP, and trajectory export. It writes
`estimated_camera_trajectory.txt`, `rgbd_odometry_report.csv`, and
`trajectory_metrics.json` when ground truth is present. The current tuned
full-resolution 500-frame probe tracks 499 / 499 frame transitions with
translation RMSE `0.2166`; the 1000-frame probe tracks 999 / 999 with RMSE
`0.7587`. Full-sequence visual odometry still drifts without loop closure.
More details are in `docs/STAGE3_SETUP.md`.

Stage 3 loop-edge discovery also supports an optional learned correspondence
frontend with `--loop-matcher superpoint-lightglue`. It changes only feature
matching; RGB-D geometric verification and pose-graph optimization remain the
same. See `docs/LEARNING_BASED_AUGMENTATION.md` for installation and the
benchmark command.

Full-sequence learned place recognition is available with
`--loop-candidate-mode dinov2`. DINOv2 retrieval plus local endpoint refinement
recovers the promoted `4387 -> 32` loop automatically; no second distinct loop
improved the official result in the current benchmark.

Loop-closure post-processing is available for trajectories where the sequence
returns near the start:

```powershell
$env:PYTHONPATH = "src"
& $python -m sfm_reconstruction.stage3 `
  --dataset "..\Experiments\Stage_3_Data\stage3" `
  --output-dir "outputs\stage3_rgbd_vo_full_guarded_scale05_loop_closed_window70_translation" `
  --input-trajectory "outputs\stage3_rgbd_vo_full_guarded_scale05\estimated_camera_trajectory.txt" `
  --apply-loop-closure
```

## Dense Fusion

Use the estimated cameras as a pose backbone for OpenCV stereo fusion:

```powershell
$env:PYTHONPATH = "src"
& $python_open3d -m sfm_reconstruction.dense_fusion `
  --dataset "..\Experiments\Stage_2_Data\stage2\boot" `
  --result-dir "outputs\stage2_boot_sift_akaze_probe" `
  --output "outputs\stage2_boot_sift_akaze_probe\dense_points_allpairs_filtered.ply" `
  --image-scale 0.25 `
  --max-pair-step 2 `
  --sample-stride 2 `
  --max-sparse-distance 0.2
```

Clean the dense point cloud for presentation:

```powershell
$env:PYTHONPATH = "src"
& $python_open3d -m sfm_reconstruction.point_cloud_cleanup `
  --input "outputs\stage2_boot_sift_akaze_probe\dense_points_allpairs_filtered.ply" `
  --output "outputs\stage2_boot_sift_akaze_probe\dense_points_allpairs_cleaned.ply" `
  --statistical-neighbors 20 `
  --statistical-std-ratio 2.0 `
  --voxel-size 0.005
```

Dense runs write:

- `*.summary.json`
- `*.pairs.csv`
- `*.pair_heatmap.csv`
- `*.cameras.csv`
- `*.spatial_density.csv`

## Visualization

Matplotlib static view:

```powershell
$env:PYTHONPATH = "src"
& $python -m sfm_reconstruction.visualize `
  --result-dir "outputs\stage2_boot_improved"
```

Interactive Open3D viewer:

```powershell
$env:PYTHONPATH = "src"
& $python_open3d -m sfm_reconstruction.open3d_viewer `
  --result-dir "outputs\stage2_boot_sift_akaze_probe" `
  --point-cloud "outputs\stage2_boot_sift_akaze_probe\dense_points_allpairs_cleaned.ply" `
  --color-mode rgb
```

Blender scene export:

```powershell
$env:PYTHONPATH = "src"
& $python -m sfm_reconstruction.blender_viewer `
  --result-dir "outputs\stage2_boot_improved" `
  --trim-percentile 0
```

Open3D color modes for sparse rich PLYs:

```text
rgb
track_length
reprojection_error
triangulation_angle
height
```

## Tests

```powershell
$env:PYTHONPATH = "src"
$env:PYTHONDONTWRITEBYTECODE = "1"
& $python_open3d -m pytest -q -p no:cacheprovider
```

Latest local verification: `48 passed`. The video-renderer helper tests add
coverage for Open3D demo asset generation without requiring a render window,
and the Stage 3 tests cover RGB-D layout validation, trajectory IO, odometry
reports, trajectory metrics, RGB-D rigid-transform helpers, and loop-closure
post-processing.


## Planned extensions:

- Continue benchmarking the implemented SuperPoint+LightGlue frontend on boot
  and wider Stage 3 loop windows; the initial milk and Stage 3 results are
  documented in `docs/LEARNING_BASED_AUGMENTATION.md`.
- Add mesh reconstruction from dense fused points using Poisson or ball-pivoting
  surface reconstruction.
- Export SfM cameras/points to a COLMAP-style layout and train a small 3D
  Gaussian Splatting demo initialized from this reconstruction.
