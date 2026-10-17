# Architecture

## Design goals

The package keeps geometry, matching, orchestration, evaluation, and
visualization separate enough to test independently. NumPy arrays are the main
numerical interchange type; small dataclasses make camera, pose, track, and
configuration contracts explicit.

## Image-based execution path

```text
dataset.py
  -> matching.py / learned_matching.py
  -> tracks.py
  -> reconstruction.py
       -> geometry.py
       -> bundle_adjustment.py
  -> io.py
  -> dense_fusion.py / visualization
```

1. `dataset.py` validates calibration, images, and supplied or generated
   correspondences.
2. `matching.py` extracts descriptors, performs mutual matching, applies
   essential-matrix and coverage gates, and writes reusable pair caches.
3. `tracks.py` merges pairwise observations while rejecting identity conflicts.
4. `reconstruction.py` initializes from a strong image pair, registers cameras
   with PnP, triangulates accepted tracks, and invokes final bundle adjustment.
5. `io.py` writes camera JSON, simple/rich PLYs, and machine-readable summaries.

## RGB-D SLAM execution path

```text
stage3.py
  -> depth backprojection + feature tracking + PnP
  -> estimated_camera_trajectory.txt
  -> place_recognition.py
  -> pose_graph.py
  -> official_eval.py / stage3_visualize.py
  -> stage3_gaussian_splatting_export.py
```

The SLAM implementation separates the sequential odometry backbone from
nonlocal constraint discovery. Candidate loops are retrieved classically or
with DINOv2, verified using local features and RGB-D geometry, then optimized in
the pose graph. TUM-format trajectories are the stable interchange format.

## Gaussian rendering path

The project does not reimplement rasterization. It exports fixed recovered poses
and point seeds to COLMAP text format, trains Nerfstudio Splatfacto, diagnoses
temporal/spatial support, and optionally removes unsupported Gaussians from a
checkpoint. This keeps pose estimation claims separate from rendering quality.

```text
SfM/SLAM cameras + RGB images + point seeds
  -> COLMAP export
  -> fixed-pose Splatfacto
  -> support diagnostics / cleanup
  -> matched camera-path rendering
```

## Core data contracts

| Concept | Representation | Owner |
|---|---|---|
| Calibrated camera | Intrinsic matrix plus world-to-camera `Pose` | `models.py` |
| Observation | Image index and 2D point | dataset/track structures |
| Track | Conflict-free multi-view observations | `tracks.py` |
| Sparse point | XYZ, RGB, and diagnostic metadata | reconstruction/IO |
| RGB-D pose | Timestamp, translation, XYZW quaternion | `stage3.py` |
| Loop edge | Relative transform, weights, diagnostics | `pose_graph.py` |
| Splat seed/export | COLMAP cameras, images, and points | Gaussian exporters |

## Coordinate conventions

- SfM camera poses use explicit camera/world transformations through `Pose`.
- Stage 3 trajectories are stored as camera-to-world TUM poses.
- COLMAP exports convert these to world-to-camera extrinsics.
- Nerfstudio dataparser normalization is stored and reused for matched portfolio
  cameras.
- GT similarity alignment is evaluation-only and never feeds promoted geometry.

## Extension points

An optional C++17/pybind11 backend provides geometry kernels and analytic
Jacobians for bundle adjustment and pose graphs. These kernels retain SciPy;
a separate optional Ceres 2.2 backend solves bundle adjustment with a Schur
solver. Python remains the default reference. See [C++ integration](CPP_INTEGRATION.md)
for building, backend selection, and numerical/runtime comparisons.

All optional backends use the Python implementation and deterministic tests
as their numerical reference.

## Measured-depth research path

`prepare_depth_research.py` excludes the held-out cameras, estimates heuristic
depth confidence from nearby training RGB-D agreement, and fuses a TSDF baseline.
`train_depth_research.py` resumes the unchanged fresh 5k splat with a robust
confidence-weighted relative depth loss for 5k further steps. RGB-only continuation
is the matched control. The experiment patch is process-local; installed
Nerfstudio and selected checkpoints are unchanged. The candidate stores final
inference weights without Adam state to bound disk use.

`evaluate_splat_checkpoint.py` evaluates every held-out RGB frame and optionally
compares aligned sensor depth. `report_depth_research.py` applies existing RGB
regression gates and predeclared geometric checks and builds the local web report.
See [results and reproduction](DEPTH_RESEARCH_RESULTS.md). This is an adaptation
of depth-prior ideas, not a reproduction of Dense Depth Priors or BundleFusion.
