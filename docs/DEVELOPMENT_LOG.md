# Development and Experiment Log

> Historical working notes. Paths and environment details below document the
> development machine and are not required by the public quick start.

Last updated: 2026-06-21

## Current Status

The incomplete notebooks and scripts were mapped and replaced with a clean,
testable Python package under:

```text
C:\Master Thesis\3d_motion_gen\Others\Project\sfm_reconstruction
```

Stage 1, Stage 2, and Stage 3 are operational:

- Stage 1 reconstructs from supplied calibrated correspondences.
- Stage 2 extracts SIFT features, with optional AKAZE augmentation, matches
  descriptors, filters matches geometrically, builds tracks, and runs the same
  incremental SfM backend.
- Both stages export evaluator-compatible camera JSON and PLY point clouds.
- Matplotlib and Blender visualization are available.
- Rich colored PLY export and optional Open3D visualization are available.
- Improved Stage 2 settings suppress the AprilTag calibration sheet and weak
  geometry, producing a much cleaner object reconstruction.
- Stage 3 validates RGB-D data, estimates RGB-D visual odometry, estimates
  visual loop edges, optimizes a keyframe pose graph, runs the official
  evaluator, and provides static plus Open3D trajectory/scene visualizations.

This workspace has a Git repository, but the current Stage 3 work includes
untracked files. Inspect `git status` before packaging or committing.

## Source Context

The implementation was informed by the existing project material, while the
legacy code was left untouched as reference:

- `..\Experiments`: original scripts, notebooks, outputs, and datasets.
- `..\project_3dcv-main`: additional legacy implementation.
- `..\Project_Overview.pdf`
- `..\Stage_2_Description.pdf`
- `..\Experiments\Stage_1_Description.pdf`
- `..\Computer_Vision_Lecture_12_by_Thomas_Brox_Slides.pdf`
- `..\Stage_3_Description.pdf`: implemented in the Stage 3 package modules.

The new package does not import the legacy notebooks or scripts.

## Environment

Verified interpreter:

```powershell
$python = "C:\Users\mopu01\AppData\Local\anaconda3\envs\mardm\python.exe"
```

Verified versions:

| Component | Version |
|---|---:|
| Python | 3.10.13 |
| OpenCV | 4.13.0 |
| NumPy | 1.23.5 |
| SciPy | 1.15.3 |
| pytest | 8.3.3 |
| trimesh | 3.23.5 |
| Matplotlib | 3.8.4 |
| Blender | 4.4.1 |

Blender remains available in the original environment. Open3D 0.19.0 was
installed in a cloned environment named `mardm_open3d`; the original `mardm`
environment was left unchanged.

From the package directory:

```powershell
$env:PYTHONPATH = "src"
```

Open3D clone interpreter:

```powershell
$python_open3d = "C:\Users\mopu01\AppData\Local\anaconda3\envs\mardm_open3d\python.exe"
```

## Package Layout

```text
pyproject.toml
README.md
handoff.md
scripts/
  blender_import_reconstruction.py
src/sfm_reconstruction/
  __init__.py
  __main__.py
  blender_viewer.py
  bundle_adjustment.py
  cli.py
  dataset.py
  dense_fusion.py
  evaluation.py
  geometry.py
  io.py
  matching.py
  models.py
  official_eval.py
  point_cloud_cleanup.py
  pose_graph.py
  reconstruction.py
  reprojection_diagnostics.py
  stage3.py
  stage3_open3d_viewer.py
  stage3_visualize.py
  tracks.py
  visualize.py
  _tum_eval/
tests/
  test_blender_viewer.py
  test_dataset.py
  test_dense_fusion.py
  test_evaluation.py
  test_geometry.py
  test_io.py
  test_matching.py
  test_official_eval.py
  test_open3d_viewer.py
  test_point_cloud_cleanup.py
  test_pose_graph.py
  test_reconstruction.py
  test_reprojection_diagnostics.py
  test_stage3.py
  test_stage3_open3d_viewer.py
  test_stage3_visualize.py
  test_tracks.py
  test_visualize.py
```

Important module responsibilities:

- `dataset.py`: dataset discovery, camera intrinsics, images, correspondence
  files, ground-truth extrinsics, and dataset subsetting.
- `matching.py`: SIFT, optional AKAZE, RootSIFT, binary-descriptor matching,
  custom mutual nearest-neighbor ratio matching, circular image-pair graph,
  essential-matrix filtering, AprilTag masks, and feature/correspondence
  caching.
- `tracks.py`: conflict-aware union-find construction of multi-view tracks.
- `geometry.py`: essential pose recovery, triangulation, cheirality,
  triangulation angle, reprojection errors, and PnP.
- `reconstruction.py`: initialization, incremental camera registration,
  triangulation, point filtering, and bundle-adjustment orchestration.
- `bundle_adjustment.py`: sparse robust global BA using SciPy
  `least_squares` with `soft_l1`.
- `io.py`: camera JSON, ASCII PLY, and summary export.
- `evaluation.py`: scale-normalized pose metrics against supplied ground truth.
- `visualize.py`: Matplotlib point cloud, trajectory, and viewing directions.
- `reprojection_diagnostics.py`: per-observation residual CSVs, per-camera
  summaries, and measured-versus-projected overlays.
- `point_cloud_cleanup.py`: Open3D statistical/radius outlier filtering and
  optional voxel downsampling for presentation PLYs.
- `blender_viewer.py` and `scripts/blender_import_reconstruction.py`: generate
  ready-to-open Blender scenes without requiring project packages inside
  Blender's Python runtime.
- `stage3.py`: RGB-D dataset validation, trajectory IO, RGB-D visual odometry,
  loop-closure post-processing, and Stage 3 CLI orchestration.
- `pose_graph.py`: visual loop-edge estimation, pose graph construction,
  keyframe optimization, duplicate-loop spacing, trajectory export, and
  comparison summaries.
- `official_eval.py`: wrapper around the assignment evaluator with vendored TUM
  association support.
- `stage3_visualize.py`: static aligned trajectory and error plots.
- `stage3_open3d_viewer.py`: interactive Stage 3 trajectory overlays, sampled
  RGB-D scene clouds, optional mesh preview, and GT-pose diagnostic overlays.

Console entry points:

```text
sfm-reconstruct
sfm-visualize
sfm-blender
sfm-open3d
sfm-dense-fuse
sfm-clean-cloud
sfm-stage3-setup
sfm-stage3-plot
sfm-stage3-open3d
```

## Stage 1 Implementation

Stage 1 performs:

1. Load calibrated images and supplied pairwise correspondences.
2. Merge observations into tracks with duplicate-image conflict rejection.
3. Select a high-support initialization pair containing the first image.
4. Estimate an essential matrix and recover relative pose.
5. Triangulate and reject points using cheirality, triangulation angle, and
   reprojection error.
6. Register remaining cameras incrementally with PnP RANSAC.
7. Triangulate newly visible tracks.
8. Run robust global bundle adjustment.
9. Export cameras, points, and metrics.

Verified command:

```powershell
& $python -m sfm_reconstruction `
  --dataset "..\Experiments\Stage_1_Data_ver._4\Stage_1_Data_ver_4\stage1\box" `
  --output-dir "outputs\stage1_box_filter_regression" `
  --bundle-adjustment-max-nfev 20
```

Verified Stage 1 result:

| Metric | Value |
|---|---:|
| Registered cameras | 46 / 46 |
| Reconstructed points | 5,746 |
| Tracks | 5,856 |
| Mean rotation error | 0.8751 degrees |
| Median rotation error | 0.9241 degrees |
| Mean translation error | 0.06662 |
| Median translation error | 0.06462 |

The final filtering changes produced exactly the same Stage 1 regression
metrics as the previous verified result.

## Stage 2 Implementation

Stage 2 adds:

1. SIFT extraction with configurable feature count and thresholds.
2. RootSIFT descriptor normalization.
3. Optional AKAZE extraction with binary-descriptor matching.
4. A custom mutual nearest-neighbor matcher with Lowe's ratio test.
5. Circular local image matching, including loop-closure neighbors.
6. Essential-matrix RANSAC filtering.
7. Cached features and accepted correspondence files.
8. The Stage 1 incremental reconstruction backend.

Default matching values remain:

```text
max_features = 1500
pair_window = 3
ratio_threshold = 0.75
essential_threshold = 1.0 px
minimum pair inliers = 15
```

Matching cache behavior was made deterministic. If the saved configuration
matches, the cache is treated as complete, including previously rejected
pairs. Use `--overwrite-correspondences` or a new output directory to rebuild.

## Stage 2 Quality Improvements

The original result spent much of its feature budget on the black-and-white
calibration sheet. Boot also contained distant disconnected point clusters.

The selected balanced configuration adds:

- `--max-features 2500`: detect enough non-marker features after masking.
- `--mask-apriltags`: detect AprilTag 36h11 markers and exclude their padded
  polygons from SIFT extraction.
- `--min-track-observations 3`: reject two-view-only points.
- `--max-point-distance-factor 1.5`: reject points farther than 1.5 times the
  reconstructed camera-loop radius from its median center, before and after BA.
- `--bundle-adjustment-max-nfev 20`: verified BA budget used for comparisons.

The distance filter is scale-relative and does not use ground truth.

Four-view tracks are available as a high-confidence alternative:

```text
--min-track-observations 4
```

They improve validation accuracy slightly but substantially reduce point count.

## Recommended Stage 2 Commands

Milk validation:

```powershell
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

Milk Open3D quality probe:

```powershell
& $python_open3d -m sfm_reconstruction `
  --stage 2 `
  --dataset "..\Experiments\Stage_2_Data\stage2\milk" `
  --output-dir "outputs\stage2_milk_open3d_quality_probe" `
  --matching-cache "outputs\stage2_milk_masked_track3_ba\matching_cache" `
  --max-features 2500 `
  --mask-apriltags `
  --min-track-observations 2 `
  --two-view-min-triangulation-angle 10 `
  --two-view-max-reprojection-error 2 `
  --max-point-distance-factor 1.5 `
  --bundle-adjustment-max-nfev 20
```

Boot:

```powershell
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

Boot Open3D quality probe:

```powershell
& $python_open3d -m sfm_reconstruction `
  --stage 2 `
  --dataset "..\Experiments\Stage_2_Data\stage2\boot" `
  --output-dir "outputs\stage2_boot_open3d_quality_probe" `
  --matching-cache "outputs\stage2_boot_improved\matching_cache" `
  --max-features 2500 `
  --mask-apriltags `
  --min-track-observations 2 `
  --two-view-min-triangulation-angle 10 `
  --two-view-max-reprojection-error 2 `
  --max-point-distance-factor 1.5 `
  --bundle-adjustment-max-nfev 20
```

Boot SIFT+AKAZE coverage probe:

```powershell
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
  --bundle-adjustment-max-nfev 20
```

## Validated Stage 2 Results

Balanced milk:

| Metric | Original | Improved |
|---|---:|---:|
| Registered cameras | 50 / 50 | 50 / 50 |
| Reconstructed points | 7,836 | 1,739 |
| Mean rotation error | 7.8304 degrees | 6.7789 degrees |
| Median rotation error | 6.6536 degrees | 5.6318 degrees |
| Mean translation error | 0.06964 | 0.06254 |
| Median translation error | 0.06411 | 0.05262 |
| Mean point-to-mesh vertex distance | 0.0667 | 0.0204 |
| Points within 0.05 of mesh vertices | 89.5% | 99.1% |

The point-to-mesh values above are a nearest-vertex validation proxy using the
supplied milk mesh after applying the evaluator's estimated scale. They are not
a substitute for an official hidden evaluator.

High-confidence four-view milk:

| Metric | Value |
|---|---:|
| Registered cameras | 50 / 50 |
| Reconstructed points | 961 |
| Mean rotation error | 6.5075 degrees |
| Median rotation error | 5.0354 degrees |
| Mean translation error | 0.06103 |
| Median translation error | 0.04402 |
| Mean point-to-mesh vertex distance | 0.0162 |
| Points within 0.05 of mesh vertices | 99.8% |

Improved boot:

| Metric | Original | Improved |
|---|---:|---:|
| Registered cameras | 51 / 51 | 51 / 51 |
| Reconstructed points | 7,939 | 1,254 |
| Point-distance 99th percentile | 22.58 | 5.28 |
| Maximum point distance | 79.13 | 8.42 |
| Median camera step | 1.10 | 0.94 |
| Maximum camera step | 1.41 | 1.16 |

Boot has no full ground-truth extrinsics in `poses.json`, so there are no
validated pose-error numbers for that sequence.

## Important Output Directories

```text
outputs/stage1_box_filter_regression/
outputs/stage2_milk_masked_track3_ba/
outputs/stage2_milk_masked_track4_ba/
outputs/stage2_milk_open3d_quality_probe/
outputs/stage2_boot_improved/
outputs/stage2_boot_open3d_track2_probe/
outputs/stage2_boot_open3d_quality_probe/
outputs/stage2_boot_sift_akaze_probe/
outputs/stage2_milk_sift_akaze_probe/
```

Each reconstruction directory contains:

```text
  estimated_camera_parameters.json
  estimated_points.ply
  estimated_points_rich.ply
  summary.json
```

The recommended milk and boot directories also contain:

```text
reconstruction.png
reconstruction.blend
matching_cache/
```

Direct Blender files:

```text
outputs/stage2_milk_masked_track3_ba/reconstruction.blend
outputs/stage2_boot_improved/reconstruction.blend
```

Both Blender scenes were reopened headlessly and verified to contain the
sparse point object, camera trajectory, overview camera, and all reconstructed
cameras.

The recommended milk and boot outputs were regenerated on 2026-06-17 with the
same verified point counts and now include `estimated_points_rich.ply`:

| Output | Rich vertices |
|---|---:|
| `outputs/stage2_milk_masked_track3_ba/estimated_points_rich.ply` | 1,739 |
| `outputs/stage2_boot_improved/estimated_points_rich.ply` | 1,254 |

Open3D density probe:

| Output | Points | Notes |
|---|---:|---|
| `outputs/stage2_milk_open3d_quality_probe/` | 3,237 | Same two-view quality thresholds as boot. Registers all 50 cameras, improves milk pose metrics slightly versus track-3, and improves nearest-mesh-vertex mean distance from 0.0204 to 0.0186. |
| `outputs/stage2_boot_open3d_track2_probe/` | 3,194 | Same boot matching cache, AprilTag mask, and distance pruning as the verified boot output, but `--min-track-observations 2`. Use for visual density comparison only until inspected. Median triangulation angle drops from about 30 degrees to about 14 degrees. |
| `outputs/stage2_boot_open3d_quality_probe/` | 2,125 | Same as the track-2 probe, plus `--two-view-min-triangulation-angle 10` and `--two-view-max-reprojection-error 2`. Keeps 825 two-view points, registers all 51 cameras, and raises median triangulation angle to about 21.5 degrees. |
| `outputs/stage2_boot_sift_akaze_probe/` | 5,291 | Coverage-focused current path. Uses tuned SIFT (`contrastThreshold=0.015`, `edgeThreshold=12`) plus AKAZE. Registers all 51 cameras, generates 46,569 accepted correspondences, and keeps 2,436 two-view points after quality filtering. |
| `outputs/stage2_milk_sift_akaze_probe/` | 8,004 | Coverage-focused comparison path. Same SIFT+AKAZE settings as boot. Registers all 50 cameras and improves sparse nearest-mesh mean distance to 0.0163, but worsens milk mean rotation/translation versus the SIFT-only quality probe. |

Hybrid dense-fusion probes:

| Output | Dense points | Notes |
|---|---:|---|
| `outputs/stage2_milk_open3d_quality_probe/dense_points_filtered.ply` | 39,479 | OpenCV stereo fusion from the quality-filtered SfM poses. Uses 20 neighboring pairs, quarter-scale images, sample stride 2, and `--max-sparse-distance 0.2`. Milk mesh proxy: mean nearest-vertex distance 0.0150, median 0.0138, 99.15% within 0.05. |
| `outputs/stage2_boot_open3d_quality_probe/dense_points_filtered.ply` | 18,657 | Same dense-fusion settings as milk. Opened in Open3D for visual inspection; boot has no mesh proxy. |
| `outputs/stage2_milk_sift_akaze_probe/dense_points_filtered.ply` | 43,372 | Dense fusion from SIFT+AKAZE poses. Milk mesh proxy mean/median improve to 0.0120/0.0099, but within-0.05 drops to 98.48%; use as a coverage probe. |
| `outputs/stage2_boot_sift_akaze_probe/dense_points_filtered.ply` | 22,982 | Dense fusion from SIFT+AKAZE poses. Opened in Open3D; compare boot front against the 18,657-point SIFT-only dense cloud before promoting. |
| `outputs/stage2_milk_sift_akaze_probe/dense_points_allpairs_filtered.ply` | 67,124 | Coverage-focused all-neighbor-pairs dense cloud regenerated with pair-step 1-2 scoring. Uses 97 candidates, keeps 36 non-empty useful pairs, and skips 61 near-empty pairs. Milk mesh proxy: mean nearest-vertex distance 0.0123, median 0.0100, 98.25% within 0.05. |
| `outputs/stage2_boot_sift_akaze_probe/dense_points_allpairs_filtered.ply` | 83,932 | Coverage-focused all-neighbor-pairs dense cloud regenerated with pair-step 1-2 scoring. Uses 99 candidates, keeps 51 non-empty useful pairs, and skips 48 near-empty pairs. |
| `outputs/stage2_milk_sift_akaze_probe/dense_points_allpairs_cleaned.ply` | 63,938 | Open3D statistical outlier filtering plus `--voxel-size 0.005` for presentation. Raw dense PLY is preserved. |
| `outputs/stage2_boot_sift_akaze_probe/dense_points_allpairs_cleaned.ply` | 79,872 | Open3D statistical outlier filtering plus `--voxel-size 0.005` for presentation. Raw dense PLY is preserved. |

Dense fusion command template:

```powershell
& $python_open3d -m sfm_reconstruction.dense_fusion `
  --dataset "..\Experiments\Stage_2_Data\stage2\boot" `
  --result-dir "outputs\stage2_boot_sift_akaze_probe" `
  --output "outputs\stage2_boot_sift_akaze_probe\dense_points_allpairs_filtered.ply" `
  --image-scale 0.25 `
  --max-pair-step 2 `
  --sample-stride 2 `
  --max-sparse-distance 0.2
```

Dense fusion now scores neighbor-pair candidates by sparse support and
triangulation angle. If `--max-pairs` is set, the limit is applied by score
instead of evenly sampling the sequence. Pairs below
`--min-pair-sparse-support`, below `--min-pair-triangulation-angle`, or below
`--min-dense-points-per-pair` after stereo are skipped from the final output.

Each dense run writes:

```text
*.summary.json
*.pairs.csv
*.pair_heatmap.csv
*.cameras.csv
*.spatial_density.csv
```

## Stage 3 Implementation

Stage 3 now includes:

1. RGB-D dataset validation for `rgb/`, `depth/`, `camera_parameters.json`,
   and `gt_camera_trajectory.txt`.
2. Assignment-format trajectory IO:

```text
timestamp tx ty tz qx qy qz qw
```

3. RGB-D visual odometry with descriptor matching plus PnP.
4. Experimental LK-PnP and 3D-3D RGB-D motion models.
5. Visual loop-edge estimation from RGB-D data.
6. Pose graph construction and robust optimization.
7. Keyframe pose graph acceleration with correction interpolation back to all
   original frames.
8. Official evaluator wrapper around the provided `eval.py` / `eval_ate.py`.
9. Static trajectory plots and Open3D trajectory/scene visualization.

Current promoted Stage 3 run:

```powershell
& $python_open3d -m sfm_reconstruction.stage3 `
  --dataset "..\Experiments\Stage_3_Data\stage3" `
  --output-dir "outputs\stage3_pose_graph_keyframe_stride8_nfev50" `
  --input-trajectory "outputs\stage3_pose_graph_full\estimated_camera_trajectory.txt" `
  --run-pose-graph `
  --pose-graph-keyframe-stride 8 `
  --max-loop-edges 3 `
  --loop-endpoint-spacing 100 `
  --optimizer-max-nfev 50 `
  --run-official-eval
```

Current Stage 3 result:

| Output | Optimized nodes | Internal RMSE | Official ATE |
|---|---:|---:|---:|
| `outputs/stage3_pose_graph_full/` | 4,396 | 6.6549 | 3.6164 |
| `outputs/stage3_pose_graph_keyframe_stride8_nfev50/` | 552 | **5.2430** | **3.0305** |

Important Stage 3 outputs:

```text
outputs/stage3_pose_graph_keyframe_stride8_nfev50/
  estimated_camera_trajectory_pose_graph.txt
  official_eval_summary.json
  trajectory_comparison.png
  stage3_scene_dense_voxel.ply
  stage3_scene_gt_pose_dense.ply
  stage3_scene_est_pose_dense.ply
  stage3_scene_est_pose_dense_aligned_to_gt.ply
```

The GT-aligned scene cloud is diagnostic only. It uses
`gt_camera_trajectory.txt` to align the estimated fused scene to the GT-pose
scene and must not be treated as a GT-free reconstruction result.

Presentation cleanup command:

```powershell
& $python_open3d -m sfm_reconstruction.point_cloud_cleanup `
  --input "outputs\stage2_boot_sift_akaze_probe\dense_points_allpairs_filtered.ply" `
  --output "outputs\stage2_boot_sift_akaze_probe\dense_points_allpairs_cleaned.ply" `
  --statistical-neighbors 20 `
  --statistical-std-ratio 2.0 `
  --voxel-size 0.005
```

This preserves raw dense PLYs and writes a cleaned RGB presentation PLY plus a
cleanup summary JSON.

Rich PLY vertex attributes:

```text
x y z red green blue track_id observations registered_observations
mean_reprojection_error max_reprojection_error max_triangulation_angle
```

Reprojection diagnostics can be exported during SfM runs with:

```powershell
--write-reprojection-diagnostics --reprojection-overlay-limit 300
```

This creates:

```text
reprojection_diagnostics/observations.csv
reprojection_diagnostics/per_camera.csv
reprojection_diagnostics/summary.json
reprojection_diagnostics/overlays/*.png
```

Generated reprojection diagnostics:

| Output | Observations | Overlays | Median error | P90 error |
|---|---:|---:|---:|---:|
| `outputs/stage2_boot_sift_akaze_probe/reprojection_diagnostics/` | 16,643 | 51 | 0.388 px | 1.304 px |
| `outputs/stage2_milk_sift_akaze_probe/reprojection_diagnostics/` | 26,522 | 50 | 0.335 px | 1.098 px |

## Visualization

Matplotlib PNG:

```powershell
& $python -m sfm_reconstruction.visualize `
  --result-dir "outputs\stage2_boot_improved"
```

Interactive Matplotlib:

```powershell
& $python -m sfm_reconstruction.visualize `
  --result-dir "outputs\stage2_boot_improved" `
  --show
```

Open3D colored viewer:

```powershell
& $python_open3d -m sfm_reconstruction.open3d_viewer `
  --result-dir "outputs\stage2_boot_improved" `
  --color-mode rgb
```

Useful Open3D color modes:

```text
rgb
track_length
reprojection_error
triangulation_angle
height
```

Create or open a Blender scene:

```powershell
& $python -m sfm_reconstruction.blender_viewer `
  --result-dir "outputs\stage2_boot_improved" `
  --trim-percentile 0 `
  --open
```

Blender controls:

- Middle mouse: orbit.
- Mouse wheel: zoom.
- `Home`: frame all.
- `Numpad 1`, `3`, `7`: front, side, and top.
- `Numpad 5`: toggle perspective/orthographic.
- Toggle the `Reconstructed Cameras` collection to inspect points alone.

The Blender importer represents each sparse point as a small tetrahedral
marker so points remain visible in solid and rendered views.

Stage 3 static trajectory plot:

```powershell
& $python_open3d -m sfm_reconstruction.stage3_visualize `
  --ground-truth "..\Experiments\Stage_3_Data\stage3\gt_camera_trajectory.txt" `
  --trajectory "Raw VO=outputs\stage3_pose_graph_full\estimated_camera_trajectory.txt" `
  --trajectory "Full pose graph=outputs\stage3_pose_graph_full\estimated_camera_trajectory_pose_graph.txt" `
  --trajectory "Keyframe pose graph=outputs\stage3_pose_graph_keyframe_stride8_nfev50\estimated_camera_trajectory_pose_graph.txt" `
  --output "outputs\stage3_pose_graph_keyframe_stride8_nfev50\trajectory_comparison.png"
```

Stage 3 Open3D scene viewer:

```powershell
& $python_open3d -m sfm_reconstruction.stage3_open3d_viewer `
  --dataset "..\Experiments\Stage_3_Data\stage3" `
  --scene-trajectory "outputs\stage3_pose_graph_keyframe_stride8_nfev50\estimated_camera_trajectory_pose_graph.txt" `
  --scene-frame-stride 10 `
  --scene-pixel-stride 4 `
  --scene-max-points 1200000 `
  --scene-voxel-size 0.02 `
  --scene-remove-outliers `
  --trajectory "Keyframe pose graph=outputs\stage3_pose_graph_keyframe_stride8_nfev50\estimated_camera_trajectory_pose_graph.txt" `
  --no-align `
  --trajectory-smoothing-window 9 `
  --point-size 1.5
```

## Tests

Latest verification:

```text
76 passed
```

Run:

```powershell
$env:PYTHONPATH = "src"
$env:PYTHONDONTWRITEBYTECODE = "1"
& $python_open3d -m pytest -q -p no:cacheprovider
```

Coverage includes dataset loading, matching, pair selection, track merging,
geometry, synthetic reconstruction, evaluation, visualization loaders, Blender
command construction, rich PLY export, Open3D viewer helpers, dense-fusion
diagnostics, point-cloud cleanup, reprojection diagnostics, Stage 3 trajectory
IO, RGB-D odometry helpers, loop-edge estimation, pose graph optimization,
official evaluator wrapping, static Stage 3 plotting, and Stage 3 Open3D scene
geometry/export helpers.

## Experiments and Lessons

- Dense or all-pairs matching produced many more correspondences but was less
  stable and generated more conflicting tracks.
- Permissive ratio thresholds increased correspondence count without improving
  validation quality.
- Requiring three-view tracks alone cleaned the cloud but still reconstructed
  many calibration-sheet points.
- Masking AprilTags without scene-relative distance pruning produced distant
  clusters that could have low reprojection error due to weak parallax.
- The balanced combination of masking, three-view support, distance pruning,
  and robust BA produced the best coverage/accuracy tradeoff.
- Four-view tracks are more accurate but too sparse to use as the only default.
- A visually clean sparse cloud is not sufficient evidence by itself; run the
  reprojection diagnostics before promoting a final output.

## Known Limitations

- The primary reconstruction remains SfM plus fused point clouds; there is no
  watertight dense mesh or texture atlas yet.
- The evaluator-compatible `estimated_points.ply` intentionally stays minimal;
  rich colors and metadata are in `estimated_points_rich.ply`.
- Reprojection diagnostics are opt-in and are written only when
  `--write-reprojection-diagnostics` is supplied during reconstruction.
- AprilTag masking is intentionally dataset-specific.
- The scene-relative distance filter assumes the target lies inside or near
  the camera loop; it should not be enabled blindly for arbitrary scenes.
- Camera trajectory smoothness is not explicitly regularized.
- Intrinsics are fixed; lens distortion is not modeled.
- Boot has no complete pose ground truth for quantitative pose validation.
- Stage 3 has GT trajectory but no GT full mesh. GT-pose RGB-D scene fusion is
  only a qualitative proxy, not a true ground-truth mesh comparison.
- GT-aligned Stage 3 scene overlays are diagnostics only and should not be used
  as final GT-free outputs.
- Open3D support is installed only in the cloned `mardm_open3d` environment.
- Generated outputs and Python cache files have accumulated; do not delete
  experimental outputs until the desired deliverables are archived.
## Pending Items

- [x] Add reprojection diagnostics: persist track IDs, compute residuals per
  observation, summarize per-camera median/p90/max errors, and generate
  measured-versus-projected image overlays.
- [x] Export richer point-cloud metadata: RGB sampled from source images, track
  length, reprojection error, and triangulation angle in PLY attributes or a
  sidecar file.
- [x] Improve dense fusion pair selection: score neighbor-pair candidates by
  sparse support and triangulation angle, support multiple pair steps, and
  skip weak or near-empty pairs.
- [x] Add side/coverage diagnostics: dense per-pair table, pair heatmap matrix,
  per-camera dense counts, and spatial density bins.
- [x] Add dense outlier cleanup: Open3D statistical/radius filtering and
  optional voxel downsampling for presentation clouds.
- [x] Update `README.md` visualization examples that still point at older
  `stage2_*_final` directories; prefer the improved
  `stage2_milk_masked_track3_ba` and `stage2_boot_improved` outputs.
- [ ] Add reproducibility infrastructure: inspect Git state, add any missing
  ignore rules for generated outputs/caches, commit the verified package, and
  keep a script around the 76 unit tests plus metric regressions.
- [ ] Archive desired deliverables before cleaning generated outputs, matching
  caches, Python caches, or experimental reconstruction directories.
- [x] Add a hybrid dense-fusion probe using verified SfM poses, sparse-support
  filtering, and Open3D inspection.
- [x] Stage 3 assignment implementation: RGB-D VO, visual loop closure, pose
  graph, keyframe acceleration, official evaluator, and visualization.
- [ ] Learning-based augmentation backlog: evaluate SuperPoint/LightGlue,
  DISK/LightGlue, or LoFTR-style matching as an optional frontend for all
  stages, with Stage 3 loop-closure discovery as the first priority.

## Prioritized Next Steps

1. Use AprilTag corners as pose constraints.
   Keep exact detected tag corners for camera stabilization, but exclude tag
   tracks from the final object cloud. This may improve boot trajectory
   smoothness without returning to a marker-dominated reconstruction.

2. Improve initialization and loop closure.
   Score initial pairs using baseline, support, and spatial coverage. Add
   explicit loop-closure consistency checks and consider pose-graph refinement.

3. Generate final diagnostics and cleaned presentation clouds.
   Re-run dense fusion with the chosen pair-step settings, inspect the dense
   CSVs for missing-side coverage, run reprojection overlays on promoted sparse
   runs, and export cleaned PLY copies for presentation.

4. Add reproducibility infrastructure.
   Initialize Git, ignore generated caches/outputs appropriately, commit the
   verified package, and add a small regression script or CI job for the
   31 unit tests plus Stage 1 metrics.

5. Learning-based augmentation.
   Treat learned matching as the recommended next research step across all
   stages. For Stage 1, compare against supplied correspondences as an
   ablation. For Stage 2, target wider-baseline object coverage. For Stage 3,
   use learned matches first for loop closure and only then for frame-to-frame
   VO. Gaussian Splatting remains a downstream visualization/demo once poses
   are stable, not a trajectory-drift fix.

## Recommended Resume Point

Start from:

```text
outputs/stage2_milk_masked_track3_ba
outputs/stage2_boot_improved
```

Before changing reconstruction behavior:

1. Run the 76 unit tests.
2. Re-run the Stage 1 regression if common geometry/reconstruction code changes.
3. Validate on milk using pose metrics and the supplied mesh proxy.
4. Only then regenerate boot and its Blender scene.

## Resume/Portfolio Cleanup Update - 2026-06-18

The README was rewritten into a portfolio-facing project page while preserving
the verified reconstruction commands and metrics. It now leads with the
geometric reconstruction signal, summarizes the pipeline, lists verified
results, links demo assets, and clearly frames planned work as future work
rather than completed functionality.

Added an Open3D orbit-video renderer:

```text
src/sfm_reconstruction/demo_video.py
tests/test_demo_video.py
```

The package entry points now include:

```text
sfm-demo-video = "sfm_reconstruction.demo_video:main"
```

Generated demo assets:

```text
assets/demo/boot_dense_open3d.png
assets/demo/boot_dense_open3d.mp4
assets/demo/milk_dense_open3d.png
assets/demo/milk_dense_open3d.mp4
```

Both MP4s were verified with OpenCV:

| Asset | Frames | Size | FPS |
|---|---:|---:|---:|
| `assets/demo/boot_dense_open3d.mp4` | 120 | 960x540 | 20 |
| `assets/demo/milk_dense_open3d.mp4` | 120 | 960x540 | 20 |

Updated `.gitignore` to keep generated outputs and local cache/probe files out
of future commits:

```text
outputs/
__pycache__/
*.pyc
*.pyc.*
pytest-cache-files-*/
.matplotlib*/
assets/demo/_*
```

Latest verification after this cleanup:

```text
34 passed
```

Notes:

- A temporary `assets/demo/_open3d_probe.png` and existing Python cache files
  could not be deleted because Windows denied access. They are ignored and are
  not referenced by the README.
- This update did not change reconstruction behavior. It only added demo-video
  rendering, README polish, tests for the new helper logic, and ignore rules.
- The pending reproducibility item remains open: Git still needs to be
  initialized, staged carefully, and committed with datasets/outputs excluded.

## Deliverables/Reproducibility Update - 2026-06-18

Created a local deliverables archive under:

```text
deliverables/reconstruction_deliverables_2026-06-18.zip
```

Archive size:

```text
299,763,299 bytes
```

SHA-256:

```text
D5C1BBD635358D5FB7B9BFA1B533CCCD039F864E46927BC0F86AB600D9E359EA
```

The archive contains:

```text
outputs/stage1_box_filter_regression/
outputs/stage2_milk_masked_track3_ba/
outputs/stage2_boot_improved/
outputs/stage2_milk_masked_track4_ba/
outputs/stage2_milk_sift_akaze_probe/
outputs/stage2_boot_sift_akaze_probe/
```

Added:

```text
deliverables/README.md
docs/REPRODUCIBILITY.md
requirements-repro.txt
scripts/verify_regression.py
scripts/run_verification.ps1
```

Updated `.gitignore` so the local zip is not accidentally committed:

```text
deliverables/*.zip
```

`docs/REPRODUCIBILITY.md` documents:

- what should and should not be committed,
- the pinned package snapshot,
- how to run the full verification command,
- the deliverables archive hash and contents,
- the pre-push checklist for initializing Git.

Full verification command:

```powershell
.\scripts\run_verification.ps1
```

Verification result:

```text
34 passed
summary ok: outputs/stage1_box_filter_regression
summary ok: outputs/stage2_milk_masked_track3_ba
summary ok: outputs/stage2_milk_masked_track4_ba
summary ok: outputs/stage2_boot_improved
summary ok: outputs/stage2_milk_sift_akaze_probe
summary ok: outputs/stage2_boot_sift_akaze_probe
ply ok: outputs/stage1_box_filter_regression/estimated_points.ply
ply ok: outputs/stage2_milk_masked_track3_ba/estimated_points.ply
ply ok: outputs/stage2_milk_masked_track4_ba/estimated_points.ply
ply ok: outputs/stage2_boot_improved/estimated_points.ply
ply ok: outputs/stage2_milk_sift_akaze_probe/estimated_points.ply
ply ok: outputs/stage2_boot_sift_akaze_probe/estimated_points.ply
ply ok: outputs/stage2_milk_sift_akaze_probe/dense_points_allpairs_cleaned.ply
ply ok: outputs/stage2_boot_sift_akaze_probe/dense_points_allpairs_cleaned.ply
video ok: assets/demo/boot_dense_open3d.mp4
video ok: assets/demo/milk_dense_open3d.mp4
archive ok: deliverables\reconstruction_deliverables_2026-06-18.zip
Regression artifacts verified.
```

Remaining Git-only step:

1. Initialize Git from `sfm_reconstruction/`.
2. Run `git add .`.
3. Inspect `git status` carefully.
4. Confirm ignored items are not staged:
   `outputs/`, `deliverables/*.zip`, caches, datasets, and `.matplotlib*/`.
5. Commit and push.

## Stage 3 Setup Update - 2026-06-20

Added a Stage 3 RGB-D SLAM setup scaffold:

```text
src/sfm_reconstruction/stage3.py
tests/test_stage3.py
docs/STAGE3_SETUP.md
```

New console entry point:

```text
sfm-stage3-setup = "sfm_reconstruction.stage3:main"
```

The setup helper validates the expected assignment layout:

```text
rgb/
depth/
camera_parameters.json
gt_camera_trajectory.txt
```

It writes `stage3_manifest.json` with frame timestamps, RGB/depth paths,
intrinsics, depth-match counts, and the ground-truth trajectory path when
present. With `--write-trajectory-template`, it also writes
`estimated_camera_trajectory.txt` in the evaluator format:

```text
timestamp tx ty tz qx qy qz qw
```

The trajectory template intentionally uses identity poses for every timestamp;
it is only a format scaffold, not a SLAM result.

Current Stage 3 scope:

- [x] RGB-D dataset layout validation.
- [x] Camera intrinsics loading.
- [x] RGB/depth timestamp matching.
- [x] Ground-truth and estimated trajectory parsing/writing.
- [x] Manifest export.
- [ ] RGB-D visual odometry.
- [ ] Loop closure using the sequence returning to the start.
- [ ] Pose-graph or bundle-adjustment refinement.
- [ ] Evaluation wrapper around the assignment `evaluate.py` script.

Latest local test result after the setup update:

```text
38 passed
```

Note: the root `stage3.zip` was present but `tar -tf stage3.zip` reported a
damaged or unsupported zip archive from this shell, so no Stage 3 dataset was
unpacked during this setup pass.

## Stage 3 RGB-D Odometry Update - 2026-06-20

The Stage 3 data was unpacked and validated at:

```text
C:\Master Thesis\3d_motion_gen\Others\Project\Experiments\Stage_3_Data\stage3
```

Dataset integrity checks:

- 4,396 RGB PNGs.
- 4,396 depth PNGs.
- 4,396 ground-truth trajectory rows.
- RGB/depth timestamp mismatch: 0.
- RGB/ground-truth timestamp mismatch: 0.
- PNG decode failures: 0.
- RGB shape: 1080x1920x3.
- Depth shape: 1080x1920.

Implemented a first-pass RGB-D visual odometry baseline in:

```text
src/sfm_reconstruction/stage3.py
```

It reuses existing Stage 1/2 package utilities:

- `matching.match_descriptors`
- `matching.root_sift`
- `geometry.solve_pnp`

The odometry path performs AKAZE or SIFT matching, backprojects reference-frame
depth to 3D, estimates current-from-reference motion with PnP RANSAC, composes
camera-to-world poses, and writes the assignment trajectory format:

```text
timestamp tx ty tz qx qy qz qw
```

It also writes:

```text
stage3_manifest.json
rgbd_odometry_report.csv
trajectory_metrics.json
```

Useful command:

```powershell
$env:PYTHONPATH = "src"
& $python_open3d -m sfm_reconstruction.stage3 `
  --dataset "..\Experiments\Stage_3_Data\stage3" `
  --output-dir "outputs\stage3_rgbd_vo_subset500_fullres" `
  --max-frames 500 `
  --run-rgbd-odometry `
  --max-features 1800 `
  --min-matches 20 `
  --min-pnp-inliers 8
```

Measured probes:

| Run | Tracked transitions | Translation RMSE |
|---|---:|---:|
| `outputs/stage3_rgbd_vo_subset30/` | 29 / 29 | 0.0164 |
| `outputs/stage3_rgbd_vo_subset100_scale05/` | 99 / 99 | 0.0540 |
| `outputs/stage3_rgbd_vo_subset500_scale05/` | 499 / 499 | 0.2874 |
| `outputs/stage3_rgbd_vo_subset200_fullres/` | 199 / 199 | 0.0474 |
| `outputs/stage3_rgbd_vo_subset500_fullres/` | 499 / 499 | 0.2419 |
| `outputs/stage3_rgbd_vo_full_guarded_scale05/` | 4298 / 4395 | 8.4356 |

The full-sequence half-scale baseline has heavy accumulated drift. It is a
working VO baseline, not a final SLAM result. Full-resolution probes are more
accurate but slower. The next Stage 3 work should add loop closure and
pose-graph refinement before treating the full trajectory as a deliverable.

Latest local test result after the odometry update:

```text
42 passed
```

## Stage 3 Local VO Improvement Update - 2026-06-20

Added two additional local motion modes to `stage3.py`:

```text
--motion-model lk_pnp
--motion-model rgbd
```

`lk_pnp` uses Lucas-Kanade tracking between adjacent RGB frames before
reference-depth PnP. `rgbd` uses depth from both frames and estimates a rigid
3D-3D transform with deterministic RANSAC. Both remain useful experiment modes,
but neither beat descriptor matching plus PnP on the first 500 frames.

Promoted descriptor-PnP defaults:

```text
--motion-model pnp
--ratio-threshold 0.8
--pnp-reprojection-error 2
```

Updated probe results:

| Run | Tracked transitions | Translation RMSE |
|---|---:|---:|
| `outputs/stage3_pnp_vo_subset500_fullres_ratio08_reproj2/` | 499 / 499 | 0.2166 |
| `outputs/stage3_pnp_vo_subset500_fullres_ratio08_reproj1/` | 499 / 499 | 0.2261 |
| `outputs/stage3_pnp_vo_subset500_fullres_ratio08_reproj8/` | 499 / 499 | 0.2759 |
| `outputs/stage3_pnp_vo_subset500_fullres_ratio085_reproj2/` | 499 / 499 | 0.2551 |
| `outputs/stage3_pnp_vo_subset1000_fullres_tuned/` | 999 / 999 | 0.7587 |
| `outputs/stage3_rgbd3d_vo_subset500_fullres/` | 499 / 499 | 0.6341 |
| `outputs/stage3_lk_pnp_vo_subset500_fullres/` | 499 / 499 | 0.4476 |

The best pre-loop-closure local VO setting so far is tuned descriptor-PnP with
ratio `0.8` and PnP RANSAC reprojection threshold `2 px`.

Latest local test result after this update:

```text
44 passed
```

## Stage 3 Loop-Closure Setup Update - 2026-06-20

Added loop-closure post-processing to `stage3.py`.

New CLI options:

```text
--input-trajectory
--apply-loop-closure
--loop-closure-window-fraction
--loop-closure-correct-rotation
```

The current default is conservative:

```text
--loop-closure-window-fraction 0.7
rotation correction disabled
```

It closes the final translation back to the first pose while preserving the raw
VO trajectory in a separate file. Output files:

```text
estimated_camera_trajectory_loop_closed.txt
loop_closure_summary.json
trajectory_metrics_loop_closed.json
```

Useful command:

```powershell
$env:PYTHONPATH = "src"
& $python_open3d -m sfm_reconstruction.stage3 `
  --dataset "..\Experiments\Stage_3_Data\stage3" `
  --output-dir "outputs\stage3_rgbd_vo_full_guarded_scale05_loop_closed_window70_translation" `
  --input-trajectory "outputs\stage3_rgbd_vo_full_guarded_scale05\estimated_camera_trajectory.txt" `
  --apply-loop-closure
```

Measured loop-closure probes on the full half-scale trajectory:

| Output | Translation RMSE |
|---|---:|
| raw `outputs/stage3_rgbd_vo_full_guarded_scale05/` | 8.4356 |
| `outputs/stage3_rgbd_vo_full_guarded_scale05_loop_closed_window30_translation/` | 7.3352 |
| `outputs/stage3_rgbd_vo_full_guarded_scale05_loop_closed_window50_translation/` | 6.5310 |
| `outputs/stage3_rgbd_vo_full_guarded_scale05_loop_closed_window70_translation/` | 6.4475 |
| `outputs/stage3_rgbd_vo_full_guarded_scale05_loop_closed_window100_translation/` | 6.8768 |

Earlier full-trajectory closure with rotation correction worsened RMSE, so
rotation correction is not enabled by default. This loop closure is a setup
baseline and was later superseded by the pose-graph implementation documented
below.

Historical local test result after this update:

```text
48 passed at that point; latest result is 76 passed in the 2026-06-21 update.
```

Added a detailed Stage 3 TODO document:

```text
docs/STAGE3_EXPERIMENT_LOG.md
```

It originally broke down the remaining work into visual loop-edge estimation,
pose graph representation, pose graph optimization, evaluation/comparison, and
official evaluator wrapping. Those items are now complete.

## Stage 3 Completion Update - 2026-06-21

Stage 3 is complete for the current assignment scope. The promoted result is:

```text
outputs/stage3_pose_graph_keyframe_stride8_nfev50/
```

Key facts:

- Full sequence: 4,396 frames.
- Optimized graph: 552 keyframe nodes, 551 odometry edges, 1 loop edge.
- Official ATE: `3.030521`.
- Internal translation RMSE: `5.242974`.
- Older full-node pose graph official ATE: `3.616357`.
- Raw VO official ATE from the same run family: `3.70073`.

The key Stage 3 implementation files are:

```text
src/sfm_reconstruction/stage3.py
src/sfm_reconstruction/pose_graph.py
src/sfm_reconstruction/official_eval.py
src/sfm_reconstruction/stage3_visualize.py
src/sfm_reconstruction/stage3_open3d_viewer.py
src/sfm_reconstruction/_tum_eval/associate.py
```

The Stage 3 visualization outputs include:

```text
outputs/stage3_pose_graph_keyframe_stride8_nfev50/trajectory_comparison.png
outputs/stage3_pose_graph_keyframe_stride8_nfev50/stage3_scene_dense_voxel.ply
outputs/stage3_pose_graph_keyframe_stride8_nfev50/stage3_scene_gt_pose_dense.ply
outputs/stage3_pose_graph_keyframe_stride8_nfev50/stage3_scene_est_pose_dense.ply
outputs/stage3_pose_graph_keyframe_stride8_nfev50/stage3_scene_est_pose_dense_aligned_to_gt.ply
```

The aligned estimated scene cloud is diagnostic only because it uses
`gt_camera_trajectory.txt` to compute a similarity transform into the GT frame.
The valid GT-free reconstruction view is the scene fused directly from
`estimated_camera_trajectory_pose_graph.txt`.

Latest full test result:

```text
76 passed
```

Recommended next research step: learned matching as an optional augmentation
for all stages. Stage 3 loop closure is the first high-value target because
better distinct loop constraints are the most plausible path to further ATE
improvement. Stage 2 can use learned matching for object coverage and
wide-baseline robustness. Stage 1 can use it as an ablation against supplied
correspondences. Gaussian Splatting is best treated as a downstream rendering
demo after poses are stable.

## Stage 2 Learned Wide-Baseline Update - 2026-08-07

Implemented an optional DINOv2 retrieval plus SuperPoint/LightGlue verification
path for nonlocal Stage 2 pairs. It includes essential-matrix inlier ratio,
spatial coverage, feature-level triangle-cycle filtering, a retrieved-edge
degree cap, and per-pair conflict/rejection diagnostics in
`matching_cache/pair_diagnostics.csv`.

The conservative milk run added 15 distributed wide pairs. Compared with the
local learned threshold-0.2 run, conflicts were `2,968` versus `2,645`, mean
translation error was `0.06427` versus `0.06275`, and the full-cloud
nearest-mesh-vertex proxy was `0.02539` versus `0.01510`. Its p90 reprojection error improved from
`3.667 px` to `3.176 px`, but the overall result does not pass the promotion
gate. The unrestricted 77-pair run was worse still, with `3,898` conflicts.

Added `sfm_reconstruction.stage2_evaluation`, which writes the historical milk
mesh-proxy metrics to `stage2_mesh_proxy.json`. Boot and dense fusion were not
regenerated because sparse pose and mesh-proxy quality did not stabilize or
improve. Keep the local learned run as the current learned Stage 2 experiment;
keep balanced SIFT as the promoted Stage 2 result.

## Stage 2 Pose-Only Learned Constraint Update - 2026-08-07

Added `--wide-pose-only`. Retrieved correspondences are stored separately and
used only to estimate relative-pose edges; the local learned track graph remains
unchanged. A strict GT-free consistency gate retained one edge (`84 -> 1387`,
340 inliers), followed by robust pose-graph optimization, retriangulation, and
local bundle adjustment.

The one-edge milk run produced 2,128 points, mean/median rotation errors of
`6.736/4.935` degrees, median/p90 reprojection errors of `0.753/1.906` px, mean
translation error `0.06341`, and mesh proxy `0.01874`. A zero-weight control
showed that the learned edge improves rotation, translation, and mesh quality
relative to repeated triangulation/BA alone. However, the original local
learned run remains better in translation (`0.06275`) and mesh proxy (`0.01510`).
The pose-only result is therefore experimental; boot and dense fusion remain
deferred.

## Stage 1 Learned-Matching Ablation - 2026-08-07

Completed a full 46-image ablation on the exact 366-pair supplied graph. Added
`--matching-pair-source supplied`, all-pair learned cycle filtering, a supplied
initial-pose correspondence override, optional supplied-plus-learned
augmentation, and a reproducible wrapper for the supplied Chamfer metric.

The best pure learned Chamfer was `0.6111` at LightGlue threshold 0.3, versus
`0.5124` supplied. Its mean rotation/translation errors were `2.087` degrees
and `0.1980`, versus `0.875` degrees and `0.0666` supplied. Cycle filtering
reduced conflicts from `5,847` to `695`, but did not recover pose or Chamfer
quality. Preserving supplied initialization and augmenting later pairs restored
all cameras and 10,084 points, but still worsened pose and Chamfer.

Conclusion: keep supplied correspondences as the promoted Stage 1 path. Keep
the learned frontend as a reproducible negative ablation; further threshold
tuning against box ground truth is not justified.

## Stage 1 Fixed-Pose Learned Augmentation - 2026-08-07

The negative replacement ablation led to a successful constrained integration.
Added `sfm_reconstruction.stage1_augmentation`, which freezes the promoted
supplied cameras and points, triangulates cycle-filtered learned tracks in at
least three views, optimizes only each new point, and applies strict cheirality,
reprojection, triangulation-angle, pair-consistency, bounds, and duplicate
checks.

The promoted augmentation is:

```text
outputs/stage1_box_fixed_pose_learned_augmentation/
```

It accepts 1,142 of 4,640 learned tracks and increases the cloud from 5,746 to
6,888 points. GT-to-estimate distance improves from `0.40805` to `0.40171`,
estimate-to-GT from `0.10437` to `0.10291`, and total Chamfer from `0.51242` to
`0.50461`. Camera poses are copied from the promoted supplied result, so its
`0.875` degree mean rotation and `0.0666` mean translation remain unchanged.
Promote this as a learned point-cloud augmentation, not as a learned replacement
for supplied Stage 1 correspondences.

## Stage 3 Portfolio Video Update - 2026-08-26

The promoted SLAM presentation is now:

```text
assets/portfolio/slam_reconstruction_timeline.mp4
assets/portfolio/slam_reconstruction_timeline_contact_sheet.png
```

It is a 98-second, 960x540, 24 fps inside-view edit. Its first 49 seconds move
from raw RGB-D VO
through full/keyframe pose graphs and learned loop retrieval into the 1k, 5k,
and frustum-cleaned Gaussian splats. There are no trajectory or metric panels.
Stages last seven seconds and use 1.25-second left-to-right wipes.
The remaining 49 seconds restart the captured route and show only the final
frustum-cleaned splat at the same camera speed as the first traversal, making
the loop closure visible while providing an uninterrupted best-quality pass.

The aligned exterior alternate is:

```text
assets/portfolio/slam_reconstruction_outside_timeline.mp4
assets/portfolio/slam_reconstruction_outside_timeline_contact_sheet.png
```

It remains a 72-second alternate: 49 seconds of progression plus a 23-second
exterior showcase. It applies the original classical outside-view orbit to
every point cloud and splat checkpoint. Both sides of every wipe use the
identical camera matrix and orbit frame.

`src/sfm_reconstruction/stage3_portfolio_video.py` makes the edit reproducible.
It writes a Nerfstudio camera-path JSON, applies the same camera-to-world poses
to Open3D clouds, and assembles frame-synchronized wipes. Each cloud uses its
own recovered poses at the same chronological source-frame indices. The learned
cloud and all splats share the promoted keyframe camera path exactly.
The 1k, 5k, and cleaned splat configs were verified to have identical
dataparser transforms and scale. Intermediate inside-view clips are under
`outputs/stage3_portfolio_inside/`; the shared exterior-path clips are under
`outputs/stage3_portfolio_freeview/`.

Latest full test result:

```text
123 passed
```
