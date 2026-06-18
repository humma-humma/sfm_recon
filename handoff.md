# SfM Project Handoff

Last updated: 2026-06-17

## Current Status

The incomplete notebooks and scripts were mapped and replaced with a clean,
testable Python package under:

```text
C:\Master Thesis\3d_motion_gen\Others\Project\sfm_reconstruction
```

Stage 1 and Stage 2 are operational:

- Stage 1 reconstructs from supplied calibrated correspondences.
- Stage 2 extracts SIFT features, with optional AKAZE augmentation, matches
  descriptors, filters matches geometrically, builds tracks, and runs the same
  incremental SfM backend.
- Both stages export evaluator-compatible camera JSON and PLY point clouds.
- Matplotlib and Blender visualization are available.
- Rich colored PLY export and optional Open3D visualization are available.
- Improved Stage 2 settings suppress the AprilTag calibration sheet and weak
  geometry, producing a much cleaner object reconstruction.

This workspace is not currently detected as a Git repository. There is no
commit or branch to use as a restore point.

## Source Context

The implementation was informed by the existing project material, while the
legacy code was left untouched as reference:

- `..\Experiments`: original scripts, notebooks, outputs, and datasets.
- `..\project_3dcv-main`: additional legacy implementation.
- `..\Project_Overview.pdf`
- `..\Stage_2_Description.pdf`
- `..\Experiments\Stage_1_Description.pdf`
- `..\Computer_Vision_Lecture_12_by_Thomas_Brox_Slides.pdf`
- `..\Stage_3_Description.pdf`: not implemented.

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
  point_cloud_cleanup.py
  reconstruction.py
  reprojection_diagnostics.py
  tracks.py
  visualize.py
tests/
  test_blender_viewer.py
  test_dataset.py
  test_dense_fusion.py
  test_evaluation.py
  test_geometry.py
  test_io.py
  test_matching.py
  test_open3d_viewer.py
  test_point_cloud_cleanup.py
  test_reconstruction.py
  test_reprojection_diagnostics.py
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

Console entry points:

```text
sfm-reconstruct
sfm-visualize
sfm-blender
sfm-open3d
sfm-dense-fuse
sfm-clean-cloud
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

## Tests

Latest verification:

```text
31 passed
```

Run:

```powershell
$env:PYTHONPATH = "src"
$env:PYTHONDONTWRITEBYTECODE = "1"
& $python_open3d -m pytest -q -p no:cacheprovider
```

Coverage includes dataset loading, matching, pair selection, track merging,
geometry, synthetic reconstruction, evaluation, visualization loaders, and
Blender command construction. Added coverage includes rich PLY export, the
Open3D viewer's PLY parsing, color modes, point-cloud selection, camera
geometry construction, dense-fusion helper behavior, pair scoring, dense
diagnostics, disparity-range clamping for OpenCV SGBM, point-cloud cleanup
export, and reprojection diagnostic export.

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
- [ ] Add reproducibility infrastructure: initialize Git, add ignore rules for
  generated outputs/caches, commit the verified package, and script the 31 unit
  tests plus Stage 1 metric regression.
- [ ] Archive desired deliverables before cleaning generated outputs, matching
  caches, Python caches, or experimental reconstruction directories.
- [x] Add a hybrid dense-fusion probe using verified SfM poses, sparse-support
  filtering, and Open3D inspection.
- [ ] Stage 3 and advanced matching backlog: implement the Stage 3 assignment
  from `..\Stage_3_Description.pdf`; if object coverage is still insufficient,
  evaluate learned matching such as SuperPoint/LightGlue, DISK/LightGlue, or
  LoFTR-style matching after validating on milk.

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

5. Stage 3 / advanced matching.
   Treat learned matching as part of the Stage 3 backlog rather than the
   current Stage 2 completion path.

## Recommended Resume Point

Start from:

```text
outputs/stage2_milk_masked_track3_ba
outputs/stage2_boot_improved
```

Before changing reconstruction behavior:

1. Run the 31 unit tests.
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
