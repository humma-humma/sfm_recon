# Reconstruction hyperparameter check - 2026-10-09

The check covers Stage 1 box, Stage 2 boot and milk, and the Stage 3 scene. Every field in the reconstruction, matching, odometry and pose-graph configuration classes is audited in `outputs/hyperparameter_check/parameter_audit.json`. The experiment is bounded screening plus interaction checks, not an exhaustive Cartesian search over continuous parameters. Ground truth is used only for evaluation and selection. These are dataset-specific exploratory choices, not held-out validation or automatically promoted defaults.

Object sweeps rerun complete reconstruction on cached correspondence files. Single-run timings exclude matching and evaluation. Geometry uses the previously validated C++ kernels; BA uses Ceres except the explicit Python control. Camera count, point retention and image/mesh coverage guards prevent a sparse or partially registered result from winning on a smaller residual alone. Timing repeats use OpenCV seed 0; baseline and selected candidates are also checked with seed 1.

Boot has no full camera or mesh ground truth: its supplied `poses.json` contains only the origin pose. Its ranking measures internal reprojection consistency and supported-image coverage, not absolute camera/surface accuracy. The nearby estimated-camera JSON is not treated as ground truth.

## Box

Recorded 33 runs: 32 successful, 1 failed.

Screen-selected candidate: `min_triangulation_angle_0.5`. Median complete reconstruction time across its initial call and repeats: 10.237 seconds. Selection rule: At least baseline camera count, 90% baseline point count and 95% baseline image and mesh coverage. Rank mean normalized camera errors plus geometry proxy (boot: reprojection consistency only). This is exploratory GT tuning, not independent validation.

```json
{
  "track_tolerance": 0.001,
  "min_track_observations": 2,
  "essential_threshold": 1.0,
  "pnp_reprojection_error": 4.0,
  "max_reprojection_error": 5.0,
  "min_triangulation_angle": 0.5,
  "min_pnp_points": 12,
  "min_pnp_inliers": 10,
  "bundle_adjustment": true,
  "bundle_adjustment_max_nfev": 40,
  "max_point_distance_factor": null,
  "two_view_max_reprojection_error": null,
  "two_view_min_triangulation_angle": null,
  "initial_pose_correspondence_dir": null,
  "bundle_adjustment_backend": "ceres",
  "geometry_backend": "cpp",
  "ceres_max_iterations": 100,
  "bundle_adjustment_loss_scale": 2.0,
  "bundle_adjustment_max_reprojection_error": 5.0
}
```

Baseline median reconstruction time (3 calls): 10.117 seconds. Candidate median uses 3 calls. Matching and evaluation are excluded for both.

Measured timing ranges: baseline 10.049–10.254 s; selected 10.074–11.318 s. Calls ran sequentially on the same machine; differences are not a statistical speed guarantee.

The Stage 1 Chamfer metric is the sum of two directed means. GT-to-estimate changes from 0.384227 to 0.358736; estimate-to-GT changes from 0.067680 to 0.069218. The selected candidate is a tradeoff, not an improvement in every metric.

| Case | Cameras | Points | Translation | Rotation (deg) | Reprojection RMSE (px) | Geometry metric | Coverage | Seconds |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| baseline | 46 | 5746 | 0.033429 | 0.561156 | 1.3305 | 0.451907 | 38.45% | 10.254 |
| bundle_adjustment_loss_scale_0.5 | 46 | 5745 | 0.033242 | 0.603272 | 1.3554 | 0.441260 | 38.45% | 11.494 |
| bundle_adjustment_loss_scale_1 | 46 | 5746 | 0.033260 | 0.580559 | 1.3355 | 0.445647 | 38.45% | 10.587 |
| bundle_adjustment_loss_scale_4 | 46 | 5746 | 0.033912 | 0.558917 | 1.3289 | 0.455747 | 38.45% | 9.939 |
| bundle_adjustment_max_reprojection_error_2 | 46 | 5672 | 0.097780 | 1.044598 | 1.9960 | 0.511114 | 38.11% | 9.362 |
| bundle_adjustment_max_reprojection_error_3 | 46 | 5728 | 0.058413 | 0.699511 | 1.7619 | 0.469037 | 38.45% | 9.995 |
| bundle_adjustment_max_reprojection_error_8 | 46 | 5746 | 0.033617 | 0.556565 | 1.1469 | 0.453499 | 38.45% | 11.434 |
| min_track_observations_3 | 46 | 2624 | 0.061194 | 0.848985 | 1.2313 | 0.800472 | 32.47% | 9.232 |
| min_track_observations_4 | FAILED | | | | | No valid initialization pair could be reconstructed | | |
| min_triangulation_angle_0.5 | 46 | 5776 | 0.033505 | 0.560584 | 1.3323 | 0.427954 | 38.55% | 11.318 |
| min_triangulation_angle_2 | 46 | 5560 | 0.035251 | 0.585301 | 1.3248 | 0.535218 | 37.50% | 11.394 |
| min_triangulation_angle_3 | 46 | 5265 | 0.034737 | 0.581793 | 1.3107 | 0.636741 | 36.99% | 10.726 |
| pnp_reprojection_error_2 | 46 | 5742 | 0.038732 | 0.635277 | 1.5186 | 0.450336 | 38.35% | 51.245 |
| pnp_reprojection_error_3 | 46 | 5741 | 0.040688 | 0.712558 | 1.3051 | 0.452545 | 38.45% | 22.830 |
| pnp_reprojection_error_6 | 46 | 5748 | 0.034479 | 0.577089 | 1.3035 | 0.452668 | 38.45% | 7.382 |
| essential_threshold_0.5 | 46 | 5760 | 0.187413 | 2.269916 | 1.2989 | 0.638551 | 38.42% | 9.868 |
| essential_threshold_2 | 5 | 257 | 0.095590 | 0.628327 | 0.4679 | 1.369225 | 30.63% | 0.647 |
| max_reprojection_error_3 | 46 | 5718 | 0.051513 | 0.674063 | 1.2918 | 0.467521 | 38.32% | 9.674 |
| max_reprojection_error_8 | 46 | 5753 | 0.035296 | 0.622991 | 1.3136 | 0.451390 | 38.45% | 9.813 |
| max_point_distance_factor_1.25 | 46 | 5716 | 0.033552 | 0.561554 | 1.3235 | 0.824796 | 37.36% | 10.347 |
| max_point_distance_factor_1.5 | 46 | 5727 | 0.033362 | 0.561091 | 1.3321 | 0.814629 | 37.60% | 10.305 |
| max_point_distance_factor_2 | 46 | 5728 | 0.033365 | 0.561622 | 1.3316 | 0.804457 | 37.67% | 10.445 |
| ceres_max_iterations_5 | 46 | 5746 | 0.046543 | 0.659448 | 1.3307 | 0.480617 | 38.45% | 10.121 |
| two_view_quality5 | 46 | 4729 | 0.037214 | 0.587084 | 1.2943 | 0.629277 | 36.28% | 9.967 |
| two_view_quality10 | 46 | 3612 | 0.044413 | 0.648150 | 1.1828 | 0.768799 | 34.00% | 9.608 |
| python_control20 | 46 | 5746 | 0.066623 | 0.875133 | 1.3311 | 0.512420 | 38.45% | 47.176 |
| combined_screen_winners | 46 | 5776 | 0.033505 | 0.560584 | 1.3323 | 0.427954 | 38.55% | 10.279 |
| selected_repeat1 | 46 | 5776 | 0.033505 | 0.560584 | 1.3323 | 0.427954 | 38.55% | 10.074 |
| selected_repeat2 | 46 | 5776 | 0.033505 | 0.560584 | 1.3323 | 0.427954 | 38.55% | 10.237 |
| baseline_seed1 | 46 | 5746 | 0.033429 | 0.561156 | 1.3305 | 0.451907 | 38.45% | 11.409 |
| selected_seed1 | 46 | 5776 | 0.033505 | 0.560584 | 1.3323 | 0.427954 | 38.55% | 11.246 |
| baseline_repeat1 | 46 | 5746 | 0.033429 | 0.561156 | 1.3305 | 0.451907 | 38.45% | 10.117 |
| baseline_repeat2 | 46 | 5746 | 0.033429 | 0.561156 | 1.3305 | 0.451907 | 38.45% | 10.049 |

Geometry metric is Stage 1 Chamfer or milk's cropped nearest-mesh-vertex proxy. Coverage is milk's fraction of GT mesh vertices within 0.05 units of an estimated point; otherwise it is the mean occupied fraction of 8x8 image bins supported by reconstructed tracks. Image coverage is not surface coverage. Changing cached frontend families also changes the observation set.

## Boot

Recorded 34 runs: 34 successful, 0 failed.

Screen-selected candidate: `essential_threshold_0.5`. Median complete reconstruction time across its initial call and repeats: 11.195 seconds. Selection rule: At least baseline camera count, 90% baseline point count and 95% baseline image and mesh coverage. Rank mean normalized camera errors plus geometry proxy (boot: reprojection consistency only). This is exploratory GT tuning, not independent validation.

```json
{
  "track_tolerance": 0.001,
  "min_track_observations": 3,
  "essential_threshold": 0.5,
  "pnp_reprojection_error": 4.0,
  "max_reprojection_error": 5.0,
  "min_triangulation_angle": 1.0,
  "min_pnp_points": 12,
  "min_pnp_inliers": 10,
  "bundle_adjustment": true,
  "bundle_adjustment_max_nfev": 40,
  "max_point_distance_factor": 1.5,
  "two_view_max_reprojection_error": null,
  "two_view_min_triangulation_angle": null,
  "initial_pose_correspondence_dir": null,
  "bundle_adjustment_backend": "ceres",
  "geometry_backend": "cpp",
  "ceres_max_iterations": 100,
  "bundle_adjustment_loss_scale": 2.0,
  "bundle_adjustment_max_reprojection_error": 3.0
}
```

Baseline median reconstruction time (3 calls): 7.553 seconds. Candidate median uses 3 calls. Matching and evaluation are excluded for both.

Measured timing ranges: baseline 7.553–10.438 s; selected 8.903–11.614 s. Calls ran sequentially on the same machine; differences are not a statistical speed guarantee.

| Case | Cameras | Points | Translation | Rotation (deg) | Reprojection RMSE (px) | Geometry metric | Coverage | Seconds |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| baseline | 51 | 1210 | — | — | 4.0262 | — | 14.22% | 10.438 |
| bundle_adjustment_loss_scale_0.5 | 51 | 1208 | — | — | 4.0122 | — | 14.12% | 7.742 |
| bundle_adjustment_loss_scale_1 | 51 | 1209 | — | — | 4.0169 | — | 14.12% | 7.686 |
| bundle_adjustment_loss_scale_4 | 51 | 1210 | — | — | 4.0286 | — | 14.15% | 7.703 |
| bundle_adjustment_max_reprojection_error_2 | 51 | 1164 | — | — | 4.2780 | — | 13.66% | 7.581 |
| bundle_adjustment_max_reprojection_error_5 | 51 | 1254 | — | — | 6.8693 | — | 14.58% | 7.771 |
| bundle_adjustment_max_reprojection_error_8 | 51 | 1254 | — | — | 6.8440 | — | 14.58% | 7.885 |
| min_track_observations_2 | 51 | 3048 | — | — | 4.5637 | — | 20.19% | 10.338 |
| min_track_observations_4 | 51 | 520 | — | — | 7.7406 | — | 10.69% | 8.740 |
| min_triangulation_angle_0.5 | 51 | 1481 | — | — | 5.5398 | — | 18.08% | 9.062 |
| min_triangulation_angle_2 | 51 | 1217 | — | — | 6.9770 | — | 14.28% | 6.672 |
| min_triangulation_angle_3 | 51 | 1298 | — | — | 6.7776 | — | 15.04% | 5.387 |
| pnp_reprojection_error_2 | 51 | 1222 | — | — | 6.9859 | — | 14.09% | 17.665 |
| pnp_reprojection_error_3 | 51 | 1267 | — | — | 6.8553 | — | 14.55% | 13.252 |
| pnp_reprojection_error_6 | 51 | 1282 | — | — | 6.8332 | — | 15.10% | 5.898 |
| essential_threshold_0.5 | 51 | 1257 | — | — | 2.9913 | — | 14.31% | 8.903 |
| essential_threshold_2 | 51 | 1036 | — | — | 8.1892 | — | 12.35% | 9.617 |
| max_reprojection_error_3 | 51 | 1061 | — | — | 4.2714 | — | 13.39% | 7.903 |
| max_reprojection_error_8 | 51 | 1276 | — | — | 6.8472 | — | 14.83% | 8.690 |
| max_point_distance_factor_1.25 | 51 | 1209 | — | — | 4.0270 | — | 14.12% | 8.060 |
| max_point_distance_factor_2 | 51 | 1368 | — | — | 3.8783 | — | 17.71% | 8.611 |
| ceres_max_iterations_5 | 51 | 1210 | — | — | 4.5563 | — | 14.22% | 8.146 |
| two_view_quality5 | 51 | 2961 | — | — | 4.6098 | — | 19.85% | 14.867 |
| two_view_quality10 | 51 | 2077 | — | — | 3.8088 | — | 16.12% | 14.311 |
| python_control20 | 51 | 1211 | — | — | 4.3094 | — | 14.22% | 26.784 |
| sift_akaze_track3 | 51 | 2674 | — | — | 3.4728 | — | 15.99% | 23.714 |
| sift_akaze_two_view_quality | 51 | 5203 | — | — | 3.2425 | — | 19.06% | 29.507 |
| combined_screen_winners | 51 | 2185 | — | — | 3.7420 | — | 16.85% | 14.591 |
| selected_repeat1 | 51 | 1257 | — | — | 2.9913 | — | 14.31% | 11.195 |
| selected_repeat2 | 51 | 1257 | — | — | 2.9913 | — | 14.31% | 11.614 |
| baseline_seed1 | 51 | 1210 | — | — | 4.0262 | — | 14.22% | 8.113 |
| selected_seed1 | 51 | 1257 | — | — | 2.9913 | — | 14.31% | 8.606 |
| baseline_repeat1 | 51 | 1210 | — | — | 4.0262 | — | 14.22% | 7.553 |
| baseline_repeat2 | 51 | 1210 | — | — | 4.0262 | — | 14.22% | 7.553 |

Geometry metric is Stage 1 Chamfer or milk's cropped nearest-mesh-vertex proxy. Coverage is milk's fraction of GT mesh vertices within 0.05 units of an estimated point; otherwise it is the mean occupied fraction of 8x8 image bins supported by reconstructed tracks. Image coverage is not surface coverage. Changing cached frontend families also changes the observation set.

## Milk

Recorded 34 runs: 34 successful, 0 failed.

Screen-selected candidate: `essential_threshold_0.5`. Median complete reconstruction time across its initial call and repeats: 8.753 seconds. Selection rule: At least baseline camera count, 90% baseline point count and 95% baseline image and mesh coverage. Rank mean normalized camera errors plus geometry proxy (boot: reprojection consistency only). This is exploratory GT tuning, not independent validation.

```json
{
  "track_tolerance": 0.001,
  "min_track_observations": 3,
  "essential_threshold": 0.5,
  "pnp_reprojection_error": 4.0,
  "max_reprojection_error": 5.0,
  "min_triangulation_angle": 1.0,
  "min_pnp_points": 12,
  "min_pnp_inliers": 10,
  "bundle_adjustment": true,
  "bundle_adjustment_max_nfev": 40,
  "max_point_distance_factor": 1.5,
  "two_view_max_reprojection_error": null,
  "two_view_min_triangulation_angle": null,
  "initial_pose_correspondence_dir": null,
  "bundle_adjustment_backend": "ceres",
  "geometry_backend": "cpp",
  "ceres_max_iterations": 100,
  "bundle_adjustment_loss_scale": 2.0,
  "bundle_adjustment_max_reprojection_error": 3.0
}
```

Baseline median reconstruction time (3 calls): 7.561 seconds. Candidate median uses 3 calls. Matching and evaluation are excluded for both.

Measured timing ranges: baseline 7.455–11.266 s; selected 8.712–8.818 s. Calls ran sequentially on the same machine; differences are not a statistical speed guarantee.

Coverage alternative: `min_track_observations_2` retains 4018 points, reaches 75.63% mesh-vertex coverage, and has cropped distance 0.015326. It trades some distance-proxy accuracy against greater vertex coverage compared with the selected candidate.

| Case | Cameras | Points | Translation | Rotation (deg) | Reprojection RMSE (px) | Geometry metric | Coverage | Seconds |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| baseline | 50 | 1729 | 0.062064 | 6.717859 | 2.9487 | 0.016054 | 61.39% | 11.266 |
| bundle_adjustment_loss_scale_0.5 | 50 | 1730 | 0.061977 | 6.734303 | 2.9490 | 0.016085 | 61.41% | 11.666 |
| bundle_adjustment_loss_scale_1 | 50 | 1729 | 0.062038 | 6.727873 | 2.9475 | 0.016104 | 61.33% | 11.110 |
| bundle_adjustment_loss_scale_4 | 50 | 1729 | 0.062076 | 6.712622 | 2.9488 | 0.016027 | 61.43% | 11.975 |
| bundle_adjustment_max_reprojection_error_2 | 50 | 1707 | 0.061631 | 6.699338 | 3.1810 | 0.013769 | 62.15% | 10.621 |
| bundle_adjustment_max_reprojection_error_5 | 50 | 1739 | 0.062267 | 6.703475 | 2.8976 | 0.017080 | 61.55% | 11.575 |
| bundle_adjustment_max_reprojection_error_8 | 50 | 1739 | 0.062665 | 6.722009 | 2.8017 | 0.017894 | 61.40% | 13.430 |
| min_track_observations_2 | 50 | 4018 | 0.061859 | 6.705138 | 3.6014 | 0.015326 | 75.63% | 14.434 |
| min_track_observations_4 | 50 | 955 | 0.060449 | 6.541727 | 2.2142 | 0.012762 | 42.30% | 9.314 |
| min_triangulation_angle_0.5 | 50 | 1748 | 0.064417 | 6.805319 | 2.8203 | 0.016222 | 61.74% | 11.215 |
| min_triangulation_angle_2 | 50 | 1723 | 0.061404 | 6.675115 | 2.7703 | 0.013782 | 58.72% | 11.456 |
| min_triangulation_angle_3 | 50 | 1685 | 0.061586 | 6.618057 | 2.9818 | 0.012882 | 58.80% | 10.952 |
| pnp_reprojection_error_2 | 50 | 1750 | 0.062765 | 6.602594 | 2.6980 | 0.014173 | 60.90% | 26.013 |
| pnp_reprojection_error_3 | 50 | 1733 | 0.062819 | 6.751346 | 2.8793 | 0.014280 | 61.45% | 15.484 |
| pnp_reprojection_error_6 | 50 | 1658 | 0.064288 | 6.889326 | 2.9701 | 0.014716 | 60.00% | 6.421 |
| essential_threshold_0.5 | 50 | 1745 | 0.060950 | 6.536904 | 2.6195 | 0.012784 | 62.20% | 8.818 |
| essential_threshold_2 | 50 | 1728 | 0.062706 | 6.930745 | 3.1736 | 0.009637 | 58.20% | 9.937 |
| max_reprojection_error_3 | 50 | 1605 | 0.063200 | 6.753798 | 2.5345 | 0.017379 | 56.44% | 6.485 |
| max_reprojection_error_8 | 50 | 1808 | 0.061145 | 6.630523 | 2.9263 | 0.014182 | 62.85% | 9.419 |
| max_point_distance_factor_1.25 | 50 | 1712 | 0.061747 | 6.672004 | 1.7171 | 0.015953 | 61.14% | 11.443 |
| max_point_distance_factor_2 | 50 | 1734 | 0.062422 | 6.745352 | 2.9516 | 0.016367 | 61.34% | 8.270 |
| ceres_max_iterations_5 | 50 | 1729 | 0.062906 | 6.928229 | 3.0516 | 0.015076 | 61.34% | 7.795 |
| two_view_quality5 | 50 | 3899 | 0.061836 | 6.690754 | 3.5481 | 0.015529 | 73.89% | 10.878 |
| two_view_quality10 | 50 | 3235 | 0.062105 | 6.714153 | 3.7276 | 0.015547 | 65.72% | 10.998 |
| python_control20 | 50 | 1729 | 0.062124 | 6.740640 | 2.9600 | 0.015953 | 61.38% | 24.196 |
| sift_akaze_track3 | 50 | 4238 | 0.064508 | 6.989070 | 2.9913 | 0.014138 | 73.47% | 18.118 |
| sift_akaze_two_view_quality | 50 | 7959 | 0.064262 | 7.037795 | 3.0567 | 0.013143 | 78.50% | 19.009 |
| combined_screen_winners | 50 | 1677 | 0.065726 | 7.165520 | 2.7242 | 0.015304 | 56.61% | 8.026 |
| selected_repeat1 | 50 | 1745 | 0.060950 | 6.536904 | 2.6195 | 0.012784 | 62.20% | 8.753 |
| selected_repeat2 | 50 | 1745 | 0.060950 | 6.536904 | 2.6195 | 0.012784 | 62.20% | 8.712 |
| baseline_seed1 | 50 | 1729 | 0.062064 | 6.717859 | 2.9487 | 0.016054 | 61.39% | 8.821 |
| selected_seed1 | 50 | 1745 | 0.060950 | 6.536904 | 2.6195 | 0.012784 | 62.20% | 9.649 |
| baseline_repeat1 | 50 | 1729 | 0.062064 | 6.717859 | 2.9487 | 0.016054 | 61.39% | 7.455 |
| baseline_repeat2 | 50 | 1729 | 0.062064 | 6.717859 | 2.9487 | 0.016054 | 61.39% | 7.561 |

Geometry metric is Stage 1 Chamfer or milk's cropped nearest-mesh-vertex proxy. Coverage is milk's fraction of GT mesh vertices within 0.05 units of an estimated point; otherwise it is the mean occupied fraction of 8x8 image bins supported by reconstructed tracks. Image coverage is not surface coverage. Changing cached frontend families also changes the observation set.

## Scene Graph

Full trajectory evaluation on one fixed verified loop. C++ is screening only; Python confirmation required. Ground truth is used only by the official evaluator. Single-call solver timing excludes graph construction, expansion and evaluation.

| Case | Input | Backend | Nodes | NFEV | Official ATE | Seconds | Converged |
|---|---|---|---:|---:|---:|---:|---:|---|
| raw_no_loop | fixed raw | raw | — | — | 3.700730 | 0.000 | — |
| cpp_baseline | fixed raw | cpp | 552 | 50 | 3.362049 | 8.290 | False |
| python_baseline | fixed raw | python | 552 | 50 | 3.030521 | 72.718 | False |
| keyframe_stride_4 | fixed raw | cpp | 1101 | 50 | 3.243243 | 15.158 | False |
| keyframe_stride_16 | fixed raw | cpp | 277 | 50 | 3.460019 | 2.476 | False |
| keyframe_stride_32 | fixed raw | cpp | 140 | 50 | 3.511798 | 1.511 | False |
| optimizer_loss_linear | fixed raw | cpp | 552 | 50 | 3.375762 | 5.686 | False |
| optimizer_loss_huber | fixed raw | cpp | 552 | 50 | 3.662856 | 0.235 | False |
| optimizer_loss_cauchy | fixed raw | cpp | 552 | 50 | 3.684704 | 0.225 | False |
| optimizer_f_scale_0.1 | fixed raw | cpp | 552 | 50 | 3.364537 | 7.180 | False |
| optimizer_f_scale_0.5 | fixed raw | cpp | 552 | 50 | 3.364343 | 6.524 | False |
| optimizer_f_scale_2.0 | fixed raw | cpp | 552 | 50 | 3.379516 | 6.214 | False |
| optimizer_max_nfev_20 | fixed raw | cpp | 552 | 20 | 3.327180 | 4.300 | False |
| optimizer_max_nfev_100 | fixed raw | cpp | 552 | 100 | 3.373446 | 9.758 | False |
| loop_translation_weight_0.1 | fixed raw | cpp | 552 | 50 | 3.370329 | 7.119 | False |
| loop_translation_weight_0.25 | fixed raw | cpp | 552 | 50 | 3.362950 | 6.946 | False |
| loop_translation_weight_1.0 | fixed raw | cpp | 552 | 50 | 3.359789 | 7.994 | False |
| loop_translation_weight_2.0 | fixed raw | cpp | 552 | 50 | 3.558118 | 6.880 | False |
| loop_rotation_weight_0.1 | fixed raw | cpp | 552 | 50 | 3.892788 | 6.318 | False |
| loop_rotation_weight_0.25 | fixed raw | cpp | 552 | 50 | 3.449835 | 6.846 | False |
| loop_rotation_weight_1.0 | fixed raw | cpp | 552 | 50 | 3.349696 | 5.353 | False |
| loop_rotation_weight_2.0 | fixed raw | cpp | 552 | 50 | 3.333826 | 6.682 | False |
| odometry_rotation_weight_0.5 | fixed raw | cpp | 552 | 50 | 3.243625 | 10.102 | False |
| odometry_rotation_weight_2.0 | fixed raw | cpp | 552 | 50 | 3.424792 | 3.681 | False |
| odometry_translation_weight_0.5 | fixed raw | cpp | 552 | 50 | 3.340025 | 4.581 | False |
| odometry_translation_weight_2.0 | fixed raw | cpp | 552 | 50 | 3.262048 | 9.792 | False |
| cpp_combined | fixed raw | cpp | 1101 | 50 | 3.104176 | 18.911 | False |
| python_confirm_0_cpp_combined | fixed raw | python | 1101 | 50 | 3.800912 | 128.899 | False |
| python_confirm_1_keyframe_stride_4 | fixed raw | python | 1101 | 50 | 4.170732 | 127.694 | False |
| python_confirm_2_odometry_rotation_weight_0.5 | fixed raw | python | 552 | 50 | 3.002751 | 69.978 | False |
| python_direct_keyframe_stride_16 | fixed raw | python | 277 | 50 | 3.431926 | 35.002 | False |
| python_direct_keyframe_stride_32 | fixed raw | python | 140 | 50 | 3.523035 | 17.719 | False |
| python_direct_loop_translation_weight_0.25 | fixed raw | python | 552 | 50 | 3.359016 | 63.107 | False |
| python_direct_loop_translation_weight_1.0 | fixed raw | python | 552 | 50 | 3.147192 | 63.247 | False |
| python_direct_loop_rotation_weight_1.0 | fixed raw | python | 552 | 50 | 3.041505 | 64.703 | False |
| python_direct_loop_rotation_weight_2.0 | fixed raw | python | 552 | 50 | 3.166135 | 66.822 | False |
| python_direct_odometry_translation_weight_2.0 | fixed raw | python | 552 | 50 | 3.240622 | 71.444 | False |
| python_direct_optimizer_loss_linear | fixed raw | python | 552 | 50 | 3.353014 | 69.906 | False |
| python_direct_optimizer_f_scale_0.1 | fixed raw | python | 552 | 50 | 2.621383 | 60.942 | False |
| python_direct_optimizer_max_nfev_20 | fixed raw | python | 552 | 20 | 3.490237 | 20.894 | False |
| python_direct_optimizer_max_nfev_100 | fixed raw | python | 552 | 100 | 3.093728 | 147.744 | False |
| frontend_lk_dense_graph_python_direct_optimizer_f_scale_0.1 | lk_dense | python | 552 | 50 | 1.609394 | 59.513 | False |
| python_combined | fixed raw | python | 552 | 50 | 2.848940 | 59.388 | False |

Selected full-trajectory Python candidate: `python_direct_optimizer_f_scale_0.1`. Native screening is not used as proof of Python convergence behavior. Official evaluator logs and full exported trajectories are retained per run.

Combined frontend/graph check: `frontend_lk_dense_graph_python_direct_optimizer_f_scale_0.1`. Selected frontend plus selected graph, reusing the same image-verified loop measurement. Evaluated separately from fixed-input graph screening; no held-out claim.

Lowest measured scene ATE across the Python graph input families: `frontend_lk_dense_graph_python_direct_optimizer_f_scale_0.1`, ATE 1.609394. This comparison includes the combined frontend checks and is distinct from the fixed-input graph selection.

Selected graph recipe:

```json
{
  "loop_method": "pnp",
  "loop_source_window": 40,
  "loop_target_window": 40,
  "loop_source_stride": 8,
  "loop_target_stride": 8,
  "loop_min_separation": 50,
  "max_loop_edges": 1,
  "loop_endpoint_spacing": 0,
  "loop_candidate_mode": "window",
  "retrieval_stride": 16,
  "retrieval_min_separation": 500,
  "retrieval_max_candidates": 50,
  "retrieval_proposal_spacing": 16,
  "retrieval_refine_radius": 4,
  "retrieval_refine_top_k": 10,
  "retrieval_min_similarity": 0.6,
  "retrieval_batch_size": 16,
  "retrieval_device": "auto",
  "loop_matcher": "classical",
  "learned_device": "auto",
  "learned_filter_threshold": 0.1,
  "loop_min_matches": 30,
  "loop_min_depth_points": 12,
  "loop_min_inliers": 15,
  "loop_max_translation": 3.0,
  "loop_max_rotation_degrees": 60.0,
  "loop_max_reprojection_error": 4.0,
  "odometry_translation_weight": 1.0,
  "odometry_rotation_weight": 1.0,
  "loop_translation_weight": 0.5,
  "loop_rotation_weight": 0.5,
  "keyframe_stride": 8,
  "optimizer_loss": "soft_l1",
  "optimizer_f_scale": 0.1,
  "optimizer_max_nfev": 50,
  "optimizer_backend": "python"
}
```

Measured frontend + graph-solve time: baseline 803.730 s, selected 924.783 s. Frontend tracked transitions: baseline 4288, selected 4387, out of 4395. These are single full frontend/solver calls; graph construction, trajectory expansion, output writing and evaluation are excluded.

## Scene Frontend

Exploratory RGB-D screening on three disjoint windows. GT is evaluation-only. RPE is consecutive relative-pose error without scale fitting. Window runtimes include feature-cache misses and are not comparable cold-start full-pipeline benchmarks. Failure fraction includes held poses. Full-run validation, if performed, uses official ATE.

| Case | Tracked fraction | Translation RPE RMSE | Rotation RPE RMSE (deg) |
|---|---:|---:|---:|
| baseline | 1.0000 | 0.009103 | 0.303387 |
| ratio_threshold_0.7 | 1.0000 | 0.008368 | 0.276219 |
| ratio_threshold_0.75 | 1.0000 | 0.008940 | 0.297139 |
| ratio_threshold_0.85 | 1.0000 | 0.009235 | 0.308354 |
| pnp_reprojection_error_1.0 | 1.0000 | 0.008578 | 0.285118 |
| pnp_reprojection_error_3.0 | 1.0000 | 0.008642 | 0.285780 |
| pnp_reprojection_error_4.0 | 1.0000 | 0.008724 | 0.288944 |
| min_depth_0.2 | 1.0000 | 0.009103 | 0.303387 |
| max_depth_4.0 | 1.0000 | 0.009114 | 0.303693 |
| max_depth_12.0 | 1.0000 | 0.009079 | 0.302104 |
| max_translation_step_0.1 | 1.0000 | 0.009103 | 0.303387 |
| max_translation_step_None | 1.0000 | 0.009103 | 0.303387 |
| max_rotation_step_degrees_10.0 | 1.0000 | 0.009103 | 0.303387 |
| max_rotation_step_degrees_40.0 | 1.0000 | 0.009103 | 0.303387 |
| max_features_1500 | 1.0000 | 0.009103 | 0.303387 |
| max_features_3500 | 1.0000 | 0.009103 | 0.303387 |
| feature_mode_sift | 1.0000 | 0.008514 | 0.282159 |
| image_scale_1.0 | 1.0000 | 0.008243 | 0.272308 |
| min_pnp_inliers_20 | 1.0000 | 0.009103 | 0.303387 |
| rgbd | 1.0000 | 0.008647 | 0.285071 |
| rgbd_ransac02 | 1.0000 | 0.008661 | 0.285542 |
| rgbd_ransac10 | 1.0000 | 0.008668 | 0.285723 |
| lk_pnp | 1.0000 | 0.008222 | 0.270617 |
| lk_dense | 1.0000 | 0.008170 | 0.268974 |

Full Baseline:

```json
{
  "label": "baseline",
  "config": {
    "motion_model": "pnp",
    "feature_mode": "akaze",
    "max_features": 2500,
    "image_scale": 0.5,
    "ratio_threshold": 0.8,
    "depth_scale": 1000.0,
    "min_depth": 0.1,
    "max_depth": 8.0,
    "min_matches": 30,
    "min_pnp_points": 12,
    "min_pnp_inliers": 10,
    "pnp_reprojection_error": 2.0,
    "ransac_iterations": 200,
    "ransac_3d_threshold": 0.05,
    "lk_quality_level": 0.01,
    "lk_min_distance": 7.0,
    "max_translation_step": 0.2,
    "max_rotation_step_degrees": 20.0
  },
  "seconds": 731.011560899904,
  "official_ate": 3.70073,
  "compared_pose_pairs": 4396,
  "tracked_frames": 4288,
  "poses": 4396
}
```


Full Validation:

```json
{
  "label": "lk_dense",
  "config": {
    "motion_model": "lk_pnp",
    "feature_mode": "akaze",
    "max_features": 2500,
    "image_scale": 0.5,
    "ratio_threshold": 0.8,
    "depth_scale": 1000.0,
    "min_depth": 0.1,
    "max_depth": 8.0,
    "min_matches": 30,
    "min_pnp_points": 12,
    "min_pnp_inliers": 10,
    "pnp_reprojection_error": 2.0,
    "ransac_iterations": 200,
    "ransac_3d_threshold": 0.05,
    "lk_quality_level": 0.005,
    "lk_min_distance": 4.0,
    "max_translation_step": 0.2,
    "max_rotation_step_degrees": 20.0
  },
  "seconds": 865.269858899992,
  "official_ate": 5.578293,
  "compared_pose_pairs": 4396,
  "tracked_frames": 4387,
  "poses": 4396
}
```

Full-trajectory frontend ATE change: +50.73%. Window selection is accepted as a full-sequence improvement only when this value is negative.

## Scope and limitations

Total recorded configurations/calls so far: 170. Physical camera/depth calibration is fixed. Numerical identity tolerances and minimum data-sufficiency guards are not freely optimized. Matching is checked through existing cached frontend families rather than independent re-extraction for every possible feature/matching setting. Scene loop discovery/retrieval thresholds are not inferred from a fixed graph: the graph sweep reuses its verified loop. The parameter audit records these distinctions explicitly.

Ceres is checked for object bundle adjustment. The current Stage 3 pose graph has Python and C++ backends; it has no Ceres backend.

Dense fusion, Gaussian training, viewer and rendering parameters are outside this SfM/SLAM geometry check; they require distinct surface or photometric validation. Existing promoted outputs and defaults are preserved.

## Reproduction

```powershell
$env:PYTHONPATH = 'src'
python scripts/sweep_reconstruction.py
python scripts/sweep_scene.py
python scripts/sweep_scene_frontend.py
python scripts/sweep_scene.py --confirm-frontend
python scripts/report_hyperparameter_check.py
```

Run phases sequentially to avoid timing contention. Each sweep checkpoints individual runs and resumes using its output directory. Full JSON configurations, raw metrics, exported reconstructions/trajectories, parameter audit and environment metadata are under `outputs/hyperparameter_check/`.
