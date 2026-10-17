# C++ follow-up - 2026-10-09

The three follow-ups are implemented: native graph numerical investigation, configurable Ceres solver/thread options with real-data measurements, and opt-in platform-wheel builds with a hosted CI matrix. Dataset-specific presets preserve the original recipes and expose verified native alternatives. Global defaults remain unchanged.

## Pose graph

The historical Python residual uses OpenCV's matrix-to-angle-axis logarithm. Near identity it zeroes small rotations, so default finite differences can erase odometry rotation derivatives. C++ uses a smooth principal SO(3) log and analytic derivatives. This explains the different bounded-budget trajectories; the native Jacobian is not shown to be incorrect. A new `python_rotation_log='stable'` / `--python-rotation-log stable` uses SciPy's smooth rotation log. Native and stable Python pose packing also retain tiny initial/fixed rotations. The default historical OpenCV mode remains available.

Real-data subgraph diagnostics:

```json
{
  "stable_residual_max_difference": 8.881784197001252e-16,
  "stable_fd_jacobian_max_difference": 9.586238469694308e-10,
  "opencv_fd_jacobian_max_difference": 0.9999997729772637,
  "opencv_zero_rotation_rows": 9,
  "rotation_rows": 12
}
```

Selected scene graph: `lk_dense_cpp_odometry_rotation_weight_2.0`. Three native repeats give ATE 0.817261; median solver time 4.255 s. Stable Python independently gives ATE 0.817260 in 109.049 s. Maximum ATE disagreement: 1e-06.

Relative to the previous joint scene recipe (ATE 1.609394), the new native recipe reduces error by 49.22%. Relative to the original frontend plus Python graph (ATE 3.030521), the reduction is 73.03%. Full trajectories retain all 4,396 poses. The optical-flow frontend is reused from its complete measured run; it is not re-estimated for every graph case.

| Case | ATE | Solver seconds | NFEV | Converged |
|---|---:|---:|---:|---|
| raw_python_stable | 3.365130 | 121.260 | 50 | False |
| raw_cpp_analytic | 3.364537 | 7.196 | 50 | False |
| lk_dense_python_stable | 0.838145 | 122.061 | 50 | False |
| lk_dense_cpp_analytic | 0.838146 | 6.000 | 50 | False |
| lk_dense_cpp_optimizer_f_scale_0.01 | 5.912160 | 0.434 | 50 | False |
| lk_dense_cpp_optimizer_f_scale_0.05 | 0.838338 | 6.147 | 50 | False |
| lk_dense_cpp_optimizer_f_scale_0.5 | 0.836972 | 4.824 | 50 | False |
| lk_dense_cpp_optimizer_f_scale_1.0 | 0.838444 | 7.218 | 50 | False |
| lk_dense_cpp_odometry_rotation_weight_0.1 | 1.069385 | 12.287 | 50 | False |
| lk_dense_cpp_odometry_rotation_weight_0.25 | 0.892564 | 7.354 | 50 | False |
| lk_dense_cpp_odometry_rotation_weight_0.5 | 0.865076 | 7.130 | 50 | False |
| lk_dense_cpp_odometry_rotation_weight_2.0 | 0.817260 | 3.979 | 50 | False |
| lk_dense_cpp_optimizer_max_nfev_100 | 0.836406 | 8.685 | 100 | False |
| selected_cpp_repeat1 | 0.817261 | 4.140 | 50 | False |
| selected_cpp_repeat2 | 0.817261 | 4.255 | 50 | False |
| selected_cpp_repeat3 | 0.817261 | 4.258 | 50 | False |
| selected_python_stable | 0.817260 | 109.049 | 50 | False |

All graph comparisons use one fixed image-verified loop and the same frame identities. Odometry measurements are rebuilt from each frontend's own poses. The reported candidates hit their evaluation budgets; agreement is at that budget, not a claim of full convergence or a global optimum. GT guides selection on this sequence; this is not held-out validation.

## Ceres options

New Python/CLI controls: `ceres_num_threads` / `--ceres-num-threads` and `ceres_linear_solver` / `--ceres-linear-solver` (`dense_schur`, `sparse_schur`, `iterative_schur`). Defaults remain one thread and dense Schur. Sparse Schur explicitly uses Eigen sparse and fails clearly if unavailable. Iterative Schur uses Schur-Jacobi preconditioning. Reports record requested and actual solver/thread settings.

Each option has three complete BA calls on identical selected pre-BA inputs. Setup, output, post-filtering and GT evaluation are outside BA timing.

| Dataset | Solver | Threads | Median BA seconds | Points | Final robust cost | Converged |
|---|---|---:|---:|---:|---:|---|
| box | dense_schur | 1 | 3.208 | 5776 | 7199.288786 | True |
| box | dense_schur | 2 | 3.014 | 5776 | 7199.288786 | True |
| box | dense_schur | 4 | 2.814 | 5776 | 7199.288786 | True |
| box | sparse_schur | 1 | 3.232 | 5776 | 7199.288786 | True |
| box | sparse_schur | 2 | 3.001 | 5776 | 7199.288786 | True |
| box | sparse_schur | 4 | 2.855 | 5776 | 7199.288786 | True |
| box | iterative_schur | 1 | 7.858 | 5776 | 7199.288863 | True |
| box | iterative_schur | 2 | 5.452 | 5776 | 7199.288889 | True |
| box | iterative_schur | 4 | 4.798 | 5776 | 7199.289388 | True |
| boot | dense_schur | 1 | 0.881 | 1257 | 544.999743 | True |
| boot | dense_schur | 2 | 0.754 | 1257 | 544.999743 | True |
| boot | dense_schur | 4 | 0.708 | 1257 | 544.999743 | True |
| boot | sparse_schur | 1 | 0.884 | 1257 | 544.999743 | True |
| boot | sparse_schur | 2 | 0.761 | 1257 | 544.999743 | True |
| boot | sparse_schur | 4 | 0.693 | 1257 | 544.999743 | True |
| boot | iterative_schur | 1 | 2.309 | 1257 | 544.999747 | True |
| boot | iterative_schur | 2 | 1.907 | 1257 | 544.999745 | True |
| boot | iterative_schur | 4 | 1.605 | 1257 | 544.999746 | True |
| milk | dense_schur | 1 | 1.144 | 1745 | 983.001071 | True |
| milk | dense_schur | 2 | 1.058 | 1745 | 983.001071 | True |
| milk | dense_schur | 4 | 1.046 | 1745 | 983.001071 | True |
| milk | sparse_schur | 1 | 1.133 | 1745 | 983.001071 | True |
| milk | sparse_schur | 2 | 1.109 | 1745 | 983.001071 | True |
| milk | sparse_schur | 4 | 1.018 | 1745 | 983.001071 | True |
| milk | iterative_schur | 1 | 1.628 | 1745 | 983.001073 | True |
| milk | iterative_schur | 2 | 1.413 | 1745 | 983.001073 | True |
| milk | iterative_schur | 4 | 1.282 | 1745 | 983.001073 | True |

Selected options were also checked through complete reconstruction with the same seed and cached correspondences. These paired full-pipeline times are single calls; three-call BA medians above provide the narrower solver comparison. Dense/sparse differences are small on these datasets; fastest observed is not universally fastest.

| Dataset | Selected option | Full reconstruction baseline / selected (s) | Max camera difference | Max point distance | Identical IDs |
|---|---|---:|---:|---:|---|
| box | dense_schur_4threads | 10.214 / 9.947 | 8.62e-14 | 9.45e-13 | True |
| boot | sparse_schur_4threads | 8.007 / 7.824 | 1.22e-12 | 4.91e-13 | True |
| milk | sparse_schur_4threads | 8.225 / 8.055 | 7.77e-15 | 2.01e-14 | True |

Threading saves about 2–3% of measured complete reconstruction time here. It improves runtime without a demonstrated accuracy gain. Boot still has no full camera or surface ground truth.

## Native distribution

`setup.py` adds opt-in CMake extensions through the standard Python build interface. Pure, kernel-only and Ceres builds use separate build directories to avoid stale binaries leaking into pure wheels. `build_backend.py` requests pybind11 headers only for native builds unless an existing provider/header path is available. Standard builds remain pure Python.

`.github/workflows/native-wheels.yml` builds Python 3.10/3.11 wheels on Windows, Linux and macOS, bundles shared-library dependencies with delvewheel/auditwheel/delocate, includes dependency license notices, and imports/tests installed repaired wheels from outside the checkout. Artifacts are uploaded; nothing is published to a package index. Current targets are glibc 2.39+ on Linux and macOS 14+. Hosted jobs have not been run in this local session.

The local package index cannot supply delvewheel, so local Windows Ceres wheels are unrepaired and still require the installed Ceres dependency DLL directory. They must not be described as self-contained redistributable wheels. The hosted Windows job performs DLL repair and tests the repaired result.

## Presets and reproduction

`configs/presets/` contains the original object recipes, their pipeline-verified fast variants, milk's coverage alternative, the new native scene recipe, and stable/legacy Python scene alternatives. Object recipes retain their exact cached correspondence family. Scene recipes retain their verified loop and exact frame indices; they are specific to this dataset.

```powershell
$env:PYTHONPATH = 'src'
python scripts/diagnose_native_graph.py --confirm-selected
python scripts/benchmark_ceres_options.py --confirm-pipeline
python scripts/save_selected_presets.py
python scripts/report_cpp_followup.py

# Complete object reconstruction:
python scripts/run_preset.py --preset configs/presets/stage2_milk_fast.json --output outputs/milk_fast

# Reuse the already validated complete scene frontend:
python scripts/run_preset.py --preset configs/presets/stage3_scene.json --trajectory outputs/hyperparameter_check/scene_frontend/full_validation/trajectory.txt --output outputs/scene_native
# Omit --trajectory to recompute the entire frontend.
```

Native wheel build (same Python ABI/compiler as target):

```powershell
$env:SFM_BUILD_NATIVE = '1'
$env:SFM_WITH_CERES = '1'
$env:SFM_CERES_PREFIX = (Resolve-Path outputs/ceres_deps).Path
python -m build --wheel
# Set SFM_WITH_CERES=0 for kernels only. Clear both flags for pure Python.
```

Set `SFM_CMAKE` if cmake is not on PATH. `PYBIND11_INCLUDE_DIR` can supply existing headers. `--no-isolation` requires installed build dependencies or that explicit header provider.

## Local validation

Publication follow-up (2026-10-17): native source, bindings, build scripts,
presets and these reports are now included in Git. The hosted macOS wheel uses
a pinned Ceres 2.2 source build with Eigen sparse support and miniglog, without
SuiteSparse/LAPACK/OpenMP dependencies. This avoids the duplicate OpenMP runtime
abort observed with the initial repaired conda Ceres wheel. Linux and Windows
continue to use the conda Ceres dependency profile. The original benchmark
measurements above are unchanged; they are not macOS performance claims.
The JSON below preserves the earlier local validation record.

```json
{
  "source_tests": "160 passed in 11.39 s",
  "installed_kernels_wheel_tests": "17 passed in 2.21 s",
  "installed_ceres_wheel_tests": "31 passed in 2.34 s",
  "installed_pure_wheel": "default Python BA, no native modules, new CLI options: passed",
  "wheel_archive_checks": "pure=0, kernels=1, Ceres=2 extensions; platform tags verified",
  "preset_smokes": {
    "scene_poses": 4396,
    "scene_ate": 0.817261,
    "milk_cameras": 50,
    "milk_points": 1745
  },
  "hosted_ci": "configured, not run in this session",
  "windows_dll_repair": "not locally validated; delvewheel unavailable from configured package index; local Ceres wheel tested with SFM_CERES_DLL_DIR"
}
```
