# Optional C++ integration

The `cpp` backend uses C++17 residual and analytic-Jacobian kernels with
pybind11 and the existing SciPy TRF solver. The optional `ceres` backend solves
bundle adjustment with Ceres 2.2 and Eigen. The two extensions are separate:
the geometry and analytic kernels do not require Ceres or its dependency DLLs.

Python remains the default. An explicit native selection raises an actionable
error when the extension is missing; it never silently falls back.

The [subsequent C++ follow-up](CPP_FOLLOWUP.md) diagnoses the rotation-log
discrepancy, validates a native graph preset for the optical-flow scene frontend,
benchmarks Ceres solver/thread controls, and adds platform-wheel builds. The
original-input graph caveat below still applies; global defaults are unchanged.

See [measured results and promotion decisions](CPP_RESULTS.md). In particular,
the native pose graph regresses ATE on the original raw Stage 3 input. The
follow-up validates a native preset on the optical-flow frontend instead.
Keep Python for the historical promoted trajectory; use the matching dataset
preset for the new native recipe.
The subsequent [Ceres comparison](CERES_RESULTS.md) covers BA runtime,
convergence, and the mixed geometry-accuracy results.

## Build

Requirements: a C++17 compiler, CMake 3.19+, Python development headers, and
pybind11 headers (a standalone install or those bundled with PyTorch).
Use the same Python interpreter for compilation, tests, and benchmarks.

Windows / MSVC:

```powershell
.\scripts\build_native.ps1 -Python "C:\path\to\python.exe" -CMake "C:\path\to\cmake.exe"
```

Linux/macOS:

```bash
cmake -S native -B outputs/native_build -DCMAKE_BUILD_TYPE=Release -DPython_EXECUTABLE="$(which python)"
cmake --build outputs/native_build --parallel 2
cmake --install outputs/native_build --prefix "$PWD/src/sfm_reconstruction"
```

If discovery is unavailable, pass `-DPYBIND11_INCLUDE_DIR=/path/to/include`.
The compiled extension is local, ignored by Git, and must be rebuilt for each
Python ABI/platform. The standard Python package build remains unchanged.

### Optional Ceres build

Install dependencies externally if this session cannot access package servers:

```powershell
conda create --prefix outputs/ceres_deps --override-channels -c conda-forge "ceres-solver=2.2.0=cpu*" eigen --yes
.\scripts\build_native.ps1 -Python "C:\path\to\python.exe" -CMake "C:\path\to\cmake.exe" -CeresPrefix outputs/ceres_deps
```

For manual CMake builds, add `-DSFM_WITH_CERES=ON` and set
`-DCMAKE_PREFIX_PATH` to the dependency prefix (`outputs/ceres_deps/Library`
on Windows). Windows runtime loading looks in that directory's `bin` folder;
set `SFM_CERES_DLL_DIR` when using another location. The DLL directory is added
only to the current Python process. The ordinary build leaves Ceres disabled.

## Candidate 1: bundle adjustment

Select `--bundle-adjustment-backend cpp` in `sfm-reconstruct`, or pass
`backend="cpp"` to `bundle_adjust`. The native callback evaluates projection
residuals and analytic derivatives for rotation vectors, translations, and
points. NumPy buffers cross the binding boundary once per evaluation; the
kernel releases the GIL. Python assembles the sparse Jacobian and runs SciPy.

The same observation gate is applied before solving. Both initialization
cameras remain fixed, unused points are preserved, and post-BA filtering stays
in the existing reconstruction pipeline. Results should be compared with
tolerances: exact derivatives can follow a different convergence path than
SciPy's finite differences at a bounded evaluation budget.

Select `--bundle-adjustment-backend ceres` to use Ceres instead of SciPy.
`--ceres-max-iterations` defaults to 100 and is independent of SciPy's
`--bundle-adjustment-max-nfev`: iteration and function-evaluation limits are
different budgets. By default the solver uses double precision, one Ceres thread,
`DENSE_SCHUR`, and function/gradient/parameter tolerances of `1e-8`.
`--ceres-num-threads` and `--ceres-linear-solver` expose measured alternatives;
see the follow-up for their complete-pipeline validation and timing boundaries.
Each image coordinate has a separate scalar residual block with
`SoftLOneLoss(2.0)`, matching SciPy's coordinate-wise objective. It uses the same
observation gate and fixed cameras as the other BA backends. Ceres is available
only for BA; pose graphs and geometry still accept `python` or `cpp`.

`--bundle-adjustment-loss-scale` controls that robust-loss scale in pixels
(default 2, finite and positive). Lower values reduce the relative influence
of large residuals; they change the objective, not just solver accuracy. The
same option applies to Python and C++/SciPy BA. The Python API argument is
`bundle_adjust(..., loss_scale=...)`. Ceres iteration limits control stopping;
increasing the limit after convergence does not cause additional refinement.

`--bundle-adjustment-max-reprojection-error` overrides only the pre-solve BA
observation gate in pixels. By default it inherits `--max-reprojection-error`.
The independent override leaves initialization and post-BA filtering unchanged,
matching the controlled tuning experiment. It selects observations using their
initial reprojection errors; it is not a final residual bound.

`scripts/tune_ceres.py` runs a small exploratory sweep of scales, observation
gates, and iteration caps on a shared pre-BA input. It writes every candidate
separately, retains post-filter point counts/IDs and raw mesh proxies, and
keeps the post-filter settings fixed. Its single-call times are exploratory,
not replacement median benchmarks. Ground-truth metrics guide this experiment
only; tuned settings require checks on other data before becoming defaults.

`--min-triangulation-angle` exposes the existing reconstruction threshold in
degrees (default 1). It applies during triangulation and filtering. The separate
`scripts/tune_milk_geometry.py` pipeline sweep tests track length, angle,
distance cutoff and PnP threshold on cached correspondences. Its report adds
reverse mesh-to-point distances and a vertex coverage proxy to distinguish
cleaner points from loss of scene coverage. See the [milk pipeline tuning
results](CERES_RESULTS.md#reconstruction-parameter-follow-up).

Add `--include-ceres --ceres-max-iterations 100` to a bundle benchmark to compare
all three backends on the same inputs. Reports include solver convergence,
final accepted-observation robust cost, and complete-call timing. The existing
`soft_l1_cost` and reprojection RMSE fields evaluate all observations, including
those excluded by the initial solver gate; they are evaluation metrics rather
than the solver's gated objective.

## Candidate 2: pose graph

Select `--optimizer-backend cpp` in `sfm-stage3-setup`, or use
`PoseGraphConfig(optimizer_backend="cpp")`. The kernel evaluates the existing
`inverse(measured) * inverse(target_pose) * source_pose` error with weighted
principal angle-axis rotation and translation residuals. Analytic SO(3)
Jacobians feed the unchanged SciPy solver. The first graph node stays fixed;
noncontiguous node IDs and keyframe expansion retain their existing semantics.

Exact derivatives and a smooth small-angle logarithm can change convergence
relative to OpenCV's rotation-log precision and finite differences. Compare
the objective, evaluation count, and official trajectory ATE together.

## Candidate 3: geometry

Select `--geometry-backend cpp` in `sfm-reconstruct`. Projection, reprojection
errors, positive-depth tests, and triangulation angles have native kernels.
Triangulation, essential-matrix estimation, and PnP retain their existing
compiled OpenCV implementation. Individual geometry functions also accept
`backend="cpp"`. Geometry selection is independent of BA selection.

Tests cover strided/empty arrays, skewed intrinsics, degenerate rays, depth
decisions, and point-filter selection parity. Scalar binding costs are included
in the realistic point-filter comparison, alongside large-batch microbenchmarks.

## Evaluation

```powershell
$env:PYTHONPATH = "src"
python -m pytest tests/test_native_backend.py -q -p no:cacheprovider
python scripts/benchmark_native.py --candidate bundle --repeats 3 --output outputs/native_benchmarks/bundle_synthetic/results.json
python scripts/benchmark_native.py --candidate graph --max-nfev 50 --repeats 3 --output outputs/native_benchmarks/graph_synthetic/results.json
python scripts/benchmark_native.py --candidate geometry --repeats 3 --output outputs/native_benchmarks/geometry_synthetic/results.json
```

Dataset runs accept `--dataset`, optional `--correspondences`,
`--min-track-observations`, and `--max-point-distance-factor`. They reconstruct
one shared pre-BA input and time each backend on that identical input. Setup,
evaluation, and output writing are excluded from the timings; Python packing,
native conversion, sparse Jacobian assembly, and solver time are included.
Each timing is a complete call, using unchanged initial inputs.

Reports contain individual timings, medians, solution differences, all-observation
reprojection metrics, pose metrics when GT exists, and Chamfer/mesh-proxy metrics
when the relevant dataset files exist. Separate output folders preserve the
promoted artifacts. Synthetic tests compare analytic derivatives against
independent finite differences and check recovered geometry and fixed poses.

For the real Stage 3 comparison, pass `--candidate graph --dataset /path/to/stage3
--trajectory /path/to/raw_vo.txt --max-nfev 50`. The benchmark verifies the
documented `4387 -> 32` edge once, saves the shared graph as `graph_input.npz`,
and compares both optimizers on identical nodes/edges. This dataset-specific
recipe keeps retrieval and geometric verification outside optimizer timing.
Both exported full trajectories are checked with the supplied official evaluator.

For geometry, dataset options time complete point filtering on one shared
reconstruction and require identical accepted point IDs. Every report retains
the synthetic batch-size sweep as well.
