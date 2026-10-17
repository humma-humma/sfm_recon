# C++ backend evaluation — 2026-10-09

The three optional integrations are implemented and measured. Bundle adjustment
and geometry show useful speedups with small numerical differences. The
analytic pose-graph backend is faster but fails the trajectory-accuracy
promotion gate on the documented Stage 3 input. Python remains the default.

The initial evaluation below uses C++17/pybind11 kernels with the existing SciPy
TRF solver. After dependencies were installed externally, a separate Ceres
backend was added; see [the subsequent Ceres comparison](CERES_RESULTS.md).
See [build and usage](CPP_INTEGRATION.md).

The later [hyperparameter check](HYPERPARAMETER_CHECK.md) screens configuration
changes across all stages and confirms scene candidates with Python. The
backend comparisons below retain their original settings.

## Runtime and quality

Median of three complete calls on identical inputs; setup and evaluation are
excluded. These are component timings, not end-to-end reconstruction speedups.

| Candidate / input | Python | C++ | Speedup | Result |
|---|---:|---:|---:|---|
| Bundle adjustment, Stage 1 box | 39.509 s | 8.499 s | 4.65× | 46 cameras, 5,746 points retained in both |
| Bundle adjustment, Stage 2 milk | 16.845 s | 4.553 s | 3.70× | 50 cameras, 1,739 points retained in both |
| Pose graph, documented Stage 3 input | 69.299 s | 6.840 s | 10.13× | ATE worsens; experimental only |
| Geometry point filtering, Stage 2 milk | 0.909 s | 0.231 s | 3.94× | Identical 1,739 accepted point IDs |

### Bundle adjustment

Both backends use the same accepted observations, fixed initialization cameras,
`soft_l1` loss with scale 2, and 20 function-evaluation budget. The C++ path
provides analytic Jacobians; Python uses sparse finite differences.

| Metric | Python | C++ |
|---|---:|---:|
| Stage 1 mean rotation error | 0.875133° | 0.875489° |
| Stage 1 mean translation error | 0.0666230 | 0.0666426 |
| Stage 1 Chamfer | 0.512419980 | 0.512453530 |
| Milk mean rotation error | 6.778942° | 6.778815° |
| Milk mean translation error | 0.062537264 | 0.062536755 |
| Milk mean nearest-mesh-vertex proxy | 0.020403068 | 0.020404125 |

Stage 1 Chamfer increases by about 0.0065%; the result is a performance
improvement with approximate numerical parity, not an accuracy improvement.
Fixed cameras remain identical and both post-filtered point counts match.
The synthetic 8-camera / 300-point run takes 2.385 s versus 0.394 s (6.05×),
with maximum camera-matrix difference `6.72e-7` and point difference `1.09e-6`.

### Pose graph

The authoritative comparison uses
`outputs/stage3_pose_graph_full/estimated_camera_trajectory.txt`, the same raw
input documented for the promoted stride-8 solution. The verified loop is
`4387 -> 32` (456 inliers, median error 0.499778 px). The graph has 552 nodes
and 552 edges; the exported full trajectory has 4,396 poses.

| Metric | Python | C++ |
|---|---:|---:|
| Initial unrobust cost | 8.011531268 | 8.011531268 |
| Final robust graph cost | 0.008856890 | 0.007621297 |
| Official Horn-aligned ATE | 3.030521 | 3.362049 |
| Function evaluations | 50 | 50 |
| Converged | No; budget exhausted | No; budget exhausted |

The Python run reproduces the promoted ATE exactly. The native run decreases
the graph objective but worsens ATE by about 10.94%, and its maximum pose-matrix
element difference is 2.899. This fails accuracy parity. Lower graph cost must
not be presented as improved trajectory accuracy. Keep `--optimizer-backend
python` for promoted runs; `cpp` is an explicit experiment.

The synthetic 100-node graph takes 2.733 s versus 0.110 s (24.88×). It converges
in 17 versus 4 evaluations, so that speedup includes convergence differences.
Both backends pass known-geometry recovery and fixed-first-pose tests.

A supplemental comparison used the earlier guarded VO file. It takes 69.757 s
versus 7.082 s (9.85×), with ATE 2.286473 versus 2.264734. This different input
does not reproduce the promoted benchmark and is reported separately; it does
not justify promoting the native solver on the documented input.

### Geometry

The real filter comparison includes scalar Python-to-C++ binding overhead and
requires identical accepted track IDs. No point coordinates or camera poses
are changed by these kernels.

For batches of 100,000 points:

| Kernel | Python | C++ | Speedup |
|---|---:|---:|---:|
| Projection | 2.316 ms | 0.818 ms | 2.83× |
| Reprojection error | 3.335 ms | 1.304 ms | 2.56× |
| Positive-depth test | 1.301 ms | 0.144 ms | 9.04× |
| Triangulation angles | 11.954 ms | 1.992 ms | 6.00× |

The largest numerical difference across this batch is `2.21e-13`; depth
decisions match exactly. Individual microbenchmark samples average 10 calls,
with three samples. The report also includes batch sizes 1, 100, and 10,000.
Triangulation, PnP, and essential-matrix estimation already use compiled OpenCV
and were not rewritten.

## Reproduction and provenance

Machine: Windows 10.0.26200, Intel family 6/model 158; MSVC 19.43, Release `/O2`.
Runtime: Python 3.10.13, NumPy 1.23.5, SciPy 1.15.3, OpenCV 4.13.0. The native
loops use double precision, release the GIL, and run on the CPU.

Run from the repository root with `PYTHONPATH=src`:

```powershell
python scripts/benchmark_native.py --candidate bundle --dataset "../Experiments/Stage_1_Data_ver._4/Stage_1_Data_ver_4/stage1/box" --repeats 3 --output outputs/native_benchmarks/bundle_stage1/results.json
python scripts/benchmark_native.py --candidate bundle --dataset "../Experiments/Stage_2_Data/stage2/milk" --correspondences outputs/stage2_milk_masked_track3_ba/matching_cache/correspondences --min-track-observations 3 --max-point-distance-factor 1.5 --repeats 3 --output outputs/native_benchmarks/bundle_milk/results.json
python scripts/benchmark_native.py --candidate graph --dataset "../Experiments/Stage_3_Data/stage3" --trajectory outputs/stage3_pose_graph_full/estimated_camera_trajectory.txt --max-nfev 50 --repeats 3 --output outputs/native_benchmarks/graph_stage3_promoted/results.json
python scripts/benchmark_native.py --candidate geometry --dataset "../Experiments/Stage_2_Data/stage2/milk" --correspondences outputs/stage2_milk_masked_track3_ba/matching_cache/correspondences --min-track-observations 3 --max-point-distance-factor 1.5 --repeats 3 --output outputs/native_benchmarks/geometry_milk/results.json
```

Raw JSONs, comparison reconstructions/trajectories, evaluator stdout, and shared
graph snapshots remain under ignored `outputs/native_benchmarks/`. Existing
promoted outputs were preserved. Both optimizer timings include Python packing,
native conversion, Jacobian assembly, and SciPy solving; geometry timings include
the entire point-filter call. Builds, feature extraction, reconstruction setup,
evaluation, and file writing are excluded. Backends run sequentially.

Validation: 138 tests pass with the compiled extension, including 15 new tests
for independent derivative checks, known-geometry recovery, fixed poses, cached
array ownership, degenerate/strided/empty geometry, filtering parity, and combined
backend reconstruction. Source and pure-Python wheel builds succeed; the source
distribution includes the optional native build sources.
With the extension deliberately unavailable, the portable suite also passes:
`123 passed, 1 skipped` (the optional native test module).

## Recommendation

Use `--bundle-adjustment-backend cpp --geometry-backend cpp` for opt-in performance
runs after reviewing the stated numerical tolerances. Keep the Python pose-graph
backend for the current promoted trajectory. The subsequent Ceres BA comparison
is reported separately; graph-model changes still require the same official
trajectory metrics before promotion.
