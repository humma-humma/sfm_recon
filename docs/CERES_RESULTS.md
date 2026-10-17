# Ceres bundle adjustment evaluation - 2026-10-09

Ceres 2.2.0 is installed, built, and available as the optional `ceres` bundle
adjustment backend. It converges faster than the existing SciPy backends on
these inputs. Stage 1 accuracy improves, while milk has mixed accuracy results.
Python remains the default; milk's geometry proxy does not justify automatic
promotion of Ceres for every dataset.

For the subsequent cross-stage parameter sweep, selected recipes, coverage
tradeoffs and complete reconstruction timings, see the
[hyperparameter check](HYPERPARAMETER_CHECK.md). The comparisons below retain
their original inputs and settings.

## Runtime and accuracy

Median of three complete BA calls on identical pre-BA inputs. Setup, evaluation,
and output writing are excluded. Python observation gating, packing, conversion,
and solving are included. Ceres uses one thread, Eigen dense linear algebra,
`DENSE_SCHUR`, and automatic derivatives. The robust objective is the same
coordinate-wise soft-L1 loss with scale 2; initialization cameras stay fixed.

| Input | Python/SciPy | Ceres | Speedup |
|---|---:|---:|---:|
| Stage 1 box: 46 cameras, 5,746 points | 39.524 s | 3.327 s | 11.88x |
| Stage 2 milk: 50 cameras, 1,739 points | 15.898 s | 1.216 s | 13.07x |

C++/SciPy medians are 9.026 s for Stage 1 and 4.071 s for milk; Ceres is
2.71x and 3.35x faster respectively. Stage 1 uses a separate clean timing check
because the original C++ timing overlapped a compiler process. Its objective
and accuracy metrics reproduce the initial C++ run exactly.

| Metric (lower is better) | Python/SciPy | C++/SciPy | Ceres |
|---|---:|---:|---:|
| Stage 1 mean rotation error, degrees | 0.875133 | 0.875489 | 0.561156 |
| Stage 1 mean translation error | 0.0666230 | 0.0666426 | 0.0334293 |
| Stage 1 Chamfer | 0.512420 | 0.512454 | 0.451907 |
| Milk mean rotation error, degrees | 6.778942 | 6.778815 | 6.703475 |
| Milk mean translation error | 0.0625373 | 0.0625368 | 0.0622666 |
| Milk mean nearest-mesh-vertex proxy | 0.0204031 | 0.0204041 | 0.0213772 |

Compared with the bounded Python run, Stage 1 translation error improves by
49.8% and Chamfer by 11.8%. Milk translation error improves by about 0.43%,
but its mean mesh-distance proxy worsens by 4.77%. The proxy is a nearest-vertex
metric, not complete surface reconstruction accuracy. All backends retain the
same post-filtered point counts in both datasets.

## Convergence and interpretation

SciPy uses a 20-function-evaluation cap for this comparison, matching the prior
native-kernel evaluation. Both SciPy backends exhaust that budget. Ceres has a
separate 100-iteration cap and converges in 12 recorded iterations on each
dataset, with function/gradient/parameter tolerances of `1e-8`.
Iterations and function evaluations are not equivalent budgets, and the
solvers' stopping conditions are not identical. These measurements demonstrate
practical runtime and convergence benefits at the tested settings; they do not
establish that Ceres always gives a better converged optimum than SciPy.

| Gated robust solver cost | Python/SciPy | C++/SciPy | Ceres |
|---|---:|---:|---:|
| Stage 1 | 7192.42195 | 7192.42768 | 7187.51769 |
| Milk | 1491.42095 | 1491.41794 | 1479.31675 |

Lower reprojection objective does not guarantee better ground-truth geometry;
milk demonstrates this directly. Existing pose-graph recommendations are
unchanged: use Python for the promoted Stage 3 trajectory. Ceres was integrated
only into bundle adjustment.

A supplemental Stage 1 C++/SciPy run with 100 evaluations takes 23.999 s
(one timing sample), still exhausts its budget, and reaches cost 7190.16506,
translation error 0.0561695, and Chamfer 0.495449. It moves toward Ceres's
solution but remains behind at that budget. This supports faster convergence
as the main benefit; it does not imply a different reconstruction objective.
The script and JSON are `outputs/ceres_benchmarks/recheck_stage1.py` and
`outputs/ceres_benchmarks/stage1_scipy_check.json`.

## Reproduction and artifacts

See [build and usage](CPP_INTEGRATION.md) for dependency installation, building,
Windows DLL loading, and CLI options. Run with the same Python interpreter used
to compile the extensions and `PYTHONPATH=src`:

```powershell
python scripts/benchmark_native.py --candidate bundle --dataset "../Experiments/Stage_1_Data_ver._4/Stage_1_Data_ver_4/stage1/box" --include-ceres --repeats 3 --output outputs/ceres_benchmarks/bundle_stage1/results.json
python scripts/benchmark_native.py --candidate bundle --dataset "../Experiments/Stage_2_Data/stage2/milk" --correspondences outputs/stage2_milk_masked_track3_ba/matching_cache/correspondences --min-track-observations 3 --max-point-distance-factor 1.5 --include-ceres --repeats 3 --output outputs/ceres_benchmarks/bundle_milk/results.json
```

Raw reports and separate reconstruction exports for each backend are under
`outputs/ceres_benchmarks/`. Existing promoted outputs are preserved.
The independent clean Stage 1 check script and results are also kept there.
Machine and Python dependency versions are embedded in the main JSON reports.

Validation: 141 tests pass with both extensions installed. The three new Ceres
tests verify scalar robust-objective equality (including skewed intrinsics),
known-solution agreement with SciPy, fixed-camera and input preservation,
unused points, empty problems, and invalid observation indices. With both
extensions deliberately unavailable, 123 tests pass and four optional tests
skip. Ceres remains optional for ordinary Python use.

Enable an experiment with `--bundle-adjustment-backend ceres
--ceres-max-iterations 100`. Retain the Python default until the dataset's
ground-truth camera and geometry metrics support promotion.

## Parameter tuning follow-up

An exploratory milk sweep tested 12 configurations on one shared pre-BA input. Ground truth was used only for evaluation and setting selection, never fed into the solver. Post-filter settings stayed fixed. These results are dataset-specific tuning, not held-out validation.

| Loss scale (px) | BA gate (px) | Iteration cap | Retained points | Mean translation error | Mean mesh proxy | BA seconds |
|---:|---:|---:|---:|---:|---:|---:|
| 2 | 5 | 100 | 1739 | 0.06226655 | 0.02137719 | 1.316 |
| 0.5 | 5 | 100 | 1739 | 0.06202392 | 0.02130217 | 1.857 |
| 1 | 5 | 100 | 1738 | 0.06217882 | 0.02138305 | 1.270 |
| 4 | 5 | 100 | 1739 | 0.06231276 | 0.02139930 | 1.232 |
| 2 | 3 | 100 | 1729 | 0.06206418 | 0.02036737 | 1.111 |
| 0.5 | 3 | 100 | 1730 | 0.06197746 | 0.02042198 | 1.349 |
| 1 | 3 | 100 | 1729 | 0.06203833 | 0.02042721 | 1.145 |
| 4 | 3 | 100 | 1729 | 0.06207619 | 0.02033571 | 1.227 |
| 2 | 8 | 100 | 1739 | 0.06266480 | 0.02221852 | 1.221 |
| 2 | 5 | 3 | 1739 | 0.06294810 | 0.01901451 | 0.988 |
| 2 | 5 | 5 | 1739 | 0.06211759 | 0.02048527 | 1.042 |
| 2 | 5 | 8 | 1739 | 0.06225479 | 0.02134590 | 1.107 |

Timings above are single complete calls, not three-run medians. Solver costs cannot be compared directly across different loss scales or observation gates. All-observation soft-L1 evaluation uses the original scale 2 for consistency.

The preferred conservative milk experiment keeps loss scale 2 and uses a 3-pixel BA observation gate. Its mesh proxy is 0.02036737 versus Python/SciPy 0.02040307 (about 0.17% better) and original Ceres 0.02137719 (about 4.72% better). Translation error improves to 0.06206418. It retains 1,729 of the original 1,739 points; it is not exact coverage parity. On all original 1,739 points before final filtering, its proxy is 0.02032424, so dropping the ten points does not explain the improvement.

At gate 3, scale 4 gives a slightly smaller proxy (0.02033571), but the change is small and translation is marginally worse than scale 2. Loss-scale changes alone at gate 5 do not eliminate the original geometry regression. A looser 8-pixel gate makes the proxy worse.

Early stopping is a tradeoff: a five-iteration cap retains every point and gives proxy 0.02048527 (about 0.40% worse than Python) with translation error 0.06211759. A three-iteration cap gives the best mesh proxy, 0.01901451, but worsens camera rotation and translation errors. Both runs stop before convergence; optimizing only the mesh proxy would be misleading. More iterations beyond the existing converged solution are not the remedy.

The Stage 1 cross-check confirms that the milk setting should not become a global default:

| Stage 1 setting | Mean translation error | Chamfer | Retained points |
|---|---:|---:|---:|
| scale 2, gate 5 | 0.03342927 | 0.45190695 | 5746 |
| scale 2, gate 3 | 0.05841280 | 0.46903660 | 5728 |
| scale 4, gate 3 | 0.06026163 | 0.47223220 | 5728 |

Keep gate 5 / scale 2 for Stage 1. For a milk experiment, add these flags to the existing track-3, distance-factor-1.5 configuration:

```powershell
--bundle-adjustment-backend ceres --bundle-adjustment-loss-scale 2 --bundle-adjustment-max-reprojection-error 3 --ceres-max-iterations 100
```

The new BA-specific gate override changes only solver observation selection. Using `--max-reprojection-error 3` instead would also change reconstruction filtering and would not reproduce this controlled experiment. Defaults remain unchanged.

Reproduce the sweep:

```powershell
python scripts/tune_ceres.py --dataset "../Experiments/Stage_2_Data/stage2/milk" --correspondences outputs/stage2_milk_masked_track3_ba/matching_cache/correspondences --min-track-observations 3 --max-point-distance-factor 1.5 --output outputs/ceres_tuning/milk/results.json
```

Every candidate is exported separately under `outputs/ceres_tuning/milk/`. The Stage 1 check script and report are `outputs/ceres_tuning/check_stage1.py` and `outputs/ceres_tuning/stage1_check.json`. Tests now cover robust-objective agreement with SciPy at scales 0.5, 2, and 4. See the official [Ceres loss-function documentation](https://ceres-solver.readthedocs.io/latest/nnls_modeling.html#lossfunction) for robust-loss scaling semantics.

## Reconstruction-parameter follow-up

Nine milk configurations were evaluated using the same cached masked correspondences, Ceres loss scale 2, and BA-only observation gate 3. Each reruns the complete reconstruction with OpenCV seed 0, so changing track, triangulation, distance or PnP settings can change initialization, registration and the observations supplied to BA. This is a pipeline comparison, not a solver-only comparison. The baseline reproduces the prior tuned milk result exactly. Ground truth is evaluation-only.

| Configuration | Points | Mean translation error | Full mesh proxy | Cropped mesh proxy | Mesh-vertex coverage at 0.05 | Pipeline seconds |
|---|---:|---:|---:|---:|---:|---:|
| baseline | 1729 | 0.06206418 | 0.02036737 | 0.01605419 | 61.39% | 9.637 |
| track4 | 955 | 0.06044948 | 0.01368952 | 0.01276171 | 42.30% | 8.279 |
| track5 | 613 | 0.06355670 | 0.02044144 | 0.02044144 | 28.76% | 6.897 |
| angle2 | 1723 | 0.06140405 | 0.01761103 | 0.01378214 | 58.72% | 9.525 |
| angle3 | 1685 | 0.06158628 | 0.01709246 | 0.01288213 | 58.80% | 9.203 |
| distance1.25 | 1712 | 0.06174683 | 0.01595256 | 0.01595256 | 61.14% | 9.431 |
| distance2 | 1734 | 0.06242188 | 0.02252300 | 0.01636669 | 61.34% | 9.318 |
| pnp2 | 1750 | 0.06276475 | 0.01705303 | 0.01417291 | 60.90% | 26.151 |
| angle2_distance1.25 | 1710 | 0.06113061 | 0.01343761 | 0.01343761 | 58.75% | 9.972 |

All nine runs register all 50 cameras. Timings are single complete pipeline calls, including track building, initialization, registration, triangulation, BA and filtering; feature matching and evaluation are excluded. They are not comparable to the earlier BA-only timings.

The baseline uses minimum track length 3, triangulation angle 1 degree, distance factor 1.5, and PnP threshold 4 pixels. The combined candidate uses triangulation angle 2 degrees and distance factor 1.25, with the other baseline values unchanged.

The combined candidate retains 1,710 of 1,729 points. Its full mesh proxy improves about 34%, but part of that is simply removal of distant outliers. The cropped object-region proxy improves about 16.3%, from 0.01605419 to 0.01343761; translation error improves about 1.5%, from 0.06206418 to 0.06113061. The fraction of GT mesh vertices within distance 0.05 of a reconstructed point falls from 61.39% to 58.75%, and the reverse mesh-to-points mean distance increases from 0.05791664 to 0.06144985. This is an accuracy/coverage tradeoff, not an unconditional improvement. Coverage is vertex-density dependent and uses the existing scale-only proxy alignment.

Distance factor 1.25 alone is the conservative option for preserving coverage: 61.14% versus baseline 61.39%, retaining 1,712 points. Its cropped proxy improves only about 0.63%; most of its full-proxy gain is outlier removal. Angle 2 alone gives a stronger object-region accuracy gain but reduces the coverage proxy to 58.72%.

Four-view tracks improve camera accuracy but drop to 955 points and 42.30% coverage, so they are suitable only if a much sparser result is acceptable. Five-view tracks are worse. A 2-pixel PnP threshold takes 26.15 seconds versus baseline 9.64 seconds, improves rotation and the mesh proxy, but worsens mean translation; it is not the preferred setting.

For the combined milk experiment, add these options to the existing masked-correspondence recipe:

```text
--bundle-adjustment-backend ceres
--bundle-adjustment-max-reprojection-error 3
--min-track-observations 3
--min-triangulation-angle 2
--max-point-distance-factor 1.25
```

The triangulation-angle option is now exposed in the CLI and still defaults to 1 degree. All other defaults also remain unchanged. These settings have not been validated on Stage 1 or held-out objects and should remain milk-specific experiments.

Reproduce all nine cases (or select one using `--case angle2_distance1.25`):

```powershell
python scripts/tune_milk_geometry.py --dataset "../Experiments/Stage_2_Data/stage2/milk" --correspondences outputs/stage2_milk_masked_track3_ba/matching_cache/correspondences --output outputs/ceres_tuning/milk_geometry/results.json
```

The actual eight-case sweep is retained under `outputs/ceres_tuning/milk_geometry/`; the independently run combined case is under `outputs/ceres_tuning/milk_geometry_combo/`. Both contain raw JSON and individual exported reconstructions. Existing promoted outputs remain untouched. For background on triangulation and calibration settings, see the official [COLMAP FAQ](https://colmap.github.io/faq.html#increase-number-of-matches-sparse-3d-points).
