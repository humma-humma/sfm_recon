# Dense GNT supervision ablation

Date: 2026-10-13. Review: http://localhost:8088/gnt-dense-supervision.html

## Question and protocol

Can existing closely spaced GNT targets improve Gaussian renders at intermediate cameras, compared with spending the same number of optimization steps on the original targets?

Both arms start from the same GNT-only 10k checkpoint (`7bf3d21f2d8ac4a5cead41f487a9685514e8890960eff7cf4dde95253048a42f`) and its optimizer state. They each continue for 2,000 steps (10000 through 11999), with seed 42 and identical learning-rate schedules extended to 12k. Gaussian refinement is disabled in both arms, preserving 1,892,910 primitives. Fixed cameras, CPU image caching, classic rasterization, native resolution, background and the RGB objective are unchanged. Original photographs are not RGB training targets.

Of 180 existing dense-turn frames, 11 within 0.1 ms of an original training timestamp are excluded from the novel-view split. The other 169 are sorted chronologically and alternated into 85 added targets and 84 withheld evaluation targets. Control trains on the original 275 GNT images. Dense trains on those plus the 85 added targets (360 total). Six of the 48 adaptive views overlap added training timestamps and are excluded from unseen evaluation; the remaining 42 are evaluated. The original 275 training views are also re-evaluated for regressions.

The added views change both pose coverage and the sampling distribution: the turn receives a greater share of training batches. This test assesses that combined practical intervention; it does not isolate coverage from weighting. Equal steps do not guarantee identical wall time. Optimizers are rebuilt after loading the live Gaussian tensors and then restored, with binding assertions. Source-checkpoint hashes, target hashes, data splits, normalization, finite parameters and unchanged primitive count are checked.

## Interpretation limits

PSNR and SSIM measure agreement with a GNT teacher, not photographic truth. The views are correlated, cover one outdoor turn, and were already inspected during diagnosis. Adaptive views deliberately sample difficult gaps. This is a single-seed exploratory ablation, not a pristine benchmark or proof that a full-scene dense fit will work. A local gain would justify checking visual quality and testing another difficult segment before generating dense targets for an entire scene. No presentation selection is changed automatically.

## Completed results

| Evaluation views | Count | Initial PSNR / SSIM | Control +2k | Dense +2k |
|---|---:|---:|---:|---:|
| Withheld turn | 84 | 22.802 / 0.7427 | 22.595 / 0.7390 | **28.370 / 0.8506** |
| Added turn training targets | 85 | 22.739 / 0.7423 | 22.551 / 0.7390 | 28.549 / 0.8536 |
| Non-overlapping adaptive | 42 | 17.488 / 0.7192 | 17.344 / 0.7131 | 18.338 / 0.7334 |
| Adaptive outside turn only | 36 | 16.819 / 0.7235 | 16.684 / 0.7174 | 16.802 / 0.7192 |
| Original GNT training views | 275 | 29.368 / 0.9171 | **29.975 / 0.9249** | 29.498 / 0.9197 |

Every withheld turn frame improves relative to the matched control. Mean paired PSNR gain is 5.774 dB; median gain is 6.147 dB. Aggregate mean-square error falls 77.13%. Added-target and withheld-target scores are close, which argues against only memorizing the added images. The same-duration control improves known-view fit but slightly worsens withheld-view fit. This supports dense supervision (including its changed sampling distribution) as an effective local intervention in this recipe.

The 42-view adaptive aggregate includes six additional withheld poses within the turn. Its 0.995 dB gain should not be described as scene-wide improvement. The 36 poses outside the turn improve only 0.117 dB versus control and remain essentially equal to the initial checkpoint. Original-view fit is 0.476 dB worse than control on average, with a severe outlier: training view 175 falls from 32.773 to 21.100 dB (initial 31.388). Visual inspection shows a red foreground haze in the dense candidate. All 275 original views are included in the gallery with regression sorting.

Withheld turn 22 shows the control's large dark cloudy patches greatly reduced by dense supervision, while fine edges remain softer than the GNT target. Turn 90 also has fewer cloudy patches but still loses detail. Known-view regressions prevent automatic promotion. Before generating dense targets across the scene, test preservation of original views through controlled sampling or regularization, and evaluate another difficult segment.

Control training took 144.78 s; dense training took 140.62 s. Peak allocated GPU memory was 2268.19 / 2265.89 MiB, respectively. Both retain exactly 1,892,910 Gaussians. These timings exclude preparation, evaluation and publication. No new GNT inference was performed.

Validation: all 202 repository tests passed. All dense/adaptive input hashes, their training-photo source exclusions and all 85 added COLMAP poses were rechecked before publication. The initial renderer reproduces previous native PSNR values exactly. Isolated Chrome passed 486 comparison selections / 1,944 image decodes at native dimensions, metric rows, full-size links, split counts, sorting and next/previous navigation. Raw results, paired analysis and both fit reports are stored beside the durable artifacts.

The follow-up [90/10 sampling experiment](GNT_BALANCED_SUPERVISION.md) recovers most original-view agreement and removes the inspected red haze, while sacrificing some turn accuracy. Its comparisons are available separately; no model was promoted.

## Reproduce

Run `scripts/run_gnt_dense_supervision.ps1`. It uses the cached local Nerfstudio image and no new GNT inference or model download. Large artifacts go under `%USERPROFILE%/sfm_reconstruction_artifacts/gnt_finalization/gnt_dense_supervision`; comparison PNGs and JSON go under `assets/gnt_dense_supervision`. Completed checkpoints and evaluation reports can be reused. An interrupted arm without a completed checkpoint is restarted from the same source.

`scripts/publish_gnt_dense_supervision.py` publishes all 84 withheld, 42 non-overlapping adaptive, 85 added-frame and 275 original-training comparisons, with frame order and improvement/regression sorting. `scripts/verify_gnt_dense_supervision_browser.py` checks image decoding, dimensions, full-size links, split sizes and navigation. The pure split test checks timestamp exclusion and alternation order.
