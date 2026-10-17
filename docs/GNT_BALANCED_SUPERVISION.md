# Balanced GNT supervision

Date: 2026-10-13. Review: http://localhost:8088/gnt-balanced-supervision.html

## Protocol

Follow-up to [dense supervision](GNT_DENSE_SUPERVISION.md): retain the improvement at unseen turn cameras while protecting the original scene views.

The new `balanced90` arm starts from the original GNT-only 10k checkpoint and optimizer state, not the already refined dense model. It receives the same 2,000 additional steps, 275 original targets, 85 added targets, fixed 1,892,910 Gaussians, seed 42 and schedules as the previous dense arm. Only camera sampling changes. Every shuffled ten-step block contains nine original-view batches and one added-view batch. Each group cycles through a shuffled permutation before repeating. Across training, all original views appear six or seven times (1,800 total batches) and all added views two or three times (200 total batches). This tests a practical group-weighting policy; it also changes sample order and remains a single-seed exploratory trial.

The original matched control and uniform dense fit are reused without retraining. Evaluation reuses their cached renders and scores, rendering only the new arm. The same 84 withheld turn poses, 42 non-overlapping adaptive poses, 85 added targets and 275 original targets are evaluated. Six adaptive poses within the turn are separated from the 36 outside it when interpreting transfer. All scoring measures agreement with GNT, not photographic truth. No presentation model is automatically changed.

## Results

| Evaluation | Count | Original-only control | Uniform dense | Balanced 90/10 |
|---|---:|---:|---:|---:|
| Withheld turn PSNR / SSIM | 84 | 22.595 / 0.7390 | **28.370 / 0.8506** | 27.185 / 0.8301 |
| Original training PSNR / SSIM | 275 | **29.975 / 0.9249** | 29.498 / 0.9197 | 29.860 / 0.9228 |
| Adaptive outside turn PSNR / SSIM | 36 | 16.684 / 0.7174 | 16.802 / 0.7192 | 16.745 / 0.7178 |
| Previously hazy training view 175 PSNR | 1 | 32.773 | 21.100 | **30.963** |

Balanced sampling recovers 0.361 dB of original-view agreement relative to uniform dense and ends only 0.115 dB behind the original-only control on average. Training view 175 improves by 9.863 dB; visual inspection confirms the conspicuous red foreground haze is no longer apparent. The view remains somewhat softer than its GNT target.

This costs 1.184 dB on withheld turn views versus uniform dense, but still leaves a 4.590 dB gain over the original-only control. At inspected turn view 22, large dark cloudy patches remain much reduced relative to control, while the uniform dense version better matches some details. Only 12/84 withheld views improve over uniform dense; 181/275 original views improve. The tradeoff is real rather than a universal gain.

Outside the turn, the change is negligible (-0.057 dB versus uniform dense). Some original views still regress: the worst change versus uniform dense is -3.257 dB at view 255, and the worst versus original-only control is -2.957 dB at view 2. View 255 remains very soft in both Gaussian variants. These results favor balanced sampling as a preservation-focused experimental candidate, but do not establish full-scene quality or justify replacing accepted GNT renders automatically. Another difficult segment and visual review remain necessary before costly full-scene densification.

Training took 142.37 seconds with peak allocated GPU memory 2267.93 MiB. All 1,892,910 Gaussians remain; the source checkpoint is unchanged. Sampling audits confirmed exactly 1,800 / 200 batches and the required per-view coverage. All 203 repository tests passed. Publication rechecked input hashes, source-photo exclusions, added poses, checkpoint identity and sampling reports. The initial comparison reproduces previous native PSNR exactly. Isolated Chrome passed all 486 selections and 1,944 image decodes, dimensions, matching sources/links, metric rows, split counts, sorting and navigation.

## Reproduce

Run `scripts/run_gnt_balanced_supervision.ps1`. Training and evaluation use the cached Docker image and existing targets; no new GNT inference or model download is needed. Large checkpoints and PNGs stay in the existing C: artifact directory. The new checkpoint and sampling counts are under `gnt_dense_supervision/models/balanced90/splatfacto/continued`; new comparison images are under `assets/gnt_dense_supervision/balanced90_*.png`. The extended results JSON contains all four arms, and `balanced90_paired_analysis.json` compares the balanced arm with uniform dense. The earlier dense gallery and its paired analysis remain available.

The browser gallery shows three aligned panels: **Best GNT — quality reference**, **Previous test — uniform dense supervision**, and **Current test — balanced supervision (90% original / 10% added)**. Choose original training views and frame 175 to inspect the previously identified red-haze regression; sort each group by current improvement or regression relative to the previous test. Detailed metric tables remain in the result files and this document.
