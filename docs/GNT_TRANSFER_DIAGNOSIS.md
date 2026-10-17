# GNT-to-splat transfer diagnosis — 2026-10-13

The frozen GNT-only 10k checkpoint was evaluated at all 275 exact training poses, all 180 previously generated dense-turn poses, and all 48 previously generated adaptive poses. No new GNT generation or training was performed. The model contains 1,892,910 Gaussians; SHA-256 is recorded in `outputs/gnt_finalization/transfer_diagnosis.json`.

Review: <http://localhost:8088/gnt-transfer-diagnosis.html>. Nine training-view comparisons include four worst training fits and route anchors. Eight local-gap comparisons show an unseen target/fit alongside its two fitted neighbors. All images link to native 960×540 PNGs; the background selector switches Gaussian renders without changing targets.

## Results

Native Nerfstudio inference background:

| Set | Count | Mean teacher PSNR | Mean SSIM | Mean local PSNR gap | More than 3 dB worse than local endpoints |
|---|---:|---:|---:|---:|---:|
| Exact training poses | 275 | 29.37 | 0.9171 | — | — |
| Unseen dense-turn poses | 169 | 22.77 | 0.7425 | -8.06 dB | 94.7% |
| Adaptive poses | 48 | 18.05 | 0.7153 | -12.00 dB | 97.9% |

Eleven dense poses within 0.1 ms of a training timestamp are excluded from unseen summaries; all 180 were still rendered and measured. The previous 23.26 dB dense-turn average included all 180 poses, so the lower 22.77 dB value here is a different subset, not a changed checkpoint or render regression.

The local comparator interpolates the two endpoint mean-square errors according to the unseen pose's fraction along that training interval, converts to PSNR, and subtracts it from the unseen-view PSNR. Negative means the unseen fit is worse. It does not average logarithmic PSNRs. All 169 unseen dense poses have a negative local gap; 47/48 adaptive poses do.

## What this establishes

The model reproduces known teacher views much better than intermediate views, including when compared with local fitted endpoints rather than the full-route mean. The gap persists beyond the outdoor turn in the deliberately difficult adaptive set. Visual inspection of adaptive pose 25 (source time 106.97 s, between training poses 199/200) shows pronounced cloudy/fragmented patches despite substantially better fits at the endpoints. The corresponding GNT target itself is imperfect and streaked; its agreement is not photographic truth.

Training fit is also imperfect. The worst training view is about 20.48 dB, and the training PSNR 10th percentile is 25.19 dB. At training pose 38 (source time 20.84 s), the GNT corridor floor is comparatively smooth while the Gaussian fit adds cloudy patches. More optimization could address some known-view errors, but known-view error alone does not explain the large local unseen-view gaps.

This supports a substantial between-view generalization problem in this RGB-only transfer recipe. It does not identify geometry as the sole cause. Incorrect/underconstrained Gaussian geometry, view-dependent appearance fitting and inconsistency in GNT targets remain possible contributors. The adaptive views were chosen because they were difficult; the dense views are correlated and cover only one outdoor turn. Local endpoint comparisons still change visible content and source-view choices, so neither collection is an independent, controlled geometric benchmark.

## Software checks

All 275 dataparser training cameras were checked against the independently normalized COLMAP poses and supplied intrinsics. Maximum world-to-camera matrix discrepancy was 2.09e-7. Image dimensions were 960×540. Three official Nerfstudio output checks (training poses 0/137/274) matched the custom renderer exactly in float RGB when using the same camera and native background.

Re-rendering all 180 previous black-background comparisons reproduced their PSNRs within 2.59e-6 dB. Native background RGB is approximately [0.1490, 0.1647, 0.2157]. Switching from black changes aggregate PSNR by less than 0.003 dB and SSIM by less than 0.0003 in each set. Background compositing is not a material explanation for the observed fit loss on these images.

The checkpoint hash remained unchanged. Target hashes and saved image hashes were verified. All 201 repository tests passed; the timestamp/exact-pose test was rerun after adding the 0.1 ms exclusion. Isolated headless Chrome passed all 66 gallery image decodes/dimensions, six metric rows, both background switches, full-size links and unchanged target checks.

## Next controlled experiment

The cheapest causal follow-up is a local dense-supervision ablation using already generated GNT targets. Divide the 169 genuinely unseen dense-turn poses into alternating added-training and withheld-evaluation subsets; keep the 48 adaptive poses for evaluation. Compare a short continuation with the added targets against a duration-matched continuation without them, preserving initialization, optimizer schedule, capacity and rendering settings. Previously inspected targets are exploratory evaluation, not a pristine benchmark.

If this improves withheld turn views but not distant adaptive views, it supports a local supervision-density limitation. If fitting improves only on the added targets, prioritize geometry/appearance consistency rather than globally generating more targets or blindly increasing training. This follow-up is proposed, not run by the diagnosis.

The follow-up is now complete as a separate experiment: [dense supervision results](GNT_DENSE_SUPERVISION.md). Withheld turn views improve substantially, but distant views change little and some known views regress. The accepted model remains unchanged.

## Reproduce

Run `scripts/run_gnt_transfer_diagnosis.ps1` from the repository. It uses the cached local Nerfstudio image, writes large artifacts under `%USERPROFILE%/sfm_reconstruction_artifacts/gnt_finalization/assets/gnt_transfer_diagnosis`, publishes the gallery and runs browser verification. Existing checkpoints and presentation selections are not changed. Saved comparisons support resuming; no model download is required.
