# GNT detail experiment

The user accepted the cleaner balanced 5,000-step scene model as the visual baseline. This bounded experiment continues it for 1,000 steps in two arms. It does not train box, boot, or milk or replace presentation models automatically.

Both arms restore the same weights, Adam state, and random states. They use the same target-camera sequence across six balanced capture intervals, 256 rays per step, six source images, 64 inverse-depth samples, transformer/feature learning rates 1e-5 / 5e-6, and gradient clipping at 1.

- Control: uniform random rays and RGB MSE.
- Candidate: one contiguous 16 by 16 patch and RGB MSE + 0.01 DSSIM. DSSIM is the mean of (1 - SSIM) / 2 on valid 7 by 7 uniform windows in RGB, with C1 = 0.01 squared and C2 = 0.03 squared.

The structural term follows the [SSIM formulation](https://ece.uwaterloo.ca/~z70wang/research/ssim/), using the RGB/window choices above rather than reproducing the original grayscale Gaussian-window protocol. Sampling and loss change together: this is a combined-method comparison, not an isolated structural-loss ablation. Adjacent patch rays also provide fewer independent spatial samples per step; improvement is not guaranteed.

Every interval-8 photograph remains excluded from training targets and source RGB. Each training target is excluded from its own source set. Camera estimation predates the RGB split. Validation cameras 24, 72, 112, 152, 200, and 248 and the prior native views are already inspected, so they are exploratory. Fresh confirmation cameras 56, 144, and 224 were declared before this run and are evaluated afterward at 480 by 270 without checkpoint selection. Each arm has only one final checkpoint at 6,000 steps.

The gallery compares all three models on seven native stills/close-ups, seventeen independent virtual path poses, six validation views, and three fresh confirmation views. Path playback covers a short indoor capture segment at 320 by 180; it does not demonstrate real-time inference. Native stills are 960 by 540. Matching photo references are encoded at JPEG quality 95 for each method. Delivered-image metrics include correctly warped close-up references; there is no ground-truth photograph for virtual path poses.

```powershell
& 'E:\Others\Project\sfm_reconstruction\scripts\run_gnt_detail.ps1'
```

The runner preserves completed arms, refuses to overwrite weights with incomplete validation, and reuses completed renders. Use `-Stage Render` or `-Stage Report` to finish those stages. Large checkpoints and lossless renders stay under `%TEMP%\sfm_lightweight_methods_v1\gnt_tuning` on C:. The separate review page is `web_demo/gnt-detail.html`; results and verification records are under `outputs/gnt_detail`. Durable weight locations are recorded in `models.json` after completion.

Visible detail, ghosting, and stability determine whether to keep the candidate. Training loss alone does not establish sharper novel views. Only a successful visual result would justify extending the approach to the three object datasets.

## Training and validation

Both 1,000-step arms completed with identical target-camera sequence hashes. Optimization took 394.0 seconds for the RGB control and 389.8 seconds for the patch candidate, excluding validation and final comparison renders. These similar timings do not establish a general performance advantage.

| Model | Validation PSNR | Validation SSIM |
| --- | ---: | ---: |
| Accepted visual baseline, 5,000 | 21.95169 dB | 0.72130 |
| RGB control, 6,000 | 22.02940 dB | 0.72232 |
| Patch + structural loss, 6,000 | 21.94631 dB | 0.72313 |

The patch candidate is effectively unchanged in mean PSNR versus the baseline and 0.08310 dB below the matched RGB control, with a small SSIM increase. These averages are not evidence of a clear sharpness improvement. Full-resolution and fresh-confirmation results are evaluated separately.

## Native views

| Camera | Baseline 5k | RGB control 6k | Patch 6k |
| --- | ---: | ---: | ---: |
| 40 | 22.04210 | 22.20222 | 22.22219 |
| 176 | 22.57365 | 22.83131 | 22.64599 |
| 264 | 24.58633 | 24.61103 | 24.37599 |
| 120 | 27.36356 | 27.23346 | 27.45149 |
| 240 | 17.98135 | 17.91529 | 18.15490 |

PSNR is in dB, computed on floating-point renders before JPEG delivery. Cameras 120 and 240 are the previously inspected outdoor and stair views. The candidate gains slightly on four of these five views but regresses on camera 264. Side-by-side inspection still shows smearing and missing surface detail; these scores do not establish photorealistic recovery.

## Fresh confirmation and conclusion

| Model | PSNR | SSIM | Delivered JPEG LPIPS (lower is better) |
| --- | ---: | ---: | ---: |
| balanced5000 | 20.44331 dB | 0.61116 | 0.422002 |
| detail_control6000 | 20.57535 dB | 0.61056 | 0.418205 |
| detail_patch6000 | 20.35023 dB | 0.61579 | 0.426139 |

The RGB control gains 0.13205 dB over the baseline on these three fresh cameras and slightly improves mean LPIPS. The patch candidate loses 0.09307 dB versus the baseline and 0.22512 dB versus the matched RGB control, with higher mean SSIM but worse mean LPIPS. Its LPIPS also worsens on average across the six validation views and seven native stills/close-ups. At camera 224, patch PSNR worsens while LPIPS improves: individual metrics can disagree, so inspect the matching photographs.

Keep the accepted 5000-step baseline as default. RGB control 6000 is a modest alternative for visual review. Do not extend patch/structural training to object datasets without a clear visual benefit.

This is one run per arm with three fresh confirmation cameras; these small changes are not an estimate of statistical significance. No claim of improved temporal stability is made from the path playback alone.

## Runtime and verification

Optimization took about 6.5 minutes per arm. Including six validation renders and checkpoint writing, the control took 559.7 seconds and the patch arm 554.1 seconds. Both peaked at 6,422 MiB allocated GPU memory. Most native renders took approximately 107 to 110 seconds, with control close-up outliers at 152 and 221 seconds; these variable timings do not establish a method speed advantage. Path frames took about 12 seconds and confirmation previews about 27 seconds. Rendering is offline, not real time.

All 189 unit tests passed. Delivered-image verification passed for 33 cases, 115 images, 48 scored comparisons and 117 HTTP resources. Browser QA exercised all 115 method choices, shared zoom, wipe, and path playback/pause. All 17 path poses use different source sets. Weight hashes confirm the accepted baseline and original GNT checkpoint are unchanged; all four original presentation checkpoints also remain unchanged. The research container exits after rendering; the six unrelated Docker services are preserved.

Review: [local comparison](http://localhost:8088/gnt-detail.html). Numeric records: `outputs/gnt_detail/results.json`, `published_metrics.json`, `conclusion.json`, `verification.json`, and `browser_qa.json`. Checkpoints were copied to durable C: storage and verified against `models.json`.
