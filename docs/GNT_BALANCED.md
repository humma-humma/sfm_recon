# Whole-scene GNT experiment

Continue the preserved 1,800-step scene checkpoint to 5,000 total steps. This experiment broadens training beyond the original two local regions; it does not fine-tune box, boot, or milk.

The 276-camera capture is divided into six contiguous intervals with equal training frequency. These are capture intervals, not semantic indoor/outdoor labels. Every interval-8 photograph is excluded from both training targets and source RGB. Training targets are also excluded from their own source sets. Camera estimation predates this split, so this is an exploratory reconstruction experiment rather than a pristine benchmark.

Validation cameras are fixed before optimization: 24, 72, 112, 152, 200, and 248. Checkpoints at 3,200 and 5,000 total steps are compared with the existing 1,800-step model using mean PSNR at 480 × 270. Separate cameras 40, 176, and 264 are evaluated afterward at 960 × 540. Previously inspected outdoor/stair cameras 120 and 240 and their 1.7× close-ups are included for visual review, not selection. Seventeen independently rendered poses follow an indoor segment of the capture trajectory between cameras 232 and 248 at 320 × 180. Positions interpolate linearly and rotations use spherical interpolation; the endpoints and intermediate samples avoid exact training-camera poses. This does not establish quality over the whole scene or real-time rendering performance.

Optimization retains six source photographs, 64 inverse-depth samples, 256 random rays, full-resolution training photos, RGB MSE, and gradient clipping at 1. Transformer/feature learning rates are 1e-5 / 5e-6. Fresh Adam state is necessary when loading the earlier weights-only checkpoint. New checkpoint files separately preserve optimizer and random-number generator state for a future continuation. No run beyond 5,000 steps is automatic.

```powershell
& 'E:\Others\Project\sfm_reconstruction\scripts\run_gnt_balanced.ps1'
```

Use `-Stage Render` to finish/reuse rendered comparisons, or `-Stage Report` to republish. The runner refuses to restart training over an existing balanced checkpoint. Checkpoints and lossless renders stay on C: under `%TEMP%\sfm_lightweight_methods_v1\gnt_tuning`; the project drive has very little free space. The final durable model-copy locations are recorded in `outputs/gnt_balanced/models.json`.

The separate comparison page is `web_demo/gnt-balanced.html`, with synchronized wipe/zoom and camera-path playback. Original galleries remain available. Native inference PSNR uses floating-point predictions; delivered JPEG metrics use a common photo reference and include LPIPS. Close-up photo references are warped with the target intrinsics, and their quality is measured only by the delivered-image verifier, because the legacy inference metric compares against an uncropped reference.

Results and validation are recorded in `outputs/gnt_balanced/results.json`, `published_metrics.json`, `verification.json`, and `browser_qa.json`. A validation winner remains a candidate for visual review; no presentation model is automatically replaced.

## Validation results

| Model | Mean PSNR at 480 × 270 | Mean SSIM |
| --- | ---: | ---: |
| Current, 1,800 steps | 21.5318 dB | 0.7131 |
| Balanced, 3,200 steps | 21.7631 dB | 0.7169 |
| Balanced, 5,000 steps | 21.9517 dB | 0.7213 |

The 5,000-step checkpoint won the predeclared validation comparison: +0.4198 dB over the current model and +0.1886 dB over 3,200 steps. Four validation cameras gained PSNR; two were slightly lower. Continuation and its two checkpoint-validation passes took 25.7 minutes, excluding the initial control previews and subsequent native/path renders. Peak training allocation was approximately 6.3 GiB on an 8 GB RTX 2080. These are modest aggregate gains, not evidence of photorealistic reconstruction.

## Native transfer and retention

| Camera | Scope | Current PSNR | 5,000-step PSNR | Change |
| --- | --- | ---: | ---: | ---: |
| 40 | Separate test | 21.598 | 22.042 | +0.444 dB |
| 176 | Separate test | 21.590 | 22.574 | +0.984 dB |
| 264 | Separate test | 24.089 | 24.586 | +0.498 dB |
| 120 | Previously inspected outdoors | 27.453 | 27.364 | −0.089 dB |
| 240 | Previously inspected stairs | 18.878 | 17.981 | −0.896 dB |

The three separate native test cameras average +0.642 dB, and all three improve SSIM. This is a mixed result: the broader model improves reserved-camera averages while losing some of the earlier stair-region gain. Paired previews still show smeared wall/ground detail and outdoor ghosting. Keep both models available for visual review; the 5,000-step model is not a universal replacement. More training is not automatically launched because average validation gains coexist with a material regression on the main previously preferred stair view. Native inference remains about 107–108 seconds per 960 × 540 frame.

## Perceptual results and verification

Delivered JPEG LPIPS is lower when better:

| Camera | Current LPIPS | 5,000-step LPIPS |
| --- | ---: | ---: |
| 40, separate test | 0.39022 | 0.39662 |
| 176, separate test | 0.38134 | 0.36962 |
| 264, separate test | 0.24654 | 0.24883 |
| 120, outdoors | 0.17871 | 0.18546 |
| 240, stairs | 0.40315 | 0.40018 |
| 120, outdoor close-up | 0.15721 | 0.16390 |
| 240, stair close-up | 0.35949 | 0.36547 |

Perceptual gains are inconsistent. Two of the three separate test views have slightly worse LPIPS despite better PSNR; their mean LPIPS changes only about −0.001. The main stair view slightly improves LPIPS despite lower PSNR. Neither metric supports a consistent fine-detail breakthrough. Reference photographs are freshly encoded at JPEG quality 95, including intrinsics-adjusted close-ups. Their JPEG bytes differ from the earlier gallery's references, so compare model LPIPS within this report rather than comparing absolute LPIPS across galleries.

Verification passed: 186 unit tests; 30 comparison views; 73 decoded images; 75 HTTP resources; 26 scored comparisons. Isolated Chrome exercised all 73 method choices, wipe, shared zoom, path playback/pause, and reference availability. The 17 rendered trajectory poses use 17 distinct source sets, and paired models use identical source indices, depth bounds, and inference settings. Held-out RGB exclusion and all three durable-model hashes passed. The original GNT checkpoint and all four presentation checkpoints match their recorded hashes. The research container has exited; the pre-existing six-service Docker stack remains running.
