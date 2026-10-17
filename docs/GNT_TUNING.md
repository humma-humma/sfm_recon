# Local GNT improvement trials

The baseline is the pinned official pretrained GNT checkpoint from `LIGHTWEIGHT_METHODS.md`. The original trial, preferred restoration, and traversable demo models are preserved. 2DGS is rejected and is not part of this work.

## Experiment design

`scripts/tune_gnt.py` evaluates one-factor inference changes on previously inspected scene cameras 120 (outdoors) and 240 (indoors), at 480×270 target resolution while retaining 960×540 source photographs:

- 64, 128, or 192 samples per ray.
- 6, 8, or 12 nearest eligible source photographs.
- Source selection using sparse-point frustum overlap, additional point coverage, and image sharpness among the 24 nearest eligible cameras. Frustum overlap is not an occlusion test.
- Uniform depth versus inverse-depth sampling.
- Wider sparse-point depth bounds versus the original bounds. Wide bounds use depth quantiles .001/.999 with .7/1.3 margins; the control uses .005/.995 with .8/1.2 margins.
- A separate combined 192-sample/overlap-selection check; gains are not assumed additive.

Inference settings are ranked by equal-weight mean development PSNR. Among settings within 0.05 dB of the best mean, the lowest measured render time is selected; this is a practical cost tolerance rather than a statistical significance threshold. Indoor and outdoor results are also reported individually. This exploratory selection does not guarantee the visually best setting.

Adaptation starts from the same original checkpoint and updates both the feature extractor and transformer with RGB MSE. It uses a balanced union of the nearest 24 eligible cameras around each development region, 480×270 source/target images, six source views, 256 random rays, 64 samples, transformer learning rate 2e-5, feature learning rate 1e-5, and gradient clipping at 1. Checkpoints at steps 200 and 600 are evaluated on the development views with the selected inference configuration. This bounded pilot is much shorter than the official 60k-step fine-tuning configuration.

Because the 600-step checkpoint improved both development views, a second phase continues its weights for another 1,200 steps using 960×540 source/target images. It uses fresh Adam state, transformer learning rate 1e-5, and feature learning rate 5e-6; other training settings remain the same. The resulting 1,800-step checkpoint is compared against the earlier checkpoints rather than assumed better. Peak allocation in this full-resolution phase is recorded in the run metadata.

Every interval-8 image is excluded from both adaptation targets and source photographs. Training targets are explicitly excluded from their own source sets. Camera and sparse geometry estimation predates the split, so this is not an end-to-end held-out reconstruction benchmark. The original comparison targets have already been visually inspected; development results are not untouched test results.

The final phase checks the selected setting at 960×540 on the two development views and separate scene cameras 96, 184, and 272. Those additional views are not used to select inference settings or checkpoints. Three scene camera-path positions are checked at 480×270 without reference photographs. When inference tuning retains the original configuration, the existing object anchor comparisons are reused rather than rendering identical settings again. Scene-adapted weights are not applied to those unrelated object captures. Existing original-GNT scene controls are also reused where available, with matching checkpoint hashes, source indices and dimensions asserted; their runtime entries retain the original timings.

## Reproduce and inspect

```powershell
& 'E:\Others\Project\sfm_reconstruction\scripts\run_gnt_tuning.ps1'
```

The runner supports `-Stage Sweep`, `Adapt`, `Refine`, `Final`, and `Report`. Large checkpoints and lossless renders stay in `%TEMP%\sfm_lightweight_methods_v1\gnt_tuning` on C:. The sweep and final phases reuse completed labelled renders. Adaptation starts fresh, so rerunning that phase replaces its adaptation checkpoints. `Refine` continues the 600-step weights with fresh optimizer state. A separate durable copy of the selected weights is recorded with its SHA256 in `outputs/gnt_tuning/model.json`, outside temporary storage on C: because the project drive is nearly full.

The comparison page is `web_demo/gnt-tuning.html`. Persistent run metadata is `outputs/gnt_tuning/results.json`. Native PSNR/SSIM use fresh floating-point predictions or reused lossless PNG controls; rounding can affect very small differences. Published JPEGs use quality 95. `scripts/verify_gnt_tuning.py` measures all delivered JPEGs against the same reference JPEG, including PSNR, SSIM and LPIPS, in `outputs/gnt_tuning/published_metrics.json`. Those measurements provide a consistent comparison with the preferred restoration. Do not compare metrics across different target resolutions.

Timing is observed on a shared RTX 2080. More samples or sources can substantially increase cost. A static-view gain does not establish temporal consistency or interactive rendering speed; camera-path checks remain separate.

## Completed local results

Inference tuning retained 64 inverse-depth samples and eight nearest eligible views. The best denser-sampling mean increased only about 0.04 dB while tripling runtime. Overlap-based selection substantially regressed outdoors; wider bounds and the source-count changes did not provide a clear overall improvement.

The adapted checkpoint used 48 eligible training photographs and 1,800 total steps: 600 at half resolution followed by 1,200 at full resolution. Training and its development validation took 12.6 minutes; the full-resolution training peak was 6,419.9 MiB. Fresh full-resolution inference took about 107–108 seconds per frame.

These scores compare the delivered JPEGs against the same reference photograph. LPIPS is lower when better:

| Camera | Scope | PSNR change versus original GNT | Original LPIPS | Adapted LPIPS |
| --- | --- | ---: | ---: | ---: |
| 120, outdoors | Development | +0.30 dB | 0.2060 | 0.2019 |
| 240, stairs | Development | +3.98 dB | 0.5815 | 0.4317 |
| 96 | Confirmation | −0.12 dB | 0.4817 | 0.4926 |
| 184 | Confirmation | −0.50 dB | 0.6262 | 0.6270 |
| 272 | Confirmation | −0.40 dB | 0.3158 | 0.3079 |

The main stair view has clearer boundaries, with about 26% lower full-resolution LPIPS. Fine detail and noise remain unresolved. The three additional cameras average −0.34 dB, so the result supports a targeted stair-region candidate rather than a global replacement. Their perceptual results are mixed. The three motion samples are coarse comparisons, not proof of temporal stability.

Keep the preferred restoration for box, boot and milk, and keep both original GNT and the preferred restoration available for scene. No live/demo model is promoted automatically. The inference script now accepts `--checkpoint` with a required `--output-tag` to render adapted weights into a separate folder.

Verification completed: 185 unit tests passed; all 79 JPEGs and 81 HTTP resources passed; isolated Chrome exercised 13 views and 79 method selections. Held-out RGB exclusion checks passed, all four original presentation-checkpoint hashes match, and the durable adapted-model copy matches its SHA256. See `outputs/gnt_tuning/verification.json`, `browser_qa.json`, and `published_metrics.json`.
