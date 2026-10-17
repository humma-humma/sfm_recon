# GNT anchor correction of splat gap frames — 2026-10-16

The existing six-second outdoor turn now has three corrected variants, compared
with the original interleaved splats and the accepted 15fps GNT motion-interpolation
control. All videos are 960×540, 180 frames, 30fps.

Open http://localhost:8088/gnt-splat-correction.html. Three panels show genuine
30fps GNT, existing interpolation, and a selectable splat candidate. The default
candidate uses correction plus interpolation fallback.

## Method and separation of evaluation

Construction reads only even-numbered GNT anchor images, 0–178. Per-channel affine
exposure fits are bounded to gains 0.8–1.2 and offsets ±20. OpenCV DIS optical flow
at half resolution maps anchor appearance residuals into intervening native splat
views. Forward/backward flow consistency, image agreement and valid image bounds
gate correction. Residual magnitude is limited to ±60 intensity units.

The fallback additionally blends corrected splats with the existing motion
interpolation where confidence is low. Mean splat contribution is 75.8% across
gap pixels (including the final frame); this is a video postprocessing method,
not a newly trained or independently improved splat model. The final frame has
no future GNT anchor and uses colour correction only, or interpolation fallback.

Only after construction are withheld odd GNT images loaded for evaluation. Scores
use the same 89 interior gap frames, 1–177; frame 179 is reported separately.
GNT anchors do not contribute to these averages. The fixed splat model already
trained on some views in this clip, so this is not an independent scene-level
generalization benchmark. Agreement with GNT is not photographic ground truth.

## Results

| Gap-frame method | Mean PSNR (dB) | Mean SSIM | Temporal residual MAE |
|---|---:|---:|---:|
| Original splats | 27.31 | 0.8269 | 7.696 |
| Colour matching | 27.22 | 0.8258 | 7.664 |
| Motion-aligned detail correction | 32.93 | 0.9181 | 3.591 |
| Correction + interpolation fallback | 34.18 | 0.9287 | 3.343 |
| Existing GNT motion interpolation | **35.08** | **0.9317** | **3.260** |

Detail correction improves PSNR on all 89 gap frames relative to original splats.
The fallback beats interpolation on 17/89 frames, but loses on average. Colour
matching alone does not solve the appearance mismatch. Temporal residual MAE
measures changes in video error relative to genuine GNT, rather than penalizing
actual camera motion; lower is better, but does not establish perceptual smoothness.

Matching-frame inspection of frames 151 (largest fallback loss) and 77 (largest
fallback gain) confirms that corrected splats recover considerable detail from
anchors, with residual blur or edge inconsistency in difficult regions. Full
motion-quality judgment remains a user viewing decision. The completed full-scene
GNT/interpolation videos remain the preferred output; this test does not justify
replacing them or exporting an entire corrected-splat route yet.

## Runtime and checks

Generating all three candidate clips, including simultaneous encoding, took
39.1 seconds on CPU; input auditing and evaluation are excluded. No new GNT
inference, GPU training or full-scene processing was performed.

Five focused tests passed, including known translation/warp direction, visibility
rejection and known exposure recovery. All six videos decoded at the expected
dimensions and duration, all GNT anchor fidelities exceeded 32dB, and browser
verification passed layout, candidate switching, byte-range delivery, shared
seek/play/pause and slow playback.

Artifacts: `C:\Users\mopu01\sfm_reconstruction_artifacts\gnt_finalization\assets\splat_correction`.
`results.json` records source hashes, checkpoint identities, correction settings,
per-frame confidence, gap/tail metrics and output hashes. Existing inputs and
selected models were preserved.

Reproduce with `scripts/gnt_splat_correction.py --root <gnt_finalization>` using
the project's host Python environment, then run
`scripts/verify_gnt_splat_correction_browser.py` with the local server available.
