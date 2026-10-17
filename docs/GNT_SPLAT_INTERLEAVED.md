# GNT anchors with native splat gap frames

Date: 2026-10-14. Review: http://localhost:8088/gnt-splat-interleaved.html

## Method

The accepted six-second, 180-view outdoor GNT clip provides anchors at 5, 10 or 15 views per second. Every remaining output frame uses the existing balanced 90/10 splat image rendered at that frame's exact recorded camera pose. This includes the end of the clip, without held frames. There is no optical-flow interpolation, warping, crossfade, blending or new inference/training. FFmpeg only encodes the selected 180-image sequence to H.264 at 30fps / 960 × 540 / CRF16.

The same fixed splat model is used at every rate. It already trained on some GNT views in this clip. This isolates playback composition, not retraining splats from 5/10/15fps supervision or independent generalization. Source GNT hashes match the native splat evaluation's target hashes, the checkpoint identity is pinned, and all selected PNGs have per-frame source identities/hashes recorded. Every decoded output frame is checked against its selected input image with PSNR above 32 dB, confirming fidelity and alignment.

## Results

| GNT views/s | GNT / splat frames | Overall mean PSNR | All splat-gap mean PSNR | Shared-gap PSNR / SSIM |
|---|---:|---:|---:|---:|
| 5 | 30 / 150 | 29.47 | 27.34 | 27.17 / 0.8246 |
| 10 | 60 / 120 | 31.56 | 27.32 | 27.17 / 0.8247 |
| 15 | 90 / 90 | 33.86 | 27.32 | 27.18 / 0.8252 |

Scores measure agreement with lossless GNT, not photographs. Shared-gap scores use 58 frames that are GNT anchors at none of the three rates, before the last 5fps anchor. All rates use the same native splat images on those frames; small differences come from video encoding. Overall score rises with more genuine GNT anchors while the splat gaps remain approximately unchanged. Native comparison at frame 83 shows softer wall/railing/stair detail in the splat, so alternating renderers can change appearance even when camera timing is exact. Playback review should judge whether that switching is acceptable.

For context, rescoring the previous motion-compensated videos on exactly these 58 frames gives 26.49 dB / 0.7541 SSIM at 5fps, 31.17 / 0.8722 at 10fps, and 35.09 / 0.9316 at 15fps. Thus native splat filling slightly improves the measured 5fps gaps but trails interpolation at 10/15fps with this model. These are exploratory playback comparisons; the splat's training exposure differs from the interpolation inputs, and no indoor or dynamic-person result is established.

## Review and reproduce

Three aligned panels default to genuine 30fps GNT / 10fps GNT plus splat fillers / 15fps GNT plus splat fillers. Either candidate can switch to 5, 10 or 15 views/s. Shared play/pause, seek, and slow playback help inspect appearance changes. Paused-frame labels identify GNT versus splat sources.

Run `scripts/gnt_splat_interleaved.py --root <artifact-root>`, then `scripts/verify_gnt_rate_decision_browser.py --splats`. Assets are under `%USERPROFILE%/sfm_reconstruction_artifacts/gnt_finalization/assets/splat_interleaved`. Source frames, encoded videos, per-frame provenance and metrics are retained. Existing interpolation galleries remain separate.

Verification: all 207 repository tests passed. Isolated Chrome passed all four video sources, three aligned panels, dimensions/decode/duration, HTTP206, every rate switch, shared seek/play/pause and slow playback. No selected presentation model changed.
