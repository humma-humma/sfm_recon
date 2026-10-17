# GNT 15-views-per-second decision test

Date: 2026-10-14. Review: http://localhost:8088/gnt-rate-decision-15.html

Follow-up to [5/10fps](GNT_RATE_DECISION.md). The accepted six-second outdoor GNT clip supplies 90 anchors at indices 0,2,…,178. FFmpeg motion compensation reconstructs 30fps output using exactly the same MCI/AOBMC/bidirectional/VSBMC/scene-change-threshold-10 settings and CRF16 encoding as the earlier test. A held-anchor control is also generated. Only repeated anchors pad the boundaries; frame 179 is held because there is no later input anchor. No new GNT or splat inference is needed.

The original 10fps motion/held videos and genuine reference are reused byte-for-byte, with hashes checked. Both rates are rescored on the same 59 frames that are anchors in neither input and precede the shorter route's last anchor. This new subset explains why the 10fps score is 31.26 dB here instead of 31.09 in the previous 116-frame comparison. Per-rate withheld PSNR and held tails are recorded separately. Scores measure delivered-video agreement with lossless GNT, not photographic truth.

| Video | Mean PSNR | Mean SSIM |
|---|---:|---:|
| Genuine 30fps GNT, encoded reference | 40.87 dB | 0.9733 |
| 10fps + motion compensation | 31.26 dB | 0.8736 |
| **15fps + motion compensation** | **35.16 dB** | **0.9322** |
| 10fps held control | 22.56 dB | 0.6334 |
| 15fps held control | 24.06 dB | 0.6696 |

15fps improves all 59 common withheld frames in PSNR, with a mean gain of 3.90 dB over 10fps. This uses half the genuine renders of a native 30fps sequence, but 50% more than 10fps. On this outdoor clip it is the closest tested interpolation candidate. It is not equivalent to genuine 30fps and does not establish performance indoors or around moving people; no full-route generation is launched automatically.

Visual inspection includes frame 31 (lowest common 15fps PSNR, 30.82 dB versus 28.03 at 10fps) and frame 139 (largest gain, 37.18 versus 29.17 dB). Fifteen-fps interpolation better preserves door/railing alignment at frame 139. At frame 31, residual softening and edge distortion remain relative to genuine GNT. GNT reference images themselves contain smearing and imperfect geometry. Native inspection PNGs are archived beside the reports.

Validation: all source PNG hashes checked; copied reference/10fps MP4 hashes unchanged; all five videos decode as 180 frames at 960 × 540 and 30fps. Minimum preserved-anchor PSNR exceeds 39 dB in the new candidates, checking temporal alignment. All 206 repository tests pass, including the new split/exclusion test. Isolated Chrome passes three aligned panels, five video sources, dimensions/duration/decode, HTTP206, held/motion toggles, shared seek/play/pause and slow playback.

Reproduce with `scripts/gnt_rate_decision_15.py --root <artifact-root>`, then `scripts/verify_gnt_rate_decision_browser.py --fifteen`. Host Python/FFmpeg are sufficient. Artifacts are under `%USERPROFILE%/sfm_reconstruction_artifacts/gnt_finalization/assets/rate_decision_15`; small metrics and browser QA are copied to `outputs/gnt_finalization`. The original 5/10fps comparison remains available.
