# GNT 5/10 fps decision test

Date: 2026-10-14. Review: http://localhost:8088/gnt-rate-decision.html

## Method

Reuse the accepted six-second outdoor turning clip: 180 independently rendered GNT views at 960 × 540 and 30 fps. Verify all source PNG hashes. The 5fps input consists only of indices 0,6,…,174 (30 anchors); 10fps uses 0,3,…,177 (60 anchors). Withheld views are never passed to FFmpeg. No new GNT inference, splat fitting or splat rendering is performed.

Each input is upsampled with FFmpeg `minterpolate`: `fps=30:mi_mode=mci:mc_mode=aobmc:me_mode=bidir:vsbmc=1:scd=fdiff:scd_threshold=10`. Held-frame controls use the same anchors and `fps=30`. Two input-frame durations of start padding and three of end padding repeat existing anchors only; trim back to the original six-second timeline. Videos use identical H.264 CRF16 encoding. The reference is a byte-identical copy of the accepted genuine 30fps GNT video.

Common scores use 116 indices that are withheld from both rates and precede the 5fps version's last anchor. This avoids comparing one method's input anchors against another method's synthesized frames. Per-rate withheld PSNR is also recorded separately. The five frames after anchor 174 at 5fps, and the two after anchor 177 at 10fps, have no later anchor and are held; their scores are separate. These boundary frames remain visible in the full video. PSNR/SSIM measure delivered-video agreement with lossless GNT PNGs, not photographic ground truth. The reference row quantifies video compression error.

## Results on common withheld frames

| Video | Mean PSNR | Mean SSIM |
|---|---:|---:|
| Genuine 30fps GNT, encoded reference | 40.74 dB | 0.9721 |
| 5fps + motion compensation | 25.43 dB | 0.7360 |
| 5fps held control | 20.37 dB | 0.5804 |
| **10fps + motion compensation** | **31.09 dB** | **0.8708** |
| 10fps held control | 22.41 dB | 0.6292 |

10fps motion compensation improves by 5.66 dB over 5fps and by 8.67 dB over its held control. It uses one third as many genuine GNT renders as the 30fps reference. This supports 10fps as the stronger candidate on this clip, not equivalence to genuine 30fps or confirmation of indoor performance.

Visual checks: frame 122 (largest 10-versus-5fps PSNR gain) shows substantially stronger deformation/ghosting of the door frame and railings at 5fps; 10fps is closer to the reference. Frame 83 (lowest-scoring common 10fps frame, about 25.00 dB) retains softened/warped stair and railing edges even at 10fps. Five-fps artifacts remain apparent. User playback review should determine acceptability; no full-scene render has been launched or selected automatically. Indoor occlusions and moving people require a separate test.

## Verification and reproduction

The subsequent [15fps comparison](GNT_RATE_DECISION_15.md) improves on 10fps on a newly shared 59-frame subset. Its separate gallery preserves this original 5/10fps test.

All source hashes and six-second / 180-frame exports passed. Every preserved input anchor was checked against its corresponding genuine view; minimum anchor PSNR exceeds 39 dB in each candidate/control, verifying timing and retained fidelity. All 205 repository tests passed, including split/exclusion/boundary identities. Isolated Chrome passed three aligned panels, all five video sources, dimensions/decoding/duration, HTTP206, held/motion switches, shared seek/play/pause and slow playback.

Run `scripts/gnt_rate_decision.py --root <artifact-root>` with the project host Python environment, followed by `scripts/verify_gnt_rate_decision_browser.py`. FFmpeg defaults to `C:/ffmpeg/bin/ffmpeg.exe`. Artifacts are under `%USERPROFILE%/sfm_reconstruction_artifacts/gnt_finalization/assets/rate_decision`, including input anchors, videos, metrics and inspection PNGs for frames 83/122. The small report and browser verification are also recorded under `outputs/gnt_finalization`.
