# Full-scene GNT at 15 views per second

Authorized scope: scene only. Review/progress: http://localhost:8088/gnt-scene-15fps.html

The run samples the accepted captured route at 15 genuine GNT views/s using the accepted `detail_control_006000` checkpoint, FP32 compiled view attention, chunk4096, eight eligible source views and 64 inverse-depth samples. No splats, crossfades, exposure adjustments or sharpening enter native frames. The camera route is unchanged; a whole-frame duration requires 2,198 inputs over 146.533 seconds, 0.033 seconds longer in playback than the prior 146.5-second video. Last camera time remains 146.4667 seconds along the route.

An exact pose/time/intrinsics check finds 109 reusable frames from the existing sparse, adaptive and dense scene exports, leaving 2,089 new renders. Input RGB, pose/point files, checkpoint, plan and completed PNG hashes are pinned. All cameras undergo source/depth preflight. Every rendered view saves atomically and records source indices excluding interval-eight photographs. Completed frames are reused on restart; changed inputs/configuration are rejected. At the measured approximately 64.35 seconds/view, remaining rendering is about 37.3 hours before export overhead. This is an estimate, not a completion guarantee.

`scripts/run_gnt_scene15.ps1` owns a serial GPU job in the named container `sfm-gnt-scene15`, using the cached Nerfstudio image. Other Docker services are left alone. Output is `%USERPROFILE%/sfm_reconstruction_artifacts/gnt_finalization/assets/scene15`: `plan.json`, `progress.json`, per-frame PNGs, `render.log`, `job.json`, worker logs and completed videos. It is safe to rerun the script to resume verified frames. Keep Docker Desktop running and prevent system sleep.

After rendering, the host automatically exports:

1. Native 15fps video: every output frame is independently rendered GNT (except verified reusable identical camera poses).
2. 30fps motion-compensated video: the same FFmpeg MCI/AOBMC/bidirectional/VSBMC settings and scene-change threshold10 used in the accepted short decision test. Padding repeats input anchors only; the final output frame is held because there is no later anchor.

All 2,198 PNG hashes/dimensions/source exclusions are rechecked. Video export verifies dimensions, fps, frame counts and every decoded frame; native frames and preserved motion-export anchors must exceed32dB PSNR against their selected lossless GNT inputs. Final isolated-browser QA verifies delivery, decoding, durations, HTTP range access, shared seeking/play/pause and slow playback. These are technical delivery checks, not evidence of artifact-free indoor interpolation. Indoor occlusions, moving subjects, and source inconsistencies still need user visual review. Existing presentations and completed object renders are unchanged.
