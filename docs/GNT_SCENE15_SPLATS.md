# Full-scene 15fps GNT and corrected splat comparison

Comparison page: http://localhost:8088/gnt-scene-15fps-splats.html.

The three panels use the same 146.533-second route at 960×540:

1. Genuine GNT: 2,198 independent views at 15fps.
2. Existing GNT motion compensation: 4,396 frames at 30fps.
3. GNT anchors with corrected splat gaps: 4,396 frames at 30fps.

The third panel defaults to the six-second test's best corrected-splat variant,
which blends with existing motion interpolation where flow confidence is low.
A selector also provides corrected splats without interpolation fallback. Neither
variant changes model weights or geometry. The underlying model is the existing
balanced 90/10 Splatfacto checkpoint, not a newly trained dynamic model.

`render_scene15_splats.py` exports lossless PNGs for every camera at 30fps. All
2,198 even camera poses and intrinsics must match the GNT plan to tolerance 1e-8.
The existing renderer additionally audits normalized training cameras against
the original poses. Input plans, checkpoint identities and images are hashed;
completed splat PNGs can be resumed after verification.

`publish_scene15_splat.py` reuses the existing native and interpolated GNT videos
byte-for-byte. It applies the bounded exposure fit, DIS residual warping,
visibility/flow consistency gating and fallback from the six-second experiment.
Every GNT anchor is passed through unchanged before video compression. The last
gap has no future GNT anchor, so the pure corrected variant uses only exposure
correction, and the fallback uses the existing interpolation's final held frame.

Video verification decodes every new frame, checks frame count, dimensions,
duration and all GNT anchor fidelities. Only the previously rendered six-second
30fps outdoor GNT clip provides genuine gap-frame reference: 89 interior odd
frames are checked at exactly matching poses and scored with PSNR/SSIM. There is
no genuine 30fps reference for the rest of the route; full-scene intermediate
quality must be judged visually, especially indoors and around moving people.

Timing results distinguish the shared original GNT renderer-call time, reused
views, additional splat renderer-call time, splat setup/PNG wall time, correction
with two encodes, and verification. Original GNT MP4 encoding times were not logged
separately and are not invented. Existing model training is excluded. These are
incremental processing costs on top of the shared GNT anchors, not a claim that
the entire corrected pipeline avoids GNT rendering.

Artifacts and progress are under
`C:\Users\mopu01\sfm_reconstruction_artifacts\gnt_finalization\assets\scene15_splat`.
`render_progress.json` records the native splat views and renderer timings;
`comparison_progress.json` records correction/export progress; `results.json`
records final videos, metrics, masks' mean weights and timings. `job.json` and
`worker.stdout.log` / `worker.stderr.log` track the finishing worker.

The background `finish_scene15_splat.ps1` waits for the GPU export, runs correction
and verification, then runs `verify_gnt_splat_correction_browser.py --scene` to
check the three-panel page, both corrected candidates and synchronized controls.
The previously accepted scene page and selected model files remain unchanged.

## Completed results — 2026-10-16

All videos are complete. Native GNT is 15fps; motion compensation, corrected
splats and corrected splats with reliability fallback are 30fps. The three-panel
page includes a selector for the two corrected candidates.

| Recorded timing | Time |
|---|---:|
| Shared GNT renderer calls (2,089 new views, 109 cached) | 37 h 10 min 42 s |
| Original job start to native GNT MP4 ready | 37 h 17 min 49 s |
| Original job start to motion-compensated MP4 ready | 37 h 26 min 2 s |
| Additional 4,396 splat renderer calls | 2 min 29 s |
| Splat setup, rendering and lossless export wall time | 8 min 13 s |
| Correction and both corrected video encodes | 11 min 31 s |
| Full video verification | 2 min 5 s |

The original MP4 milestones use the original job's UTC start and the unchanged
video modification timestamps. They include other work since job start and are
not isolated encoder timings. New generation costs total 19 min 44 s before
verification, in addition to the already-rendered GNT anchors and existing
interpolation used by the fallback. Pure corrected splats do not require the
interpolation result for their pixels. Earlier cached-view rendering and model
training are excluded.

On the 89 previously rendered outdoor gap references, motion interpolation scores
35.13dB / 0.9321 SSIM, corrected splats 32.81dB / 0.9154, and corrected splats with
fallback 34.21dB / 0.9291. These are GNT-agreement scores on the short outdoor
section only, not full-route or photograph-quality scores. Slight differences
from the six-second-only videos reflect encoding in the longer sequence.

Matching-frame inspection at 10.03, 46.03, 90.03 and 130.03 seconds shows substantial
indoor corruption in pure corrected splats. Reliability fallback reduces it but
does not eliminate it. Its average corrected-splat weight is 53.9% over gap pixels,
so almost half of that blend comes from existing interpolation. Existing GNT
motion interpolation remains the preferred full-scene output.

Every new video frame decoded, all 2,198 GNT anchors exceeded 32dB fidelity in
both corrected videos, six focused tests passed, and browser verification passed
the three-panel layout, all four video sources, dimensions/duration, HTTP206,
variant switching, shared seek/play/pause and slow playback.
