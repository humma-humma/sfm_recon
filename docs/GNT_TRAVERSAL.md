# Full-route GNT traversal

The native traversal uses the 6,000-step RGB control from the matched detail experiment. This is an offline scene render for visual evaluation; it does not replace the accepted 5,000-step model or the four presentation splat models.

Review: [live progress and final video](http://localhost:8088/gnt-traversal.html). The page shows the latest completed lossless frame while rendering and switches to the video after export.

## Route and rendering

The route contains 275 independently rendered virtual cameras, one midpoint between each adjacent pair of the 276 chronologically ordered captured cameras. Positions are interpolated linearly and rotations use spherical interpolation. It covers the complete recorded indoor and outdoor route, rather than the previous short indoor demonstration.

Each view is rendered at 960 by 540 with eight eligible nearest source images, 64 inverse-depth samples, original depth bounds, full FP32 precision, and a ray chunk of 4,096. Interval-8 photographs remain excluded from source RGB. The control checkpoint SHA-256 is `2d09429d432e9f5cbfbf8c92e284142a58fdc4f469e81cc367c66030ea9706bd`.

Capture timestamps determine the playback timing: approximately 146.5 seconds and 1.9 independently rendered views per second. The MP4 uses 30 fps by repeating each rendered view until the next timestamp. Motion is stepped. There is no image blending, optical-flow interpolation, post-sharpening, or claim of 30 novel views per second. Pause the video or open a lossless PNG to inspect detail.

## Runtime experiments

At 160-pixel preview width, ray chunks of 512, 2,048, 4,096 and 8,192 took approximately 3.53, 2.83, 2.56 and 14.11 seconds. The 4,096 chunk used about 4,418 MiB allocated GPU memory and agreed with the smaller reference to within one decoded 8-bit pixel level. The 8,192 chunk exceeded the physical 8 GB GPU memory budget and was rejected.

The first three native views took approximately 88 seconds each, using about 4,428 MiB allocated GPU memory. This implies roughly 6 to 7 hours for the full route, excluding startup, archive copying, and encoding.

FP16 was tested separately. Its attention-mask sentinel first overflowed; a temporary dtype-safe sentinel allowed testing, but an outdoor view then produced nonfinite RGB. FP16 was rejected, and the temporary research-code patch was restored. The full traversal uses FP32 throughout.

## Reproduction and resuming

```powershell
& 'E:\Others\Project\sfm_reconstruction\scripts\run_gnt_traversal.ps1' -Width 960
```

Run only one traversal job at a time. Completed frames are cached with their rendering configuration; rerunning after a failure reuses them. `-Stage Export` archives and encodes an already complete render without starting Docker. `-Stage Render` renders without publishing.

Large frames and model files remain on C: because E: has little free space. The completed archive is under `%USERPROFILE%\sfm_reconstruction_artifacts\gnt_traversal_control6000_960`, containing `scene_control6000.mp4`, all lossless PNGs, `frames.ffconcat`, and `manifest.json`. The browser asset folder is a junction into that specific generated archive. The separate preview junction points to the render cache.

The runner automatically exports after all 275 views finish. It verifies frame dimensions and SHA-256 hashes, checkpoint identity, held-out source exclusion, video dimensions and duration, and decoded samples at the start, middle, and end. Each sampled decoded video frame must exceed 35 dB PSNR against its lossless PNG. The player offers previous/next rendered-view controls, speed selection, PNG inspection, and MP4 download.

## Verification status

All 190 unit tests passed after the trajectory test was added. An isolated exporter smoke test on three real native frames passed: 960 by 540, 30 fps, 1.6 seconds, and decoded sample PSNR of 43.77 to 44.38 dB. Its record is `outputs/gnt_traversal/export_smoke_test.json`.

Full-run progress is shown on the live page. Results are written to `outputs/gnt_traversal/results.json` only after all views and export checks succeed. The smoke test is not the completed full traversal. Live-page browser checks passed; a separate verification watcher waits for export and then checks video decoding, dimensions, duration, seeking, previous/next controls, speed selection, play/pause, PNG inspection, and MP4 download. Its final record is `outputs/gnt_traversal/browser_qa.json`.

## Completed native export

All 275 views completed, with 24,402.2 seconds of recorded rendering time (6 hours 47 minutes). The intended route timing is 146.58 seconds; the encoded video measures 146.50 seconds at 960 by 540, within the export tolerance. There are 1.876 independently rendered views per second. The H.264 MP4 is 21,558,410 bytes. Its SHA-256 is `27aa6363891f5b5a974902f6ccb3c10a470331264903e6ba83d13639dc1c9299`.

All archived frame hashes, dimensions, route intervals and source exclusions passed verification. Decoded video samples at views 0, 137 and 274 scored 43.77, 44.84 and 43.69 dB against their lossless PNGs. These are encoding-fidelity checks, not reconstruction accuracy against photographs.

The initial export succeeded, but a transient file lock prevented the progress marker from being updated. The marker was repaired without rendering or encoding again, and the runner now retries that write. The local HTTP server also needed byte-range responses and the MP4 MIME type for browser seeking. Eight range-response checks and the existing fourteen HTTP-resource checks passed after this fix. Final browser QA passed for playback, pause, seeking, previous/next view controls, speed selection, lossless PNG decoding and MP4 download. The rendering container exited; all six unrelated Docker services remained running.

Open the [finished player](http://localhost:8088/gnt-traversal.html) or [download the MP4](http://localhost:8088/assets/gnt_traversal_960/scene_control6000.mp4). The player also links every lossless frame.
