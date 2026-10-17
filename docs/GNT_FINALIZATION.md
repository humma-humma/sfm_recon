# GNT pipeline finalization trials

The accepted scene traversal remains the visual reference. This work covers independent adaptation for box, boot and milk, smoother playback tests, and measured inference optimizations. It does not automatically replace accepted presentation models.

## Individual traversal videos

Run `scripts/run_gnt_object_videos.ps1` to render box, boot and milk with their separate 6,000-step adapted checkpoints. Each object gets 60 independent 960×540 views sampled uniformly along the captured camera order, with linear camera centers and SLERP rotations. The route does not invent a closing orbit beyond capture coverage. All cameras undergo depth/source preflight; interval-8 source exclusions remain in force.

Each object exports a 32-second native held-view video and a separate motion-compensated 30 fps video. Inference uses the verified compiled-view-attention backend, FP32, chunk 4096, eight eligible source views and 64 inverse-depth samples. Hash-checked PNG frames and checkpoint identity make the job resumable. Video publication checks frame dimensions, source exclusions, duration, frame rate and decoding at the start, middle and end.

The [four-video page](http://localhost:8088/gnt-videos.html) combines these exports with the existing scene native and motion-compensated traversals. It updates after each object finishes. Large files remain under `%USERPROFILE%/sfm_reconstruction_artifacts/gnt_finalization/assets/videos`, served through the existing asset junction; `outputs/gnt_finalization/videos.json` records publication. These videos do not alter or convert the existing Nerfstudio splats.

## Adaptive scene export

For a matched six-second test of genuine 30 fps output, run `scripts/run_gnt_dense_turn.ps1`. It samples the existing scene route at 46 + i/30 seconds for i=0…179, preserving the original piecewise camera-center interpolation and SLERP orientations. Every frame is independently evaluated with the accepted scene GNT checkpoint; exact timestamp/pose matches may reuse verified native renders. FP32 compiled view attention, chunk 4096, eight eligible sources and 64 inverse-depth samples remain unchanged. No optical-flow interpolation or crossfade enters the dense video. Saved PNG hashes support resuming.

The [dense turn comparison](http://localhost:8088/gnt-dense-turn.html) includes the matched adaptive held-view clip and threshold-10 motion-compensated control. Publication checks all 180 PNG hashes/dimensions/source exclusions, uniform camera timestamps, all 180 decoded video frames, 30 fps, six-second duration and HTTP byte-range access. Artifacts are under `%USERPROFILE%/sfm_reconstruction_artifacts/gnt_finalization/assets/dense_turn`; completed verification is `outputs/gnt_finalization/dense_turn.json`. This test addresses frame spacing without changing camera pacing or restarting the stopped object-video job.

`scripts/run_gnt_adaptive.ps1` runs the adaptive scene workflow. Its defaults cap the first pass at 96 genuine 160×90 midpoint previews and 48 additional 960×540 GNT views, with at most three subdivisions per original interval. It reuses all 275 accepted native frames, verifies their hashes and the scene checkpoint, and keeps capture timestamps. Ranking uses rotation, bidirectional optical-flow displacement and endpoint source-set churn. A genuine midpoint preview is compared with a cheap bidirectional-flow midpoint approximation; this is an error proxy, not an exact simulation of FFmpeg's block-motion interpolator.

An interval is refined when rotation exceeds 3 degrees, preview midpoint mean absolute RGB disagreement exceeds 0.035, the proxy's in-bounds fraction is below 90%, or source-set churn exceeds 60%. These are initial engineering thresholds, not a validated perceptual guarantee. Sparse points visible to the target are also checked against eligible source frustums. Less than 70% supported by at least two sources flags limited support and prevents recursive spending beyond that midpoint. This frustum check does not model occlusion or prove coverage of dense surfaces. No attention weights are presented as calibrated model uncertainty.

The original captured route is unchanged; added poses interpolate camera centers and SLERP rotations between existing virtual views. The global priority queue spends the fixed budget on difficult gaps. Budget-limited and depth/support-limited intervals are reported explicitly. Checkpoint and saved-frame hashes support resuming with the same budgets. Sparse support alone cannot establish reconstruction accuracy, and more output frames cannot supply missing source observations.

Separate adaptive native and motion-compensated videos appear at [adaptive scene review](http://localhost:8088/gnt-adaptive.html) after completion. Both retain the original 146.5-second playback duration. The original videos stay available for comparison. Progress is under `%USERPROFILE%/sfm_reconstruction_artifacts/gnt_finalization/assets/adaptive_scene/progress.json`; the completed proof is `outputs/gnt_finalization/adaptive.json`. This is the adaptive runner's default behavior; the legacy uniform traversal runner remains reproducible, and the stopped object-video job is not resumed.

## Object adaptation

Each object starts from the official generalization checkpoint, not the scene-adapted weights. All interval-8 photographs remain excluded from training targets and source RGB. A training target also cannot be its own source. Six capture intervals receive equal target-sampling probability.

The bounded schedule follows the scene recipe: 600 steps with 480-pixel source/target images and transformer/feature learning rates 2e-5 / 1e-5, followed by 1,200 steps at 960 pixels with rates 1e-5 / 5e-6, then 4,200 more native steps with a fresh Adam at those rates. All phases use 256 uniform rays, up to six source photographs, 64 inverse-depth samples, RGB MSE, and gradient clipping at 1. Balanced coverage replaces the scene pilot's initial local targeting. This is a local adaptation experiment, not reproduction of the authors' full training schedule.

Both 1,800- and 6,000-step models are retained. Each is compared with pretrained GNT on every interval-8 camera at 480 by 270. Alternating held-out cameras form validation and review subsets; these datasets have been inspected before, so neither subset is claimed to be a pristine benchmark. Native stills include the prior camera-24 comparison, a focal close-up, and two separated held-out views. Three virtual motion views per model are rendered at 480 by 270 without photograph references.

Review: [object gallery](http://localhost:8088/gnt-objects.html). The gallery updates after each object completes training and rendering. Visual quality determines promotion; there is no automatic winner based only on PSNR.

## Frame blending

The [released GNT renderer](https://github.com/VITA-Group/GNT/blob/main/render.py) independently renders cameras along a path. Its view/ray transformers are described by the [authors](https://github.com/VITA-Group/GNT). The released code does not provide a learned temporal frame-interpolation module. Rendering more intermediate poses is native novel-view synthesis and costs additional inference. Blending already rendered images is a separate video operation.

The [motion gallery](http://localhost:8088/gnt-motion.html) includes indoor, outdoor and stair clips with four versions: native independent views, a half-rate held control, crossfade, and motion-compensated interpolation. Alternate native GNT frames are withheld from the interpolation inputs and used for consistency checks. This tests larger gaps than the full traversal's normal cadence. All interpolation output is 30 fps; these are not 30 independently rendered GNT views per second.

Implementation uses [FFmpeg minterpolate](https://ffmpeg.org/ffmpeg-filters.html#minterpolate): blend mode for crossfades, and bidirectional motion estimation with adaptive overlapped block motion compensation for the flow variant. These modes can ghost or warp disocclusions. Neither adds measured scene information or establishes better reconstruction accuracy.

| Clip | Held PSNR | Crossfade PSNR | Motion compensation PSNR |
| --- | ---: | ---: | ---: |
| Indoor | 12.70 | 14.32 | 15.00 |
| Outdoor | 19.19 | 21.82 | 22.83 |
| Stairs | 13.31 | 15.25 | 15.78 |

These means compare four withheld GNT views per clip, including video encoding error. They measure consistency with independent GNT renders, not with ground-truth photographs. Motion compensation improves average PSNR and SSIM over crossfading in all three clips; visual inspection remains necessary. A separate full-route crossfade is also available. Native lossless renders remain unchanged.

## Runtime measurements

The installed runtime is PyTorch 2.1.2 / CUDA 11.8 on an RTX 2080 with 8 GB. BF16 hardware support is unavailable; the earlier FP16 trial produced nonfinite outdoor renders and remains rejected.

Source loading and feature extraction account for less than 1% of native render time. Nearly all time is spent in ray rendering. Caching helps repeated small previews but cannot eliminate the native transformer computation. A broad host-side C++ rewrite is therefore not the first optimization target; PyTorch already dispatches its tensor operations to native CPU/CUDA kernels.

The first trial combines [PyTorch SDPA](https://docs.pytorch.org/docs/2.1/generated/torch.nn.functional.scaled_dot_product_attention.html) for qk ray attention, GPU output assembly, and bounded CPU feature caching. At native resolution, three scene views took 88.81 / 89.53 / 90.86 seconds in the reference and 84.99 / 84.99 / 85.83 seconds in the candidate: a median speedup of 1.053. The indoor view's largest floating-point error was 1.138e-5, slightly above the initial 1e-5 gate, so this trial is not the selected default. Larger batches were much slower and filled GPU memory; preview size does not protect against per-chunk activation pressure.

The second trial uses compiler fusion only for view-attention forward methods, retaining the original ray attention, FP32 precision, and checkpoint keys. With a 4,096-ray chunk, the same native views took 63.08 / 64.32 / 64.69 seconds: median speedup 1.405, or about 29% lower latency. Peak allocated GPU memory fell to 2,994 MiB from approximately 4,467 MiB. First-call compilation plus rendering took 23.0 seconds at preview resolution; this cost is separate from steady-state timing.

Float previews had maximum error below 1e-4 and PSNR above 80 dB. Each native PNG differed from its archived reference by at most one 8-bit level, in 0.0009% to 0.0021% of channel values. These delivery-based engineering tolerances differ from the stricter initial SDPA gate and are recorded explicitly. This is not bitwise identity or proof of equivalent training behavior: compilation is enabled only after adaptation, for inference. An 8,192-ray compiler trial checks whether the lower memory use permits a further gain; the fastest verified native candidate is retained with a 2% selection margin.

The 8,192-ray compiled trial also passed image-delivery checks, but its three native times were 70.95 / 71.55 / 71.58 seconds, using 5,812 MiB peak allocated memory. Keep the faster 4,096-ray compiled backend. Compiler caches are saved on C: for reuse by object inference; training remains uncompiled.

All timings are local observations, not a hardware-independent performance claim. Rendering remains offline. No hand-written C++ extension or additional downloaded model is required for these optimizations. The reusable `gnt_local_runtime.Renderer` applies the selected backend to new views; the accepted scene video's existing frames are not regenerated merely to change runtime.

## Running and resuming

```powershell
& 'E:\Others\Project\sfm_reconstruction\scripts\run_gnt_finalization.ps1'
```

The runner has `Runtime`, `Compile`, `Objects`, `Blend`, and `Report` stages. `-Stage Blend -FullFlow` additionally exports a full-route motion-compensated video. Run only one GPU stage at a time. Completed checkpoints and renders are preserved. Training resumes from a phase boundary with saved random states; interrupted partial phases are repeated from the last completed checkpoint. Adam is deliberately fresh at each schedule boundary.

The runner archives all object renders and model checkpoints under `%USERPROFILE%\sfm_reconstruction_artifacts\gnt_finalization` on C:, verifies their hashes, and serves only its generated `assets` subfolder through the demo junction. Checkpoints are outside that served subfolder. GPU work uses the existing Nerfstudio image without pulling. Original scene and splat checkpoints are not trained or overwritten.

Numeric records are under `outputs/gnt_finalization`. Full object archive and browser QA run automatically after all three object datasets complete. All 192 unit tests passed after the loader tests were added. The twelve motion-comparison videos passed native decoding and seek checks. Pipeline selection remains pending visual review of the object models and motion variants.
