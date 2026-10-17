# GNT-guided scene splat fitting

The accepted dense GNT turn is the quality reference. This experiment uses all 275 existing sparse GNT renders as additional RGB training targets for a separate Splatfacto model. It is a custom teacher-guided fit using the installed Nerfstudio 1.1.5 / gsplat stack, not an implementation of ReSplat, DepthSplat, or RadSplat. [RadSplat](https://m-niemeyer.github.io/radsplat/) motivates transferring radiance-field information to a fast Gaussian representation; this experiment transfers rendered RGB and does not reproduce its geometry supervision.

## Data and controls

The original 276 posed scene photographs yield 241 training photos and 35 interval-8 held-out photos. The mixed arm trains on those 241 photos plus all 275 independent GNT midpoint views. The photo-only arm uses the same 241 photos. Explicit split lists avoid assigning synthetic images to validation by filename order. All 180 dense GNT turning frames and all 48 adaptive native additions are excluded from fitting. Sparse GNT source RGB already excludes interval-8 photographs. Geometry/camera estimation predates this split; this is an exploratory project comparison.

The two fresh fits share the same 200,000-point COLMAP initialization, fixed poses, 0.05 global scale, no centering/orientation estimation, seed 42, 10,000-step optimizer schedules, CPU image cache and native images after the first 1,000 steps. Absgrad remains enabled. Refinement uses gradient threshold 0.0016 every 200 steps, opacity culling threshold 0.01, and stops splitting at step 5,000. These capacity settings differ from the older selected 5k scene model, so the matched photo-only fit is required to isolate the effect of GNT supervision. The augmented sampling pool is approximately 53% teacher targets, with the datamanager's random sampling strategy.

The 5k mixed checkpoint is the saved step 5,000; final 10k checkpoints are step 9,999. Checkpoints are resumable and saved separately. No selected presentation checkpoint/config is overwritten or promoted automatically.

## Evaluation and exports

Both fits and the existing selected splat render the same 180 dense-turn camera poses at 960×540. Native gsplat SH3/classic rasterization uses a black background and the exact recorded normalization transform. Dense GNT PSNR/SSIM measures agreement with the teacher, not photographic truth. All 35 original held-out photos provide separate PSNR/SSIM checks. GPU timing excludes image writing/encoding and initialization. Native video frames are encoded at 30 fps without interpolation.

The final mixed model also renders the full 146.5-second route at 30 fps. As in the original traversal, the final pose is held for the short tail after camera motion ends. A temporary raw RGB stream is deleted only after verified MP4 encoding. A Nerfstudio-compatible config/checkpoint and PLY export are retained. The full-route video is an experimental candidate, not a claim that teacher detail survives the fit.

Run `scripts/run_gnt_splat_fit.ps1` from the repository. The runner waits for an active initial fit, completes mixed-model evaluations, fits/evaluates the photo-only control and exports the mixed PLY. Large data and models are under `%USERPROFILE%/sfm_reconstruction_artifacts/gnt_finalization/gnt_splat_scene`; served comparisons are under its sibling `assets/gnt_splat_scene`. The [review page](http://localhost:8088/gnt-splat-fit.html) updates as evaluations finish. Summary: `outputs/gnt_finalization/splat_fit.json`.

## Noise-reduction trials

After preparing datasets with the updated `prepare_gnt_splat.py`, run `scripts/run_gnt_splat_noise.ps1`. It waits for the original export, then evaluates three additional candidates on the same 180 dense-turn views and 35 test photographs:

- `teacher_10000`: 275 GNT RGB targets only, using the same initial points and 10k settings. No original-photo RGB participates in its loss. The teacher dataset retains only the 35 original test images for validation.
- `geometry_12000`: continue the teacher checkpoint for 2k steps with a sampling pool of 275 GNT views and 241 original training-camera views. GNT views contribute the existing RGB loss. Original-camera views contribute only a robust relative sensor-depth loss weighted by 0.05; their RGB never contributes. Confidence masks require at least two neighboring training views to agree and exclude depth edges. The existing depth-cache pose hash, eligible views and neighbor exclusions are checked before training. This comparison changes both loss and training duration, so any improvement cannot be attributed solely to depth supervision. Gaussian refinement is disabled during this continuation.
- `pruned_12000`: remove Gaussians in front of reliable training-camera sensor surfaces in at least three views, with no supporting surface observation. Tolerance is 0.10 m + 5% of depth. Occluded and unobserved points survive; lack of evidence alone never removes a point. This is depth-consistency pruning, not RadSplat's contribution-based pruning.

No held-out sensor depth enters these new losses or pruning decisions. The inherited seed cloud and camera estimates still predate the split. Teacher RGB supervision alone does not supply GNT's internal representations or reliable per-ray geometry. Geometry and pruned checkpoints contain inference weights only. The teacher keeps its latest checkpoint to bound storage; resume code rebuilds optimizers after loading Gaussian tensors and computes remaining steps explicitly.

The runner publishes all completed candidates to the same comparison page. `noise_trials_status` in the summary distinguishes the three new experiments from completion of the original four comparisons. No full-route video is regenerated for these noise trials until visual comparison justifies one.

### Completed scene results (2026-10-12)

| Candidate | Dense GNT PSNR / SSIM | Held-out photo PSNR / SSIM | Median GPU ms | Gaussians |
|---|---:|---:|---:|---:|
| Mixed 10k | 24.17 / 0.7724 | 16.68 / 0.6429 | 48.5 | 1,949,881 |
| GNT-only 10k | 23.26 / 0.7516 | 16.64 / 0.6285 | 49.2 | 1,892,910 |
| Depth-guided 12k | 23.09 / 0.7507 | 16.60 / 0.6265 | 49.7 | 1,892,910 |
| Pruned 12k | 22.77 / 0.7422 | 16.60 / 0.6173 | 47.0 | 1,708,556 |

GNT-only training took 537.35 seconds and peaked at 3395.16 MiB of CUDA allocation. Depth refinement took 181.92 seconds, with 1067 teacher RGB batches and 933 sensor-depth batches; none had empty depth confidence masks. Pruning took 14.35 seconds and removed 184,354 points (9.74%). Timings are local measurements, not guaranteed application throughput or speedups.

Inspection of matching turn frames 30/90 and the indoor held-out photo 0 found persistent cloudiness and soft edges. GNT-only smooths some fine speckling but does not match the clean teacher. Depth guidance produces small changes without an obvious quality win. Pruning reveals stronger patch/edge artifacts in photo 0. Both teacher agreement and held-out photo SSIM decline across these new variants. No candidate is promoted; the accepted GNT remains the quality reference. These results reject the tested recipes, not all possible teacher-to-splat methods. The failure of GNT-only to beat mixed supervision also means original-photo RGB conflict is not a sufficient explanation for the artifacts.

Validation: 199 repository tests passed; actual parser splits, pose/depth-cache hashes, optimizer bindings, finite final parameters, unchanged teacher checkpoint hash and nonempty pruning were checked. Browser validation checks all published video dimensions, durations, decoding/seeking, play/pause controls and photographic images.
