# Photograph-guided, surface and pretrained-prior sharpness pilot

Run completed on all four datasets. Open `web_demo/sharpness.html`, or the
**Sharpness tests** link in the local demo. There are 30 held-out reference
comparisons and 36 motion cameras, with six experimental/control outputs per
camera. Main presentation models and viewer assets were not replaced.

## What was actually tested

1. **Photograph-guided rendering:** four nearby training photographs are
   reprojected using expected splat depth. Source depth consistency, colour
   agreement and at least two supporting sources gate the contribution.
   Unsupported pixels retain the splat rendering. This is a classical
   image-based rendering prototype, not a reproduction of the learned
   [IBRNet](https://ibrnet.github.io/) feature aggregator.
2. **Surface and scale:** inference-only gsplat antialiasing is measured
   separately. For scene, the existing training-only TSDF supplies target and
   source visibility depths for photograph projection, with antialiased splats
   as fallback. Objects have no fused TSDF in this pilot; their combined variant
   changes the fallback only. This is not full
   [Mip-Splatting](https://github.com/autonomousvision/mip-splatting) retraining or
   a 2DGS integration.
3. **Pretrained learned prior:** official
   [Real-ESRGAN](https://github.com/xinntao/Real-ESRGAN) x2plus restores the combined
   result. A second control applies the same model directly to the original
   splats. This GAN supplies a learned image-restoration prior, not a multiview
   reconstruction prior such as [ReconFusion](https://reconfusion.github.io/).
   No diffusion model, neural feature field or learned source-view aggregator
   was integrated. These pilot results do not establish their performance.

The pretrained output is 2x resolution, then area-downsampled to the reference
resolution for numerical comparisons. One full 2x example per dataset is also
linked. Per-frame restoration can invent edges or textures.

## Fidelity results

Mean PSNR change in dB relative to each dataset's fixed direct-render baseline:

| Dataset | Photo reuse | Antialiasing | Surface/photo combination | Prior on combination | Prior on original |
|---|---:|---:|---:|---:|---:|
| Box | -0.114 | -0.552 | -0.626 | -1.028 | -0.571 |
| Boot | +0.016 | -0.280 | -0.252 | -0.228 | +0.038 |
| Milk | -0.025 | -0.572 | -0.552 | -0.531 | -0.034 |
| Scene | +0.238 | +0.000 | **+0.826** | +0.339 | -0.250 |

AlexNet LPIPS on the delivered JPEGs (lower is better):

| Dataset | Original | Photo reuse | Surface/photo combination | Prior on combination | Prior on original |
|---|---:|---:|---:|---:|---:|
| Box | 0.2506 | 0.2423 | 0.2466 | 0.2680 | 0.2746 |
| Boot | 0.3493 | 0.3485 | 0.3621 | 0.3606 | 0.3503 |
| Milk | 0.3002 | 0.3016 | 0.3199 | 0.3207 | 0.3042 |
| Scene | 0.5325 | 0.4691 | **0.4297** | 0.4550 | 0.5063 |

The scene surface/photo result improves LPIPS by about 19.3%. Its strongest
PSNR improvement is pavement at 1.7x framing (+4.273 dB). Indoor stair structure
partly returns, but projected detail has conspicuous holes and seams. The
entrance remains essentially unchanged because its projections lack reliable
support. Sampled object views retain smeared table markers; restoration
accentuates existing distorted edges instead of recovering the reference pattern.

The numeric fidelity gate is mean PSNR delta >= -0.1 dB, worst-view delta >=
-0.5 dB and mean SSIM delta >= -0.002. It is a screening diagnostic, not proof
of visual improvement. Results are not automatically promoted. Individual
per-view metrics, support fractions and timings are in
`outputs/sharpness_trials/results.json`.

## Motion and runtime

Nine cameras sample a lateral path of +/-0.02 original COLMAP world units.
The metric scale of all object scenes was not independently verified. There
are no ground-truth images along this path.

All methods share the same warp/mask within a comparison. For scene, the JSON
reports **both** TSDF and splat-depth warps. Their conclusions differ because
the reconstructed depths disagree. With TSDF geometry, scene warp MAE changes
from 0.01014 (original) to 0.00546 (surface/photo), then 0.00669 (with prior).
With splat depth it changes from 0.00447 to 0.02166, then 0.02343. Thus the
splat-depth score alone would unfairly penalize the surface substitution.
Neither estimate proves temporal correctness. Use the synchronized motion
players to inspect seams and texture changes.

Measured median native rendering costs range from roughly 6 to 57 ms per
frame across datasets. CPU photo reprojection costs approximately 0.44–0.50 s;
the separate surface/photo pass costs approximately 0.42–0.46 s. Each FP16
Real-ESRGAN pass costs approximately 0.45 s on the RTX 2080. The GPU was shared
with the existing stack. These are offline prototype timings, not a browser
frame-rate claim; startup/source-depth preparation is separate.

## Reproduction, verification and limits

Run `scripts/run_sharpness_trials.ps1` with Docker GPU access and the existing
Open3D/PyTorch host environment. The cached Nerfstudio 1.1.5 image renders the
fixed checkpoints; host Python handles reprojection, TSDF raycasting and
restoration; Docker supplies cached AlexNet LPIPS. Intermediates and the 64 MB
pretrained weights stay in `%TEMP%/sfm_sharpness_trials_v1` because the project
drive is nearly full. The small architecture adaptation includes its upstream
license. No dependencies were installed into the shared stack.

The sample contains three held-out views per object and six diagnostic scene
views, each at 1x and 1.7x focal length, using 960x540 exported images. No target
RGB participates in source selection, confidence or rendering. All projected
RGB sources exclude interval-8 held-out views. Existing camera estimation and
Gaussian seed clouds predate the split, so this is not an end-to-end
split-isolated reconstruction benchmark. The scene TSDF excludes held-out
sensor depths as documented in `DEPTH_RESEARCH_RESULTS.md`.

The baseline is direct RGB+ED gsplat rasterization of frozen selected weights
with a black background; it is not a claim of bitwise equality to every native
or browser viewer. PSNR/SSIM are measured before JPEG compression; LPIPS is
explicitly measured on the delivered quality-90 JPEGs. Source checkpoint
SHA-256 values are checked before and after the experiment.

Visual inspection covered one close-up per object, all six scene diagnostics
at normal framing, and a full-resolution indoor stair comparison. Browser
interaction and a continuously navigable hybrid renderer remain unverified
and unimplemented respectively. The gallery is an offline pilot with short
precomputed motion sequences. No full-scene photo-level quality claim is made.

Recommendation: keep the current presentations. Surface-guided photograph
reuse is the strongest scene experiment to investigate further; neither
inference antialiasing nor per-frame GAN restoration is a universal sharpness
fix. The remaining indoor geometry and coverage errors are visible in these
results rather than solved by the pretrained prior.

Final verification: **180 pytest tests passed**, all 426 comparison JPEGs and
four 2x examples validated, all 180 method/reference metric records finite,
241 local HTTP page/resource checks passed, gallery JavaScript and PowerShell
runner syntax passed, and `git diff --check` passed. Browser interaction remains
unverified. Full checkpoint hashes and held-out source exclusion were checked.
See `outputs/sharpness_trials/verification.json` and `visual_review.json`.

## User review and comparison controls

The user initially reported no discernible difference in the comparison
gallery, then refined that assessment: **prior on original** is a visibly
better render for the static objects, while **surface/photo plus prior** reveals
more fine-grained scene detail whose meaning remains difficult to recognize.
This supersedes the blanket no-improvement assessment. Record the static-object
appearance preference as positive; scene detail visibility does not yet
establish faithful or recognizable structure. No presentation is automatically
promoted. The updated user review is in `visual_review.json`.

The gallery now includes a single-canvas A/B wipe, full-image toggle and shared
1–4x crop zoom/pan. The reference photograph can occupy the candidate side to
show the remaining gap. All 210 selectable image paths and the two HTTP
resources were verified; JavaScript syntax passed. Browser interaction remains
unverified. This changes inspection only, not reconstruction or rendering.
