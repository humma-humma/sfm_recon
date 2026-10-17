# Scene diagnosis

Follow-up completed: [confidence-weighted sensor-depth experiment and TSDF baseline](DEPTH_RESEARCH_RESULTS.md).
The candidate passes the numeric RGB and geometry gates; local fog remains visible.

Status: complete for CPU coverage/alignment and official Nerfstudio RGB/expected-
depth checks. The user completed the Docker rendering on 2026-10-11. All 159 RGB/
depth pairs (47 training + six held-out cameras, three checkpoints) were validated.
Surface thickness itself was not measured. No training, pruning, or selected-model
changes were made during diagnosis.

## Completed render findings

| Checkpoint | 47 training views: mean PSNR / LPIPS | Six held-out views: mean PSNR / LPIPS |
|---|---:|---:|
| Fresh 5k | 21.064 / 0.506 | 17.468 / 0.578 |
| Continued 10k | 23.468 / 0.413 | 17.336 / 0.541 |
| Continued 15k | 25.140 / 0.340 | 17.397 / 0.521 |

These six views are deliberately selected problem/control views, not the full
35-view evaluation. The full earlier evaluation improved in mean PSNR; that does
not contradict the flat mean PSNR of this subset. Held-out LPIPS improves overall
here, so continuation does not worsen every quality measure. Training and held-out
are different frames; their gap is descriptive rather than an exact controlled
generalization-error measurement.

| Region | Nearby train mean PSNR: 5k -> 15k | Held-out PSNR: 5k -> 15k | Held-out median absolute depth residual at 15k |
|---|---:|---:|---:|
| Corridor | 20.90 -> 26.26 | 16.80 -> 15.71 | 0.758 m |
| Exterior stairs | 21.74 -> 24.22 | 18.51 -> 18.85 | 0.432 m |
| Outdoor pavement control | 25.68 -> 28.83 | 25.28 -> 27.04 | 0.106 m |
| Stair entrance | 21.69 -> 26.54 | 10.48 -> 9.91 | 0.812 m |
| Stair interior | 15.37 -> 22.36 | 12.27 -> 12.10 | 0.693 m |
| Lab sign | 20.99 -> 22.28 | 21.47 -> 20.78 | 0.524 m |

The largest local train/held-out gaps widen considerably. At 15k the corridor gap
is 10.55 dB, the stair-entrance gap 16.62 dB, and the stair-interior gap 10.27 dB.
The control gap is only 1.79 dB. This supports poor local generalization alongside
the previously measured uneven coverage/alignment. It does not establish that
every training render is good: visual review of training frame 127.768511 still
shows substantially blurred stairs, and corridor training walls remain streaky.

Depth adds a separate indication of incorrect surface placement. In the 15k stair
interior, median rendered depth is 0.231 m versus measured 0.933 m. At the entrance
it is 0.394 versus 1.231 m, and in the corridor 0.925 versus 1.665 m. The control
is 1.985 versus 1.991 m. Medians are summaries of the same comparison mask, not
correspondences between particular points; the absolute residual column above is
computed per pixel before taking its median.

Using the diagnostic tolerance of 5 cm + 3% of measured depth, expected rendered
depth is in front of the observed surface at 100% of compared stair-interior
pixels, 94.3% at the entrance, and 97.2% in the corridor. Comparison uses valid
0.1–8 m sensor depth and opacity >0.9. These percentages describe rendered-depth
conflict, not the fraction of Gaussian primitives that are floaters. Edges are
not excluded in this render-depth check. Neighboring training views also have
large depth residuals: their mean per-view median errors at 15k are 0.609 m in
the interior and 0.991 m in the corridor, versus 0.102 m in the control. Thus
the geometry mismatch is present even where training RGB scores improve.

Nerfstudio 1.1.5 requests `RGB+ED` and returns its depth channel directly during
evaluation; this is expected accumulated splat depth, not a mesh surface or
thickness measurement. See the [versioned Splatfacto implementation](https://github.com/nerfstudio-project/nerfstudio/blob/v1.1.5/nerfstudio/models/splatfacto.py#L546-L601).
The saved dataparser scale converts that depth to source units. Aligned optical-
axis sensor depth in millimetres remains an assumption inherited from this
project's RGB-D pipeline. The good outdoor depth agreement and strongly varying
regional depth ratios argue against a single global conversion factor explaining
the failures, but do not fully validate sensor calibration.

**Conclusion:** the fresh-control scene has substantial local geometric mismatch
and weak generalization in the stair/corridor regions. Longer RGB-only training
fits neighboring training images better while leaving those held-out regions
foggy. This is stronger evidence than the initial coverage check alone, but it
does not isolate whether poses, initialization, missing depth constraints, or
their interaction is the principal cause.

**Recommended next experiment:** validate/refine local RGB-D alignment first, then
test a confidence-masked depth-supervised splat model against the unchanged fresh
control. Gate on held-out RGB quality and sensor-depth agreement in these regions,
with the pavement control guarding against regressions. Treat camera refinement
and depth supervision as separate ablations. Another longer RGB-only run or a
blanket post-training pruning pass is not supported as the next priority. No new
training experiment has been launched.

Review `outputs/scene_diagnosis/renders.html` for all 53 source/5k/10k/15k rows;
the six held-out views appear first. Validated aggregate and per-region numbers
are in `outputs/scene_diagnosis/summary.json`.

## Measured results

Six held-out scene views were checked against up to eight neighboring training
views, excluding every held-out camera by filename. Exported intrinsics exactly
match the raw calibration scaled by 0.5. The inspected raw RGB and depth arrays
are both 1920x1080. Exported JPEG versus resized original RGB mean absolute
differences are 0.68–1.40 on the 0–255 scale: no gross wrong-frame or image-scale
error is apparent in these six views. Matching filenames do not prove physical
RGB/depth synchronization or calibration.

| Region / timestamp | Valid measured depth in image | Valid depth supported by >=2 nearby training views | Surface-seed tile coverage |
|---|---:|---:|---:|
| Corridor / 4.569 s | 37.3% | 62.3% | 53.5% |
| Exterior stairs / 51.502 s | 26.8% | 48.3% | 57.6% |
| Outdoor pavement control / 64.302 s | 54.8% | 90.0% | 68.8% |
| Stair entrance / 98.435 s | 44.2% | 34.5% | 58.3% |
| Stair interior / 128.302 s | 50.6% | 74.3% | 65.3% |
| Lab sign / 145.369 s | 61.5% | 99.0% | 68.8% |

Support is a deliberately local, depth-aware measure, not total camera coverage.
The denominator excludes invalid depth but includes depth discontinuities removed
from cross-view testing; these values are conservative. Missing depth, occlusion,
view changes, and pose/depth inconsistency can all reduce support. Seed coverage
counts occupied 16x9 image tiles containing a seed consistent with observed depth;
its denominator is the whole image, including invalid-depth regions. It is not
mesh completeness, and frustum seed counts include occluded points.

## Independent appearance/alignment check

SIFT matches at 480x270 were filtered with a fitted fundamental matrix independent
of the exported poses. Source measured depth and exported poses then predicted the
target feature positions. Errors below are converted to 960x540 export pixels.

| Region | Median reprojection error, preceding / following training frame | Depth-valid fitted-epipolar inliers |
|---|---:|---:|
| Corridor | 6.05 / 6.28 px | 19 / 10 |
| Exterior stairs | 1.45 / 1.46 px | 33 / 60 |
| Pavement control | 1.24 / 1.41 px | 126 / 157 |
| Stair entrance | 4.79 / unavailable | 19 / 0 |
| Stair interior | 123.75 / 11.20 px | 3 / 6: inconclusive |
| Lab sign | 10.54 / 9.62 px | 53 / 57 |

The stair interior's repetitive steps and very low inlier counts make its feature
numbers unreliable; they are retained in JSON for audit, not treated as evidence
of a measured 124-pixel pose error. Exterior stairs have large upper-tail errors
(p90 about 31–34 px), despite low median errors; depth edges or mismatches remain
possible. The control's p90 is only 2.2–2.5 px.

Export-pose epipolar line distances, which do not use the depth values in their
calculation, are about 5.0–5.1 px in the corridor and 0.7–1.1 px in the control.
Near the lab sign they are asymmetric, 1.8 / 8.1 px. This adds evidence of local
camera/image geometric inconsistency, but descriptor errors, calibration, and
motion/rolling shutter remain possible. This is not proof of a single pose bug.

## Interpretation

1. A simple global intrinsics scaling error is not supported by these checks.
2. The stair entrance and exterior stairs have weaker measured local support than
   the pavement control. The source stair-entrance image also visibly has motion
   blur; the stair-interior image has little texture and missing depth on walls.
3. Corridor and lab-sign appearance alignment is worse than the control. Large
   smooth walls can agree in depth while their texture is misregistered, so depth
   agreement alone is insufficient to establish training consistency.
4. More iterations cannot recover missing measurements or resolve contradictory
   fixed-pose image evidence by themselves. This is a plausible explanation for
   the previously observed local regressions during continuation, not yet a
   demonstrated causal attribution.

The completed train/held-out comparison above supports both local generalization
failure and geometric mismatch. It does not support a broad parameter sweep.

## Reproducing the render checks

Already completed in the user's Docker-enabled PowerShell:

```powershell
& 'E:\Others\Project\sfm_reconstruction\scripts\run_scene_diagnosis.ps1'
```

This evaluates the existing 5k, 10k, and 15k checkpoints at the six target cameras
and their neighboring training cameras. It saves source/render pairs, PSNR/SSIM/
LPIPS, and rendered depth versus measured depth, with no optimization. Each
checkpoint is explicitly loaded from its own directory so resume configuration
fields cannot silently select the original checkpoint. Depth is converted using
the saved dataparser scale; comparison requires valid measured depth and rendered
opacity >0.9. Expected rendered depth does not measure surface thickness; that
part of the original proposed diagnosis remains unresolved.

Outputs: `outputs/scene_diagnosis/index.html`, `geometry.json`,
`render_selection.json`, then `render_{5000,10000,15000}/metrics.json` and
`render_complete.json`. Run `scripts/summarize_scene_diagnosis.py` locally to
revalidate these outputs and rebuild `renders.html` and `summary.json`.

## Validation

Two synthetic tests passed: a known camera translation projects a plane by the
expected pixel displacement; foreground occlusion is distinguished from a
free-space conflict. All six real-data diagnostics ran successfully. Visual QA
of stair entrance and interior panels confirmed that depth/support maps follow
the visible scene. The completed GPU results have the intended checkpoint steps,
exact expected camera selections, finite RGB metrics, and 159 readable 960x540
depth/opacity arrays and RGB pairs. Depth statistics and split means were
recomputed from saved data; six held-out PSNR values per checkpoint reproduce
the earlier evaluation within 0.0001 dB. Visual review covered held-out stairs,
training stairs, training corridor, and the held-out pavement control.
