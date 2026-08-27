# Results and experiment provenance

## Promotion policy

A run is promoted only when it improves the intended quantity without silently
changing another one. Camera accuracy, point coverage, rendering quality, and
presentation cleanup are reported separately.

## Stage 1: supplied correspondences

| Run | Points | GT -> estimate | Estimate -> GT | Chamfer |
|---|---:|---:|---:|---:|
| Supplied baseline | 5,746 | `0.40805` | `0.10437` | `0.51242` |
| Fixed-pose learned augmentation | 6,888 | `0.40171` | `0.10291` | `0.50461` |

The augmentation accepts 1,142 learned tracks while copying the promoted
classical cameras. This supports a narrow claim: learned correspondences add
useful geometry when pose quality is protected.

Pure learned replacement was worse (`0.6111` best Chamfer) and is retained as a
negative ablation rather than presented as an improvement.

## Stage 2: milkbox and boot

Balanced SIFT remains the stable geometric result. SuperPoint + LightGlue
produced a useful milkbox presentation run, but learned wide-baseline variants
did not consistently improve translation or mesh-proxy metrics. The boot result
therefore retains the promoted SIFT cameras before 30k-step fixed-pose splat
training.

The portfolio labels this distinction explicitly; it does not fabricate a
learned boot result.

## Stage 3: full RGB-D sequence

| Pose solution | Official ATE |
|---|---:|
| Guarded raw RGB-D VO | `3.70073` |
| Full pose graph | `3.6164` |
| Keyframe pose graph | `3.030521` |
| Learned automatic loop recovery | `3.0305` |

DINOv2 retrieval plus local refinement automatically recovers the useful
`4387 -> 32` loop. It validates the retrieval system without claiming a metric
gain beyond the promoted keyframe solution.

## Gaussian splatting

- Milkbox and boot use 30k-step fixed-pose Splatfacto checkpoints.
- Stage 3 experiments compare 1k and 5k checkpoints because longer unrestricted
  training improves some regions while worsening unsupported indoor/stair areas.
- Frustum-support cleanup is the promoted presentation checkpoint.
- Cleanup changes Gaussian support only; it does not improve SLAM ATE.

The inside-view video shows the chronological route, followed by a speed-matched
full cleaned-splat pass. The exterior alternate applies the exact same orbit
matrices to classical clouds and all splat checkpoints.

## Reproducibility boundary

The repository contains code, tests, configs encoded in commands, and curated
videos. Assignment datasets, trained checkpoints, generated COLMAP datasets,
and multi-gigabyte reconstruction outputs are intentionally excluded. Result
paths in the historical development log provide local provenance but are not
required for installation.
