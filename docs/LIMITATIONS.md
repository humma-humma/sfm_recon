# Limitations and next steps

## Current limitations

### Reconstruction

- Incremental SfM is designed for calibrated project datasets rather than every
  camera model or arbitrary internet photo collection.
- Repeated texture, weak baselines, and sparse coverage can still produce track
  conflicts or underconstrained points.
- Dense fusion is an OpenCV reference implementation, not a production MVS
  system.

### Learned features

- SuperPoint/LightGlue and DINOv2 increase dependency and GPU complexity.
- Learned matching did not universally beat balanced SIFT in these experiments.
- A learned frontend is promoted only where geometric verification protects the
  existing pose solution.

### RGB-D SLAM

- The full sequence is not real time.
- Stairs and sparsely observed surroundings remain the weakest regions.
- One useful loop is recovered; additional unrestricted loop edges can degrade
  the trajectory.

### Gaussian splatting

- Rendering cannot repair incorrect or weakly covered camera geometry.
- More training steps may densify unsupported regions into noise.
- Checkpoint cleanup is support-based presentation processing, not a trajectory
  metric improvement.

### Distribution

- Original datasets and trained checkpoints cannot be reproduced from the Git
  repository alone because they are not redistributed.
- Published videos are evidence artifacts; source frames and editing footage
  remain local to keep repository size controlled.

## High-value next steps

1. Validate physical RGB/depth alignment and improve local pose consistency in
   the corridor and stair entrance. The current diagnostics do not isolate a
   single cause of their errors.
2. Extend the measured-depth research comparison with independent datasets and
   repeated runs; the present ablation is exploratory, not statistical validation.
3. Compare surface representations such as 2DGS if reliable alignment/depth
   supervision still leaves poor geometry. These methods are not implemented here.
4. Perform browser/device interaction QA and add collision only against validated
   surfaces. The demo currently provides free flight, not collision-safe walking.

Optional C++/Ceres backends and benchmark tooling are now implemented and
evaluated; see [native follow-up](CPP_FOLLOWUP.md). Measured-depth supervision
and a confidence-masked TSDF baseline are documented in
[research results](DEPTH_RESEARCH_RESULTS.md). Missing sensor observations and
low local coverage remain limitations even when visible mesh surfaces are accurate.
