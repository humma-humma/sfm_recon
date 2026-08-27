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

1. Add a small redistributable synthetic fixture for end-to-end CI.
2. Introduce benchmark tooling for matching, triangulation, and optimization.
3. Port a bounded geometry/optimization backend to C++ using Eigen, Ceres,
   CMake, and pybind11 while preserving Python numerical regression tests.
4. Add stronger spatial support priors for the stairs and weak surroundings.
