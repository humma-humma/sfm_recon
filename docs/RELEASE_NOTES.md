# Portfolio release notes

## v1.0.1

- Skip the optional PyTorch checkpoint-pruning test when PyTorch is not
  installed, allowing the lightweight CI matrix to test the remaining suite.
- Constrain OpenCV to the validated 4.x API used by AKAZE feature extraction.
- Add an upload-ready 1280×640 JPEG social preview below GitHub's 1 MB limit.

## v1.0-portfolio

This release packages the complete reconstruction project as a public-facing
engineering portfolio.

Highlights:

- Incremental calibrated SfM for supplied and generated correspondences.
- Classical and learned matching ablations with controlled promotion criteria.
- RGB-D visual odometry, loop discovery, and pose-graph optimization.
- Fixed-pose Gaussian exports, temporal/spatial diagnostics, and checkpoint
  cleanup.
- Three curated presentation videos: object reconstruction, inside-view SLAM,
  and exact-orbit exterior SLAM.
- 123 deterministic tests and Python 3.10/3.11 CI.

Large datasets, trained checkpoints, local outputs, and source editing footage
are intentionally excluded from Git. Attach any separately distributable
artifacts to the GitHub release rather than committing them to repository
history.
