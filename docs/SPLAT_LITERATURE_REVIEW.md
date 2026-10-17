# Literature review: persistent splat fog and missing scene detail

Reviewed 2026-10-10, focusing on Matthias Niessner's Visual Computing & AI Lab
and Angela Dai's 3D AI Lab at TUM. This is a targeted literature review, not an
exhaustive survey or a new reconstruction experiment.

Follow-up on 2026-10-11: [diagnosis](SCENE_DIAGNOSIS.md) and the controlled
[measured-depth adaptation plus TSDF baseline](DEPTH_RESEARCH_RESULTS.md) are now
complete. The chronological observations below describe the evidence available
at the time of this review; the linked reports contain the later results.

The evidence supports changing geometric constraints, initialization or
representation before further broad tuning. It does not prove that this dataset
is irrecoverable or establish a single cause of its artifacts.

## What our experiments establish

- Browser projection and spherical-harmonic fixes addressed rendering errors,
  but native gsplat also shows missing detail.
- Earlier full resolution and extended splitting worsened scene quality at 5k.
- Continuing the exact fresh control to 10k/15k improves average held-out quality
  by 0.328/0.363 dB, while stairs and lettering remain poorly reconstructed.
- Aggressive depth-based checkpoint pruning damaged detail. This evaluates
  deleting existing Gaussians, not supervising geometry during training.
- The reviewed comparisons are held-out views. We have not yet established
  whether the problematic regions fit their TRAINING images well. A photograph's
  captured pose is not necessarily a training pose.
- The scene contains 276 exported cameras spanning a long trajectory. Total
  camera count alone does not measure local surface coverage or angular diversity.
- Camera optimization and bilateral appearance correction are off in the fresh
  control. Their absence is a possible contributor, not proof of a failure.

## Relevant work from the requested groups

### QuickSplat (ICCV 2025; Liu, Hollein, Niessner, Dai)

Learns initialization, densification and updates for 2D Gaussian surface
reconstruction, targeting under-observed and textureless indoor regions.
This is particularly relevant to our walls and stairs. Its reported indoor
benchmarks do not establish performance on our mixed indoor/outdoor sequence.
The paper also identifies reflections and moving people as limitations.

[Paper](https://arxiv.org/html/2505.05591v1),
[project](https://liu115.github.io/quicksplat),
[official code](https://github.com/liu115/QuickSplat).

The official README currently lists checkpoints as unreleased. It also requires
MinkowskiEngine and custom rasterizers. Treat it as a strong method reference,
not an immediately available pretrained fix. We did not contact its authors.

### Dense Depth Priors for NeRF (CVPR 2022; Roessle et al., Niessner)

Uses completed depth and uncertainty estimates to constrain reconstruction from
sparse observations. This is a NeRF method; adopting uncertainty-weighted depth
constraints for our splat trainer would be an adaptation, not a reproduction.
Our first depth source should be existing RGB-D measurements where reliable.

[Paper and official code link](https://barbararoessle.github.io/dense_depth_priors_nerf/).

### Neural RGB-D Surface Reconstruction (CVPR 2022; Azinovic et al., Niessner)

Combines an implicit signed-distance surface, RGB-D supervision and pose/camera
refinement. It supports testing surface-oriented geometry and local alignment,
rather than expecting RGB appearance fitting alone to recover clean surfaces.

[Paper](https://arxiv.org/abs/2104.04532).

### BundleFusion (TOG 2017; Dai, Niessner et al.)

Jointly optimizes global camera alignment with sparse/dense correspondences and
reintegrates the surface as poses change. The relevant lesson is to validate
geometry alignment and rebuild dependent geometry consistently. Having Ceres or
a good aggregate trajectory score does not itself establish local image alignment.

[Project, paper and code](https://graphics.stanford.edu/projects/bundlefusion/).

### Diff3R (2026; Liu, Hladky, Niessner, Dai)

Learns initialization for subsequent optimization and uncertainty-dependent
bounds on parameter changes to reduce overfitting in under-constrained regions.
Relevant as a research direction for reliable learned initialization/refinement.
The checked project page links the paper but does not provide a ready inference
recipe/checkpoint; deployment readiness was not established.

[Project](https://liu115.github.io/diff3r).

### 3DGS-LM (ICCV 2025; Hollein et al., Niessner)

Uses a custom CUDA Levenberg-Marquardt optimizer to accelerate fitting at similar
quality. It is primarily a speed reference, not evidence that switching solvers
will remove our fog. Official paper-reproduction instructions use an A100 80GB;
this is not stated as a universal minimum memory requirement.

[Project](https://lukashoel.github.io/3DGS-LM/),
[official code](https://github.com/lukasHoel/3DGS-LM).

### GenRecon (2026; Schmid et al., Dai, Niessner)

Generates scene meshes using a pretrained 3D prior and posed multi-view
conditioning in overlapping chunks. Code and checkpoints are released. This is
interesting for a cleaner traversable demo, but inferred unobserved content
cannot be treated as measured reconstruction. Its CUDA/Flash-Attention dependency
stack needs a separate hardware compatibility check before planning a local run.

[Project](https://kasothaphie.github.io/GenRecon/),
[official code](https://github.com/kasothaphie/GenRecon).

## Related methods outside those groups

- [2DGS](https://github.com/hbb1/2d-gaussian-splatting): oriented surface disks,
  depth-distortion and normal-consistency regularization, mesh extraction.
  A useful representation baseline; no pretrained scene model is needed to
  optimize a scene, though integration and compute remain necessary.
- [Mip-Splatting](https://niujinshuchong.github.io/mip-splatting/): addresses
  scale-dependent rendering artifacts with 3D smoothing and a 2D Mip filter.
  Relevant to zoom behavior, not a guarantee of repairing poor geometry or
  recovering unobserved detail. Our earlier inference-only AA test was not
  a reproduction of its training method.
- [SplatFields](https://github.com/markomih/SplatFields): predicts splat properties
  through neural fields to regularize sparse-view optimization. A possible later
  comparison; adapting it to our long mixed scene is not a drop-in replacement.

## Ranked hypotheses and discriminating checks

1. Weak local constraints/initialization: compare training and held-out renders
   in the same bad regions; measure local view support and seed coverage. Strong
   training fits with weak held-out fits would support a generalization problem.
2. Pose/calibration inconsistency: check cross-view RGB-D reprojection, depth
   surface alignment, timestamp association, image resizing and intrinsics. Poor
   fits even in training views would make this a higher-priority investigation,
   while also leaving representation/optimization issues possible.
3. Inadequate surface constraints: inspect rendered depth and surface thickness
   against reliable measured depth. Color similarity alone cannot validate shape.
4. Appearance/capture violations: inspect motion, exposure, reflections and blur
   in the original images. None has yet been quantified as the dominant cause.
5. Scale filtering: assess separately with matched focal-length/dolly tests;
   this cannot explain all artifacts that already occur at captured views.

## Recommended next decision

Pause broad parameter sweeps. First perform the above diagnostics on one
preselected stair/corridor subscene. Then compare the unmodified continuation
against confidence-weighted measured-depth training, using identical poses,
images, initialization and optimization budget. Validate units, depth convention,
RGB/depth alignment, invalid-pixel masks and depth discontinuities first. Keep
held-out observations out of supervision.

If alignment is sound but surfaces remain poor, compare 2DGS or an RGB-D TSDF
mesh baseline. Mesh geometry also provides collision surfaces for traversal.
If reliable depth/local observations are missing, consider learned priors or
additional capture; label generated completion separately from measured geometry.

Retain numerical gates and user render review. Report surface/depth quality and
worst-view behavior in addition to PSNR/SSIM/LPIPS. No new training, dependency
installation, author contact or selected-model promotion was performed in this review.
