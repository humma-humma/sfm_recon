# Dataset layouts

Datasets are local inputs and are not redistributed by this repository. Do not
open an issue requesting the original assignment data.

## Stage 1

Stage 1 expects calibrated images and supplied pairwise correspondence files.
The loader validates calibration dimensions, image identifiers, and observation
indices before reconstruction. Use `sfm-reconstruct --stage 1 --help` for the
current CLI contract.

## Stage 2

A Stage 2 object directory contains calibrated RGB images and the calibration
metadata consumed by `load_image_dataset`. Correspondences are generated into
the output matching cache.

Conceptual layout:

```text
stage2/object/
  images or numbered RGB files
  calibration metadata
```

Run with `--mask-apriltags` when calibration boards should be excluded from
feature generation and point-cloud presentation.

## Stage 3 RGB-D

Stage 3 expects synchronized RGB and depth streams, camera intrinsics, and
timestamps. Ground-truth trajectory data is optional and evaluation-only.

```text
stage3/
  rgb/
  depth/
  rgb.txt
  depth.txt
  camera intrinsics / calibration
  gt_camera_trajectory.txt   # optional
```

Run the validation path before a long sequence:

```bash
sfm-stage3-setup --dataset /path/to/stage3 --output-dir outputs/layout_check
```

File names and association rules vary between dataset releases; validation
errors identify missing paths and timestamp mismatches. See `STAGE3_SETUP.md`
for additional diagnostics.

## Data policy

- Keep datasets outside the repository or under ignored `data/`/`datasets/`.
- Never commit learned weights, Nerfstudio caches, or Gaussian checkpoints.
- Record dataset version, image scale, frame stride, and pose source in result
  summaries.
- Ground-truth-aligned clouds are diagnostics, not GT-free reconstruction output.
