# Gaussian Splatting

Gaussian Splatting is a downstream rendering experiment in this project. It
does not replace the Stage 1/2 geometry metrics or the Stage 3 trajectory ATE.

## Saved state - 2026-08-07

The experiment is prepared but training has not started:

- The completed Stage 1-3 implementation was committed and pushed separately
  as `6cd1392`.
- Docker Desktop 4.47.0 is installed and its WSL2 Linux engine works.
- CUDA container passthrough was verified on the NVIDIA GeForce RTX 2080 with
  8,192 MiB VRAM and driver 591.86.
- `nvidia/cuda:11.8.0-base-ubuntu22.04` is installed locally.
- Nerfstudio is pinned to `ghcr.io/nerfstudio-project/nerfstudio:1.1.5`.
- The Nerfstudio image pull was interrupted before completion. There is no
  completed Nerfstudio image and no training container or model checkpoint.
- The half-resolution milk COLMAP export is complete and validated at
  `outputs/stage2_milk_gaussian_colmap_half/` (50 cameras, 1,739 seed points,
  960 x 540 images).
- The Gaussian exporter, tests, and this document are intentionally local and
  uncommitted until the experiment is completed.

Resume from the repository root with:

```powershell
docker pull ghcr.io/nerfstudio-project/nerfstudio:1.1.5
docker image inspect ghcr.io/nerfstudio-project/nerfstudio:1.1.5 `
  --format '{{index .RepoDigests 0}}'
```

After the pull, first run a CUDA/parser smoke test and a 1,000-iteration
no-viewer training probe. Proceed to the 30,000-iteration fixed-pose benchmark
only if the probe loads all 50 cameras, initializes from 1,739 points, and stays
within the 8 GB VRAM limit.

## First target

Start with the promoted Stage 2 milk reconstruction, not Stage 3:

- 50/50 registered cameras.
- Short, bounded camera orbit.
- Existing colored SfM initialization with 1,739 points.
- Stage 3 still has enough long-loop drift to risk duplicated or smeared
  appearance.

The first export is:

```text
outputs/stage2_milk_gaussian_colmap_half/
  images/                     50 JPEG images at 960 x 540
  sparse/0/cameras.txt        scaled PINHOLE intrinsics
  sparse/0/images.txt         estimated world-to-camera poses
  sparse/0/points3D.txt       colored SfM initialization
  gaussian_splatting_export.json
```

Recreate it with:

```powershell
sfm-export-gaussian `
  --dataset "..\Experiments\Stage_2_Data\stage2\milk" `
  --result-dir "outputs\stage2_milk_masked_track3_ba" `
  --output-dir "outputs\stage2_milk_gaussian_colmap_half" `
  --image-scale 0.5
```

The export contains COLMAP text files so it can be read without rerunning
COLMAP. The points intentionally omit observation tracks because Gaussian
Splatting loaders use their positions and colors for initialization; the
camera poses remain the promoted SfM estimates.

## Recommended trainer

Use Nerfstudio Splatfacto in a separate environment. Its default model is
documented at about 6 GB of GPU memory, making it a better first fit for the
local RTX 2080 (8 GB) than `splatfacto-big` or the paper-quality reference
configuration. Do not add it to the SfM runtime environment.

After installing Nerfstudio according to its current official instructions:

```powershell
ns-train splatfacto `
  --data "outputs\stage2_milk_gaussian_colmap_half" `
  --output-dir "outputs\stage2_milk_splatfacto" `
  colmap `
  --colmap-path "sparse/0" `
  --downscale-factor 1 `
  --eval-mode interval `
  --eval-interval 8
```

Keep camera optimization off for the first benchmark (it is off by default in
Splatfacto). This ensures the rendering experiment measures the promoted SfM
poses instead of silently changing them.

Evaluate held-out images and export the final splat with:

```powershell
ns-eval --load-config "<run>\config.yml" `
  --output-path "outputs\stage2_milk_splatfacto\evaluation.json"

ns-export gaussian-splat --load-config "<run>\config.yml" `
  --output-dir "outputs\stage2_milk_splatfacto\export"
```

## Acceptance criteria

Record all of the following before calling the demo successful:

1. Held-out PSNR, SSIM, and LPIPS from the fixed interval split.
2. Side-by-side held-out RGB and rendered images, especially near the back of
   the milk object where Stage 2 coverage is weakest.
3. Peak GPU memory, training time, final Gaussian count, and exported PLY size.
4. A novel-view orbit inspected for floaters, elongated splats, background
   leakage, and duplicated surfaces.
5. The exact source reconstruction and export scale.

Compare a later learned-pose or learned-point run only with the same split,
resolution, trainer version, seed, and training settings. Gaussian metrics are
presentation metrics; the existing pose, reprojection, coverage, and mesh-proxy
metrics remain authoritative for geometric accuracy.

## Stage 3 gate

The initial Stage 3 fixed-pose probe is implemented. It uses the promoted
pose-graph trajectory, not raw visual odometry or ground-truth alignment:

```powershell
sfm-export-stage3-gaussian `
  --dataset "..\Experiments\Stage_3_Data\stage3" `
  --trajectory `
    "outputs\stage3_pose_graph_keyframe_stride8_nfev50\estimated_camera_trajectory_pose_graph.txt" `
  --point-cloud `
    "outputs\stage3_pose_graph_keyframe_stride8_nfev50\stage3_scene_est_pose_dense.ply" `
  --output-dir "outputs\stage3_gaussian_colmap_stride16_half" `
  --frame-stride 16 `
  --image-scale 0.5 `
  --max-points 200000
```

The validated export contains 276 keyframes at 960 x 540 and 200,000
deterministically sampled GT-free seed points. It explicitly includes the final
loop frame. The fixed-pose 1,000-iteration Splatfacto probe completed on the
RTX 2080 with about 3.4 GB of observed VRAM usage:

```powershell
docker run --rm --gpus all `
  -v "${PWD}:/workspace" `
  -v "${PWD}/outputs/nerfstudio_cache:/root/.cache" `
  -w /workspace `
  ghcr.io/nerfstudio-project/nerfstudio:1.1.5 `
  ns-train splatfacto `
  --data /workspace/outputs/stage3_gaussian_colmap_stride16_half `
  --output-dir /workspace/outputs/stage3_splatfacto_probe `
  --experiment-name stage3_stride16_half_fixed_pose `
  --timestamp probe_1000 `
  --max-num-iterations 1000 `
  --steps-per-save 1000 `
  --pipeline.model.camera-optimizer.mode off `
  --vis tensorboard `
  colmap --colmap-path sparse/0 --downscale-factor 1 `
  --eval-mode interval --eval-interval 8
```

Stage 3 remains a scene-reconstruction presentation experiment. Long-sequence
pose drift may appear as doubled or smeared geometry, and the splat should not
be presented as an improvement over the official Stage 3 ATE of `3.0305`.
