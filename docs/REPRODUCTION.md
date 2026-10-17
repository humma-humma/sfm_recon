# Reproduction guide

This document separates fast code verification from dataset-dependent research
runs and GPU-dependent Gaussian rendering.

## 1. Environment

Python 3.10 and 3.11 are supported.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -c configs/environments/cpu-constraints.txt -e ".[dev,evaluation,visualization]"
```

On Windows PowerShell, activate with `.\.venv\Scripts\Activate.ps1` instead
of `source .venv/bin/activate`. Run the remaining commands in that environment
from the repository root.

Optional extras:

```bash
python -m pip install -e ".[open3d]"
python -m pip install -e ".[learned]"
```

The learned extra does not install third-party model repositories. Kornia
provides the SuperPoint/LightGlue path; DINOv2 weights are loaded through
PyTorch Hub when the place-recognition mode is requested.

## 2. Code-only verification

No project dataset is required for the unit suite:

```bash
python -m pytest -q -p no:cacheprovider
python -m build
```

All collected tests must pass. Native and PyTorch tests skip when their optional
components are absent; a skipped optional test is not evidence that that component
was exercised. The separate native-wheel workflow builds, repairs and tests Ceres
wheels on three operating systems and Python 3.10/3.11.

The CPU constraints file records tested top-level versions, not a complete
transitive lock. The neural Docker environment is separate; do not install these
CPU constraints into the recorded CUDA image.

On Windows, the wrapper also checks locally available result artifacts:

```powershell
.\scripts\run_verification.ps1 -Python python
```

Artifact checks are skipped only when their documented local outputs are not
present; the unit suite remains the portable acceptance gate.

## 3. Image-based SfM

```bash
sfm-reconstruct \
  --stage 2 \
  --dataset /path/to/stage2/milk \
  --output-dir outputs/stage2_milk \
  --max-features 2500 \
  --feature-mode sift \
  --mask-apriltags \
  --min-track-observations 3 \
  --max-point-distance-factor 1.5 \
  --bundle-adjustment-max-nfev 20
```

Use `--feature-mode superpoint-lightglue` after installing the `learned` extra.
The promoted object results retain fixed cameras when learned points are added;
see `LEARNING_BASED_AUGMENTATION.md` for the controlled ablations.

## 4. RGB-D SLAM

A bounded smoke run:

```bash
sfm-stage3-setup \
  --dataset /path/to/stage3 \
  --output-dir outputs/stage3_smoke \
  --max-frames 500 \
  --run-rgbd-odometry \
  --max-features 1800 \
  --min-matches 20 \
  --min-pnp-inliers 8
```

Full pose-graph and learned loop-closure options are listed by:

```bash
sfm-stage3-setup --help
```

The original published benchmark uses guarded full-sequence RGB-D VO, keyframe-stride-8
pose-graph optimization, and automatic DINOv2 retrieval with local geometric
verification. Ground truth is used only for evaluation, never for the promoted
reconstruction or Gaussian initialization.

## 5. Fixed-pose Gaussian export

Object export:

```bash
sfm-export-gaussian \
  --dataset /path/to/stage2/milk \
  --result-dir outputs/stage2_milk \
  --output-dir outputs/stage2_milk_colmap \
  --image-scale 0.5
```

SLAM export:

```bash
sfm-export-stage3-gaussian \
  --dataset /path/to/stage3 \
  --trajectory outputs/stage3_pose_graph/estimated_camera_trajectory_pose_graph.txt \
  --point-cloud outputs/stage3_pose_graph/stage3_scene_est_pose_dense.ply \
  --output-dir outputs/stage3_colmap \
  --frame-stride 16 \
  --image-scale 0.5 \
  --max-points 200000
```

Training is performed with Nerfstudio Splatfacto in its official container.
Exact checkpoint cleanup and rendering commands are documented in
`GAUSSIAN_SPLATTING.md`. Checkpoints, container caches, and generated datasets
remain outside Git.

## 6. Portfolio rendering

`sfm-stage3-portfolio-video --help` exposes the camera-path, point-cloud,
trajectory, and composition helpers used for the published timelines. All
published transitions use matched camera-frame indices. Three curated
reconstruction timeline MP4s and contact sheets are tracked under
`assets/portfolio`; intermediate clips remain local. The final neural demo
videos are distributed through the hash-checked release described in section 10.

## 7. Expected hardware and time

| Task | Hardware | Typical scope |
|---|---|---|
| Unit tests | CPU | Seconds |
| Stage 2 SfM | CPU | Minutes, dataset dependent |
| Stage 3 smoke VO | CPU | Minutes for 500 frames |
| Full Stage 3 | CPU | Hours, configuration dependent |
| Splat training/rendering | NVIDIA GPU | Minutes to hours |

Runtime varies substantially with image count, resolution, feature count, and
optimization limits. The repository does not claim real-time execution.

## 8. Current saved presets, without the author's directory layout

Build the optional kernels/Ceres solver as described in
[CPP_INTEGRATION.md](CPP_INTEGRATION.md) before using native presets. Missing
backends fail before matching; they never silently fall back to Python.

For supplied Stage 1 correspondences:

```bash
python scripts/run_preset.py --preset configs/presets/stage1_box_fast.json --dataset /data/stage1/box --output outputs/my_box --seed 0
```

For Stage 2, generate matches in the new output folder instead of depending on an
ignored historical cache:

```bash
python scripts/run_preset.py --preset configs/presets/stage2_milk_fast.json --dataset /data/stage2/milk --generate-matches --output outputs/my_milk --seed 0
python scripts/run_preset.py --preset configs/presets/stage2_boot_fast.json --dataset /data/stage2/boot --generate-matches --output outputs/my_boot --seed 0
```

Alternatively pass `--correspondences /path/to/correspondences`. Overrides resolve
relative to your current working directory. Archived paths in presets retain their
repository-relative meaning. `preset.json` in the output records effective paths
and settings; `environment.json` records Python/package versions, seed, source Git
revision/dirty state, runner hash and input-preset hash. `run.json` records the achieved counts/runtime.
A seed alone does not ensure cross-version or cross-platform numerical identity.
The full historical matching settings are now embedded in each Stage 2 preset.

For a fresh complete scene frontend, omit `--trajectory`:

```bash
python scripts/run_preset.py --preset configs/presets/stage3_scene.json --dataset /data/stage3 --output outputs/my_scene --seed 0
```

This recipe uses the recorded verified-loop measurement. Re-running automatic
learned loop discovery is a distinct experiment; see Stage 3 documentation. Scene
output evaluation still needs the dataset's ground truth and official evaluator.
To reuse a frontend, explicitly add `--trajectory /path/to/trajectory.txt`; frame
count and timestamps are validated.

To regenerate the fresh splat input exports, copy the four `*_fast` / scene
recipes to a local preset directory, edit their `dataset` paths, then run:

```bash
python scripts/prepare_splat_refresh.py --preset-dir /path/to/my_presets --datasets box boot milk scene --generate-matches
```

Outputs go to `outputs/splat_refresh_v1`. This command no longer silently depends
on the ignored `scene_frontend/full_validation/trajectory.txt`; it computes the
scene frontend unless `--scene-trajectory` is explicitly supplied. It can be an
expensive CPU run. Existing completed manifests are preserved and their preset
hashes checked. Use fresh outputs when intentionally changing a recipe.

## 9. Neural runtime and dependency order

The base image is pinned by immutable digest in the runners and in
[neural-runtime.json](../configs/environments/neural-runtime.json). Pull it once:

```bash
docker pull ghcr.io/nerfstudio-project/nerfstudio@sha256:b59b8e1012d7a43679d3234b3de9c8416a4b8435fcbf21b9d8c4494b8563f19e
```

It contains Python 3.10.12, PyTorch 2.1.2+cu118, Nerfstudio 1.1.5 and gsplat 1.4.0.
The recorded platform is Linux x86-64 with an NVIDIA GPU (originally RTX 2080),
accessed from Linux or Docker Desktop/WSL2. A compatible host NVIDIA driver and
NVIDIA Container Toolkit/GPU passthrough are separate requirements. This old
CUDA runtime is not promised to support every newer GPU architecture.

The recorded GPU orchestration is Windows/PowerShell-oriented; a complete native
Linux host runner is not provided. PowerShell runners now default to `python` and `ffmpeg` on PATH; `-Python` and
`-FFmpeg` accept explicit executable paths. Node 20+, Chrome (for the existing
Windows browser QA runners), Git, PowerShell and FFmpeg are additional host tools.
Some historical browser helpers remain Windows-specific; portable site packaging
and link/hash checks do not require Chrome.

Install/check the pinned GNT source and official pretrained checkpoint without
building the rejected 2DGS CUDA extensions:

```powershell
.\scripts\run_lightweight_methods.ps1 -Stage Setup -GNTOnly -TrialRoot /path/to/gnt_trial
```

This stage clones external code and downloads/checks the model, but does not
render or adapt GNT. Use a fresh trial directory when changing upstream source or
inputs. The historical full setup still supports 2DGS and now derives its CUDA
architecture from the detected GPU instead of hard-coding 7.5.

The neural stages consume generated artifacts; launching a later runner in an
empty clone is not sufficient. Follow this dependency order:

| Step | Producer / guide | Inputs required before launching |
|---|---|---|
| Geometry and COLMAP inputs | Sections 3-5/8 above; [Gaussian export and training](GAUSSIAN_SPLATTING.md) | Dataset, poses, reconstructed points and fixed train/test split |
| Fixed-pose splats and camera-case manifest | Splat training guide, then `run_sharpness_trials.ps1` | Trained fixed-pose splat checkpoints and their matching COLMAP/dataparser transforms; writes `render_manifest.json` in scratch |
| GNT initialization / adaptation | `run_gnt_tuning.ps1`, `run_gnt_balanced.ps1`, `run_gnt_detail.ps1`; [GNT tuning](GNT_TUNING.md), [detail](GNT_DETAIL.md) | Pinned upstream GNT/weights, scratch manifest, COLMAP data; final accepted scene checkpoint is `gnt_tuning/detail_control_006000.pth` |
| Object GNT adaptation and inference tuning | `run_gnt_finalization.ps1`; [Finalization](GNT_FINALIZATION.md) | Prior scene adaptation, object source data and scratch manifest; creates runtime/compile reports and object `balanced_006000.pth` weights |
| Scene sparse/dense views and fitting controls | Traversal/dense-turn/splat-fit/dense-supervision/balanced-supervision runners; [traversal](GNT_TRAVERSAL.md), [dense supervision](GNT_DENSE_SUPERVISION.md), [balanced supervision](GNT_BALANCED_SUPERVISION.md) | Adapted scene GNT, native route plans/frames, teacher fit and continuation checkpoints |
| Final object fits and exports | `run_gnt_balanced_all.ps1`; [all-dataset fitting](GNT_BALANCED_ALL.md) | Adapted object GNT; scene control/dense/balanced models already exist because the runner also exports the scene comparisons |
| Full scene 15-view/s route | `run_gnt_scene15.ps1`; [scene route](GNT_SCENE15.md) | Accepted scene GNT, route inputs and source photos; cached views are reusable only after pose/hash validation |
| Intermediate splat alternative | `render_scene15_splats.py`, `finish_scene15_splat.ps1`; [comparison](GNT_SCENE15_SPLATS.md) | Complete 15-view/s route, scene balanced splat model, dataparser/camera transforms and existing motion interpolation for fallback |

`configs/splatting/gnt_inputs.json` now lists the four COLMAP input directories
used by GNT and the fixed-camera manifest renderer. Defaults preserve the original
experiment paths. To use newly generated exports, edit this map to their
repository-relative paths and train the associated fixed-pose splats against those
same cameras. Different camera solutions create a new experiment; do not reuse
incompatible models/transforms or old trial results.

For learned loop retrieval, `SFM_DINOV2_REF` can select a specific upstream Git
commit/tag for PyTorch Hub (for example `export SFM_DINOV2_REF=<commit>`). The default
retains the historical unpinned Hub branch. Record the selected ref, downloaded
weight hashes and learned package versions for a new experiment. The original
Hub revision and every learned dependency were not fully locked at experiment
time; exact historical learned-retrieval reproduction is not established.

These dependency links expose the original research progression, including
rejected controls. They are not a newly validated one-command end-to-end GPU
rebuild. Training and rendering the full selected route again would take many
GPU-hours; this audit does not claim to have done that.

## 10. Reproduce the delivered demo independently of training

For the already accepted outputs, [web_demo/README.md](../web_demo/README.md)
provides a fresh-clone workflow that downloads the versioned release videos,
verifies their SHA-256 hashes, packages the site and serves it locally. This
reproduces the delivered demo, not the learned training process. GitHub Actions
runs the same packaging and integrity checks.

See [the reproducibility audit](REPRODUCIBILITY_AUDIT.md) for verified cases,
remaining limitations and the distinction between replay and regeneration.
