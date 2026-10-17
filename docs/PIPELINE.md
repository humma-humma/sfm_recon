# Published pipeline and reproduction entry points

The repository contains the classical reconstruction and RGB-D SLAM package,
optional native C++/Ceres backends, research adaptation and rendering scripts,
dataset recipes, tests, experiment reports, and the static demo deployment.
The project is a research pipeline with multiple stages, not one command that
downloads datasets and recreates every experiment.

## Stage map

| Stage | Main implementation / entry point | Supporting documentation |
|---|---|---|
| Classical SfM | `src/sfm_reconstruction/{matching,tracks,geometry,reconstruction}.py`; `sfm-reconstruct` | [Reproduction](REPRODUCTION.md) |
| Bundle adjustment | `bundle_adjustment.py`, `native_backend.py`, `native/kernels.cpp`, `native/ceres_bundle.cpp` | [Native build and usage](CPP_INTEGRATION.md), [Ceres results](CERES_RESULTS.md) |
| RGB-D odometry and learned loop closure | `stage3.py`, `pose_graph.py`, Stage 3 CLI tools | [Stage 3 setup](STAGE3_SETUP.md), [learned augmentation](LEARNING_BASED_AUGMENTATION.md) |
| Dataset-specific geometry recipes | `configs/presets/`, `scripts/run_preset.py` | [Parameter checks](HYPERPARAMETER_CHECK.md), [native follow-up](CPP_FOLLOWUP.md) |
| Splat exports and native camera support | `gaussian_splatting_export.py`, `stage3_*gaussian*.py`, `native/kernels.cpp` | [Splat workflow](GAUSSIAN_SPLATTING.md), [native splat results](SPLATTING_CPP_RESULTS.md) |
| GNT adaptation and inference | `scripts/run_gnt_tuning.ps1`, `run_gnt_finalization.ps1`, `gnt_local_runtime.py` | [GNT tuning](GNT_TUNING.md), [finalization](GNT_FINALIZATION.md) |
| Balanced Gaussian fitting | `scripts/run_gnt_balanced_all.ps1`, `fit_gnt_rollout_object.py` | [Balanced fitting](GNT_BALANCED_ALL.md) |
| Full scene and intermediate-frame comparison | `scripts/run_gnt_scene15.ps1`, `render_scene15_splats.py`, `finish_scene15_splat.ps1` | [15-view/s scene](GNT_SCENE15.md), [corrected-splat comparison](GNT_SCENE15_SPLATS.md) |
| Interactive demo | `web_demo/`, `scripts/build_demo_site.py`, `.github/workflows/demo-pages.yml` | [Portable demo build](../web_demo/README.md) |

Paths in the implementation column are relative to the repository root except
for package modules, which are under `src/sfm_reconstruction`.

## CPU and native setup

Follow the root README for a Python installation. Use the same interpreter for
building and testing native extensions. C++17, CMake, pybind11 headers and optional
Ceres 2.2/Eigen dependencies are documented in [CPP_INTEGRATION.md](CPP_INTEGRATION.md).
Python remains the default. Fast presets explicitly select native backends and
therefore require their corresponding extensions.
The macOS wheel workflow uses a pinned Eigen-only/miniglog Ceres source build
to avoid bundling conflicting OpenMP runtimes; Linux and Windows use the
recorded conda Ceres dependency profile.

Presets retain historical paths, but `run_preset.py --dataset` overrides the
dataset location and `--generate-matches` rebuilds Stage 2 correspondences using
the recorded matching configuration. See [the current reproduction commands](REPRODUCTION.md#8-current-saved-presets-without-the-authors-directory-layout).
A basic saved-recipe invocation is:

```bash
python scripts/run_preset.py --preset /path/to/local_recipe.json --output outputs/my_run
```

Datasets and official dataset evaluators must be obtained separately. See
[DATASETS.md](DATASETS.md). Benchmark scripts expose synthetic checks and
dataset-specific options; experiments that use saved reconstruction inputs need
those earlier stages first.

## Neural stages and external dependencies

GPU training/rendering uses PyTorch, Nerfstudio/gsplat and an NVIDIA CUDA runtime.
The recorded orchestration scripts use Docker and PowerShell, using the
immutable image digest recorded in `configs/environments/neural-runtime.json`
(original Nerfstudio tag `1.1.5`). Many runners use `--pull never`;
the image must already be present. Review each runner's parameter block before
launching. Override `Python`, `FFmpeg`, `TrialRoot`, `Destination` and dataset paths
where available; executables default to PATH. Other artifact paths preserve
the historical stage layout. See the reproduction guide for dependency order.

`scripts/run_lightweight_methods.ps1 -Stage Setup -GNTOnly` records and retrieves the
upstream GNT checkout at commit `53e0fc3d39b4244503d6c4de1e34bbf3199dd2a8`, its
official pretrained checkpoint, while skipping the unrelated 2DGS trial build. Omit `-GNTOnly` for the
historical two-method setup. Adaptation and
final rendering require the earlier fixed-camera exports/manifests described in
the stage reports. The upstream GNT and 2DGS code, datasets and model weights
are not vendored; their licenses and terms apply. The small BasicSR architecture
adaptation in `scripts/vendor` includes its attribution and Apache-2.0 license.

The accepted outputs are balanced Gaussian object routes and the 15-view/s GNT
scene route motion compensated to 30fps. The corrected-splat scene remains a
comparison option. Historical restoration, 2DGS, depth and splat-fitting branches
are retained as experiments, not all presented as selected methods.

## What is deliberately excluded

Original datasets, learned checkpoints, compiled extensions, solver DLLs,
generated output directories, temporary profiles and caches remain outside Git.
Final demo videos are versioned GitHub release assets, downloaded and verified
by the Pages workflow. Source and wheel distributions are separate from those
media assets. The detailed local retrospective and its evidence bundle remain
ignored at the owner's request; the technical stage reports in `docs/` are public.

Historical reports preserve their original measurements and artifact paths.
Some rejected checkpoints and intermediate files were removed during cleanup;
those paths are provenance records, not promises of downloadable artifacts.
Reproduction may require rerunning upstream stages. GPU training and full-route
rendering are not exercised by ordinary CPU CI.
