# Reproducibility audit - 17 October 2026

## Conclusion

The published code and delivered demo have verified installation/build/test paths.
The classical/native object pipeline can reconstruct from the dataset without
requiring the author's saved matching cache. The selected neural pipeline is
source-complete but remains a multi-stage research reproduction, not a verified
fresh-clone-to-final-render command. Dataset availability alone does not supply
its intermediate trained models, manifests, camera exports and control fits.

[REPRODUCTION.md](REPRODUCTION.md) is the command guide. [PIPELINE.md](PIPELINE.md)
is the stage map. This audit distinguishes a tested rebuild, replay of accepted
outputs, and a documented but unexecuted expensive training chain.

## Findings and improvements

| Area | Original gap | Change / verification |
|---|---|---|
| CPU environment | Broad requirements, missing image-metrics dependency in the old reproduction file; stale test count | Added tested top-level CPU constraints; CI uses them; updated requirements and instructions |
| Dataset locations | Presets pointed to the original workstation layout | `run_preset.py --dataset` and `--correspondences` accept relocated inputs |
| Matching caches | Stage 2 recipes referenced ignored output directories | Embedded the archived matching settings; `--generate-matches` rebuilds features and pairs under the new output directory instead of reusing stale matches |
| Native selection | A missing extension could fail after expensive preparation | Explicit backend validation now happens before matching; no silent Python fallback |
| Run provenance | Effective overrides and versions were not captured by the recipe runner | Saved effective preset, seed, versions, source revision/dirty state, runner and preset hashes |
| Scene preparation | Splat preparation implicitly required an ignored full frontend trajectory | Fresh frontend by default; optional explicit `--scene-trajectory`; selectable datasets and local preset directory |
| Neural runtime | Mutable version tag in GPU runners | Pinned the observed image by SHA-256 digest; recorded base-image package versions |
| Workstation executables | Personal Python and FFmpeg paths | Runners now use executables on PATH, while retaining explicit overrides |
| GNT setup | Setup required the rejected 2DGS CUDA build | Added `-Stage Setup -GNTOnly`; cached pinned source/weight validation passed |
| 2DGS compiler | CUDA architecture hard-coded to RTX 2080's 7.5 | Compiler setup derives the architecture from the detected GPU |
| GNT camera inputs | Four input paths embedded in Python source | Shared `configs/splatting/gnt_inputs.json`; original paths preserved, replacements explicit |
| Learned loop discovery | DINOv2 Hub source followed an unpinned branch | Added `SFM_DINOV2_REF` for an explicit upstream commit/tag; historical default remains unchanged |
| Stage ordering | Late runners depended on artifacts not apparent from their names | Documented the input/producer order, selected checkpoints and historical-control dependencies |

The CPU constraints are not a full transitive package lock and are not the
neural training environment. The recorded image fixes its base contents, not
later installs into trial directories. Neither pinning nor random seeds imply
bitwise equality across hardware, BLAS versions or GPU kernels.

## Tests actually performed

- Full local suite: **212 passed**, including the installed C++ and Ceres
  components and the two new relocated-input CLI tests.
- [Clean constrained CI](https://github.com/humma-humma/sfm_recon/actions/runs/39049429141)
  passed on Python 3.10 and 3.11: **167 passed, 24 optional tests skipped** per
  environment. Source/pure-wheel builds and CLI import checks passed too.
- [Native wheel CI](https://github.com/humma-humma/sfm_recon/actions/runs/39049429129)
  built, repaired and tested **all six Ceres wheels** across Linux, Windows and
  macOS with Python 3.10/3.11 for the updated source. Optional skips in CPU CI do
  not replace this separate native matrix.
- Fresh milk run: selected native preset, new matching/output directory, seed 0;
  **50 cameras, 1,745 points**. Matching files were byte-identical to the archived
  cache (tree SHA-256 `a80cb18d7ec7b7163628a9c5ef28fa99eadd1ca00c83ac141f3123d772dddf61`).
- That reconstruction call took **9.48 seconds**, excluding matching generation,
  output writing and evaluation. This is a verification sample, not a new runtime
  benchmark or cross-platform accuracy claim. Its observed package versions are
  in [repro-audit-result.json](../configs/environments/repro-audit-result.json).
- A project-local NumPy **2.2.6** / OpenCV **4.12.0.88** overlay repeated that
  milk run: **50 cameras, 1,745 points**, **9.27 seconds** for reconstruction;
  correspondence files were again byte-identical. This verifies the compatible
  version pair on the same workstation, not a clean environment installation.
- Pinned neural image launched in a network-disabled container; package metadata
  matched [neural-runtime.json](../configs/environments/neural-runtime.json).
  A separate driver-only probe saw the RTX 2080 with driver 591.86. No training
  or view rendering was launched for those checks.
- PowerShell runner syntax parsed successfully; the GNT-only setup verified the
  existing pinned upstream commit and official pretrained-checkpoint digest.
  This cached check does not establish that the remote weight URL is always
  available or that every external download is content-addressed.

## Remaining limits

1. The complete scene training/rendering chain has not been rerun from an empty
   workspace. The long route alone previously required about 37 hours of new GNT
   rendering, before all of the upstream training and exports.
2. Historical checkpoints were not redistributed; rejected/intermediate weights
   were pruned during cleanup. Recipes that export comparisons still need their
   control/dense/balanced checkpoints regenerated through the earlier stages.
   The final accepted MP4s can instead be replayed from the versioned release.
3. Original camera/data exports and train/test splits must match the associated
   splat normalization and GNT inputs. Choosing a new geometry recipe is a new
   experiment, not a guaranteed exact reproduction of an old accepted model.
4. The original DINOv2 Hub revision, all learned-model weight hashes and every
   learned dependency were not fully locked. A new explicit Hub ref improves new
   experiments; it cannot retroactively prove the old source revision. Stage 3
   saved-loop recipes reproduce a stored constraint, not automatic loop discovery.
5. Some metadata records the original Windows filesystem paths. Current CPU
   overrides and PATH defaults remove common blockers, but GPU orchestration and
   Chrome QA are still Windows/PowerShell-oriented. A single native Linux host
   orchestration command is not provided.
6. The recorded Linux x86-64 / CUDA 11.8 / PyTorch 2.1 runtime is not a universal
   runtime for every GPU architecture. Upstream repositories, weights, container
   access, host drivers and licenses remain external requirements.

The detailed local retrospective and its evidence bundle remain ignored. Public
stage reports preserve historical measurements and artifact names; an old path
in a report does not mean that the corresponding artifact is shipped in Git.
