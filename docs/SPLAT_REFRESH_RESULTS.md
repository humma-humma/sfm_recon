# Completed cleanup and splat refresh

All four fresh models were trained and evaluated, closing the missing link from
selected geometry presets through fresh seeds to actual splats. Completion does
not imply universal rendering improvement. The predeclared per-view gate
prevents a better average from concealing a substantial regression.

## Refreshed models

| Dataset | Held-out views | PSNR old → new | SSIM old → new | LPIPS old → new | Worst-view PSNR delta | Presentation decision |
|---|---:|---:|---:|---:|---:|---|
| Boot | 7 | 18.006 → 18.621 | 0.7118 → 0.7448 | 0.3218 → 0.2775 | -1.719 dB | Retain old; new is an experimental alternative |
| Milk | 7 | 17.601 → 17.257 | 0.7141 → 0.7106 | 0.3110 → 0.3461 | -6.817 dB | Retain old |
| Scene | 35 | 16.460 → 17.227 | 0.6663 → 0.6787 | 0.6290 → 0.5919 | -3.151 dB | Retain old frustum-cleaned presentation |
| Box | 6 | — → 21.330 | — → 0.8179 | — → 0.2311 | No historical baseline | Record first box baseline |

Higher PSNR/SSIM and lower LPIPS are better. These use the same held-out images
and pixels as the earlier runs. Boot improves PSNR in 6/7 views; scene in 23/35;
milk in 5/7, but its severe regression dominates the mean. These are single
seed-42 training runs, not estimates of seed-to-seed variability.

The worst boot regression is `00000.jpg` (20.744 → 19.025 dB); milk is
`00226.jpg` (17.942 → 11.125 dB); scene is `51.5018330.jpg`
(18.555 → 15.404 dB). Visual review confirms the milk view becomes washed out,
and the scene staircase view loses detail. Other scene views improve visibly,
including the laboratory signage. No gate was relaxed after seeing these results.

## Cleanup decision: rejected

Relative to the existing frustum-cleaned scene:

| Depth mask | Gaussians kept | Mean PSNR delta | Mean SSIM delta | Mean LPIPS delta | Worst PSNR delta |
|---|---:|---:|---:|---:|---:|
| Strict | 660,118 | -0.214 | -0.01883 | +0.03945 | -1.863 |
| Middle | 697,763 | -0.079 | -0.01337 | +0.03080 | -1.423 |
| Loose | 753,617 | +0.057 | -0.00724 | +0.02062 | -0.994 |

The fresh scene's conservative cleanup reduces 914,526 Gaussians to 795,663,
but changes PSNR by +0.113 dB, SSIM by -0.00740 and LPIPS by +0.02137, with a
worst-view PSNR drop of 2.783 dB. It also fails. Sampled rendered pairs show
damaged surface detail despite some sharpening of remaining edges.

All tested depth masks are rejected; keep the existing frustum-only presentation
cleanup. Opacity coverage is approximately 100% even in visibly poor renders,
so this metric provides little discrimination here. It must not be interpreted
as preserved geometric detail or sufficient scene coverage.

## Runtime and model size

Fresh final training, including trainer setup/evaluation overhead but excluding
the separate probes, external held-out evaluation and cleanup:

| Dataset | Iterations | Training wall time | Final Gaussians | Recorded peak allocated CUDA memory |
|---|---:|---:|---:|---:|
| Boot | 30,000 | 38.49 min | 321,911 | 3,352 MiB |
| Milk | 30,000 | 24.14 min | 292,238 | 3,351 MiB |
| Scene | 5,000 | 4.66 min | 914,526 | 3,895 MiB |
| Box | 30,000 | 23.12 min | 233,512 | 3,350 MiB |

There is no matched historical training-runtime benchmark, so no training speedup
is claimed. Per-view evaluation render timings include first-render effects and
are not a warmed FPS benchmark. The earlier native support-counting speedup is
independent of these training times.

## Final pipeline choices

- Geometry: keep the selected C++/Ceres object presets and selected scene
  trajectory (0.817261 ATE); their geometry metrics remain independently valid.
- Renderer: retain dense gsplat/Splatfacto settings.
- Presentation: existing boot/milk models and existing frustum-cleaned scene;
  first fresh box model available as a baseline.
- Depth cleanup: tested and rejected, not enabled by default.
- Fresh boot/milk/scene models: retained as evaluated alternatives, with their
  exact source geometry and results documented.

[`configs/splatting/selected_models.json`](../configs/splatting/selected_models.json)
records separate geometry, fresh-model and selected-presentation paths. It is a
selection manifest, not an automatic change to existing viewers or published
videos. No old checkpoint, default backend or published asset was overwritten.

## Review and validation

Reviewed the sampled source/render contact sheets for old/new scene, new scene
depth cleanup, new milk, boot and box, plus the old milk and boot worst views and
the old/new scene worst view. This is a sampled still-image review, not exhaustive
novel-view video validation. Box retains background artifacts and is not claimed
to be a clean object-only reconstruction.

`scripts/finalize_splat_refresh.py` validated all four final checkpoints for
expected step/count and finite Gaussian tensors, recomputed the selection gates,
and saved `outputs/splat_refresh_v1/final_selection.json`. GPU workflow status is
`complete`. The earlier full source suite passed 167 tests; finalization changes
only validation/reporting and selection manifests.
