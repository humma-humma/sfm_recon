# Final local demo — 2026-10-16

The main entry point is now **http://localhost:8088/**, served by
`node web_demo/serve.mjs`. It provides four datasets (box, milk, shoe and scene)
with route navigation:

- **Traverse route:** full pre-rendered routes with playback, timeline scrubbing,
  keyboard forward/back navigation and fullscreen. Scene defaults to the accepted
  15fps GNT motion-compensated 30fps video. Corrected splats with reliability
  fallback remain a switchable alternative at the same playback time.

The outside/orbit mode was removed at user request because of its noisy
appearance. The demo no longer loads live splat binaries or an embedded 3D
viewer. Existing `?mode=outside` bookmarks open route navigation instead.
Source checkpoints and archived model assets are preserved.

The five-video portfolio is at `portfolio.html`; detailed scene comparison and
recorded generation times remain at `gnt-scene-15fps-splats.html`. See
[demo setup and controls](../web_demo/README.md) and
[full-scene corrected-splat results](GNT_SCENE15_SPLATS.md).

Browser QA exercises four full routes, scene-render switching, keyboard travel,
responsive layout and absence of the removed orbit setup. Reports are under
`outputs/gnt_finalization/final_demo_browser_qa.json` and
`portfolio_browser_qa.json`. JavaScript syntax and worker sorting also pass.
This supersedes the earlier browser-unverified status below.

The demo is local, not publicly hosted. GNT route navigation is constrained to
the rendered path. This closes the requested local presentation work without claiming
photorealism or collision-safe free walking.

## Historical local release — 2026-10-11

The implementation, controlled research experiment, local demo assets, and
evaluation reports are complete. This is a reproducible local research project,
not a claim of perfect reconstruction or a publicly hosted release.

## Open the demo

From the repository root:

```powershell
node web_demo/serve.mjs
```

Open <http://127.0.0.1:8088>. Box, boot and milk support orbit; all four datasets
support free flight and captured-view tours. The **Research results** link opens
the controlled scene comparison and measured-versus-TSDF depth images.

Free flight is traversable viewing without collisions. The confidence-masked
TSDF is incomplete, especially at the stair entrance, and is not advertised as a
collision-safe walking surface. Original presentation assets remain selected;
the new depth-supervised model is available as a measured research candidate.

## Research contribution

- Adapted confidence-aware depth constraints to the existing Splatfacto pipeline.
- Used only 241 training views and their training depth neighbors for supervision;
  excluded all 35 held-out RGB-D views from the added loss, confidence and TSDF.
  Average supervised image fraction: 29.3%. The inherited seed cloud and poses
  were not rebuilt under a train-only split; this is a continuation ablation,
  not a strictly held-out end-to-end reconstruction benchmark.
- Compared equal 5k additional optimization budgets from the identical fresh 5k
  checkpoint. Cameras, Gaussian initialization and resolution schedule stayed fixed.
- Improved mean held-out PSNR by 0.586 dB, SSIM by 0.00649 and LPIPS by 0.01164
  versus the matched RGB-only 10k control. Worst PSNR regression: 0.383 dB.
- Reduced the median of six diagnostic per-view sensor-depth errors by 21.9%.
  The stair entrance remains almost unchanged; the corridor remains streaky.
- Built an independent train-only TSDF mesh: 680,600 vertices / 1,189,339 triangles.
  Surface error is low where supported, with substantial gaps in weak regions.

See [research results and reproduction](DEPTH_RESEARCH_RESULTS.md),
[diagnosis](SCENE_DIAGNOSIS.md), and [literature review](SPLAT_LITERATURE_REVIEW.md).
The depth loss adapts a research principle; it is not a reproduction of Dense
Depth Priors. TSDF fusion is not a BundleFusion implementation. 2DGS, QuickSplat,
Mip-Splatting and learned completion are not claimed as implemented.

## Verification

- Full local Python suite: **176 passed**.
- Four model assets: **1,705,968** finite Gaussian records; **423** camera exports
  checked against independent COLMAP projection and orthonormality.
- Offscreen GPU comparison: **72/72** shader cases passed. This uses a desktop
  GLSL adaptation; it is not browser compilation/interaction testing.
- Worker sorting, JavaScript syntax and PowerShell runner syntax passed.
- HTTP delivery: 14 core/research/model resources passed; outside-root access denied.
- Training produced finite weights, verified optimizer bindings, 5,000 nonempty
  depth-supervised batches and the intended step-9999 checkpoint.
- All 35 candidate held-out pairs use identical source pixels to the matched
  control. Numeric RGB gates and predeclared geometric gates passed.

Raw results remain in `outputs/depth_research/` and
`outputs/web_renderer_validation/report.json`; generated web comparisons are in
`web_demo/assets/research/`.

## Practical limits

No browser surface was available for browser interaction QA; this remains
unverified, alongside hosted CI execution and public deployment. Missing capture
information and local alignment issues remain visible in the scene. Dataset-
specific experiments and single-run gains do not establish general performance.

The 2 TB project drive filled during checkpoint saving. Only the incomplete new
checkpoint was removed. The experiment reran successfully from the untouched
source and saves final inference weights without Adam state; it cannot resume
optimizer training from that candidate. Approximately 100 MB remained after
evaluation. Free storage before generating more large artifacts; no original data
or historical models were deleted. Existing unrelated containers were left running.
