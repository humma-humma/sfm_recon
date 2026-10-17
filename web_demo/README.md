# Reconstruction Atlas

**[Open the interactive demo](https://humma-humma.github.io/sfm_recon/)** | [Five-video portfolio](https://humma-humma.github.io/sfm_recon/portfolio.html)

Explore box, milk, shoe and the full building route. Click the view and hold W/S
or up/down arrows to travel; Shift moves faster, Space plays/pauses, left/right
arrows step a frame, and dragging scrubs the route. Timeline, playback speed and
fullscreen controls are also available, including on mobile.

Scene defaults to **motion-compensated GNT**: 15 independently rendered views/s,
motion compensated to 30fps. Switch to **improved splats + reliability fallback**
at the same playback time. Objects use the final balanced Gaussian-splat renders.
Object routes last 32 seconds; the scene lasts 146.533 seconds. All five portfolio
videos play at 960 x 540, 30fps.

Navigation follows pre-rendered camera routes; it does not provide arbitrary
camera movement or run reconstruction in the browser. Motion compensation is the
preferred scene result; indoor splat artifacts remain. The rejected outside-orbit
viewer is not part of the demo.

## GitHub Actions deployment

`.github/workflows/demo-pages.yml` builds and deploys GitHub Pages on relevant
pushes to `main`, or through **Actions > Deploy interactive demo > Run workflow**.
The repository uses **Settings > Pages > Source: GitHub Actions**.

The workflow downloads seven original MP4s from the `demo-media-v1` GitHub release,
checks their byte sizes and SHA-256 hashes against `media-manifest.json`, and copies
an explicit list of pages, scripts, posters and comparison data into `dist/demo`.
The large videos stay outside Git history. The published site contains regular
files, with no Windows junctions, model checkpoints, local reports or runtime
training dependencies. GitHub Actions uploads that artifact and deploys it using
the `github-pages` environment. No custom token or secret is required.

Do not replace a release asset in place. For new renders, create a new versioned
release and update the tag, hashes and sizes in `media-manifest.json`.

## Build and view a portable copy

Python 3.10+, GitHub CLI and Node 20+ are sufficient. No npm install, CUDA,
PyTorch or Docker is needed for viewing. From the repository root:

```powershell
gh release download demo-media-v1 --repo humma-humma/sfm_recon --dir dist/media --pattern "*.mp4"
python scripts/build_demo_site.py --assets-dir dist/media
python scripts/verify_demo_site.py dist/demo
node web_demo/serve.mjs dist/demo
```

Open **http://localhost:8088/**. Build into an empty output directory; existing
files are rejected to prevent stale or private files from entering deployments.
If original local media junctions are available, omit `--assets-dir` when building.

The final site includes route navigation, `portfolio.html`, and
`gnt-scene-15fps-splats.html` with native GNT, motion compensation, corrected splats,
reliability fallback, synchronized seeking and measured render times. The detailed
local retrospective and evidence bundle are ignored by Git and excluded from the
site; they remain on the project owner's machine.
