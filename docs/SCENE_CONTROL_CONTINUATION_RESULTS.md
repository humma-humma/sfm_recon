# Fresh scene control: continuation results

Both continuations completed: 35 held-out views each, starting from the exact
5k fresh control. Gaussian count stayed 882,871; runtime checks verified optimizer
parameter bindings and changed means. Selected models and web assets are unchanged.

| Total steps | Mean PSNR | PSNR delta vs 5k | SSIM delta | LPIPS delta | PSNR-improved views | Worst PSNR delta | Additional training vs 5k |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 5,000 | 16.5612 | — | — | — | — | — | — |
| 10,000 | 16.8896 | +0.3284 | +0.00926 | -0.04277 | 30/35 | -0.8027 | 4.92 min |
| 15,000 | 16.9239 | +0.3627 | +0.00616 | -0.07354 | 26/35 | -1.0918 | 9.75 min |

LPIPS decreases are improvements. Peak allocated CUDA memory was about
4,539–4,542 MiB. These are incremental continuation times, excluding separate
evaluation and the original 5k training. Both candidates fail only the
predeclared worst-view PSNR gate (-0.5 dB); all other checks pass.

## Render review

Reviewed nine held-out views covering worst regressions, largest improvements,
interior and exterior stairs, corridor, pavement and lab signage. Sheets are
`outputs/scene_control_continuation/review/comparison_{0,1}.jpg`.

Outdoor pavement texture becomes more defined, especially in `64.3018440.jpg`
and `81.3685110.jpg`. Some wall boundaries improve. Both models retain cloudy
surfaces and floating artifacts. The internal stairs at `128.301833.jpg` remain
poorly reconstructed; the lab lettering at `145.368511.jpg` remains unreadable.
This continuation does not solve the main geometric/detail failures.

The largest regression for both candidates is corridor view `4.56851100.jpg`.
The 15k result has more conspicuous streaked wall structure there. The modest
additional mean PSNR gain from 10k to 15k (+0.0343 dB) accompanies lower mean
SSIM and more PSNR-regressing views, despite further improvement in LPIPS.

10k is the more conservative candidate for user visual review; 15k remains an
alternative if its perceived texture improvement is preferred. Neither is
automatically promoted. Avoid describing either as universally better or as
successful noise removal. A separate depth-constrained training experiment is
the next proposed test; no such test was run in this continuation.

Full gallery: `outputs/scene_control_continuation/index.html`.
Columns: source / fresh control 5k / continued 10k / continued 15k.
Full gates/runtime: `outputs/scene_control_continuation/comparisons.json`.
