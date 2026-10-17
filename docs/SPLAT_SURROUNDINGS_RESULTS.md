# Surroundings experiment results

All six experiments completed. Selected checkpoints and web assets remain
unchanged. Milk person masking is a modest candidate improvement; the other
changes are not recommended for selection.

| Experiment | Reference | Mean PSNR change | Worst-view PSNR change | Training minutes | Gaussians |
| --- | --- | ---: | ---: | ---: | ---: |
| Scene fresh control | Selected historical scene | +0.101 dB | -1.500 dB | 4.74 | 882,871 |
| Scene faster resolution | Fresh control | -0.461 dB | -6.608 dB | 4.81 | 841,675 |
| Scene faster resolution + splitting to 4,000 | Fresh control | -1.866 dB | -7.137 dB | 5.41 | 1,381,469 |
| Box person mask | Selected box | -0.027 dB | -0.820 dB | 23.56 | 241,388 |
| Boot person mask | Selected boot | -0.263 dB | -0.934 dB | 24.54 | 294,163 |
| Milk person mask | Selected milk | +0.187 dB | -0.292 dB | 23.89 | 306,707 |

Only milk passes the full-image numerical gate. Its SSIM improves by 0.00443
and LPIPS decreases by 0.00702. On the same static evaluation pixels, milk gains
0.300 dB PSNR and 0.00834 SSIM. These are small gains from one training seed,
not evidence that the surroundings or close-up detail are solved.

Box static-region PSNR improves by only 0.069 dB; boot decreases by 0.203 dB.
These static scores use automatic masks, so segmentation errors weaken their
interpretation. Full-image metrics are retained separately.

## Visual findings

Reviewed the largest four detections per object, all seven milk held-out
triptychs, and the two worst held-out views per scene candidate. Review sheets
are under `outputs/splat_surroundings_tests/review`; the complete gallery is
`outputs/splat_surroundings_tests/index.html`.

- Scene faster resolution visibly washes out the outdoor pavement view
  `68.5685110.jpg` and worsens the entrance-floor view `21.6351770.jpg`.
- Extended splitting visibly blurs/washes out the entrance-floor view
  `25.9018330.jpg` and the exterior view `94.1685110.jpg`. It produces about
  56.5% more Gaussians than the fresh control without better quality.
- Box masks have serious false positives: large parts of the table and box are
  excluded in high-angle frames. The report records about 47% exclusion in
  `00017.jpg` and 60% in `00019.jpg`. This candidate should not be promoted.
- Boot masks misclassify substantial parts of the boot as a person, including
  the upper and laces. This is a confounded masking experiment and should not
  be promoted or used to conclude that correct person masking is ineffective.
- The sampled milk masks target the person while preserving the carton and
  table. Held-out renders show modest changes, with persistent background
  artifacts and soft table markers. No dramatic detail recovery is apparent.

## User visual review

The user clarified that the fresh scene control looks better than the
experimental candidates in roughly 85% of comparisons, but is mostly equivalent
to the selected model, looking better roughly 10% of the time. These are visual
judgments, not measured image-count statistics. Separately, held-out PSNR improves in
19/35 views, with a mean gain of 0.101 dB and worst regression of 1.500 dB.
Use the fresh control as the reference for further noise-reduction experiments
at the user's subsequent request to try it; selected checkpoints have not been changed.

## Interpretation and next steps

Earlier full-resolution training alone does not improve this scene at the
tested 5,000-step budget. The tests do not isolate whether learning-rate timing,
camera errors, initialization, or coverage causes the regressions. They also do
not establish that full-resolution training is generally harmful.

Retain selected scene, box and boot models. Keep milk masking as a separately
reviewable alternative pending user render-quality judgment. Correct the
object-mask false positives before rerunning box or boot; increasing mask
confidence alone is not a demonstrated fix. Confidence-weighted RGB-D depth
supervision is a more informative next scene experiment than further broad
schedule sweeps.

Object comparisons use historical selected baselines, rather than fresh
repeated controls. Scene candidates use a fresh control with identical saved
recipe and no post-training cleanup; the historical selected scene has frustum
cleanup. Training timings have no matched historical object runtime controls.
Peak allocated CUDA memory was about 3,896 MiB for scene and 3,373–3,377 MiB for
objects; these are allocator measurements, not total device memory.

Workflow, complete metrics, gates and runtime records:
`outputs/splat_surroundings_tests/{workflow,comparisons}.json`.
