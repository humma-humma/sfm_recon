# Surroundings experiments

Run from a regular PowerShell window with working Docker/GPU access:

```powershell
& 'E:\Others\Project\sfm_reconstruction\scripts\run_splat_surroundings_tests.ps1'
```

The runner preserves selected checkpoints and trains sequentially in
`outputs/splat_surroundings_tests`. Completed cases are reused on rerun;
an interrupted training case starts again. No candidate is automatically selected.

| Case | Steps | Change |
| --- | ---: | --- |
| scene_control | 5,000 | Saved scene training recipe, fresh training |
| scene_fast_resolution | 5,000 | Resolution schedule 1,000 instead of 3,000 |
| scene_fast_resolution_split4000 | 5,000 | Faster resolution schedule plus splitting until step 4,000 |
| box/boot/milk_person_mask | 30,000 each | Exclude detected people from training loss; skip datasets without detections |

Scene controls and candidates receive no post-training pruning. The selected
historical scene has frustum cleanup, so compare experimental scene changes
primarily against the fresh control. Historical object comparisons are not
fresh repeated controls and cannot measure training variance.

Masks use official pretrained Torchvision LRASPP MobileNetV3 weights (about
12.5 MB), downloaded into the existing mounted cache if needed. They exclude
the person class at probability 0.6 with a seven-pixel dilation. This tests
people removal, not arbitrary moving objects or missing background geometry.
Inspect the red mask overlays for false detections before accepting results.
If weight download fails, scene experiments still finish and the masking
failure is recorded separately.

The gallery `outputs/splat_surroundings_tests/index.html` contains every
held-out source/selected/candidate comparison. `comparisons.json` records
PSNR, SSIM, LPIPS, worst-view regression gates, training time, peak allocated
CUDA memory, and Gaussian count. Object tests additionally compare PSNR and
SSIM on identical static regions, since full-image scores penalize removing
people that remain visible in reference photographs. These static scores use
saved eight-bit renders and do not include LPIPS. Logs and workflow status
remain in the same output directory.

These are training experiments, considerably longer than the earlier render
comparisons. Improved held-out scores still require visual review for boot
detail and scene surroundings; training cannot recover surfaces absent from
the photographs.
