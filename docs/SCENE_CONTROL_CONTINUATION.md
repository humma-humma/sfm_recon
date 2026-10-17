# Fresh-control continuation

The user chose the fresh scene control as the starting point for the next
experiment. Their clarified visual review: it beats experimental settings in
roughly 85% of comparisons, but is mostly equivalent to the selected model,
looking better roughly 10% of the time. The selected presentation is preserved.

```powershell
& 'E:\Others\Project\sfm_reconstruction\scripts\run_scene_control_continuation.ps1'
```

The runner loads the exact 5,000-step fresh-control checkpoint, continues to
10,000 total steps, evaluates, then continues to 15,000 total steps and evaluates
again. The original resolution schedule remains 3,000: training becomes full
input resolution at step 6,000. Splitting/refinement stays stopped at 2,500;
camera optimization remains off. No pruning or new loss is introduced.

Nerfstudio 1.1.5's resumed trainer counts max_num_iterations as additional
iterations. Each segment therefore uses 5,000 additional steps, with final
checkpoint indices 9,999 and 14,999. Gaussian count must remain 882,871, all
Gaussian tensors must remain finite, and means must change after optimization.

The runner rebuilds optimizers after Splatfacto replaces its Gaussian Parameters
while loading the checkpoint, then restores Adam and scheduler state. It checks
parameter identities to prevent optimization of stale tensors. The patch exists
only in each training subprocess; installed Nerfstudio files are not edited.
Image sampling starts a new seeded sequence in each segment; this is not a claim
of bitwise equivalence with uninterrupted 15k training.

Completed segments are reused. Interrupted segments restart from the previous
completed segment. Source config and checkpoint hashes guard against mixing
runs. Timing is reported per additional 5k segment, not as a from-scratch run.

Results: `outputs/scene_control_continuation/index.html`, all 35 held-out views.
Columns: source / fresh control 5k / continued 10k / continued 15k. Quality gates
compare each candidate to the exact fresh control. Runtime and complete metrics
are recorded alongside the gallery. No model is automatically promoted.

Depth supervision is a separate follow-up, not part of this continuation.
Existing scene depth inputs are available in the Stage 3 dataset. Adding depth
loss requires matching training-image timestamps, confirming RGB/depth alignment,
scaling measured depth into Nerfstudio's normalized coordinates, filtering
invalid depth and boundaries, and validating the rendered-depth convention.
It should use only training observations and its own matched continuation control.
