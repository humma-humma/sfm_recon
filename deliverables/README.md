# Deliverables Archive

This directory is for local archives of generated reconstruction outputs. The
archive files are intentionally ignored by Git because they contain large PLY,
diagnostic, and visualization outputs.

Current archive target:

```text
reconstruction_deliverables_2026-06-18.zip
```

Local archive size: `299,763,299` bytes.

SHA-256:

```text
D5C1BBD635358D5FB7B9BFA1B533CCCD039F864E46927BC0F86AB600D9E359EA
```

Archived output directories:

```text
outputs/stage1_box_filter_regression/
outputs/stage2_milk_masked_track3_ba/
outputs/stage2_boot_improved/
outputs/stage2_milk_masked_track4_ba/
outputs/stage2_milk_sift_akaze_probe/
outputs/stage2_boot_sift_akaze_probe/
```

Why these were selected:

- `stage1_box_filter_regression`: verified Stage 1 regression baseline.
- `stage2_milk_masked_track3_ba`: balanced validated Stage 2 milk result.
- `stage2_boot_improved`: conservative boot output.
- `stage2_milk_masked_track4_ba`: high-confidence four-view milk ablation.
- `stage2_milk_sift_akaze_probe`: coverage-focused milk sparse/dense result
  with reprojection diagnostics.
- `stage2_boot_sift_akaze_probe`: coverage-focused boot sparse/dense result
  with reprojection diagnostics.

The repository README and reproducibility scripts should be committed, but the
zip archive itself should usually be stored outside Git or attached separately.
