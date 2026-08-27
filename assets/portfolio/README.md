# Curated portfolio media

Only final presentation exports are tracked. Source renders, comparison drafts,
training frames, and intermediate edits remain local and are excluded by
`.gitignore`.

| File | Content | Duration |
|---|---|---:|
| `object_reconstruction_timeline.mp4` | Milkbox and boot from classical geometry through fixed-pose splats | 21 s |
| `slam_reconstruction_timeline.mp4` | Inside-view seven-stage progression plus speed-matched cleaned-splat loop | 98 s |
| `slam_reconstruction_outside_timeline.mp4` | Exact exterior orbit across every method plus cleaned-splat showcase | 72 s |

Each MP4 has a matching `*_contact_sheet.png` used as the GitHub preview.

## Presentation semantics

- Object panels use left-to-right linear wipes and fixed orientation.
- Inside-view SLAM follows matched chronological source-frame indices. Early
  clouds use the poses recovered by their stage; the learned cloud and all
  splats share the promoted final path exactly.
- Exterior-view SLAM applies identical camera matrices and orbit phase to all
  point clouds and splat checkpoints.
- The final Gaussian loop is a presentation showcase, not an additional pose
  or ATE improvement.

## Provenance

- Stage 1 classical: `outputs/stage1_box_filter_regression`.
- Stage 1 learned augmentation:
  `outputs/stage1_box_fixed_pose_learned_augmentation`.
- Stage 2 promoted milk: `outputs/stage2_milk_masked_track3_ba`.
- Stage 2 learned milk: `outputs/stage2_milk_superpoint_lightglue_filter02`.
- Stage 2 milk/boot splats: 30k-step fixed-pose Splatfacto checkpoints.
- Stage 3 promoted trajectory:
  `outputs/stage3_pose_graph_keyframe_stride8_nfev50` (ATE `3.030521`).
- Learned retrieval: `outputs/stage3_pose_graph_dinov2_akaze_refine4`.
- Stage 3 final presentation: frustum-support cleanup of the strongest global
  5k checkpoint.

The output names document local provenance; `outputs/` and checkpoints are not
distributed with the repository.
