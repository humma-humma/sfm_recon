from __future__ import annotations

import argparse
from pathlib import Path

from .dataset import (
    load_image_dataset,
    load_stage1_dataset,
    load_stage2_dataset,
)
from .io import (
    write_camera_parameters,
    write_point_cloud,
    write_rich_point_cloud,
    write_summary,
)
from .matching import MatchingConfig, generate_correspondences
from .reconstruction import ReconstructionConfig, reconstruct
from .reprojection_diagnostics import write_reprojection_diagnostics


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Incremental calibrated Structure-from-Motion."
    )
    parser.add_argument("--stage", type=int, choices=(1, 2), default=1)
    parser.add_argument("--dataset", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--max-images", type=int)
    parser.add_argument(
        "--no-bundle-adjustment",
        action="store_true",
        help="Skip final global bundle adjustment.",
    )
    parser.add_argument("--bundle-adjustment-max-nfev", type=int, default=40)
    parser.add_argument("--min-pnp-points", type=int, default=12)
    parser.add_argument("--min-pnp-inliers", type=int, default=10)
    parser.add_argument("--min-track-observations", type=int, default=2)
    parser.add_argument("--max-point-distance-factor", type=float)
    parser.add_argument("--two-view-max-reprojection-error", type=float)
    parser.add_argument("--two-view-min-triangulation-angle", type=float)
    parser.add_argument("--matching-cache", type=Path)
    parser.add_argument("--max-features", type=int, default=1500)
    parser.add_argument(
        "--feature-mode",
        choices=("sift", "akaze", "sift+akaze"),
        default="sift",
    )
    parser.add_argument("--sift-contrast-threshold", type=float, default=0.03)
    parser.add_argument("--sift-edge-threshold", type=float, default=10.0)
    parser.add_argument("--pair-window", type=int, default=3)
    parser.add_argument("--ratio-threshold", type=float, default=0.75)
    parser.add_argument("--matching-essential-threshold", type=float, default=1.0)
    parser.add_argument("--min-match-inliers", type=int, default=15)
    parser.add_argument("--mask-apriltags", action="store_true")
    parser.add_argument("--apriltag-padding", type=float, default=0.12)
    parser.add_argument("--overwrite-correspondences", action="store_true")
    parser.add_argument("--write-reprojection-diagnostics", action="store_true")
    parser.add_argument("--reprojection-overlay-limit", type=int, default=300)
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.stage == 1:
        dataset = load_stage1_dataset(args.dataset).subset(args.max_images)
    else:
        image_dataset = load_image_dataset(args.dataset).subset(args.max_images)
        cache_root = args.matching_cache or args.output_dir / "matching_cache"
        correspondence_dir, matching_summary = generate_correspondences(
            image_dataset,
            cache_root,
            MatchingConfig(
                max_features=args.max_features,
                feature_mode=args.feature_mode,
                sift_contrast_threshold=args.sift_contrast_threshold,
                sift_edge_threshold=args.sift_edge_threshold,
                pair_window=args.pair_window,
                ratio_threshold=args.ratio_threshold,
                essential_threshold=args.matching_essential_threshold,
                min_inliers=args.min_match_inliers,
                mask_apriltags=args.mask_apriltags,
                apriltag_padding=args.apriltag_padding,
            ),
            overwrite=args.overwrite_correspondences,
        )
        print(
            f"Generated {matching_summary.correspondences} correspondences "
            f"across {matching_summary.accepted_pairs} image pairs."
        )
        dataset = load_stage2_dataset(
            args.dataset, correspondence_dir
        ).subset(args.max_images)

    config = ReconstructionConfig(
        bundle_adjustment=not args.no_bundle_adjustment,
        bundle_adjustment_max_nfev=args.bundle_adjustment_max_nfev,
        min_pnp_points=args.min_pnp_points,
        min_pnp_inliers=args.min_pnp_inliers,
        min_track_observations=args.min_track_observations,
        max_point_distance_factor=args.max_point_distance_factor,
        two_view_max_reprojection_error=args.two_view_max_reprojection_error,
        two_view_min_triangulation_angle=args.two_view_min_triangulation_angle,
    )
    result = reconstruct(dataset, config)

    write_camera_parameters(
        args.output_dir / "estimated_camera_parameters.json", dataset, result
    )
    write_point_cloud(args.output_dir / "estimated_points.ply", result)
    write_rich_point_cloud(
        args.output_dir / "estimated_points_rich.ply", dataset, result
    )
    write_summary(args.output_dir / "summary.json", dataset, result)
    if args.write_reprojection_diagnostics:
        write_reprojection_diagnostics(
            args.output_dir / "reprojection_diagnostics",
            dataset,
            result,
            max_overlay_points=args.reprojection_overlay_limit,
        )

    print(
        f"Registered {len(result.poses)}/{len(dataset.image_paths)} cameras; "
        f"reconstructed {len(result.points)} points."
    )
    print(f"Outputs: {args.output_dir.resolve()}")
