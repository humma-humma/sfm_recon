from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from .stage3 import load_trajectory
from .stage3_visualize import NamedTrajectory, aligned_trajectory_points


def frame_prefix_lengths(point_count: int, frame_count: int) -> np.ndarray:
    if point_count < 1 or frame_count < 1:
        raise ValueError("point_count and frame_count must be positive")
    return np.maximum(
        1,
        np.ceil(np.linspace(1.0 / frame_count, 1.0, frame_count) * point_count).astype(int),
    )


def render_stage3_trajectory_video(
    ground_truth: NamedTrajectory,
    estimates: list[NamedTrajectory],
    output: Path,
    *,
    width: int = 1280,
    height: int = 720,
    fps: int = 24,
    seconds: float = 10.0,
) -> None:
    import cv2
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if width < 1 or height < 1 or fps < 1 or seconds <= 0.0:
        raise ValueError("video dimensions, fps, and duration must be positive")
    if not estimates:
        raise ValueError("at least one estimated trajectory is required")

    reference_points = np.asarray(
        [pose.translation for pose in ground_truth.poses], dtype=np.float64
    )
    aligned = []
    for estimate in estimates:
        matched_reference, matched_estimate, all_estimate = aligned_trajectory_points(
            ground_truth.poses, estimate.poses
        )
        errors = np.linalg.norm(matched_estimate - matched_reference, axis=1)
        aligned.append((estimate.label, all_estimate, errors))

    frame_count = int(round(fps * seconds))
    output = output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    figure, (path_axis, error_axis) = plt.subplots(
        1,
        2,
        figsize=(width / 100.0, height / 100.0),
        dpi=100,
        constrained_layout=True,
    )
    figure.patch.set_facecolor("#101216")
    for axis in (path_axis, error_axis):
        axis.set_facecolor("#171a20")
        axis.tick_params(colors="white")
        axis.xaxis.label.set_color("white")
        axis.yaxis.label.set_color("white")
        axis.title.set_color("white")
        axis.grid(True, alpha=0.18, color="white")
        for spine in axis.spines.values():
            spine.set_color("#777777")

    all_points = np.vstack([reference_points] + [record[1] for record in aligned])
    x_margin = max(0.5, 0.05 * float(np.ptp(all_points[:, 0])))
    z_margin = max(0.5, 0.05 * float(np.ptp(all_points[:, 2])))
    path_axis.set_xlim(all_points[:, 0].min() - x_margin, all_points[:, 0].max() + x_margin)
    path_axis.set_ylim(all_points[:, 2].min() - z_margin, all_points[:, 2].max() + z_margin)
    path_axis.set_aspect("equal", adjustable="box")
    path_axis.set_title("Top-down trajectory (similarity-aligned)")
    path_axis.set_xlabel("X")
    path_axis.set_ylabel("Z")

    colors = ["#4da3ff", "#ff9f43", "#44d17a"]
    reference_line, = path_axis.plot([], [], color="white", linewidth=2.2, label=ground_truth.label)
    path_lines = []
    error_lines = []
    for index, (label, _, errors) in enumerate(aligned):
        color = colors[index % len(colors)]
        path_line, = path_axis.plot([], [], color=color, linewidth=2.0, label=label)
        error_line, = error_axis.plot([], [], color=color, linewidth=1.8, label=label)
        path_lines.append(path_line)
        error_lines.append(error_line)
    path_legend = path_axis.legend(loc="best", facecolor="#20242b", edgecolor="#777777")
    for text in path_legend.get_texts():
        text.set_color("white")

    maximum_error = max(float(np.max(record[2])) for record in aligned)
    error_axis.set_xlim(0, max(len(record[2]) for record in aligned) - 1)
    error_axis.set_ylim(0, maximum_error * 1.05)
    error_axis.set_title("Aligned translation error")
    error_axis.set_xlabel("Frame")
    error_axis.set_ylabel("Error")
    error_legend = error_axis.legend(loc="upper left", facecolor="#20242b", edgecolor="#777777")
    for text in error_legend.get_texts():
        text.set_color("white")
    progress_text = figure.text(0.5, 0.98, "", ha="center", va="top", color="white", fontsize=14)

    reference_prefix = frame_prefix_lengths(len(reference_points), frame_count)
    estimate_prefixes = [frame_prefix_lengths(len(record[1]), frame_count) for record in aligned]
    error_prefixes = [frame_prefix_lengths(len(record[2]), frame_count) for record in aligned]
    writer = cv2.VideoWriter(
        str(output), cv2.VideoWriter_fourcc(*"mp4v"), float(fps), (width, height)
    )
    if not writer.isOpened():
        plt.close(figure)
        raise RuntimeError(f"Could not open video writer for {output}")
    try:
        for frame_index in range(frame_count):
            ref_end = reference_prefix[frame_index]
            reference_line.set_data(reference_points[:ref_end, 0], reference_points[:ref_end, 2])
            for index, (_, points, errors) in enumerate(aligned):
                path_end = estimate_prefixes[index][frame_index]
                error_end = error_prefixes[index][frame_index]
                path_lines[index].set_data(points[:path_end, 0], points[:path_end, 2])
                error_lines[index].set_data(np.arange(error_end), errors[:error_end])
            progress_text.set_text(
                f"Stage 3 RGB-D SLAM — {(frame_index + 1) / frame_count * 100:5.1f}% of sequence"
            )
            figure.canvas.draw()
            rgba = np.asarray(figure.canvas.buffer_rgba())
            writer.write(cv2.cvtColor(rgba[:, :, :3], cv2.COLOR_RGB2BGR))
    finally:
        writer.release()
        plt.close(figure)


def _parse_trajectory(value: str) -> NamedTrajectory:
    if "=" not in value:
        raise argparse.ArgumentTypeError("trajectory must be LABEL=PATH")
    label, path = value.split("=", 1)
    if not label.strip():
        raise argparse.ArgumentTypeError("trajectory label must be non-empty")
    return NamedTrajectory(label.strip(), load_trajectory(Path(path)))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Animate aligned Stage 3 SLAM trajectories.")
    parser.add_argument("--ground-truth", required=True, type=Path)
    parser.add_argument("--trajectory", required=True, action="append", type=_parse_trajectory)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--fps", type=int, default=24)
    parser.add_argument("--seconds", type=float, default=10.0)
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    render_stage3_trajectory_video(
        NamedTrajectory("Ground truth (evaluation only)", load_trajectory(args.ground_truth)),
        args.trajectory,
        args.output,
        width=args.width,
        height=args.height,
        fps=args.fps,
        seconds=args.seconds,
    )
    print(f"Saved Stage 3 trajectory video: {args.output.resolve()}")


if __name__ == "__main__":
    main()
