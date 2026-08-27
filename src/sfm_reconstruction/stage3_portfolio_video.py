from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .open3d_viewer import colors_for_mode, load_ascii_ply_table
from .stage3 import load_trajectory
from .stage3_visualize import NamedTrajectory, aligned_trajectory_points


def transform_points(points: np.ndarray, transform: np.ndarray, scale: float) -> np.ndarray:
    """Apply Nerfstudio's saved world-to-dataparser transform."""
    points = np.asarray(points, dtype=np.float64)
    transform = np.asarray(transform, dtype=np.float64)
    return scale * (points @ transform[:, :3].T + transform[:, 3])


def look_at_camera_to_world(
    position: np.ndarray, target: np.ndarray, up: np.ndarray
) -> np.ndarray:
    """Build a Nerfstudio/OpenGL camera-to-world matrix."""
    position = np.asarray(position, dtype=np.float64)
    backward = position - np.asarray(target, dtype=np.float64)
    backward /= np.linalg.norm(backward)
    right = np.cross(np.asarray(up, dtype=np.float64), backward)
    right /= np.linalg.norm(right)
    camera_up = np.cross(backward, right)
    matrix = np.eye(4, dtype=np.float64)
    matrix[:3, :3] = np.column_stack((right, camera_up, backward))
    matrix[:3, 3] = position
    return matrix


def orbit_camera_path(
    points: np.ndarray,
    up: np.ndarray,
    start_direction: np.ndarray,
    frame_count: int,
    *,
    fov_degrees: float = 55.0,
    elevation_degrees: float = 18.0,
) -> list[np.ndarray]:
    if frame_count < 1:
        raise ValueError("frame_count must be positive")
    points = np.asarray(points, dtype=np.float64)
    finite = points[np.isfinite(points).all(axis=1)]
    if len(finite) == 0:
        raise ValueError("points must contain a finite point")
    lower, upper = np.percentile(finite, [2.0, 98.0], axis=0)
    center = (lower + upper) / 2.0
    radius = float(np.max(np.linalg.norm(finite.clip(lower, upper) - center, axis=1)))

    up = np.asarray(up, dtype=np.float64)
    up /= np.linalg.norm(up)
    radial = np.asarray(start_direction, dtype=np.float64)
    radial -= up * np.dot(radial, up)
    radial /= np.linalg.norm(radial)
    tangent = np.cross(up, radial)
    elevation = np.deg2rad(elevation_degrees)
    distance = max(radius * 1.55, 0.5)

    cameras = []
    for frame_index in range(frame_count):
        angle = 2.0 * np.pi * frame_index / frame_count
        horizontal = radial * np.cos(angle) + tangent * np.sin(angle)
        offset = distance * (horizontal * np.cos(elevation) + up * np.sin(elevation))
        cameras.append(look_at_camera_to_world(center + offset, center, up))
    return cameras


def write_nerfstudio_camera_path(
    output: Path,
    cameras: list[np.ndarray],
    *,
    width: int,
    height: int,
    seconds: float,
    fov_degrees: float,
) -> None:
    payload = {
        "render_height": height,
        "render_width": width,
        "camera_type": "perspective",
        "seconds": seconds,
        "camera_path": [
            {"camera_to_world": camera.reshape(-1).tolist(), "fov": fov_degrees}
            for camera in cameras
        ],
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def trajectory_camera_path(
    poses: list,
    transform: np.ndarray,
    scale: float,
    *,
    frame_stride: int = 16,
) -> list[np.ndarray]:
    """Convert sampled OpenCV trajectory poses to normalized Nerfstudio poses."""
    from scipy.spatial.transform import Rotation

    if frame_stride < 1:
        raise ValueError("frame_stride must be positive")
    indices = list(range(0, len(poses), frame_stride))
    if indices[-1] != len(poses) - 1:
        indices.append(len(poses) - 1)
    rotation = np.asarray(transform, dtype=np.float64)[:, :3]
    convention_flip = np.diag([1.0, -1.0, -1.0])
    cameras = []
    for index in indices:
        pose = poses[index]
        camera = np.eye(4, dtype=np.float64)
        camera[:3, :3] = (
            rotation
            @ Rotation.from_quat(pose.quaternion_xyzw).as_matrix()
            @ convention_flip
        )
        camera[:3, 3] = transform_points(
            pose.translation[None, :], transform, scale
        )[0]
        cameras.append(camera)
    return cameras


def read_nerfstudio_camera_path(path: Path) -> tuple[dict, list[np.ndarray]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    cameras = [
        np.asarray(item["camera_to_world"], dtype=np.float64).reshape(4, 4)
        for item in payload["camera_path"]
    ]
    return payload, cameras


def render_point_cloud(
    ply_path: Path,
    camera_path: Path,
    output: Path,
    *,
    transform: np.ndarray,
    scale: float,
    point_size: float = 1.5,
) -> None:
    import cv2
    import open3d as o3d

    payload, cameras = read_nerfstudio_camera_path(camera_path)
    width, height = int(payload["render_width"]), int(payload["render_height"])
    fps = len(cameras) / float(payload["seconds"])
    fov = float(payload["camera_path"][0]["fov"])
    focal = 0.5 * height / np.tan(np.deg2rad(fov) / 2.0)

    table = load_ascii_ply_table(ply_path)
    cloud = o3d.geometry.PointCloud()
    cloud.points = o3d.utility.Vector3dVector(transform_points(table.points, transform, scale))
    cloud.colors = o3d.utility.Vector3dVector(colors_for_mode(table, "rgb"))

    visualizer = o3d.visualization.Visualizer()
    if not visualizer.create_window(width=width, height=height, visible=False):
        raise RuntimeError("Open3D could not create a render window")
    output.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(output), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height)
    )
    if not writer.isOpened():
        visualizer.destroy_window()
        raise RuntimeError(f"Could not open video writer for {output}")
    try:
        visualizer.add_geometry(cloud)
        options = visualizer.get_render_option()
        options.background_color = np.asarray([0.02, 0.02, 0.025])
        options.point_size = point_size
        control = visualizer.get_view_control()
        intrinsic = o3d.camera.PinholeCameraIntrinsic(
            width, height, focal, focal, width / 2.0, height / 2.0
        )
        convention_flip = np.diag([1.0, -1.0, -1.0, 1.0])
        for camera in cameras:
            parameters = o3d.camera.PinholeCameraParameters()
            parameters.intrinsic = intrinsic
            parameters.extrinsic = np.linalg.inv(camera @ convention_flip)
            control.convert_from_pinhole_camera_parameters(parameters, allow_arbitrary=True)
            visualizer.poll_events()
            visualizer.update_renderer()
            image = np.asarray(visualizer.capture_screen_float_buffer(do_render=True))
            frame = (np.clip(image, 0.0, 1.0) * 255.0).astype(np.uint8)
            writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
    finally:
        writer.release()
        visualizer.destroy_window()


def _map_path(
    points: np.ndarray,
    width: int,
    height: int,
    bounds: tuple[np.ndarray, np.ndarray] | None = None,
) -> np.ndarray:
    planar = np.asarray(points, dtype=np.float64)[:, (0, 2)]
    lower, upper = (
        (np.min(planar, axis=0), np.max(planar, axis=0)) if bounds is None else bounds
    )
    span = np.maximum(upper - lower, 1e-9)
    margin = 55.0
    scale = min((width - 2 * margin) / span[0], (height - 2 * margin) / span[1])
    mapped = (planar - (lower + upper) / 2.0) * scale
    mapped[:, 0] += width / 2.0
    mapped[:, 1] = height / 2.0 - mapped[:, 1]
    return np.rint(mapped).astype(np.int32)


def render_trajectory_panel(
    ground_truth: NamedTrajectory,
    estimate: NamedTrajectory,
    output: Path,
    *,
    width: int,
    height: int,
    fps: int,
    seconds: float,
    start_fraction: float,
    end_fraction: float,
) -> None:
    import cv2

    reference = np.asarray([pose.translation for pose in ground_truth.poses])
    _, _, aligned_estimate = aligned_trajectory_points(ground_truth.poses, estimate.poses)
    reference_planar = reference[:, (0, 2)]
    bounds = (np.min(reference_planar, axis=0), np.max(reference_planar, axis=0))
    reference_xy = _map_path(reference, width, height, bounds)
    estimate_xy = _map_path(aligned_estimate, width, height, bounds)
    frame_count = int(round(fps * seconds))
    output.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(output), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height)
    )
    if not writer.isOpened():
        raise RuntimeError(f"Could not open video writer for {output}")
    try:
        for frame_index in range(frame_count):
            progress = (frame_index + 1) / frame_count
            fraction = start_fraction + progress * (end_fraction - start_fraction)
            end = max(2, min(len(estimate_xy), int(round(fraction * len(estimate_xy)))))
            frame = np.full((height, width, 3), (20, 18, 16), dtype=np.uint8)
            cv2.polylines(frame, [reference_xy], False, (92, 92, 92), 2, cv2.LINE_AA)
            cv2.polylines(frame, [estimate_xy], False, (55, 42, 32), 2, cv2.LINE_AA)
            cv2.polylines(frame, [estimate_xy[:end]], False, (245, 170, 55), 4, cv2.LINE_AA)
            cv2.circle(frame, tuple(estimate_xy[0]), 7, (100, 220, 120), -1, cv2.LINE_AA)
            cv2.circle(frame, tuple(estimate_xy[end - 1]), 8, (245, 210, 90), -1, cv2.LINE_AA)
            cv2.putText(frame, "FULL CAMERA TRAJECTORY", (34, 48), cv2.FONT_HERSHEY_SIMPLEX,
                        0.78, (235, 235, 235), 2, cv2.LINE_AA)
            cv2.putText(frame, "reference", (34, height - 44), cv2.FONT_HERSHEY_SIMPLEX,
                        0.55, (120, 120, 120), 1, cv2.LINE_AA)
            cv2.putText(frame, "reconstructed", (170, height - 44), cv2.FONT_HERSHEY_SIMPLEX,
                        0.55, (245, 170, 55), 1, cv2.LINE_AA)
            writer.write(frame)
    finally:
        writer.release()


@dataclass(frozen=True)
class TimelineStage:
    label: str
    left_video: Path
    right_video: Path


@dataclass(frozen=True)
class SingleStage:
    label: str
    video: Path


def _read_frames(path: Path, size: tuple[int, int], count: int) -> list[np.ndarray]:
    import cv2

    capture = cv2.VideoCapture(str(path))
    frames = []
    try:
        while len(frames) < count:
            ok, frame = capture.read()
            if not ok:
                break
            frames.append(cv2.resize(frame, size, interpolation=cv2.INTER_AREA))
    finally:
        capture.release()
    if not frames:
        raise RuntimeError(f"Could not read frames from {path}")
    frames.extend([frames[-1]] * (count - len(frames)))
    return frames


def _decorate(left: np.ndarray, right: np.ndarray, label: str) -> np.ndarray:
    import cv2

    frame = np.hstack((left, right))
    cv2.line(frame, (left.shape[1], 0), (left.shape[1], frame.shape[0]), (230, 230, 230), 2)
    cv2.rectangle(frame, (0, 0), (frame.shape[1], 72), (12, 12, 15), -1)
    cv2.putText(frame, label, (34, 47), cv2.FONT_HERSHEY_SIMPLEX, 0.92,
                (245, 245, 245), 2, cv2.LINE_AA)
    cv2.putText(frame, "FREE-VIEW RECONSTRUCTION", (34, 102), cv2.FONT_HERSHEY_SIMPLEX,
                0.56, (220, 220, 220), 1, cv2.LINE_AA)
    cv2.putText(frame, "PROGRESSIVE CAMERA TRAJECTORY", (left.shape[1] + 34, 102),
                cv2.FONT_HERSHEY_SIMPLEX, 0.56, (220, 220, 220), 1, cv2.LINE_AA)
    return frame


def compose_timeline(
    stages: list[TimelineStage],
    output: Path,
    *,
    panel_width: int = 960,
    panel_height: int = 540,
    fps: int = 24,
    stage_seconds: float = 8.0,
    wipe_seconds: float = 1.25,
) -> None:
    import cv2

    if not stages:
        raise ValueError("at least one stage is required")
    frame_count = int(round(fps * stage_seconds))
    wipe_count = int(round(fps * wipe_seconds))
    stage_frames = []
    for stage in stages:
        left = _read_frames(stage.left_video, (panel_width, panel_height), frame_count)
        right = _read_frames(stage.right_video, (panel_width, panel_height), frame_count)
        stage_frames.append([_decorate(a, b, stage.label) for a, b in zip(left, right)])

    output.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(output), cv2.VideoWriter_fourcc(*"mp4v"), fps, (2 * panel_width, panel_height)
    )
    if not writer.isOpened():
        raise RuntimeError(f"Could not open video writer for {output}")
    try:
        for stage_index, frames in enumerate(stage_frames):
            start = wipe_count if stage_index > 0 else 0
            if stage_index == len(stage_frames) - 1:
                for frame in frames[start:]:
                    writer.write(frame)
                continue
            for frame in frames[start:-wipe_count]:
                writer.write(frame)
            next_frames = stage_frames[stage_index + 1]
            for wipe_index in range(wipe_count):
                old, new = frames[-wipe_count + wipe_index], next_frames[wipe_index]
                boundary = int(round((wipe_index + 1) / wipe_count * old.shape[1]))
                wiped = old.copy()
                wiped[:, :boundary] = new[:, :boundary]
                writer.write(wiped)

    finally:
        writer.release()


def _read_all_frames(path: Path, size: tuple[int, int]) -> list[np.ndarray]:
    import cv2

    capture = cv2.VideoCapture(str(path))
    frames = []
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            frames.append(cv2.resize(frame, size, interpolation=cv2.INTER_AREA))
    finally:
        capture.release()
    if not frames:
        raise RuntimeError(f"Could not read frames from {path}")
    return frames


def _decorate_single(frame: np.ndarray, label: str) -> np.ndarray:
    import cv2

    decorated = frame.copy()
    cv2.rectangle(decorated, (0, 0), (decorated.shape[1], 66), (12, 12, 15), -1)
    cv2.putText(
        decorated,
        label,
        (30, 44),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.82,
        (245, 245, 245),
        2,
        cv2.LINE_AA,
    )
    return decorated


def compose_synced_timeline(
    stages: list[SingleStage],
    output: Path,
    *,
    width: int = 960,
    height: int = 540,
    fps: int = 24,
    stage_seconds: float = 7.0,
    wipe_seconds: float = 1.25,
) -> None:
    """Compose one continuous orbit with frame-synchronous method wipes."""
    import cv2

    if not stages:
        raise ValueError("at least one stage is required")
    stage_frame_count = int(round(fps * stage_seconds))
    wipe_frame_count = int(round(fps * wipe_seconds))
    total_frame_count = stage_frame_count * len(stages)
    sources = [_read_all_frames(stage.video, (width, height)) for stage in stages]

    output.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(output), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height)
    )
    if not writer.isOpened():
        raise RuntimeError(f"Could not open video writer for {output}")
    try:
        for frame_index in range(total_frame_count):
            stage_index = min(frame_index // stage_frame_count, len(stages) - 1)
            source_fraction = frame_index / max(total_frame_count - 1, 1)
            source_index = int(round(source_fraction * (len(sources[stage_index]) - 1)))
            current = sources[stage_index][source_index].copy()
            label = stages[stage_index].label

            local_index = frame_index % stage_frame_count
            wipe_start = stage_frame_count - wipe_frame_count
            if stage_index < len(stages) - 1 and local_index >= wipe_start:
                next_source_index = int(
                    round(source_fraction * (len(sources[stage_index + 1]) - 1))
                )
                following = sources[stage_index + 1][next_source_index]
                wipe_progress = (local_index - wipe_start + 1) / wipe_frame_count
                boundary = int(round(wipe_progress * width))
                current[:, :boundary] = following[:, :boundary]
                if wipe_progress >= 0.5:
                    label = stages[stage_index + 1].label
            writer.write(_decorate_single(current, label))
    finally:
        writer.release()


def _parse_matrix(values: list[float]) -> np.ndarray:
    if len(values) != 12:
        raise argparse.ArgumentTypeError("transform requires 12 numbers")
    return np.asarray(values, dtype=np.float64).reshape(3, 4)


def _parse_stage(value: str) -> TimelineStage:
    parts = value.split("=", 2)
    if len(parts) != 3:
        raise argparse.ArgumentTypeError("stage must be LABEL=LEFT_VIDEO=RIGHT_VIDEO")
    return TimelineStage(parts[0], Path(parts[1]), Path(parts[2]))


def _parse_single_stage(value: str) -> SingleStage:
    parts = value.split("=", 1)
    if len(parts) != 2:
        raise argparse.ArgumentTypeError("stage must be LABEL=VIDEO")
    return SingleStage(parts[0], Path(parts[1]))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build aligned Stage 3 portfolio footage.")
    subparsers = parser.add_subparsers(dest="command", required=True)
    path_parser = subparsers.add_parser("camera-path")
    path_parser.add_argument("--point-cloud", required=True, type=Path)
    path_parser.add_argument("--output", required=True, type=Path)
    path_parser.add_argument("--transform", required=True, nargs=12, type=float)
    path_parser.add_argument("--scale", required=True, type=float)
    path_parser.add_argument("--width", type=int, default=960)
    path_parser.add_argument("--height", type=int, default=540)
    path_parser.add_argument("--fps", type=int, default=24)
    path_parser.add_argument("--seconds", type=float, default=8.0)
    path_parser.add_argument("--fov", type=float, default=55.0)
    trajectory_path_parser = subparsers.add_parser("trajectory-path")
    trajectory_path_parser.add_argument("--trajectory", required=True, type=Path)
    trajectory_path_parser.add_argument("--output", required=True, type=Path)
    trajectory_path_parser.add_argument("--transform", required=True, nargs=12, type=float)
    trajectory_path_parser.add_argument("--scale", required=True, type=float)
    trajectory_path_parser.add_argument("--frame-stride", type=int, default=16)
    trajectory_path_parser.add_argument("--width", type=int, default=960)
    trajectory_path_parser.add_argument("--height", type=int, default=540)
    trajectory_path_parser.add_argument("--seconds", type=float, default=8.0)
    trajectory_path_parser.add_argument("--fov", type=float, default=61.4)
    cloud_parser = subparsers.add_parser("point-cloud")
    cloud_parser.add_argument("--point-cloud", required=True, type=Path)
    cloud_parser.add_argument("--camera-path", required=True, type=Path)
    cloud_parser.add_argument("--output", required=True, type=Path)
    cloud_parser.add_argument("--transform", required=True, nargs=12, type=float)
    cloud_parser.add_argument("--scale", required=True, type=float)
    cloud_parser.add_argument("--point-size", type=float, default=1.5)
    trajectory_parser = subparsers.add_parser("trajectory")
    trajectory_parser.add_argument("--ground-truth", required=True, type=Path)
    trajectory_parser.add_argument("--estimate", required=True, type=Path)
    trajectory_parser.add_argument("--label", required=True)
    trajectory_parser.add_argument("--output", required=True, type=Path)
    trajectory_parser.add_argument("--width", type=int, default=960)
    trajectory_parser.add_argument("--height", type=int, default=540)
    trajectory_parser.add_argument("--fps", type=int, default=24)
    trajectory_parser.add_argument("--seconds", type=float, default=8.0)
    trajectory_parser.add_argument("--start-fraction", required=True, type=float)
    trajectory_parser.add_argument("--end-fraction", required=True, type=float)
    compose_parser = subparsers.add_parser("compose")
    compose_parser.add_argument("--stage", required=True, action="append", type=_parse_stage)
    compose_parser.add_argument("--output", required=True, type=Path)
    compose_parser.add_argument("--stage-seconds", type=float, default=8.0)
    compose_parser.add_argument("--wipe-seconds", type=float, default=1.25)
    single_parser = subparsers.add_parser("compose-single")
    single_parser.add_argument(
        "--stage", required=True, action="append", type=_parse_single_stage
    )
    single_parser.add_argument("--output", required=True, type=Path)
    single_parser.add_argument("--stage-seconds", type=float, default=7.0)
    single_parser.add_argument("--wipe-seconds", type=float, default=1.25)
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    if args.command == "camera-path":
        transform = _parse_matrix(args.transform)
        table = load_ascii_ply_table(args.point_cloud)
        points = transform_points(table.points, transform, args.scale)
        rotation = transform[:, :3]
        up = rotation @ np.array([0.0, -1.0, 0.0])
        front = rotation @ np.array([0.5, 0.0, -0.8])
        cameras = orbit_camera_path(
            points, up, front, int(round(args.fps * args.seconds)), fov_degrees=args.fov
        )
        write_nerfstudio_camera_path(
            args.output, cameras, width=args.width, height=args.height,
            seconds=args.seconds, fov_degrees=args.fov
        )
    elif args.command == "trajectory-path":
        transform = _parse_matrix(args.transform)
        cameras = trajectory_camera_path(
            load_trajectory(args.trajectory),
            transform,
            args.scale,
            frame_stride=args.frame_stride,
        )
        write_nerfstudio_camera_path(
            args.output,
            cameras,
            width=args.width,
            height=args.height,
            seconds=args.seconds,
            fov_degrees=args.fov,
        )
    elif args.command == "point-cloud":
        transform = _parse_matrix(args.transform)
        render_point_cloud(
            args.point_cloud, args.camera_path, args.output,
            transform=transform, scale=args.scale, point_size=args.point_size
        )
    elif args.command == "trajectory":
        render_trajectory_panel(
            NamedTrajectory("Reference", load_trajectory(args.ground_truth)),
            NamedTrajectory(args.label, load_trajectory(args.estimate)),
            args.output,
            width=args.width,
            height=args.height,
            fps=args.fps,
            seconds=args.seconds,
            start_fraction=args.start_fraction,
            end_fraction=args.end_fraction,
        )
    elif args.command == "compose":
        compose_timeline(
            args.stage,
            args.output,
            stage_seconds=args.stage_seconds,
            wipe_seconds=args.wipe_seconds,
        )
    else:
        compose_synced_timeline(
            args.stage,
            args.output,
            stage_seconds=args.stage_seconds,
            wipe_seconds=args.wipe_seconds,
        )


if __name__ == "__main__":
    main()
