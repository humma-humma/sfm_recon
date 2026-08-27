from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import cv2
import numpy as np


def image_metrics(reference: np.ndarray, rendered: np.ndarray) -> tuple[float, float]:
    if reference.shape != rendered.shape:
        raise ValueError("reference and rendered images must have the same shape")
    reference_float = reference.astype(np.float64)
    rendered_float = rendered.astype(np.float64)
    mse = float(np.mean((reference_float - rendered_float) ** 2))
    psnr = float("inf") if mse == 0.0 else 10.0 * np.log10(255.0**2 / mse)

    c1 = (0.01 * 255.0) ** 2
    c2 = (0.03 * 255.0) ** 2
    mu_reference = cv2.GaussianBlur(reference_float, (11, 11), 1.5)
    mu_rendered = cv2.GaussianBlur(rendered_float, (11, 11), 1.5)
    sigma_reference = cv2.GaussianBlur(reference_float**2, (11, 11), 1.5) - mu_reference**2
    sigma_rendered = cv2.GaussianBlur(rendered_float**2, (11, 11), 1.5) - mu_rendered**2
    covariance = (
        cv2.GaussianBlur(reference_float * rendered_float, (11, 11), 1.5)
        - mu_reference * mu_rendered
    )
    ssim_map = (
        (2.0 * mu_reference * mu_rendered + c1) * (2.0 * covariance + c2)
    ) / (
        (mu_reference**2 + mu_rendered**2 + c1)
        * (sigma_reference + sigma_rendered + c2)
    )
    return float(psnr), float(np.mean(ssim_map))


def _qvec_to_rotation(qvec: np.ndarray) -> np.ndarray:
    qw, qx, qy, qz = qvec
    return np.asarray(
        [
            [1 - 2 * (qy * qy + qz * qz), 2 * (qx * qy - qw * qz), 2 * (qx * qz + qw * qy)],
            [2 * (qx * qy + qw * qz), 1 - 2 * (qx * qx + qz * qz), 2 * (qy * qz - qw * qx)],
            [2 * (qx * qz - qw * qy), 2 * (qy * qz + qw * qx), 1 - 2 * (qx * qx + qy * qy)],
        ],
        dtype=np.float64,
    )


def pose_increments(
    rotations: list[np.ndarray], translations: list[np.ndarray]
) -> tuple[np.ndarray, np.ndarray]:
    if len(rotations) != len(translations):
        raise ValueError("rotation and translation counts must match")
    translation_steps = np.zeros(len(rotations), dtype=np.float64)
    rotation_steps = np.zeros(len(rotations), dtype=np.float64)
    centers = [-rotation.T @ translation for rotation, translation in zip(rotations, translations)]
    for index in range(1, len(rotations)):
        translation_steps[index] = np.linalg.norm(centers[index] - centers[index - 1])
        relative = rotations[index] @ rotations[index - 1].T
        cosine = np.clip((np.trace(relative) - 1.0) / 2.0, -1.0, 1.0)
        rotation_steps[index] = np.degrees(np.arccos(cosine))
    return translation_steps, rotation_steps


def projected_point_support(
    points: np.ndarray,
    rotation: np.ndarray,
    translation: np.ndarray,
    intrinsics: tuple[float, float, float, float, int, int],
) -> tuple[int, float]:
    fx, fy, cx, cy, width, height = intrinsics
    camera_points = points @ rotation.T + translation
    depth = camera_points[:, 2]
    positive = depth > 1e-6
    x = np.full(len(points), -1.0, dtype=np.float64)
    y = np.full(len(points), -1.0, dtype=np.float64)
    x[positive] = fx * camera_points[positive, 0] / depth[positive] + cx
    y[positive] = fy * camera_points[positive, 1] / depth[positive] + cy
    visible = positive & (x >= 0.0) & (x < width) & (y >= 0.0) & (y < height)
    count = int(np.count_nonzero(visible))
    median_depth = float(np.median(depth[visible])) if count else float("nan")
    return count, median_depth


def _read_colmap(colmap_dir: Path) -> tuple[
    tuple[float, float, float, float, int, int],
    list[tuple[str, np.ndarray, np.ndarray]],
    np.ndarray,
]:
    camera_lines = [
        line.strip()
        for line in (colmap_dir / "cameras.txt").read_text(encoding="ascii").splitlines()
        if line.strip() and not line.startswith("#")
    ]
    if len(camera_lines) != 1:
        raise ValueError("expected exactly one COLMAP camera")
    camera = camera_lines[0].split()
    if camera[1] != "PINHOLE":
        raise ValueError("expected a PINHOLE COLMAP camera")
    width, height = int(camera[2]), int(camera[3])
    fx, fy, cx, cy = map(float, camera[4:8])

    poses: list[tuple[str, np.ndarray, np.ndarray]] = []
    for line in (colmap_dir / "images.txt").read_text(encoding="ascii").splitlines():
        fields = line.split()
        if not fields or line.startswith("#") or len(fields) < 10:
            continue
        rotation = _qvec_to_rotation(np.asarray(fields[1:5], dtype=np.float64))
        translation = np.asarray(fields[5:8], dtype=np.float64)
        poses.append((fields[9], rotation, translation))

    point_rows = []
    for line in (colmap_dir / "points3D.txt").read_text(encoding="ascii").splitlines():
        fields = line.split()
        if fields and not line.startswith("#"):
            point_rows.append([float(value) for value in fields[1:4]])
    return (fx, fy, cx, cy, width, height), poses, np.asarray(point_rows, dtype=np.float64)


def _image_map(root: Path) -> dict[str, Path]:
    paths = list(root.rglob("*.jpg")) + list(root.rglob("*.png"))
    result: dict[str, Path] = {}
    for path in paths:
        if path.name in result:
            raise ValueError(f"duplicate image name under {root}: {path.name}")
        result[path.name] = path
    return result


def _read_image(path: Path) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"failed to read image: {path}")
    return image


def _write_contact_sheet(
    output: Path,
    rows: list[dict[str, object]],
    source_map: dict[str, Path],
    render_1k_map: dict[str, Path],
    render_5k_map: dict[str, Path],
    title: str,
) -> None:
    panels = []
    for row in rows:
        name = str(row["image_name"])
        images = [_read_image(mapping[name]) for mapping in (source_map, render_1k_map, render_5k_map)]
        combined = np.hstack(images)
        label = (
            f"t={float(row['timestamp']):.3f}s  delta PSNR={float(row['delta_psnr']):+.2f} dB  "
            f"1k={float(row['psnr_1k']):.2f}  5k={float(row['psnr_5k']):.2f}"
        )
        cv2.putText(combined, label, (12, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 0), 4, cv2.LINE_AA)
        cv2.putText(combined, label, (12, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 1, cv2.LINE_AA)
        panels.append(combined)
    sheet = np.vstack(panels)
    header = np.zeros((60, sheet.shape[1], 3), dtype=np.uint8)
    cv2.putText(header, f"{title} | source / 1k fixed / 5k fixed", (12, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2, cv2.LINE_AA)
    if not cv2.imwrite(str(output), np.vstack([header, sheet])):
        raise RuntimeError(f"failed to write contact sheet: {output}")


def run_diagnostics(
    source_dir: Path,
    render_1k_dir: Path,
    render_5k_dir: Path,
    colmap_dir: Path,
    output_dir: Path,
) -> dict[str, object]:
    import matplotlib.pyplot as plt

    output_dir.mkdir(parents=True, exist_ok=True)
    source_map = _image_map(source_dir)
    render_1k_map = _image_map(render_1k_dir)
    render_5k_map = _image_map(render_5k_dir)
    intrinsics, poses, points = _read_colmap(colmap_dir)
    rotations = [pose[1] for pose in poses]
    translations = [pose[2] for pose in poses]
    translation_steps, rotation_steps = pose_increments(rotations, translations)

    rows: list[dict[str, object]] = []
    for index, (name, rotation, translation) in enumerate(poses):
        for mapping, label in ((source_map, "source"), (render_1k_map, "1k"), (render_5k_map, "5k")):
            if name not in mapping:
                raise ValueError(f"missing {label} image: {name}")
        source = _read_image(source_map[name])
        render_1k = _read_image(render_1k_map[name])
        render_5k = _read_image(render_5k_map[name])
        psnr_1k, ssim_1k = image_metrics(source, render_1k)
        psnr_5k, ssim_5k = image_metrics(source, render_5k)
        grayscale = cv2.cvtColor(source, cv2.COLOR_BGR2GRAY)
        support, median_depth = projected_point_support(points, rotation, translation, intrinsics)
        rows.append(
            {
                "frame_index": index,
                "timestamp": float(Path(name).stem),
                "image_name": name,
                "psnr_1k": psnr_1k,
                "psnr_5k": psnr_5k,
                "delta_psnr": psnr_5k - psnr_1k,
                "ssim_1k": ssim_1k,
                "ssim_5k": ssim_5k,
                "delta_ssim": ssim_5k - ssim_1k,
                "source_luminance": float(np.mean(grayscale)),
                "source_sharpness": float(cv2.Laplacian(grayscale, cv2.CV_64F).var()),
                "translation_step": float(translation_steps[index]),
                "rotation_step_deg": float(rotation_steps[index]),
                "visible_seed_points": support,
                "median_seed_depth": median_depth,
            }
        )

    csv_path = output_dir / "temporal_metrics.csv"
    with csv_path.open("w", encoding="ascii", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    timestamps = np.asarray([float(row["timestamp"]) for row in rows])
    delta_psnr = np.asarray([float(row["delta_psnr"]) for row in rows])
    figure, axes = plt.subplots(4, 1, figsize=(16, 12), sharex=True, constrained_layout=True)
    axes[0].plot(timestamps, [row["psnr_1k"] for row in rows], label="1k fixed")
    axes[0].plot(timestamps, [row["psnr_5k"] for row in rows], label="5k fixed")
    axes[0].set_ylabel("PSNR (dB)")
    axes[0].legend()
    axes[1].axhline(0.0, color="black", linewidth=0.8)
    axes[1].plot(timestamps, delta_psnr, color="tab:purple")
    axes[1].set_ylabel("5k - 1k PSNR")
    axes[2].plot(timestamps, [row["translation_step"] for row in rows], label="translation")
    axes[2].plot(timestamps, np.asarray([row["rotation_step_deg"] for row in rows]) / 10.0, label="rotation deg / 10")
    axes[2].set_ylabel("Pose increment")
    axes[2].legend()
    axes[3].plot(timestamps, [row["source_sharpness"] for row in rows], label="sharpness")
    axes[3].plot(timestamps, np.asarray([row["visible_seed_points"] for row in rows]) / 100.0, label="visible seeds / 100")
    axes[3].set_ylabel("Input/support")
    axes[3].set_xlabel("Sequence timestamp (s)")
    axes[3].legend()
    for axis in axes:
        axis.grid(True, alpha=0.25)
    plot_path = output_dir / "temporal_diagnostics.png"
    figure.savefig(plot_path, dpi=170)
    plt.close(figure)

    ordered = sorted(rows, key=lambda row: float(row["delta_psnr"]))
    _write_contact_sheet(output_dir / "largest_5k_losses.png", ordered[:6], source_map, render_1k_map, render_5k_map, "Largest 5k losses")
    _write_contact_sheet(output_dir / "largest_5k_gains.png", ordered[-6:][::-1], source_map, render_1k_map, render_5k_map, "Largest 5k gains")

    summary = {
        "frames": len(rows),
        "mean_psnr_1k": float(np.mean([row["psnr_1k"] for row in rows])),
        "mean_psnr_5k": float(np.mean([row["psnr_5k"] for row in rows])),
        "mean_delta_psnr": float(np.mean(delta_psnr)),
        "frames_5k_better": int(np.count_nonzero(delta_psnr > 0.0)),
        "frames_1k_better": int(np.count_nonzero(delta_psnr < 0.0)),
        "largest_losses": [{key: row[key] for key in ("frame_index", "timestamp", "image_name", "delta_psnr")} for row in ordered[:10]],
        "largest_gains": [{key: row[key] for key in ("frame_index", "timestamp", "image_name", "delta_psnr")} for row in ordered[-10:][::-1]],
        "metrics_csv": str(csv_path),
        "plot": str(plot_path),
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Compare Stage 3 splat quality over sequence time.")
    parser.add_argument("--source-dir", required=True, type=Path)
    parser.add_argument("--render-1k-dir", required=True, type=Path)
    parser.add_argument("--render-5k-dir", required=True, type=Path)
    parser.add_argument("--colmap-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    summary = run_diagnostics(args.source_dir, args.render_1k_dir, args.render_5k_dir, args.colmap_dir, args.output_dir)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
