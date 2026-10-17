"""Run a saved dataset recipe; optionally reuse a full scene frontend trajectory."""
import argparse
import json
import hashlib
import platform
import importlib.metadata
import subprocess
from pathlib import Path
from time import perf_counter

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--trajectory", type=Path, help="Reuse scene odometry; exact dataset frame timestamps required")
    parser.add_argument("--rich-cloud", action="store_true", help="Export colored points and track quality for splat initialization")
    parser.add_argument("--dataset", type=Path, help="Override the archived dataset path; relative to the working directory")
    parser.add_argument("--correspondences", type=Path, help="Override archived matches; relative to the working directory")
    parser.add_argument("--generate-matches", action="store_true", help="Generate Stage 2 matches using the preset's recorded matching configuration")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)
    recipe = json.loads(args.preset.read_text())
    recipe['dataset'] = str(args.dataset.resolve() if args.dataset else (ROOT/recipe['dataset']).resolve())
    if args.correspondences and args.generate_matches:
        parser.error('--correspondences and --generate-matches are mutually exclusive')
    if (args.correspondences or args.generate_matches) and 'reconstruction' not in recipe:
        parser.error('Correspondence options apply only to image-based presets')
    if args.trajectory and 'reconstruction' in recipe:
        parser.error('--trajectory applies only to scene presets')
    if args.correspondences:
        recipe['correspondences'] = str(args.correspondences.resolve())
    if args.generate_matches and 'matching' not in recipe:
        parser.error('This preset has no recorded matching configuration; use supplied or explicit correspondences')
    if not Path(recipe['dataset']).is_dir():
        parser.error(f"Dataset directory not found: {recipe['dataset']}; pass --dataset")
    from sfm_reconstruction.native_backend import validate_backend
    if 'reconstruction' in recipe:
        validate_backend(recipe['reconstruction'].get('geometry_backend', 'python'))
        validate_backend(recipe['reconstruction'].get('bundle_adjustment_backend', 'python'), allow_ceres=True)
    else:
        validate_backend(recipe['pose_graph'].get('optimizer_backend', 'python'))
    import cv2
    cv2.setRNGSeed(args.seed)
    np.random.seed(args.seed)
    args.output.mkdir(parents=True, exist_ok=True)
    if args.generate_matches:
        from sfm_reconstruction.dataset import load_image_dataset
        from sfm_reconstruction.matching import MatchingConfig, generate_correspondences
        matches, _ = generate_correspondences(load_image_dataset(recipe['dataset']),
            args.output/'matching_cache', MatchingConfig(**recipe['matching']), overwrite=True)
        recipe['correspondences'] = str(matches.resolve())
    elif recipe.get('correspondences'):
        cache = (ROOT/recipe['correspondences']).resolve()
        if not cache.is_dir():
            parser.error(f'Matching cache not found: {cache}; pass --generate-matches or --correspondences')
        recipe['correspondences'] = str(cache)
    (args.output/"preset.json").write_text(json.dumps(recipe, indent=2), encoding="utf-8")
    versions = {}
    for package in ('sfm-reconstruction', 'numpy', 'scipy', 'opencv-python', 'torch'):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            pass
    try:
        revision = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True,
                                           stderr=subprocess.DEVNULL, timeout=5).strip()
        dirty = bool(subprocess.check_output(['git', 'status', '--porcelain'], cwd=ROOT,
                                            stderr=subprocess.DEVNULL, timeout=5).strip())
    except (OSError, subprocess.SubprocessError):
        revision = None
        dirty = None
    (args.output/'environment.json').write_text(json.dumps(dict(
        python=platform.python_version(), platform=platform.platform(), versions=versions,
        seed=args.seed, source_revision=revision, source_dirty=dirty,
        runner_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        preset_sha256=hashlib.sha256(args.preset.read_bytes()).hexdigest(),
        generated_matches=args.generate_matches), indent=2), encoding='utf-8')
    if "reconstruction" in recipe:
        from sfm_reconstruction.dataset import load_dataset
        from sfm_reconstruction.reconstruction import ReconstructionConfig, reconstruct
        from sfm_reconstruction.io import write_camera_parameters, write_point_cloud, write_summary
        cache = recipe.get("correspondences")
        dataset = load_dataset(ROOT/recipe["dataset"], correspondence_dir=ROOT/cache if cache else None)
        import cv2
        cv2.setRNGSeed(args.seed)
        start = perf_counter()
        result = reconstruct(dataset, ReconstructionConfig(**recipe["reconstruction"]))
        seconds = perf_counter()-start
        write_camera_parameters(args.output/"estimated_camera_parameters.json", dataset, result)
        write_point_cloud(args.output/"estimated_points.ply", result)
        write_summary(args.output/"summary.json", dataset, result)
        if args.rich_cloud:
            from sfm_reconstruction.io import write_rich_point_cloud
            write_rich_point_cloud(args.output/"estimated_points_rich.ply", dataset, result)
        metrics = {"seconds":seconds, "cameras":len(result.poses), "points":len(result.points)}
    else:
        import cv2
        from sfm_reconstruction.stage3 import (load_stage3_dataset, Stage3OdometryConfig,
            estimate_rgbd_trajectory, load_trajectory, write_trajectory)
        from sfm_reconstruction.pose_graph import (LoopEdgeCandidate, PoseGraphConfig,
            build_keyframe_pose_graph, optimize_pose_graph, expand_keyframe_optimization)
        from sfm_reconstruction.official_eval import run_official_eval
        dataset = load_stage3_dataset(ROOT/recipe["dataset"])
        cv2.setRNGSeed(args.seed)
        start = perf_counter()
        trajectory = load_trajectory(args.trajectory) if args.trajectory else estimate_rgbd_trajectory(
            dataset, Stage3OdometryConfig(**recipe["odometry"])).poses
        if len(trajectory) != len(dataset.frames) or any(abs(p.timestamp-f.timestamp)>1e-6 for p,f in zip(trajectory,dataset.frames)):
            raise ValueError("Trajectory does not match preset dataset frames")
        edge = recipe["verified_loop"]
        source, target = edge["source_index"], edge["target_index"]
        measured = np.asarray(edge["source_from_target"])
        loop = LoopEdgeCandidate(source, target, trajectory[source].timestamp, trajectory[target].timestamp,
            edge["method"], 500, 492, edge["inliers"], edge["median_reprojection_error"],
            float(np.linalg.norm(measured[:3,3])), float(np.degrees(np.linalg.norm(cv2.Rodrigues(measured[:3,:3])[0]))),
            True, "", source_from_target=measured)
        config = PoseGraphConfig(**recipe["pose_graph"])
        result = optimize_pose_graph(build_keyframe_pose_graph(trajectory, [loop], config), config)
        poses = expand_keyframe_optimization(trajectory, result.poses)
        seconds = perf_counter()-start
        export = args.output/"estimated_camera_trajectory.txt"
        write_trajectory(export, poses)
        evaluation = run_official_eval(dataset.ground_truth_trajectory_path, export, args.output,
            label="preset", evaluator_dir=dataset.root)
        if evaluation.returncode or evaluation.rmse is None:
            raise RuntimeError("Official scene evaluation failed")
        metrics = {"seconds":seconds, "reused_frontend":bool(args.trajectory), "poses":len(poses),
            "official_ate":evaluation.rmse, "solver_converged":result.success, "nfev":result.iterations}
    (args.output/"run.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(json.dumps(metrics))


if __name__ == "__main__":
    main()
