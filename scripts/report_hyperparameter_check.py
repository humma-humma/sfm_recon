"""Build a reviewable report and CSVs from checkpointed sweep results."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
from statistics import median

ROOT = Path(__file__).resolve().parents[1]


def plot_comparisons(root):
    """Keep coverage/error tradeoffs visible alongside the full numerical tables."""
    os.environ.setdefault("MPLCONFIGDIR", str(root/"matplotlib_cache"))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(11, 8), constrained_layout=True)
    for axis, name in zip(axes.flat, ("box", "boot", "milk", "scene_graph")):
        path = root/name/"results.json"
        if not path.exists():
            axis.set_visible(False)
            continue
        data = json.loads(path.read_text())
        runs = [r for r in data["runs"] if r.get("status") == "ok"
                and "repeat" not in r["label"] and "seed1" not in r["label"]]
        selected = data.get("selected", data.get("selected_python"))
        for r in runs:
            if name == "milk":
                x, y = 100*r["mesh_coverage"], r["mesh_proxy"]["cropped_mean_nearest_vertex_distance"]
            elif name == "box":
                x, y = r["seconds"], r["point_metrics"]["chamfer_distance"]
            elif name == "boot":
                x, y = r["points"], r["reprojection_rmse_px"]
            else:
                if r["label"] == "raw_no_loop" or r.get("input_family"):
                    continue
                x, y = r["seconds"], r["official_ate"]
            is_selected = r["label"] == selected
            is_baseline = r["label"] == ("python_baseline" if name == "scene_graph" else "baseline")
            color = "#188453" if is_selected else "#c45328" if is_baseline else "#4775a3"
            axis.scatter(x, y, c=color, marker="*" if is_selected else "o",
                s=150 if is_selected else 65 if is_baseline else 25,
                alpha=1 if is_selected or is_baseline else .55, zorder=3 if is_selected else 2)
        labels = {"box": ("Reconstruction time (s)", "GT Chamfer distance"),
            "boot": ("Reconstructed points", "Reprojection RMSE (px)"),
            "milk": ("GT mesh-vertex coverage (%)", "Cropped nearest-vertex distance"),
            "scene_graph": ("Solver time (s)", "Official full-trajectory ATE")}
        axis.set(xlabel=labels[name][0], ylabel=labels[name][1], title=name.replace("_", " ").title())
        axis.grid(alpha=.2)
    fig.suptitle("Exploratory sweep: green star = selected; orange = baseline\nLower error is better; boot measures consistency only")
    target = root/"comparison.png"
    fig.savefig(target, dpi=160)
    plt.close(fig)
    return target


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=ROOT/"outputs/hyperparameter_check")
    parser.add_argument("--report", type=Path, default=ROOT/"docs/HYPERPARAMETER_CHECK.md")
    args = parser.parse_args()
    lines = ["# Reconstruction hyperparameter check - 2026-10-09\n",
        "The check covers Stage 1 box, Stage 2 boot and milk, and the Stage 3 scene. Every field in the reconstruction, matching, odometry and pose-graph configuration classes is audited in `outputs/hyperparameter_check/parameter_audit.json`. The experiment is bounded screening plus interaction checks, not an exhaustive Cartesian search over continuous parameters. Ground truth is used only for evaluation and selection. These are dataset-specific exploratory choices, not held-out validation or automatically promoted defaults.\n",
        "Object sweeps rerun complete reconstruction on cached correspondence files. Single-run timings exclude matching and evaluation. Geometry uses the previously validated C++ kernels; BA uses Ceres except the explicit Python control. Camera count, point retention and image/mesh coverage guards prevent a sparse or partially registered result from winning on a smaller residual alone. Timing repeats use OpenCV seed 0; baseline and selected candidates are also checked with seed 1.\n",
        "Boot has no full camera or mesh ground truth: its supplied `poses.json` contains only the origin pose. Its ranking measures internal reprojection consistency and supported-image coverage, not absolute camera/surface accuracy. The nearby estimated-camera JSON is not treated as ground truth.\n"]
    total = 0
    for name in ("box", "boot", "milk"):
        path = args.input/name/"results.json"
        if not path.exists():
            continue
        data = json.loads(path.read_text())
        runs = data["runs"]
        total += len(runs)
        good = [r for r in runs if r["status"] == "ok"]
        selected = next((r for r in runs if r["label"] == data.get("selected")), None)
        lines += [f"## {name.title()}\n", f"Recorded {len(runs)} runs: {len(good)} successful, {len(runs)-len(good)} failed.\n"]
        if selected:
            repeats = [selected] + [r for r in good if r["label"] in ("selected_repeat1", "selected_repeat2")]
            baseline_repeats = [r for r in good if r["label"] in
                ("baseline", "baseline_repeat1", "baseline_repeat2")]
            lines += [f"Screen-selected candidate: `{selected['label']}`. Median complete reconstruction time across its initial call and repeats: {median(r['seconds'] for r in repeats):.3f} seconds. Selection rule: {data['selection_rule']}\n",
                "```json\n"+json.dumps(selected["config"], indent=2)+"\n```\n"]
            lines.append(f"Baseline median reconstruction time ({len(baseline_repeats)} calls): {median(r['seconds'] for r in baseline_repeats):.3f} seconds. Candidate median uses {len(repeats)} calls. Matching and evaluation are excluded for both.\n")
            lines.append(f"Measured timing ranges: baseline {min(r['seconds'] for r in baseline_repeats):.3f}–{max(r['seconds'] for r in baseline_repeats):.3f} s; "
                f"selected {min(r['seconds'] for r in repeats):.3f}–{max(r['seconds'] for r in repeats):.3f} s. "
                "Calls ran sequentially on the same machine; differences are not a statistical speed guarantee.\n")
            baseline = next(r for r in good if r["label"] == "baseline")
            if name == "box":
                lines.append("The Stage 1 Chamfer metric is the sum of two directed means. "
                    f"GT-to-estimate changes from {baseline['point_metrics']['mean_ground_truth_to_estimate']:.6f} "
                    f"to {selected['point_metrics']['mean_ground_truth_to_estimate']:.6f}; "
                    f"estimate-to-GT changes from {baseline['point_metrics']['mean_estimate_to_ground_truth']:.6f} "
                    f"to {selected['point_metrics']['mean_estimate_to_ground_truth']:.6f}. "
                    "The selected candidate is a tradeoff, not an improvement in every metric.\n")
            if name == "milk":
                density = next((r for r in good if r["label"] == "min_track_observations_2"), None)
                if density:
                    lines.append(f"Coverage alternative: `min_track_observations_2` retains {density['points']} points, "
                        f"reaches {100*density['mesh_coverage']:.2f}% mesh-vertex coverage, and has cropped distance "
                        f"{density['mesh_proxy']['cropped_mean_nearest_vertex_distance']:.6f}. "
                        "It trades some distance-proxy accuracy against greater vertex coverage compared with the selected candidate.\n")
        lines += ["| Case | Cameras | Points | Translation | Rotation (deg) | Reprojection RMSE (px) | Geometry metric | Coverage | Seconds |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
        csv_rows = []
        for r in runs:
            if r["status"] != "ok":
                lines.append(f"| {r['label']} | FAILED | | | | | {r.get('error','')} | | |")
                continue
            pose = r.get("pose_metrics") or {}
            geometry = r.get("point_metrics", {}).get("chamfer_distance", r.get("mesh_proxy", {}).get("cropped_mean_nearest_vertex_distance"))
            coverage = r.get("mesh_coverage", r["image_coverage"])
            fmt = lambda v: "—" if v is None else f"{v:.6f}"
            lines.append(f"| {r['label']} | {r['cameras']} | {r['points']} | {fmt(pose.get('mean_translation'))} | {fmt(pose.get('mean_rotation_degrees'))} | {r['reprojection_rmse_px']:.4f} | {fmt(geometry)} | {coverage*100:.2f}% | {r['seconds']:.3f} |")
            csv_rows.append({"case": r["label"], "cameras": r["cameras"], "points": r["points"],
                "seconds": r["seconds"], "translation": pose.get("mean_translation"),
                "rotation_degrees": pose.get("mean_rotation_degrees"), "geometry_metric": geometry,
                "coverage": coverage, "reprojection_rmse_px": r["reprojection_rmse_px"],
                "config": json.dumps(r["config"], sort_keys=True)})
        if csv_rows:
            with (args.input/name/"comparison.csv").open("w", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=list(csv_rows[0]))
                writer.writeheader(); writer.writerows(csv_rows)
        lines += ["\nGeometry metric is Stage 1 Chamfer or milk's cropped nearest-mesh-vertex proxy. Coverage is milk's fraction of GT mesh vertices within 0.05 units of an estimated point; otherwise it is the mean occupied fraction of 8x8 image bins supported by reconstructed tracks. Image coverage is not surface coverage. Changing cached frontend families also changes the observation set.\n"]
    for family in ("scene_graph", "scene_frontend"):
        path = args.input/family/"results.json"
        if not path.exists():
            continue
        data = json.loads(path.read_text()); total += len(data["runs"])
        if family == "scene_frontend":
            total += int("full_baseline" in data)
            total += int("full_validation" in data and
                data["full_validation"]["label"] != data.get("full_baseline", {}).get("label"))
        lines += [f"## {family.replace('_',' ').title()}\n", data["note"]+"\n"]
        if family == "scene_graph":
            lines += ["| Case | Input | Backend | Nodes | NFEV | Official ATE | Seconds | Converged |", "|---|---|---|---:|---:|---:|---:|---:|---|"]
            for r in data["runs"]:
                if r["status"] == "ok":
                    lines.append(f"| {r['label']} | {r.get('input_family','fixed raw')} | {r.get('config',{}).get('optimizer_backend','raw')} | {r.get('nodes','—')} | {r.get('nfev','—')} | {r['official_ate']:.6f} | {r['seconds']:.3f} | {r.get('converged','—')} |")
                else:
                    lines.append(f"| {r['label']} | FAILED | | | | {r.get('error','')} | | |")
            lines.append(f"\nSelected full-trajectory Python candidate: `{data.get('selected_python','pending')}`. Native screening is not used as proof of Python convergence behavior. Official evaluator logs and full exported trajectories are retained per run.\n")
            if data.get("combined_frontend_graph"):
                lines.append(f"Combined frontend/graph check: `{data['combined_frontend_graph']}`. {data['combined_note']}\n")
            python_runs = [r for r in data["runs"] if r.get("status") == "ok"
                and r.get("config", {}).get("optimizer_backend") == "python"]
            if python_runs:
                overall = min(python_runs, key=lambda r: r["official_ate"])
                lines.append(f"Lowest measured scene ATE across the Python graph input families: "
                    f"`{overall['label']}`, ATE {overall['official_ate']:.6f}. "
                    "This comparison includes the combined frontend checks and is distinct from the fixed-input graph selection.\n")
                lines.append("Selected graph recipe:\n\n```json\n"+json.dumps(overall["config"], indent=2)+"\n```\n")
                frontend_path = args.input/"scene_frontend/results.json"
                if frontend_path.exists():
                    front = json.loads(frontend_path.read_text())
                    if "full_baseline" in front and "full_validation" in front:
                        source = front["full_validation"] if overall.get("input_family") else front["full_baseline"]
                        graph_baseline = next(r for r in python_runs if r["label"] == "python_baseline")
                        baseline_seconds = front["full_baseline"]["seconds"]+graph_baseline["seconds"]
                        selected_seconds = source["seconds"]+overall["seconds"]
                        lines.append(f"Measured frontend + graph-solve time: baseline {baseline_seconds:.3f} s, "
                            f"selected {selected_seconds:.3f} s. Frontend tracked transitions: "
                            f"baseline {front['full_baseline']['tracked_frames']}, selected {source['tracked_frames']}, "
                            f"out of {source['poses']-1}. These are single full frontend/solver calls; graph construction, "
                            "trajectory expansion, output writing and evaluation are excluded.\n")
        else:
            lines += ["| Case | Tracked fraction | Translation RPE RMSE | Rotation RPE RMSE (deg) |", "|---|---:|---:|---:|"]
            for r in data["runs"]:
                lines.append(f"| {r['label']} | {r['tracked_fraction']:.4f} | {r['translation_rpe_rmse']:.6f} | {r['rotation_rpe_rmse_degrees']:.6f} |")
            for key in ("full_baseline", "full_validation"):
                lines.append(f"\n{key.replace('_', ' ').title()}:\n\n```json\n"+json.dumps(data.get(key, {"status":"pending"}), indent=2)+"\n```\n")
            if "full_baseline" in data and "full_validation" in data:
                baseline_ate = data["full_baseline"]["official_ate"]
                selected_ate = data["full_validation"]["official_ate"]
                lines.append(f"Full-trajectory frontend ATE change: {(selected_ate/baseline_ate-1)*100:+.2f}%. Window selection is accepted as a full-sequence improvement only when this value is negative.\n")
    lines += [f"## Scope and limitations\n\nTotal recorded configurations/calls so far: {total}. Physical camera/depth calibration is fixed. Numerical identity tolerances and minimum data-sufficiency guards are not freely optimized. Matching is checked through existing cached frontend families rather than independent re-extraction for every possible feature/matching setting. Scene loop discovery/retrieval thresholds are not inferred from a fixed graph: the graph sweep reuses its verified loop. The parameter audit records these distinctions explicitly.\n",
        "Ceres is checked for object bundle adjustment. The current Stage 3 pose graph has Python and C++ backends; it has no Ceres backend.\n",
        "Dense fusion, Gaussian training, viewer and rendering parameters are outside this SfM/SLAM geometry check; they require distinct surface or photometric validation. Existing promoted outputs and defaults are preserved.\n",
        "## Reproduction\n\n```powershell\n$env:PYTHONPATH = 'src'\npython scripts/sweep_reconstruction.py\npython scripts/sweep_scene.py\npython scripts/sweep_scene_frontend.py\npython scripts/sweep_scene.py --confirm-frontend\npython scripts/report_hyperparameter_check.py\n```\n\nRun phases sequentially to avoid timing contention. Each sweep checkpoints individual runs and resumes using its output directory. Full JSON configurations, raw metrics, exported reconstructions/trajectories, parameter audit and environment metadata are under `outputs/hyperparameter_check/`.\n"]
    args.report.write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote {args.report}; {total} recorded calls")
    environment_path = args.input/"environment.json"
    if environment_path.exists():
        environment = json.loads(environment_path.read_text())
        sources = [ROOT/"scripts"/name for name in ("sweep_reconstruction.py",
            "sweep_scene.py", "sweep_scene_frontend.py", "report_hyperparameter_check.py")]
        environment["source_hashes"] = {p.relative_to(ROOT).as_posix():
            hashlib.sha256(p.read_bytes()).hexdigest() for p in sources}
        environment["method"] = "Bounded object screening, coverage guards, interaction checks, repeats and seed-1 checks; native graph screening plus direct Python tests; three-window RGB-D screening followed by full baseline/selected runs and combined frontend/graph validation."
        environment_path.write_text(json.dumps(environment, indent=2), encoding="utf-8")
    try:
        print(f"Wrote {plot_comparisons(args.input)}")
    except ImportError:
        print("Comparison plot skipped: install the visualization extra for matplotlib.")


if __name__ == "__main__":
    main()
