"""Summarize the native graph investigation, Ceres options and distribution work."""
import json
import hashlib
import platform
import sys
from pathlib import Path
from statistics import median

ROOT = Path(__file__).resolve().parents[1]


def main():
    graph = json.loads((ROOT/"outputs/cpp_followup/graph/results.json").read_text())
    runs = graph["runs"]
    chosen = next(r for r in runs if r["label"] == "selected_cpp_repeat1")
    reference = next(r for r in runs if r["label"] == "selected_python_stable")
    repeats = [r["seconds"] for r in runs if r["label"].startswith("selected_cpp_repeat")]
    lines = ["# C++ follow-up - 2026-10-09\n",
        "The three follow-ups are implemented: native graph numerical investigation, configurable Ceres solver/thread options with real-data measurements, and opt-in platform-wheel builds with a hosted CI matrix. Dataset-specific presets preserve the original recipes and expose verified native alternatives. Global defaults remain unchanged.\n",
        "## Pose graph\n",
        "The historical Python residual uses OpenCV's matrix-to-angle-axis logarithm. Near identity it zeroes small rotations, so default finite differences can erase odometry rotation derivatives. C++ uses a smooth principal SO(3) log and analytic derivatives. This explains the different bounded-budget trajectories; the native Jacobian is not shown to be incorrect. A new `python_rotation_log='stable'` / `--python-rotation-log stable` uses SciPy's smooth rotation log. Native and stable Python pose packing also retain tiny initial/fixed rotations. The default historical OpenCV mode remains available.\n",
        "Real-data subgraph diagnostics:\n\n```json\n"+json.dumps(graph["derivative_diagnostics"], indent=2)+"\n```\n",
        f"Selected scene graph: `{graph['selected_native']}`. Three native repeats give ATE {chosen['official_ate']:.6f}; median solver time {median(repeats):.3f} s. Stable Python independently gives ATE {reference['official_ate']:.6f} in {reference['seconds']:.3f} s. Maximum ATE disagreement: {graph['stable_reference_ate_agreement']:.6g}.\n",
        f"Relative to the previous joint scene recipe (ATE 1.609394), the new native recipe reduces error by {(1-chosen['official_ate']/1.609394)*100:.2f}%. Relative to the original frontend plus Python graph (ATE 3.030521), the reduction is {(1-chosen['official_ate']/3.030521)*100:.2f}%. Full trajectories retain all 4,396 poses. The optical-flow frontend is reused from its complete measured run; it is not re-estimated for every graph case.\n",
        "| Case | ATE | Solver seconds | NFEV | Converged |", "|---|---:|---:|---:|---|"]
    for r in runs:
        lines.append(f"| {r['label']} | {r['official_ate']:.6f} | {r['seconds']:.3f} | {r['nfev']} | {r['converged']} |")
    lines += ["\nAll graph comparisons use one fixed image-verified loop and the same frame identities. Odometry measurements are rebuilt from each frontend's own poses. The reported candidates hit their evaluation budgets; agreement is at that budget, not a claim of full convergence or a global optimum. GT guides selection on this sequence; this is not held-out validation.\n",
        "## Ceres options\n",
        "New Python/CLI controls: `ceres_num_threads` / `--ceres-num-threads` and `ceres_linear_solver` / `--ceres-linear-solver` (`dense_schur`, `sparse_schur`, `iterative_schur`). Defaults remain one thread and dense Schur. Sparse Schur explicitly uses Eigen sparse and fails clearly if unavailable. Iterative Schur uses Schur-Jacobi preconditioning. Reports record requested and actual solver/thread settings.\n",
        "Each option has three complete BA calls on identical selected pre-BA inputs. Setup, output, post-filtering and GT evaluation are outside BA timing.\n",
        "| Dataset | Solver | Threads | Median BA seconds | Points | Final robust cost | Converged |",
        "|---|---|---:|---:|---:|---:|---|"]
    confirmations = []
    for name in ("box", "boot", "milk"):
        data = json.loads((ROOT/f"outputs/cpp_followup/ceres_options/{name}.json").read_text())
        for r in data["runs"]:
            if r["status"] != "ok":
                lines.append(f"| {name} | {r['label']} | | FAILED | | {r['error']} | |")
                continue
            solver = r["solver_reports"][-1]
            lines.append(f"| {name} | {r['solver']} | {r['threads']} | {r['median_seconds']:.3f} | {r['points']} | {solver['final_cost']:.6f} | {solver['converged']} |")
        c = data["pipeline_confirmation"]
        confirmations.append((name, c))
    lines += ["\nSelected options were also checked through complete reconstruction with the same seed and cached correspondences. These paired full-pipeline times are single calls; three-call BA medians above provide the narrower solver comparison. Dense/sparse differences are small on these datasets; fastest observed is not universally fastest.\n",
        "| Dataset | Selected option | Full reconstruction baseline / selected (s) | Max camera difference | Max point distance | Identical IDs |",
        "|---|---|---:|---:|---:|---|"]
    for name, c in confirmations:
        lines.append(f"| {name} | {c['option']} | {c['seconds']['baseline']:.3f} / {c['seconds']['selected']:.3f} | {c['max_camera_matrix_difference']:.3g} | {c['max_point_distance']:.3g} | {c['identical_ids']} |")
    lines += ["\nThreading saves about 2–3% of measured complete reconstruction time here. It improves runtime without a demonstrated accuracy gain. Boot still has no full camera or surface ground truth.\n",
        "## Native distribution\n",
        "`setup.py` adds opt-in CMake extensions through the standard Python build interface. Pure, kernel-only and Ceres builds use separate build directories to avoid stale binaries leaking into pure wheels. `build_backend.py` requests pybind11 headers only for native builds unless an existing provider/header path is available. Standard builds remain pure Python.\n",
        "`.github/workflows/native-wheels.yml` builds Python 3.10/3.11 wheels on Windows, Linux and macOS, bundles shared-library dependencies with delvewheel/auditwheel/delocate, includes dependency license notices, and imports/tests installed repaired wheels from outside the checkout. Artifacts are uploaded; nothing is published to a package index. Current targets are glibc 2.39+ on Linux and macOS 14+. Hosted jobs have not been run in this local session.\n",
        "The local package index cannot supply delvewheel, so local Windows Ceres wheels are unrepaired and still require the installed Ceres dependency DLL directory. They must not be described as self-contained redistributable wheels. The hosted Windows job performs DLL repair and tests the repaired result.\n",
        "## Presets and reproduction\n",
        "`configs/presets/` contains the original object recipes, their pipeline-verified fast variants, milk's coverage alternative, the new native scene recipe, and stable/legacy Python scene alternatives. Object recipes retain their exact cached correspondence family. Scene recipes retain their verified loop and exact frame indices; they are specific to this dataset.\n",
        "```powershell\n$env:PYTHONPATH = 'src'\npython scripts/diagnose_native_graph.py --confirm-selected\npython scripts/benchmark_ceres_options.py --confirm-pipeline\npython scripts/save_selected_presets.py\npython scripts/report_cpp_followup.py\n\n# Complete object reconstruction:\npython scripts/run_preset.py --preset configs/presets/stage2_milk_fast.json --output outputs/milk_fast\n\n# Reuse the already validated complete scene frontend:\npython scripts/run_preset.py --preset configs/presets/stage3_scene.json --trajectory outputs/hyperparameter_check/scene_frontend/full_validation/trajectory.txt --output outputs/scene_native\n# Omit --trajectory to recompute the entire frontend.\n```\n",
        "Native wheel build (same Python ABI/compiler as target):\n\n```powershell\n$env:SFM_BUILD_NATIVE = '1'\n$env:SFM_WITH_CERES = '1'\n$env:SFM_CERES_PREFIX = (Resolve-Path outputs/ceres_deps).Path\npython -m build --wheel\n# Set SFM_WITH_CERES=0 for kernels only. Clear both flags for pure Python.\n```\n\nSet `SFM_CMAKE` if cmake is not on PATH. `PYBIND11_INCLUDE_DIR` can supply existing headers. `--no-isolation` requires installed build dependencies or that explicit header provider.\n"]
    validation = ROOT/"outputs/cpp_followup/validation.json"
    if validation.exists():
        lines.append("## Local validation\n\n```json\n"+validation.read_text()+"\n```\n")
    (ROOT/"docs/CPP_FOLLOWUP.md").write_text("\n".join(lines), encoding="utf-8")
    sources = [ROOT/p for p in ("native/kernels.cpp", "native/ceres_bundle.cpp", "native/CMakeLists.txt",
        "setup.py", "build_backend.py", "pyproject.toml", ".github/workflows/native-wheels.yml",
        "src/sfm_reconstruction/native_backend.py", "src/sfm_reconstruction/pose_graph.py",
        "src/sfm_reconstruction/bundle_adjustment.py", "scripts/diagnose_native_graph.py",
        "scripts/benchmark_ceres_options.py", "scripts/run_preset.py", "scripts/save_selected_presets.py",
        "scripts/report_cpp_followup.py", "scripts/collect_native_licenses.py", "MANIFEST.in",
        "src/sfm_reconstruction/reconstruction.py", "src/sfm_reconstruction/cli.py",
        "tests/test_native_backend.py", "tests/test_ceres_backend.py")]
    sources += sorted((ROOT/"configs/presets").glob("*.json"))
    metadata = {"python":sys.version, "platform":platform.platform(),
        "source_hashes":{p.relative_to(ROOT).as_posix():hashlib.sha256(p.read_bytes()).hexdigest() for p in sources}}
    (ROOT/"outputs/cpp_followup/environment.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print("Wrote docs/CPP_FOLLOWUP.md")


if __name__ == "__main__":
    main()
