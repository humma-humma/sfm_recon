"""Numerical acceptance checks for separately evaluated splat candidates."""
from __future__ import annotations

import numpy as np


def compare_candidate(reference: dict, candidate: dict) -> dict:
    """Require matching held-out views and limit aggregate and local regressions.

    A passing gate still requires visual review; reduced splat counts alone are
    not evidence of better reconstruction or rendering quality.
    """
    before = {r["image"]: r for r in reference["views"]}
    after = {r["image"]: r for r in candidate["views"]}
    if set(before) != set(after) or not before:
        raise ValueError("Evaluation must use the same nonempty held-out image set")
    if len(before) != len(reference["views"]) or len(after) != len(candidate["views"]):
        raise ValueError("Evaluation images must be unique")
    keys = ("psnr", "ssim", "lpips", "opacity_coverage")
    delta = {}
    for key in keys:
        values = np.array([[before[n][key], after[n][key]] for n in before])
        if not np.isfinite(values).all():
            raise ValueError("Evaluation metrics must be finite")
        delta[key] = float(np.mean(values[:, 1] - values[:, 0]))
    coverage_ratios = [after[n]["opacity_coverage"] / max(before[n]["opacity_coverage"], 1e-12)
                       if before[n]["opacity_coverage"] > 0 else 1. for n in before]
    worst_psnr_delta = min(after[n]["psnr"] - before[n]["psnr"] for n in before)
    checks = {
        "mean_psnr": delta["psnr"] >= -.1,
        "mean_ssim": delta["ssim"] >= -.002,
        "mean_lpips": delta["lpips"] <= .005,
        "worst_view_psnr": worst_psnr_delta >= -.5,
        "mean_coverage_retention": float(np.mean(coverage_ratios)) >= .98,
        "worst_view_coverage_retention": min(coverage_ratios) >= .95,
    }
    passed = all(checks.values())
    return {"status": "passes_numeric_gate_visual_review_pending" if passed else "rejected",
            "mean_deltas": delta, "worst_view_psnr_delta": worst_psnr_delta,
            "mean_coverage_retention": float(np.mean(coverage_ratios)),
            "worst_view_coverage_retention": min(coverage_ratios), "checks": checks}
