"""Selection must reject misleading gains from lost cameras or coverage."""
from copy import deepcopy
import importlib

import numpy as np
import pytest


@pytest.fixture
def screen(monkeypatch):
    from pathlib import Path
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1]/"scripts"))
    return importlib.import_module("sweep_reconstruction").screen_score


@pytest.fixture
def baseline():
    return {"status": "ok", "cameras": 50, "points": 1000, "image_coverage": .4,
        "pose_metrics": {"mean_translation": .1, "mean_rotation_degrees": 1.},
        "mesh_proxy": {"cropped_mean_nearest_vertex_distance": .02},
        "mesh_coverage": .6, "reprojection_rmse_px": 1.}


@pytest.mark.parametrize("field,value", [("cameras", 49), ("points", 899),
    ("image_coverage", .37), ("mesh_coverage", .56)])
def test_screen_rejects_coverage_loss_even_with_lower_errors(screen, baseline, field, value):
    better = deepcopy(baseline)
    better["pose_metrics"] = {"mean_translation": .01, "mean_rotation_degrees": .1}
    better["mesh_proxy"]["cropped_mean_nearest_vertex_distance"] = .001
    better[field] = value
    assert np.isinf(screen(better, baseline))


def test_screen_accepts_improvement_with_preserved_coverage(screen, baseline):
    better = deepcopy(baseline)
    better["pose_metrics"] = {"mean_translation": .08, "mean_rotation_degrees": .8}
    better["mesh_proxy"]["cropped_mean_nearest_vertex_distance"] = .016
    assert screen(baseline, baseline) == 1.
    assert screen(better, baseline) == pytest.approx(.8)
    assert np.isinf(screen({"status": "failed"}, baseline))


def test_boot_score_is_consistency_only_without_ground_truth(screen, baseline):
    baseline["pose_metrics"] = None
    baseline.pop("mesh_proxy")
    baseline.pop("mesh_coverage")
    better = deepcopy(baseline)
    better["reprojection_rmse_px"] = .8
    assert screen(better, baseline) == pytest.approx(.8)
