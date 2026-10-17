"""Fresh-path CLI checks: no original dataset location or matching cache required."""
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest


def runner():
    path = Path(__file__).parents[1]/'scripts/run_preset.py'
    spec = importlib.util.spec_from_file_location('run_preset', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_portable_supplied_matches_and_effective_provenance(tmp_path):
    from sfm_reconstruction.geometry import project_points
    from sfm_reconstruction.models import Pose
    data = tmp_path/'relocated_data'
    images = data/'images'; images.mkdir(parents=True)
    matches = tmp_path/'relocated_matches'; matches.mkdir()
    k = np.array([[700., 0., 320.], [0., 700., 240.], [0., 0., 1.]])
    rng = np.random.default_rng(11)
    points = np.column_stack([rng.uniform(-1., 1., (70, 2)), rng.uniform(4., 8., 70)])
    uv = []
    for i in range(4):
        (images/f'{i:05d}.jpg').touch()
        uv.append(project_points(points, Pose(np.eye(3), [-.45*i, 0., 0.]), k))
    for i in range(3):
        np.savetxt(matches/f'{i}_{i+1}.txt', np.hstack([uv[i], uv[i+1]]))
    (data/'camera_parameters.json').write_text(json.dumps(dict(intrinsics=k.tolist(), extrinsics={})))
    preset = tmp_path/'preset.json'
    preset.write_text(json.dumps(dict(dataset='/unavailable/original', correspondences='/unavailable/cache',
        reconstruction=dict(bundle_adjustment=False, min_pnp_points=8, min_pnp_inliers=8,
                            max_reprojection_error=1., pnp_reprojection_error=1., min_triangulation_angle=.25))))
    out = tmp_path/'result'
    runner().main(['--preset', str(preset), '--dataset', str(data), '--correspondences', str(matches),
                   '--output', str(out), '--seed', '7'])
    assert json.loads((out/'run.json').read_text())['cameras'] == 4
    assert json.loads((out/'preset.json').read_text())['dataset'] == str(data.resolve())
    assert json.loads((out/'environment.json').read_text())['seed'] == 7


def test_missing_cache_points_to_regeneration(tmp_path, capsys):
    preset = tmp_path/'preset.json'
    preset.write_text(json.dumps(dict(dataset=str(tmp_path), correspondences=str(tmp_path/'missing'), reconstruction={})))
    with pytest.raises(SystemExit):
        runner().main(['--preset', str(preset), '--output', str(tmp_path/'out')])
    assert '--generate-matches' in capsys.readouterr().err
