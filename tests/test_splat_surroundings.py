"""Check report generation using tiny, known source/render pairs."""
import importlib.util
import json
from pathlib import Path

import numpy as np
from PIL import Image


def test_static_region_comparison_and_gallery(tmp_path, monkeypatch):
    path = Path(__file__).resolve().parents[1] / 'scripts/test_splat_surroundings.py'
    spec = importlib.util.spec_from_file_location('surroundings_runner', path)
    runner = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(runner)
    monkeypatch.setattr(runner, 'ROOT', tmp_path)
    monkeypatch.setattr(runner, 'OUT', tmp_path / 'experiment')
    baseline = tmp_path / 'outputs/splat_refresh_v1/evaluation/boot_old'
    candidate = runner.OUT / 'evaluation/boot_person_mask'
    masks = runner.OUT / 'masked_data/boot/masks'
    for directory in [baseline, candidate, masks]:
        directory.mkdir(parents=True)
    source = np.full((32, 32, 3), 100, dtype=np.uint8)
    old = source + 10
    new = source.copy()
    # A deliberate error only in the excluded person region must not harm
    # the static-region score; full-image metrics remain separately reported.
    new[:, :8] = 255
    Image.fromarray(np.concatenate([source, old], axis=1)).save(baseline / '000_img.png')
    Image.fromarray(np.concatenate([source, new], axis=1)).save(candidate / '000_img.png')
    mask = np.full((32, 32), 255, dtype=np.uint8)
    mask[:, :8] = 0
    Image.fromarray(mask).save(masks / 'test.png')
    row = dict(image='test.jpg', psnr=20., ssim=.8, lpips=.2, opacity_coverage=1.)
    for directory in [baseline, candidate]:
        runner.save(directory / 'metrics.json', dict(views=[row]))
    runner.save(runner.OUT / 'workflow.json', dict(experiments={
        'boot_person_mask': dict(status='evaluated', dataset='boot',
                                runtime=dict(seconds=60, peak_cuda_allocated_mb=100, gaussians=20))}))
    runner.finalize({})
    report = json.loads((runner.OUT / 'comparisons.json').read_text())['boot_person_mask']
    assert report['same_static_pixels']['mean_psnr_delta'] > 50
    assert abs(report['same_static_pixels']['mean_ssim_delta']) < 1
    assert report['against_selected']['mean_deltas']['psnr'] == 0
    assert Image.open(runner.OUT / 'comparisons/boot_person_mask/000.jpg').size == (96, 32)
    assert 'fresh control' not in (runner.OUT / 'index.html').read_text()
