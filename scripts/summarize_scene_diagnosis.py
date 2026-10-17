"""Validate completed diagnosis artifacts and assemble a read-only comparison."""
from pathlib import Path
import html
import json
import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'outputs/scene_diagnosis'
LABELS = ('5000', '10000', '15000')


def main():
    if not json.loads((OUT / 'render_complete.json').read_text())['complete']:
        raise ValueError('Rendering not complete')
    selection = json.loads((OUT / 'render_selection.json').read_text())
    geometry = json.loads((OUT / 'geometry.json').read_text())
    expected = {(s['split'], s['image']) for s in selection}
    reports, lookup = {}, {}
    review = OUT / 'review'
    review.mkdir(exist_ok=True)
    checked = 0
    for label in LABELS:
        folder = OUT / f'render_{label}'
        report = json.loads((folder / 'metrics.json').read_text())
        assert report['step'] == int(label)-1, (label, report['step'])
        rows = report['views']
        assert len(rows) == len(expected)
        assert {(r['split'], r['image']) for r in rows} == expected
        lookup[label] = {(r['split'], r['image']): r for r in rows}
        for row in rows:
            for metric in ('psnr', 'ssim', 'lpips'):
                assert np.isfinite(row[metric]), (label, row)
            stem = f"{row['split']}_{Path(row['image']).stem}"
            pair = cv2.imread(str(folder / f'{stem}.jpg'))
            assert pair is not None and pair.shape == (540, 1920, 3), stem
            with np.load(folder / f'{stem}_depth.npz') as data:
                predicted, measured, alpha = (data[k] for k in ('rendered_m', 'measured_m', 'opacity'))
                assert predicted.shape == measured.shape == alpha.shape == (540, 960)
                valid = (measured > .1) & (measured < 8) & np.isfinite(predicted) & (alpha > .9)
                residual = predicted[valid] - measured[valid]
                error = float(np.median(np.abs(residual)))
                conflict = float((residual < -(.05+.03*measured[valid])).mean())
                assert int(valid.sum()) == row['depth_comparison_pixels']
                assert abs(error-row['median_absolute_rendered_depth_residual_m']) < 1e-9
                assert abs(conflict-row['rendered_depth_free_space_fraction']) < 1e-9
                row['median_rendered_depth_m'] = float(np.median(predicted[valid]))
                row['median_measured_depth_m'] = float(np.median(measured[valid]))
                row['median_rendered_to_measured_depth_ratio'] = float(np.median(predicted[valid]/measured[valid]))
            checked += 1
        for split in ('train', 'eval'):
            for metric in ('psnr', 'ssim', 'lpips'):
                mean = float(np.mean([r[metric] for r in rows if r['split'] == split]))
                assert abs(mean-report['mean_by_split'][split][metric]) < 1e-9
        # Confirm these six held-out renders reproduce the earlier evaluations.
        old_path = ROOT / ('outputs/splat_surroundings_tests/evaluation/scene_control/metrics.json' if label == '5000'
                           else f'outputs/scene_control_continuation/evaluation/scene_{label}/metrics.json')
        previous = {r['image']: r for r in json.loads(old_path.read_text())['views']}
        for row in rows:
            if row['split'] == 'eval':
                assert abs(row['psnr']-previous[row['image']]['psnr']) < 1e-4
        reports[label] = report

    regions = []
    for target in geometry['views']:
        neighbors = {p['image'] for p in target['pairs']}
        item = {'image': target['image'], 'checkpoints': {}}
        for label in LABELS:
            training = [r for r in reports[label]['views'] if r['split']=='train' and r['image'] in neighbors]
            heldout = lookup[label]['eval', target['image']]
            mean_train = float(np.mean([r['psnr'] for r in training]))
            item['checkpoints'][label] = {
                'neighbor_train_views': len(training), 'neighbor_train_mean_psnr': mean_train,
                'heldout_psnr': heldout['psnr'], 'psnr_gap': mean_train-heldout['psnr'],
                'neighbor_train_mean_median_depth_error_m': float(np.mean([r['median_absolute_rendered_depth_residual_m'] for r in training])),
                **{k: heldout[k] for k in ('median_rendered_depth_m', 'median_measured_depth_m',
                                          'median_absolute_rendered_depth_residual_m', 'rendered_depth_free_space_fraction')}}
        regions.append(item)

    # Every selected view: source plus checkpoint renders, one common size.
    gallery = ['<!doctype html><html><head><meta charset="utf-8"><title>Scene render diagnosis</title>',
               '<style>body{background:#15181e;color:#edf0f5;font:16px system-ui;margin:24px}img{width:100%;max-width:1920px}a{color:#91c9ff}section{margin:36px 0}table{border-collapse:collapse}td,th{padding:8px;border:1px solid #555}</style></head><body>',
               '<h1>Scene render diagnosis</h1><p>Source | fresh 5k | continued 10k | continued 15k. All views rendered with Nerfstudio at 960x540; gallery panels reduced for display.</p>',
               '<p>Train and held-out views are different nearby frames. The PSNR gap is descriptive, not a controlled same-image experiment. Depth assumes aligned optical-axis measurements in millimetres; expected splat depth is not surface thickness.</p>',
               '<p><a href="index.html">RGB-D coverage diagnosis</a> · <a href="summary.json">Validated numeric results</a></p>',
               '<h2>Six selected held-out views</h2>']
    for split in ('eval','train'):
        if split == 'train':
            gallery.append('<h2>47 neighboring training views</h2>')
        chosen = sorted((s for s in selection if s['split']==split), key=lambda s:float(Path(s['image']).stem))
        for entry in chosen:
            stem = f"{split}_{Path(entry['image']).stem}"
            panels = []
            for label in LABELS:
                pair = cv2.imread(str(OUT/f'render_{label}'/f'{stem}.jpg'))
                if not panels:
                    panels.append(pair[:, :960])
                panels.append(pair[:, 960:])
            composite = cv2.resize(np.hstack(panels),(1920,270),interpolation=cv2.INTER_AREA)
            cv2.imwrite(str(review/f'{stem}.jpg'),composite)
            metrics = ' | '.join(f'{label}: PSNR {lookup[label][split,entry["image"]]["psnr"]:.2f}, LPIPS {lookup[label][split,entry["image"]]["lpips"]:.3f}' for label in LABELS)
            gallery.append(f'<section><h3>{html.escape(split+": "+entry["image"])}</h3><p>{metrics}</p><a href="review/{stem}.jpg"><img loading="lazy" src="review/{stem}.jpg"></a></section>')
    gallery.append('</body></html>')
    (OUT/'renders.html').write_text('\n'.join(gallery),encoding='utf-8')
    index_path = OUT/'index.html'
    original = index_path.read_text(encoding='utf-8')
    if 'href="renders.html"' not in original:
        original = original.replace('</body>', '<p><a href="renders.html">Completed training and held-out render comparisons</a></p></body>')
        index_path.write_text(original,encoding='utf-8')
    summary = {'validated_render_depth_pairs': checked, 'train_views_per_checkpoint':47, 'heldout_views_per_checkpoint':6,
               'means':{label:reports[label]['mean_by_split'] for label in LABELS}, 'regions':regions,
               'limitations':['Six held-out problem/control views, not all 35 held-out views.',
                              'Training versus held-out compares different neighboring frames.',
                              'Expected rendered depth is not surface thickness; assumes sensor alignment and units.']}
    (OUT/'summary.json').write_text(json.dumps(summary,indent=2))
    print(json.dumps(summary,indent=2))


if __name__ == '__main__':
    main()
