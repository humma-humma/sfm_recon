"""Publish the local GNT tuning outputs without changing selected demo models."""
import argparse
import html
import json
from pathlib import Path
import cv2

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--trial', type=Path, required=True)
    args = parser.parse_args()
    source = args.trial/'gnt_tuning'
    report = json.loads((source/'results.json').read_text())
    out = ROOT/'outputs/gnt_tuning'
    out.mkdir(exist_ok=True)
    assets = ROOT/'web_demo/assets/gnt_tuning'
    assets.mkdir(exist_ok=True)
    old = json.loads((ROOT/'outputs/lightweight_methods/results.json').read_text())
    labels = dict(baseline='Original splats',preferred='Preferred restoration',
                  gnt_base='GNT original settings',gt='Held-out photograph')
    cases = {}
    for run in report['runs']:
        dataset, cid = run['dataset'],run['case']
        development = run['width']==480 and run.get('kind')!='motion'
        if development:
            dataset = 'scene_development_480'
        entries = cases.setdefault(dataset,{})
        if cid not in entries:
            previous = next(c for c in old['datasets'][run['dataset']]['cases'] if c['id']==cid)
            entries[cid] = dict(id=cid,image=run['image'],kind=run.get('kind','heldout'),zoom=1,
                common_comparison=True,width=run['width'],height=run['width']*9//16,
                urls={key:previous['urls'][key] for key in ('baseline','preferred','gt') if key in previous['urls']})
            if development:
                for key,url in list(entries[cid]['urls'].items()):
                    src = cv2.imread(str(ROOT/'web_demo'/url))
                    path = assets/f'{dataset}_{cid}_{key}.jpg'
                    assert cv2.imwrite(str(path),cv2.resize(src,(480,270),interpolation=cv2.INTER_AREA),[cv2.IMWRITE_JPEG_QUALITY,95])
                    entries[cid]['urls'][key] = 'assets/gnt_tuning/'+path.name
            if 'gnt' in previous['urls'] and not development:
                entries[cid]['urls']['gnt_base'] = previous['urls']['gnt']
        label = run['label']
        key = 'gnt_base' if label=='control' else label
        if key not in labels:
            labels[key] = 'GNT '+label+(' (adapted to scene)' if label.startswith('adapt') else '')
        image = cv2.imread(str(source/run['path']))
        assert image is not None and image.shape[:2]==(run['width']*9//16,run['width'])
        path = assets/f'{dataset}_{cid}_{label}.jpg'
        assert cv2.imwrite(str(path),image,[cv2.IMWRITE_JPEG_QUALITY,95])
        entries[cid]['urls'][key] = 'assets/gnt_tuning/'+path.name
    for dataset in report.get('object_decision',{}):
        previous = next(c for c in old['datasets'][dataset]['cases'] if c['id']=='eval_024_z1.0')
        entry = dict(previous)
        entry['urls'] = {key:previous['urls'][key] for key in ('baseline','preferred','gt')}
        entry['urls']['gnt_base'] = previous['urls']['gnt']
        cases[dataset] = {entry['id']:entry}
    data = dict(labels=labels,cases={name:list(entries.values()) for name,entries in cases.items()})
    best = report.get('final_selection',{}).get('label',report['inference_selection']['label'])
    pairs = {}
    for run in report['runs']:
        if run['dataset']=='scene' and run['width']==960 and run['label'] in ('control',best):
            pairs.setdefault(run['case'],{})[run['label']] = run
    paired = [(cid,pair) for cid,pair in pairs.items() if 'control' in pair and best in pair]
    deltas = {cid:pair[best]['psnr']-pair['control']['psnr'] for cid,pair in paired}
    confirmation = [value for cid,value in deltas.items() if cid not in ('eval_120_z1.0','eval_240_z1.0')]
    report['assessment'] = dict(selected_on_development_only=True,full_resolution_psnr_changes=deltas,
        confirmation_mean_psnr_change=sum(confirmation)/len(confirmation) if confirmation else None,
        decision='Targeted stair-view improvement; additional cameras do not establish a global upgrade. Keep original GNT and preferred restoration available; no live model is promoted.')
    summary_rows = []
    for cid,pair in paired:
        scope = 'Development' if cid in ('eval_120_z1.0','eval_240_z1.0') else 'Confirmation'
        summary_rows.append(f'<tr><td>{pair[best]["image"]}</td><td>{scope}</td><td>{pair["control"]["psnr"]:.2f}</td><td>{pair[best]["psnr"]:.2f}</td><td>{deltas[cid]:+.2f}</td></tr>')
    summary = '<h2>Full-resolution scene check</h2><p><strong>Targeted improvement, not a global replacement.</strong> The checkpoint was selected on development views. Additional cameras must be judged separately; their native PSNR changes are shown below. Original GNT and the preferred restoration remain available.</p><table><tr><th>Photo</th><th>Scope</th><th>Original GNT PSNR</th><th>Adapted GNT PSNR</th><th>Change</th></tr>'+''.join(summary_rows)+'</table>'
    rows = []
    for run in report['runs']:
        rows.append('<tr>'+''.join(f'<td>{html.escape(str(value))}</td>' for value in
            (run['dataset'],run['case'],run['label'],run['width'],f'{run["psnr"]:.2f}' if run['psnr'] is not None else 'n/a',f'{run["ssim"]:.4f}' if run['ssim'] is not None else 'n/a',f'{run["seconds"]:.1f}'))+'</tr>')
    selection = html.escape(json.dumps(report.get('final_selection',report.get('inference_selection',{})),indent=2))
    page = '''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>GNT tuning comparison</title><style>body{background:#15191e;color:#eee;font:16px system-ui;margin:24px}a{color:#9ed6ff}p{max-width:1100px;line-height:1.6}select,button,input{margin:8px;padding:6px}canvas{display:block;width:100%;max-width:960px;touch-action:none;cursor:grab}table{border-collapse:collapse}td,th{padding:8px;border:1px solid #777}pre{white-space:pre-wrap}</style>
<h1>GNT: local sampling and adaptation tests</h1><p><a href="index.html">Main demo</a> · <a href="lightweight.html">Original trials</a></p>
<p>Compare the same camera with original GNT, inference tuning, scene adaptation, the preferred restoration, or the reference photograph. These are exploratory tests. Development cameras 120/240 selected settings; additional scene cameras check transfer. Adaptation excludes every eighth photograph, including every comparison target. Existing camera geometry predates this split. No live demo model has been replaced.</p>
<label>Dataset <select id="dataset"></select></label><label>View <select id="view"></select></label><br>
<label>Left <select id="left"></select></label><label>Right <select id="right"></select></label><br>
<label>Wipe <input id="wipe" type="range" min="0" max="100" value="50"></label><button id="toggle">Toggle A/B</button>
<label>Crop zoom <input id="zoom" type="range" min="1" max="4" step=".25" value="1"></label><button id="reset">Reset crop</button>
<p id="status" role="status"></p><canvas id="comparison" width="960" height="540" aria-label="Matched image comparison; drag to pan"></canvas>
SUMMARY<h2>Selection on development views</h2><pre>SELECTION</pre><p>Metrics below are native-render measurements at the listed width. They measure photo agreement, not guaranteed sharpness or temporal consistency. Development selection is by mean PSNR; visual review remains decisive. Runtime is observed on the shared RTX 2080.</p>
<table><tr><th>Dataset</th><th>Camera</th><th>Setting</th><th>Width</th><th>PSNR ↑</th><th>SSIM ↑</th><th>Seconds</th></tr>ROWS</table>
<script id="comparison-data" type="application/json">DATA</script><script src="gnt-tuning-inspector.js" defer></script></html>'''
    page = page.replace('SUMMARY',summary).replace('SELECTION',selection).replace('ROWS',''.join(rows)).replace('DATA',json.dumps(data).replace('<','\\u003c'))
    (ROOT/'web_demo/gnt-tuning.html').write_text(page,encoding='utf-8')
    # Keep the existing inspector behavior; choose the tuned candidate by default.
    js = (ROOT/'web_demo/lightweight-inspector.js').read_text(encoding='utf-8')
    best = report.get('final_selection',{}).get('label',report['inference_selection']['label'])
    fallback = report['inference_selection']['label']
    expression = f"(current().urls[{json.dumps(best)}]?{json.dumps(best)}:(current().urls[{json.dumps(fallback)}]?{json.dumps(fallback)}:'gnt_base'))"
    js = js.replace("(current().urls.gnt?'gnt':'2dgs')",expression)
    (ROOT/'web_demo/gnt-tuning-inspector.js').write_text(js,encoding='utf-8')
    (out/'results.json').write_text(json.dumps(report,indent=2))
    (assets/'results.json').write_text(json.dumps(report,indent=2))
    print('Complete: web_demo/gnt-tuning.html')


if __name__ == '__main__':
    main()
