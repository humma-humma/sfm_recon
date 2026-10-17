"""Publish exact training targets and locally matched unseen-view diagnostics."""
import argparse
import html
import json
from pathlib import Path
import shutil
import urllib.request

ROOT=Path(__file__).resolve().parents[1]


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--root',type=Path,required=True)
    args=parser.parse_args();folder=args.root/'assets/gnt_transfer_diagnosis'
    report=json.loads((folder/'results.json').read_text());assert report['status']=='complete'
    training=[r for r in report['rows'] if r['group']=='training']
    unseen=[r for r in report['rows'] if r['group']!='training' and not r['exact_training_time']]
    table=[]
    for group,modes in report['summaries'].items():
        for mode,summary in modes.items():
            gap='—' if group=='training' else f'{summary["mean_local_psnr_gap"]:+.2f}'
            fraction='—' if group=='training' else f'{100*summary["fraction_more_than_3db_worse"]:.1f}%'
            table.append(f'<tr><td>{group}</td><td>{mode}</td><td>{summary["count"]}</td><td>{summary["mean_psnr"]:.2f}</td><td>{summary["mean_ssim"]:.4f}</td><td>{gap}</td><td>{fraction}</td></tr>')
    prefix='assets/gnt_finalization/gnt_transfer_diagnosis/'
    def picture(row,target=False):
        name=row['target'] if target else row['native_file']
        attributes='' if target else f' class="fit" data-native="{prefix+row["native_file"]}" data-black="{prefix+row["black_file"]}"'
        label='GNT target' if target else 'Gaussian fit'
        return f'<figure><a href="{prefix+name}" target="_blank"><img loading="lazy" src="{prefix+name}"{attributes} alt="{label} {row["key"]}"></a><figcaption>{label}</figcaption></figure>'
    selected=[]
    indices=[r['index'] for r in sorted(training,key=lambda r:r['native']['psnr'])[:4]]+[0,38,91,183,274]
    for i in indices:
        if i not in selected:selected.append(i)
    stills=[]
    for i in selected:
        row=training[i]
        stills.append(f'<section><h3>Training {i} · route time {row["source_time"]-training[0]["source_time"]:.2f}s · native PSNR {row["native"]["psnr"]:.2f} dB</h3><div class="pair">{picture(row,True)}{picture(row)}</div></section>')
    gaps=[]
    # At most one dense example per training interval to avoid showing duplicates.
    for group in ('dense','adaptive'):
        selected_brackets=set();selected_rows=[]
        for row in sorted([r for r in unseen if r['group']==group],key=lambda r:r['native']['psnr_gap']):
            bracket=tuple(row['bracket'])
            if bracket in selected_brackets:continue
            selected_brackets.add(bracket);selected_rows.append(row)
            if len(selected_rows)==4:break
        for row in selected_rows:
            left,right=row['bracket'];neighbors=[training[left],row,training[right]]
            heading=f'{group} {row["index"]} · route time {row["source_time"]-training[0]["source_time"]:.2f}s · local PSNR gap {row["native"]["psnr_gap"]:+.2f} dB'
            columns=''.join(f'<div><h4>{label} · PSNR {neighbor["native"]["psnr"]:.2f}</h4>{picture(neighbor,True)}{picture(neighbor)}</div>' for label,neighbor in zip(('Fitted left endpoint','Unseen pose','Fitted right endpoint'),neighbors))
            gaps.append(f'<section><h3>{heading}</h3><div class="triple">{columns}</div></section>')
    audit=html.escape(json.dumps(report['renderer_audit'],indent=2))
    page='''<!doctype html><html><meta charset="utf-8"><title>GNT-to-splat transfer diagnosis</title>
<style>body{margin:24px auto;max-width:1400px;padding:0 16px;font:16px system-ui;background:#121720;color:#eee}a{color:#8cf}table{border-collapse:collapse}td,th{padding:9px;border-bottom:1px solid #455;text-align:left}.pair,.triple{display:grid;gap:12px}.pair{grid-template-columns:repeat(2,1fr)}.triple{grid-template-columns:repeat(3,1fr)}img{width:100%;display:block}figure{margin:0 0 12px}section{border-top:1px solid #455;padding:12px 0}figcaption{color:#aab}pre{white-space:pre-wrap}select{padding:8px}@media(max-width:800px){.triple,.pair{grid-template-columns:1fr}}</style>
<h1>Where does GNT-to-splat transfer fail?</h1>
<p>The frozen GNT-only 10k Gaussian model is compared with all 275 exact RGB training targets, 180 existing dense-turn targets and 48 existing adaptive targets. Views within 0.1 ms of a fitted timestamp are excluded from unseen summaries. No new GNT images or training were needed. Teacher agreement is not photographic ground truth.</p>
<p>Local gap compares an unseen pose with the interpolated mean-square error of its two neighboring fitted poses. A negative gap means worse agreement between training cameras. Different visible content and the deliberately difficult adaptive selection still confound this comparison; it does not prove a geometric cause. Dense frames cover one outdoor turn and are temporally correlated.</p>
<table><tr><th>Poses</th><th>Background</th><th>Count</th><th>Mean PSNR</th><th>Mean SSIM</th><th>Mean local gap, dB</th><th>Over 3 dB worse locally</th></tr>__TABLE__</table>
<p>Black reproduces the earlier direct-render comparison. Native uses Nerfstudio's inference background: __BACKGROUND__. Click an image for its native 960×540 PNG.</p>
<label>Visual background: <select id="mode"><option value="native">Native Nerfstudio</option><option value="black">Previous black control</option></select></label>
<p>Table values and card PSNR labels remain explicitly identified; card labels describe native background.</p>
<h2>Exact training views: four worst fits plus route anchors</h2>__STILLS__
<h2>Worst local gaps, with fitted neighbors</h2>__GAPS__
<h2>Renderer and camera audit</h2><pre>__AUDIT__</pre>
<script>document.getElementById('mode').onchange=e=>document.querySelectorAll('img.fit').forEach(img=>{img.src=img.dataset[e.target.value];img.parentElement.href=img.src;});</script></html>'''
    page=page.replace('__TABLE__',''.join(table)).replace('__BACKGROUND__',str(report['native_background']))
    page=page.replace('__STILLS__',''.join(stills)).replace('__GAPS__',''.join(gaps)).replace('__AUDIT__',audit)
    path=ROOT/'web_demo/gnt-transfer-diagnosis.html';path.write_text(page,encoding='utf-8')
    output=ROOT/'outputs/gnt_finalization/transfer_diagnosis.json';shutil.copy2(folder/'results.json',output)
    with urllib.request.urlopen('http://localhost:8088/gnt-transfer-diagnosis.html') as response:
        assert response.status==200 and response.read()==path.read_bytes()
    print(json.dumps(dict(status='published',training_cards=len(selected),gap_cards=len(gaps),url='http://localhost:8088/gnt-transfer-diagnosis.html')),flush=True)


if __name__=='__main__':main()
