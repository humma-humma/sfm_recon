"""Publish matched withheld-frame comparisons for the density ablation."""
import argparse
import json
from pathlib import Path
import urllib.request
import hashlib
import numpy as np

ROOT=Path(__file__).resolve().parents[1]


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--balanced',action='store_true')
    args=parser.parse_args()
    report=json.loads((args.root/'assets/gnt_dense_supervision/results.json').read_text())
    assert report['status']=='complete'
    active=('initial','control','dense','balanced90') if args.balanced else ('initial','control','dense')
    candidate,reference=('balanced90','dense') if args.balanced else ('dense','control')
    arms={a:report['arms'][a] for a in active};table=[]
    if args.balanced:
        fit=json.loads((args.root/'gnt_dense_supervision/models/balanced90/splatfacto/continued/fit_results.json').read_text())
        assert fit['source_sha256']==arms['initial']['checkpoint_sha256']
        assert fit['checkpoint_sha256']==arms['balanced90']['checkpoint_sha256']
        assert fit['sampling']['original_batches']==1800 and fit['sampling']['added_batches']==200
        assert fit['gaussians']==1892910 and fit['optimizer_bindings_verified']
        (args.root/'assets/gnt_dense_supervision/balanced90_protocol.json').write_text(json.dumps(fit,indent=2))
    paired={}
    for group in ('withheld','adaptive','added','original_training'):
        control={r['index']:r for r in arms[reference]['rows'] if r['group']==group}
        compared={r['index']:r for r in arms[candidate]['rows'] if r['group']==group}
        assert control.keys()==compared.keys()
        deltas=np.array([compared[i]['psnr']-control[i]['psnr'] for i in control])
        paired[group]=dict(count=len(deltas),mean_psnr_delta=float(deltas.mean()),
                           median_psnr_delta=float(np.median(deltas)),fraction_improved=float(np.mean(deltas>0)),
                           aggregate_mse_reduction_percent=float(100*(1-sum(r['mse'] for r in compared.values())/sum(r['mse'] for r in control.values()))))
    # Recheck all target bytes, including evaluation-only views, before publication.
    dense=json.loads((args.root/'assets/dense_turn/progress.json').read_text())
    for row in dense['frames']:
        assert all(i%8 for i in row['sources'])
        path=args.root/'assets/dense_turn'/row['file']
        assert hashlib.sha256(path.read_bytes()).hexdigest()==row['sha256']
    adaptive=json.loads((args.root/'assets/adaptive_scene/progress.json').read_text())
    for row in adaptive['native'].values():
        assert all(i%8 for i in row['sources'])
        assert hashlib.sha256((args.root/'assets/adaptive_scene'/row['file']).read_bytes()).hexdigest()==row['sha256']
    ordered=sorted(adaptive['native'].values(),key=lambda r:r['case']['source_time'])
    lo,hi=dense['cases'][0]['source_time'],dense['cases'][-1]['source_time']
    outside={i for i in report['plan']['adaptive'] if not lo<=ordered[i]['case']['source_time']<=hi}
    outside_rows={a:[r for r in arm['rows'] if r['group']=='adaptive' and r['index'] in outside] for a,arm in arms.items()}
    deltas=np.array([d['psnr']-c['psnr'] for d,c in zip(outside_rows[candidate],outside_rows[reference])])
    paired['adaptive_outside_turn']=dict(count=len(deltas),mean_psnr_delta=float(deltas.mean()),
        median_psnr_delta=float(np.median(deltas)),fraction_improved=float(np.mean(deltas>0)),
        means={a:{m:float(np.mean([r[m] for r in rows])) for m in ('psnr','ssim')} for a,rows in outside_rows.items()})
    diagnosis=json.loads((args.root/'assets/gnt_transfer_diagnosis/results.json').read_text())
    previous={(r['group'],r['index']):r for r in diagnosis['rows']}
    errors=[]
    for row in arms['initial']['rows']:
        group='dense' if row['group'] in ('added','withheld') else 'training' if row['group']=='original_training' else row['group']
        errors.append(abs(row['psnr']-previous[(group,row['index'])]['native']['psnr']))
    assert max(errors)<.02,'Initial native renderer changed'
    from sfm_reconstruction.stage3_splat_temporal_diagnostics import _read_colmap
    _,poses,_=_read_colmap(args.root/'gnt_dense_supervision/data/dense/sparse/0')
    poses={name:(r,t) for name,r,t in poses}
    for row in report['plan']['added_rows']:
        r,t=poses[row['name']]
        np.testing.assert_allclose(r,row['r'],atol=1e-12)
        np.testing.assert_allclose(t,row['t'],atol=1e-12)
    analysis_name='balanced90_paired_analysis.json' if args.balanced else 'paired_analysis.json'
    (args.root/'assets/gnt_dense_supervision'/analysis_name).write_text(json.dumps(paired,indent=2))
    for group in ('withheld','adaptive','added','original_training'):
        for arm in active:
            s=arms[arm]['summaries'][group]
            table.append(f'<tr><td>{group}</td><td>{arm}</td><td>{s["count"]}</td><td>{s["psnr"]:.2f}</td><td>{s["ssim"]:.4f}</td></tr>')
    for arm in active:
        s=paired['adaptive_outside_turn']['means'][arm]
        table.append(f'<tr><td>adaptive outside turn only</td><td>{arm}</td><td>{len(outside)}</td><td>{s["psnr"]:.2f}</td><td>{s["ssim"]:.4f}</td></tr>')
    data=[]
    for row in arms['initial']['rows']:
        target=row['target'] or 'gnt_transfer_diagnosis/'+previous[('training',row['index'])]['target']
        item=dict(group=row['group'],index=row['index'],target=target)
        for arm in active:
            matched=next(r for r in arms[arm]['rows'] if (r['group'],r['index'])==(row['group'],row['index']))
            item[arm]=dict(file=matched['file'],psnr=matched['psnr'])
        item['delta']=item[candidate]['psnr']-item[reference]['psnr']
        data.append(item)
    page='''<!doctype html><html><meta charset="utf-8"><title>Dense GNT supervision test</title>
<style>body{margin:24px auto;max-width:1700px;padding:0 16px;font:16px system-ui;background:#121720;color:#eee}a{color:#8cf}table{border-collapse:collapse}td,th{padding:8px;border-bottom:1px solid #455;text-align:left}select,button{padding:8px;margin:4px}.grid{display:grid;grid-template-columns:repeat(2,1fr);gap:12px}img{width:100%;display:block}figure{margin:0}figcaption{padding:6px}@media(max-width:700px){.grid{grid-template-columns:1fr}}</style>
<h1>__HEADING__</h1>__BALANCED_PROTOCOL__
<p>Both models continue the same 10k-step GNT-only checkpoint for 2,000 steps with identical optimizer state, schedules, seed and fixed Gaussian count. Control uses 275 original targets; Dense adds 85 alternating turn views. The other 84 turn views are withheld from both. There are __ADAPTIVE__ non-overlapping adaptive views; __EXCLUDED__ adaptive views near added training timestamps are excluded. The table separately reports adaptive views outside the six-second turn.</p>
<p>These scores measure agreement with GNT, not photographic truth. This is a single-seed exploratory test on previously inspected poses. No model is automatically selected. All renders use the native inference background at 960 &times; 540.</p>
<table><tr><th>Views</th><th>Model</th><th>Count</th><th>PSNR, dB</th><th>SSIM</th></tr>__TABLE__</table>
<p><label>Views <select id="group"><option value="withheld">Withheld turn frames</option><option value="adaptive">Adaptive views</option><option value="added">Added training frames</option><option value="original_training">Original training views</option></select></label>
<label>Order <select id="order"><option value="index">Frame order</option><option value="best">Largest candidate improvement</option><option value="worst">Largest candidate regression</option></select></label>
<button id="prev">Previous</button><select id="frame"></select><button id="next">Next</button></p>
<p id="score"></p><div class="grid">__FIGURES__</div>
<script>
const data=__DATA__,prefix='assets/gnt_finalization/',g=document.getElementById('group'),o=document.getElementById('order'),f=document.getElementById('frame');
let rows=[];
function show(){const r=rows[+f.value];if(!r)return;document.getElementById('score').textContent=`${r.group} ${r.index}: __DELTA_LABEL__ ${r.delta.toFixed(2)} dB`;
for(const arm of __DISPLAY_ARMS__){const image=document.getElementById(arm);image.src=prefix+(arm==='target'?r.target:'gnt_dense_supervision/'+r[arm].file);image.parentElement.href=image.src;}}
function list(){rows=data.filter(r=>r.group===g.value).sort((a,b)=>o.value==='best'?b.delta-a.delta:o.value==='worst'?a.delta-b.delta:a.index-b.index);f.replaceChildren(...rows.map((r,i)=>new Option('Frame '+r.index,i)));show();}
g.onchange=o.onchange=list;f.onchange=show;
document.getElementById('prev').onclick=()=>{f.value=(+f.value+rows.length-1)%rows.length;show();};document.getElementById('next').onclick=()=>{f.value=(+f.value+1)%rows.length;show();};
const params=new URLSearchParams(location.search);if([...g.options].some(x=>x.value===params.get('group')))g.value=params.get('group');list();
if(params.has('frame')){const pos=rows.findIndex(r=>r.index===+params.get('frame'));if(pos>=0){f.value=pos;show();}}
</script></html>'''
    labels=[('target','1. Best GNT — quality reference'),('dense','2. Previous test — uniform dense supervision'),('balanced90','3. Current test — balanced supervision (90% original / 10% added)')] if args.balanced else [('target','GNT target'),('initial','Initial GNT-only 10k'),('control','Control: original targets, +2k'),('dense','Dense: +85 targets, +2k')]
    figures=''.join(f'<figure><figcaption>{label}</figcaption><a target="_blank"><img id="{arm}" alt="{label}"></a></figure>' for arm,label in labels)
    page=page.replace('__TABLE__',''.join(table)).replace('__FIGURES__',figures).replace('__DATA__',json.dumps(data))
    page=page.replace('__ADAPTIVE__',str(len(report['plan']['adaptive']))).replace('__EXCLUDED__',str(48-len(report['plan']['adaptive'])))
    page=page.replace('__DISPLAY_ARMS__',json.dumps([a for a,_ in labels])).replace('__DELTA_LABEL__','Balanced minus uniform dense' if args.balanced else 'Dense minus control')
    page=page.replace('__HEADING__','Can balanced sampling protect existing views?' if args.balanced else 'Does denser GNT supervision improve unseen views?')
    page=page.replace('__BALANCED_PROTOCOL__','<p>The balanced model starts from the same original 10k checkpoint, with the same 2,000 steps, targets, optimizer state and fixed Gaussian count. Sampling changes to exactly 1,800 original-view batches and 200 added-view batches, shuffled within each group. Original views are visited 6 or 7 times; added views 2 or 3 times. Sorting compares balanced against uniform dense.</p>' if args.balanced else '')
    name='gnt-balanced-supervision.html' if args.balanced else 'gnt-dense-supervision.html'
    page=page.replace('<title>Dense GNT supervision test</title>',f'<title>{name[:-5]}</title>')
    if args.balanced:
        start=page.index('<h1>');end=page.index('<p><label>Views')
        page=page[:start]+'<h1>GNT and the latest two tests</h1><p>All three panels show the same camera view. Click an image to open it at full resolution. Improvement/regression sorting compares the current test with the previous test.</p>\n'+page[end:]
        page=page.replace('grid-template-columns:repeat(2,1fr)','grid-template-columns:repeat(3,1fr)')
        page=page.replace('figcaption{padding:6px}','figcaption{padding:10px 12px;font-weight:600;height:84px;box-sizing:border-box;display:flex;align-items:center}')
        page=page.replace(f'<title>{name[:-5]}</title>','<title>GNT / Previous test / Current test</title>')
        page=page.replace('Largest candidate improvement','Largest current improvement').replace('Largest candidate regression','Largest current regression')
        page=page.replace('${r.group} ${r.index}: Balanced minus uniform dense ${r.delta.toFixed(2)} dB','${g.selectedOptions[0].textContent} · Frame ${r.index}')
    path=ROOT/'web_demo'/name;path.write_text(page,encoding='utf-8')
    url='http://localhost:8088/'+name
    with urllib.request.urlopen(url) as response:
        assert response.read()==path.read_bytes()
    print(json.dumps(dict(status='published',url=url,frames=len(data),candidate=candidate,reference=reference,paired=paired,max_initial_psnr_difference=max(errors))))


if __name__=='__main__':main()
