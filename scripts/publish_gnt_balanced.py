"""Publish the bounded whole-scene GNT experiment without replacing earlier galleries."""
import argparse
import hashlib
import json
from pathlib import Path
import cv2
import numpy as np
from run_gnt_trial import DATA, ROOT
from sfm_reconstruction.stage3_splat_temporal_diagnostics import _read_colmap


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--trial', type=Path, required=True)
    parser.add_argument('--experiment', choices=['balanced','detail'], default='balanced')
    args = parser.parse_args()
    source = args.trial/'gnt_tuning'
    report = json.loads((source/'results.json').read_text())
    detail = args.experiment=='detail'
    slug = f'gnt_{args.experiment}'
    baseline_label = 'balanced5000' if detail else 'adapt1800'
    candidate_label = 'detail_patch6000' if detail else 'balanced5000'
    selection = (dict(label=candidate_label,validation_psnr={label:row['psnr'] for label,row in report['detail_validation'].items()},
                      policy='Candidate shown for visual review; no automatic model selection.') if detail else report['balanced_selection'])
    labels = ['balanced5000','detail_control6000','detail_patch6000'] if detail else list(dict.fromkeys(['adapt1800',selection['label'],'balanced5000']))
    training = (dict(initial_step=5000,total_steps=6000,arms=report['detail_training'],
                     targets=report['balanced_training']['targets'],validation=[24,72,112,152,200,248],test=[56,144,224],
                     baseline_sha256=report['detail_training']['control']['baseline_sha256']) if detail else report['balanced_training'])
    assets = ROOT/'web_demo/assets'/slug
    assets.mkdir(parents=True,exist_ok=True)
    output = ROOT/'outputs'/slug
    output.mkdir(exist_ok=True)
    k,poses,_ = _read_colmap(ROOT/DATA['scene']/'sparse/0')
    cases = {c['id']:c for c in report['detail_cases' if detail else 'balanced_cases']}
    for idx in training['validation']:
        name,r,t = poses[idx]
        cases[f'eval_{idx:03}_z1.0'] = dict(id=f'eval_{idx:03}_z1.0',index=idx,image=name,
            k=list(k),kind='validation',zoom=1.)
    published = []
    for case_id,case in cases.items():
        width = 320 if case_id.startswith('path_') else (480 if case['kind'] in ('validation','confirmation') else 960)
        item = dict(id=case_id,kind=case['kind'],width=width,images={},metrics={})
        if case['kind']=='motion':
            item['title'] = f'Indoor path · frame {int(case_id.rsplit("_",1)[1])+1:02}'
        elif case['kind']=='validation':
            item['title'] = f'Validation · camera {case["index"]}'
        elif case['index'] in (120,240):
            item['title'] = ('Outdoor' if case['index']==120 else 'Indoor stairs') + (' · 1.7× close-up' if case.get('zoom',1.)!=1. else ' · previously inspected')
        else:
            item['title'] = f'Separate test · camera {case["index"]}'
        if detail and case['kind']=='confirmation':
            item['title'] = f'Fresh confirmation · camera {case["index"]}'
        elif detail and case['index'] in (40,176,264) and case['kind']!='motion':
            item['title'] = f'Previously inspected · camera {case["index"]}'
        for label in labels:
            rows = [r for r in report['runs'] if r['case']==case_id and r['width']==width and r['label']==label]
            assert len(rows)==1, (case_id,label,width)
            row = rows[0]
            image = cv2.imread(str(source/row['path']))
            assert image is not None and image.shape[1]==width
            name = f'{case_id}_{label}_{width}.jpg'
            cv2.imwrite(str(assets/name),image,[cv2.IMWRITE_JPEG_QUALITY,95])
            item['images'][label] = f'assets/{slug}/{name}'
            item['metrics'][label] = {key:row[key] for key in ('psnr','ssim','seconds')}
            if case.get('zoom',1.)!=1.:
                item['metrics'][label].update(psnr=None,ssim=None)
        if case['kind']!='motion':
            photo = cv2.imread(str(ROOT/DATA['scene']/'images'/case['image']))
            target_k = case['k']
            def matrix(intrinsics):
                fx,fy,cx,cy = intrinsics[:4]
                return np.array([[fx,0,cx],[0,fy,cy],[0,0,1.]])
            gt = cv2.warpPerspective(photo,matrix(target_k)@np.linalg.inv(matrix(k)),(k[4],k[5]))
            gt = cv2.resize(gt,(width,round(k[5]*width/k[4])),interpolation=cv2.INTER_AREA)
            name = f'{case_id}_photo_{width}.jpg'
            cv2.imwrite(str(assets/name),gt,[cv2.IMWRITE_JPEG_QUALITY,95])
            item['images']['photo'] = f'assets/{slug}/{name}'
        published.append(item)
    checkpoints = []
    entries = ([dict(arm=arm,step=6000,**row['checkpoint']) for arm,row in report['detail_training'].items()]
               if detail else report['balanced_training']['checkpoints'])
    for entry in entries:
        path = source/entry['path']
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        if detail:
            assert digest==entry['sha256'], 'Experiment weights changed after training'
        checkpoints.append(dict(entry,sha256=digest,bytes=path.stat().st_size))
    test_ids = {f'eval_{idx:03}_z1.0' for idx in training['test']}
    differences = []
    historical = []
    for case in published:
        if case['id'] in test_ids:
            differences.append(dict(case=case['id'],psnr_delta=case['metrics'][candidate_label]['psnr']-case['metrics'][baseline_label]['psnr']))
            if detail:
                differences[-1]['control_delta'] = case['metrics']['detail_control6000']['psnr']-case['metrics'][baseline_label]['psnr']
                differences[-1]['patch_vs_control'] = case['metrics'][candidate_label]['psnr']-case['metrics']['detail_control6000']['psnr']
        if case['id'] in (('eval_040_z1.0','eval_176_z1.0','eval_264_z1.0','eval_120_z1.0','eval_240_z1.0') if detail else ('eval_120_z1.0','eval_240_z1.0')):
            historical.append(dict(case=case['id'],psnr_delta=case['metrics'][candidate_label]['psnr']-case['metrics'][baseline_label]['psnr']))
    summary = dict(selection=selection,training=training,checkpoints=checkpoints,
        experiment=args.experiment,baseline_label=baseline_label,candidate_label=candidate_label,
        historical_differences=historical,
        assessment='Mixed improvement. Validation selects 5,000 steps, but PSNR falls on two previously inspected views. Keep both models available; no automatic global replacement. Blur, smearing and ghosting remain.',
        test_differences=differences,mean_test_psnr_delta=float(np.mean([r['psnr_delta'] for r in differences])),
        cases=published,note='PSNR/SSIM only valid for uncropped held-out and validation views. Close-up references use adjusted intrinsics; no GT exists for the virtual path. Geometry was estimated before the RGB split.')
    if detail:
        summary['assessment']='Compare the 5,000-step visual baseline, matched 6,000-step RGB control, and 6,000-step patch/structural candidate. Both arms share the same target sequence and initial optimizer state. Sampling and loss change together. Previously inspected views remain exploratory; three fresh confirmation views are excluded from selection. Choose by visible detail and stability, not training loss alone.'
    (output/'results.json').write_text(json.dumps(summary,indent=2))
    (assets/'results.json').write_text(json.dumps(summary,indent=2))
    preview_methods = [('balanced5000','Baseline: 5,000'),('detail_control6000','RGB control: 6,000'),('detail_patch6000','Patch/structure: 6,000'),('photo','Reference photograph')] if detail else [('adapt1800','Current: 1,800 steps'),('balanced5000','Balanced: 5,000 steps'),('photo','Reference photograph')]
    preview = np.zeros((800,640*len(preview_methods),3),dtype=np.uint8)
    for row,case_id in enumerate(('eval_040_z1.0','eval_240_z1.0')):
        case = next(c for c in published if c['id']==case_id)
        for col,(label,title) in enumerate(preview_methods):
            image = cv2.imread(str(ROOT/'web_demo'/case['images'][label]))
            preview[row*400+35:row*400+395,col*640:(col+1)*640] = cv2.resize(image,(640,360),interpolation=cv2.INTER_AREA)
            cv2.putText(preview,f'{title} / camera {case_id[5:8]}',(col*640+10,row*400+25),cv2.FONT_HERSHEY_SIMPLEX,.65,(255,255,255),1)
    cv2.imwrite(str(output/'comparison.jpg'),preview,[cv2.IMWRITE_JPEG_QUALITY,95])
    html = '''<!doctype html><meta charset="utf-8"><meta name="viewport" content="width=device-width"><title>GNT whole-scene comparison</title>
<style>body{background:#111;color:#eee;font:16px system-ui;margin:24px auto;max-width:1250px;padding:0 16px}a{color:#9bd}select,button{padding:8px;margin:5px}#viewport{overflow:auto;max-height:80vh;border:1px solid #555}canvas{display:block;width:100%}table{border-collapse:collapse}td,th{padding:7px;border:1px solid #555}small{color:#bbb}</style>
<h1>GNT: balanced whole-scene training</h1><p>Current 1,800-step model compared with 5,000 total steps. All interval-8 images are excluded from training and source photos. Validation selects the checkpoint; three separate test cameras check transfer. Original galleries and models are preserved.</p>
<p id="summary"></p><p id="assessment"></p><a href="gnt-tuning.html">Earlier comparison</a>
<p><select id="case"></select><select id="left"></select><select id="right"></select><button id="play">Play indoor path</button></p>
<p><label>Wipe <input id="wipe" type="range" min="0" max="100" value="50"></label> <label>Zoom <input id="zoom" type="range" min="100" max="300" value="100"></label></p>
<div id="viewport"><canvas id="canvas"></canvas></div><p id="status"></p>
<small>The path contains 17 independently rendered poses at 320 × 180, played forward and backward. It follows a short indoor capture segment, not the whole scene. Native stills are 960 × 540; validation previews are 480 × 270. Close-ups have photo references but no reported PSNR, because the original renderer's metric uses an uncropped photo.</small>
<h2>Validation</h2><div id="validation"></div><h2>Separate test cameras</h2><div id="test"></div><h2>Previously inspected views</h2><div id="historical"></div>
<script id="comparison-data" type="application/json">__DATA__</script>
<script>
const data=JSON.parse(document.getElementById('comparison-data').textContent),cases=data.cases;
const el=id=>document.getElementById(id), selector=el('case'),left=el('left'),right=el('right'),canvas=el('canvas'),ctx=canvas.getContext('2d');
const names={adapt1800:'Current · 1,800 steps',balanced3200:'Balanced · 3,200 steps',balanced5000:'Balanced · 5,000 steps',detail_control6000:'RGB control · 6,000 steps',detail_patch6000:'Patch + structure · 6,000 steps',photo:'Reference photograph'};
const base=data.baseline_label,candidate=data.candidate_label;
cases.forEach((c,i)=>selector.add(new Option(c.title+' · '+c.width+'px',i)));
let loaded={},generation=0,timer=null,pathIndex=0,direction=1;
function options(){const c=cases[+selector.value];for(const s of [left,right]){const old=s.value;s.replaceChildren();Object.keys(c.images).forEach(key=>s.add(new Option(names[key]||key,key)));if(c.images[old])s.value=old;}if(!left.value)left.value=base;if(right.value===left.value)right.value=left.value===candidate?base:candidate;}
async function load(){const g=++generation,c=cases[+selector.value];const pairs=await Promise.all(Object.entries(c.images).map(async([key,url])=>{const im=new Image();im.src=url;await im.decode();return [key,im]}));if(g!==generation)return;loaded=Object.fromEntries(pairs);draw();}
function draw(){const c=cases[+selector.value],a=loaded[left.value],b=loaded[right.value];if(!a||!b)return;canvas.width=a.naturalWidth;canvas.height=a.naturalHeight;ctx.drawImage(b,0,0);ctx.save();ctx.beginPath();ctx.rect(0,0,canvas.width*el('wipe').value/100,canvas.height);ctx.clip();ctx.drawImage(a,0,0);ctx.restore();ctx.strokeStyle='#fff';ctx.beginPath();ctx.moveTo(canvas.width*el('wipe').value/100,0);ctx.lineTo(canvas.width*el('wipe').value/100,canvas.height);ctx.stroke();canvas.style.width=el('zoom').value+'%';el('status').textContent=`Left: ${names[left.value]} | Right: ${names[right.value]} | ${c.id}`;}
function stop(){clearInterval(timer);timer=null;el('play').textContent='Play indoor path';}
selector.onchange=()=>{stop();options();load()};left.onchange=right.onchange=draw;el('wipe').oninput=el('zoom').oninput=draw;
el('play').onclick=()=>{if(timer){stop();return;}const path=cases.map((c,i)=>c.id.startsWith('path_')?i:-1).filter(i=>i>=0);pathIndex=0;direction=1;const advance=()=>{selector.value=path[pathIndex];options();load();pathIndex+=direction;if(pathIndex>=path.length-1||pathIndex<=0)direction*=-1;};advance();timer=setInterval(advance,180);el('play').textContent='Pause path';};
el('summary').textContent=data.experiment==='detail'?`Patch candidate versus 5,000-step baseline: mean fresh-confirmation PSNR change ${data.mean_test_psnr_delta.toFixed(3)} dB at 480px.`:`Validation selected ${names[data.selection.label]}. Mean separate-test PSNR change at 5,000 steps: ${data.mean_test_psnr_delta.toFixed(3)} dB.`;
el('assessment').textContent=data.assessment;
el('validation').innerHTML='<table><tr><th>Model</th><th>Mean PSNR</th></tr>'+Object.entries(data.selection.validation_psnr).map(([label,value])=>`<tr><td>${names[label]}</td><td>${value.toFixed(3)} dB</td></tr>`).join('')+'</table>';
el('test').innerHTML=data.experiment==='detail'?'<table><tr><th>Camera</th><th>RGB control minus baseline</th><th>Patch minus baseline</th><th>Patch minus RGB control</th></tr>'+data.test_differences.map(c=>`<tr><td>${c.case}</td><td>${c.control_delta.toFixed(3)} dB</td><td>${c.psnr_delta.toFixed(3)} dB</td><td>${c.patch_vs_control.toFixed(3)} dB</td></tr>`).join('')+'</table>':'<table><tr><th>Camera</th><th>5,000 minus 1,800</th></tr>'+data.test_differences.map(c=>`<tr><td>${c.case}</td><td>${c.psnr_delta.toFixed(3)} dB</td></tr>`).join('')+'</table>';
el('historical').innerHTML='<table><tr><th>Camera</th><th>5,000 minus 1,800</th></tr>'+data.historical_differences.map(c=>`<tr><td>${c.case}</td><td>${c.psnr_delta.toFixed(3)} dB</td></tr>`).join('')+'</table>';
options();left.value=base;right.value=candidate;load();
</script>'''
    if detail:
        html = html.replace('GNT whole-scene comparison','GNT detail experiment').replace('GNT: balanced whole-scene training','GNT: detail recovery experiment')
        html = html.replace('Current 1,800-step model compared with 5,000 total steps.','Two matched 1,000-step continuations of the cleaner 5,000-step model.')
        html = html.replace('Validation selects the checkpoint; three separate test cameras check transfer.','Three fresh confirmation cameras check transfer; previously inspected views remain exploratory.')
        html = html.replace('href="gnt-tuning.html"','href="gnt-balanced.html"').replace('5,000 minus 1,800','Patch 6,000 minus baseline 5,000').replace('Separate test cameras','Fresh confirmation cameras (480px)')
    (ROOT/'web_demo'/f'gnt-{args.experiment}.html').write_text(html.replace('__DATA__',json.dumps(summary)),encoding='utf-8')
    print(json.dumps(dict(selection=selection,test= differences,mean_test_psnr_delta=summary['mean_test_psnr_delta'],cases=len(published)),indent=2))


if __name__=='__main__':
    main()
