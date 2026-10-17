"""Correct existing splat gap frames using only the surrounding 15fps GNT anchors."""
import argparse
import json
from pathlib import Path
import shutil
import subprocess
import time

import cv2
import numpy as np
from skimage.metrics import structural_similarity
from gnt_rollout_io import digest, save
from publish_gnt_videos import verify

ROOT = Path(__file__).resolve().parents[1]


def colour_fit(splat, anchor):
    """Bounded affine fit on blurred anchor pairs, avoiding saturated pixels."""
    x = cv2.GaussianBlur(splat.astype(np.float32), (0, 0), 5)[::8, ::8]
    y = cv2.GaussianBlur(anchor.astype(np.float32), (0, 0), 5)[::8, ::8]
    gain, bias = [], []
    for c in range(3):
        a, b = x[..., c].ravel(), y[..., c].ravel()
        valid = (a > 10) & (a < 245) & (b > 10) & (b < 245)
        if valid.sum() < 32:
            gain.append(1.); bias.append(0.); continue
        a, b = a[valid], b[valid]
        g = np.clip(np.mean((a-a.mean())*(b-b.mean())) / max(a.var(), 1), .8, 1.2)
        gain.append(g); bias.append(np.clip(np.median(b-g*a), -20, 20))
    return np.array(gain, np.float32), np.array(bias, np.float32)


def apply_colour(image, fit):
    return np.clip(image.astype(np.float32)*fit[0]+fit[1], 0, 255)


def flow_to_anchor(current, anchor, residual, dis):
    """Backward warp an anchor residual; confidence uses splat-only consistency."""
    gray = lambda im: cv2.resize(cv2.cvtColor(im, cv2.COLOR_BGR2GRAY), (480, 270))
    a, b = gray(current), gray(anchor)
    forward = cv2.resize(dis.calc(a, b, None), (960, 540))*2
    backward = cv2.resize(dis.calc(b, a, None), (960, 540))*2
    yy, xx = np.mgrid[:540, :960].astype(np.float32)
    mx, my = xx+forward[..., 0], yy+forward[..., 1]
    warp = lambda im: cv2.remap(im, mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    fb = np.linalg.norm(forward+warp(backward), axis=2)
    photo = np.mean(np.abs(current.astype(np.float32)-warp(anchor).astype(np.float32)), axis=2)
    valid = (mx >= 0) & (mx <= 959) & (my >= 0) & (my <= 539)
    confidence = valid*np.exp(-(fb/1.5)**2)*np.exp(-(photo/25)**2)
    return warp(residual), confidence.astype(np.float32)


def publish(folder, report):
    prefix = 'assets/gnt_finalization/splat_correction/'
    choices = [('raw', 'Previous: uncorrected splat gaps'), ('colour', 'Colour-matched splat gaps'),
               ('detail', 'Motion-aligned detail correction'), ('fallback', 'Corrected splats + interpolation fallback')]
    options = ''.join(f'<option value="{k}"'+(' selected' if k=='fallback' else '')+f'>{v}</option>' for k,v in choices)
    cards = ''.join(f'<figure><figcaption>{title}</figcaption><video id="{key}" controls muted playsinline preload="auto" src="{prefix}{file}.mp4"></video></figure>' for key,title,file in
                    [('reference','1. Best GNT: 30 independent views/s','reference'),
                     ('baseline','2. Current reference: 15fps GNT + motion interpolation','motion'),
                     ('candidate','3. Corrected splats + interpolation fallback','fallback')])
    rows = ''.join(f'<tr><td>{key}</td><td>{value["gap_psnr"]:.2f}</td><td>{value["gap_ssim"]:.4f}</td><td>{value["temporal_residual_mae"]:.3f}</td></tr>' for key,value in report['scores'].items())
    page = '''<!doctype html><html><meta charset="utf-8"><title>15fps GNT: corrected splat gaps</title>
<style>body{max-width:1700px;margin:24px auto;padding:0 16px;background:#121720;color:#eee;font:16px system-ui}a{color:#8cf}.grid{display:grid;grid-template-columns:repeat(3,1fr);gap:14px}figure{margin:0}figcaption{min-height:65px;font-weight:600}video{width:100%;display:block}button,select{padding:8px;margin:5px}td,th{padding:8px;text-align:left}@media(max-width:750px){.grid{grid-template-columns:1fr}}</style>
<h1>15fps GNT with corrected splat gap frames</h1>
<p>Six-second outdoor turn, 960 × 540, 30fps. All candidates keep the same 90 GNT anchors. Only intermediate frames change. No new GNT rendering or splat training.</p>
<p>Colour and detail corrections use only the neighbouring even-numbered GNT anchors. Withheld odd GNT views are used only for scoring. The fallback variant blends with the existing motion interpolation where splat flow is unreliable; it is not a pure splat render.</p>
<button id="play">Play all from start</button><button id="pause">Pause all</button>
<label>Speed <select id="speed"><option value="1">1×</option><option value="0.5">0.5×</option><option value="0.25">0.25×</option></select></label>
<label>Third panel <select id="variant">__OPTIONS__</select></label>
<label>Seek <input id="seek" type="range" min="0" max="5.966666667" step="0.033333333" value="0"></label><p id="frame"></p>
<div class="grid">__CARDS__</div><details><summary>Gap-only results and limitations</summary>
<p>Scores compare 89 intervening frames (1–177) with GNT, not ground-truth photos. Higher PSNR/SSIM is better. Lower temporal residual means less change in the error relative to genuine GNT; it is not a perceptual smoothness guarantee. Final frame 179 has no future anchor and is scored separately in results.json. The fixed splat model already trained on some views here. Indoor and dynamic scenes remain untested.</p>
<table><tr><th>Method</th><th>Gap PSNR</th><th>Gap SSIM</th><th>Temporal residual MAE</th></tr>__ROWS__</table><a href="__PREFIX__results.json">Full results / provenance</a></details>
<script>const videos=[...document.querySelectorAll('video')],prefix='__PREFIX__';
function label(){const i=Math.min(179,Math.floor(videos[0].currentTime*30+1e-6));document.getElementById('frame').textContent='Frame '+i+' · '+(i%2===0?'GNT anchor':'intermediate frame');}
document.getElementById('play').onclick=async()=>{videos.forEach(v=>{v.pause();v.currentTime=0;});await Promise.all(videos.map(v=>v.play()));};
document.getElementById('pause').onclick=()=>{videos.forEach(v=>v.pause());label();};
document.getElementById('seek').oninput=e=>{videos.forEach(v=>{v.pause();v.currentTime=+e.target.value;});label();};
document.getElementById('speed').onchange=e=>videos.forEach(v=>v.playbackRate=+e.target.value);
document.getElementById('variant').onchange=e=>{const t=videos[0].currentTime;videos.forEach(v=>v.pause());const v=videos[2];v.src=prefix+e.target.value+'.mp4';v.load();v.currentTime=t;v.previousElementSibling.textContent='3. '+e.target.selectedOptions[0].textContent;};
videos[0].addEventListener('seeked',label);videos[0].addEventListener('pause',label);label();</script></html>'''
    for a,b in [('__OPTIONS__',options),('__CARDS__',cards),('__ROWS__',rows),('__PREFIX__',prefix)]: page=page.replace(a,b)
    (ROOT/'web_demo/gnt-splat-correction.html').write_text(page,encoding='utf-8')


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--ffmpeg',default=r'C:\ffmpeg\bin\ffmpeg.exe');args=parser.parse_args()
    folder=args.root/'assets/splat_correction';folder.mkdir(parents=True,exist_ok=True)
    gnt=args.root/'assets/dense_turn';splat=args.root/'assets/balanced_rollout/scene'
    progress=json.loads((gnt/'progress.json').read_text());evaluation=json.loads((splat/'balanced90_evaluation.json').read_text())
    assert progress['status']==evaluation['status']=='complete'
    rows=[r for r in evaluation['rows'] if r['group']=='dense'];assert [r['index'] for r in rows]==list(range(180))
    gt_paths=[gnt/r['file'] for r in progress['frames']];sp_paths=[splat/r['file'] for r in rows]
    for i in range(180):
        assert digest(gt_paths[i])==progress['frames'][i]['sha256']==rows[i]['target_sha256']
    read=lambda p: cv2.imread(str(p))
    # Construction loads only even GNT anchors. Odd views remain sealed until evaluation.
    anchors={i:read(gt_paths[i]) for i in range(0,180,2)}
    fits={i:colour_fit(read(sp_paths[i]),anchors[i]) for i in anchors}
    residuals={i:anchors[i].astype(np.float32)-apply_colour(read(sp_paths[i]),fits[i]) for i in anchors}
    baseline=args.root/'assets/rate_decision_15/fps15_motion.mp4'
    if not baseline.exists():
        previous=json.loads((args.root/'assets/rate_decision_15/results.json').read_text())
        baseline=args.root/'assets/rate_decision_15'/previous['videos']['fps15']['file']
    originals={'reference':gnt/'scene_turn_genuine_30fps.mp4','motion':baseline,
               'raw':args.root/'assets/splat_interleaved/fps15_splat.mp4'}
    for key,path in originals.items():
        target=folder/f'{key}.mp4'
        if not target.exists():
            try:target.hardlink_to(path)
            except OSError:shutil.copy2(path,target)
        assert digest(target)==digest(path)
    report=dict(status='running',teacher_checkpoint_sha256=progress['checkpoint_sha256'],
                splat_checkpoint_sha256=evaluation['checkpoint_sha256'],
                construction_gnt_indices=list(anchors),evaluation_gap_indices=list(range(1,178,2)),
                settings=dict(flow='OpenCV DIS medium, half-resolution',gain_limits=[.8,1.2],bias_limits=[-20,20],residual_limit=60,fb_scale=1.5,photo_scale=25),
                source_hashes=dict(gnt=[digest(p) for p in gt_paths],splat=[digest(p) for p in sp_paths]),scores={})
    save(folder/'results.json',report)
    encoders={};started=time.perf_counter();cap=cv2.VideoCapture(str(baseline));dis=cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_MEDIUM)
    cv2.setNumThreads(4)
    try:
        for key in ('colour','detail','fallback'):
            encoders[key]=subprocess.Popen([args.ffmpeg,'-hide_banner','-loglevel','error','-y','-threads','2','-f','rawvideo','-pix_fmt','bgr24','-s','960x540','-r','30','-i','pipe:0','-an','-c:v','libx264','-threads','2','-preset','fast','-crf','16','-pix_fmt','yuv420p','-movflags','+faststart',str(folder/f'{key}.partial.mp4')],stdin=subprocess.PIPE)
        confidence_rows=[]
        for i in range(180):
            ok,motion=cap.read();assert ok
            if i%2==0:
                outputs={key:anchors[i] for key in encoders}
            else:
                current=read(sp_paths[i]);lo=i-1;hi=min(i+1,178)
                fit=tuple((a+b)/2 for a,b in zip(fits[lo],fits[hi]));colour=apply_colour(current,fit)
                if hi==lo:
                    detail=colour;confidence=np.zeros((540,960),np.float32)
                else:
                    ra,ca=flow_to_anchor(current,read(sp_paths[lo]),residuals[lo],dis)
                    rb,cb=flow_to_anchor(current,read(sp_paths[hi]),residuals[hi],dis)
                    weight=ca+cb
                    correction=(ra*ca[...,None]+rb*cb[...,None])/np.maximum(weight[...,None],1e-6)
                    detail=np.clip(colour+np.clip(correction,-60,60)*np.minimum(weight,1)[...,None],0,255)
                    disagreement=np.mean(np.abs(ra-rb),axis=2)
                    confidence=np.minimum(ca,cb)*np.exp(-(disagreement/30)**2)
                    confidence=cv2.GaussianBlur(confidence,(0,0),2)
                fallback=detail*confidence[...,None]+motion.astype(np.float32)*(1-confidence[...,None])
                outputs=dict(colour=colour,detail=detail,fallback=fallback)
                confidence_rows.append(dict(index=i,mean_splat_weight=float(confidence.mean()),pixels_weight_over_half=float(np.mean(confidence>.5))))
            for key,image in outputs.items():encoders[key].stdin.write(np.clip(np.rint(image),0,255).astype(np.uint8).tobytes())
            if i%20==0:print(json.dumps(dict(phase='correcting',frame=i,elapsed=time.perf_counter()-started)),flush=True)
        assert not cap.read()[0]
        for key,process in encoders.items():
            process.stdin.close();assert process.wait()==0
            temp=folder/f'{key}.partial.mp4';verify(temp,6.);temp.replace(folder/f'{key}.mp4')
    finally:
        cap.release()
        for process in encoders.values():
            if process.poll() is None:process.kill();process.wait()
    report.update(correction_seconds=time.perf_counter()-started,confidence=confidence_rows)
    # Evaluate only after construction; no withheld GNT informs fits, masks or fallback.
    for key in ('raw','motion','colour','detail','fallback'):
        cap=cv2.VideoCapture(str(folder/f'{key}.mp4'));metrics=[];temporal=[];prev_error=None;anchor_psnr=[]
        for i in range(180):
            ok,image=cap.read();assert ok
            target=read(gt_paths[i]);error=image.astype(np.float32)-target.astype(np.float32)
            row=dict(index=i,psnr=float(cv2.PSNR(image,target)))
            if i%2:row['ssim']=float(structural_similarity(image,target,channel_axis=2,data_range=255))
            else:anchor_psnr.append(row['psnr'])
            if prev_error is not None:temporal.append(float(np.mean(np.abs(error-prev_error))))
            prev_error=error;metrics.append(row)
        assert not cap.read()[0];cap.release();gaps=[metrics[i] for i in range(1,178,2)]
        report['scores'][key]=dict(**verify(folder/f'{key}.mp4',6.),gap_psnr=float(np.mean([r['psnr'] for r in gaps])),gap_ssim=float(np.mean([r['ssim'] for r in gaps])),tail_psnr=metrics[-1]['psnr'],temporal_residual_mae=float(np.mean(temporal[:178])),minimum_anchor_psnr=min(anchor_psnr),rows=metrics)
        assert min(anchor_psnr)>32
        save(folder/'results.json',report);print(json.dumps(dict(method=key,**{k:v for k,v in report['scores'][key].items() if k not in ('rows',)})),flush=True)
    report['status']='complete';save(folder/'results.json',report);save(ROOT/'outputs/gnt_finalization/splat_correction.json',report);publish(folder,report)
    print('Complete: http://localhost:8088/gnt-splat-correction.html',flush=True)


if __name__=='__main__':main()
