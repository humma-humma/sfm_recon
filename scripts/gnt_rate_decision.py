"""Subsample genuine 30fps GNT and evaluate interpolation on withheld views."""
import argparse
import json
from pathlib import Path
import shutil
import subprocess
import urllib.request
import cv2
import numpy as np
from skimage.metrics import structural_similarity
from gnt_rollout_io import digest,save
from publish_gnt_videos import verify
from gnt_rate_sampling import split_frames

ROOT=Path(__file__).resolve().parents[1]


def encode(ffmpeg,folder,output,fps,method):
    pad=2/fps
    interpolation='fps=30' if method=='held' else 'minterpolate=fps=30:mi_mode=mci:mc_mode=aobmc:me_mode=bidir:vsbmc=1:scd=fdiff:scd_threshold=10'
    filters=f'tpad=start_mode=clone:start_duration={pad:.9f}:stop_mode=clone:stop_duration={3/fps:.9f},{interpolation},trim=start={pad:.9f}:duration=6,setpts=PTS-STARTPTS'
    temp=output.with_suffix('.partial.mp4')
    subprocess.run([ffmpeg,'-hide_banner','-loglevel','error','-y','-threads','4','-framerate',str(fps),'-i',str(folder/'anchor_%04d.png'),
                    '-vf',filters,'-frames:v','180','-an','-c:v','libx264','-preset','fast','-crf','16','-pix_fmt','yuv420p','-movflags','+faststart',str(temp)],check=True)
    verify(temp,6.);temp.replace(output)
    return filters


def evaluate(path,source,anchors,common):
    cap=cv2.VideoCapture(str(path));rows=[];previous=None;near_duplicates=[]
    for i in range(180):
        ok,image=cap.read();assert ok,(path,i)
        reference=cv2.imread(str(source/f'frame_{i:04}.png'));assert reference.shape==image.shape==(540,960,3)
        mse=float(np.mean(((image.astype(np.float32)-reference)/255)**2))
        row=dict(index=i,psnr=float(-10*np.log10(max(mse,1e-16))))
        if i in common:row['ssim']=float(structural_similarity(image,reference,channel_axis=2,data_range=255))
        rows.append(row)
        if previous is not None:near_duplicates.append(float(np.mean(abs(image.astype(np.float32)-previous)))<.5)
        previous=image.astype(np.float32)
    assert not cap.read()[0];cap.release()
    anchor_min=min(rows[i]['psnr'] for i in anchors)
    assert anchor_min>32,('Anchor timing or fidelity mismatch',path,anchor_min)
    return dict(rows=rows,common_withheld=dict(count=len(common),psnr=float(np.mean([rows[i]['psnr'] for i in common])),
                ssim=float(np.mean([rows[i]['ssim'] for i in common]))),minimum_anchor_psnr=anchor_min,
                near_duplicate_fraction=float(np.mean(near_duplicates)))


def publish(folder,report,rates=(5,10),asset_folder='rate_decision',page_name='gnt-rate-decision.html'):
    prefix=f'assets/gnt_finalization/{asset_folder}/'
    labels=[('reference','1. Genuine GNT — 30 independent views/s')]+[(f'fps{rate}',f'{i+2}. GNT at {rate} views/s → motion compensation to 30 fps') for i,rate in enumerate(rates)]
    cards=''.join(f'<figure><figcaption>{title}</figcaption><video id="{label}" controls muted playsinline preload="auto" src="{prefix+report["videos"][label]["file"]}"></video><a id="{label}_download" href="{prefix+report["videos"][label]["file"]}" download>Download MP4</a></figure>' for label,title in labels)
    table=''.join(f'<tr><td>{label}</td><td>{r["common_withheld"]["psnr"]:.2f}</td><td>{r["common_withheld"]["ssim"]:.4f}</td></tr>' for label,r in report['videos'].items())
    page='''<!doctype html><html><meta charset="utf-8"><title>GNT 5 / 10 fps decision test</title>
<style>body{max-width:1700px;margin:24px auto;padding:0 16px;background:#121720;color:#eee;font:16px system-ui}a{color:#8cf}.grid{display:grid;grid-template-columns:repeat(3,1fr);gap:14px}figure{margin:0}figcaption{height:70px;font-weight:600;display:flex;align-items:center}video{width:100%;display:block}button,select{padding:9px;margin:6px}input{width:300px}td,th{padding:8px;text-align:left}@media(max-width:750px){.grid{grid-template-columns:1fr}}</style>
<h1>Can 5–10 GNT views/s replace dense 30 fps rendering?</h1>
<p>Same six-second outdoor turn, 960 × 540. The candidates use only 30 or 60 genuine GNT anchor frames. Their intermediate frames never enter interpolation. No splats or new GNT rendering are used.</p>
<p>Left: genuine GNT reference. Middle: 5 views/s. Right: 10 views/s. Motion compensation uses the previously tested FFmpeg settings, with scene-change threshold 10. The final 5 frames at 5 views/s and final 2 at 10 views/s are held because no later anchor exists.</p>
<button id="play">Play all from start</button><button id="pause">Pause all</button><label>Playback speed <select id="speed"><option value="1">1×</option><option value="0.5">0.5×</option><option value="0.25">0.25×</option></select></label>
<label>Seek together <input id="seek" type="range" min="0" max="5.966666667" step="0.033333333" value="0"></label>
<label>Candidate mode <select id="mode"><option value="motion">Motion compensated</option><option value="held">Held anchors (control)</option></select></label>
<div class="grid">__CARDS__</div>
<details><summary>Withheld-frame scores and test limits</summary><p>All methods are compared on the same 116 non-anchor frames before the shorter candidate's last anchor. Reference scores measure video compression error; candidate scores include interpolation and compression. These are agreement scores against GNT, not photographs. Boundary frames are excluded from the main scores. This outdoor turn does not establish indoor or moving-person performance.</p>
<table><tr><th>Video</th><th>Mean PSNR, dB</th><th>Mean SSIM</th></tr>__TABLE__</table></details>
<script>const videos=[...document.querySelectorAll('video')];
document.getElementById('play').onclick=async()=>{videos.forEach(v=>{v.pause();v.currentTime=0;});await Promise.all(videos.map(v=>v.play()));};
document.getElementById('pause').onclick=()=>videos.forEach(v=>v.pause());
document.getElementById('speed').onchange=e=>videos.forEach(v=>v.playbackRate=+e.target.value);
document.getElementById('seek').oninput=e=>videos.forEach(v=>{v.pause();v.currentTime=+e.target.value;});
document.getElementById('mode').onchange=e=>{const time=videos[0].currentTime;videos.forEach(v=>v.pause());for(const rate of __RATES__){const v=document.getElementById('fps'+rate);v.src='__PREFIX__fps'+rate+'_'+e.target.value+'.mp4';v.load();v.currentTime=time;document.getElementById('fps'+rate+'_download').href=v.src;v.previousElementSibling.textContent=(rate===__FIRST_RATE__?'2. ':'3. ')+'GNT at '+rate+' views/s → '+(e.target.value==='motion'?'motion compensation to 30 fps':'held anchors at 30 fps');}};
</script></html>'''.replace('__CARDS__',cards).replace('__TABLE__',table).replace('__PREFIX__',prefix)
    page=page.replace('__RATES__',json.dumps(rates)).replace('__FIRST_RATE__',str(rates[0]))
    if rates!=(5,10):
        page=page.replace('5 / 10 fps',f'{rates[0]} / {rates[1]} fps').replace('5–10 GNT',f'{rates[0]}–{rates[1]} GNT')
        page=page.replace('only 30 or 60 genuine','only 60 or 90 genuine').replace('Middle: 5 views/s. Right: 10 views/s.','Middle: 10 views/s. Right: 15 views/s.')
        page=page.replace('The final 5 frames at 5 views/s and final 2 at 10 views/s','The final 2 frames at 10 views/s and final 1 at 15 views/s')
        page=page.replace('same 116 non-anchor frames',f'same {len(report["common_withheld"])} non-anchor frames')
    path=ROOT/'web_demo'/page_name;path.write_text(page,encoding='utf-8')
    with urllib.request.urlopen('http://localhost:8088/'+page_name) as response:assert response.read()==path.read_bytes()


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--ffmpeg',default=r'C:\ffmpeg\bin\ffmpeg.exe');args=parser.parse_args()
    source=args.root/'assets/dense_turn';progress=json.loads((source/'progress.json').read_text())
    assert progress['status']=='complete' and len(progress['frames'])==180
    for i,row in enumerate(progress['frames']):
        assert row['index']==i and digest(source/row['file'])==row['sha256']
    folder=args.root/'assets/rate_decision';folder.mkdir(parents=True,exist_ok=True)
    splits={fps:split_frames(fps) for fps in (5,10)}
    common=sorted(set(splits[5]['withheld'])&set(splits[10]['withheld']));assert len(common)==116
    report=dict(status='running',checkpoint_sha256=progress['checkpoint_sha256'],common_withheld=common,splits=splits,videos={},
                note='Exploratory six-second outdoor-turn rate test. No new GNT or splat inference. Padding uses only repeated input anchors, never withheld images. FPS refers to anchors per second, not recovered scene information.')
    reference=folder/'reference.mp4'
    if not reference.exists():shutil.copy2(source/'scene_turn_genuine_30fps.mp4',reference)
    assert digest(reference)==digest(source/'scene_turn_genuine_30fps.mp4')
    report['videos']['reference']={**verify(reference,6.),**evaluate(reference,source,list(range(180)),common)}
    for fps in (5,10):
        inputs=folder/f'inputs_{fps}';inputs.mkdir(exist_ok=True)
        for j,i in enumerate(splits[fps]['anchors']):
            target=inputs/f'anchor_{j:04}.png'
            if not target.exists():shutil.copy2(source/f'frame_{i:04}.png',target)
            assert digest(target)==digest(source/f'frame_{i:04}.png')
        for method in ('motion','held'):
            label=f'fps{fps}' if method=='motion' else f'fps{fps}_held'
            output=folder/f'fps{fps}_{method}.mp4'
            filters=encode(args.ffmpeg,inputs,output,fps,method)
            row={**verify(output,6.),**evaluate(output,source,splits[fps]['anchors'],common),'anchor_fps':fps,'method':method,'filter':filters}
            row['rate_withheld']=dict(count=len(splits[fps]['withheld']),psnr=float(np.mean([row['rows'][i]['psnr'] for i in splits[fps]['withheld']])))
            row['held_tail_mean_psnr']=float(np.mean([row['rows'][i]['psnr'] for i in splits[fps]['held_tail']]))
            report['videos'][label]=row;save(folder/'results.json',report)
            print(json.dumps(dict(label=label,**row['common_withheld'],minimum_anchor_psnr=row['minimum_anchor_psnr'])),flush=True)
    report['status']='complete';save(folder/'results.json',report)
    save(ROOT/'outputs/gnt_finalization/rate_decision.json',report)
    publish(folder,report)
    print('Complete: http://localhost:8088/gnt-rate-decision.html',flush=True)


if __name__=='__main__':main()
