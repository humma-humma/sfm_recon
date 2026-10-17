"""Keep GNT anchors and fill gaps with native balanced-splat camera renders."""
import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import urllib.request
import cv2
import numpy as np
from skimage.metrics import structural_similarity
from gnt_rollout_io import digest,save
from gnt_rate_sampling import split_frames,interleaved_sources
from publish_gnt_videos import verify

ROOT=Path(__file__).resolve().parents[1]


def publish(report):
    prefix='assets/gnt_finalization/splat_interleaved/'
    labels=[('reference','1. Genuine GNT — 30 independent views/s','reference.mp4'),
            ('left','2. GNT at 10 views/s + native splat gap frames','fps10_splat.mp4'),
            ('right','3. GNT at 15 views/s + native splat gap frames','fps15_splat.mp4')]
    cards=[]
    for key,title,file in labels:
        selector='' if key=='reference' else f'<label>GNT views/s <select id="{key}_rate">'+''.join(f'<option value="{fps}"'+(' selected' if fps==(10 if key=='left' else 15) else '')+f'>{fps}</option>' for fps in (5,10,15))+'</select></label>'
        cards.append(f'<figure><figcaption id="{key}_title">{title}</figcaption>{selector}<video id="{key}" controls muted playsinline preload="auto" src="{prefix+file}"></video><a id="{key}_download" href="{prefix+file}" download>Download MP4</a></figure>')
    table=''.join(f'<tr><td>{rate}</td><td>{row["overall_psnr"]:.2f}</td><td>{row["gap_only_psnr"]:.2f}</td><td>{row["common_gap"]["psnr"]:.2f}</td><td>{row["common_gap"]["ssim"]:.4f}</td></tr>' for rate,row in report['videos'].items())
    page='''<!doctype html><html><meta charset="utf-8"><title>GNT anchors + native splat gap frames</title>
<style>body{max-width:1700px;margin:24px auto;padding:0 16px;background:#121720;color:#eee;font:16px system-ui}a{color:#8cf}.grid{display:grid;grid-template-columns:repeat(3,1fr);gap:14px}figure{margin:0}figcaption{height:70px;font-weight:600;display:flex;align-items:center}figure>label{display:block;height:40px}figure:first-child video{margin-top:40px}video{width:100%;display:block}button,select{padding:8px;margin:5px}input{width:280px}td,th{padding:8px;text-align:left}@media(max-width:750px){.grid{grid-template-columns:1fr}}</style>
<h1>GNT anchors with intermediate splats</h1><p>Same outdoor turning clip, 960 × 540 at 30 fps. GNT supplies the anchor frames; the existing balanced 90/10 splat supplies every other frame at its exact recorded camera pose. No interpolation, optical-flow warping, crossfade, blending or held gap frames.</p>
<p>Choose 5, 10 or 15 GNT views/s in either candidate panel. Look for changes in detail or brightness as playback switches between GNT and splat frames. The same fixed splat model is used at all rates; this is a playback-composition test, not new training at those rates.</p>
<button id="play">Play all from start</button><button id="pause">Pause all</button><label>Playback speed <select id="speed"><option value="1">1×</option><option value="0.5">0.5×</option><option value="0.25">0.25×</option></select></label>
<label>Seek together <input id="seek" type="range" min="0" max="5.966666667" step="0.033333333" value="0"></label>
<p id="frame"></p><div class="grid">__CARDS__</div>
<details><summary>Scores and limitations</summary><p>Gap-only scores measure the splat frames, without GNT anchors inflating the average. The shared gap comparison uses 58 frames that are splat-filled at all three rates. Those frames have the same source images at every rate. Overall scores also include GNT anchors. Agreement is measured against GNT, not photographs; the splat has already trained on some views in this clip. This outdoor test does not establish indoor/dynamic-scene quality.</p>
<table><tr><th>GNT views/s</th><th>Overall PSNR</th><th>Gap-only PSNR</th><th>Shared-gap PSNR</th><th>Shared-gap SSIM</th></tr>__TABLE__</table></details>
<script>const videos=[...document.querySelectorAll('video')],prefix='__PREFIX__';
function sourceLabel(){const i=Math.min(179,Math.floor(videos[0].currentTime*30+1e-6));document.getElementById('frame').textContent='Frame '+i+' — '+['left','right'].map(key=>{const rate=+document.getElementById(key+'_rate').value;return rate+' views/s: '+(i%(30/rate)===0?'GNT anchor':'native splat');}).join(' · ');}
document.getElementById('play').onclick=async()=>{videos.forEach(v=>{v.pause();v.currentTime=0;});await Promise.all(videos.map(v=>v.play()));};
document.getElementById('pause').onclick=()=>{videos.forEach(v=>v.pause());sourceLabel();};
document.getElementById('speed').onchange=e=>videos.forEach(v=>v.playbackRate=+e.target.value);
document.getElementById('seek').oninput=e=>{videos.forEach(v=>{v.pause();v.currentTime=+e.target.value;});sourceLabel();};
for(const key of ['left','right'])document.getElementById(key+'_rate').onchange=e=>{const time=videos[0].currentTime;videos.forEach(v=>v.pause());const v=document.getElementById(key);v.src=prefix+'fps'+e.target.value+'_splat.mp4';v.load();v.currentTime=time;document.getElementById(key+'_download').href=v.src;document.getElementById(key+'_title').textContent=(key==='left'?'2. ':'3. ')+'GNT at '+e.target.value+' views/s + native splat gap frames';sourceLabel();};
videos[0].addEventListener('play',()=>document.getElementById('frame').textContent='Playing — pause to inspect the frame source.');videos[0].addEventListener('pause',sourceLabel);videos[0].addEventListener('seeked',sourceLabel);sourceLabel();
</script></html>'''.replace('__CARDS__',''.join(cards)).replace('__TABLE__',table).replace('__PREFIX__',prefix)
    path=ROOT/'web_demo/gnt-splat-interleaved.html';path.write_text(page,encoding='utf-8')
    with urllib.request.urlopen('http://localhost:8088/gnt-splat-interleaved.html') as response:assert response.read()==path.read_bytes()


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--ffmpeg',default=r'C:\ffmpeg\bin\ffmpeg.exe');args=parser.parse_args()
    gnt=args.root/'assets/dense_turn';splat=args.root/'assets/balanced_rollout/scene'
    progress=json.loads((gnt/'progress.json').read_text());evaluation=json.loads((splat/'balanced90_evaluation.json').read_text())
    checkpoint=args.root/'gnt_dense_supervision/models/balanced90/splatfacto/continued/nerfstudio_models/step-000011999.ckpt'
    assert evaluation['status']==progress['status']=='complete' and digest(checkpoint)==evaluation['checkpoint_sha256']
    rows=[r for r in evaluation['rows'] if r['group']=='dense'];assert [r['index'] for r in rows]==list(range(180))
    gnt_files=[];splat_files=[]
    for i,row in enumerate(rows):
        target=gnt/progress['frames'][i]['file'];fit=splat/row['file']
        assert digest(target)==row['target_sha256']==progress['frames'][i]['sha256']
        assert cv2.imread(str(fit)).shape==cv2.imread(str(target)).shape==(540,960,3)
        gnt_files.append(target);splat_files.append(fit)
    folder=args.root/'assets/splat_interleaved';folder.mkdir(parents=True,exist_ok=True)
    reference=folder/'reference.mp4'
    if not reference.exists():shutil.copy2(gnt/'scene_turn_genuine_30fps.mp4',reference)
    assert digest(reference)==digest(gnt/'scene_turn_genuine_30fps.mp4')
    common=sorted(set.intersection(*(set(split_frames(rate)['withheld']) for rate in (5,10,15))));assert len(common)==58
    report=dict(status='running',teacher_checkpoint_sha256=progress['checkpoint_sha256'],splat_checkpoint_sha256=evaluation['checkpoint_sha256'],
                common_gap_indices=common,videos={},interpolation=False,blending=False,new_training=False,
                note='Same pre-existing balanced splat model at all rates; some source views already used in fitting. Exact-pose native splat frames fill every gap, including the tail. Playback composition only, not an independent sparse-supervision generalization benchmark.')
    for rate in (5,10,15):
        inputs=folder/f'inputs_{rate}';inputs.mkdir(exist_ok=True);sources=interleaved_sources(rate);provenance=[]
        for i,kind in enumerate(sources):
            source=gnt_files[i] if kind=='gnt' else splat_files[i];target=inputs/f'frame_{i:04}.png'
            if not target.exists():
                try:os.link(source,target)
                except OSError:shutil.copy2(source,target)
            assert digest(target)==digest(source)
            provenance.append(dict(index=i,source=kind,source_sha256=digest(source)))
        output=folder/f'fps{rate}_splat.mp4';temp=output.with_suffix('.partial.mp4')
        subprocess.run([args.ffmpeg,'-hide_banner','-loglevel','error','-y','-threads','4','-framerate','30','-i',str(inputs/'frame_%04d.png'),
                        '-frames:v','180','-an','-c:v','libx264','-preset','fast','-crf','16','-pix_fmt','yuv420p','-movflags','+faststart',str(temp)],check=True)
        verify(temp,6.);temp.replace(output)
        cap=cv2.VideoCapture(str(output));metrics=[];minimum_source_psnr=100.
        for i,kind in enumerate(sources):
            ok,image=cap.read();assert ok
            target=cv2.imread(str(gnt_files[i]));selected=cv2.imread(str(inputs/f'frame_{i:04}.png'))
            source_psnr=float(cv2.PSNR(image,selected));minimum_source_psnr=min(minimum_source_psnr,source_psnr)
            mse=float(np.mean(((image.astype(np.float32)-target)/255)**2));row=dict(index=i,source=kind,psnr=float(-10*np.log10(max(mse,1e-16))))
            if i in common:row['ssim']=float(structural_similarity(image,target,channel_axis=2,data_range=255))
            metrics.append(row)
        assert not cap.read()[0];cap.release();assert minimum_source_psnr>32,minimum_source_psnr
        gaps=[r for r in metrics if r['source']=='splat']
        item={**verify(output,6.),'gnt_frames':sources.count('gnt'),'splat_frames':sources.count('splat'),
              'minimum_selected_source_psnr':minimum_source_psnr,'overall_psnr':float(np.mean([r['psnr'] for r in metrics])),
              'gap_only_psnr':float(np.mean([r['psnr'] for r in gaps])),
              'common_gap':dict(count=len(common),psnr=float(np.mean([metrics[i]['psnr'] for i in common])),ssim=float(np.mean([metrics[i]['ssim'] for i in common]))),
              'provenance':provenance,'rows':metrics}
        report['videos'][str(rate)]=item;save(folder/'results.json',report)
        print(json.dumps(dict(rate=rate,gnt_frames=item['gnt_frames'],splat_frames=item['splat_frames'],overall_psnr=item['overall_psnr'],gap_only_psnr=item['gap_only_psnr'],common_gap=item['common_gap'])),flush=True)
    report['status']='complete';save(folder/'results.json',report);save(ROOT/'outputs/gnt_finalization/splat_interleaved.json',report);publish(report)
    print('Complete: http://localhost:8088/gnt-splat-interleaved.html',flush=True)


if __name__=='__main__':main()
