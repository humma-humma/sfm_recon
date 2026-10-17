"""Publish scene15 progress and encode native/30fps motion-compensated videos."""
import argparse
import json
from pathlib import Path
import subprocess
import urllib.request
import cv2
from gnt_rollout_io import digest,save

ROOT=Path(__file__).resolve().parents[1]


def verify(path,fps,frames):
    cap=cv2.VideoCapture(str(path));assert cap.isOpened(),str(path)
    assert cap.get(cv2.CAP_PROP_FPS)==fps and int(cap.get(cv2.CAP_PROP_FRAME_COUNT))==frames
    assert (cap.get(cv2.CAP_PROP_FRAME_WIDTH),cap.get(cv2.CAP_PROP_FRAME_HEIGHT))==(960,540)
    for i in (0,frames//2,frames-1):
        cap.set(cv2.CAP_PROP_POS_FRAMES,i);assert cap.read()[0]
    cap.release()
    return dict(file=path.name,sha256=digest(path),fps=fps,frames=frames,duration_seconds=frames/fps)


def encode(folder,report,ffmpeg):
    plan=json.loads((folder/'plan.json').read_text());assert report['plan_sha256']==digest(folder/'plan.json')
    count=plan['total'];assert count==2198 and len(report['frames'])==count
    for i in range(count):
        row=report['frames'][str(i)];assert row['index']==i and all(j%8 for j in row['sources'])
        path=folder/row['file'];assert digest(path)==row['sha256'] and cv2.imread(str(path)).shape==(540,960,3)
    export_path=folder/'exports.json'
    exports=json.loads(export_path.read_text()) if export_path.exists() else dict(plan_sha256=report['plan_sha256'],videos={})
    assert exports['plan_sha256']==report['plan_sha256']
    filters='tpad=start_mode=clone:start_duration=0.133333333:stop_mode=clone:stop_duration=0.2,minterpolate=fps=30:mi_mode=mci:mc_mode=aobmc:me_mode=bidir:vsbmc=1:scd=fdiff:scd_threshold=10,trim=start=0.133333333:duration=146.533333333,setpts=PTS-STARTPTS'
    for label,fps,frames in [('native',15,count),('motion',30,2*count)]:
        output=folder/f'scene15_{label}.mp4'
        if label in exports['videos']:
            assert verify(output,fps,frames)['sha256']==exports['videos'][label]['sha256'];continue
        temp=output.with_suffix('.partial.mp4')
        command=[ffmpeg,'-hide_banner','-loglevel','error','-y','-threads','4','-framerate','15','-i',str(folder/'frame_%04d.png')]
        if label=='motion':command+=['-vf',filters]
        command+=['-frames:v',str(frames),'-an','-c:v','libx264','-preset','fast','-crf','16','-pix_fmt','yuv420p','-movflags','+faststart',str(temp)]
        print(json.dumps(dict(phase='encoding',label=label,frames=frames)),flush=True)
        subprocess.run(command,check=True);info=verify(temp,fps,frames)
        cap=cv2.VideoCapture(str(temp));minimum=100.;decoded=0
        for i in range(frames):
            ok,image=cap.read();assert ok,(label,i)
            # Every native frame, and every preserved GNT anchor of the 30fps export.
            if label=='native' or i%2==0:
                target=cv2.imread(str(folder/f'frame_{i if label=="native" else i//2:04}.png'))
                minimum=min(minimum,float(cv2.PSNR(image,target)))
            decoded+=1
        assert not cap.read()[0];cap.release();assert minimum>32,(label,minimum)
        temp.replace(output);info['file']=output.name
        info.update(decoded_frames=decoded,minimum_anchor_psnr=minimum,frame_interpolation=label=='motion')
        if label=='motion':info.update(filter=filters,final_held_output_frames=1)
        exports['videos'][label]=info;save(export_path,exports)
    report.update(status='complete',videos=exports['videos']);save(folder/'progress.json',report)
    save(ROOT/'outputs/gnt_finalization/scene15.json',dict(status='complete',plan=plan,videos=exports['videos'],
         completed=report['completed'],reused=report['reused'],new_rendered=report['new_rendered']))


def publish(folder,report):
    ready=report.get('status')=='complete' and len(report.get('videos',{}))==2
    cards=[]
    for label,title in [('native','1. Genuine GNT — 15 independent views/s'),('motion','2. GNT at 15 views/s — motion compensated to 30 fps')]:
        if ready:
            url='assets/gnt_finalization/scene15/'+report['videos'][label]['file']
            cards.append(f'<section><h2>{title}</h2><video id="{label}" controls muted playsinline preload="metadata" src="{url}"></video><a href="{url}" download>Download MP4</a></section>')
    content=''.join(cards) if ready else '<p>The videos will appear automatically after rendering, encoding and verification finish.</p>'
    page='''<!doctype html><html><meta charset="utf-8"><title>Full scene — 15-view GNT</title>
<style>body{max-width:1400px;margin:24px auto;padding:0 16px;background:#121720;color:#eee;font:16px system-ui}a{color:#8cf}.grid{display:grid;grid-template-columns:repeat(2,1fr);gap:18px}video{width:100%;display:block}h2{font-size:18px;min-height:50px}button,select{padding:9px;margin:5px}input{width:300px}#preview{max-width:640px;width:100%}@media(max-width:800px){.grid{grid-template-columns:1fr}}</style>
<h1>Full scene — GNT at 15 views per second</h1><p>The captured scene route at 960 × 540. Native output contains 2,198 GNT views over 146.53 seconds; the second export uses the tested FFmpeg motion compensation to produce 30 fps playback. No splats are mixed in. This run extends the outdoor decision test to the full route; indoor and dynamic-scene interpolation quality still needs visual review.</p>
<p id="progress">Checking progress…</p><div id="preview_box"><p>Latest newly rendered view</p><img id="preview" alt="Latest GNT render"></div>
<button id="play">Play both from start</button><button id="pause">Pause both</button><label>Speed <select id="speed"><option value="1">1×</option><option value="0.5">0.5×</option><option value="0.25">0.25×</option></select></label>
<label>Seek together <input id="seek" type="range" min="0" max="146.5" step="0.033333333" value="0"></label>
<div class="grid">__CARDS__</div>
<script>const ready=__READY__,videos=[...document.querySelectorAll('video')],prefix='assets/gnt_finalization/scene15/';
async function status(){try{const r=await fetch(prefix+'progress.json',{cache:'no-store'});if(!r.ok)throw Error('pending');const p=await r.json();const eta=p.estimated_remaining_seconds>0?' · approximately '+(p.estimated_remaining_seconds/3600).toFixed(1)+' hours of rendering left':'';
document.getElementById('progress').textContent=p.status+' · '+p.completed+'/'+p.total+' views · '+p.reused+' reused'+eta;
if(p.latest_frame&&!ready){document.getElementById('preview_box').hidden=false;document.getElementById('preview').src=prefix+p.latest_frame;}else document.getElementById('preview_box').hidden=true;
if(p.status==='complete'&&!ready)location.reload();}catch(e){document.getElementById('progress').textContent='Queued / preflight';document.getElementById('preview_box').hidden=true;}}
status();setInterval(status,30000);
document.getElementById('play').disabled=document.getElementById('pause').disabled=!ready;
document.getElementById('play').onclick=async()=>{videos.forEach(v=>{v.pause();v.currentTime=0;});await Promise.all(videos.map(v=>v.play()));};
document.getElementById('pause').onclick=()=>videos.forEach(v=>v.pause());
document.getElementById('seek').oninput=e=>videos.forEach(v=>{v.pause();v.currentTime=+e.target.value;});
document.getElementById('speed').onchange=e=>videos.forEach(v=>v.playbackRate=+e.target.value);
</script></html>'''.replace('__CARDS__',content).replace('__READY__','true' if ready else 'false')
    path=ROOT/'web_demo/gnt-scene-15fps.html';path.write_text(page,encoding='utf-8')
    with urllib.request.urlopen('http://localhost:8088/gnt-scene-15fps.html') as response:assert response.read()==path.read_bytes()
    print(json.dumps(dict(status=report.get('status','pending'),url='http://localhost:8088/gnt-scene-15fps.html')),flush=True)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--ffmpeg',default=r'C:\ffmpeg\bin\ffmpeg.exe');args=parser.parse_args()
    folder=args.root/'assets/scene15';folder.mkdir(parents=True,exist_ok=True)
    report=json.loads((folder/'progress.json').read_text()) if (folder/'progress.json').exists() else dict(status='pending')
    if report['status'] in ('rendered','complete'):encode(folder,report,args.ffmpeg)
    publish(folder,report)


if __name__=='__main__':main()
