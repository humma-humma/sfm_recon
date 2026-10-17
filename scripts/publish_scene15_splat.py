"""Apply the tested anchor correction to a complete, matched splat route."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
import subprocess
import time
import cv2
import numpy as np
from skimage.metrics import structural_similarity
from gnt_splat_correction import colour_fit, apply_colour, flow_to_anchor
from gnt_rollout_io import digest, save
from publish_gnt_scene15 import verify

ROOT=Path(__file__).resolve().parents[1]


def publish(folder, report):
    prefix='assets/gnt_finalization/scene15_splat/'
    ready=report['status']=='complete'
    cards=''.join(f'<figure><figcaption>{title}</figcaption><video id="{key}" controls muted playsinline preload="metadata" src="{prefix}{file}.mp4"></video><a href="{prefix}{file}.mp4" download>Download</a></figure>' for key,title,file in
        [('native','1. Genuine GNT — 15 independent views/s','native'),
         ('motion','2. GNT at 15 views/s — motion compensated to 30 fps','motion'),
         ('splats','3. GNT at 15 views/s — improved splats + reliability fallback to 30 fps','fallback')]) if ready else '<p>Videos appear after correction, encoding and verification.</p>'
    timing=''
    if ready:
        t=report['timings'];timing=f'<p>All versions share the same GNT anchors: {t["gnt_new_views"]} newly rendered and {t["gnt_reused_views"]} reused. Their cost is shared, so splat processing does not replace that GNT work. Original MP4-ready times come from job-start and artifact timestamps; isolated original encoding times were not logged. Existing model training and the earlier cost of cached GNT views are excluded.</p>'
        timing='<table><tr><th>Timing</th><th>Measured time</th></tr>'+''.join(f'<tr><td>{label}</td><td>{value:.2f} {unit}</td></tr>' for label,value,unit in [
            ('Shared GNT renderer calls',t['gnt_recorded_render_seconds']/3600,'hours'),
            ('Original job: native GNT MP4 ready',t['original_native_ready_wall_seconds']/3600,'hours'),
            ('Original job: motion-compensated MP4 ready',t['original_motion_ready_wall_seconds']/3600,'hours'),
            ('Additional splat renderer calls',t['splat_render_seconds']/60,'minutes'),
            ('Splat setup + renderer + lossless export',t['splat_wall_seconds']/60,'minutes'),
            ('Correction + both corrected MP4 encodes',t['correction_two_encodes_seconds']/60,'minutes'),
            ('Full video verification',t['verification_seconds']/60,'minutes')])+'</table>'+timing
    page='''<!doctype html><html><meta charset="utf-8"><title>Full scene: GNT / interpolation / improved splats</title>
<style>body{max-width:1800px;margin:24px auto;padding:0 16px;background:#121720;color:#eee;font:16px system-ui}a{color:#8cf}.grid{display:grid;grid-template-columns:repeat(3,1fr);gap:14px}figure{margin:0}figcaption{min-height:85px;font-weight:600}video{width:100%;display:block}button,select{padding:8px;margin:5px}input{width:300px}td,th{padding:6px 16px 6px 0;text-align:left}@media(max-width:800px){.grid{grid-template-columns:1fr}}</style>
<h1>Full scene: 15fps GNT to 30fps</h1><p>Same 146.53-second captured route, 960 × 540. Native GNT has 2,198 genuine views; both 30fps videos have 4,396 frames. Corrected splats transfer appearance from neighbouring GNT anchors using splat optical flow. The default third panel falls back to existing motion interpolation where alignment is unreliable. Choose corrected splats only to see the version without interpolation fallback.</p>
<p id="progress">Checking progress…</p>__TIMING__
<button id="play">Play all from start</button><button id="pause">Pause all</button>
<label>Speed <select id="speed"><option value="1">1×</option><option value="0.5">0.5×</option><option value="0.25">0.25×</option></select></label>
<label>Third panel <select id="variant"><option value="fallback">Improved splats + reliability fallback</option><option value="detail">Corrected splats only — no interpolation fallback</option></select></label>
<label>Seek together <input id="seek" type="range" min="0" max="146.5" step="0.033333333" value="0"></label><p id="frame"></p>
<div class="grid">__CARDS__</div><p>There is no genuine 30fps GNT reference for the full route. The results file reports fidelity at every existing GNT anchor and gap-frame agreement only on the previously rendered six-second outdoor clip. Indoor and dynamic quality requires visual inspection; splat correction does not create a dynamic scene model. Last output frame has no future GNT anchor.</p><a href="__PREFIX__results.json">Results and timing details</a>
<script>const ready=__READY__,videos=[...document.querySelectorAll('video')],prefix='__PREFIX__';
async function status(){const r=await fetch(prefix+'comparison_progress.json',{cache:'no-store'});const p=await r.json();if(p.phase==='splat rendering'){try{const s=await(await fetch(prefix+'render_progress.json',{cache:'no-store'})).json();p.completed=s.completed;}catch(e){}}document.getElementById('progress').textContent=p.status+' · '+p.phase+' · '+(p.completed||0)+'/'+(p.total||4396)+' frames';if(p.status==='complete'&&!ready)location.reload();}status();setInterval(status,30000);
function label(){document.getElementById('frame').textContent='Time '+(videos[0]?.currentTime||0).toFixed(2)+' s';}
for(const id of ['play','pause','variant','seek'])document.getElementById(id).disabled=!ready;
document.getElementById('play').onclick=async()=>{videos.forEach(v=>{v.pause();v.currentTime=0;});await Promise.all(videos.map(v=>v.play()));};
document.getElementById('pause').onclick=()=>{videos.forEach(v=>v.pause());label();};
document.getElementById('seek').oninput=e=>{videos.forEach(v=>{v.pause();v.currentTime=+e.target.value;});label();};
document.getElementById('speed').onchange=e=>videos.forEach(v=>v.playbackRate=+e.target.value);
document.getElementById('variant').onchange=e=>{const t=videos[0].currentTime;videos.forEach(v=>v.pause());const v=videos[2];v.src=prefix+e.target.value+'.mp4';v.load();v.currentTime=t;v.previousElementSibling.textContent='3. GNT at 15 views/s — '+e.target.selectedOptions[0].textContent+' to 30 fps';v.nextElementSibling.href=v.src;};
</script></html>'''
    for key,value in [('__CARDS__',cards),('__TIMING__',timing),('__PREFIX__',prefix),('__READY__','true' if ready else 'false')]:page=page.replace(key,value)
    (ROOT/'web_demo/gnt-scene-15fps-splats.html').write_text(page,encoding='utf-8')


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--root',type=Path,required=True);parser.add_argument('--publish-only',action='store_true')
    parser.add_argument('--ffmpeg',default=r'C:\ffmpeg\bin\ffmpeg.exe');args=parser.parse_args()
    folder=args.root/'assets/scene15_splat';folder.mkdir(parents=True,exist_ok=True)
    progress_path=folder/'comparison_progress.json'
    if args.publish_only:
        report=json.loads((folder/'results.json').read_text()) if (folder/'results.json').exists() else dict(status='running',phase='splat rendering',completed=0,total=4396)
        if not progress_path.exists():save(progress_path,report)
        publish(folder,report);return
    native=args.root/'assets/scene15';plan=json.loads((native/'plan.json').read_text());native_progress=json.loads((native/'progress.json').read_text())
    renders=json.loads((folder/'render_progress.json').read_text());splat_plan=json.loads((folder/'splat_plan.json').read_text())
    assert native_progress['status']==renders['status']=='complete';assert renders['total']==4396 and plan['total']==2198
    assert splat_plan['native_plan_sha256']==digest(native/'plan.json')==native_progress['plan_sha256']
    count=renders['total'];n=plan['total'];duration=count/30
    for i in range(n):assert digest(native/native_progress['frames'][str(i)]['file'])==native_progress['frames'][str(i)]['sha256']
    for i in range(count):assert digest(folder/renders['frames'][str(i)]['file'])==renders['frames'][str(i)]['sha256']
    for label in ('native','motion'):
        source=native/f'scene15_{label}.mp4';target=folder/f'{label}.mp4'
        if not target.exists():
            try:target.hardlink_to(source)
            except OSError:shutil.copy2(source,target)
        assert digest(target)==digest(source)==native_progress['videos'][label]['sha256']
    report=dict(status='running',phase='correction',completed=0,total=count,native_plan_sha256=digest(native/'plan.json'),
        teacher_checkpoint_sha256=plan['checkpoint_sha256'],splat_checkpoint_sha256=splat_plan['checkpoint_sha256'],
        anchor_count=n,gap_count=n,settings=dict(flow='OpenCV DIS medium at half resolution',gain_limits=[.8,1.2],bias_limits=[-20,20],residual_limit=60,fb_scale=1.5,photo_scale=25),timings={})
    save(progress_path,report);publish(folder,report)
    read=lambda path:cv2.imread(str(path))
    sp=lambda i:read(folder/renders['frames'][str(i)]['file'])
    gt=lambda j:read(native/native_progress['frames'][str(j)]['file'])
    def bundle(j):
        image=gt(j);splat=sp(2*j);fit=colour_fit(splat,image)
        return dict(gnt=image,splat=splat,fit=fit,residual=image.astype(np.float32)-apply_colour(splat,fit))
    dis=cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_MEDIUM);cv2.setNumThreads(4)
    cap=cv2.VideoCapture(str(folder/'motion.mp4'));encoders={};started=time.perf_counter();compute_seconds=0.;weights=[]
    try:
        for key in ('detail','fallback'):
            encoders[key]=subprocess.Popen([args.ffmpeg,'-hide_banner','-loglevel','error','-y','-threads','2','-f','rawvideo','-pix_fmt','bgr24','-s','960x540','-r','30','-i','pipe:0','-an','-c:v','libx264','-threads','2','-preset','fast','-crf','16','-pix_fmt','yuv420p','-movflags','+faststart',str(folder/f'{key}.partial.mp4')],stdin=subprocess.PIPE)
        left=bundle(0);right=bundle(1)
        for j in range(n):
            tick=time.perf_counter()
            if j:
                left=right;right=bundle(min(j+1,n-1))
            ok,motion=cap.read();assert ok
            for process in encoders.values():process.stdin.write(left['gnt'].tobytes())
            ok,motion=cap.read();assert ok
            current=sp(2*j+1);fit=tuple((a+b)/2 for a,b in zip(left['fit'],right['fit']));colour=apply_colour(current,fit)
            if j==n-1:
                detail=colour;confidence=np.zeros((540,960),np.float32)
            else:
                ra,ca=flow_to_anchor(current,left['splat'],left['residual'],dis)
                rb,cb=flow_to_anchor(current,right['splat'],right['residual'],dis)
                weight=ca+cb;correction=(ra*ca[...,None]+rb*cb[...,None])/np.maximum(weight[...,None],1e-6)
                detail=np.clip(colour+np.clip(correction,-60,60)*np.minimum(weight,1)[...,None],0,255)
                confidence=np.minimum(ca,cb)*np.exp(-(np.mean(np.abs(ra-rb),axis=2)/30)**2)
                confidence=cv2.GaussianBlur(confidence,(0,0),2)
            fallback=detail*confidence[...,None]+motion.astype(np.float32)*(1-confidence[...,None])
            for key,image in [('detail',detail),('fallback',fallback)]:encoders[key].stdin.write(np.clip(np.rint(image),0,255).astype(np.uint8).tobytes())
            compute_seconds+=time.perf_counter()-tick;weights.append(float(confidence.mean()))
            if (j+1)%20==0 or j==n-1:
                elapsed=time.perf_counter()-started;report.update(completed=2*(j+1),elapsed_seconds=elapsed,estimated_remaining_seconds=elapsed/(j+1)*(n-j-1))
                save(progress_path,report);print(json.dumps({k:report[k] for k in ('phase','completed','total','elapsed_seconds','estimated_remaining_seconds')}),flush=True)
        assert not cap.read()[0]
        for key,process in encoders.items():
            process.stdin.close();assert process.wait()==0
            temp=folder/f'{key}.partial.mp4';verify(temp,30,count);temp.replace(folder/f'{key}.mp4')
    finally:
        cap.release()
        for process in encoders.values():
            if process.poll() is None:process.kill();process.wait()
    report['timings']=dict(gnt_recorded_render_seconds=sum(r['seconds'] for r in native_progress['frames'].values() if not r['reused']),gnt_new_views=native_progress['new_rendered'],gnt_reused_views=native_progress['reused'],splat_render_seconds=renders['render_seconds'],splat_wall_seconds=renders['current_run_wall_seconds'],correction_two_encodes_seconds=time.perf_counter()-started,correction_loop_including_pipe_writes_seconds=compute_seconds)
    original_job=json.loads((native/'job.json').read_text(encoding='utf-8-sig'))
    job_start=datetime.fromisoformat(original_job['started_utc'][:26]).replace(tzinfo=timezone.utc).timestamp()
    report['timings'].update(original_native_ready_wall_seconds=(native/'scene15_native.mp4').stat().st_mtime-job_start,original_motion_ready_wall_seconds=(native/'scene15_motion.mp4').stat().st_mtime-job_start,original_job_start_utc=original_job['started_utc'])
    report.update(phase='verification',mean_splat_gap_weight=float(np.mean(weights)),gap_weights=weights);save(progress_path,report)
    check_started=time.perf_counter();dense=args.root/'assets/dense_turn';dense_progress=json.loads((dense/'progress.json').read_text())
    # Only the existing independent 30fps outdoor clip provides gap-frame reference.
    sample={}
    for idx in range(1,178,2):
        case=dense_progress['cases'][idx];i=1380+idx;actual=splat_plan['cases'][i]
        assert abs(case['source_time']-actual['source_time'])<1e-8
        for key in ('r','t','k'):np.testing.assert_allclose(case[key],actual[key],atol=1e-8,rtol=0)
        path=dense/dense_progress['frames'][idx]['file'];assert digest(path)==dense_progress['frames'][idx]['sha256'];sample[i]=read(path)
    videos={};scores={}
    for key in ('motion','detail','fallback'):
        path=folder/f'{key}.mp4';cap=cv2.VideoCapture(str(path));minimum=100.;metrics=[]
        for i in range(count):
            ok,image=cap.read();assert ok
            if i%2==0:minimum=min(minimum,float(cv2.PSNR(image,gt(i//2))))
            if i in sample:metrics.append(dict(index=i,psnr=float(cv2.PSNR(image,sample[i])),ssim=float(structural_similarity(image,sample[i],channel_axis=2,data_range=255))))
        assert not cap.read()[0];cap.release();assert minimum>32
        videos[key]=dict(**verify(path,30,count),decoded_frames=count,minimum_anchor_psnr=minimum)
        scores[key]=dict(count=len(metrics),gap_psnr=float(np.mean([r['psnr'] for r in metrics])),gap_ssim=float(np.mean([r['ssim'] for r in metrics])),rows=metrics)
        print(json.dumps(dict(phase='verification',method=key,gap_psnr=scores[key]['gap_psnr'],gap_ssim=scores[key]['gap_ssim'],minimum_anchor_psnr=minimum)),flush=True)
    videos['native']=verify(folder/'native.mp4',15,n)
    report['timings']['verification_seconds']=time.perf_counter()-check_started
    report.update(status='complete',phase='complete',videos=videos,outdoor_gap_scores=scores,estimated_remaining_seconds=0,
        note='Full route has only 15fps genuine GNT. Gap scores apply only to the previous independent 30fps outdoor clip, not the full route. Training excluded from timings. Correction does not modify the underlying splat model. Fallback includes motion interpolation.')
    save(folder/'results.json',report);save(progress_path,report);save(ROOT/'outputs/gnt_finalization/scene15_splat.json',report);publish(folder,report)
    print('Complete: http://localhost:8088/gnt-scene-15fps-splats.html',flush=True)


if __name__=='__main__':main()
