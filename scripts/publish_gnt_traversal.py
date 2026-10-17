"""Archive native GNT views and encode a faithful full-route inspection video."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import cv2
import numpy as np

ROOT=Path(__file__).resolve().parents[1]


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--trial',type=Path,required=True)
    parser.add_argument('--destination',type=Path,required=True)
    parser.add_argument('--ffmpeg',default='ffmpeg')
    args=parser.parse_args()
    source=args.trial/'gnt_tuning'
    report=json.loads((source/'results.json').read_text(encoding='utf-8'))
    traversal=report['traversal'];cases=traversal['cases'];width=traversal['width']
    checkpoint=source/'detail_control_006000.pth'
    assert hashlib.sha256(checkpoint.read_bytes()).hexdigest()==traversal['checkpoint_sha256']
    selected=[]
    for case in cases:
        rows=[r for r in report['runs'] if r['label']==traversal['label'] and r['case']==case['id'] and r['width']==width]
        assert len(rows)==1, ('Missing traversal frame',case['id'])
        row=rows[0]
        assert all(i%8!=0 for i in row['sources'])
        assert row['config']['chunk']==traversal['chunk']
        selected.append((case,row))
    destination=args.destination.resolve();frames=destination/'frames';frames.mkdir(parents=True,exist_ok=True)
    times=np.array([c['source_time'] for c in cases]);starts=times-times[0]
    durations=np.r_[np.diff(times),np.median(np.diff(times))]
    manifest=[];concat=['ffconcat version 1.0']
    for i,((case,row),start,duration) in enumerate(zip(selected,starts,durations)):
        original=source/row['path'];name=original.name;target=frames/name
        digest=hashlib.sha256(original.read_bytes()).hexdigest()
        if not target.exists():shutil.copy2(original,target)
        assert hashlib.sha256(target.read_bytes()).hexdigest()==digest
        image=cv2.imread(str(target));assert image is not None and image.shape[:2]==(round(width*540/960),width)
        manifest.append(dict(index=i,id=case['id'],start=float(start),duration=float(duration),
            source_time=case['source_time'],frame='frames/'+name,sha256=digest,sources=row['sources'],
            render_seconds=row['seconds'],bracket=case['bracket']))
        concat.extend([f"file 'frames/{name}'",f'duration {duration:.9f}'])
    concat.append(f"file '{manifest[-1]['frame']}'")
    (destination/'frames.ffconcat').write_text('\n'.join(concat)+'\n',encoding='utf-8')
    movie=destination/'scene_control6000.mp4'
    duration=float(durations.sum())
    subprocess.run([args.ffmpeg,'-hide_banner','-loglevel','warning','-y','-safe','0','-i','frames.ffconcat',
        '-t',str(duration),'-r','30','-c:v','libx264','-preset','medium','-crf','16','-pix_fmt','yuv420p',
        '-movflags','+faststart',movie.name],cwd=destination,check=True)
    capture=cv2.VideoCapture(str(movie));assert capture.isOpened()
    assert int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))==width
    assert int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))==round(width*540/960)
    encoded_duration=capture.get(cv2.CAP_PROP_FRAME_COUNT)/capture.get(cv2.CAP_PROP_FPS)
    assert abs(encoded_duration-duration)<.1
    checks=[]
    for i in (0,len(manifest)//2,len(manifest)-1):
        item=manifest[i];capture.set(cv2.CAP_PROP_POS_MSEC,(item['start']+item['duration']/2)*1000)
        ok,decoded=capture.read();assert ok and decoded.std()>1
        original=cv2.imread(str(destination/item['frame']))
        mse=np.mean((decoded.astype(float)-original.astype(float))**2)
        psnr=float(-10*np.log10(max(mse,1e-12)/255**2))
        assert psnr>35, ('Unexpectedly large video encoding error',i,psnr)
        checks.append(dict(index=i,encoding_psnr=psnr))
    capture.release()
    summary=dict(model='6000-step RGB control',checkpoint_sha256=traversal['checkpoint_sha256'],
        width=width,height=round(width*540/960),independent_frames=len(manifest),duration_seconds=duration,
        encoded_fps=30,rendered_fps=len(manifest)/duration,frame_interpolation=False,
        note='Each view is independently rendered at a midpoint of consecutive captured poses. Video holds each view until the next capture time; 30fps encoding repeats frames and does not create smoother motion. No blending, optical-flow interpolation or post-sharpening.',
        source_time_range=[float(times[0]),float(times[-1])],
        recorded_render_seconds=float(sum(row['seconds'] for _,row in selected)),
        video=str(movie),video_sha256=hashlib.sha256(movie.read_bytes()).hexdigest(),
        frames=manifest,verification=dict(status='passed',decoded_samples=checks,heldout_source_exclusion=True))
    (destination/'manifest.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
    output=ROOT/'outputs/gnt_traversal';output.mkdir(exist_ok=True)
    (output/'results.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
    html='''<!doctype html><meta charset="utf-8"><meta name="viewport" content="width=device-width"><title>GNT full traversal</title>
<style>body{max-width:1200px;margin:24px auto;padding:0 16px;background:#111;color:#eee;font:16px system-ui}a{color:#9bd}video{display:block;width:100%;background:#000}button,select{margin:6px;padding:8px}p{line-height:1.5}</style>
<h1>GNT: full captured route</h1><p id="description"></p>
<video id="video" controls preload="metadata" src="assets/gnt_traversal/scene_control6000.mp4"></video>
<p><button id="previous">Previous rendered view</button><button id="next">Next rendered view</button>
<label>Speed <select id="speed"><option>0.5</option><option selected>1</option><option>1.5</option><option>2</option></select></label>
<a id="native" target="_blank">Open lossless frame</a> · <a href="assets/gnt_traversal/scene_control6000.mp4" download>Download MP4</a> · <a href="gnt-detail.html">Still comparisons</a></p>
<p id="status"></p><p>Each view is independently rendered. The video holds it until the next capture time; 30fps encoding repeats frames. No frame blending, generated intermediate images, or post-sharpening is applied. Pause or open the lossless PNG to evaluate fine detail.</p>
<script>const data=__DATA__,video=document.getElementById('video'),frames=data.frames;
const index=()=>Math.max(0,frames.findLastIndex(f=>f.start<=video.currentTime));
function update(){const i=index(),f=frames[i];document.getElementById('status').textContent=`View ${i+1}/${frames.length} · capture time ${f.source_time.toFixed(2)}s`;document.getElementById('native').href='assets/gnt_traversal/'+f.frame;}
function step(delta){video.pause();const i=Math.max(0,Math.min(frames.length-1,index()+delta));video.currentTime=frames[i].start+.03;update();}
document.getElementById('previous').onclick=()=>step(-1);document.getElementById('next').onclick=()=>step(1);document.getElementById('speed').onchange=e=>video.playbackRate=+e.target.value;
video.ontimeupdate=video.onseeked=update;document.getElementById('description').textContent=`6,000-step RGB control · ${data.width} × ${data.height} · ${data.independent_frames} rendered views · ${data.duration_seconds.toFixed(1)} seconds. Follows the complete captured route through the indoor and outdoor regions.`;update();</script>'''
    html=html.replace('assets/gnt_traversal/',f'assets/gnt_traversal_{width}/')
    (ROOT/'web_demo/gnt-traversal.html').write_text(html.replace('__DATA__',json.dumps(summary)),encoding='utf-8')
    print(json.dumps({k:v for k,v in summary.items() if k!='frames'},indent=2))


if __name__=='__main__':main()
