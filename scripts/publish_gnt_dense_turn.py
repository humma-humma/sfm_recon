"""Publish the direct 30 fps GNT clip alongside matched sparse controls."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import urllib.request
import cv2
from publish_gnt_videos import verify

ROOT=Path(__file__).resolve().parents[1]


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--ffmpeg',required=True);args=parser.parse_args()
    progress=args.output/'progress.json'
    report=json.loads(progress.read_text()) if progress.exists() else dict(status='pending')
    cards=[]
    if report['status'] in ('rendered','complete'):
        frames=sorted(report['frames'],key=lambda row:row['index'])
        assert [row['index'] for row in frames]==list(range(180))
        for row in frames:
            file=args.output/row['file']
            assert hashlib.sha256(file.read_bytes()).hexdigest()==row['sha256']
            image=cv2.imread(str(file));assert image is not None and image.shape[:2]==(540,960)
            assert all(i%8 for i in row['sources'])
        assert all(abs((b['source_time']-a['source_time'])-1/30)<1e-8 for a,b in zip(frames[:-1],frames[1:]))
        video=args.output/'scene_turn_genuine_30fps.mp4'
        if not video.exists():
            temp=video.with_suffix('.partial.mp4')
            subprocess.run([args.ffmpeg,'-hide_banner','-loglevel','error','-y','-threads','4',
                '-framerate','30','-start_number','0','-i',str(args.output/'frame_%04d.png'),
                '-frames:v','180','-an','-c:v','libx264','-preset','fast','-crf','16','-pix_fmt','yuv420p',
                '-movflags','+faststart',str(temp)],check=True)
            verify(temp,6);temp.replace(video)
        proof=verify(video,6)
        cap=cv2.VideoCapture(str(video));count=0
        while cap.read()[0]:count+=1
        cap.release();assert count==180
        url='http://localhost:8088/assets/gnt_finalization/dense_turn/'+video.name
        with urllib.request.urlopen(urllib.request.Request(url,headers={'Range':'bytes=0-1023'})) as response:
            assert response.status==206 and len(response.read())==1024
        report.update(status='complete',video=proof,verification=dict(status='passed',decoded_frames=180,
            uniform_camera_time=True,heldout_source_exclusion=True,frame_interpolation=False,http_seeking=True))
        progress.write_text(json.dumps(report,indent=2))
        (ROOT/'outputs/gnt_finalization/dense_turn.json').write_text(json.dumps(report,indent=2))
        cards.append(('Genuine GNT: 30 independent views per second','assets/gnt_finalization/dense_turn/'+video.name))
    else:cards.append(('Dense GNT clip is rendering',None))
    cards += [('Previous adaptive native (held views)','assets/gnt_finalization/scd_thresholds/native.mp4'),
        ('Previous motion compensation (threshold 10)','assets/gnt_finalization/scd_thresholds/threshold_10.mp4')]
    content=''.join(f'<section><h2>{title}</h2>'+ (f'<video controls muted playsinline preload="metadata" src="{url}"></video><a href="{url}" download>Download MP4</a>' if url else '<p>180 frames requested; refresh after export completes.</p>')+'</section>' for title,url in cards)
    html='''<!doctype html><meta charset="utf-8"><title>Genuine 30 fps GNT turn</title>
<style>body{max-width:1100px;margin:24px auto;background:#121720;color:#eee;font:17px system-ui}video{width:100%;max-width:960px}a{color:#8cf}button{margin:8px;padding:10px}</style>
<h1>Dense GNT turning comparison</h1><p>Same route segment: 46–52 seconds, 960×540. Dense output renders every frame with GNT; no motion interpolation, crossfade or sharpening. Existing camera timing and piecewise path are preserved.</p>
<button id="play">Play all from start</button><button id="pause">Pause all</button>__CARDS__
<script>const videos=[...document.querySelectorAll('video')];document.getElementById('play').onclick=()=>{videos.forEach(v=>{v.pause();v.currentTime=0;});Promise.all(videos.map(v=>v.play()));};document.getElementById('pause').onclick=()=>videos.forEach(v=>v.pause());</script>'''.replace('__CARDS__',content)
    (ROOT/'web_demo/gnt-dense-turn.html').write_text(html,encoding='utf-8')
    print(json.dumps(dict(status=report['status'],completed=len(report.get('frames',[])))),flush=True)


if __name__=='__main__':main()
