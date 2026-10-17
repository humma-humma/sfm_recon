"""Matched scene-change threshold clips using existing adaptive GNT renders."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import urllib.request
import cv2
import numpy as np
from publish_gnt_videos import verify

ROOT=Path(__file__).resolve().parents[1]


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--destination',type=Path,required=True)
    parser.add_argument('--ffmpeg',required=True);args=parser.parse_args()
    source=args.destination/'assets/adaptive_scene'
    adaptive=json.loads((source/'progress.json').read_text())
    assert adaptive['status']=='complete'
    output=args.destination/'assets/scd_thresholds';output.mkdir(parents=True,exist_ok=True)
    report=dict(status='running',route_start_seconds=46,route_end_seconds=52,duration_seconds=6,
        source='Existing adaptive scene lossless frames, original timestamps',
        concat_sha256=hashlib.sha256((source/'frames.ffconcat').read_bytes()).hexdigest(),
        settings=dict(fps=30,mi_mode='mci',mc_mode='aobmc',me_mode='bidir',vsbmc=1,scd='fdiff'),
        note='Only scd_threshold differs between motion-compensated candidates. Native is a separate held-view control. Duplicate fraction describes playback behavior, not visual accuracy.',videos={})
    for label,threshold in [('native',None),('threshold_3',3),('threshold_5',5),('threshold_7',7),('threshold_9',9),('threshold_10',10)]:
        file=output/f'{label}.mp4'
        interpolation='fps=30' if threshold is None else f'minterpolate=fps=30:mi_mode=mci:mc_mode=aobmc:me_mode=bidir:vsbmc=1:scd=fdiff:scd_threshold={threshold}'
        # Two seconds of temporal context on each side; identical timing for every variant.
        filter_value=f'trim=start=44:end=54,setpts=PTS-STARTPTS,{interpolation},trim=start=2:duration=6,setpts=PTS-STARTPTS'
        if not file.exists():
            temporary=file.with_suffix('.partial.mp4')
            subprocess.run([args.ffmpeg,'-hide_banner','-loglevel','error','-y','-threads','4',
                '-safe','0','-i','frames.ffconcat','-vf',filter_value,'-t','6','-an',
                '-c:v','libx264','-preset','fast','-crf','16','-pix_fmt','yuv420p','-movflags','+faststart',
                str(temporary)],cwd=source,check=True)
            verify(temporary,6);temporary.replace(file)
        row=verify(file,6);row['threshold']=threshold;row['filter']=filter_value
        cap=cv2.VideoCapture(str(file));previous=None;errors=[]
        while True:
            ok,image=cap.read()
            if not ok:break
            gray=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY).astype(float)
            if previous is not None:errors.append(float(np.mean(np.abs(gray-previous))))
            previous=gray
        cap.release();assert len(errors)==179
        row['near_duplicate_fraction']=float(np.mean(np.array(errors)<.5))
        url='http://localhost:8088/assets/gnt_finalization/scd_thresholds/'+file.name
        with urllib.request.urlopen(urllib.request.Request(url,headers={'Range':'bytes=0-1023'})) as response:
            assert response.status==206 and len(response.read())==1024
        report['videos'][label]=row
        (output/'results.json').write_text(json.dumps(report,indent=2))
        print(json.dumps(dict(label=label,status='verified',near_duplicate_fraction=row['near_duplicate_fraction'])),flush=True)
    report['status']='complete'
    (output/'results.json').write_text(json.dumps(report,indent=2))
    (ROOT/'outputs/gnt_finalization/scd_thresholds.json').write_text(json.dumps(report,indent=2))
    cards=[]
    for label,row in report['videos'].items():
        title='Native held views' if row['threshold'] is None else f'Threshold {row["threshold"]}'+(' (previous default)' if row['threshold']==10 else '')
        url='assets/gnt_finalization/scd_thresholds/'+row['file']
        cards.append(f'<section><h2>{title}</h2><video controls muted playsinline preload="auto" src="{url}"></video><a href="{url}" download>Download</a></section>')
    html='''<!doctype html><meta charset="utf-8"><title>GNT threshold comparison</title>
<style>body{margin:24px;background:#121720;color:#eee;font:16px system-ui}main{display:grid;grid-template-columns:repeat(auto-fit,minmax(420px,1fr));gap:20px}video{width:100%}a{color:#8cf}button{padding:10px;margin:8px}input{width:320px}</style>
<h1>Scene-change thresholds: same turning clip</h1><p>Adaptive scene route, 46–52 seconds. 960×540 at 30 fps. Only the scene-change threshold differs between interpolated versions; native held views and threshold 10 are controls. Lower thresholds can replace warping with frame jumps.</p>
<button id="play">Play all from start</button><button id="pause">Pause all</button><label>Seek together <input id="seek" type="range" min="0" max="5.9" step="0.033" value="0"></label>
<main>__CARDS__</main><script>
const videos=[...document.querySelectorAll('video')];
document.getElementById('play').onclick=async()=>{videos.forEach(v=>v.pause());await Promise.all(videos.map(v=>new Promise(resolve=>{v.currentTime=0;if(v.readyState>=3)resolve();else v.addEventListener('canplay',resolve,{once:true});})));await Promise.all(videos.map(v=>v.play()));};
document.getElementById('pause').onclick=()=>videos.forEach(v=>v.pause());
document.getElementById('seek').oninput=e=>videos.forEach(v=>{v.pause();v.currentTime=+e.target.value;});
</script>'''.replace('__CARDS__',''.join(cards))
    (ROOT/'web_demo/gnt-thresholds.html').write_text(html,encoding='utf-8')
    print('Complete: http://localhost:8088/gnt-thresholds.html',flush=True)


if __name__=='__main__':main()
