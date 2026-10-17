"""Encode verified native splat videos and publish incremental all-four review."""
import argparse
import json
from pathlib import Path
import subprocess
import cv2
import numpy as np
from publish_gnt_videos import verify
from gnt_rollout_io import save

ROOT=Path(__file__).resolve().parents[1]


def encode(root,name,ffmpeg):
    folder=root/'assets/balanced_rollout'/name;path=folder/'video.json'
    report=json.loads(path.read_text());raw=folder/report['raw'];video=folder/f'{name}_balanced90.mp4'
    if report['status']!='encoded':
        assert raw.stat().st_size==report['frames']*960*540*3
        temp=video.with_suffix('.partial.mp4')
        subprocess.run([ffmpeg,'-hide_banner','-loglevel','error','-y','-threads','4','-f','rawvideo','-pix_fmt','rgb24',
            '-s','960x540','-r','30','-i',str(raw),'-an','-c:v','libx264','-preset','fast','-crf','16','-pix_fmt','yuv420p','-movflags','+faststart',str(temp)],check=True)
        info=verify(temp,report['duration_seconds'])
        cap=cv2.VideoCapture(str(temp));psnrs=[]
        with raw.open('rb') as stream:
            for i in (0,report['frames']//2,report['frames']-1):
                stream.seek(i*960*540*3);reference=np.frombuffer(stream.read(960*540*3),np.uint8).reshape(540,960,3)
                cap.set(cv2.CAP_PROP_POS_FRAMES,i);ok,image=cap.read();assert ok
                mse=np.mean(((cv2.cvtColor(image,cv2.COLOR_BGR2RGB).astype(np.float32)-reference)/255)**2)
                psnrs.append(float(-10*np.log10(max(mse,1e-16))))
        cap.release();assert min(psnrs)>32,psnrs
        temp.replace(video);info['file']=video.name
        report.update(status='encoded',video=info,decoded_sample_psnr=psnrs);save(path,report)
        # Delete only the verified generated raw stream, never a directory.
        assert raw.resolve().parent==folder.resolve() and raw.name=='full_route.rgb';raw.unlink()
    else:
        info=verify(video,report['duration_seconds']);assert info['sha256']==report['video']['sha256']
    teacher=root/'assets/dense_turn' if name=='scene' else folder/'teachers'
    clips={}
    for label,input_pattern in [('gnt',teacher/'frame_%04d.png'),('dense',folder/'dense_dense_%04d.png'),('balanced90',folder/'balanced90_dense_%04d.png')]:
        output=folder/f'{name}_{label}_turn.mp4'
        if not output.exists():
            temp=output.with_suffix('.partial.mp4')
            subprocess.run([ffmpeg,'-hide_banner','-loglevel','error','-y','-threads','4','-framerate','30','-i',str(input_pattern),
                '-frames:v','180','-an','-c:v','libx264','-preset','fast','-crf','16','-pix_fmt','yuv420p','-movflags','+faststart',str(temp)],check=True)
            verify(temp,6.);temp.replace(output)
        clips[label]=verify(output,6.)
    report['clips']=clips;save(path,report)


def publish(root):
    summaries={};sections=[]
    for name in ('scene','box','boot','milk'):
        folder=root/'assets/balanced_rollout'/name;path=folder/'video.json'
        if not path.exists() or json.loads(path.read_text()).get('status')!='encoded' or 'clips' not in json.loads(path.read_text()):
            sections.append(f'<section><h2>{name.title()}</h2><p>Queued or processing. Refresh to check for the next completed dataset.</p></section>');continue
        report=json.loads(path.read_text());summaries[name]=report
        prefix=f'assets/gnt_finalization/balanced_rollout/{name}/'
        main=prefix+report['video']['file']
        figures=''.join(f'<figure><figcaption>{label}</figcaption><video class="comparison" controls preload="metadata" src="{prefix+report["clips"][arm]["file"]}"></video></figure>' for arm,label in [('gnt','1. Best GNT — quality reference'),('dense','2. Uniform dense supervision'),('balanced90','3. Current — balanced 90/10')])
        evaluations={a:json.loads((folder/f'{a}_evaluation.json').read_text())['summaries'] for a in ('control','dense','balanced90')}
        metrics=''.join(f'<tr><td>{arm}</td><td>{g}</td><td>{s["count"]}</td><td>{s["psnr"]:.2f}</td><td>{s["ssim"]:.4f}</td></tr>' for arm,groups in evaluations.items() for g,s in groups.items())
        sections.append(f'<section><h2>{name.title()}</h2><h3>Full route — balanced splats</h3><video class="full" controls preload="metadata" src="{main}"></video><p><a href="{main}" download>Download full video</a> · {report["duration_seconds"]:.1f} seconds · 960 × 540 · 30 fps</p><h3>Matched six-second section</h3><p>These three videos share camera timing. Playing, pausing or seeking one controls all three.</p><div class="triple">{figures}</div><details><summary>Evaluation scores</summary><p>Teacher agreement is not photographic ground truth. Original photos are held out of RGB fitting; camera estimation and seed geometry predate that split. The dense teacher covers one section, not the entire route.</p><table><tr><th>Model</th><th>Views</th><th>Count</th><th>PSNR</th><th>SSIM</th></tr>{metrics}</table></details></section>')
    page='''<!doctype html><html><meta charset="utf-8"><title>Four balanced-splat traversals</title>
<style>body{max-width:1700px;margin:24px auto;padding:0 16px;background:#121720;color:#eee;font:16px system-ui}a{color:#8cf}video{width:100%;display:block}.full{max-width:960px}.triple{display:grid;grid-template-columns:repeat(3,1fr);gap:12px}figure{margin:0}figcaption{min-height:55px;font-weight:600}section{padding:24px 0;border-top:1px solid #456}td,th{padding:8px;text-align:left}table{border-collapse:collapse}@media(max-width:750px){.triple{grid-template-columns:1fr}}</style>
<h1>Box, boot, milk and scene — balanced splat traversals</h1><p>Full videos use native Gaussian rendering at 30 fps, without motion-compensated frame interpolation. They prioritize speed and retain less detail than GNT. Each object uses 60 sparse teacher views plus one dense six-second section; scene reuses its existing balanced model. Existing presentation selections are preserved.</p>__SECTIONS__
<script>document.querySelectorAll('.triple').forEach(group=>{const videos=[...group.querySelectorAll('video')];let changing=false;
for(const v of videos){v.addEventListener('play',()=>{if(changing)return;changing=true;for(const other of videos)if(other!==v){other.currentTime=v.currentTime;other.play().catch(()=>{});}changing=false;});
v.addEventListener('pause',()=>{if(changing)return;changing=true;for(const other of videos)if(other!==v)other.pause();changing=false;});
v.addEventListener('seeking',()=>{if(changing)return;changing=true;for(const other of videos)if(other!==v&&Math.abs(other.currentTime-v.currentTime)>.08)other.currentTime=v.currentTime;changing=false;});}});</script></html>'''.replace('__SECTIONS__',''.join(sections))
    path=ROOT/'web_demo/gnt-balanced-all.html';path.write_text(page,encoding='utf-8')
    save(root/'assets/balanced_rollout/summary.json',dict(status='complete' if len(summaries)==4 else 'running',datasets=summaries,url='http://localhost:8088/gnt-balanced-all.html'))
    print(json.dumps(dict(status='published',complete=list(summaries),url='http://localhost:8088/gnt-balanced-all.html')),flush=True)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--dataset',choices=['scene','box','boot','milk']);parser.add_argument('--ffmpeg',default=r'C:\ffmpeg\bin\ffmpeg.exe')
    args=parser.parse_args()
    if args.dataset:encode(args.root,args.dataset,args.ffmpeg)
    publish(args.root)


if __name__=='__main__':main()
