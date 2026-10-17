"""Encode independently rendered objects and index all four GNT videos."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import cv2

ROOT=Path(__file__).resolve().parents[1]


def verify(path,duration):
    cap=cv2.VideoCapture(str(path));assert cap.isOpened(),str(path)
    fps=cap.get(cv2.CAP_PROP_FPS);frames=cap.get(cv2.CAP_PROP_FRAME_COUNT)
    assert fps==30 and abs(frames/fps-duration)<.05
    assert (cap.get(cv2.CAP_PROP_FRAME_WIDTH),cap.get(cv2.CAP_PROP_FRAME_HEIGHT))==(960,540)
    for fraction in (0,.5,.95):
        cap.set(cv2.CAP_PROP_POS_FRAMES,int((frames-1)*fraction));ok,_=cap.read();assert ok
    cap.release()
    return dict(file=path.name,sha256=hashlib.sha256(path.read_bytes()).hexdigest(),duration_seconds=frames/fps)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--destination',type=Path,required=True)
    parser.add_argument('--ffmpeg',required=True);args=parser.parse_args()
    base=args.destination/'assets/videos';base.mkdir(parents=True,exist_ok=True)
    results={};sections=[]
    for name in ('box','boot','milk'):
        folder=base/name;progress=folder/'progress.json'
        if not progress.exists():sections.append(f'<h2>{name.title()}</h2><p>Pending rendering.</p>');continue
        report=json.loads(progress.read_text())
        if report['status'] not in ('rendered','complete'):
            sections.append(f'<h2>{name.title()}</h2><p>{len(report["frames"])}/60 native views rendered.</p>');continue
        assert len(report['frames'])==60
        for row in report['frames']:
            frame=folder/row['file'];assert hashlib.sha256(frame.read_bytes()).hexdigest()==row['sha256']
            image=cv2.imread(str(frame));assert image is not None and image.shape[:2]==(540,960)
            assert all(i%8 for i in row['sources'])
        videos={}
        for label,filter_value in (
            ('native','fps=30'),
            ('motion_compensated','tpad=stop_mode=clone:stop_duration=2,minterpolate=fps=30:mi_mode=mci:mc_mode=aobmc:me_mode=bidir:vsbmc=1')):
            output=folder/f'{name}_{label}.mp4'
            if not output.exists():
                temporary=output.with_suffix('.partial.mp4')
                subprocess.run([args.ffmpeg,'-hide_banner','-loglevel','error','-y','-threads','4',
                    '-framerate','15/8','-i',str(folder/'frame_%04d.png'),'-vf',filter_value,
                    '-t','32','-an','-c:v','libx264','-preset','fast','-crf','16','-pix_fmt','yuv420p',
                    '-movflags','+faststart',str(temporary)],check=True)
                verify(temporary,32);temporary.replace(output)
            videos[label]=verify(output,32)
        report['status']='complete';report['videos']=videos
        progress.write_text(json.dumps(report,indent=2));results[name]=report
        sections.append(f'<h2>{name.title()}</h2>')
        for label,video in videos.items():
            url=f'assets/gnt_finalization/videos/{name}/{video["file"]}'
            sections.append(f'<h3>{label.replace("_"," ").title()}</h3><video controls preload="metadata" src="{url}"></video> <a href="{url}" download>Download MP4</a>')
    scene=json.loads((ROOT/'outputs/gnt_traversal/results.json').read_text())
    native=Path(scene['video']);verify(native,146.5)
    flow=args.destination/'assets/blending/scene_motion_compensated.mp4';verify(flow,146.5)
    results['scene']=dict(status='complete',native=str(native),motion_compensated=str(flow))
    sections.append('<h2>Scene</h2>')
    for label,url in [('Native','assets/gnt_traversal_960/scene_control6000.mp4'),('Motion compensated','assets/gnt_finalization/blending/scene_motion_compensated.mp4')]:
        sections.append(f'<h3>{label}</h3><video controls preload="metadata" src="{url}"></video> <a href="{url}" download>Download MP4</a>')
    page='<!doctype html><meta charset="utf-8"><title>Four GNT traversals</title><style>body{max-width:1100px;margin:32px auto;background:#121720;color:#eee;font:17px system-ui}video{display:block;width:100%;max-width:960px}a{color:#8cc8ff}h2{margin-top:48px}</style><h1>Four GNT traversals</h1><p>960×540 renders. Native playback holds independent views; motion compensation synthesizes intermediate frames at 30 fps. Object paths follow captured cameras without an artificial closing orbit. Object videos use the 6,000-step adapted models.</p>' + ''.join(sections)
    (ROOT/'web_demo/gnt-videos.html').write_text(page,encoding='utf-8')
    out=ROOT/'outputs/gnt_finalization/videos.json'
    out.write_text(json.dumps(dict(status='complete' if len(results)==4 else 'running',datasets=results),indent=2),encoding='utf-8')
    print(json.dumps({name:value['status'] for name,value in results.items()}),flush=True)


if __name__=='__main__':main()
