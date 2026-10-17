"""Encode a bounded adaptive scene route without replacing the accepted videos."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
from publish_gnt_videos import verify

ROOT=Path(__file__).resolve().parents[1]


def main():
    parser=argparse.ArgumentParser()
    for key in ('original','output'):parser.add_argument('--'+key,type=Path,required=True)
    parser.add_argument('--ffmpeg',required=True);args=parser.parse_args()
    manifest=json.loads((args.original/'manifest.json').read_text())
    progress=args.output/'progress.json'
    report=json.loads(progress.read_text()) if progress.exists() else dict(status='pending')
    if report['status'] in ('rendered','complete'):
        assert report['checkpoint_sha256']==manifest['checkpoint_sha256']
        originals=args.output/'original';originals.mkdir(exist_ok=True)
        frames=[]
        for row in manifest['frames']:
            source=args.original/row['frame'];target=originals/source.name
            if not target.exists():os.link(source,target)
            assert hashlib.sha256(target.read_bytes()).hexdigest()==row['sha256']
            frames.append(dict(time=row['source_time'],file='original/'+target.name))
        for row in report['native'].values():
            assert hashlib.sha256((args.output/row['file']).read_bytes()).hexdigest()==row['sha256']
            assert all(i%8 for i in row['sources'])
            frames.append(dict(time=row['case']['source_time'],file=row['file']))
        frames.sort(key=lambda row:row['time'])
        assert all(b['time']>a['time'] for a,b in zip(frames[:-1],frames[1:]))
        start=manifest['frames'][0]['source_time'];end=start+manifest['duration_seconds']
        concat=['ffconcat version 1.0']
        for i,row in enumerate(frames):
            next_time=frames[i+1]['time'] if i+1<len(frames) else end
            concat += [f"file '{row['file']}'",f"duration {next_time-row['time']:.9f}"]
        concat.append(f"file '{frames[-1]['file']}'")
        (args.output/'frames.ffconcat').write_text('\n'.join(concat)+'\n')
        videos={}
        for label,filter_value in [('native','fps=30'),('motion_compensated',
            'tpad=stop_mode=clone:stop_duration=1,minterpolate=fps=30:mi_mode=mci:mc_mode=aobmc:me_mode=bidir:vsbmc=1')]:
            file=args.output/f'scene_adaptive_{label}.mp4'
            if not file.exists():
                temp=file.with_suffix('.partial.mp4')
                subprocess.run([args.ffmpeg,'-hide_banner','-loglevel','error','-y','-threads','4',
                    '-safe','0','-i','frames.ffconcat','-vf',filter_value,'-t','146.5','-an',
                    '-c:v','libx264','-preset','fast','-crf','16','-pix_fmt','yuv420p',
                    '-movflags','+faststart',temp.name],cwd=args.output,check=True)
                verify(temp,146.5);temp.replace(file)
            videos[label]=verify(file,146.5)
        report.update(status='complete',independent_frames=len(frames),videos=videos,
            verification=dict(status='passed',strict_timestamps=True,original_hashes=True,heldout_source_exclusion=True))
        progress.write_text(json.dumps(report,indent=2))
        (ROOT/'outputs/gnt_finalization/adaptive.json').write_text(json.dumps(report,indent=2))
    sections=[]
    for label,url in [('Previous motion compensation','assets/gnt_finalization/blending/scene_motion_compensated.mp4')]:
        sections.append(f'<h2>{label}</h2><video controls preload="metadata" src="{url}"></video>')
    if report['status']=='complete':
        sections.insert(0,f'<p>{report["independent_frames"]} independent views; {len(report["native"])} added. Unprocessed intervals: {report["remaining_intervals"]}. Depth/support limited: {len(report["depth_limited"])}.</p>')
        for label,row in report['videos'].items():
            url='assets/gnt_finalization/adaptive_scene/'+row['file']
            sections.append(f'<h2>Adaptive {label.replace("_"," ")}</h2><video controls preload="metadata" src="{url}"></video><a href="{url}" download>Download</a>')
    else:sections.insert(0,'<p>Adaptive rendering is in progress. This page updates after export completes.</p>')
    html='<!doctype html><meta charset="utf-8"><title>Adaptive GNT traversal</title><style>body{max-width:1050px;margin:32px auto;background:#121720;color:#eee;font:17px system-ui}video{width:100%;max-width:960px}a{color:#8cf}</style><h1>Adaptive GNT traversal</h1><p>Original route timing and model preserved. Extra genuine GNT views target difficult transitions. Motion compensation still synthesizes intermediate frames; sparse source support is not calibrated uncertainty.</p>'+''.join(sections)
    (ROOT/'web_demo/gnt-adaptive.html').write_text(html,encoding='utf-8')
    print(json.dumps(dict(status=report['status'],added=len(report.get('native',{})))),flush=True)


if __name__=='__main__':main()
