"""Compare native GNT views with post-render blending, without changing originals."""
import argparse
import json
from pathlib import Path
import shutil
import subprocess
import cv2
import numpy as np
from skimage.metrics import structural_similarity

ROOT=Path(__file__).resolve().parents[1]

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--destination',type=Path,required=True)
    parser.add_argument('--ffmpeg',required=True)
    parser.add_argument('--full-flow',action='store_true');args=parser.parse_args()
    source=json.loads((ROOT/'outputs/gnt_traversal/results.json').read_text(encoding='utf-8'))
    original=Path(source['video']).parent
    destination=args.destination/'assets/blending';destination.mkdir(parents=True,exist_ok=True)
    report=dict(status='complete',native_frames=source['independent_frames'],clips=[],
        note='GNT has no temporal frame blending module in the released code. These are post-render FFmpeg variants. Native source renders remain unchanged. Evaluation withholds alternate native GNT views, not photographs: metrics measure interpolation consistency with GNT and cannot establish photorealistic accuracy. Clips use uniform 1.875 fps camera spacing; 30 fps output synthesizes intermediate images.')
    for name,start in [('indoor',32),('outdoor',120),('stairs',232)]:
        folder=destination/name;folder.mkdir(exist_ok=True)
        # Withhold alternate independent renders for an interpolation check.
        for j,i in enumerate(range(start,start+13)):
            shutil.copy2(original/source['frames'][i]['frame'],folder/f'all_{j:03}.png')
            if j%2==0:shutil.copy2(original/source['frames'][i]['frame'],folder/f'key_{j//2:03}.png')
        for label,filter_value in [('native','fps=30'),('held','fps=30'),
            ('blend','minterpolate=fps=30:mi_mode=blend'),
            ('flow','minterpolate=fps=30:mi_mode=mci:mc_mode=aobmc:me_mode=bidir:vsbmc=1')]:
            rate='15/8' if label=='native' else '15/16'
            pattern='all_%03d.png' if label=='native' else 'key_%03d.png'
            output=folder/f'{label}.mp4'
            if not output.exists():
                subprocess.run([args.ffmpeg,'-hide_banner','-loglevel','error','-y','-threads','4',
                    '-framerate',rate,'-i',pattern,'-vf',filter_value,'-an','-c:v','libx264',
                    '-preset','fast','-crf','16','-pix_fmt','yuv420p','-movflags','+faststart',output.name],cwd=folder,check=True)
        scores={}
        for label in ('held','blend','flow'):
            capture=cv2.VideoCapture(str(folder/f'{label}.mp4'));assert capture.isOpened()
            samples=[]
            # Avoid filter start/end buffering, and compare interior withheld views.
            for j in (3,5,7,9):
                capture.set(cv2.CAP_PROP_POS_MSEC,j*8/15*1000)
                ok,image=capture.read();assert ok
                reference=cv2.imread(str(folder/f'all_{j:03}.png'))
                mse=np.mean((image.astype(float)-reference.astype(float))**2)/255**2
                samples.append(dict(index=start+j,psnr=float(-10*np.log10(max(mse,1e-16))),
                    ssim=float(structural_similarity(image,reference,channel_axis=2,data_range=255))))
            capture.release();scores[label]=dict(samples=samples,
                psnr=float(np.mean([s['psnr'] for s in samples])),ssim=float(np.mean([s['ssim'] for s in samples])))
        report['clips'].append(dict(name=name,start_index=start,indices=list(range(start,start+13)),scores=scores))
        print(json.dumps(report['clips'][-1]),flush=True)
        (destination/'results.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    # A full-route crossfade is inexpensive and is explicitly separate from native.
    concat=original/'frames.ffconcat'
    full=destination/'scene_crossfade.mp4'
    capture=cv2.VideoCapture(source['video']);assert capture.isOpened()
    native_duration=capture.get(cv2.CAP_PROP_FRAME_COUNT)/capture.get(cv2.CAP_PROP_FPS);capture.release()
    duration=0.
    if full.exists():
        capture=cv2.VideoCapture(str(full));assert capture.isOpened()
        duration=capture.get(cv2.CAP_PROP_FRAME_COUNT)/capture.get(cv2.CAP_PROP_FPS);capture.release()
    if not full.exists() or abs(duration-native_duration)>.05:
        subprocess.run([args.ffmpeg,'-hide_banner','-loglevel','error','-y','-threads','4',
            '-safe','0','-i',str(concat),'-vf','tpad=stop_mode=clone:stop_duration=1,minterpolate=fps=30:mi_mode=blend',
            '-t',str(native_duration),
            '-c:v','libx264','-preset','fast','-crf','16','-pix_fmt','yuv420p','-movflags','+faststart',str(full)],check=True)
    report['full_crossfade']='scene_crossfade.mp4'
    capture=cv2.VideoCapture(str(full));assert capture.isOpened()
    report['full_crossfade_duration_seconds']=capture.get(cv2.CAP_PROP_FRAME_COUNT)/capture.get(cv2.CAP_PROP_FPS)
    assert abs(report['full_crossfade_duration_seconds']-native_duration)<.05
    capture.release()
    flow=destination/'scene_motion_compensated.mp4'
    flow_valid=False
    if flow.exists():
        capture=cv2.VideoCapture(str(flow))
        if capture.isOpened():
            duration=capture.get(cv2.CAP_PROP_FRAME_COUNT)/capture.get(cv2.CAP_PROP_FPS)
            flow_valid=abs(duration-native_duration)<.05 and capture.get(cv2.CAP_PROP_FRAME_WIDTH)==960
        capture.release()
    if args.full_flow and not flow_valid:
        temporary=flow.with_name('scene_motion_compensated.partial.mp4')
        subprocess.run([args.ffmpeg,'-hide_banner','-loglevel','error','-y','-threads','4',
            '-safe','0','-i',str(concat),'-vf','tpad=stop_mode=clone:stop_duration=1,minterpolate=fps=30:mi_mode=mci:mc_mode=aobmc:me_mode=bidir:vsbmc=1',
            '-t',str(native_duration),'-c:v','libx264','-preset','fast','-crf','16','-pix_fmt','yuv420p',
            '-movflags','+faststart',str(temporary)],check=True)
        temporary.replace(flow)
    if flow.exists():
        capture=cv2.VideoCapture(str(flow));assert capture.isOpened()
        duration=capture.get(cv2.CAP_PROP_FRAME_COUNT)/capture.get(cv2.CAP_PROP_FPS)
        assert abs(duration-native_duration)<.05 and capture.get(cv2.CAP_PROP_FRAME_WIDTH)==960
        capture.release();report['full_flow']=flow.name;report['full_flow_duration_seconds']=duration
    (destination/'results.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    out=ROOT/'outputs/gnt_finalization';out.mkdir(exist_ok=True)
    (out/'blending.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    clips=''.join(f'<h2>{c["name"].title()}</h2><div class="grid">'+''.join(
        f'<div><p>{label}</p><video controls muted loop preload="metadata" src="assets/gnt_finalization/blending/{c["name"]}/{key}.mp4"></video></div>'
        for key,label in [('native','Native independent GNT views'),('held','Half-rate held control'),('blend','Crossfade from half-rate views'),('flow','Motion compensation from half-rate views')])+'</div>' for c in report['clips'])
    page='''<!doctype html><meta charset="utf-8"><title>GNT motion tests</title><style>body{background:#111;color:#eee;font:16px system-ui;margin:24px}a{color:#9bd}.grid{display:grid;grid-template-columns:1fr 1fr;gap:12px}video{width:100%}p{max-width:1100px;line-height:1.5}</style><h1>GNT: smoother playback tests</h1><p>These are post-render video operations, not a temporal GNT model. Crossfades can ghost; motion compensation can distort occlusions. The native scene remains the quality reference. Below, alternate GNT frames are deliberately withheld so interpolation can be compared to independent renders.</p><p><a href="gnt-traversal.html">Native full route</a> | <a href="assets/gnt_finalization/blending/scene_crossfade.mp4">Full-route crossfade</a> | <a href="gnt-objects.html">Object adaptation</a></p>'''+clips
    if flow.exists():page=page.replace('Full-route crossfade</a>','Full-route crossfade</a> | <a href="assets/gnt_finalization/blending/scene_motion_compensated.mp4">Full-route motion compensation</a>')
    (ROOT/'web_demo/gnt-motion.html').write_text(page,encoding='utf-8')

if __name__=='__main__':main()
