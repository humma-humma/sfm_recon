"""Encode and publish isolated teacher-guided splat comparisons."""
import argparse
import json
from pathlib import Path
import shutil
import subprocess
import urllib.request
from publish_gnt_videos import verify

ROOT=Path(__file__).resolve().parents[1]
TITLES={'baseline':'Existing scene splat','photos_10000':'Matched photos-only fit, 10k',
    'mixed_5000':'Photos + sparse GNT views, 5k','mixed_10000':'Photos + sparse GNT views, 10k',
    'teacher_10000':'GNT-only RGB fit, 10k',
    'geometry_12000':'GNT RGB + training sensor depth, 12k',
    'pruned_12000':'Depth-guided fit + conservative pruning, 12k'}


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--ffmpeg',required=True);args=parser.parse_args()
    base=args.root/'assets/gnt_splat_scene';base.mkdir(parents=True,exist_ok=True)
    results={};videos=[];table=[]
    for label,title in TITLES.items():
        folder=base/label;path=folder/'results.json'
        if not path.exists():continue
        report=json.loads(path.read_text());assert report['status']=='complete'
        video=folder/'turn.mp4'
        if not video.exists():
            temporary=video.with_suffix('.partial.mp4')
            subprocess.run([args.ffmpeg,'-hide_banner','-loglevel','error','-y','-threads','4',
                '-framerate','30','-i',str(folder/'frame_%04d.png'),'-frames:v','180','-an',
                '-c:v','libx264','-preset','fast','-crf','16','-pix_fmt','yuv420p','-movflags','+faststart',str(temporary)],check=True)
            verify(temporary,6);temporary.replace(video)
        report['video']=verify(video,6)
        if report.get('full_route'):
            raw=folder/report['full_route']['raw'];full=folder/'full_route.mp4'
            if not full.exists():
                assert raw.stat().st_size==4395*960*540*3
                temporary=full.with_suffix('.partial.mp4')
                subprocess.run([args.ffmpeg,'-hide_banner','-loglevel','error','-y','-threads','4',
                    '-f','rawvideo','-pixel_format','rgb24','-video_size','960x540','-framerate','30',
                    '-i',str(raw),'-frames:v','4395','-an','-c:v','libx264','-preset','fast','-crf','16',
                    '-pix_fmt','yuv420p','-movflags','+faststart',str(temporary)],check=True)
                verify(temporary,146.5);temporary.replace(full)
            report['full_route']['video']=verify(full,146.5)
            if raw.exists():raw.unlink()  # Disposable raw RGB stream, after verified video export.
        url=f'assets/gnt_finalization/gnt_splat_scene/{label}/turn.mp4'
        with urllib.request.urlopen(urllib.request.Request('http://localhost:8088/'+url,headers={'Range':'bytes=0-1023'})) as response:
            assert response.status==206 and len(response.read())==1024
        results[label]=report;videos.append((title,url))
        metrics=report['means']
        table.append(f'<tr><td>{title}</td><td>{metrics["gnt_psnr"]:.2f}</td><td>{metrics["photo_psnr"]:.2f}</td><td>{metrics["render_seconds"]*1000:.1f}</td><td>{report["gaussians"]:,}</td></tr>')
    videos.insert(0,('Approved genuine GNT, 30 fps','assets/gnt_finalization/dense_turn/scene_turn_genuine_30fps.mp4'))
    panels=''.join(f'<section id="{url.split("/")[-2]}"><h2>{title}</h2><video controls muted playsinline preload="metadata" src="{url}"></video><a href="{url}" download>Download</a></section>' for title,url in videos)
    if 'mixed_10000' in results and results['mixed_10000'].get('full_route'):
        panels+='<h2>GNT-guided splat: full scene traversal</h2><video controls preload="metadata" src="assets/gnt_finalization/gnt_splat_scene/mixed_10000/full_route.mp4"></video>'
    pictures=[]
    reference=base/'reference';reference.mkdir(exist_ok=True)
    for i in (0,96,120,240):
        source=args.root/'gnt_splat_scene/data/photos/images'/f'photo_{i:04}.jpg'
        target=reference/source.name
        if not target.exists():shutil.copy2(source,target)
        pictures.append(f'<h3>Held-out photograph {i}</h3><div class="stills"><figure><img loading="lazy" src="assets/gnt_finalization/gnt_splat_scene/reference/{source.name}"><figcaption>Original photo</figcaption></figure>')
        for label in results:
            pictures.append(f'<figure><a href="assets/gnt_finalization/gnt_splat_scene/{label}/photo_{i:04}.png"><img loading="lazy" src="assets/gnt_finalization/gnt_splat_scene/{label}/photo_{i:04}.png"></a><figcaption>{TITLES[label]}</figcaption></figure>')
        pictures.append('</div>')
    html='''<!doctype html><meta charset="utf-8"><title>GNT-guided splat fitting</title>
<style>body{max-width:1200px;margin:24px auto;background:#121720;color:#eee;font:16px system-ui}video{width:100%;max-width:960px}a{color:#8cf}td,th{padding:10px;text-align:left}.stills{display:grid;grid-template-columns:repeat(auto-fit,minmax(250px,1fr));gap:8px}figure{margin:0}img{width:100%}button{padding:10px;margin:8px}</style>
<h1>GNT-guided splat fitting</h1><p>Separate candidate models; existing selections are unchanged. Mixed fits combine 275 sparse GNT views with 241 original training photographs. GNT-only uses only the 275 rendered RGB targets. The geometry variant continues GNT-only for 2k steps using GNT RGB and confidence-filtered sensor depth from the 241 training cameras, with no original-photo RGB loss; its comparison also changes training duration. Pruning removes repeated free-space conflicts with no surface support. The dense GNT clip and 35 held-out photographs/depth maps are excluded from these losses. Inherited camera estimates and seed points predate the split. These are local Splatfacto experiments, not reproductions of ReSplat, DepthSplat or RadSplat. All splat videos render genuine camera views without motion interpolation.</p>
<table><tr><th>Model</th><th>Dense GNT agreement (PSNR)</th><th>Held-out photo PSNR</th><th>Median GPU render ms</th><th>Gaussians</th></tr>__TABLE__</table><p>GNT agreement measures similarity to the teacher, not ground-truth reconstruction. GPU timing excludes encoding and image IO. Direct gsplat SH3/classic render, black background.</p>
<button id="play">Play six-second clips from start</button><button id="pause">Pause all</button>__VIDEOS__<h2>Photographic checks</h2>__STILLS__
<script>const videos=[...document.querySelectorAll('video')];document.getElementById('play').onclick=()=>videos.filter(v=>!v.src.endsWith('full_route.mp4')).forEach(v=>{v.currentTime=0;v.play();});document.getElementById('pause').onclick=()=>videos.forEach(v=>v.pause());</script>'''.replace('__TABLE__',''.join(table)).replace('__VIDEOS__',panels).replace('__STILLS__',''.join(pictures))
    (ROOT/'web_demo/gnt-splat-fit.html').write_text(html,encoding='utf-8')
    report=dict(status='complete' if {'baseline','photos_10000','mixed_5000','mixed_10000'}<=results.keys() else 'running',candidates=results,
        noise_trials_status='complete' if {'teacher_10000','geometry_12000','pruned_12000'}<=results.keys() else 'pending',
        selected_models_unchanged=True,method='GNT-guided Splatfacto fit with matched photo-only control')
    (ROOT/'outputs/gnt_finalization/splat_fit.json').write_text(json.dumps(report,indent=2))
    print(json.dumps(dict(status=report['status'],models=list(results))),flush=True)


if __name__=='__main__':main()
