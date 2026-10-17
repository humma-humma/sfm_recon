"""Source/full-resolution vs training-resolution vs captured render detail review."""
from pathlib import Path
import json
import numpy as np
from PIL import Image,ImageDraw
from skimage.metrics import peak_signal_noise_ratio,structural_similarity

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'outputs/splat_detail_review'
CLOSE=ROOT/'outputs/splat_closeups'
REGIONS={
    'boot':{0:(340,140,580,380),24:(250,230,490,470),48:(270,230,510,470)},
    'scene':{0:(240,80,720,320),136:(180,160,660,400),272:(180,80,660,320)},
}

def main():
    OUT.mkdir(parents=True,exist_ok=True)
    report=json.loads((CLOSE/'report.json').read_text())
    manifest=json.loads((ROOT/'web_demo/assets/manifest.json').read_text())
    summary=dict(method='Selected, explicitly located detail crops at captured camera poses. Original capture is 1920x1080; training/export input is 960x540. Crops share coordinates. Novel camera close-ups have no ground truth.',datasets={})
    html=['<!doctype html><meta charset="utf-8"><title>Boot and scene detail review</title><style>body{font:14px Segoe UI;background:#111710;color:#edf0e9;margin:24px;line-height:1.6}img{width:100%;display:block}a{color:#c8e5b0}.grid{display:grid;grid-template-columns:repeat(2,1fr);gap:12px}p{max-width:1000px}</style><h1>Where is detail lost?</h1><p>Left to right in each sheet: original 1920x1080 capture, actual 960x540 input, saved Nerfstudio render at that camera, native gsplat full-SH render at that camera. Each row is a different view. Regions are specified in report.json. These enlarged crops show detail, not new ground truth close-up captures.</p>']
    for name,regions in REGIONS.items():
        rows=[];panels=[]
        for index,box in regions.items():
            view=manifest[name]['views'][index];image_name=view['name']
            if name=='boot':raw_path=ROOT.parent/'Experiments/Stage_2_Data/stage2/boot/images'/image_name
            else:raw_path=ROOT.parent/'Experiments/Stage_3_Data/stage3/rgb'/Path(image_name).with_suffix('.png')
            raw=Image.open(raw_path).convert('RGB');pair=Image.open(CLOSE/report['datasets'][name]['references'][str(index)]).convert('RGB')
            source=pair.crop((0,0,960,540));nerf=pair.crop((960,0,1920,540))
            native=Image.open(CLOSE/f'{name}/view{index:03d}_captured_native_sh3.png').convert('RGB')
            assert raw.size==(1920,1080) and source.size==(960,540)
            raw_box=tuple(v*2 for v in box)
            sources=[('Original capture',raw.crop(raw_box)),('Half-resolution input',source.crop(box)),('Nerfstudio captured pose',nerf.crop(box)),('Native captured pose',native.crop(box))]
            width=400;tile_height=round((box[3]-box[1])*width/(box[2]-box[0]));panel=Image.new('RGB',(width*4,tile_height+32),'#1b251a');draw=ImageDraw.Draw(panel)
            for x,(label,image) in enumerate(sources):
                panel.paste(image.resize((width,tile_height),Image.Resampling.NEAREST),(x*width,32));draw.text((x*width+8,8),f'view{index:03d} / {label}',fill='white')
            panels.append(panel)
            reference=np.asarray(source.crop(box));render=np.asarray(nerf.crop(box))
            rows.append(dict(camera_index=index,source_image=image_name,original_path=str(raw_path),region_half_resolution=box,original_resolution=raw.size,input_resolution=source.size,
                nerfstudio_detail_psnr=float(peak_signal_noise_ratio(reference,render,data_range=255)),nerfstudio_detail_ssim=float(structural_similarity(reference,render,channel_axis=2,data_range=255,gaussian_weights=True,sigma=1.5,use_sample_covariance=False))))
        sheet=Image.new('RGB',(1600,sum(p.height for p in panels)),'#1b251a');y=0
        for panel in panels:sheet.paste(panel,(0,y));y+=panel.height
        file=f'{name}_detail.jpg';sheet.save(OUT/file,quality=96)
        summary['datasets'][name]=dict(views=rows)
        html.append(f'<h2>{name.title()}</h2><a href="{file}"><img src="{file}" alt="{name} detail crops"></a>')
        html.append('<h3>Native close-ups: optical zoom vs forward movement</h3><div class="grid">')
        for index in regions:
            for case in ['optical4x','dolly50']:
                image=Image.open(CLOSE/f'{name}/view{index:03d}_{case}_native_sh3.png');file=f'{name}_view{index:03d}_{case}.png';image.save(OUT/file)
                html.append(f'<div><p>view{index:03d} / {case} — no close-up ground truth</p><a href="{file}"><img loading="lazy" src="{file}"></a></div>')
        html.append('</div>')
    summary['findings']={
        'boot':'Source photos contain finer seams/laces/texture than captured-pose renders. Some loss occurs in 1920-to-960 reduction, but visible smearing is already present in Nerfstudio/native at captured poses; it is not solely a browser projection or novel-view problem.',
        'scene':'Sign lettering and pavement texture are visible in source inputs but heavily blurred/missing in captured-pose renders. Current renderer correction cannot recover these. Sampling, poses, training and representation are candidates; this comparison does not isolate their individual contributions.',
        'recommendation':'Keep corrected projection and full SH. Next controlled training tests should isolate full-resolution input (first), then camera optimization or densification changes separately. Preserve selected presentations until per-view and detail-region checks pass. No retraining or pruning was performed.'}
    (OUT/'report.json').write_text(json.dumps(summary,indent=2))
    (OUT/'index.html').write_text('\n'.join(html),encoding='utf-8')
    lines=['# Boot and scene detail diagnosis','',summary['method'],'']
    for name,finding in summary['findings'].items():lines.extend([f'**{name}:** {finding}',''])
    lines+=['These are selected held-out captured views. They establish detail loss in the fitted model at known viewpoints, but do not establish whether the same loss exists on all training views. Camera pose error, image subsampling, optimization and inadequate geometric coverage remain confounded. Full-resolution retraining is a hypothesis to test, not a guaranteed fix.','', 'Open index.html for original/input/render crops and matching native close-ups.']
    (OUT/'RESULTS.md').write_text('\n'.join(lines),encoding='utf-8');print('\n'.join(lines),flush=True)

if __name__=='__main__':main()
