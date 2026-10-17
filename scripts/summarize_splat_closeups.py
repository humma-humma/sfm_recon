"""Summarize diagnostic renders; close-up images have no ground truth."""
from pathlib import Path
import json
import numpy as np
from PIL import Image, ImageDraw
from skimage.metrics import peak_signal_noise_ratio, structural_similarity

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'outputs/splat_closeups'

def main():
    report=json.loads((OUT/'report.json').read_text())
    summary={'scope':'3 captured cameras x 6 movement/zoom cases x 7 variants x 4 datasets = 504 offscreen GPU renders. Browser interaction and native CUDA close-ups remain blocked unless separately run.',
             'metrics_note':'Captured-view source comparisons only; novel close-ups have no ground truth. Same pose, but demo uses centered square-pixel intrinsics rather than exact training intrinsics. These are diagnostic metrics, not benchmark replacements.', 'datasets':{}}
    lines=['# Close-up diagnosis — local GPU tests','',summary['scope'],'',summary['metrics_note'],'',
           '| Dataset | Variant | Captured PSNR dB | Captured SSIM | Median GPU draw ms |','|---|---|---:|---:|---:|']
    for name,dataset in report['datasets'].items():
        rows=[]
        for row in dataset['renders']:
            if row['case']!='captured':continue
            reference=Image.open(OUT/dataset['references'][str(row['camera_index'])]).convert('RGB')
            ground=np.array(reference.crop((0,0,reference.width//2,reference.height)))
            render=np.array(Image.open(OUT/row['file']).convert('RGB'))
            rows.append(dict(camera_index=row['camera_index'],variant=row['variant'],psnr=float(peak_signal_noise_ratio(ground,render,data_range=255)),ssim=float(structural_similarity(ground,render,channel_axis=2,data_range=255,gaussian_weights=True,sigma=1.5,use_sample_covariance=False))))
        grouped={}
        for variant in dict.fromkeys(r['variant'] for r in dataset['renders']):
            chosen=[r for r in rows if r['variant']==variant]
            times=[r['gpu_render_ms'] for r in dataset['renders'] if r['variant']==variant]
            grouped[variant]=dict(captured_psnr=float(np.mean([r['psnr'] for r in chosen])),captured_ssim=float(np.mean([r['ssim'] for r in chosen])),median_gpu_draw_ms=float(np.median(times)))
            g=grouped[variant];lines.append(f'| {name} | {variant} | {g["captured_psnr"]:.2f} | {g["captured_ssim"]:.3f} | {g["median_gpu_draw_ms"]:.2f} |')
        near=[];clamp=[]
        for case_id in dict.fromkeys(r['case_id'] for r in dataset['renders']):
            def read(variant):return np.asarray(Image.open(OUT/f'{name}/{case_id}_{variant}.png'),dtype=float)
            original=read('browser');near.append(np.abs(original-read('near_only')).mean());clamp.append(np.abs(original-read('clamp_only')).mean())
        summary['datasets'][name]=dict(variants=grouped,captured_views=rows,mean_pixel_difference_0_255=dict(near_only=float(np.mean(near)),clamp_only=float(np.mean(clamp))),median_cpu_sort_ms=float(np.median([r['sort_ms'] for r in dataset['renders']])))
    lines+=['','GPU draw times exclude sorting, uploads, SH evaluation and readback. They are not browser FPS. Sorting is CPU NumPy here; the browser uses a JavaScript Worker.','',
            'No checkpoint or web demo settings were changed. Full SH restores view-dependent color; it does not repair geometry. Near plane and projection clamp are isolated in separate columns. Oversized-splat removal is diagnostic and can erase valid surfaces.','',
            'Open index.html for all full-resolution images. Run scripts/run_splat_closeups_cuda.ps1 in a Docker-capable PowerShell for native SH0/full SH/antialiased comparisons. Native output goes to native.html. This is a gsplat rasterizer reference, not the complete Nerfstudio pipeline.']
    (OUT/'summary.json').write_text(json.dumps(summary,indent=2));(OUT/'RESULTS.md').write_text('\n'.join(lines),encoding='utf-8')
    variants=['browser','near_only','clamp_only','full_sh_guard']
    for case in ['captured','dolly25','optical4x']:
        sheet=Image.new('RGB',(1280,4*210),'#182018');draw=ImageDraw.Draw(sheet)
        for y,name in enumerate(report['datasets']):
            for x,variant in enumerate(variants):
                image=Image.open(OUT/f'{name}/view000_{case}_{variant}.png').resize((320,180))
                sheet.paste(image,(x*320,y*210+30));draw.text((x*320+8,y*210+9),f'{name} / {variant}',fill='white')
        sheet.save(OUT/f'decision_{case}.jpg',quality=95)
    print('\n'.join(lines),flush=True)

if __name__=='__main__':main()
