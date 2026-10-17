"""Verify and review completed native CUDA close-up renders without model edits."""
from pathlib import Path
import json
import numpy as np
from PIL import Image, ImageDraw
from skimage.metrics import peak_signal_noise_ratio, structural_similarity

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'outputs/splat_closeups'

def read(path):return np.asarray(Image.open(path).convert('RGB'))

def main():
    native=json.loads((OUT/'native_report.json').read_text())
    local=json.loads((OUT/'report.json').read_text())
    assert native['status']=='complete' and native['renders']==216
    summary=dict(native_renders=216,local_renders=504,total_renders=720,
        scope='4 datasets, 3 captured cameras each, 6 camera distance/zoom cases. Native gsplat rasterizer, not full Nerfstudio pipeline. Browser UI remains untested.',
        metrics_note='Source PSNR/SSIM only for captured cameras. Novel close-ups have no ground truth. Guard/native PSNR measures renderer agreement, not scene accuracy. Square-pixel centered intrinsics are shared by diagnostic renderers.',datasets={})
    lines=['# Completed native close-up comparison','',summary['scope'],'',summary['metrics_note'],'',
        '| Dataset | Native variant | Captured PSNR dB | Captured SSIM | Median native render ms |',
        '|---|---|---:|---:|---:|']
    for name,dataset in native['datasets'].items():
        assert dataset['checkpoint']==local['datasets'][name]['checkpoint']
        assert len(dataset['renders'])==54
        metrics=[];agreement=[]
        native_keys=set()
        for row in dataset['renders']:
            file=OUT/row['file'];assert file.exists() and Image.open(file).size==(960,540)
            key=(row['case_id'],row['variant']);assert key not in native_keys;native_keys.add(key)
            image=read(file)
            if row['case_id'].endswith('_captured'):
                index=str(int(row['case_id'][4:7]));reference=read(OUT/local['datasets'][name]['references'][index]);ground=reference[:,:reference.shape[1]//2]
                metrics.append(dict(case_id=row['case_id'],variant=row['variant'],psnr=float(peak_signal_noise_ratio(ground,image,data_range=255)),ssim=float(structural_similarity(ground,image,channel_axis=2,data_range=255,gaussian_weights=True,sigma=1.5,use_sample_covariance=False))))
            if row['variant']=='native_sh3':
                guarded=read(OUT/f'{name}/{row["case_id"]}_full_sh_guard.png')
                agreement.append(dict(case_id=row['case_id'],psnr=float(peak_signal_noise_ratio(image,guarded,data_range=255)),mean_abs_rgb_0_255=float(np.abs(image.astype(float)-guarded).mean())))
        variants={}
        for variant in ['native_sh0','native_sh3','native_sh3_antialiased']:
            chosen=[r for r in metrics if r['variant']==variant];assert len(chosen)==3
            times=[r['gpu_render_ms'] for r in dataset['renders'] if r['variant']==variant];assert len(times)==18
            variants[variant]=dict(captured_psnr=float(np.mean([r['psnr'] for r in chosen])),captured_ssim=float(np.mean([r['ssim'] for r in chosen])),median_native_render_ms=float(np.median(times)))
            v=variants[variant];lines.append(f'| {name} | {variant} | {v["captured_psnr"]:.2f} | {v["captured_ssim"]:.3f} | {v["median_native_render_ms"]:.2f} |')
        summary['datasets'][name]=dict(variants=variants,captured_views=metrics,guard_native_agreement=agreement,
            median_guard_native_psnr=float(np.median([r['psnr'] for r in agreement])))
    lines+=['','## Renderer agreement','', '| Dataset | Guarded full-SH shader vs native full-SH, median PSNR dB |','|---|---:|']
    for name,dataset in summary['datasets'].items():lines.append(f'| {name} | {dataset["median_guard_native_psnr"]:.2f} |')
    lines+=['','Native timings include projection, native sorting, color evaluation and rasterization, excluding loading/readback/PNG writing. Local draw-only timings exclude sorting and CPU SH evaluation, so these timing columns must not be treated as browser FPS or directly compared as end-to-end runtimes. Only two timed samples per case remain after warm-up; timings are diagnostic.','',
        'Antialiased is inference-only filtering on classic-trained models; this does not test a Mip-Splatting retrain. No source checkpoints or web demo settings were modified. Native output images and gallery links were verified; browser interaction and complete Nerfstudio pipeline rendering were not tested.','',
        'Open native.html for all native comparisons and index.html for all isolated local shader tests.']
    (OUT/'NATIVE_RESULTS.md').write_text('\n'.join(lines),encoding='utf-8')
    (OUT/'native_summary.json').write_text(json.dumps(summary,indent=2))
    columns=[('browser','Current demo'),('full_sh_guard','Guarded + full SH'),('native_sh3','Native classic + full SH'),('native_sh3_antialiased','Native antialiased + full SH')]
    for case in ['captured','dolly25','dolly50','optical4x']:
        sheet=Image.new('RGB',(1600,4*255),'#172019');draw=ImageDraw.Draw(sheet)
        for y,name in enumerate(native['datasets']):
            # Use the middle scene camera to show stairs; all 3 cameras remain in gallery.
            view=136 if name=='scene' else 0
            for x,(variant,label) in enumerate(columns):
                im=Image.open(OUT/f'{name}/view{view:03d}_{case}_{variant}.png').resize((400,225))
                sheet.paste(im,(400*x,255*y+30));draw.text((400*x+7,255*y+8),f'{name} view{view:03d} / {label}',fill='white')
        sheet.save(OUT/f'native_decision_{case}.jpg',quality=95)
    # Clear only stale blocker labels now superseded by the completed native run.
    local['native_cuda_comparison']='complete: 216 renders; native_report.json'
    local['nerfstudio_closeups']='Full pipeline not run; native gsplat rasterizer comparison completed separately.'
    (OUT/'report.json').write_text(json.dumps(local,indent=2))
    html=(OUT/'index.html').read_text(encoding='utf-8')
    html=html.replace('Novel close-up Nerfstudio rendering remains blocked by Docker access.', 'Native gsplat close-up comparisons are complete: <a href="native.html">open native comparison gallery</a>. Full Nerfstudio pipeline rendering was not run.')
    (OUT/'index.html').write_text(html,encoding='utf-8')
    print('\n'.join(lines),flush=True)

if __name__=='__main__':main()
