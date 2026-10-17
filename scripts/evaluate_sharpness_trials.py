"""Compare photo reprojection, surface/scale changes and a pretrained GAN prior.

Host Open3D/PyTorch environment. Scratch renders and weights stay outside the
nearly-full project drive. Persistent results contain only JPEGs and reports.
"""
import argparse
import hashlib
import html
import json
from pathlib import Path
from time import perf_counter
import cv2
import numpy as np
import open3d as o3d
import torch
from skimage.metrics import structural_similarity
from sfm_reconstruction.photo_reprojection import photo_render, project
from vendor.rrdbnet_arch import RRDBNet

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT/'outputs/sharpness_trials'
ASSETS = ROOT/'web_demo/assets/sharpness'
METHODS = ('baseline', 'photo', 'scale', 'surface_photo', 'prior', 'baseline_prior')


def read(path):
    bgr = cv2.imread(str(path))
    if bgr is None: raise FileNotFoundError(path)
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB).astype(np.float32)/255


def write(path, rgb):
    cv2.imwrite(str(path), cv2.cvtColor(np.uint8(np.clip(rgb, 0, 1)*255+.5), cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 90])


def quality(gt, image):
    mse = float(np.mean((gt-image)**2))
    return dict(psnr=float(-10*np.log10(max(mse, 1e-12))),
                ssim=float(structural_similarity(gt, image, channel_axis=2, data_range=1)))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--scratch', type=Path, required=True)
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True); ASSETS.mkdir(parents=True, exist_ok=True)
    manifest = json.loads((args.scratch/'render_manifest.json').read_text())
    weights_path = args.scratch/'RealESRGAN_x2plus.pth'
    net = RRDBNet(3, 3, scale=2, num_feat=64, num_block=23, num_grow_ch=32)
    net.load_state_dict(torch.load(weights_path, map_location='cpu', weights_only=True)['params_ema'], strict=True)
    net = net.eval().cuda().half()
    def restore(image):
        tensor = torch.from_numpy(image.transpose(2, 0, 1).copy()).unsqueeze(0).cuda().half()
        torch.cuda.synchronize(); tick = perf_counter()
        with torch.inference_mode(): result = net(tensor).float()[0].permute(1, 2, 0).cpu().numpy().clip(0, 1)
        torch.cuda.synchronize()
        seconds = perf_counter()-tick
        # Fidelity comparisons use exactly the same resolution as the reference.
        return cv2.resize(result, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_AREA), result, seconds
    mesh = o3d.io.read_triangle_mesh(str(ROOT/'outputs/depth_research/scene_tsdf.ply'))
    raycaster = o3d.t.geometry.RaycastingScene()
    raycaster.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(mesh))
    def mesh_depth(pose, k):
        r, t = pose
        yy, xx = np.indices((k[5], k[4]), dtype=np.float32)
        direction = np.stack(((xx-k[2])/k[0], (yy-k[3])/k[1], np.ones_like(xx)), axis=-1)@r
        rays = np.concatenate((np.broadcast_to(-r.T@t, direction.shape), direction), axis=-1).astype(np.float32)
        result = raycaster.cast_rays(o3d.core.Tensor(rays))['t_hit'].numpy()
        result[~np.isfinite(result)] = 0
        return result
    report = dict(status='running', methods={
        'baseline': 'Fixed selected checkpoint, direct classic gsplat, black background.',
        'photo': 'Training-photo reprojection, expected splat depth visibility, >=2 agreeing sources, 90% maximum photo contribution.',
        'scale': 'Same fixed checkpoint, gsplat antialiased rasterization; inference-only, not full Mip-Splatting retraining.',
        'surface_photo': 'Scene: train-only TSDF-guided photo reprojection with antialiased fallback. Objects: photo reprojection with antialiased fallback; no TSDF available.',
        'prior': 'Official pretrained Real-ESRGAN x2plus applied to surface_photo, reduced back to source resolution for metrics.',
        'baseline_prior': 'Same pretrained model on baseline alone, to isolate learned restoration from photo reuse.',
    }, weights=dict(url='https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.1/RealESRGAN_x2plus.pth',
                    sha256=hashlib.sha256(weights_path.read_bytes()).hexdigest(), architecture='RRDBNet x2, 23 blocks, strict params_ema loading, FP16'),
       limitations=[
           'Rendering pilot, not IBRNet, ReconFusion, full Mip-Splatting or 2DGS reproduction; no learned multiview model was trained.',
           'Existing poses/seed clouds precede the rendering split. Target RGB is used for evaluation only; all projected source RGB excludes interval-8 held-out views.',
           'Real-ESRGAN is a per-frame GAN restoration prior, not a multiview-conditioned reconstruction prior. Apparent detail may be invented and may change with motion.',
           'Motion has no ground-truth photographs. Warp residual uses estimated splat depth and cannot prove correct geometry or temporal stability.',
           'Object surface_photo differs only in its antialiased fallback. TSDF surface substitution is tested only for scene.',
           'Results use 960x540 exported photos, three held-out object views and six scene views, at 1x and 1.7x focal length; not a full benchmark.',
           'Motion is +/-0.02 original COLMAP world units. Metric scale of all object reconstructions was not independently verified.',
           'GPU is shared with the existing Docker stack. Times are observational, not isolated benchmarks.',
       ], datasets={})
    for dataset, record in manifest['datasets'].items():
        start = perf_counter(); folder = args.scratch/dataset
        target = ASSETS/dataset; target.mkdir(exist_ok=True)
        source_cache = {}
        for index, source in record['sources'].items():
            pose = (np.array(source['r']), np.array(source['t']))
            source_cache[int(index)] = dict(rgb=read(folder/f'source_{int(index):03d}.png'), pose=pose,
                k=record['k'], depth=np.load(folder/f'source_{int(index):03d}_depth.npy'))
            if dataset == 'scene': source_cache[int(index)]['surface'] = mesh_depth(pose, record['k'])
        rows, motion = [], []
        for case in record['cases']:
            tick = perf_counter()
            pose = (np.array(case['r']), np.array(case['t']))
            depth = np.load(folder/f'{case["id"]}_depth.npy')
            images = dict(baseline=read(folder/f'{case["id"]}_classic.png'), scale=read(folder/f'{case["id"]}_antialiased.png'))
            sources = [dict(source_cache[index], weight=weight) for index, weight in zip(case['sources'], case['weights'])]
            assert set(case['sources']) <= set(record['train_indices'])
            images['photo'], mask = photo_render(images['baseline'], depth, pose, case['k'], sources)
            photo_seconds = perf_counter()-tick
            tick = perf_counter()
            surface = mesh_depth(pose, case['k']) if dataset == 'scene' else depth
            surface_sources = [dict(s, depth=s['surface']) for s in sources] if dataset == 'scene' else sources
            images['surface_photo'], surface_mask = photo_render(images['scale'], surface, pose, case['k'], surface_sources)
            surface_seconds = perf_counter()-tick
            images['prior'], high, prior_seconds = restore(images['surface_photo'])
            images['baseline_prior'], _, baseline_prior_seconds = restore(images['baseline'])
            if case['kind'] == 'heldout':
                gt = read(folder/f'{case["id"]}_gt.png')
                write(target/f'{case["id"]}_gt.jpg', gt)
                metrics = {method: quality(gt, image) for method, image in images.items()}
            else:
                metrics = {}
                motion.append(dict(images={key: value for key, value in images.items()}, pose=pose, k=case['k'], depth=depth, surface=surface))
            if case['kind'] == 'heldout' and case['index'] == record['cases'][2]['index'] and case['zoom'] == 1.7:
                write(target/'prior_2x_detail.jpg', high)
            for method, image in images.items(): write(target/f'{case["id"]}_{method}.jpg', image)
            row = dict(case, metrics=metrics, photo_coverage=float((mask > .1).mean()),
                       surface_photo_coverage=float((surface_mask > .1).mean()), surface_valid_fraction=float((surface > 0).mean()),
                       seconds=dict(baseline=case['classic_seconds'], scale=case['antialiased_seconds'],
                                    photo_cpu=photo_seconds, surface_photo_cpu=surface_seconds,
                                    prior_gpu=prior_seconds, baseline_prior_gpu=baseline_prior_seconds))
            rows.append(row)
            print(dataset, case['id'], 'photo', round(row['photo_coverage'], 3), 'surface', round(row['surface_photo_coverage'], 3), flush=True)
        def temporal_errors(depth_key):
            result = {method: [] for method in METHODS}
            for previous, current in zip(motion, motion[1:]):
                x, y, z = project(current[depth_key], current['pose'], previous['pose'], current['k'], previous['k'])
                sampled_depth = cv2.remap(previous[depth_key], x, y, cv2.INTER_NEAREST)
                h, w = current[depth_key].shape
                mask = ((current[depth_key] > 0) & (sampled_depth > 0) & (z > 0) &
                        (np.abs(sampled_depth-z) < .03+.03*z) & (x >= 2) & (x < w-3) & (y >= 2) & (y < h-3))
                mask = cv2.erode(mask.astype(np.uint8), np.ones((3, 3), np.uint8)).astype(bool)
                for method in result:
                    warped = cv2.remap(previous['images'][method], x, y, cv2.INTER_LINEAR)
                    error = float(np.abs(current['images'][method]-warped)[mask].mean()) if mask.any() else None
                    result[method].append(dict(warp_mae=error, compared_fraction=float(mask.mean())))
            return result
        temporal = temporal_errors('depth')
        temporal_tsdf = temporal_errors('surface') if dataset == 'scene' else None
        means = {method: {metric: float(np.mean([r['metrics'][method][metric] for r in rows if r['kind'] == 'heldout']))
                          for metric in ('psnr', 'ssim')} for method in METHODS}
        # A single fixed promotion gate; selected presentations remain untouched regardless.
        gates = {}
        for method in METHODS[1:]:
            deltas = [r['metrics'][method]['psnr']-r['metrics']['baseline']['psnr'] for r in rows if r['kind'] == 'heldout']
            gates[method] = dict(mean_psnr_delta=float(np.mean(deltas)), worst_psnr_delta=float(min(deltas)),
                mean_ssim_delta=means[method]['ssim']-means['baseline']['ssim'],
                passes_fidelity_gate=bool(np.mean(deltas) >= -.1 and min(deltas) >= -.5 and means[method]['ssim']-means['baseline']['ssim'] >= -.002))
        report['datasets'][dataset] = dict(cases=rows, mean=means, gates=gates, temporal=temporal,
            source_checkpoint=record['checkpoint'], source_checkpoint_sha256=record['checkpoint_sha256'],
            source_views=len(source_cache), seconds=perf_counter()-start,
            temporal_mean={method: float(np.mean([r['warp_mae'] for r in values if r['warp_mae'] is not None])) for method, values in temporal.items()})
        if temporal_tsdf:
            report['datasets'][dataset]['temporal_tsdf'] = temporal_tsdf
            report['datasets'][dataset]['temporal_tsdf_mean'] = {method: float(np.mean([r['warp_mae'] for r in values if r['warp_mae'] is not None])) for method, values in temporal_tsdf.items()}
        (OUT/'results.json').write_text(json.dumps(report, indent=2))
    report['status'] = 'complete'
    for dataset, record in manifest['datasets'].items():
        assert hashlib.sha256((ROOT/record['checkpoint']).read_bytes()).hexdigest() == record['checkpoint_sha256'], dataset
    report['selected_models_unchanged'] = True
    (OUT/'results.json').write_text(json.dumps(report, indent=2))
    (ASSETS/'results.json').write_text(json.dumps(report, indent=2))
    build_page(report)
    print('Complete: web_demo/sharpness.html', flush=True)


def build_page(report):
    labels = dict(baseline='Original splats', photo='1 · Photograph-guided', scale='2a · Antialiased splats',
                  surface_photo='2b · Surface + photographs', prior='3 · Surface/photo + learned prior', baseline_prior='3 control · Prior on original')
    parts = ['''<!doctype html><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Sharpness experiments</title>
<style>body{background:#15191e;color:#eee;font:16px system-ui;margin:24px}a{color:#9ed6ff}p{max-width:1100px;line-height:1.6}.grid{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:12px}img{width:100%;display:block}figure{margin:0}figcaption{padding:8px}button,select,input{margin:8px;padding:6px}table{border-collapse:collapse}td,th{padding:8px;border:1px solid #777}@media(max-width:900px){.grid{grid-template-columns:1fr}}</style>
<h1>Sharpness experiments</h1><p><a href="index.html">Main demo</a> · <a href="assets/sharpness/results.json">Metrics and method details</a></p>
<p>Click an image for full resolution. Photographs used for rendering are training views only. Reference photos are held out. 1.7× means a closer focal framing of the same camera. Selected presentation models remain unchanged.</p>
<p>Stage 2 uses gsplat inference antialiasing, not full Mip-Splatting training. Only scene has a TSDF surface; object stage 2b uses the same splat depth with antialiased fallback. Stage 3 uses pretrained per-frame Real-ESRGAN, not a multiview diffusion reconstruction. Sharper-looking output can contain invented detail.</p>''']
    inspection = {dataset: [dict(id=c['id'], image=c['image'], zoom=c['zoom']) for c in record['cases'] if c['kind'] == 'heldout'] for dataset, record in report['datasets'].items()}
    parts.append('''<section id="inspector"><h2>Inspect the same pixels</h2>
<p>Candidate on the left, original splats on the right. Move the wipe or use Toggle A/B. Zoom and drag to inspect the same region in both images. Choose Reference photograph to see the remaining gap.</p>
<label>Dataset <select id="inspect-dataset"></select></label><label>View <select id="inspect-case"></select></label>
<label>Candidate <select id="inspect-method"></select></label><br>
<label>Wipe <input id="inspect-wipe" type="range" min="0" max="100" value="50"></label>
<button id="inspect-toggle" type="button">Toggle A/B</button><label>Zoom <input id="inspect-zoom" type="range" min="1" max="4" step="0.25" value="1"></label>
<button id="inspect-reset" type="button">Reset crop</button><p id="inspect-status" role="status"></p>
<canvas id="inspect-canvas" width="960" height="540" style="display:block;width:100%;max-width:960px;touch-action:none;cursor:grab" aria-label="Shared crop comparison. Drag to pan; use zoom and wipe sliders."></canvas></section>''')
    parts.append('<script type="application/json" id="inspect-data">'+json.dumps(dict(cases=inspection, labels=dict(gt='Reference photograph', **labels))).replace('<', '\\u003c')+'</script><script src="sharpness-inspector.js" defer></script>')
    for dataset, record in report['datasets'].items():
        parts.append(f'<h2>{dataset.title()}</h2><table><tr><th>Method</th><th>Mean PSNR</th><th>Mean SSIM</th><th>JPEG LPIPS ↓</th><th>Motion warp MAE ↓</th></tr>')
        for method in METHODS:
            mean = record['mean'][method]
            motion = record.get('temporal_tsdf_mean', record['temporal_mean']).get(method)
            perceptual = mean.get('lpips_jpeg')
            parts.append(f'<tr><td>{labels[method]}</td><td>{mean["psnr"]:.3f}</td><td>{mean["ssim"]:.4f}</td><td>{"—" if perceptual is None else f"{perceptual:.4f}"}</td><td>{"—" if motion is None else f"{motion:.4f}"}</td></tr>')
        parts.append('</table><p>Motion error uses a common TSDF warp for scene and a common splat-depth warp for objects; the JSON includes both scene warps. Each warp uses the same mask for all methods. No ground-truth motion sequence is available: this is a diagnostic, not proof of stable or correct 3D.</p>')
        parts.append(f'<div class="motion" data-dataset="{dataset}"><button type="button">Play / pause motion</button><input type="range" min="0" max="8" value="0" step="1" aria-label="Motion frame"><span>0 / 8</span><div class="grid">')
        for method in METHODS:
            path = f'assets/sharpness/{dataset}/motion_00_{method}.jpg'
            parts.append(f'<figure><figcaption>{labels[method]}</figcaption><img data-method="{method}" src="{path}" alt="{dataset} {labels[method]} motion"></figure>')
        parts.append('</div></div>')
        for case in record['cases']:
            if case['kind'] != 'heldout': continue
            parts.append(f'<h3>{html.escape(case["image"])} · {case["zoom"]}×</h3><p>Photo coverage: {case["photo_coverage"]:.1%}; surface/photo coverage: {case["surface_photo_coverage"]:.1%}</p><div class="grid">')
            for method in ('gt', *METHODS):
                path = f'assets/sharpness/{dataset}/{case["id"]}_{method}.jpg'
                label = 'Held-out reference photograph' if method == 'gt' else labels[method]
                parts.append(f'<figure><figcaption>{label}</figcaption><a href="{path}" target="_blank"><img loading="lazy" src="{path}" alt="{dataset} {label}"></a></figure>')
            parts.append('</div>')
        parts.append(f'<p><a href="assets/sharpness/{dataset}/prior_2x_detail.jpg">One 2× learned-prior output</a> (upsampled output; metrics above use source resolution).</p>')
    parts.append('''<script>
for(const section of document.querySelectorAll('.motion')){
 const slider=section.querySelector('input'),label=section.querySelector('span');let timer=null,generation=0;
 async function update(){const token=++generation,frame=slider.value,images=[...section.querySelectorAll('img')];
  const paths=images.map(image=>'assets/sharpness/'+section.dataset.dataset+'/motion_'+String(frame).padStart(2,'0')+'_'+image.dataset.method+'.jpg');
  try{await Promise.all(paths.map(async path=>{const image=new Image();image.src=path;await image.decode();}));
   if(token!==generation)return;images.forEach((image,i)=>image.src=paths[i]);label.textContent=frame+' / 8';
  }catch{clearInterval(timer);timer=null;label.textContent='Frame unavailable';}
 }
 slider.addEventListener('input',update);
 section.querySelector('button').addEventListener('click',()=>{if(timer){clearInterval(timer);timer=null;}else timer=setInterval(()=>{slider.value=(Number(slider.value)+1)%9;update();},350);});
 document.addEventListener('visibilitychange',()=>{if(document.hidden){clearInterval(timer);timer=null;}});
}
</script>''')
    (ROOT/'web_demo/sharpness.html').write_text('\n'.join(parts), encoding='utf-8')


if __name__ == '__main__': main()
