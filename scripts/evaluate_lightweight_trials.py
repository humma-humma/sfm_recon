"""Publish matched JPEG comparisons of actual GNT and 2DGS runs.

Large checkpoints and lossless intermediate images stay in C: temporary storage.
Scores describe delivered JPEGs, not a dataset-wide benchmark.
"""
import argparse
import hashlib
import html
import json
from pathlib import Path
import subprocess
import cv2
import numpy as np
import torch
from skimage.metrics import structural_similarity
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity

ROOT=Path(__file__).resolve().parents[1]
LABELS={'baseline':'Original splats','preferred':'Previously preferred restoration',
        'gnt':'GNT · official pretrained','2dgs':'2DGS · initial pilot',
        '2dgs_default':'2DGS · default-weight check','gt':'Held-out photograph'}


def read(path):
    value=cv2.imread(str(path))
    if value is None:raise FileNotFoundError(path)
    return cv2.cvtColor(value,cv2.COLOR_BGR2RGB)


def write(path,image):
    assert cv2.imwrite(str(path),cv2.cvtColor(image,cv2.COLOR_RGB2BGR),[cv2.IMWRITE_JPEG_QUALITY,90])


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--trial',type=Path,required=True)
    parser.add_argument('--render-scratch',type=Path,required=True)
    args=parser.parse_args()
    torch.set_num_threads(4)
    lpips=LearnedPerceptualImagePatchSimilarity(net_type='alex',normalize=True).eval()
    out=ROOT/'outputs/lightweight_methods';out.mkdir(parents=True,exist_ok=True)
    assets=ROOT/'web_demo/assets/lightweight';assets.mkdir(parents=True,exist_ok=True)
    report=dict(status='complete',methods=LABELS,datasets={},provenance={},limitations=[
        'Common GNT/2DGS pilot: one held-out object view each, two scene views. Additional 2DGS views include focal zooms and more scene cameras; not a full benchmark.',
        'All quality scores use delivered JPEG90 images at 960x540. Restoration controls were already JPEG90.',
        'RGB sources exclude interval-8 held-out targets. Camera/seed geometry was estimated before this split.',
        'GNT uses the authors generalization checkpoint with no scene fine-tuning; eight eligible source photos and 64 inverse-depth samples.',
        '2DGS uses official optimization/rasterization at 10k steps, distortion weight 1000 for bounded objects and upstream general-scene default 0 for scene; normal weight .05. This is not a fully converged 30k reproduction.',
        'After boot regressed and milk collapsed, a follow-up on those two datasets tests upstream default distortion weight 0. Both original failed runs remain visible; this is exploratory setting selection.',
        'Models use different training budgets and initializations. GPU runtime is observational on a shared RTX2080, not an isolated speed benchmark.',
        '2DGS render times measure the native CUDA renderer, not browser FPS. The existing traversable demo retains its selected Gaussian models.',
        'Official 2DGS assumes a centered principal point. Scene calibration differs from image center by about (0.067, 2.424) pixels; this is uncorrected in this pilot and can bias its pixel metrics. Object principal points are centered.',
        'Three motion positions at 480x270 are coarse previews with no ground truth; they cannot establish temporal stability.',
        'Previously preferred control: prior on original for objects, surface/photo plus prior for scene. Learned restoration may invent detail.',
    ])
    for name,repo in [('gnt','GNT'),('2dgs','2dgs')]:
        report['provenance'][name]=dict(repository={'gnt':'https://github.com/VITA-Group/GNT','2dgs':'https://github.com/hbb1/2d-gaussian-splatting'}[name],
            commit=subprocess.check_output(['git','-C',str(args.trial/repo),'rev-parse','HEAD'],text=True).strip())
    weights=args.trial/'GNT/generalization_200000.pth'
    report['provenance']['gnt']['checkpoint_sha256']=hashlib.sha256(weights.read_bytes()).hexdigest()
    # Windows git-submodule invokes sh.exe, which may be denied in restricted sessions.
    # These three paths are pinned by the official checkout; rev-parse is a Git builtin.
    modules=('submodules/diff-surfel-rasterization',
             'submodules/diff-surfel-rasterization/third_party/glm','submodules/simple-knn')
    report['provenance']['2dgs']['submodules']={path:subprocess.check_output(
        ['git','-C',str(args.trial/'2dgs'/path),'rev-parse','HEAD'],text=True).strip() for path in modules}
    manifest=json.loads((args.render_scratch/'render_manifest.json').read_text())
    report['provenance']['controls']={name:{key:record[key] for key in ('checkpoint','checkpoint_sha256','data','heldout_indices','train_indices','k')}
                                      for name,record in manifest['datasets'].items()}
    for name,record in report['provenance']['controls'].items():
        digest=hashlib.sha256()
        with (ROOT/record['checkpoint']).open('rb') as stream:
            for block in iter(lambda:stream.read(1024*1024),b''):digest.update(block)
        if digest.hexdigest()!=record['checkpoint_sha256']:raise RuntimeError(f'Control checkpoint changed: {name}')
        record['checkpoint_unchanged']=True
    report['provenance']['environment']=dict(docker_image='ghcr.io/nerfstudio-project/nerfstudio:1.1.5',
        docker_digest='sha256:b59b8e1012d7a43679d3234b3de9c8416a4b8435fcbf21b9d8c4494b8563f19e',
        gpu='NVIDIA GeForce RTX 2080 8 GiB',inference_torch='2.1.2+cu118',cuda_compiler='11.8.89',
        extension_architecture='sm75',scoring_torch=str(torch.__version__))
    inspection=dict(labels=LABELS,cases={})
    tables=[]
    for dataset in ('box','boot','milk','scene'):
        gnt=args.trial/'results/gnt'/dataset
        gs=args.trial/'results/2dgs'/dataset
        runtime=json.loads((gnt/'runtime.json').read_text())
        motion_runtime=json.loads((gnt/'motion_480/runtime.json').read_text())
        gs_runtime=json.loads((gs/'runtime.json').read_text())
        default_path=gs/'dist_0/runtime.json'
        default_runtime=json.loads(default_path.read_text()) if default_path.exists() else None
        methods=['baseline','preferred','gnt','2dgs']+(['2dgs_default'] if default_runtime else [])
        destination=assets/dataset;destination.mkdir(exist_ok=True)
        records=[];inspection['cases'][dataset]=[]
        common_ids={c['id'] for c in runtime['cases']+motion_runtime['cases']}
        extra=[c for c in gs_runtime['cases'] if c['kind']=='heldout' and c['id'] not in common_ids]
        for case in runtime['cases']+motion_runtime['cases']+extra:
            motion=case['kind']=='motion';key=case['id'];size=(480,270) if motion else (960,540)
            preferred='prior' if dataset=='scene' else 'baseline_prior'
            paths={'baseline':args.render_scratch/dataset/f'{key}_classic.png',
                   'preferred':ROOT/'web_demo/assets/sharpness'/dataset/f'{key}_{preferred}.jpg',
                   '2dgs':gs/f'{key}.png'}
            if key in common_ids:paths['gnt']=gnt/('motion_480' if motion else '')/f'{key}.png'
            if default_runtime:paths['2dgs_default']=gs/'dist_0'/f'{key}.png'
            if not motion:paths['gt']=args.render_scratch/dataset/f'{key}_gt.png'
            images={};urls={}
            for method,path in paths.items():
                image=read(path)
                if (image.shape[1],image.shape[0])!=size:image=cv2.resize(image,size,interpolation=cv2.INTER_AREA)
                output=destination/f'{key}_{method}.jpg'
                if method=='preferred' and not motion:output.write_bytes(path.read_bytes())
                else:write(output,image)
                images[method]=read(output).astype(np.float32)/255
                urls[method]=f'assets/lightweight/{dataset}/{output.name}'
            scores={}
            if not motion:
                gt=images['gt'];target=torch.from_numpy(gt.transpose(2,0,1).copy())[None]
                for method,image in images.items():
                    if method=='gt':continue
                    mse=float(np.mean((gt-image)**2))
                    tensor=torch.from_numpy(image.transpose(2,0,1).copy())[None]
                    with torch.inference_mode():perceptual=float(lpips(tensor,target));lpips.reset()
                    scores[method]=dict(psnr=float(-10*np.log10(max(mse,1e-12))),
                        ssim=float(structural_similarity(gt,image,channel_axis=2,data_range=1)),lpips=perceptual)
            item=dict(id=key,image=case['image'],kind=case['kind'],zoom=case['zoom'],common_comparison=key in common_ids,width=size[0],height=size[1],urls=urls,scores=scores)
            records.append(item);inspection['cases'][dataset].append(item)
        means={method:{metric:float(np.mean([r['scores'][method][metric] for r in records if r['scores'] and r['common_comparison']]))
                       for metric in ('psnr','ssim','lpips')} for method in methods}
        extended={method:{metric:float(np.mean([r['scores'][method][metric] for r in records if r['scores']]))
                          for metric in ('psnr','ssim','lpips')} for method in methods if method!='gnt'}
        report['datasets'][dataset]=dict(cases=records,mean_quality=means,mean_quality_all_2dgs_views=extended,gnt=runtime,gnt_motion=motion_runtime,two_dgs=gs_runtime,two_dgs_default=default_runtime)
        rows=''.join(f'<tr><td>{html.escape(LABELS[m])}</td><td>{s["psnr"]:.2f}</td><td>{s["ssim"]:.4f}</td><td>{s["lpips"]:.4f}</td></tr>' for m,s in means.items())
        default_note=f' Default-weight check: {default_runtime["training_seconds"]/60:.1f} min, {default_runtime["peak_cuda_mb"]:.0f} MiB, distortion weight 0; median CUDA render {np.median([c["seconds"] for c in default_runtime["cases"]])*1000:.1f} ms.' if default_runtime else ''
        tables.append(f'<h2>{dataset.title()}</h2><p>2DGS initial training: {gs_runtime["training_seconds"]/60:.1f} min; peak allocated {gs_runtime["peak_cuda_mb"]:.0f} MiB; distortion weight {gs_runtime["lambda_dist"]:g}; median CUDA render {np.median([c["seconds"] for c in gs_runtime["cases"]])*1000:.1f} ms.{default_note} GNT full frame: {np.mean([c["seconds"] for c in runtime["cases"]]):.1f} s; peak allocated {runtime["peak_cuda_mb"]:.0f} MiB. Scores below use only views shared by both trials. Extra views in the selector evaluate 2DGS separately; GNT is unavailable for those views.</p><table><tr><th>Method</th><th>PSNR ↑</th><th>SSIM ↑</th><th>LPIPS ↓</th></tr>{rows}</table>')
        extended_rows=''.join(f'<tr><td>{html.escape(LABELS[m])}</td><td>{s["psnr"]:.2f}</td><td>{s["ssim"]:.4f}</td><td>{s["lpips"]:.4f}</td></tr>' for m,s in extended.items())
        scored=[r for r in records if r['scores']]
        tables.append(f'<h3>Wider 2DGS check</h3><p>{len(scored)} camera/zoom comparisons from {len(set(r["image"] for r in scored))} distinct held-out cameras. Zooms are correlated views of the same photograph. GNT was not evaluated across this wider set.</p><table><tr><th>Method</th><th>PSNR ↑</th><th>SSIM ↑</th><th>LPIPS ↓</th></tr>{extended_rows}</table>')
        print(dataset,json.dumps(means),flush=True)
    (out/'results.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    (assets/'results.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    page='''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>GNT and 2DGS trials</title>
<style>body{background:#15191e;color:#eee;font:16px system-ui;margin:24px}a{color:#9ed6ff}p{max-width:1100px;line-height:1.6}select,button,input{margin:8px;padding:6px}canvas{display:block;width:100%;max-width:960px;touch-action:none;cursor:grab}table{border-collapse:collapse}td,th{padding:8px;border:1px solid #777}</style>
<h1>GNT and 2DGS: actual research implementations</h1><p><strong>Visual review decision:</strong> 2DGS is rejected for all four datasets because its trial renders are unusable for this project. Its results remain here as an experiment archive. GNT improves scene indoors; outdoors and static objects are approximately on par with the preferred restoration, with possible outdoor detail loss. Retain the preferred restoration for those cases.</p><p><a href="index.html">Main demo</a> · <a href="sharpness.html">Earlier experiments</a> · <a href="assets/lightweight/results.json">Full results and settings</a></p>
<p>Choose any two methods. Both share the same crop and camera. The original render is the default on the right. Zoom and drag to inspect fine details; select the held-out photograph to see the remaining gap. Motion entries are three coarse positions at half resolution, without a reference photograph.</p>
<p>Previously preferred restoration means prior on original for box, boot and milk; surface/photo plus learned prior for scene.</p>
<label>Dataset <select id="dataset"></select></label><label>View <select id="view"></select></label><br>
<label>Left <select id="left"></select></label><label>Right <select id="right"></select></label><br>
<label>Wipe <input id="wipe" type="range" min="0" max="100" value="50"></label><button id="toggle">Toggle A/B</button>
<label>Crop zoom <input id="zoom" type="range" min="1" max="4" step=".25" value="1"></label><button id="reset">Reset crop</button>
<p id="status" role="status"></p><canvas id="comparison" width="960" height="540" aria-label="Matched image comparison; drag to pan"></canvas>
<p>GNT: official pretrained generalization model, no fine-tuning. 2DGS: official optimizer and rasterizer, 10,000-step geometry-regularized pilot. These are limited trials, not claims of full paper reproduction or convergence. RGB targets are held out, but camera/seed estimation predates the split. Times include shared GPU contention.</p>
'''+''.join(tables)+'<script id="comparison-data" type="application/json">'+json.dumps(inspection).replace('</','<\\/')+'</script><script src="lightweight-inspector.js" defer></script></html>'
    (ROOT/'web_demo/lightweight.html').write_text(page,encoding='utf-8')


if __name__=='__main__':main()
