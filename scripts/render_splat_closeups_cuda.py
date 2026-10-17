"""Native gsplat comparison for the exact offscreen diagnostic camera matrix.

Run in the existing Nerfstudio Docker image. This bypasses dataset loading and
does not train or edit checkpoints. It is a rasterizer reference, not a complete
Nerfstudio pipeline render. Full SH, SH0 and antialiased filtering are isolated.
"""
from pathlib import Path
from time import perf_counter
import json
import numpy as np
import torch
from PIL import Image

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'outputs/splat_closeups'

def main():
    from gsplat.rendering import rasterization
    import importlib.metadata
    if not torch.cuda.is_available():raise RuntimeError('CUDA unavailable')
    selected=json.loads((ROOT/'configs/splatting/selected_models.json').read_text())['datasets']
    manifest=json.loads((ROOT/'web_demo/assets/manifest.json').read_text())
    device='cuda';width,height=960,540
    report=dict(status='running',gsplat=importlib.metadata.version('gsplat'),torch=torch.__version__,gpu=torch.cuda.get_device_name(),method='Native gsplat rasterization, not complete Nerfstudio pipeline. Same camera poses, square pixels, background and near plane as local diagnostic. No novel-view ground truth.',datasets={})
    for name,choice in selected.items():
        cases=json.loads((OUT/name/'camera_cases.json').read_text())
        state=torch.load(ROOT/choice['presentation_checkpoint'],map_location='cpu',weights_only=False)['pipeline']
        values={key:state['_model.gauss_params.'+key].detach().to(device) for key in ['means','quats','scales','opacities','features_dc','features_rest']}
        if len(values['means'])!=manifest[name]['count']:raise ValueError('Exported Gaussian count mismatch')
        colors=torch.cat([values['features_dc'][:,None,:],values['features_rest']],dim=1)
        scales=values['scales'].exp();opacity=values['opacities'].sigmoid().flatten();background=torch.tensor([[.035,.045,.035]],device=device)
        dataset=dict(checkpoint=choice['presentation_checkpoint'],renders=[]);report['datasets'][name]=dataset
        for case in cases:
            eye=np.array(case['position']);forward=np.array(case['forward']);up=np.array(case['up']);right=np.cross(forward,up);right/=np.linalg.norm(right);up=np.cross(right,forward)
            # OpenCV view convention: +x right, +y down, +z forward.
            rotation=np.stack([right,-up,forward]);view=np.eye(4);view[:3,:3]=rotation;view[:3,3]=-rotation@eye
            focal=height/(2*np.tan(np.radians(case['fov']/2)));k=np.array([[focal,0,width/2],[0,focal,height/2],[0,0,1]])
            view_tensor=torch.tensor(view[None],device=device,dtype=torch.float32);k_tensor=torch.tensor(k[None],device=device,dtype=torch.float32)
            for label,degree,mode in [('native_sh0',0,'classic'),('native_sh3',int(np.sqrt(colors.shape[1]))-1,'classic'),('native_sh3_antialiased',int(np.sqrt(colors.shape[1]))-1,'antialiased')]:
                samples=[]
                with torch.no_grad():
                    for repeat in range(3):
                        torch.cuda.synchronize();tick=perf_counter()
                        rgb,alpha,_=rasterization(means=values['means'],quats=values['quats'],scales=scales,opacities=opacity,colors=colors,
                            viewmats=view_tensor,Ks=k_tensor,width=width,height=height,sh_degree=degree,packed=False,
                            rasterize_mode=mode,near_plane=.005,far_plane=1e10,backgrounds=background)
                        torch.cuda.synchronize();samples.append(1000*(perf_counter()-tick))
                file=f'{name}/{case["id"]}_{label}.png'
                Image.fromarray((rgb[0].clamp(0,1).cpu().numpy()*255).round().astype('uint8')).save(OUT/file)
                dataset['renders'].append(dict(case_id=case['id'],variant=label,file=file,gpu_render_ms=float(np.median(samples[1:])),gpu_samples_ms=samples))
            print(name,case['id'],flush=True)
            (OUT/'native_report.json').write_text(json.dumps(report,indent=2))
        del values,colors,scales,opacity,state;torch.cuda.empty_cache()
    report.update(status='complete',renders=sum(len(d['renders']) for d in report['datasets'].values()))
    (OUT/'native_report.json').write_text(json.dumps(report,indent=2))
    html=['<!doctype html><meta charset="utf-8"><title>Native GPU close-up comparison</title><style>body{background:#101510;color:#eee;font:14px Segoe UI;margin:24px}.grid{display:grid;grid-template-columns:repeat(4,1fr);gap:8px}img{width:100%}figure{margin:0}a{color:#cde6aa}</style><h1>Native gsplat vs demo</h1><p>Native CUDA rasterizer reference; not the complete Nerfstudio pipeline. Same checkpoints and cameras. Antialiased rendering is an inference-only diagnostic on classic-trained models, not a Mip-Splatting retrain. No models were modified.</p><p><a href="index.html">Local shader tests</a></p>']
    for name,dataset in report['datasets'].items():
        html.append(f'<h2>{name.title()}</h2>')
        groups={}
        for row in dataset['renders']:groups.setdefault(row['case_id'],[]).append(row)
        for case_id,rows in groups.items():
            html.append(f'<h3>{case_id}</h3><div class="grid">')
            images=[dict(file=f'{name}/{case_id}_full_sh_guard.png',variant='Guarded local shader / full SH'),*rows]
            for row in images:html.append(f'<figure><a href="{row["file"]}"><img loading="lazy" src="{row["file"]}"></a><figcaption>{row["variant"]}</figcaption></figure>')
            html.append('</div>')
    (OUT/'native.html').write_text('\n'.join(html),encoding='utf-8')
    print('Complete: outputs/splat_closeups/native.html',flush=True)

if __name__=='__main__':main()
