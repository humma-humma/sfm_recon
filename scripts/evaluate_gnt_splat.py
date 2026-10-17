"""Matched native gsplat renders: dense GNT turn and held-out photographs."""
import argparse
import hashlib
import json
from pathlib import Path
from time import perf_counter
import subprocess
import cv2
import numpy as np
import torch
from gsplat.rendering import rasterization
from skimage.metrics import structural_similarity
from sfm_reconstruction.stage3_splat_temporal_diagnostics import _read_colmap
from gnt_traversal import traversal_cases,dense_route_cases
from prepare_gnt_splat import normalized_view

ROOT=Path(__file__).resolve().parents[1]
COLMAP=ROOT/'outputs/stage3_gaussian_colmap_stride16_half_stair_rebalanced'


def main():
    parser=argparse.ArgumentParser()
    for key in ('root','dense','original'):parser.add_argument('--'+key,type=Path,required=True)
    parser.add_argument('--label',choices=['baseline','mixed_5000','mixed_10000','photos_10000','teacher_10000','geometry_12000','pruned_12000'],required=True)
    parser.add_argument('--full-route',action='store_true');args=parser.parse_args()
    torch.set_num_threads(4)
    base=args.root/'gnt_splat_scene';folder=args.root/'assets/gnt_splat_scene'/args.label
    folder.mkdir(parents=True,exist_ok=True);result_path=folder/'results.json'
    if result_path.exists():
        cached=json.loads(result_path.read_text())
        if cached['status']=='complete' and (not args.full_route or cached.get('full_route')):
            print('Already evaluated: '+args.label,flush=True);return
    if args.label=='baseline':
        choice=json.loads((ROOT/'configs/splatting/selected_models.json').read_text())['datasets']['scene']
        checkpoint=ROOT/choice['presentation_checkpoint']
    else:
        arm,step=args.label.split('_');saved_step=5000 if step=='5000' else int(step)-1
        checkpoint=base/'models'/arm/'splatfacto/run_10000/nerfstudio_models'/f'step-{saved_step:09}.ckpt'
    digest=hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    saved=torch.load(checkpoint,map_location='cpu',weights_only=False);training_step=saved['step']
    state=saved['pipeline'];del saved
    values={key:state['_model.gauss_params.'+key].detach().cuda() for key in ('means','scales','quats','opacities','features_dc','features_rest')}
    del state
    assert all(torch.isfinite(value).all() for value in values.values())
    colors=torch.cat([values['features_dc'][:,None],values['features_rest']],dim=1)
    scales=values['scales'].exp();opacities=values['opacities'].sigmoid().flatten()
    degree=int(np.sqrt(colors.shape[1]))-1
    normalization=json.loads((checkpoint.parent.parent/'dataparser_transforms.json').read_text())
    transform=np.array(normalization['transform']);scale=normalization['scale']
    k,poses,_=_read_colmap(COLMAP/'sparse/0')
    K=torch.tensor([[[k[0],0,k[2]],[0,k[1],k[3]],[0,0,1]]],dtype=torch.float32,device='cuda')
    route=traversal_cases(poses,k)
    dense=json.loads((args.dense/'progress.json').read_text());assert dense['status']=='complete'
    rows=[];timings=[]
    @torch.inference_mode()
    def render(case):
        view=torch.tensor(normalized_view(case['r'],case['t'],transform,scale)[None],dtype=torch.float32,device='cuda')
        torch.cuda.synchronize();tick=perf_counter()
        rgb,_,_=rasterization(means=values['means'],quats=values['quats'],scales=scales,opacities=opacities,
            colors=colors,viewmats=view,Ks=K,width=960,height=540,sh_degree=degree,packed=False,
            rasterize_mode='classic',backgrounds=torch.zeros((1,3),device='cuda'))
        torch.cuda.synchronize();seconds=perf_counter()-tick
        return np.uint8(np.clip(rgb[0].cpu().numpy(),0,1)*255+.5),seconds
    def measure(image,reference):
        mse=np.mean((image.astype(float)-reference.astype(float))**2)
        return dict(psnr=float(-10*np.log10(max(mse/255**2,1e-16))),
            ssim=float(structural_similarity(image,reference,channel_axis=2,data_range=255)))
    for i,case in enumerate(dense['cases']):
        image,seconds=render(case);timings.append(seconds)
        reference=cv2.cvtColor(cv2.imread(str(args.dense/f'frame_{i:04}.png')),cv2.COLOR_BGR2RGB)
        file=folder/f'frame_{i:04}.png'
        assert cv2.imwrite(str(file),cv2.cvtColor(image,cv2.COLOR_RGB2BGR))
        rows.append(dict(index=i,seconds=seconds,**measure(image,reference)))
    heldout=[]
    for i,(name,r,t) in enumerate(poses):
        if i%8:continue
        image,seconds=render(dict(r=r,t=t))
        reference=cv2.cvtColor(cv2.imread(str(COLMAP/'images'/name)),cv2.COLOR_BGR2RGB)
        file=folder/f'photo_{i:04}.png';assert cv2.imwrite(str(file),cv2.cvtColor(image,cv2.COLOR_RGB2BGR))
        heldout.append(dict(index=i,image=name,seconds=seconds,**measure(image,reference)))
    report=dict(status='complete',label=args.label,training_step=training_step,checkpoint=str(checkpoint),checkpoint_sha256=digest,
        gaussians=len(values['means']),dense_gnt_comparison=rows,heldout_photos=heldout,
        means=dict(gnt_psnr=float(np.mean([r['psnr'] for r in rows])),gnt_ssim=float(np.mean([r['ssim'] for r in rows])),
            photo_psnr=float(np.mean([r['psnr'] for r in heldout])),photo_ssim=float(np.mean([r['ssim'] for r in heldout])),
            render_seconds=float(np.median(timings[1:]))),
        note='Direct native gsplat SH3/classic rendering, black background. Dense GNT comparison measures teacher agreement, not ground-truth reconstruction. Interval-8 photo metrics are separate. No camera optimization or video interpolation.')
    if args.full_route:
        cases=dense_route_cases(route,start_seconds=0,duration=146.5,fps=30,hold_end=True)
        # Save direct RGB frames compactly for host encoding, avoiding thousands of PNG files.
        raw=folder/'full_route.rgb'
        with raw.open('wb') as output:
            for i,case in enumerate(cases):
                image,_=render(case);output.write(image.tobytes())
                if i%300==0:print(json.dumps(dict(label=args.label,full_route_frame=i,total=len(cases))),flush=True)
        report['full_route']=dict(raw=raw.name,frames=len(cases),fps=30,duration_seconds=146.5,
            last_pose_hold=True,bytes=raw.stat().st_size)
    assert hashlib.sha256(checkpoint.read_bytes()).hexdigest()==digest
    result_path.write_text(json.dumps(report,indent=2));print(json.dumps(dict(label=args.label,**report['means'])),flush=True)


if __name__=='__main__':main()
