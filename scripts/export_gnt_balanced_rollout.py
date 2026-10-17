"""Evaluate matched splat variants and export full independent 30fps routes."""
import argparse
import json
from pathlib import Path
from time import perf_counter
import cv2
import numpy as np
import torch
from gsplat.rendering import rasterization
from nerfstudio.utils.eval_utils import eval_setup
from nerfstudio.models.splatfacto import get_viewmat
from diagnose_gnt_transfer import digest,measure,quantize
from prepare_gnt_splat import normalized_view
from run_gnt_trial import ROOT,DATA
from sfm_reconstruction.stage3_splat_temporal_diagnostics import _read_colmap
from gnt_traversal import dense_route_cases,traversal_cases
from render_gnt_rollout_teachers import save


def setup(root,name):
    if name=='scene':
        base=root/'gnt_dense_supervision'
        dense=json.loads((root/'assets/dense_turn/progress.json').read_text())
        original=json.loads((root/'gnt_splat_scene/data_manifest.json').read_text())
        sparse=[r for r in original['rows'] if r['kind']=='gnt']
        plan=json.loads((base/'plan.json').read_text())
        dense_targets=[root/'assets/dense_turn'/r['file'] for r in dense['frames']]
        sparse_targets=[root/'gnt_splat_scene/data/teacher/images'/r['name'] for r in sparse]
        k,poses,_=_read_colmap(ROOT/DATA[name]/'sparse/0')
        route=traversal_cases(poses,k);duration=146.5
    else:
        base=root/'balanced_rollout'/name;plan=json.loads((base/'plan.json').read_text())
        dense=json.loads((root/'assets/balanced_rollout'/name/'teachers/progress.json').read_text())
        sparse=plan['sparse'];route=sparse;duration=32.
        dense_targets=[root/'assets/balanced_rollout'/name/'teachers'/r['file'] for r in dense['frames']]
        sparse_report=json.loads((root/'assets/videos'/name/'progress.json').read_text())
        sparse_targets=[root/'assets/videos'/name/r['file'] for r in sparse_report['frames']]
        k,poses,_=_read_colmap(ROOT/DATA[name]/'sparse/0')
    checkpoints={arm:base/f'models/{arm}/splatfacto/continued/nerfstudio_models/step-000011999.ckpt' for arm in ('control','dense','balanced90')}
    return plan,dense['cases'],dense_targets,sparse,sparse_targets,route,duration,k,poses,checkpoints


def make_renderer(checkpoint,k,audit_cases):
    _,pipeline,loaded,_=eval_setup(checkpoint.parent.parent/'config.yml',test_mode='inference')
    assert loaded==checkpoint;model=pipeline.model;model.eval()
    normalization=json.loads((checkpoint.parent.parent/'dataparser_transforms.json').read_text())
    transform=np.array(normalization['transform']);scale=normalization['scale']
    K=torch.tensor([[[k[0],0,k[2]],[0,k[1],k[3]],[0,0,1]]],device='cuda',dtype=torch.float32)
    # Check every training camera against the independently exported world pose.
    names=[Path(p).name for p in pipeline.datamanager.train_dataset.image_filenames]
    expected={f'gnt_{i:04}.png':c for i,c in enumerate(audit_cases['sparse'])}
    expected.update({f'dense_{i:04}.png':audit_cases['dense'][i] for i in audit_cases['added']})
    cameras=pipeline.datamanager.train_dataset.cameras
    for i,name in enumerate(names):
        c=expected[name];actual=get_viewmat(cameras[i:i+1].to('cuda').camera_to_worlds)[0].cpu().numpy()
        np.testing.assert_allclose(actual,normalized_view(c['r'],c['t'],transform,scale),atol=5e-5,rtol=1e-5)
        np.testing.assert_allclose(cameras[i:i+1].get_intrinsics_matrices().numpy(),K.cpu().numpy(),atol=1e-4)
    colors=torch.cat([model.features_dc[:,None],model.features_rest],dim=1)
    @torch.inference_mode()
    def render(case):
        view=torch.tensor(normalized_view(case['r'],case['t'],transform,scale)[None],device='cuda',dtype=torch.float32)
        torch.cuda.synchronize();tick=perf_counter()
        rgb,alpha,_=rasterization(means=model.means,quats=model.quats,scales=model.scales.exp(),
            opacities=model.opacities.sigmoid().flatten(),colors=colors,viewmats=view,Ks=K,
            width=960,height=540,packed=False,near_plane=.01,far_plane=1e10,sh_degree=3,
            sparse_grad=False,absgrad=model.strategy.absgrad,rasterize_mode='classic')
        image=quantize((rgb[0]+(1-alpha[0])*model._get_background_color()).cpu().numpy())
        return image,perf_counter()-tick
    return render,pipeline


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--dataset',choices=['scene','box','boot','milk'],required=True)
    args=parser.parse_args();torch.set_num_threads(4)
    plan,dense,targets,sparse,sparse_targets,route,duration,k,poses,checkpoints=setup(args.root,args.dataset)
    folder=args.root/'assets/balanced_rollout'/args.dataset;folder.mkdir(parents=True,exist_ok=True)
    # Teacher target bytes are checked before any new comparisons are accepted.
    for path in targets+sparse_targets:assert cv2.imread(str(path)).shape==(540,960,3)
    audits=dict(sparse=sparse,dense=dense,added=plan['added'])
    for arm,checkpoint in checkpoints.items():
        output=folder/f'{arm}_evaluation.json'
        if output.exists():
            assert json.loads(output.read_text())['checkpoint_sha256']==digest(checkpoint);continue
        render,pipeline=make_renderer(checkpoint,k,audits);rows=[]
        cases=[('dense',i,c,t) for i,(c,t) in enumerate(zip(dense,targets))]
        cases += [('original',i,c,t) for i,(c,t) in enumerate(zip(sparse,sparse_targets))]
        cases += [('photo',i,dict(r=r,t=t),ROOT/DATA[args.dataset]/'images'/name) for i,(name,r,t) in enumerate(poses) if i%8==0]
        for group,i,case,target in cases:
            image,seconds=render(case);reference=cv2.cvtColor(cv2.imread(str(target)),cv2.COLOR_BGR2RGB)
            file=f'{arm}_{group}_{i:04}.png';assert cv2.imwrite(str(folder/file),cv2.cvtColor(image,cv2.COLOR_RGB2BGR))
            rows.append(dict(group=group,index=i,file=file,seconds=seconds,target_sha256=digest(target),**measure(image,reference)))
            if len(rows)%50==0:print(json.dumps(dict(dataset=args.dataset,phase='evaluation',arm=arm,completed=len(rows),total=len(cases))),flush=True)
        groups={'withheld':[r for r in rows if r['group']=='dense' and r['index'] in plan['withheld']],
                'added':[r for r in rows if r['group']=='dense' and r['index'] in plan['added']],
                'original':[r for r in rows if r['group']=='original'],'photos':[r for r in rows if r['group']=='photo']}
        summaries={g:dict(count=len(items),**{m:float(np.mean([r[m] for r in items])) for m in ('psnr','ssim')}) for g,items in groups.items()}
        save(output,dict(status='complete',checkpoint_sha256=digest(checkpoint),rows=rows,summaries=summaries))
        del render,pipeline;torch.cuda.empty_cache()
    checkpoint=checkpoints['balanced90'];signature=digest(checkpoint)
    video_path=folder/'video.json'
    if video_path.exists() and json.loads(video_path.read_text()).get('status')=='encoded':return
    raw=folder/'full_route.rgb';progress_path=folder/'raw_progress.json'
    progress=json.loads(progress_path.read_text()) if progress_path.exists() else dict(completed=0,checkpoint_sha256=signature)
    assert progress['checkpoint_sha256']==signature
    cases=dense_route_cases(route,start_seconds=0,duration=duration,fps=30,hold_end=True)
    render,pipeline=make_renderer(checkpoint,k,audits)
    frame_bytes=960*540*3;completed=progress['completed']
    assert completed==0 or (raw.exists() and raw.stat().st_size>=completed*frame_bytes)
    with raw.open('r+b' if raw.exists() else 'w+b') as stream:
        stream.truncate(completed*frame_bytes);stream.seek(completed*frame_bytes)
        for i in range(completed,len(cases)):
            image,_=render(cases[i]);stream.write(image.tobytes())
            if (i+1)%120==0 or i+1==len(cases):
                stream.flush();progress['completed']=i+1;save(progress_path,progress)
                print(json.dumps(dict(dataset=args.dataset,phase='full_video',completed=i+1,total=len(cases))),flush=True)
    assert digest(checkpoint)==signature and raw.stat().st_size==len(cases)*frame_bytes
    save(video_path,dict(status='rendered',raw=raw.name,frames=len(cases),fps=30,width=960,height=540,
                        duration_seconds=duration,checkpoint_sha256=signature,last_pose_hold=True,
                        note='Native Gaussian rendering at 30fps; no motion-compensated interpolation. Supervision is sparse plus one dense section, not full-route dense GNT.'))


if __name__=='__main__':main()
