"""Audit native rendering and compare fitted versus unseen GNT target poses."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
from time import perf_counter

import cv2
import numpy as np
import torch
from gsplat.rendering import rasterization
from nerfstudio.cameras.cameras import Cameras,CameraType
from nerfstudio.models.splatfacto import get_viewmat
from nerfstudio.utils.eval_utils import eval_setup
from skimage.metrics import structural_similarity
from prepare_gnt_splat import normalized_view
from gnt_transfer_metrics import bracket_training,local_gap


def digest(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def measure(image,reference):
    mse=float(np.mean(((image.astype(np.float32)-reference)/255)**2))
    return dict(mse=mse,psnr=float(-10*np.log10(max(mse,1e-16))),
                ssim=float(structural_similarity(image,reference,channel_axis=2,data_range=255)))


def quantize(rgb):return np.uint8(np.clip(rgb,0,1)*255+.5)


def summarize(rows,mode):
    values=[r[mode] for r in rows]
    return dict(count=len(rows),mean_psnr=float(np.mean([r['psnr'] for r in values])),
                median_psnr=float(np.median([r['psnr'] for r in values])),
                p10_psnr=float(np.quantile([r['psnr'] for r in values],.1)),
                mean_ssim=float(np.mean([r['ssim'] for r in values])))


@torch.inference_mode()
def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--root',type=Path,required=True)
    args=parser.parse_args();torch.set_num_threads(4)
    root=args.root;folder=root/'assets/gnt_transfer_diagnosis';folder.mkdir(parents=True,exist_ok=True)
    checkpoint=root/'gnt_splat_scene/models/teacher/splatfacto/run_10000/nerfstudio_models/step-000009999.ckpt'
    checkpoint_hash=digest(checkpoint)
    manifest=json.loads((root/'gnt_splat_scene/data_manifest.json').read_text())
    teacher=[r for r in manifest['rows'] if r['kind']=='gnt']
    assert len(teacher)==275
    teacher.sort(key=lambda r:r['source_time'])
    train_times=np.array([r['source_time'] for r in teacher])
    dense=json.loads((root/'assets/dense_turn/progress.json').read_text())
    adaptive=json.loads((root/'assets/adaptive_scene/progress.json').read_text())
    assert dense['status']=='complete' and len(adaptive['native'])==48
    assert dense['checkpoint_sha256']==adaptive['checkpoint_sha256']==manifest['teacher_checkpoint_sha256']
    config,pipeline,loaded,step=eval_setup(checkpoint.parent.parent/'config.yml',test_mode='inference')
    assert loaded==checkpoint and step==9999
    model=pipeline.model;model.eval()
    assert not model.config.use_bilateral_grid and model.config.camera_optimizer.mode=='off'
    assert model.config.rasterize_mode=='classic' and model.step>=3000
    normalization=json.loads((checkpoint.parent.parent/'dataparser_transforms.json').read_text())
    transform=np.array(normalization['transform']);scale=normalization['scale']
    k=next(iter(dense['cases']))['k']
    K=torch.tensor([[[k[0],0,k[2]],[0,k[1],k[3]],[0,0,1]]],device='cuda',dtype=torch.float32)
    colors=torch.cat([model.features_dc[:,None],model.features_rest],dim=1)
    background=model._get_background_color().detach();assert background.shape==(3,)
    names=[Path(p).name for p in pipeline.datamanager.train_dataset.image_filenames]
    assert set(names)=={r['name'] for r in teacher}
    cameras=pipeline.datamanager.train_dataset.cameras
    camera_audit=[]
    for row in teacher:
        camera=cameras[names.index(row['name']):names.index(row['name'])+1].to('cuda')
        expected=normalized_view(row['r'],row['t'],transform,scale)
        actual=get_viewmat(camera.camera_to_worlds)[0].cpu().numpy()
        error=float(np.max(abs(expected-actual)))
        assert error<5e-5,(row['name'],error)
        np.testing.assert_allclose(camera.get_intrinsics_matrices().cpu().numpy(),K.cpu().numpy(),atol=1e-4)
        assert int(camera.width.item())==960 and int(camera.height.item())==540
        camera_audit.append(error)

    def camera_for(case):
        cv_c2w=np.linalg.inv(normalized_view(case['r'],case['t'],transform,scale))[:3]
        cv_c2w[:,1:3]*=-1
        return Cameras(camera_to_worlds=torch.tensor(cv_c2w[None],dtype=torch.float32),
                       fx=k[0],fy=k[1],cx=k[2],cy=k[3],width=960,height=540,
                       camera_type=CameraType.PERSPECTIVE).to('cuda')

    def render(case,view=None):
        if view is None:view=torch.tensor(normalized_view(case['r'],case['t'],transform,scale)[None],device='cuda',dtype=torch.float32)
        tick=perf_counter()
        rgb,alpha,_=rasterization(means=model.means,quats=model.quats,scales=model.scales.exp(),
            opacities=model.opacities.sigmoid().flatten(),colors=colors,viewmats=view,Ks=K,
            width=960,height=540,packed=False,near_plane=.01,far_plane=1e10,
            render_mode='RGB+ED',sh_degree=3,sparse_grad=False,
            absgrad=model.strategy.absgrad,rasterize_mode='classic')
        native=rgb[0,:,:,:3]+(1-alpha[0])*background
        black=rgb[0,:,:,:3]
        torch.cuda.synchronize()
        return black.cpu().numpy(),native.cpu().numpy(),alpha[0].cpu().numpy(),perf_counter()-tick

    native_audit=[]
    for i in (0,137,274):
        case=teacher[i];index=names.index(case['name']);camera=cameras[index:index+1].to('cuda')
        official=model.get_outputs_for_camera(camera)['rgb'].cpu().numpy()
        _,custom,_,_=render(case,get_viewmat(camera.camera_to_worlds))
        error=float(np.max(abs(official-custom.clip(0,1))))
        assert error<1e-5,error
        native_audit.append(dict(training_index=i,max_rgb_error=error))

    rows=[];cached_path=folder/'progress.json'
    cached=json.loads(cached_path.read_text()) if cached_path.exists() else {}
    if cached:assert cached['checkpoint_sha256']==checkpoint_hash
    previous={r['key']:r for r in cached.get('rows',[])}
    cases=[('training',i,r,root/'gnt_splat_scene/data/teacher/images'/r['name'],r['sha256']) for i,r in enumerate(teacher)]
    cases += [('dense',i,r,root/'assets/dense_turn'/f'frame_{i:04}.png',dense['frames'][i]['sha256']) for i,r in enumerate(dense['cases'])]
    cases += [('adaptive',i,r['case'],root/'assets/adaptive_scene'/r['file'],r['sha256']) for i,r in enumerate(sorted(adaptive['native'].values(),key=lambda r:r['case']['source_time']))]
    for group,i,case,target,expected_hash in cases:
        key=f'{group}_{i:04}'
        if key in previous:
            row=previous[key]
            assert all(digest(folder/row[name])==row[name+'_sha256'] for name in ('target','black_file','native_file'))
        else:
            assert digest(target)==expected_hash,str(target)
            archived=folder/f'{key}_target.png'
            if not archived.exists():
                try:os.link(target,archived)
                except OSError:shutil.copy2(target,archived)
            assert digest(archived)==expected_hash
            reference=cv2.cvtColor(cv2.imread(str(target)),cv2.COLOR_BGR2RGB)
            black,native,alpha,seconds=render(case)
            black=quantize(black);native=quantize(native)
            row=dict(key=key,group=group,index=i,source_time=case['source_time'],
                     target=archived.name,target_sha256=expected_hash,
                     black=measure(black,reference),native=measure(native,reference),
                     low_alpha_fraction=float(np.mean(alpha<.95)),render_seconds=seconds)
            for mode,image in [('black',black),('native',native)]:
                file=folder/f'{key}_{mode}.png'
                assert cv2.imwrite(str(file),cv2.cvtColor(image,cv2.COLOR_RGB2BGR))
                row[mode+'_file']=file.name;row[mode+'_file_sha256']=digest(file)
        rows.append(row)
        if len(rows)%20==0:
            cached_path.write_text(json.dumps(dict(checkpoint_sha256=checkpoint_hash,rows=rows),indent=2))
            print(json.dumps(dict(completed=len(rows),total=len(cases),last=key)),flush=True)
    training=rows[:275]
    for row in rows[275:]:
        left,right,fraction=bracket_training(train_times,row['source_time'])
        row['bracket']=[left,right];row['fraction']=fraction;row['exact_training_time']=left==right
        for mode in ('black','native'):
            row[mode].update(local_gap([r[mode] for r in training],left,right,fraction,row[mode]))
    summaries={}
    for group in ('training','dense','adaptive'):
        subset=[r for r in rows if r['group']==group and not r.get('exact_training_time',False)]
        summaries[group]={mode:summarize(subset,mode) for mode in ('black','native')}
        if group!='training':
            for mode in ('black','native'):
                gaps=np.array([r[mode]['psnr_gap'] for r in subset])
                summaries[group][mode].update(mean_local_psnr_gap=float(gaps.mean()),
                    median_local_psnr_gap=float(np.median(gaps)),fraction_worse_than_endpoints=float(np.mean(gaps<0)),
                    fraction_more_than_3db_worse=float(np.mean(gaps < -3)))
    # Reproduce old black-background metrics as an independent provenance check.
    old=json.loads((root/'assets/gnt_splat_scene/teacher_10000/results.json').read_text())
    assert old['checkpoint_sha256']==checkpoint_hash
    reproduction=max(abs(row['black']['psnr']-old['dense_gnt_comparison'][row['index']]['psnr']) for row in rows if row['group']=='dense')
    assert reproduction<.02,reproduction
    assert digest(checkpoint)==checkpoint_hash
    report=dict(status='complete',checkpoint_sha256=checkpoint_hash,teacher_checkpoint_sha256=manifest['teacher_checkpoint_sha256'],
                summaries=summaries,rows=rows,native_background=background.cpu().tolist(),
                renderer_audit=dict(camera_count=len(camera_audit),max_view_matrix_error=max(camera_audit),
                    official_rgb_checks=native_audit,max_previous_psnr_difference=reproduction),
                note='Training metrics measure fit to teacher RGB, not photographic truth. Views within 0.1 ms of a training timestamp are excluded from unseen summaries. Dense frames are correlated and cover one outdoor turn; adaptive views were selected for difficult gaps. Local endpoint comparisons reduce scene-content confounding but do not isolate a causal geometric effect. Both previous black and native inference backgrounds are measured. No retraining or new GNT generation.')
    (folder/'results.json').write_text(json.dumps(report,indent=2))
    print(json.dumps(dict(status='complete',summaries=summaries,renderer_audit=report['renderer_audit'])),flush=True)


if __name__=='__main__':main()
