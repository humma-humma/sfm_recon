"""Refine a GNT-only splat with training-only sensor depth, then prune conflicts.

RGB loss uses GNT images only. Original training photographs supply calibrated
poses and confidence-filtered sensor depth, never RGB loss. This is a local
depth-prior experiment, not a reproduction of RadSplat.
"""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
from time import perf_counter

import numpy as np
import torch
import torch.nn.functional as F
import yaml

from nerfstudio.engine.trainer import Trainer
from nerfstudio.models.splatfacto import SplatfactoModel
from nerfstudio.scripts.train import main as train
from continue_scene_control import bind_loaded_optimizers
from sfm_reconstruction.depth_supervision import relative_depth_loss
from sfm_reconstruction.stage3_gaussian_checkpoint_cleanup import prune_checkpoint
from sfm_reconstruction.stage3_gaussian_depth_cleanup import depth_keep_mask
from sfm_reconstruction.stage3_splat_temporal_diagnostics import _read_colmap

ROOT=Path(__file__).resolve().parents[1]
EXPORT=ROOT/'outputs/stage3_gaussian_colmap_stride16_half_stair_rebalanced'
CACHE=ROOT/'outputs/depth_research/depth_cache'


def audit_cache():
    k,poses,_=_read_colmap(EXPORT/'sparse/0')
    plan=json.loads((CACHE.parent/'plan.json').read_text())
    expected={name for i,(name,_,_) in enumerate(poses) if i%8}
    assert {row['image'] for row in plan['rows']}==expected
    assert hashlib.sha256((EXPORT/'sparse/0/images.txt').read_bytes()).hexdigest()==plan['pose_export_sha256']
    for row in plan['rows']:
        assert set(row['neighbors'])<=expected
    return k,poses


def refine(base,poses):
    source=base/'models/teacher/splatfacto/run_10000/nerfstudio_models/step-000009999.ckpt'
    destination=base/'models/geometry/splatfacto/run_10000'
    checkpoint=destination/'nerfstudio_models/step-000011999.ckpt'
    if checkpoint.exists():
        assert (destination/'fit_results.json').exists()
        return checkpoint
    source_hash=hashlib.sha256(source.read_bytes()).hexdigest()
    state=torch.load(source,map_location='cpu',weights_only=False)
    config=yaml.load((source.parent.parent/'config.yml').read_text(),Loader=yaml.Loader)
    config.data=base/'data/mixed'
    config.pipeline.datamanager.data=config.data
    config.pipeline.datamanager.dataparser.data=config.data
    config.experiment_name='geometry';config.timestamp='run_10000'
    config.load_dir=config.load_config=config.load_step=None;config.load_checkpoint=source
    config.max_num_iterations=2000;config.steps_per_save=1000000
    config.pipeline.model.output_depth_during_training=True
    config.pipeline.model.refine_every=1000000  # Keep the same primitives during this ablation.
    for optimizer in config.optimizers.values():
        if optimizer['scheduler'] is not None:optimizer['scheduler'].max_steps=12000
    original_load=Trainer._load_checkpoint
    original_loss=SplatfactoModel.get_loss_dict
    original_save=Trainer.save_checkpoint
    cached={}
    audit=dict(teacher_batches=0,depth_batches=0,empty_depth_batches=0,
               depth_weight=.05,original_rgb_loss_used=False,
               heldout_depth_used=False,first_depth_loss=None,last_depth_loss=None)

    def load(self):
        self._start_step=10000
        self.pipeline.load_pipeline(state['pipeline'],state['step'])
        bind_loaded_optimizers(self,state)
        names=[Path(path).name for path in self.pipeline.datamanager.train_dataset.image_filenames]
        expected={f'photo_{i:04}.jpg' for i in range(len(poses)) if i%8}
        assert {name for name in names if name.startswith('photo_')}==expected
        assert sum(name.startswith('gnt_') for name in names)==275
        for index,name in enumerate(names):
            if not name.startswith('photo_'):continue
            i=int(Path(name).stem.split('_')[1]);assert i%8
            with np.load(CACHE/f'{Path(poses[i][0]).stem}.npz') as data:
                depth=torch.from_numpy(data['depth'].copy())
                weight=torch.from_numpy(data['weight'].copy())
            assert depth.shape==weight.shape==(540,960)
            cached[index]=(depth,weight)
        output=self.pipeline.datamanager.train_dataparser_outputs
        cached['scale']=float(output.dataparser_scale)
        normalization=json.loads((source.parent.parent/'dataparser_transforms.json').read_text())
        np.testing.assert_allclose(output.dataparser_transform.numpy(),normalization['transform'])
        np.testing.assert_allclose(cached['scale'],normalization['scale'])

    def loss(self,outputs,batch,metrics_dict=None):
        index=int(batch['image_idx'])
        if index not in cached:
            audit['teacher_batches']+=1
            return original_loss(self,outputs,batch,metrics_dict)
        depth,weight=cached[index]
        size=outputs['depth'].shape[:2]
        depth=F.interpolate(depth[None,None].to(self.device),size=size,mode='nearest')[0,0]
        weight=F.interpolate(weight[None,None].to(self.device),size=size,mode='nearest')[0,0]
        predicted=outputs['depth'].squeeze(-1)/cached['scale']
        raw=relative_depth_loss(predicted,depth,weight)
        audit['depth_batches']+=1
        audit['empty_depth_batches']+=int(not bool((weight>0).any()))
        value=float(raw.detach())
        if audit['first_depth_loss'] is None:audit['first_depth_loss']=value
        audit['last_depth_loss']=value
        return {'sensor_depth_loss':audit['depth_weight']*raw}

    def save(self,step):
        if step!=11999:return
        checkpoint.parent.mkdir(parents=True,exist_ok=True)
        torch.save({'step':step,'pipeline':self.pipeline.state_dict()},checkpoint)

    Trainer._load_checkpoint=load;SplatfactoModel.get_loss_dict=loss;Trainer.save_checkpoint=save
    start=perf_counter();torch.cuda.reset_peak_memory_stats()
    try:train(config)
    finally:
        Trainer._load_checkpoint=original_load
        SplatfactoModel.get_loss_dict=original_loss
        Trainer.save_checkpoint=original_save
    result=torch.load(checkpoint,map_location='cpu',weights_only=False)
    params={key:value for key,value in result['pipeline'].items() if key.startswith('_model.gauss_params.')}
    assert result['step']==11999 and all(torch.isfinite(value).all() for value in params.values())
    assert audit['depth_batches']>0 and audit['teacher_batches']>0
    assert not torch.equal(params['_model.gauss_params.means'],state['pipeline']['_model.gauss_params.means'])
    assert hashlib.sha256(source.read_bytes()).hexdigest()==source_hash
    audit.update(status='complete',seconds=perf_counter()-start,source_sha256=source_hash,
                 peak_cuda_allocated_mb=torch.cuda.max_memory_allocated()/2**20,
                 gaussians=len(params['_model.gauss_params.means']),optimizer_bindings_verified=True,
                 note='2k-step continuation changes optimization duration as well as adding depth guidance; not an isolated depth-effect measurement.')
    (destination/'fit_results.json').write_text(json.dumps(audit,indent=2))
    print(json.dumps(audit),flush=True)
    return checkpoint


@torch.inference_mode()
def prune(base,checkpoint,k,poses):
    destination=base/'models/pruned/splatfacto/run_10000'
    output=destination/'nerfstudio_models/step-000011999.ckpt'
    if output.exists():
        assert (destination/'pruning.json').exists()
        return
    start=perf_counter()
    state=torch.load(checkpoint,map_location='cpu',weights_only=False)
    normalization=json.loads((checkpoint.parent.parent/'dataparser_transforms.json').read_text())
    transform=torch.tensor(normalization['transform'],device='cuda',dtype=torch.float32)
    means=state['pipeline']['_model.gauss_params.means'].cuda()
    points=(means/normalization['scale']-transform[:,3])@transform[:,:3]
    surface=torch.zeros(len(points),dtype=torch.int32,device='cuda')
    conflict=torch.zeros_like(surface)
    for i,(name,r,t) in enumerate(poses):
        if not i%8:continue
        with np.load(CACHE/f'{Path(name).stem}.npz') as data:
            depth=torch.from_numpy(data['depth']).cuda()
            weight=torch.from_numpy(data['weight']).cuda()
        rotation=torch.tensor(r,device='cuda',dtype=torch.float32)
        translation=torch.tensor(t,device='cuda',dtype=torch.float32)
        for offset in range(0,len(points),250000):
            camera=points[offset:offset+250000]@rotation.T+translation
            z=camera[:,2]
            xy=camera[:,:2]/z[:,None].clamp_min(1e-6)
            xy=xy*torch.tensor(k[:2],device='cuda')+torch.tensor(k[2:4],device='cuda')
            inside=(z>1e-6)&(xy[:,0]>=0)&(xy[:,0]<960)&(xy[:,1]>=0)&(xy[:,1]<540)
            ids=torch.where(inside)[0];pixel=xy[ids].long()
            measured=depth[pixel[:,1],pixel[:,0]]
            valid=torch.isfinite(measured)&(measured>.1)&(measured<8)&(weight[pixel[:,1],pixel[:,0]]>0)
            ids=ids[valid];measured=measured[valid]
            delta=z[ids]-measured;tolerance=.10+.05*measured
            surface[offset+ids]+=(delta.abs()<=tolerance).int()
            conflict[offset+ids]+=(delta < -tolerance).int()
        if i%40==1:print(json.dumps(dict(pruning_camera=i,total=len(poses))),flush=True)
    # Unknown and occluded Gaussians survive. Only repeated free-space conflicts
    # with no supporting surface measurement are culled.
    supporting=surface.cpu().numpy();conflicting=conflict.cpu().numpy()
    keep=depth_keep_mask(np.ones(len(points),dtype=bool),supporting,conflicting,min_conflicts=3)
    assert keep.any()
    output.parent.mkdir(parents=True,exist_ok=True)
    torch.save(prune_checkpoint(state,keep),output)
    shutil.copy2(checkpoint.parent.parent/'dataparser_transforms.json',destination/'dataparser_transforms.json')
    report=dict(status='complete',source=str(checkpoint),source_gaussians=len(keep),
                kept_gaussians=int(keep.sum()),removed_gaussians=int((~keep).sum()),
                training_depth_cameras=241,heldout_depth_used=False,
                min_conflicts=3,absolute_tolerance_m=.10,relative_tolerance=.05,
                unknown_and_occluded_retained=True,seconds=perf_counter()-start,
                note='Depth-consistency pruning, not RadSplat contribution-based pruning.')
    (destination/'pruning.json').write_text(json.dumps(report,indent=2));print(json.dumps(report),flush=True)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--root',type=Path,required=True)
    args=parser.parse_args();torch.set_num_threads(4)
    k,poses=audit_cache();base=args.root/'gnt_splat_scene'
    checkpoint=refine(base,poses)
    torch.cuda.empty_cache()
    prune(base,checkpoint,k,poses)


if __name__=='__main__':main()
