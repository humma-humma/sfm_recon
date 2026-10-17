"""Continue fresh 5k with confidence-weighted sensor depth; isolated process patch."""
from pathlib import Path
import hashlib
import json
from time import perf_counter
import numpy as np
import torch
import torch.nn.functional as F
import yaml
from nerfstudio.engine.trainer import Trainer
from nerfstudio.models.splatfacto import SplatfactoModel
from nerfstudio.scripts.train import main as train
from sfm_reconstruction.depth_supervision import relative_depth_loss
from continue_scene_control import bind_loaded_optimizers

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'outputs/depth_research'


def main():
    plan=json.loads((OUT/'plan.json').read_text())
    source=ROOT/plan['source_checkpoint'].replace('\\','/')
    assert hashlib.sha256(source.read_bytes()).hexdigest()==plan['source_checkpoint_sha256']
    assert hashlib.sha256((source.parent.parent/'config.yml').read_bytes()).hexdigest()==plan['source_config_sha256']
    export=ROOT/'outputs/stage3_gaussian_colmap_stride16_half_stair_rebalanced/sparse/0/images.txt'
    assert hashlib.sha256(export.read_bytes()).hexdigest()==plan['pose_export_sha256']
    state=torch.load(source,map_location='cpu',weights_only=False)
    config=yaml.load((source.parent.parent/'config.yml').read_text(),Loader=yaml.Loader)
    config.output_dir=OUT/'train';config.experiment_name='scene_depth';config.timestamp='run_10000'
    config.load_dir=config.load_config=config.load_step=None;config.load_checkpoint=source
    config.max_num_iterations=5000;config.steps_per_save=1000;config.steps_per_eval_all_images=1000000
    config.vis='tensorboard';config.pipeline.model.output_depth_during_training=True
    checkpoint=config.get_checkpoint_dir()/'step-000009999.ckpt'
    if checkpoint.exists(): raise RuntimeError('Candidate already exists; refusing to overwrite')
    original_load=Trainer._load_checkpoint
    original_loss=SplatfactoModel.get_loss_dict
    original_save=Trainer.save_checkpoint
    cached={}
    audit={'depth_batches':0,'empty_depth_batches':0,'first_raw_depth_loss':None,'last_raw_depth_loss':None}
    def load(self):
        self._start_step=state['step']+1
        self.pipeline.load_pipeline(state['pipeline'],state['step'])
        bind_loaded_optimizers(self,state)
        names=[Path(p).name for p in self.pipeline.datamanager.train_dataset.image_filenames]
        assert set(names)=={r['image'] for r in plan['rows']}
        for index,name in enumerate(names):
            with np.load(OUT/'depth_cache'/f'{Path(name).stem}.npz') as data:
                cached[index]=(torch.from_numpy(data['depth'].copy()),torch.from_numpy(data['weight'].copy()))
        cached['scale']=float(self.pipeline.datamanager.train_dataparser_outputs.dataparser_scale)
    def loss(self,outputs,batch,metrics_dict=None):
        losses=original_loss(self,outputs,batch,metrics_dict)
        if self.training:
            index=int(batch['image_idx'])
            depth,weight=cached[index]
            size=outputs['depth'].shape[:2]
            depth=F.interpolate(depth[None,None].to(self.device),size=size,mode='nearest')[0,0]
            weight=F.interpolate(weight[None,None].to(self.device),size=size,mode='nearest')[0,0]
            predicted=outputs['depth'].squeeze(-1)/cached['scale']
            raw=relative_depth_loss(predicted,depth,weight)
            losses['sensor_depth_loss']=plan['depth_weight']*raw
            audit['depth_batches']+=1
            audit['empty_depth_batches']+=int(not bool((weight>0).any()))
            value=float(raw.detach())
            if audit['first_raw_depth_loss'] is None:audit['first_raw_depth_loss']=value
            audit['last_raw_depth_loss']=value
        return losses
    def save_weights(self,step):
        # Storage-bounded experiment: only final inference weights, no Adam state.
        if step!=9999:return
        checkpoint.parent.mkdir(parents=True,exist_ok=True)
        torch.save({'step':step,'pipeline':self.pipeline.state_dict()},checkpoint)
    Trainer._load_checkpoint=load;SplatfactoModel.get_loss_dict=loss;Trainer.save_checkpoint=save_weights
    tick=perf_counter();torch.cuda.reset_peak_memory_stats()
    try:train(config)
    finally:Trainer._load_checkpoint=original_load;SplatfactoModel.get_loss_dict=original_loss;Trainer.save_checkpoint=original_save
    result=torch.load(checkpoint,map_location='cpu',weights_only=False)
    assert result['step']==9999
    params={k:v for k,v in result['pipeline'].items() if k.startswith('_model.gauss_params.')}
    assert all(torch.isfinite(v).all() for v in params.values())
    assert not torch.equal(params['_model.gauss_params.means'],state['pipeline']['_model.gauss_params.means'])
    audit.update(seconds=perf_counter()-tick,peak_cuda_allocated_mb=torch.cuda.max_memory_allocated()/2**20,
                 checkpoint=str(checkpoint),step=9999,gaussians=len(params['_model.gauss_params.means']),
                 optimizer_bindings_verified=True,depth_weight=plan['depth_weight'],dataparser_scale=cached['scale'],
                 inference_only_checkpoint=True,failed_attempt='Drive full at first save; reran from the untouched 5k source')
    (OUT/'training_runtime.json').write_text(json.dumps(audit,indent=2))
    print(json.dumps(audit,indent=2))


if __name__=='__main__':main()
