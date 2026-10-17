"""Matched fixed-pose splat fits with CPU image caching and bounded refinement."""
import argparse
import json
from pathlib import Path
from time import perf_counter
import torch
import yaml

ROOT=Path(__file__).resolve().parents[1]


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--dataset',choices=['scene','box','boot','milk'],default='scene')
    parser.add_argument('--base',type=Path)
    parser.add_argument('--arm',choices=['photos','mixed','teacher'],required=True);args=parser.parse_args()
    from nerfstudio.scripts.train import main as train
    from nerfstudio.engine.trainer import Trainer
    from continue_scene_control import bind_loaded_optimizers
    torch.set_num_threads(4)
    selected=json.loads((ROOT/'configs/splatting/selected_models.json').read_text())['datasets'][args.dataset]
    config=yaml.load((ROOT/selected['presentation_config']).read_text(),Loader=yaml.Loader)
    base=args.base or args.root/'gnt_splat_scene';config.data=base/'data'/args.arm
    config.pipeline.datamanager.data=config.data;config.pipeline.datamanager.dataparser.data=config.data
    config.output_dir=base/'models';config.experiment_name=args.arm;config.timestamp='run_10000'
    config.max_num_iterations=10000;config.steps_per_save=1000;config.save_only_latest_checkpoint=False
    if args.arm=='teacher':
        config.steps_per_save=10000;config.save_only_latest_checkpoint=True
    config.steps_per_eval_all_images=0;config.steps_per_eval_image=0;config.steps_per_eval_batch=0
    config.vis='tensorboard';config.load_config=config.load_checkpoint=config.load_dir=config.load_step=None
    config.pipeline.datamanager.cache_images='cpu'
    parser_config=config.pipeline.datamanager.dataparser
    parser_config.orientation_method='none';parser_config.center_method='none'
    parser_config.auto_scale_poses=False;parser_config.scale_factor=.05
    parser_config.downscale_factor=1
    model=config.pipeline.model;model.camera_optimizer.mode='off'
    model.num_downscales=1;model.resolution_schedule=1000
    model.refine_every=200;model.densify_grad_thresh=.0016;model.cull_alpha_thresh=.01
    model.stop_split_at=5000;model.rasterize_mode='classic';model.use_bilateral_grid=False
    for optimizer in config.optimizers.values():
        if optimizer['scheduler'] is not None:optimizer['scheduler'].max_steps=10000
    # Audit actual parser splits and pin normalization across both arms.
    dataparser=parser_config.setup()
    training=dataparser.get_dataparser_outputs('train');testing=dataparser.get_dataparser_outputs('test')
    assert not set(training.image_filenames)&set(testing.image_filenames)
    assert all(path.name.startswith('photo_') and int(path.stem.split('_')[1])%8==0 for path in testing.image_filenames)
    manifest=json.loads((base/'data_manifest.json').read_text())
    teacher_count=manifest['teacher_frames']
    expected=teacher_count if args.arm=='teacher' else manifest['original_train']+(teacher_count if args.arm=='mixed' else 0)
    assert len(training.image_filenames)==expected
    if args.arm=='teacher':assert all(path.name.startswith('gnt_') for path in training.image_filenames)
    assert len(testing.image_filenames)==manifest['heldout']
    checkpoint=config.get_checkpoint_dir()/'step-000009999.ckpt'
    if checkpoint.exists():print('Already trained: '+str(checkpoint),flush=True);return
    existing=sorted(config.get_checkpoint_dir().glob('step-*.ckpt'))
    original_load=Trainer._load_checkpoint
    if existing:
        resumed=torch.load(existing[-1],map_location='cpu',weights_only=False)
        config.load_dir=config.get_checkpoint_dir()
        config.max_num_iterations=10000-(resumed['step']+1)
        assert config.max_num_iterations>0
        def load(self):
            self._start_step=resumed['step']+1
            self.pipeline.load_pipeline(resumed['pipeline'],resumed['step'])
            bind_loaded_optimizers(self,resumed)
        Trainer._load_checkpoint=load
    start=perf_counter()
    try:train(config)
    finally:Trainer._load_checkpoint=original_load
    state=torch.load(checkpoint,map_location='cpu',weights_only=False)
    assert state['step']==9999
    for key,value in state['pipeline'].items():
        if key.startswith('_model.gauss_params.'):assert len(value) and torch.isfinite(value).all()
    report=dict(status='complete',arm=args.arm,iterations=10000,seconds=perf_counter()-start,
        checkpoint=str(checkpoint),gaussians=len(state['pipeline']['_model.gauss_params.means']),
        peak_cuda_allocated_mb=torch.cuda.max_memory_allocated()/2**20,
        train_views=len(training.image_filenames),test_views=len(testing.image_filenames),
        normalization=dict(transform=training.dataparser_transform.tolist(),scale=training.dataparser_scale),
        settings=dict(camera_optimizer='off',cache_images='cpu',num_downscales=1,resolution_schedule=1000,
            refine_every=200,densify_grad_thresh=.0016,cull_alpha_thresh=.01,stop_split_at=5000,
            use_absgrad=model.use_absgrad,rasterize_mode='classic'))
    (checkpoint.parent.parent/'fit_results.json').write_text(json.dumps(report,indent=2));print(json.dumps(report),flush=True)


if __name__=='__main__':main()
