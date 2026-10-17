"""Matched 2k-step continuations testing existing dense GNT supervision."""
import argparse
import json
from pathlib import Path
import shutil
from time import perf_counter

import cv2
import numpy as np
import torch
import yaml
from nerfstudio.engine.trainer import Trainer
from nerfstudio.scripts.train import main as train
from nerfstudio.utils.eval_utils import eval_setup
from gsplat.rendering import rasterization

from continue_scene_control import bind_loaded_optimizers
from diagnose_gnt_transfer import digest,measure,quantize
from prepare_gnt_splat import archive,normalized_view,pose_line
from gnt_transfer_metrics import alternating_unseen


def prepare(root):
    base=root/'gnt_splat_scene'
    experiment=root/'gnt_dense_supervision'
    experiment.mkdir(parents=True,exist_ok=True)
    original=json.loads((base/'data_manifest.json').read_text())
    diagnosis=json.loads((root/'assets/gnt_transfer_diagnosis/results.json').read_text())
    dense=json.loads((root/'assets/dense_turn/progress.json').read_text())
    assert dense['checkpoint_sha256']==original['teacher_checkpoint_sha256']
    assert all(all(i%8 for i in row['sources']) for row in dense['frames'])
    for i,row in enumerate(dense['frames']):
        assert row['index']==i and digest(root/'assets/dense_turn'/f'frame_{i:04}.png')==row['sha256']
    added,withheld=alternating_unseen([r for r in diagnosis['rows'] if r['group']=='dense'])
    assert len(added)==85 and len(withheld)==84 and not set(added)&set(withheld)
    teacher=[r for r in original['rows'] if r['kind']=='gnt']
    rows=[r for r in original['rows'] if r['kind']=='gnt' or not r['train']]
    added_rows=[]
    for i in added:
        case=dense['cases'][i]
        row=dict(name=f'dense_{i:04}.png',r=case['r'],t=case['t'],train=True,
                 source_time=case['source_time'],sha256=dense['frames'][i]['sha256'])
        added_rows.append(row)
    folder=experiment/'data/dense';images=folder/'images';sparse=folder/'sparse/0'
    images.mkdir(parents=True,exist_ok=True);sparse.mkdir(parents=True,exist_ok=True)
    for row in rows+added_rows:
        source=base/'data/teacher/images'/row['name'] if not row['name'].startswith('dense_') else root/'assets/dense_turn'/f"frame_{int(row['name'][6:10]):04}.png"
        assert archive(source,images/row['name'])==row['sha256']
    (sparse/'images.txt').write_text(''.join(pose_line(i+1,np.array(r['r']),r['t'],r['name']) for i,r in enumerate(rows+added_rows)))
    for name in ('cameras.txt','points3D.txt'):
        shutil.copy2(base/'data/teacher/sparse/0'/name,sparse/name)
    for split in ('train','test','val'):
        names=[r['name'] for r in rows+added_rows if r['train']==(split=='train')]
        (folder/f'{split}_list.txt').write_text('\n'.join(names)+'\n')
    checkpoint=base/'models/teacher/splatfacto/run_10000/nerfstudio_models/step-000009999.ckpt'
    assert digest(checkpoint)==diagnosis['checkpoint_sha256']
    # Avoid reporting any adaptive target near an added training time as unseen.
    added_times=np.array([r['source_time'] for r in added_rows])
    adaptive=[r['index'] for r in diagnosis['rows'] if r['group']=='adaptive' and np.min(abs(added_times-r['source_time']))>=1e-4]
    plan=dict(source_sha256=digest(checkpoint),teacher_sha256=original['teacher_checkpoint_sha256'],
              added=added,withheld=withheld,adaptive=adaptive,additional_steps=2000,
              seed=42,gaussian_refinement=False,training_views=dict(control=275,dense=360),
              added_rows=added_rows)
    path=experiment/'plan.json'
    if path.exists():assert json.loads(path.read_text())==plan,'Plan changed'
    path.write_text(json.dumps(plan,indent=2))
    print(json.dumps(dict(phase='prepared',added=len(added),withheld=len(withheld),adaptive=len(adaptive))),flush=True)
    return experiment,checkpoint,plan,teacher


def fit(root,experiment,source,plan,arm):
    destination=experiment/f'models/{arm}/splatfacto/continued'
    checkpoint=destination/'nerfstudio_models/step-000011999.ckpt'
    if checkpoint.exists():
        assert json.loads((destination/'fit_results.json').read_text())['source_sha256']==plan['source_sha256']
        return checkpoint
    state=torch.load(source,map_location='cpu',weights_only=False)
    count=len(state['pipeline']['_model.gauss_params.means'])
    config=yaml.load((source.parent.parent/'config.yml').read_text(),Loader=yaml.Loader)
    config.data=config.data if arm=='control' else experiment/'data/dense'
    config.pipeline.datamanager.data=config.pipeline.datamanager.dataparser.data=config.data
    config.output_dir=experiment/'models';config.experiment_name=arm;config.timestamp='continued'
    config.machine.seed=42
    config.load_dir=config.load_config=config.load_step=None;config.load_checkpoint=source
    config.max_num_iterations=2000;config.steps_per_save=1000000
    config.pipeline.model.refine_every=1000000
    config.steps_per_eval_all_images=config.steps_per_eval_image=config.steps_per_eval_batch=0
    for optimizer in config.optimizers.values():
        if optimizer['scheduler'] is not None:optimizer['scheduler'].max_steps=12000
    original_load,original_save=Trainer._load_checkpoint,Trainer.save_checkpoint
    from nerfstudio.data.datamanagers.full_images_datamanager import FullImageDatamanager
    from gnt_balanced_sampling import balanced_schedule
    from collections import Counter
    original_next=FullImageDatamanager.next_train
    schedule=[];sample_counts=Counter();sample_names=[]
    original_count=plan['training_views']['control']
    added_count=plan['training_views']['dense']-original_count
    def load(self):
        self._start_step=10000
        self.pipeline.load_pipeline(state['pipeline'],state['step'])
        bind_loaded_optimizers(self,state)
        output=self.pipeline.datamanager.train_dataparser_outputs
        names=[Path(p).name for p in output.image_filenames]
        assert len(names)==(original_count if arm=='control' else original_count+added_count)
        assert all(n.startswith(('gnt_','dense_')) for n in names)
        assert sum(n.startswith('dense_') for n in names)==(0 if arm=='control' else added_count)
        if arm=='balanced90':
            sample_names.extend(names)
            schedule.extend(balanced_schedule([i for i,n in enumerate(names) if n.startswith('gnt_')],
                [i for i,n in enumerate(names) if n.startswith('dense_')],2000,seed=42))
        normalization=json.loads((source.parent.parent/'dataparser_transforms.json').read_text())
        np.testing.assert_allclose(output.dataparser_transform.numpy(),normalization['transform'])
        np.testing.assert_allclose(output.dataparser_scale,normalization['scale'])
    def save(self,step):
        if step!=11999:return
        checkpoint.parent.mkdir(parents=True,exist_ok=True)
        torch.save(dict(step=step,pipeline=self.pipeline.state_dict()),checkpoint)
    def next_balanced(self,step):
        index=schedule[step-10000];sample_counts[index]+=1
        data=self.cached_train[index].copy();data['image']=data['image'].to(self.device)
        camera=self.train_cameras[index:index+1].to(self.device)
        if camera.metadata is None:camera.metadata={}
        camera.metadata['cam_idx']=index
        return camera,data
    Trainer._load_checkpoint=load;Trainer.save_checkpoint=save
    if arm=='balanced90':FullImageDatamanager.next_train=next_balanced
    start=perf_counter();torch.cuda.reset_peak_memory_stats()
    try:train(config)
    finally:
        Trainer._load_checkpoint=original_load;Trainer.save_checkpoint=original_save
        FullImageDatamanager.next_train=original_next
    result=torch.load(checkpoint,map_location='cpu',weights_only=False)
    params={k:v for k,v in result['pipeline'].items() if k.startswith('_model.gauss_params.')}
    assert result['step']==11999 and all(len(v)==count and torch.isfinite(v).all() for v in params.values())
    assert not torch.equal(params['_model.gauss_params.means'],state['pipeline']['_model.gauss_params.means'])
    assert digest(source)==plan['source_sha256']
    report=dict(status='complete',arm=arm,seconds=perf_counter()-start,gaussians=count,
                source_sha256=plan['source_sha256'],checkpoint_sha256=digest(checkpoint),
                peak_cuda_allocated_mb=torch.cuda.max_memory_allocated()/2**20,
                optimizer_bindings_verified=True,additional_steps=2000,train_views=(original_count if arm=='control' else original_count+added_count))
    if arm=='balanced90':
        original_counts=[sample_counts[i] for i,n in enumerate(sample_names) if n.startswith('gnt_')]
        added_counts=[sample_counts[i] for i,n in enumerate(sample_names) if n.startswith('dense_')]
        assert sum(original_counts)==1800 and sum(added_counts)==200
        assert set(original_counts)<={1800//original_count,(1800+original_count-1)//original_count}
        assert set(added_counts)<={200//added_count,(200+added_count-1)//added_count}
        report['sampling']=dict(original_batches=1800,added_batches=200,original_fraction=.9,
            original_minmax=[min(original_counts),max(original_counts)],added_minmax=[min(added_counts),max(added_counts)],
            counts={sample_names[i]:sample_counts[i] for i in range(len(sample_names))})
    (destination/'fit_results.json').write_text(json.dumps(report,indent=2))
    print(json.dumps(report),flush=True)
    return checkpoint


@torch.inference_mode()
def evaluate(root,source,checkpoints,plan,teacher):
    folder=root/'assets/gnt_dense_supervision';folder.mkdir(parents=True,exist_ok=True)
    dense=json.loads((root/'assets/dense_turn/progress.json').read_text())
    adaptive=sorted(json.loads((root/'assets/adaptive_scene/progress.json').read_text())['native'].values(),key=lambda r:r['case']['source_time'])
    cases=[]
    for group,indices in [('added',plan['added']),('withheld',plan['withheld'])]:
        cases.extend((group,i,dense['cases'][i],root/'assets/dense_turn'/f'frame_{i:04}.png') for i in indices)
    cases.extend(('adaptive',i,adaptive[i]['case'],root/'assets/adaptive_scene'/adaptive[i]['file']) for i in plan['adaptive'])
    cases.extend(('original_training',i,r,root/'gnt_splat_scene/data/teacher/images'/r['name']) for i,r in enumerate(teacher))
    results={}
    for arm,checkpoint in dict(initial=source,**checkpoints).items():
        report_path=folder/f'{arm}.json'
        if report_path.exists():
            results[arm]=json.loads(report_path.read_text());assert results[arm]['checkpoint_sha256']==digest(checkpoint)
            continue
        _,pipeline,loaded,step=eval_setup(checkpoint.parent.parent/'config.yml',test_mode='inference')
        assert loaded==checkpoint
        model=pipeline.model;model.eval()
        normalization=json.loads((checkpoint.parent.parent/'dataparser_transforms.json').read_text())
        transform=np.array(normalization['transform']);scale=normalization['scale']
        k=dense['cases'][0]['k'];K=torch.tensor([[[k[0],0,k[2]],[0,k[1],k[3]],[0,0,1]]],device='cuda',dtype=torch.float32)
        colors=torch.cat([model.features_dc[:,None],model.features_rest],dim=1)
        rows=[]
        for group,i,case,target in cases:
            view=torch.tensor(normalized_view(case['r'],case['t'],transform,scale)[None],device='cuda',dtype=torch.float32)
            rgb,alpha,_=rasterization(means=model.means,quats=model.quats,scales=model.scales.exp(),
                opacities=model.opacities.sigmoid().flatten(),colors=colors,viewmats=view,Ks=K,
                width=960,height=540,packed=False,near_plane=.01,far_plane=1e10,
                sh_degree=3,sparse_grad=False,absgrad=model.strategy.absgrad,rasterize_mode='classic')
            image=quantize((rgb[0]+(1-alpha[0])*model._get_background_color()).cpu().numpy())
            reference=cv2.cvtColor(cv2.imread(str(target)),cv2.COLOR_BGR2RGB)
            filename=f'{arm}_{group}_{i:04}.png'
            assert cv2.imwrite(str(folder/filename),cv2.cvtColor(image,cv2.COLOR_RGB2BGR))
            rows.append(dict(group=group,index=i,file=filename,target=str(target.relative_to(root/'assets')) if group!='original_training' else None,
                             **measure(image,reference)))
            if len(rows)%50==0:print(json.dumps(dict(phase='evaluation',arm=arm,completed=len(rows),total=len(cases))),flush=True)
        summaries={g:dict(count=sum(r['group']==g for r in rows),**{metric:float(np.mean([r[metric] for r in rows if r['group']==g])) for metric in ('psnr','ssim')}) for g in ('added','withheld','adaptive','original_training')}
        results[arm]=dict(checkpoint_sha256=digest(checkpoint),summaries=summaries,rows=rows)
        report_path.write_text(json.dumps(results[arm],indent=2))
        del model,pipeline,colors;torch.cuda.empty_cache()
    assert digest(source)==plan['source_sha256']
    report=dict(status='complete',plan=plan,arms=results,
                note='Single-seed exploratory RGB-density ablation. Same initialization, optimizer state, steps, capacity and schedule; dense arm changes training targets and sampling distribution. Agreement with GNT is not photographic ground truth. No model promoted.')
    if 'balanced90' in checkpoints:
        report['additional_trials']={'balanced90':json.loads((checkpoints['balanced90'].parent.parent/'fit_results.json').read_text())}
    (folder/'results.json').write_text(json.dumps(report,indent=2))
    print(json.dumps({a:r['summaries'] for a,r in results.items()}),flush=True)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--root',type=Path,required=True)
    args=parser.parse_args();torch.set_num_threads(4)
    experiment,source,plan,teacher=prepare(args.root)
    checkpoints={}
    for arm in ('control','dense'):
        checkpoints[arm]=fit(args.root,experiment,source,plan,arm)
        torch.cuda.empty_cache()
    evaluate(args.root,source,checkpoints,plan,teacher)


if __name__=='__main__':main()
