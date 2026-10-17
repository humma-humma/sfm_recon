"""Render selected training and held-out cameras without changing checkpoints."""
import json
from pathlib import Path
import cv2
import numpy as np
import torch
from nerfstudio.utils.eval_utils import eval_setup
from nerfstudio.data.utils.dataloaders import FixedIndicesEvalDataloader

ROOT=Path('/workspace')
OUT=ROOT/'outputs/scene_diagnosis'

def main():
    selected=json.loads((OUT/'render_selection.json').read_text())
    configurations={
        '5000':ROOT/'outputs/splat_surroundings_tests/train/scene_control/splatfacto/run_5000/config.yml',
        '10000':ROOT/'outputs/scene_control_continuation/train/scene_10000/splatfacto/continued/config.yml',
        '15000':ROOT/'outputs/scene_control_continuation/train/scene_15000/splatfacto/continued/config.yml'}
    scale=json.loads((configurations['5000'].parent/'dataparser_transforms.json').read_text())['scale']
    depth_paths={float(p.stem):p for p in (ROOT/'raw_scene/depth').glob('*.png')}
    for label,config in configurations.items():
        _,pipeline,checkpoint,step=eval_setup(config,test_mode='test')
        candidates=sorted((config.parent/'nerfstudio_models').glob('step-*.ckpt'))
        if len(candidates)!=1:
            raise ValueError(f'Expected exactly one checkpoint in {config.parent}, got {candidates}')
        checkpoint=candidates[0]
        state=torch.load(checkpoint,map_location='cpu',weights_only=False)
        step=state['step']
        pipeline.load_pipeline(state['pipeline'],step)
        del state
        pipeline.eval()
        folder=OUT/f'render_{label}'
        folder.mkdir(exist_ok=True)
        rows=[]
        with torch.no_grad():
            for split in ('train','eval'):
                dataset=getattr(pipeline.datamanager,f'{split}_dataset')
                names=[Path(p).name for p in dataset.image_filenames]
                wanted=[s['image'] for s in selected if s['split']==split]
                missing=set(wanted)-set(names)
                if missing:
                    raise ValueError(f'{split} missing: {missing}')
                indices=[names.index(name) for name in wanted]
                loader=FixedIndicesEvalDataloader(dataset,image_indices=indices,device=pipeline.device)
                for index,(camera,batch) in zip(indices,loader):
                    outputs=pipeline.model.get_outputs_for_camera(camera=camera)
                    metrics,_=pipeline.model.get_image_metrics_and_images(outputs,batch)
                    name=names[index]
                    rgb=outputs['rgb'].detach().cpu().numpy()
                    gt=batch['image'].detach().cpu().numpy()
                    if gt.dtype==np.uint8:
                        gt=gt.astype(float)/255
                    cv2.imwrite(str(folder/f'{split}_{Path(name).stem}.jpg'),cv2.cvtColor(np.uint8(np.clip(np.hstack((gt,rgb)),0,1)*255),cv2.COLOR_RGB2BGR))
                    row={'image':name,'split':split,**{k:float(metrics[k]) for k in ('psnr','ssim','lpips')}}
                    if outputs.get('depth') is not None:
                        predicted=outputs['depth'].detach().cpu().numpy().squeeze()/scale
                        measured=cv2.imread(str(depth_paths[float(Path(name).stem)]),cv2.IMREAD_UNCHANGED).astype(float)/1000
                        measured=cv2.resize(measured,(predicted.shape[1],predicted.shape[0]),interpolation=cv2.INTER_NEAREST)
                        opacity=outputs['accumulation'].detach().cpu().numpy().squeeze()
                        valid=(measured>.1)&(measured<8)&np.isfinite(predicted)&(opacity>.9)
                        residual=predicted[valid]-measured[valid]
                        row.update(depth_comparison_pixels=int(valid.sum()),
                                   median_absolute_rendered_depth_residual_m=float(np.median(np.abs(residual))) if len(residual) else None,
                                   rendered_depth_free_space_fraction=float((residual < -(.05+.03*measured[valid])).mean()) if len(residual) else None)
                        np.savez_compressed(folder/f'{split}_{Path(name).stem}_depth.npz',rendered_m=predicted,measured_m=measured,opacity=opacity)
                    rows.append(row)
                    print(json.dumps({'checkpoint':label,**row}),flush=True)
        report={'checkpoint':str(checkpoint),'step':step,'views':rows,
                'mean_by_split':{s:{k:float(np.mean([r[k] for r in rows if r['split']==s])) for k in ('psnr','ssim','lpips')} for s in ('train','eval')},
                'depth_caveat':'Expected rendered depth is not surface thickness; assumes optical-axis depth and source dataparser scale.'}
        (folder/'metrics.json').write_text(json.dumps(report,indent=2))
        del pipeline
        torch.cuda.empty_cache()
    (OUT/'render_complete.json').write_text(json.dumps({'complete':True}))

if __name__=='__main__':
    main()
