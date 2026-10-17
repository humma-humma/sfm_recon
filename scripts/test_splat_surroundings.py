"""Controlled surroundings experiments; run inside the existing Nerfstudio image.

--prepare-only performs an offline input/config audit on the host. Training is
sequential, isolated from selected models, and never automatically promoted.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
from time import perf_counter

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'outputs/splat_surroundings_tests'
DATA={
    'box':'outputs/splat_refresh_v1/colmap/box',
    'boot':'outputs/stage2_boot_gaussian_colmap_half',
    'milk':'outputs/stage2_milk_gaussian_colmap_half',
    'scene':'outputs/stage3_gaussian_colmap_stride16_half_stair_rebalanced',
}

def save(path,value):path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(value,indent=2))
def digest(path):return hashlib.sha256(path.read_bytes()).hexdigest()

def prepare():
    selected=json.loads((ROOT/'configs/splatting/selected_models.json').read_text())['datasets']
    plan=dict(status='prepared',selected_models_unchanged=True,scene_experiments={
        'scene_control':{},'scene_fast_resolution':{'resolution_schedule':1000},
        'scene_fast_resolution_split4000':{'resolution_schedule':1000,'stop_split_at':4000}},
        object_experiments=['box_person_mask','boot_person_mask','milk_person_mask'],datasets={})
    for name,choice in selected.items():
        config=ROOT/choice['presentation_config'];checkpoint=ROOT/choice['presentation_checkpoint'];source=ROOT/DATA[name]
        assert config.is_file() and checkpoint.is_file()
        for file in ['cameras.txt','images.txt','points3D.txt']:assert (source/'sparse/0'/file).is_file()
        lines=iter((source/'sparse/0/images.txt').read_text().splitlines());images=[]
        for line in lines:
            if not line or line.startswith('#'):continue
            image=line.split()[9];next(lines);assert (source/'images'/image).is_file(),image;images.append(image)
        text=config.read_text();iterations=int(re.search(r'^max_num_iterations: (\d+)',text,re.M).group(1))
        plan['datasets'][name]=dict(config=choice['presentation_config'],config_sha256=digest(config),
            checkpoint=choice['presentation_checkpoint'],data=DATA[name],iterations=iterations,images=images,
            input_signature=hashlib.sha256(''.join(digest(source/'sparse/0'/f) for f in ['cameras.txt','images.txt','points3D.txt']).encode()).hexdigest())
    plan['signature']=hashlib.sha256(json.dumps(plan,sort_keys=True).encode()).hexdigest()
    path=OUT/'plan.json'
    if path.exists() and json.loads(path.read_text())['signature']!=plan['signature']:raise RuntimeError('Experiment inputs changed; use a new output directory rather than mixing runs.')
    save(path,plan);return plan

def run(label,args):
    log=OUT/'logs'/f'{label}.log';log.parent.mkdir(parents=True,exist_ok=True)
    with log.open('a') as stream:
        process=subprocess.Popen([sys.executable,*map(str,args)],cwd=ROOT,env=dict(os.environ,PYTHONPATH=str(ROOT/'src'),PYTHONUNBUFFERED='1'),stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True)
        for line in process.stdout:print(line,end='',flush=True);stream.write(line);stream.flush()
        if process.wait():raise RuntimeError(f'{label} failed. See {log}')

def train_case(plan,label,name,changes,data=None):
    import torch
    import yaml
    from nerfstudio.scripts.train import main as train
    record=plan['datasets'][name];config=yaml.load((ROOT/record['config']).read_text(),Loader=yaml.Loader)
    config.data=(data or ROOT/record['data']).resolve()
    config.pipeline.datamanager.data=config.pipeline.datamanager.dataparser.data=config.data
    config.output_dir=OUT/'train';config.experiment_name=label;config.timestamp=f'run_{record["iterations"]}'
    config.load_config=config.load_dir=config.load_checkpoint=config.load_step=None;config.vis='tensorboard'
    config.steps_per_save=1000;config.steps_per_eval_all_images=1000000
    assert config.max_num_iterations==record['iterations'] and config.pipeline.model.camera_optimizer.mode=='off'
    assert config.pipeline.datamanager.dataparser.eval_mode=='interval' and config.pipeline.datamanager.dataparser.eval_interval==8
    for key,value in changes.items():setattr(config.pipeline.model,key,value)
    if data:config.pipeline.datamanager.dataparser.masks_path=Path('masks')
    checkpoint=config.get_checkpoint_dir()/f'step-{config.max_num_iterations-1:09d}.ckpt'
    runtime=checkpoint.parent.parent/'experiment_runtime.json'
    signature=hashlib.sha256(json.dumps(dict(plan=plan['signature'],label=label,changes=changes),sort_keys=True).encode()).hexdigest()
    if checkpoint.exists():
        if not runtime.exists() or json.loads(runtime.read_text())['signature']!=signature:raise RuntimeError(f'Incomplete/stale run at {checkpoint}; refusing silent reuse.')
        return checkpoint,json.loads(runtime.read_text())
    torch.cuda.reset_peak_memory_stats();tick=perf_counter();train(config)
    assert checkpoint.exists();state=torch.load(checkpoint,map_location='cpu',weights_only=False)
    for key,value in state['pipeline'].items():
        if key.startswith('_model.gauss_params.'):assert len(value)>0 and torch.isfinite(value).all(),key
    result=dict(signature=signature,seconds=perf_counter()-tick,iterations=config.max_num_iterations,changes=changes,
        peak_cuda_allocated_mb=torch.cuda.max_memory_allocated()/2**20,gaussians=len(state['pipeline']['_model.gauss_params.means']),seed=config.machine.seed,config=str(checkpoint.parent.parent/'config.yml'))
    save(runtime,result);return checkpoint,result

def make_masks(plan):
    import numpy as np
    import torch
    import torch.nn.functional as F
    import cv2
    from PIL import Image
    from torchvision.models.segmentation import lraspp_mobilenet_v3_large,LRASPP_MobileNet_V3_Large_Weights
    weights=LRASPP_MobileNet_V3_Large_Weights.DEFAULT
    # Uses official Torchvision weights (~12.5 MB); cached under mounted torch hub.
    model=lraspp_mobilenet_v3_large(weights=weights).eval().cuda();preprocess=weights.transforms();person=weights.meta['categories'].index('person')
    results={}
    with torch.no_grad():
        for name in ['box','boot','milk']:
            source=ROOT/plan['datasets'][name]['data'];target=OUT/'masked_data'/name
            marker=target/'mask_report.json'
            if marker.exists():
                cached=json.loads(marker.read_text())
                if cached.get('signature')!=plan['signature']:raise RuntimeError(f'Stale person masks: {marker}')
                results[name]=cached;continue
            (target/'masks').mkdir(parents=True,exist_ok=True);(target/'images').mkdir(exist_ok=True)
            shutil.copytree(source/'sparse',target/'sparse',dirs_exist_ok=True)
            rows=[];panels=[]
            for index,image_name in enumerate(plan['datasets'][name]['images']):
                path=source/'images'/image_name;image=Image.open(path).convert('RGB');array=np.array(image)
                logits=model(preprocess(image).unsqueeze(0).cuda())['out']
                probabilities=F.interpolate(logits,size=(image.height,image.width),mode='bilinear',align_corners=False).softmax(dim=1)[0]
                removed=((probabilities.argmax(dim=0)==person)&(probabilities[person]>=.6)).cpu().numpy().astype('uint8')
                removed=cv2.dilate(removed,np.ones((7,7),dtype='uint8'))
                mask=(1-removed)*255;Image.fromarray(mask).save(target/'masks'/Path(image_name).with_suffix('.png'))
                destination=target/'images'/image_name
                if not destination.exists():
                    try:os.link(path,destination)
                    except OSError:shutil.copy2(path,destination)
                fraction=float(removed.mean());rows.append(dict(image=image_name,removed_fraction=fraction))
                if fraction>.001:
                    overlay=array.copy();overlay[removed>0]=(overlay[removed>0]*.4+np.array([255,30,30])*.6).astype('uint8')
                    panels.append((fraction,cv2.resize(overlay,(480,270))))
                print(f'mask {name} {index+1}/{len(plan["datasets"][name]["images"])}: {fraction:.3%}',flush=True)
            results[name]=dict(signature=plan['signature'],status='ready' if any(r['removed_fraction']>.001 for r in rows) else 'no_detected_people',
                model='torchvision LRASPP MobileNetV3 COCO/VOC',person_confidence=.6,dilation_pixels=7,
                rows=rows,visual_review_required=True,mean_removed_fraction=float(np.mean([r['removed_fraction'] for r in rows])))
            if panels:
                panels=sorted(panels,key=lambda row:row[0],reverse=True)[:12]
                Image.fromarray(np.vstack([p[1] for p in panels])).save(target/'mask_overlay.jpg',quality=92)
            save(marker,results[name])
    return results

def finalize(plan):
    import numpy as np
    from PIL import Image
    from skimage.metrics import structural_similarity
    from scipy.ndimage import binary_erosion
    from sfm_reconstruction.splat_cleanup_evaluation import compare_candidate
    workflow=json.loads((OUT/'workflow.json').read_text());comparisons={}
    html=['<!doctype html><meta charset="utf-8"><title>Surroundings experiment results</title><style>body{background:#111710;color:#eee;font:14px Segoe UI;margin:24px}img{max-width:100%;display:block}a{color:#cde5b0}p{max-width:1100px;line-height:1.6}</style><h1>Surroundings experiments</h1><p>Each triptych: source photograph / selected historical model / experimental candidate. Same unmasked held-out images and original camera poses. No selected models were replaced. People masks are automatic and need visual review.</p>']
    for label,entry in workflow['experiments'].items():
        if entry['status']!='evaluated':continue
        name=entry['dataset'];candidate=json.loads((OUT/'evaluation'/label/'metrics.json').read_text())
        baseline_dir=ROOT/'outputs/splat_refresh_v1/evaluation'/('box_new' if name=='box' else name+'_old')
        baseline=json.loads((baseline_dir/'metrics.json').read_text())
        comparisons[label]=dict(against_selected=compare_candidate(baseline,candidate),runtime=entry['runtime'])
        if name=='scene' and label!='scene_control':
            control=json.loads((OUT/'evaluation/scene_control/metrics.json').read_text())
            comparisons[label]['against_fresh_control']=compare_candidate(control,candidate)
        html.append(f'<h2>{label}</h2><p>Training: {entry["runtime"]["seconds"]/60:.1f} min; peak allocated CUDA: {entry["runtime"]["peak_cuda_allocated_mb"]:.0f} MiB; {entry["runtime"]["gaussians"]:,} Gaussians.</p>')
        gate=comparisons[label].get('against_fresh_control',comparisons[label]['against_selected'])
        html.append(f'<p>Comparison against {"fresh control" if "against_fresh_control" in comparisons[label] else "selected model"}: {gate["status"]}; mean PSNR change {gate["mean_deltas"]["psnr"]:+.3f} dB; worst view {gate["worst_view_psnr_delta"]:+.3f} dB. Full metrics: <a href="comparisons.json">comparisons.json</a>.</p>')
        if name!='scene':html.append(f'<p>Person-mask overlay (largest detections; red = excluded pixels):</p><img src="masked_data/{name}/mask_overlay.jpg">')
        before={row['image']:i for i,row in enumerate(baseline['views'])};static=[]
        control_indices=None
        if name=='scene' and label!='scene_control':
            control_indices={row['image']:i for i,row in enumerate(control['views'])}
            html.append('<p>Scene rows below: source / selected historical model / fresh control / candidate.</p>')
        for i,row in enumerate(candidate['views']):
            old_pair=Image.open(baseline_dir/f'{before[row["image"]]:03d}_img.png').convert('RGB')
            new_pair=Image.open(OUT/f'evaluation/{label}/{i:03d}_img.png').convert('RGB');w=old_pair.width//2;h=old_pair.height
            assert new_pair.size==old_pair.size
            source=old_pair.crop((0,0,w,h));old=old_pair.crop((w,0,2*w,h));new=new_pair.crop((w,0,2*w,h))
            assert np.array_equal(np.asarray(source),np.asarray(new_pair.crop((0,0,w,h)))), 'Evaluation source pixels differ'
            panels=[source,old]
            if control_indices is not None:
                control_pair=Image.open(OUT/f'evaluation/scene_control/{control_indices[row["image"]]:03d}_img.png').convert('RGB')
                assert control_pair.size==old_pair.size
                assert np.array_equal(np.asarray(source),np.asarray(control_pair.crop((0,0,w,h))))
                panels.append(control_pair.crop((w,0,2*w,h)))
            panels.append(new)
            triptych=Image.new('RGB',(w*len(panels),h))
            for panel_index,panel in enumerate(panels):triptych.paste(panel,(w*panel_index,0))
            file=f'comparisons/{label}/{i:03d}.jpg';(OUT/file).parent.mkdir(parents=True,exist_ok=True);triptych.save(OUT/file,quality=95)
            html.append(f'<p>{row["image"]}</p><a href="{file}"><img loading="lazy" src="{file}"></a>')
            if name!='scene':
                valid=np.array(Image.open(OUT/f'masked_data/{name}/masks'/Path(row['image']).with_suffix('.png')))>0
                valid=binary_erosion(valid,iterations=5,border_value=0)
                if valid.sum()<100:raise ValueError('Too few static evaluation pixels')
                gt=np.array(source,dtype=float);scores={}
                for key,image in [('selected',old),('candidate',new)]:
                    pixels=np.array(image,dtype=float);mse=float(np.mean((pixels[valid]-gt[valid])**2))
                    _,ssim_map=structural_similarity(gt,pixels,data_range=255,channel_axis=2,gaussian_weights=True,sigma=1.5,use_sample_covariance=False,full=True)
                    scores[key]=dict(psnr=float(10*np.log10(255**2/max(mse,1e-12))),ssim=float(ssim_map[valid].mean()))
                static.append(dict(image=row['image'],valid_pixels=int(valid.sum()),**scores))
        if static:
            comparisons[label]['same_static_pixels']=dict(views=static,mean_psnr_delta=float(np.mean([r['candidate']['psnr']-r['selected']['psnr'] for r in static])),mean_ssim_delta=float(np.mean([r['candidate']['ssim']-r['selected']['ssim'] for r in static])))
            comparisons[label]['full_image_gate_caveat']='Full-image scores penalize removal of people visible in reference photos. Static-region comparison uses the same person masks for both models; inspect both and the overlays. No automatic promotion.'
    save(OUT/'comparisons.json',comparisons);(OUT/'index.html').write_text('\n'.join(html),encoding='utf-8')
    print('Gallery: outputs/splat_surroundings_tests/index.html',flush=True)

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--prepare-only',action='store_true');parser.add_argument('--finalize-only',action='store_true')
    parser.add_argument('--train-case');parser.add_argument('--make-masks',action='store_true');args=parser.parse_args()
    plan=prepare()
    if args.prepare_only:print(json.dumps(plan,indent=2));return
    if args.finalize_only:finalize(plan);return
    if args.make_masks:save(OUT/'mask_results.json',make_masks(plan));return
    if args.train_case:
        label=args.train_case
        if label in plan['scene_experiments']:checkpoint,runtime=train_case(plan,label,'scene',plan['scene_experiments'][label])
        else:
            name=label.removesuffix('_person_mask');assert name in ['box','boot','milk']
            checkpoint,runtime=train_case(plan,label,name,{},OUT/'masked_data'/name)
        save(OUT/'runs'/f'{label}.json',dict(checkpoint=str(checkpoint),runtime=runtime));return
    import torch
    if not torch.cuda.is_available():raise RuntimeError('CUDA required')
    path=OUT/'workflow.json';workflow=json.loads(path.read_text()) if path.exists() else dict(signature=plan['signature'],status='running',experiments={},masks={'status':'pending'})
    if workflow['signature']!=plan['signature']:raise RuntimeError('Input signature mismatch')
    save(path,workflow)
    cases=[(label,'scene') for label in plan['scene_experiments']]
    # Scene experiments complete even if the optional official segmentation download fails.
    for label,name in cases:
        if workflow['experiments'].get(label,{}).get('status')=='evaluated':continue
        run('train_'+label,['scripts/test_splat_surroundings.py','--train-case',label])
        result=json.loads((OUT/'runs'/f'{label}.json').read_text())
        run('eval_'+label,['scripts/evaluate_splat_checkpoint.py','--config',ROOT/plan['datasets'][name]['config'],'--checkpoint',result['checkpoint'],'--output',OUT/'evaluation'/label])
        workflow['experiments'][label]=dict(status='evaluated',dataset=name,**result);save(path,workflow)
    if workflow['masks'].get('status')!='ready':
        try:
            run('person_masks',['scripts/test_splat_surroundings.py','--make-masks'])
            workflow['masks']=dict(status='ready',datasets=json.loads((OUT/'mask_results.json').read_text()))
        except RuntimeError as error:workflow['masks']=dict(status='blocked',reason=str(error))
        save(path,workflow)
    if workflow['masks']['status']=='ready':
        for name,mask_record in workflow['masks']['datasets'].items():
            label=name+'_person_mask'
            if mask_record['status']!='ready':workflow['experiments'][label]=dict(status='skipped_no_detected_people',dataset=name);save(path,workflow);continue
            if workflow['experiments'].get(label,{}).get('status')=='evaluated':continue
            run('train_'+label,['scripts/test_splat_surroundings.py','--train-case',label])
            result=json.loads((OUT/'runs'/f'{label}.json').read_text())
            # Evaluate through ORIGINAL UNMASKED config; same image pixels, no masked-score inflation.
            run('eval_'+label,['scripts/evaluate_splat_checkpoint.py','--config',ROOT/plan['datasets'][name]['config'],'--checkpoint',result['checkpoint'],'--output',OUT/'evaluation'/label])
            workflow['experiments'][label]=dict(status='evaluated',dataset=name,**result);save(path,workflow)
    workflow['status']='finalizing';save(path,workflow)
    finalize(plan)
    workflow['status']='complete' if workflow['masks']['status']=='ready' else 'scene_complete_mask_tests_blocked';save(path,workflow)
    print('Finished:',workflow['status'],flush=True)

if __name__=='__main__':main()
