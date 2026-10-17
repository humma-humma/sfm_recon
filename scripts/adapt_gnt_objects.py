"""Balanced, isolated GNT adaptation for box, boot and milk with review renders."""
import argparse
import hashlib
import json
import os
from pathlib import Path
from time import perf_counter
import cv2
import numpy as np
import torch
from gnt_local_runtime import Dataset,Renderer,load_model,load_weights,enable_sdpa,enable_compiled_view_attention,balanced_targets


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--trial',type=Path,required=True)
    parser.add_argument('--render-scratch',type=Path,required=True)
    parser.add_argument('--dataset',choices=['box','boot','milk'],required=True)
    args=parser.parse_args();torch.set_num_threads(4)
    torch.manual_seed(2026);np.random.seed(2026)
    output=args.trial/'gnt_finalization'/args.dataset;output.mkdir(parents=True,exist_ok=True)
    report_path=output/'results.json'
    weights=args.trial/'GNT/generalization_200000.pth'
    digest=hashlib.sha256(weights.read_bytes()).hexdigest()
    dataset=Dataset(args.dataset)
    heldout=[i for i in range(len(dataset.poses)) if i%8==0]
    groups=balanced_targets(len(dataset.poses));rejected=[]
    for group in groups:
        for i in list(group):
            try:dataset.data(dataset.case(i),480,training=True)
            except AssertionError:group.remove(i);rejected.append(i)
    assert all(groups)
    report=json.loads(report_path.read_text()) if report_path.exists() else dict(
        dataset=args.dataset,status='training',initial_checkpoint_sha256=digest,groups=groups,rejected=rejected,
        validation=heldout[::2],review=heldout[1::2],losses=[],checkpoints={},runs=[],
        schedule=[dict(end=600,width=480,learning_rates=[2e-5,1e-5]),
                  dict(end=1800,width=960,learning_rates=[1e-5,5e-6]),
                  dict(end=6000,width=960,learning_rates=[1e-5,5e-6])],
        rays=256,source_views=6,samples=64,seed=2026,training_seconds=0,
        note='Separate model initialized from official pretrained GNT for each object. Balanced capture groups; RGB MSE; interval-8 RGB excluded from targets and sources, each training target excluded from sources. Camera estimation predates split. All inspected views are exploratory, not a pristine benchmark. Fresh Adam at schedule boundaries matches scene training phases; balanced coverage replaces scene-local initial targeting.')
    assert report['initial_checkpoint_sha256']==digest
    assert report['groups']==groups
    def save():report_path.write_text(json.dumps(report,indent=2))
    model=load_model(args.trial,weights)
    from gnt.sample_ray import RaySamplerSingleImage,rng as ray_rng
    from gnt.projection import Projector
    from gnt.render_ray import render_rays
    projector=Projector('cuda:0')
    completed=max([int(s) for s in report['checkpoints']] or [0])
    rng=np.random.default_rng(2027);ray_rng.seed(2026)
    if completed:
        path=output/report['checkpoints'][str(completed)]['path'];load_weights(model,path)
        state=torch.load(output/f'state_{completed:06}.pth',map_location='cpu',weights_only=False)
        rng.bit_generator.state=state['numpy_rng'];ray_rng.set_state(state['ray_rng'])
        torch.set_rng_state(state['torch_rng']);torch.cuda.set_rng_state_all(state['cuda_rng'])
    parameters=list(model.net_coarse.parameters())+list(model.feature_net.parameters())
    optimizer=None
    for phase in report['schedule']:
        if completed>=phase['end']:continue
        optimizer=torch.optim.Adam([dict(params=model.net_coarse.parameters(),lr=phase['learning_rates'][0]),
                                    dict(params=model.feature_net.parameters(),lr=phase['learning_rates'][1])])
        tick=perf_counter();previous_seconds=report['training_seconds']
        for step in range(completed+1,phase['end']+1):
            idx=int(rng.choice(groups[(step-1)%6]))
            data,indices=dataset.data(dataset.case(idx),phase['width'],training=True)
            sampler=RaySamplerSingleImage(data,'cuda:0');batch=sampler.random_sample(256,'uniform')
            model.switch_to_train();optimizer.zero_grad(set_to_none=True)
            features=model.feature_net(batch['src_rgbs'].squeeze(0).permute(0,3,1,2))
            result=render_rays(batch,model,features,projector,64,inv_uniform=True,N_importance=0,det=False,single_net=True)
            loss=((result['outputs_coarse']['rgb']-batch['rgb'])**2).mean()
            assert torch.isfinite(loss), 'Nonfinite object training loss'
            loss.backward();torch.nn.utils.clip_grad_norm_(parameters,1.);optimizer.step()
            if step%50==0:
                row=dict(step=step,mse=loss.item(),training_seconds=previous_seconds+perf_counter()-tick,
                    width=phase['width'],target=idx)
                report['losses'].append(row);save();print(json.dumps(dict(dataset=args.dataset,**row)),flush=True)
            del result,features,loss,batch,sampler,data
        report['training_seconds']=previous_seconds+perf_counter()-tick
        completed=phase['end'];path=output/f'balanced_{completed:06}.pth'
        torch.save(dict(net_coarse=model.net_coarse.state_dict(),feature_net=model.feature_net.state_dict()),path)
        torch.save(dict(numpy_rng=rng.bit_generator.state,ray_rng=ray_rng.get_state(),
            torch_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state_all()),output/f'state_{completed:06}.pth')
        report['checkpoints'][str(completed)]=dict(path=path.name,sha256=hashlib.sha256(path.read_bytes()).hexdigest())
        save()
    runtime=json.loads((args.trial/'gnt_finalization/runtime.json').read_text())
    selection=runtime['selection'];optimized=selection.get('verified') and selection['label']!='reference'
    compile_path=args.trial/'gnt_finalization/compile.json'
    compiler=json.loads(compile_path.read_text()) if compile_path.exists() else {}
    if compiler.get('verified'):
        os.environ.setdefault('TORCHINDUCTOR_CACHE_DIR',str(args.trial/'gnt_finalization/inductor_cache'))
        os.environ.setdefault('TRITON_CACHE_DIR',str(args.trial/'gnt_finalization/triton_cache'))
        enable_compiled_view_attention(model);optimized=True;backend='compiled_view_attention'
    elif optimized:enable_sdpa(model);backend='sdpa'
    else:backend='reference'
    if report['runs']:assert report['inference_backend']==backend, 'Preserve existing renders; inference backend changed'
    report['inference_backend']=backend;report['status']='rendering'
    render_chunk=compiler.get('chunk',4096) if backend=='compiled_view_attention' else (selection['chunk'] if optimized else 4096)
    if report['runs']:assert report['render_chunk']==render_chunk, 'Preserve existing renders; inference chunk changed'
    report['render_chunk']=render_chunk
    del optimizer
    torch.cuda.empty_cache()
    manifest=json.loads((args.render_scratch/'render_manifest.json').read_text())['datasets'][args.dataset]
    anchor=next(c for c in manifest['cases'] if c['id']=='eval_024_z1.0')
    review_cases=[dict(c) for c in manifest['cases'] if c['index']==anchor['index'] and c['kind']=='heldout']
    review_cases += [dict(c) for c in manifest['cases'] if c['id'] in ('motion_00','motion_04','motion_08')]
    # Native stills at separated held-out views, focal close-up and three virtual poses.
    review_cases += [dataset.case(i) for i in report['review'] if i!=anchor['index']][:2]
    report['review_cases']=review_cases;save()
    from skimage.metrics import structural_similarity
    for label,path in [('pretrained',weights),('balanced1800',output/'balanced_001800.pth'),('balanced6000',output/'balanced_006000.pth')]:
        load_weights(model,path);renderer=Renderer(model,optimized=optimized,chunk=render_chunk,cache_features=True)
        cases=[(dataset.case(i),480) for i in heldout]
        cases += [(c,480 if c['kind']=='motion' else 960) for c in review_cases]
        for case,width in cases:
            key=(label,case['id'],width)
            existing=next((r for r in report['runs'] if (r['label'],r['case'],r['width'])==key),None)
            destination=output/'renders'/label/f'{case["id"]}_{width}.png'
            if existing and destination.exists():continue
            rgb,metadata=renderer.render(dataset,case,width)
            destination.parent.mkdir(parents=True,exist_ok=True)
            assert cv2.imwrite(str(destination),cv2.cvtColor(np.uint8(np.clip(rgb,0,1)*255+.5),cv2.COLOR_RGB2BGR))
            row=metadata|dict(case=case['id'],kind=case['kind'],label=label,path=str(destination.relative_to(output)),psnr=None,ssim=None)
            if case['kind']=='heldout' and case.get('zoom',1.)==1.:
                gt=cv2.resize(dataset.image(case['index']),(rgb.shape[1],rgb.shape[0]),interpolation=cv2.INTER_AREA)/255.
                row.update(psnr=float(-10*np.log10(max(np.mean((rgb-gt)**2),1e-16))),
                    ssim=float(structural_similarity(rgb,gt,channel_axis=2,data_range=1)))
            report['runs'].append(row);save();print(json.dumps(dict(dataset=args.dataset,**row)),flush=True)
    report['validation_means']={label:float(np.mean([r['psnr'] for r in report['runs'] if r['label']==label and r['width']==480 and r['case'] in [dataset.case(i)['id'] for i in report['validation']]])) for label in ('pretrained','balanced1800','balanced6000')}
    report['status']='complete';save()
    assert hashlib.sha256(weights.read_bytes()).hexdigest()==digest
    print('Object complete: '+args.dataset,flush=True)

if __name__=='__main__':main()
