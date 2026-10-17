"""Reuse the scene route and spend a fixed budget on difficult camera intervals."""
import argparse
import hashlib
import heapq
import json
import os
from pathlib import Path
import cv2
import numpy as np
import torch
from gnt_adaptive import midpoint,rotation_degrees,motion_proxy,needs_refinement,sparse_source_support
from gnt_local_runtime import Dataset,Renderer,load_model,enable_compiled_view_attention
from gnt_traversal import traversal_cases


def main():
    parser=argparse.ArgumentParser()
    for key in ('trial','original','output'):parser.add_argument('--'+key,type=Path,required=True)
    parser.add_argument('--max-added',type=int,default=48)
    parser.add_argument('--max-probes',type=int,default=96)
    args=parser.parse_args();assert 0<args.max_added<=args.max_probes
    torch.set_num_threads(4)
    args.output.mkdir(parents=True,exist_ok=True)
    dataset=Dataset('scene');cases=traversal_cases(dataset.poses,dataset.k)
    manifest=json.loads((args.original/'manifest.json').read_text())
    checkpoint=args.trial/'gnt_tuning/detail_control_006000.pth'
    digest=hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    assert digest==manifest['checkpoint_sha256']
    path=args.output/'progress.json'
    report=json.loads(path.read_text()) if path.exists() else dict(status='planning',
        checkpoint_sha256=digest,max_added=args.max_added,max_probes=args.max_probes,
        max_depth=3,preview_width=160,width=960,chunk=4096,probes={},selected=[],native={},
        note='Geometry, source-set changes and low-resolution flow disagreement are diagnostic proxies, not calibrated uncertainty. More output views do not add source observations. Original route timing retained.')
    assert report['checkpoint_sha256']==digest and report['max_added']==args.max_added and report['max_probes']==args.max_probes
    def save():
        temp=path.with_suffix('.partial.json');temp.write_text(json.dumps(report,indent=2));temp.replace(path)
    def write_rgb(file,rgb):
        temp=file.with_name(file.stem+'.partial.png')
        assert cv2.imwrite(str(temp),cv2.cvtColor(np.uint8(np.clip(rgb,0,1)*255+.5),cv2.COLOR_RGB2BGR))
        temp.replace(file)
    def read_rgb(file):
        image=cv2.imread(str(file));assert image is not None,str(file)
        return cv2.cvtColor(cv2.resize(image,(160,90),interpolation=cv2.INTER_AREA),cv2.COLOR_BGR2RGB)
    nodes={};queue=[];counter=0
    for case,row in zip(cases,manifest['frames']):
        assert case['id']==row['id']
        file=args.original/row['frame']
        assert hashlib.sha256(file.read_bytes()).hexdigest()==row['sha256']
        nodes[case['id']]=dict(case=case,image=read_rgb(file),sources=row['sources'])
    def enqueue(a,b,depth):
        nonlocal counter
        angle=rotation_degrees(a['case'],b['case'])
        prediction,valid,flow=motion_proxy(a['image'],b['image'])
        union=set(a['sources'])|set(b['sources'])
        churn=1-len(set(a['sources'])&set(b['sources']))/len(union)
        priority=max(angle/3,flow/8,churn/.6,(1-float(valid.mean()))/.1)
        heapq.heappush(queue,(-priority,counter,a,b,depth,angle,prediction,valid,churn))
        counter+=1
    for a,b in zip(cases[:-1],cases[1:]):enqueue(nodes[a['id']],nodes[b['id']],0)
    assert json.loads((args.trial/'gnt_finalization/compile.json').read_text())['verified']
    os.environ.setdefault('TORCHINDUCTOR_CACHE_DIR',str(args.trial/'gnt_finalization/inductor_cache'))
    os.environ.setdefault('TRITON_CACHE_DIR',str(args.trial/'gnt_finalization/triton_cache'))
    model=load_model(args.trial,checkpoint);enable_compiled_view_attention(model)
    renderer=Renderer(model,optimized=True,chunk=4096,cache_features=True)
    selected=[];probe_count=0;unresolved=[]
    while queue and probe_count<args.max_probes and len(selected)<args.max_added:
        priority,_,a,b,depth,angle,prediction,valid,churn=heapq.heappop(queue)
        case=midpoint(a['case'],b['case']);key=case['id'];file=args.output/f'{key}_160.png'
        if key in report['probes']:
            row=report['probes'][key]
            assert hashlib.sha256(file.read_bytes()).hexdigest()==row['sha256']
            image=read_rgb(file)
        else:
            rgb,metadata=renderer.render(dataset,case,160);write_rgb(file,rgb)
            image=read_rgb(file)
            error=float(np.abs(image.astype(float)/255-prediction)[valid].mean()) if valid.any() else 1.
            row=dict(case=case,depth=depth,angle_degrees=angle,midpoint_mae=error,
                valid_fraction=float(valid.mean()),source_churn=churn,priority=-priority,
                sparse_frustum_support=sparse_source_support(dataset,case,metadata['sources']),
                sha256=hashlib.sha256(file.read_bytes()).hexdigest(),**metadata)
            report['probes'][key]=row;save()
        probe_count+=1
        low_support=row['sparse_frustum_support']<.7
        bad=needs_refinement(angle,row['midpoint_mae'],row['valid_fraction'],churn) or low_support
        if bad:
            selected.append(case)
            node=dict(case=case,image=image,sources=row['sources'])
            if depth<report['max_depth']-1 and not low_support:
                enqueue(a,node,depth+1);enqueue(node,b,depth+1)
            else:unresolved.append(key)
        print(json.dumps(dict(phase='preview',probes=probe_count,selected=len(selected),case=key,refine=bad)),flush=True)
    report.update(status='rendering',selected=selected,remaining_intervals=len(queue),
        depth_limited=unresolved,probes_evaluated=probe_count,budget_limited=bool(queue))
    save()
    for case in selected:
        key=case['id'];file=args.output/f'{key}_960.png'
        if key in report['native']:
            assert hashlib.sha256(file.read_bytes()).hexdigest()==report['native'][key]['sha256'];continue
        rgb,metadata=renderer.render(dataset,case,960);write_rgb(file,rgb)
        report['native'][key]=dict(case=case,file=file.name,sha256=hashlib.sha256(file.read_bytes()).hexdigest(),**metadata)
        save();print(json.dumps(dict(phase='native',completed=len(report['native']),total=len(selected),seconds=metadata['seconds'])),flush=True)
    assert hashlib.sha256(checkpoint.read_bytes()).hexdigest()==digest
    report['status']='rendered';save()


if __name__=='__main__':main()
