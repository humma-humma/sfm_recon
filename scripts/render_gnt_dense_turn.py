"""Render 180 independent GNT views for the matched six-second turning clip."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import cv2
import numpy as np
import torch
from gnt_local_runtime import Dataset,Renderer,load_model,enable_compiled_view_attention
from gnt_traversal import traversal_cases,dense_route_cases


def main():
    parser=argparse.ArgumentParser()
    for key in ('trial','original','adaptive','output'):parser.add_argument('--'+key,type=Path,required=True)
    args=parser.parse_args();torch.set_num_threads(4)
    args.output.mkdir(parents=True,exist_ok=True)
    dataset=Dataset('scene');route=traversal_cases(dataset.poses,dataset.k)
    cases=dense_route_cases(route)
    original=json.loads((args.original/'manifest.json').read_text())
    adaptive=json.loads((args.adaptive/'progress.json').read_text())
    checkpoint=args.trial/'gnt_tuning/detail_control_006000.pth'
    digest=hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    assert digest==original['checkpoint_sha256']==adaptive['checkpoint_sha256']
    path=args.output/'progress.json'
    report=json.loads(path.read_text()) if path.exists() else dict(status='preflight',
        checkpoint_sha256=digest,start_seconds=46,duration_seconds=6,fps=30,width=960,height=540,
        total=180,cases=cases,frames=[],backend='compiled_view_attention',precision='fp32',chunk=4096,
        frame_interpolation=False,note='Each video frame is an independent GNT render at its exact camera timestamp. Piecewise linear centers and SLERP preserve the previous route; no optical-flow or crossfade frames.')
    assert report['checkpoint_sha256']==digest and report['cases']==cases
    def save():
        temp=path.with_suffix('.partial.json');temp.write_text(json.dumps(report,indent=2));temp.replace(path)
    save()
    for case in cases:dataset.data(case,960)
    reusable=[dict(case=case,path=args.original/row['frame'],sha256=row['sha256'],sources=row['sources']) for case,row in zip(route,original['frames'])]
    reusable += [dict(case=row['case'],path=args.adaptive/row['file'],sha256=row['sha256'],sources=row['sources']) for row in adaptive['native'].values()]
    assert json.loads((args.trial/'gnt_finalization/compile.json').read_text())['verified']
    os.environ.setdefault('TORCHINDUCTOR_CACHE_DIR',str(args.trial/'gnt_finalization/inductor_cache'))
    os.environ.setdefault('TRITON_CACHE_DIR',str(args.trial/'gnt_finalization/triton_cache'))
    model=load_model(args.trial,checkpoint);enable_compiled_view_attention(model)
    renderer=Renderer(model,optimized=True,chunk=4096,cache_features=True)
    report['status']='rendering';save()
    for i,case in enumerate(cases):
        file=args.output/f'frame_{i:04}.png'
        previous=next((r for r in report['frames'] if r['index']==i),None)
        if previous:
            assert hashlib.sha256(file.read_bytes()).hexdigest()==previous['sha256'];continue
        cached=next((row for row in reusable if abs(row['case']['source_time']-case['source_time'])<1e-8
            and np.allclose(row['case']['r'],case['r'],rtol=0,atol=1e-8)
            and np.allclose(row['case']['t'],case['t'],rtol=0,atol=1e-8)),None)
        if cached:
            assert hashlib.sha256(cached['path'].read_bytes()).hexdigest()==cached['sha256']
            shutil.copy2(cached['path'],file)
            metadata=dict(sources=cached['sources'],seconds=0.,reused=True)
        else:
            rgb,metadata=renderer.render(dataset,case,960)
            temp=file.with_name(file.stem+'.partial.png')
            assert cv2.imwrite(str(temp),cv2.cvtColor(np.uint8(np.clip(rgb,0,1)*255+.5),cv2.COLOR_RGB2BGR))
            temp.replace(file);metadata['reused']=False
        assert all(j%8 for j in metadata['sources'])
        report['frames'].append(dict(index=i,file=file.name,source_time=case['source_time'],
            sha256=hashlib.sha256(file.read_bytes()).hexdigest(),**metadata))
        report['completed']=len(report['frames'])
        report['estimated_remaining_seconds']=(180-report['completed'])*float(np.mean([r['seconds'] for r in report['frames'] if not r['reused']] or [65.]))
        save();print(json.dumps(dict(completed=report['completed'],total=180,seconds=metadata['seconds'],estimated_remaining_seconds=report['estimated_remaining_seconds'])),flush=True)
    assert hashlib.sha256(checkpoint.read_bytes()).hexdigest()==digest
    report['status']='rendered';save()


if __name__=='__main__':main()
