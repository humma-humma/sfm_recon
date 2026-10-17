"""Resumable full-scene GNT at 15 independently rendered views per second."""
import argparse
import json
import os
from pathlib import Path
import shutil
import statistics
import cv2
import numpy as np
import torch
from gnt_local_runtime import Dataset,Renderer,load_model,enable_compiled_view_attention
from gnt_traversal import traversal_cases
from gnt_scene15_plan import scene15_cases,matching_camera
from gnt_rollout_io import digest,save


def main():
    parser=argparse.ArgumentParser()
    for key in ('root','trial','original'):parser.add_argument('--'+key,type=Path,required=True)
    args=parser.parse_args();torch.set_num_threads(4)
    folder=args.root/'assets/scene15';folder.mkdir(parents=True,exist_ok=True)
    dataset=Dataset('scene');route=traversal_cases(dataset.poses,dataset.k);cases=scene15_cases(route)
    checkpoint=args.trial/'gnt_tuning/detail_control_006000.pth';checkpoint_hash=digest(checkpoint)
    original=json.loads((args.original/'manifest.json').read_text())
    manifest=json.loads((args.root/'gnt_splat_scene/data_manifest.json').read_text())
    dense=json.loads((args.root/'assets/dense_turn/progress.json').read_text())
    adaptive=json.loads((args.root/'assets/adaptive_scene/progress.json').read_text())
    assert checkpoint_hash==original['checkpoint_sha256']==manifest['teacher_checkpoint_sha256']==dense['checkpoint_sha256']==adaptive['checkpoint_sha256']
    sparse=[r for r in manifest['rows'] if r['kind']=='gnt'];assert len(sparse)==len(route)==275
    for case,row in zip(route,sparse):
        np.testing.assert_allclose(case['r'],row['r'],atol=1e-12);np.testing.assert_allclose(case['t'],row['t'],atol=1e-12)
        assert abs(case['source_time']-row['source_time'])<1e-8
    candidates=[dict(case=case,path=args.original/row['frame'],sha256=row['sha256'],sources=row['sources']) for case,row in zip(route,original['frames'])]
    candidates += [dict(case=c,path=args.root/'assets/dense_turn'/row['file'],sha256=row['sha256'],sources=row['sources']) for c,row in zip(dense['cases'],dense['frames'])]
    candidates += [dict(case=row['case'],path=args.root/'assets/adaptive_scene'/row['file'],sha256=row['sha256'],sources=row['sources']) for row in adaptive['native'].values()]
    matches=[matching_camera(c,candidates) for c in cases]
    inputs={str(p.relative_to(dataset.root)):digest(p) for p in [dataset.root/'sparse/0'/n for n in ('cameras.txt','images.txt','points3D.txt')]}
    inputs.update({f'images/{name}':digest(dataset.root/'images'/name) for i,(name,_,_) in enumerate(dataset.poses) if i%8})
    plan=dict(checkpoint_sha256=checkpoint_hash,input_hashes=inputs,cases=cases,total=len(cases),
              fps=15,duration_seconds=len(cases)/15,width=960,height=540,cached_exact_poses=sum(r is not None for r in matches),
              settings=dict(backend='compiled_view_attention',precision='fp32',chunk=4096,source_views=8,samples=64),
              note='Scene only. Preserved captured route with exact 15fps sampling. 2198 native frames; duration rounded up by 1/30 second. No splats, interpolation, exposure adjustment or sharpening in native frames.')
    plan_path=folder/'plan.json'
    if plan_path.exists():assert json.loads(plan_path.read_text())==plan,'Inputs or configuration changed'
    save(plan_path,plan);plan_hash=digest(plan_path)
    path=folder/'progress.json'
    report=json.loads(path.read_text()) if path.exists() else dict(status='preflight',checkpoint_sha256=checkpoint_hash,plan_sha256=plan_hash,total=len(cases),frames={})
    assert report['checkpoint_sha256']==checkpoint_hash and report['plan_sha256']==plan_hash
    def update():
        values=list(report['frames'].values());timings=[r['seconds'] for r in values if not r['reused']]
        report.update(completed=len(values),reused=sum(r['reused'] for r in values),new_rendered=sum(not r['reused'] for r in values),
                      estimated_remaining_seconds=(len(cases)-len(values))*statistics.median(timings or [64.35]))
        save(path,report)
    update()
    for key,row in report['frames'].items():
        assert int(key)==row['index'] and digest(folder/row['file'])==row['sha256']
        assert all(i%8 for i in row['sources'])
    # Copy all reusable views up front, including views later in the route.
    for i,matched in enumerate(matches):
        if str(i) in report['frames'] or matched is None:continue
        assert digest(matched['path'])==matched['sha256'] and all(j%8 for j in matched['sources'])
        file=folder/f'frame_{i:04}.png'
        if not file.exists():
            try:os.link(matched['path'],file)
            except OSError:shutil.copy2(matched['path'],file)
        assert digest(file)==matched['sha256']
        report['frames'][str(i)]=dict(index=i,file=file.name,source_time=cases[i]['source_time'],sha256=matched['sha256'],
                                    sources=matched['sources'],seconds=0.,reused=True)
    update();print(json.dumps(dict(phase='cache_ready',completed=report['completed'],total=len(cases),reused=report['reused'])),flush=True)
    for case in cases:dataset.data(case,960)
    assert json.loads((args.trial/'gnt_finalization/compile.json').read_text())['verified']
    os.environ.setdefault('TORCHINDUCTOR_CACHE_DIR',str(args.trial/'gnt_finalization/inductor_cache'))
    os.environ.setdefault('TRITON_CACHE_DIR',str(args.trial/'gnt_finalization/triton_cache'))
    renderer=None;report['status']='rendering';update()
    for i,case in enumerate(cases):
        if str(i) in report['frames']:continue
        if renderer is None:
            model=load_model(args.trial,checkpoint);enable_compiled_view_attention(model)
            renderer=Renderer(model,optimized=True,chunk=4096,cache_features=True)
        rgb,metadata=renderer.render(dataset,case,960);assert rgb.shape==(540,960,3)
        file=folder/f'frame_{i:04}.png';temp=file.with_name(file.stem+'.partial.png')
        assert cv2.imwrite(str(temp),cv2.cvtColor(np.uint8(np.clip(rgb,0,1)*255+.5),cv2.COLOR_RGB2BGR));temp.replace(file)
        assert all(j%8 for j in metadata['sources'])
        report['frames'][str(i)]=dict(index=i,file=file.name,source_time=case['source_time'],sha256=digest(file),reused=False,**metadata)
        report['latest_frame']=file.name;update()
        print(json.dumps(dict(phase='rendering',index=i,completed=report['completed'],total=len(cases),seconds=metadata['seconds'],estimated_remaining_seconds=report['estimated_remaining_seconds'])),flush=True)
    assert digest(checkpoint)==checkpoint_hash
    for name,expected in inputs.items():assert digest(dataset.root/name)==expected
    report['status']='rendered';update()


if __name__=='__main__':main()
