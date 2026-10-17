"""Render lossless, pose-matched splat views for the full 15fps GNT comparison."""
import argparse
import json
from pathlib import Path
import time
import cv2
import numpy as np
import torch
from export_gnt_balanced_rollout import setup, make_renderer
from gnt_rollout_io import digest, save
from gnt_traversal import dense_route_cases


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--root',type=Path,required=True);args=parser.parse_args()
    torch.set_num_threads(4)
    folder=args.root/'assets/scene15_splat';folder.mkdir(parents=True,exist_ok=True)
    native=args.root/'assets/scene15';native_plan=json.loads((native/'plan.json').read_text())
    plan,dense,targets,sparse,sparse_targets,route,duration,k,poses,checkpoints=setup(args.root,'scene')
    checkpoint=checkpoints['balanced90'];signature=digest(checkpoint)
    cases=dense_route_cases(route,start_seconds=0,duration=native_plan['duration_seconds'],fps=30,hold_end=True)
    assert len(cases)==2*native_plan['total']==4396
    for a,b in zip(cases[::2],native_plan['cases']):
        assert abs(a['source_time']-b['source_time'])<1e-8
        for key in ('r','t','k'):np.testing.assert_allclose(a[key],b[key],atol=1e-8,rtol=0)
    settings=dict(native_plan_sha256=digest(native/'plan.json'),checkpoint_sha256=signature,total=len(cases),fps=30,cases=cases)
    settings_path=folder/'splat_plan.json'
    if settings_path.exists():assert json.loads(settings_path.read_text())==settings
    save(settings_path,settings)
    path=folder/'render_progress.json'
    report=json.loads(path.read_text()) if path.exists() else dict(status='rendering',plan_sha256=digest(settings_path),frames={})
    assert report['plan_sha256']==digest(settings_path)
    for key,row in report['frames'].items():assert digest(folder/row['file'])==row['sha256']
    started=time.perf_counter();render,pipeline=make_renderer(checkpoint,k,dict(sparse=sparse,dense=dense,added=plan['added']))
    setup_seconds=time.perf_counter()-started
    for i,case in enumerate(cases):
        if str(i) in report['frames']:continue
        image,seconds=render(case);file=folder/f'splat_{i:04}.png';temp=file.with_name(file.stem+'.partial.png')
        assert cv2.imwrite(str(temp),cv2.cvtColor(image,cv2.COLOR_RGB2BGR));temp.replace(file)
        report['frames'][str(i)]=dict(index=i,file=file.name,sha256=digest(file),seconds=seconds)
        if (i+1)%30==0 or i==len(cases)-1:
            report.update(completed=len(report['frames']),total=len(cases),render_seconds=sum(r['seconds'] for r in report['frames'].values()),current_run_wall_seconds=time.perf_counter()-started,setup_seconds=setup_seconds)
            save(path,report);print(json.dumps({key:report[key] for key in ('status','completed','total','render_seconds','current_run_wall_seconds')}),flush=True)
    assert digest(checkpoint)==signature
    report.update(status='complete',current_run_wall_seconds=time.perf_counter()-started);save(path,report)


if __name__=='__main__':main()
