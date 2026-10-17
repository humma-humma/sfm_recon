"""Measure unchanged-quality GNT inference optimizations before object adaptation."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import torch
from gnt_local_runtime import Dataset,Renderer,load_model,enable_sdpa
from gnt_traversal import traversal_cases

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--trial',type=Path,required=True)
    parser.add_argument('--native-only',action='store_true',help='Resume saved preview measurements with native checks.')
    args=parser.parse_args();torch.set_num_threads(4)
    checkpoint=args.trial/'gnt_tuning/detail_control_006000.pth'
    digest=hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    output=args.trial/'gnt_finalization';output.mkdir(exist_ok=True)
    report=json.loads((output/'runtime.json').read_text()) if args.native_only else dict(
        checkpoint_sha256=digest,torch=torch.__version__,gpu=torch.cuda.get_device_name(0),
        precision='fp32',rows=[],selection=None)
    assert report['checkpoint_sha256']==digest
    dataset=Dataset('scene');cases=traversal_cases(dataset.poses,dataset.k)
    model=load_model(args.trial,checkpoint)
    reference=Renderer(model)
    probes=[cases[i] for i in (0,120,240)]
    baselines=[]
    for case in ([] if args.native_only else probes):
        reference.render(dataset,case,160)
        samples=[reference.render(dataset,case,160) for _ in range(3)]
        baselines.append(samples[-1][0])
        row=samples[-1][1]|dict(label='reference',case=case['id'],median_seconds=float(np.median([s[1]['seconds'] for s in samples])))
        report['rows'].append(row);print(json.dumps(row),flush=True)
    enable_sdpa(model)
    for chunk in ([] if args.native_only else (4096,6144)):
        torch.cuda.empty_cache()
        renderer=Renderer(model,optimized=True,chunk=chunk,cache_features=True)
        for case,baseline in zip(probes,baselines):
            try:
                renderer.render(dataset,case,160)
                samples=[renderer.render(dataset,case,160) for _ in range(3)]
                rgb,metadata=samples[-1]
                difference=np.abs(rgb-baseline)
                row=metadata|dict(label=f'sdpa_gpu_output_{chunk}',case=case['id'],
                    median_seconds=float(np.median([s[1]['seconds'] for s in samples])),
                    rgb_max_difference=float(difference.max()),
                    rgb_psnr=float(-10*np.log10(max(float(np.mean(difference**2)),1e-16))))
                row['quality_passed']=row['rgb_max_difference']<=1e-5 and row['rgb_psnr']>=80 and row['peak_cuda_mb']<6800
                report['rows'].append(row);print(json.dumps(row),flush=True)
            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache();report['rows'].append(dict(label=f'sdpa_gpu_output_{chunk}',case=case['id'],quality_passed=False,error='CUDA OOM'))
        (output/'runtime.json').write_text(json.dumps(report,indent=2))
    labels=[f'sdpa_gpu_output_{n}' for n in (4096,6144)]
    eligible=[label for label in labels if len([r for r in report['rows'] if r['label']==label and r.get('quality_passed')])==3]
    if eligible:
        best=min(eligible,key=lambda label:np.mean([r['median_seconds'] for r in report['rows'] if r['label']==label]))
        chunk=int(best.rsplit('_',1)[1]);report['selection']=dict(label=best,chunk=chunk)
        # Native checks include indoors, outdoors and stairs. No preview-only promotion.
        native=[]
        original_model=load_model(args.trial,checkpoint)
        ref=Renderer(original_model);fast=Renderer(model,optimized=True,chunk=chunk,cache_features=True)
        for case in probes:
            torch.cuda.empty_cache()
            rgb0,m0=ref.render(dataset,case,960)
            torch.cuda.empty_cache()
            rgb1,m1=fast.render(dataset,case,960)
            difference=np.abs(rgb1-rgb0)
            row=dict(case=case['id'],reference=m0,candidate=m1,
                rgb_max_difference=float(difference.max()),
                rgb_psnr=float(-10*np.log10(max(float(np.mean(difference**2)),1e-16))))
            row['quality_passed']=row['rgb_max_difference']<=1e-5 and row['rgb_psnr']>=80 and m1['peak_cuda_mb']<6800
            native.append(row);print(json.dumps(row),flush=True)
        report['native']=native
        report['selection']['verified']=all(r['quality_passed'] for r in native)
        report['selection']['native_speedup']=float(np.median([r['reference']['seconds']/r['candidate']['seconds'] for r in native]))
    else:report['selection']=dict(label='reference',chunk=4096,verified=True,native_speedup=1.)
    assert hashlib.sha256(checkpoint.read_bytes()).hexdigest()==digest
    report['status']='complete'
    (output/'runtime.json').write_text(json.dumps(report,indent=2))
    print('Runtime benchmark complete',flush=True)

if __name__=='__main__':main()
