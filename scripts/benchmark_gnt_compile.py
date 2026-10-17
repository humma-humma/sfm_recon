"""Bounded compiler-fusion trial for the GNT view-attention bottleneck."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import signal
from time import perf_counter
import cv2
import numpy as np
import torch
from gnt_local_runtime import Dataset,Renderer,load_model,enable_compiled_view_attention
from gnt_traversal import traversal_cases

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--trial',type=Path,required=True)
    parser.add_argument('--chunk',type=int,choices=[4096,8192],default=4096)
    args=parser.parse_args();torch.set_num_threads(4)
    from torch._inductor import config as compiler_config
    compiler_config.compile_threads=2
    checkpoint=args.trial/'gnt_tuning/detail_control_006000.pth'
    digest=hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    output=args.trial/'gnt_finalization';output.mkdir(exist_ok=True)
    os.environ.setdefault('TORCHINDUCTOR_CACHE_DIR',str(output/'inductor_cache'))
    os.environ.setdefault('TRITON_CACHE_DIR',str(output/'triton_cache'))
    report=dict(status='running',checkpoint_sha256=digest,precision='fp32',chunk=args.chunk,rows=[],verified=False,
        note='Compile view attention only; retain original ray attention and checkpoint format. First-call compile cost recorded separately. Promotion requires finite outputs, native pixel error <=1 level, float preview max error <=1e-4 and PSNR >=80 dB, and a native median speedup above 1.05. These engineering tolerances reflect 8-bit delivery; they differ from the stricter initial SDPA float gate.')
    path=output/f'compile_{args.chunk}.json'
    dataset=Dataset('scene');cases=traversal_cases(dataset.poses,dataset.k)
    probes=[cases[i] for i in (0,120,240)]
    model=load_model(args.trial,checkpoint);reference=Renderer(model)
    baselines=[reference.render(dataset,c,160)[0] for c in probes]
    enable_compiled_view_attention(model)
    fast=Renderer(model,optimized=True,chunk=args.chunk,cache_features=True)
    def timeout(*_):raise TimeoutError('Compiler trial exceeded its fifteen-minute bound')
    signal.signal(signal.SIGALRM,timeout);signal.alarm(900)
    try:
        start=perf_counter()
        fast.render(dataset,probes[0],160)
        report['first_call_seconds']=perf_counter()-start
        print(json.dumps(dict(first_call_seconds=report['first_call_seconds'])),flush=True)
        for case,baseline in zip(probes,baselines):
            rgb,metadata=fast.render(dataset,case,160)
            delta=np.abs(rgb-baseline)
            row=metadata|dict(case=case['id'],width=160,rgb_max_difference=float(delta.max()),
                rgb_psnr=float(-10*np.log10(max(float(np.mean(delta**2)),1e-16))))
            row['quality_passed']=row['rgb_max_difference']<=1e-4 and row['rgb_psnr']>=80
            report['rows'].append(row);print(json.dumps(row),flush=True)
        if all(r['quality_passed'] for r in report['rows']):
            original=args.trial/'gnt_tuning/renders/scene/traversal_control6000'
            runtime=json.loads((output/'runtime.json').read_text())
            native=[]
            for case in probes:
                torch.cuda.empty_cache()
                rgb,metadata=fast.render(dataset,case,960)
                target=original/f'{case["id"]}_960.png'
                reference_image=cv2.cvtColor(cv2.imread(str(target)),cv2.COLOR_BGR2RGB)
                image=np.uint8(np.clip(rgb,0,1)*255+.5)
                difference=np.abs(image.astype(int)-reference_image.astype(int))
                baseline=next(r['reference']['seconds'] for r in runtime['native'] if r['case']==case['id'])
                row=metadata|dict(case=case['id'],rgb_pixel_max_difference=int(difference.max()),
                    changed_pixel_fraction=float(np.mean(difference!=0)),speedup=baseline/metadata['seconds'])
                row['quality_passed']=row['rgb_pixel_max_difference']<=1 and row['peak_cuda_mb']<6800
                native.append(row);print(json.dumps(row),flush=True)
                dest=output/'compile_renders';dest.mkdir(exist_ok=True)
                assert cv2.imwrite(str(dest/f'{case["id"]}_960.png'),cv2.cvtColor(image,cv2.COLOR_RGB2BGR))
            report['native']=native
            report['native_speedup']=float(np.median([r['speedup'] for r in native]))
            report['verified']=all(r['quality_passed'] for r in native) and report['native_speedup']>1.05
        report['status']='complete'
    except Exception as error:
        report.update(status='rejected',error=f'{type(error).__name__}: {error}',verified=False)
        print(report['error'],flush=True)
    finally:signal.alarm(0)
    assert hashlib.sha256(checkpoint.read_bytes()).hexdigest()==digest
    path.write_text(json.dumps(report,indent=2))
    selected=output/'compile.json'
    previous=json.loads(selected.read_text()) if selected.exists() else {}
    if not previous or (report['verified'] and (not previous.get('verified') or report['native_speedup']>previous['native_speedup']*1.02)):
        selected.write_text(json.dumps(report,indent=2))
    print('Compiler trial complete',flush=True)

if __name__=='__main__':main()
