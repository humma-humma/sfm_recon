"""Resume native GNT frames on a bounded captured-camera object route."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import cv2
import numpy as np
import torch
from gnt_local_runtime import Dataset, Renderer, load_model, enable_compiled_view_attention
from gnt_traversal import object_route_cases


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--trial',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--dataset',choices=['box','boot','milk'],required=True)
    args=parser.parse_args();torch.set_num_threads(4)
    folder=args.output/args.dataset;folder.mkdir(parents=True,exist_ok=True)
    checkpoint=args.trial/'gnt_finalization'/args.dataset/'balanced_006000.pth'
    digest=hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    trained=json.loads((checkpoint.parent/'results.json').read_text())
    assert trained['status']=='complete' and digest==trained['checkpoints']['6000']['sha256']
    dataset=Dataset(args.dataset);cases=object_route_cases(dataset.poses,dataset.k)
    # Validate every camera before beginning the expensive render.
    for case in cases:dataset.data(case,960)
    path=folder/'progress.json'
    report=json.loads(path.read_text()) if path.exists() else dict(
        dataset=args.dataset,status='rendering',checkpoint_sha256=digest,width=960,height=540,
        independent_frames=60,source_fps='15/8',duration_seconds=32.,cases=cases,frames=[],
        route='Captured camera order, linear centers and SLERP rotations; no synthetic closure',
        backend='compiled_view_attention',chunk=4096)
    assert report['checkpoint_sha256']==digest and report['cases']==cases
    def save():
        temporary=path.with_suffix('.partial.json')
        temporary.write_text(json.dumps(report,indent=2));temporary.replace(path)
    os.environ.setdefault('TORCHINDUCTOR_CACHE_DIR',str(args.trial/'gnt_finalization/inductor_cache'))
    os.environ.setdefault('TRITON_CACHE_DIR',str(args.trial/'gnt_finalization/triton_cache'))
    assert json.loads((args.trial/'gnt_finalization/compile.json').read_text())['verified']
    model=load_model(args.trial,checkpoint);enable_compiled_view_attention(model)
    renderer=Renderer(model,optimized=True,chunk=4096,cache_features=True)
    for i,case in enumerate(cases):
        frame=folder/f'frame_{i:04}.png'
        previous=next((row for row in report['frames'] if row['index']==i),None)
        if previous:
            assert frame.exists() and hashlib.sha256(frame.read_bytes()).hexdigest()==previous['sha256']
            continue
        rgb,metadata=renderer.render(dataset,case,960)
        temporary=folder/f'frame_{i:04}.partial.png'
        assert cv2.imwrite(str(temporary),cv2.cvtColor(np.uint8(np.clip(rgb,0,1)*255+.5),cv2.COLOR_RGB2BGR))
        temporary.replace(frame)
        report['frames'].append(dict(index=i,file=frame.name,sha256=hashlib.sha256(frame.read_bytes()).hexdigest(),**metadata))
        save();print(json.dumps(dict(dataset=args.dataset,completed=len(report['frames']),total=60,seconds=metadata['seconds'])),flush=True)
    assert hashlib.sha256(checkpoint.read_bytes()).hexdigest()==digest
    report['status']='rendered';save()


if __name__=='__main__':main()
