"""Generate resumable dense object teachers after the existing sparse route render."""
import argparse
import json
import os
from pathlib import Path
import shutil
import cv2
import numpy as np
import torch
from gnt_local_runtime import Dataset,Renderer,load_model,enable_compiled_view_attention
from gnt_rollout_plan import object_plan
from gnt_rollout_io import digest,save


def preflight(root,trial,name):
    dataset=Dataset(name);plan=object_plan(dataset.poses,dataset.k)
    checkpoint=trial/'gnt_finalization'/name/'balanced_006000.pth'
    trained=json.loads((checkpoint.parent/'results.json').read_text())
    assert trained['status']=='complete' and digest(checkpoint)==trained['checkpoints']['6000']['sha256']
    assert dataset.k[4:]==(960,540) or list(dataset.k[4:])==[960,540]
    for case in plan['sparse']+plan['dense']:dataset.data(case,960)
    plan.update(dataset=name,checkpoint_sha256=digest(checkpoint),width=960,height=540,
                source_photo_exclusion='interval-8',precision='fp32',chunk=4096)
    path=root/'balanced_rollout'/name/'plan.json'
    if path.exists():assert json.loads(path.read_text())==plan,'Plan changed'
    save(path,plan)
    print(json.dumps(dict(dataset=name,status='preflight_passed',sparse=60,dense=180,
                         added=len(plan['added']),withheld=len(plan['withheld']),start_seconds=plan['dense_start_seconds'])),flush=True)
    return dataset,plan,checkpoint


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--trial',type=Path,required=True)
    parser.add_argument('--dataset',choices=['box','boot','milk'],required=True)
    parser.add_argument('--preflight',action='store_true');args=parser.parse_args();torch.set_num_threads(4)
    dataset,plan,checkpoint=preflight(args.root,args.trial,args.dataset)
    if args.preflight:return
    sparse_folder=args.root/'assets/videos'/args.dataset
    sparse=json.loads((sparse_folder/'progress.json').read_text())
    assert len(sparse['frames'])==60 and sparse['checkpoint_sha256']==plan['checkpoint_sha256']
    assert all(all(i%8 for i in r['sources']) for r in sparse['frames'])
    folder=args.root/'assets/balanced_rollout'/args.dataset/'teachers';folder.mkdir(parents=True,exist_ok=True)
    path=folder/'progress.json'
    report=json.loads(path.read_text()) if path.exists() else dict(status='rendering',checkpoint_sha256=plan['checkpoint_sha256'],frames=[],cases=plan['dense'])
    assert report['checkpoint_sha256']==plan['checkpoint_sha256'] and report['cases']==plan['dense']
    os.environ.setdefault('TORCHINDUCTOR_CACHE_DIR',str(args.trial/'gnt_finalization/inductor_cache'))
    os.environ.setdefault('TRITON_CACHE_DIR',str(args.trial/'gnt_finalization/triton_cache'))
    assert json.loads((args.trial/'gnt_finalization/compile.json').read_text())['verified']
    renderer=None
    for i,case in enumerate(plan['dense']):
        file=folder/f'frame_{i:04}.png'
        if i<len(report['frames']):assert digest(file)==report['frames'][i]['sha256'];continue
        match=next((j for j,c in enumerate(plan['sparse']) if abs(c['source_time']-case['source_time'])<1e-4
                    and np.allclose(c['r'],case['r'],atol=1e-7,rtol=0) and np.allclose(c['t'],case['t'],atol=1e-7,rtol=0)),None)
        if match is not None:
            previous=sparse['frames'][match];source=sparse_folder/previous['file']
            assert digest(source)==previous['sha256'];shutil.copy2(source,file)
            metadata=dict(sources=previous['sources'],seconds=0.,reused=True)
        else:
            if renderer is None:
                model=load_model(args.trial,checkpoint);enable_compiled_view_attention(model)
                renderer=Renderer(model,optimized=True,chunk=4096,cache_features=True)
            rgb,metadata=renderer.render(dataset,case,960)
            assert rgb.shape==(540,960,3)
            temp=file.with_name(file.stem+'.partial.png')
            assert cv2.imwrite(str(temp),cv2.cvtColor(np.uint8(np.clip(rgb,0,1)*255+.5),cv2.COLOR_RGB2BGR));temp.replace(file)
            metadata['reused']=False
        assert all(j%8 for j in metadata['sources'])
        report['frames'].append(dict(index=i,file=file.name,sha256=digest(file),**metadata))
        save(path,report)
        print(json.dumps(dict(dataset=args.dataset,phase='dense_teachers',completed=i+1,total=180,seconds=metadata['seconds'])),flush=True)
    assert digest(checkpoint)==plan['checkpoint_sha256']
    report['status']='complete';save(path,report)


if __name__=='__main__':main()
