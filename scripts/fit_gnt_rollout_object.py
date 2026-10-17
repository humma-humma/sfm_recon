"""Build object teacher-only splits, fit 10k, and refine with 90/10 sampling."""
import argparse
import json
from pathlib import Path
import shutil
import subprocess
import sys
import numpy as np
from prepare_gnt_splat import archive,pose_line
from run_gnt_trial import DATA,ROOT
from diagnose_gnt_transfer import digest
from sfm_reconstruction.stage3_splat_temporal_diagnostics import _read_colmap


def prepare(root,name):
    base=root/'balanced_rollout'/name
    plan=json.loads((base/'plan.json').read_text())
    sparse_folder=root/'assets/videos'/name
    sparse=json.loads((sparse_folder/'progress.json').read_text())
    dense_folder=root/'assets/balanced_rollout'/name/'teachers'
    dense=json.loads((dense_folder/'progress.json').read_text())
    assert len(sparse['frames'])==60 and dense['status']=='complete' and len(dense['frames'])==180
    assert sparse['checkpoint_sha256']==dense['checkpoint_sha256']==plan['checkpoint_sha256']
    colmap=ROOT/DATA[name];k,poses,_=_read_colmap(colmap/'sparse/0')
    rows=[]
    for i,(case,frame) in enumerate(zip(plan['sparse'],sparse['frames'])):
        assert all(j%8 for j in frame['sources'])
        rows.append(dict(name=f'gnt_{i:04}.png',r=case['r'],t=case['t'],train=True,kind='gnt',source_time=case['source_time'],
                         source=str(sparse_folder/frame['file']),sha256=frame['sha256']))
    for i,(original,r,t) in enumerate(poses):
        if i%8:continue
        source=colmap/'images'/original
        rows.append(dict(name=f'photo_{i:04}'+Path(original).suffix,r=r.tolist(),t=t.tolist(),train=False,kind='photo',source=str(source),sha256=digest(source)))
    added=[]
    for i in plan['added']:
        case=plan['dense'][i];frame=dense['frames'][i]
        assert frame['index']==i and all(j%8 for j in frame['sources'])
        added.append(dict(name=f'dense_{i:04}.png',r=case['r'],t=case['t'],train=True,kind='added',source_time=case['source_time'],
                          source=str(dense_folder/frame['file']),sha256=frame['sha256']))
    for arm,subset in [('teacher',rows),('dense',rows+added)]:
        folder=base/'data'/arm;images=folder/'images';sparse_out=folder/'sparse/0'
        images.mkdir(parents=True,exist_ok=True);sparse_out.mkdir(parents=True,exist_ok=True)
        for row in subset:assert archive(Path(row['source']),images/row['name'])==row['sha256']
        (sparse_out/'images.txt').write_text(''.join(pose_line(i+1,np.array(r['r']),r['t'],r['name']) for i,r in enumerate(subset)))
        for filename in ('cameras.txt','points3D.txt'):shutil.copy2(colmap/'sparse/0'/filename,sparse_out/filename)
        for split in ('train','test','val'):
            names=[r['name'] for r in subset if r['train']==(split=='train')]
            (folder/f'{split}_list.txt').write_text('\n'.join(names)+'\n')
        _,loaded,_=_read_colmap(sparse_out)
        assert len(loaded)==len(subset)
        for (n,r,t),row in zip(loaded,subset):
            assert n==row['name'];np.testing.assert_allclose(r,row['r'],atol=1e-12);np.testing.assert_allclose(t,row['t'],atol=1e-12)
    manifest=dict(teacher_frames=60,original_train=sum(i%8!=0 for i in range(len(poses))),heldout=sum(i%8==0 for i in range(len(poses))),
                  teacher_checkpoint_sha256=plan['checkpoint_sha256'],rows=rows,added_rows=added,
                  note='Original photos are evaluation-only; geometry/poses predate split. Alternating unseen dense views withheld.')
    (base/'data_manifest.json').write_text(json.dumps(manifest,indent=2))
    return base,plan


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--dataset',choices=['box','boot','milk'],required=True);args=parser.parse_args()
    base,plan=prepare(args.root,args.dataset)
    subprocess.run([sys.executable,str(ROOT/'scripts/train_gnt_splat.py'),'--root',str(args.root),'--base',str(base),'--dataset',args.dataset,'--arm','teacher'],check=True)
    from test_gnt_dense_supervision import fit
    source=base/'models/teacher/splatfacto/run_10000/nerfstudio_models/step-000009999.ckpt'
    refinement=dict(source_sha256=digest(source),training_views=dict(control=60,dense=60+len(plan['added'])))
    for arm in ('control','dense','balanced90'):
        checkpoint=fit(args.root,base,source,refinement,arm)
    (base/'completed_fit.json').write_text(json.dumps(dict(status='complete',checkpoint=str(checkpoint),checkpoint_sha256=digest(checkpoint)),indent=2))


if __name__=='__main__':main()
