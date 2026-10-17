"""Create matched photo-only / GNT-augmented COLMAP training sets."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import cv2
import numpy as np
from scipy.spatial.transform import Rotation
from sfm_reconstruction.stage3_splat_temporal_diagnostics import _read_colmap
from gnt_traversal import traversal_cases

ROOT=Path(__file__).resolve().parents[1]
COLMAP=ROOT/'outputs/stage3_gaussian_colmap_stride16_half_stair_rebalanced'


def normalized_view(r,t,transform,scale):
    rotation=np.array(r)@transform[:,:3].T
    view=np.eye(4);view[:3,:3]=rotation
    view[:3,3]=scale*(np.array(t)-rotation@transform[:,3])
    return view


def pose_line(index,r,t,name):
    x,y,z,w=Rotation.from_matrix(r).as_quat()
    return f'{index} '+ ' '.join(f'{v:.17g}' for v in (w,x,y,z,*t))+f' 1 {name}\n\n'


def archive(source,target):
    digest=hashlib.sha256(source.read_bytes()).hexdigest()
    if not target.exists():
        try:os.link(source,target)
        except OSError:shutil.copy2(source,target)
    assert hashlib.sha256(target.read_bytes()).hexdigest()==digest
    image=cv2.imread(str(target));assert image is not None and image.shape[:2]==(540,960)
    return digest


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--destination',type=Path,required=True)
    parser.add_argument('--original',type=Path,required=True);args=parser.parse_args()
    base=args.destination/'gnt_splat_scene';base.mkdir(parents=True,exist_ok=True)
    k,poses,points=_read_colmap(COLMAP/'sparse/0')
    cases=traversal_cases(poses,k)
    source=json.loads((args.original/'manifest.json').read_text())
    assert len(cases)==len(source['frames'])==275
    rows=[]
    for i,(name,r,t) in enumerate(poses):
        rows.append(dict(name=f'photo_{i:04}.jpg',r=r.tolist(),t=t.tolist(),
            source=str(COLMAP/'images'/name),kind='photo',original_index=i,
            train=bool(i%8)))
    for case,frame in zip(cases,source['frames']):
        assert case['id']==frame['id'] and all(i%8 for i in frame['sources'])
        assert hashlib.sha256((args.original/frame['frame']).read_bytes()).hexdigest()==frame['sha256']
        rows.append(dict(name=f'gnt_{frame["index"]:04}.png',r=case['r'],t=case['t'],
            source=str(args.original/frame['frame']),kind='gnt',original_index=None,train=True,
            source_time=case['source_time'],sources=frame['sources']))
    for arm in ('photos','mixed','teacher'):
        folder=base/'data'/arm;images=folder/'images';sparse=folder/'sparse/0'
        images.mkdir(parents=True,exist_ok=True);sparse.mkdir(parents=True,exist_ok=True)
        subset=[row for row in rows if arm!='photos' or row['kind']=='photo']
        if arm=='teacher':
            subset=[row for row in subset if row['kind']=='gnt' or not row['train']]
        lines=[]
        for i,row in enumerate(subset):
            row['sha256']=archive(Path(row['source']),images/row['name'])
            lines.append(pose_line(i+1,np.array(row['r']),row['t'],row['name']))
        (sparse/'images.txt').write_text('# Explicit OpenCV world-to-camera poses\n'+''.join(lines),encoding='ascii')
        shutil.copy2(COLMAP/'sparse/0/cameras.txt',sparse/'cameras.txt')
        shutil.copy2(COLMAP/'sparse/0/points3D.txt',sparse/'points3D.txt')
        train=[row['name'] for row in subset if row['train']]
        test=[row['name'] for row in subset if not row['train']]
        assert not set(train)&set(test)
        for split,names in [('train',train),('test',test),('val',test)]:
            (folder/f'{split}_list.txt').write_text('\n'.join(names)+'\n')
        rk,rp,rpoints=_read_colmap(sparse)
        np.testing.assert_allclose(rk,k);np.testing.assert_allclose(rpoints,points)
        for (_,r,t),row in zip(rp,subset):
            np.testing.assert_allclose(r,row['r'],atol=1e-12)
            np.testing.assert_allclose(t,row['t'],atol=1e-12)
        print(json.dumps(dict(arm=arm,training=len(train),heldout=len(test),teacher=sum(row['kind']=='gnt' for row in subset))),flush=True)
    report=dict(status='prepared',teacher_checkpoint_sha256=source['checkpoint_sha256'],
        teacher_frames=275,original_photos=len(poses),original_train=sum(bool(i%8) for i in range(len(poses))),
        heldout=sum(i%8==0 for i in range(len(poses))),seed_points=len(points),rows=rows,
        dense_turn_excluded=True,note='Original interval-8 photographs excluded from training in all arms. Teacher arm trains only 275 sparse GNT frames. Dense GNT frames are never training targets. Teacher consistency is not photographic ground truth; original camera estimation and seed cloud predate splits.')
    (base/'data_manifest.json').write_text(json.dumps(report,indent=2))


if __name__=='__main__':main()
