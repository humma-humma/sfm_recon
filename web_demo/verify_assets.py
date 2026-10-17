"""Asset integrity and independent COLMAP reprojection checks (not browser QA)."""
import json
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation
from export_assets import ROOT, OUT, SOURCES

manifest=json.loads((OUT/'manifest.json').read_text())
for name,item in manifest.items():
    data=np.fromfile(OUT/f'{name}.bin',dtype='<f4').reshape(-1,16)
    assert len(data)==item['count'] and data.nbytes==item['bytes']
    assert np.isfinite(data).all() and np.all((data[:,3:7]>=0)&(data[:,3:7]<=1))
    sh=np.fromfile(OUT/f'{name}.sh.bin',dtype='<f2').reshape(-1,16,4)
    assert len(sh)==item['count'] and sh.nbytes==item['sh_bytes'] and item['sh_degree']==3
    assert np.isfinite(sh).all() and np.all(sh[:,:,3]==0)
    views=item['views']
    for view in views:
        f,u=np.array(view['forward']),np.array(view['up'])
        assert abs(np.linalg.norm(f)-1)<1e-5 and abs(np.linalg.norm(u)-1)<1e-5
        assert abs(f@u)<1e-5 and np.isfinite(view['position']).all()
    folder=ROOT/SOURCES[name]/'sparse/0'
    points=[]
    for line in (folder/'points3D.txt').read_text().splitlines():
        if line and not line.startswith('#'): points.append(list(map(float,line.split()[1:4])))
        if len(points)>=1000: break
    points=np.array(points)
    line=next(x for x in (folder/'images.txt').read_text().splitlines() if x and not x.startswith('#'))
    p=line.split();q=np.array(list(map(float,p[1:5])))
    r=Rotation.from_quat(q[[1,2,3,0]]).as_matrix();t=np.array(list(map(float,p[5:8])))
    original=points@r.T+t
    checkpoint=ROOT/item['checkpoint']
    trans=json.loads((checkpoint.parent.parent/'dataparser_transforms.json').read_text())
    a=np.array(trans['transform']);scale=trans['scale']
    transformed=scale*(points@a[:,:3].T+a[:,3])
    view=views[0];f=np.array(view['forward']);u=np.array(view['up']);right=np.cross(f,u)
    delta=transformed-np.array(view['position'])
    projected=np.stack([delta@right,-delta@u,delta@f],axis=1)/scale
    error=float(np.max(np.abs(projected-original)))
    assert error<1e-4,(name,error)
    print(f'{name}: {len(data):,} finite records; {len(views)} orthonormal cameras; COLMAP camera-space error {error:.2g}')
print('PASS: asset integrity and camera alignment. GPU rendering / interaction not tested.')
