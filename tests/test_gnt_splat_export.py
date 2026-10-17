import importlib
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation
from sfm_reconstruction.stage3_splat_temporal_diagnostics import _read_colmap


def test_teacher_colmap_pose_roundtrip(tmp_path,monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).parents[1]/'scripts'))
    module=importlib.import_module('prepare_gnt_splat')
    r=Rotation.from_euler('xyz',[.2,-.7,.1]).as_matrix();t=np.array([2.,-3.,.5])
    (tmp_path/'images.txt').write_text(module.pose_line(1,r,t,'gnt_0000.png'))
    (tmp_path/'cameras.txt').write_text('1 PINHOLE 960 540 455 456 480 270\n')
    (tmp_path/'points3D.txt').write_text('1 1 2 3 255 0 0 0\n')
    _,poses,_=_read_colmap(tmp_path)
    assert poses[0][0]=='gnt_0000.png'
    np.testing.assert_allclose(poses[0][1],r,atol=1e-12)
    np.testing.assert_allclose(poses[0][2],t,atol=1e-12)


def test_normalized_splat_view_preserves_camera_projection(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).parents[1]/'scripts'))
    module=importlib.import_module('prepare_gnt_splat')
    r=Rotation.from_euler('xyz',[.2,.5,-.1]).as_matrix();t=np.array([.1,.2,2.])
    tr=np.column_stack([Rotation.from_euler('xyz',[-.1,.3,.4]).as_matrix(),[1.,2.,3.]])
    points=np.random.default_rng(1).normal(size=(20,3));scale=.05
    normalized=(points@tr[:,:3].T+tr[:,3])*scale
    view=module.normalized_view(r,t,tr,scale)
    np.testing.assert_allclose(normalized@view[:3,:3].T+view[:3,3],(points@r.T+t)*scale,atol=1e-12)
