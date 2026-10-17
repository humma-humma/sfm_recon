import importlib.util
from pathlib import Path
import numpy as np

spec=importlib.util.spec_from_file_location('diagnosis',Path(__file__).parents[1]/'scripts/diagnose_scene_reconstruction.py')
module=importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)

def test_known_camera_translation_projects_plane():
    depth=np.full((30,40),2,dtype=np.float32)
    identity=(np.eye(3),np.zeros(3))
    moved=(np.eye(3),np.array([.1,0,0]))
    stats,xy,ij,agree,_=module.cross_view(depth,identity,moved,depth,(20,20,20,15))
    assert stats['agreement_fraction']==1
    np.testing.assert_array_equal(ij-xy,np.tile([1,0],(len(xy),1)))

def test_occlusion_is_distinguished_from_free_space():
    identity=(np.eye(3),np.zeros(3))
    near=np.full((30,40),1,dtype=np.float32)
    far=np.full((30,40),2,dtype=np.float32)
    stats,*_=module.cross_view(far,identity,identity,near,(20,20,20,15))
    assert stats['occluded_fraction']==1
    assert stats['free_space_conflict_fraction']==0
    stats,*_=module.cross_view(near,identity,identity,far,(20,20,20,15))
    assert stats['free_space_conflict_fraction']==1
