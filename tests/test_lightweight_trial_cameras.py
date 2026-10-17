"""Check the coordinate/split adapters independently of research networks."""
import importlib.util
from pathlib import Path
import numpy as np

spec = importlib.util.spec_from_file_location('gnt_trial', Path(__file__).parents[1]/'scripts/run_gnt_trial.py')
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def test_camera_vector_inverts_colmap_pose():
    angle = .37
    r = np.array([[np.cos(angle),0,np.sin(angle)],[0,1,0],[-np.sin(angle),0,np.cos(angle)]])
    t = np.array([.1,-.4,.7])
    vector = module.camera_vector(r,t,[800,810,480,270,960,540])
    assert vector.shape == (34,)
    np.testing.assert_equal(vector[:2],[540,960])
    camera = vector[18:].reshape(4,4)
    world = np.array([.2,.3,2.])
    recovered = camera@np.r_[r@world+t,1]
    np.testing.assert_allclose(recovered[:3],world,atol=1e-6)
    intrinsic = vector[2:18].reshape(4,4)
    np.testing.assert_allclose(intrinsic[:3,:3],[[800,0,480],[0,810,270],[0,0,1]])


def test_object_cases_keep_anchor_and_three_motion_positions():
    cases = [dict(id=f'eval_{i}_z{z}',index=i,kind='heldout') for i in (0,24,48) for z in (1,1.7)]
    cases += [dict(id=f'motion_{i:02}',index=24,kind='motion') for i in range(9)]
    actual = module.trial_cases({'cases':cases},'boot')
    assert [c['id'] for c in actual] == ['eval_24_z1','eval_24_z1.7','motion_00','motion_04','motion_08']
    assert len(module.trial_cases({'cases':cases},'scene')) == 9
