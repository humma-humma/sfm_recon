import importlib
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation


def test_midpoint_preserves_route_time_and_slerp(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).parents[1]/'scripts'))
    module=importlib.import_module('gnt_adaptive')
    a=dict(r=np.eye(3).tolist(),t=[0,0,0],source_time=10,index=0,image='a.jpg',k=[1]*6)
    r=Rotation.from_euler('y',60,degrees=True).as_matrix()
    b=a|dict(r=r.tolist(),t=(-r@np.array([2.,0,0])).tolist(),source_time=12)
    c=module.midpoint(a,b)
    assert c['source_time']==11
    np.testing.assert_allclose(-np.array(c['r']).T@np.array(c['t']),[1,0,0],atol=1e-12)
    assert abs(module.rotation_degrees(a,c)-30)<1e-10
    assert module.needs_refinement(4,0,1,0)
    assert module.needs_refinement(0,.04,1,0)
    assert module.needs_refinement(0,0,.8,0)
    assert not module.needs_refinement(0,0,1,0)


def test_static_flow_proxy_does_not_invent_motion(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).parents[1]/'scripts'))
    module=importlib.import_module('gnt_adaptive')
    image=np.random.default_rng(42).integers(0,256,(90,160,3),dtype=np.uint8)
    prediction,valid,motion=module.motion_proxy(image,image)
    assert valid.mean()>.95 and motion<.1
    assert np.mean(np.abs(prediction[valid]-image[valid]/255))<.005
