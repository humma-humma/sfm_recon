"""Ensure traversal covers every capture interval with correct camera interpolation."""
import importlib
from pathlib import Path
import numpy as np
import pytest
from scipy.spatial.transform import Rotation


def test_full_route_midpoint_centers_and_rotations(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).parents[1]/'scripts'))
    module = importlib.import_module('gnt_traversal')
    poses=[]
    for i,angle in enumerate([0.,30.,60.,90.]):
        r=Rotation.from_euler('y',angle,degrees=True).as_matrix()
        center=np.array([i*2.,1.,0.])
        poses.append((f'{i*3}.jpg',r,-r@center))
    cases=module.traversal_cases(poses,[100,100,50,50,100,100])
    assert len(cases)==3
    assert [c['bracket'] for c in cases]==[[0,1],[1,2],[2,3]]
    assert [c['source_time'] for c in cases]==[1.5,4.5,7.5]
    for i,case in enumerate(cases):
        r=np.array(case['r']);t=np.array(case['t'])
        np.testing.assert_allclose(-r.T@t,[i*2+1.,1.,0.],atol=1e-12)
        np.testing.assert_allclose(r,Rotation.from_euler('y',15+30*i,degrees=True).as_matrix(),atol=1e-12)
        assert case['kind']=='motion'
    with pytest.raises(ValueError):
        module.traversal_cases(poses[::-1],[100,100,50,50,100,100])


def test_object_route_stays_within_capture_and_does_not_close(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).parents[1]/'scripts'))
    module=importlib.import_module('gnt_traversal')
    poses=[]
    for i in range(4):
        r=Rotation.from_euler('y',i*20,degrees=True).as_matrix()
        poses.append((f'{i:05}.jpg',r,-r@np.array([i,0.,0.])))
    cases=module.object_route_cases(poses,[100,100,50,50,100,100],60)
    assert len(cases)==60
    for j,case in enumerate(cases):
        r=np.array(case['r']);t=np.array(case['t'])
        position=(j+.5)*3/60
        np.testing.assert_allclose(-r.T@t,[position,0,0],atol=1e-12)
        np.testing.assert_allclose(r,Rotation.from_euler('y',position*20,degrees=True).as_matrix(),atol=1e-12)
        assert case['bracket'][1]==case['bracket'][0]+1
    assert cases[-1]['bracket']==[2,3]
    with pytest.raises(ValueError):module.object_route_cases(poses[:1],[],60)


def test_dense_route_uses_time_not_uniform_capture_index(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).parents[1]/'scripts'))
    module=importlib.import_module('gnt_traversal')
    cases=[]
    for i,(time,x) in enumerate([(10.,0.),(11.,2.),(13.,3.)]):
        cases.append(dict(r=np.eye(3).tolist(),t=[-x,0,0],source_time=time,index=i,image=f'{i}.jpg',k=[1]*6))
    dense=module.dense_route_cases(cases,start_seconds=.5,duration=2,fps=30)
    assert len(dense)==60
    for i,case in enumerate(dense):
        time=10.5+i/30
        assert abs(case['source_time']-time)<1e-12
        x=2*(time-10) if time<11 else 2+.5*(time-11)
        np.testing.assert_allclose(-np.array(case['r']).T@np.array(case['t']),[x,0,0],atol=1e-12)
    with pytest.raises(ValueError):module.dense_route_cases(cases,3,1,30)
