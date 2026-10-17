import importlib
from pathlib import Path
import numpy as np


def test_scene15_uniform_timing_and_exact_camera_reuse(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).parents[1]/'scripts'))
    module=importlib.import_module('gnt_scene15_plan')
    route=[dict(r=np.eye(3).tolist(),t=[0.,0.,z],k=[450,450,480,270,960,540],source_time=time,index=i,image='test.jpg') for i,(time,z) in enumerate([(1.,0.),(148.,1.)])]
    cases=module.scene15_cases(route)
    assert len(cases)==2198
    np.testing.assert_allclose(np.diff([c['source_time'] for c in cases]),1/15,atol=1e-12)
    assert abs(cases[-1]['route_time']-146.4666666667)<1e-8
    candidate=dict(case=cases[1],path='cached')
    assert module.matching_camera(cases[1],[candidate]) is candidate
    wrong=dict(cases[1],t=[0.,0.,99.])
    assert module.matching_camera(wrong,[candidate]) is None
    wrong=dict(cases[1],k=[900,900,480,270,960,540])
    assert module.matching_camera(wrong,[candidate]) is None
