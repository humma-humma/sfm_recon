import importlib
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation


def test_object_rollout_is_bounded_and_keeps_dense_evaluation_unseen(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).parents[1]/'scripts'))
    module=importlib.import_module('gnt_rollout_plan')
    poses=[(f'{i}.jpg',Rotation.from_euler('y',i/50).as_matrix(),np.array([i/100,0.,0.])) for i in range(51)]
    plan=module.object_plan(poses,[450,450,480,270,960,540])
    assert len(plan['sparse'])==60 and len(plan['dense'])==180
    assert not set(plan['added'])&set(plan['withheld'])
    fitted=[c['source_time'] for c in plan['sparse']]
    for i in plan['added']+plan['withheld']:
        assert min(abs(t-plan['dense'][i]['source_time']) for t in fitted)>=1e-4
    assert len(plan['added'])+len(plan['withheld']) in (168,169)
    assert 0<=plan['dense_start_seconds']<=25
