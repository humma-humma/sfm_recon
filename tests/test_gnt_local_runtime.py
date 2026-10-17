"""Check source exclusion and independent target/source intrinsics in the new loader."""
import importlib
from pathlib import Path
import numpy as np
import pytest


def test_training_excludes_heldout_and_target_and_resizes_cameras(monkeypatch):
    pytest.importorskip('torch')
    monkeypatch.syspath_prepend(str(Path(__file__).parents[1]/'scripts'))
    module=importlib.import_module('gnt_local_runtime')
    dataset=module.Dataset.__new__(module.Dataset)
    dataset.name='synthetic';dataset.k=[100.,100.,50.,30.,100,60]
    dataset.poses=[(f'{i}.jpg',np.eye(3),np.array([i*.001,0.,0.])) for i in range(20)]
    dataset.points=np.array([[0.,0.,2.],[.1,.1,3.],[-.1,-.1,4.]])
    dataset.images={i:np.full((60,100,3),i,dtype=np.uint8) for i in range(20)}
    data,indices=dataset.data(dataset.case(1),50,training=True)
    assert len(indices)==6 and not set(indices).intersection({0,1,8,16})
    np.testing.assert_allclose(data['rgb'].numpy(),1/255.)
    assert tuple(data['src_rgbs'].shape)==(1,6,30,50,3)
    assert data['camera'][0,0]==30 and data['camera'][0,1]==50
    with pytest.raises(AssertionError):dataset.data(dataset.case(8),50,training=True)


def test_closeup_changes_target_but_keeps_native_source_intrinsics(monkeypatch):
    pytest.importorskip('torch')
    monkeypatch.syspath_prepend(str(Path(__file__).parents[1]/'scripts'))
    module=importlib.import_module('gnt_local_runtime')
    dataset=module.Dataset.__new__(module.Dataset)
    dataset.name='synthetic';dataset.k=[100.,110.,50.,30.,100,60]
    dataset.poses=[(f'{i}.jpg',np.eye(3),np.array([i*.001,0.,0.])) for i in range(20)]
    dataset.points=np.array([[0.,0.,2.],[.1,.1,3.],[-.1,-.1,4.]])
    dataset.images={i:np.zeros((60,100,3),dtype=np.uint8) for i in range(20)}
    case=dataset.case(8);case['k']=[170.,187.,50.,30.,100,60]
    data,indices=dataset.data(case,50)
    intrinsic=data['camera'][0,2:18].numpy().reshape(4,4)
    source_intrinsic=data['src_cameras'][0,0,2:18].numpy().reshape(4,4)
    np.testing.assert_allclose(np.diag(intrinsic)[:2],[85,93.5])
    np.testing.assert_allclose(np.diag(source_intrinsic)[:2],[100,110])
    assert not set(indices).intersection({0,8,16})
