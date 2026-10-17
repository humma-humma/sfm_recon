"""Protect RGB exclusion and off-center camera visibility during adaptation."""
import importlib
from pathlib import Path
import numpy as np
import pytest


@pytest.fixture
def tuning(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).parents[1]/'scripts'))
    return importlib.import_module('tune_gnt')


@pytest.mark.parametrize('mode', ['nearest', 'overlap'])
def test_sources_exclude_heldout_and_training_target(tuning, mode):
    r = np.eye(3)
    poses = [(str(i), r, np.array([i*.01,0.,0.])) for i in range(20)]
    points = np.array([[0.,0.,2.],[.1,.2,3.]])
    selected = tuning.sources(r,poses[1][2],poses,points,[100,100,50,50,100,100],6,mode,[1.]*20,exclude=1)
    assert len(selected)==6
    assert not set(selected).intersection({0,1,8,16})


def test_visibility_uses_supplied_principal_point(tuning):
    points = np.array([[-.7,0.,1.],[.7,0.,1.],[0.,0.,-1.]])
    # Off-center intrinsics make the left point visible and right point invisible.
    result = tuning.visible(points,np.eye(3),np.zeros(3),[100,100,80,50,100,100])
    np.testing.assert_array_equal(result,[True,False,False])


def test_balanced_capture_groups_cover_every_eligible_view_once(tuning):
    groups = tuning.balanced_targets(276)
    assert len(groups)==6 and all(groups)
    flattened = [i for group in groups for i in group]
    assert sorted(flattened)==[i for i in range(276) if i%8 != 0]
    assert len(flattened)==len(set(flattened))
    assert max(map(len,groups))-min(map(len,groups))<=1
