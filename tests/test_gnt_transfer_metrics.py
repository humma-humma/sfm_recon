import importlib
from pathlib import Path
import numpy as np
import pytest


def module(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).parents[1]/'scripts'))
    return importlib.import_module('gnt_transfer_metrics')


def test_training_endpoints_are_not_treated_as_unseen(monkeypatch):
    m=module(monkeypatch)
    assert m.bracket_training([1.,2.,4.],2.)==(1,1,0.)
    assert m.bracket_training([1.,2.,4.],2.00001)==(1,1,0.)
    assert m.bracket_training([1.,2.,4.],3.)==(1,2,.5)
    with pytest.raises(ValueError):m.bracket_training([1.,2.,4.],5.)


def test_local_gap_compares_linear_error_not_mean_db(monkeypatch):
    m=module(monkeypatch)
    # Endpoint errors differ by 100x. Their mean error gives ~13 dB, not 20 dB.
    training=[{'mse':.001},{'mse':.1}]
    matched_psnr=-10*np.log10(.0505)
    result=m.local_gap(training,0,1,.5,{'psnr':matched_psnr})
    assert abs(result['psnr_gap'])<1e-10
    worse=m.local_gap(training,0,1,.5,{'psnr':matched_psnr-4})
    assert worse['psnr_gap']==pytest.approx(-4)


def test_density_split_excludes_fitted_views_and_alternates_in_time(monkeypatch):
    m=module(monkeypatch)
    rows=[dict(index=i,source_time=float(i),exact_training_time=i in (1,4)) for i in range(9)]
    added,withheld=m.alternating_unseen(list(reversed(rows)))
    assert added==[0,3,6,8]
    assert withheld==[2,5,7]
    assert not set(added)&set(withheld)
    with pytest.raises(ValueError):m.alternating_unseen([rows[0],rows[0]])
