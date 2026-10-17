import importlib
from pathlib import Path
from collections import Counter
import pytest


def test_balanced_sampling_preserves_coverage_and_exact_quotas(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).parents[1]/'scripts'))
    schedule=importlib.import_module('gnt_balanced_sampling').balanced_schedule
    original=list(range(275));added=list(range(275,360))
    values=schedule(original,added,2000);counts=Counter(values)
    assert values==schedule(original,added,2000)
    assert values!=schedule(original,added,2000,seed=43)
    assert sum(counts[i] for i in original)==1800
    assert sum(counts[i] for i in added)==200
    assert {counts[i] for i in original}=={6,7}
    assert {counts[i] for i in added}=={2,3}
    assert all(sum(i in added for i in values[j:j+10])==1 for j in range(0,2000,10))
    with pytest.raises(ValueError):schedule(original,added,1999)
    with pytest.raises(ValueError):schedule([0],[0],10)
