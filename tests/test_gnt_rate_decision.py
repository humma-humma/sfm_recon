import importlib
from pathlib import Path


def test_rate_splits_hold_back_all_evaluation_images(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).parents[1]/'scripts'))
    split=importlib.import_module('gnt_rate_sampling').split_frames
    five=split(5);ten=split(10)
    assert len(five['anchors'])==30 and len(ten['anchors'])==60
    assert five['held_tail']==list(range(175,180)) and ten['held_tail']==[178,179]
    common=set(five['withheld'])&set(ten['withheld'])
    assert len(common)==116
    assert not common&set(five['anchors']) and not common&set(ten['anchors'])
    assert max(common)<174


def test_15fps_comparison_excludes_both_sets_of_anchors(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).parents[1]/'scripts'))
    split=importlib.import_module('gnt_rate_sampling').split_frames
    ten=split(10);fifteen=split(15)
    assert len(fifteen['anchors'])==90 and fifteen['held_tail']==[179]
    common=set(ten['withheld'])&set(fifteen['withheld'])
    assert len(common)==59 and max(common)==175
    assert all(i%2==1 and i%3!=0 for i in common)


def test_splat_fillers_keep_anchor_times_and_fill_the_end(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).parents[1]/'scripts'))
    module=importlib.import_module('gnt_rate_sampling')
    for rate,count in [(5,30),(10,60),(15,90)]:
        sources=module.interleaved_sources(rate)
        assert len(sources)==180 and sources.count('gnt')==count
        assert sources[0]=='gnt' and sources[-1]=='splat'
        assert [i for i,s in enumerate(sources) if s=='gnt']==module.split_frames(rate)['anchors']
    assert module.interleaved_sources(15)==['gnt','splat']*90
