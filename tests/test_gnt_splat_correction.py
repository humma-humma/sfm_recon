import importlib
from pathlib import Path

import cv2
import numpy as np


def test_anchor_residual_warp_direction_and_visibility(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).parents[1]/'scripts'))
    module=importlib.import_module('gnt_splat_correction')
    rng=np.random.default_rng(17)
    image=cv2.GaussianBlur(rng.integers(20,235,(540,960,3),dtype=np.uint8),(0,0),2)
    anchor=cv2.warpAffine(image,np.float32([[1,0,12],[0,1,0]]),(960,540),borderMode=cv2.BORDER_REFLECT)
    yy,xx=np.mgrid[:540,:960].astype(np.float32)
    residual=np.stack([xx,yy,xx*0],axis=2)
    dis=cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_MEDIUM)
    warped,confidence=module.flow_to_anchor(image,anchor,residual,dis)
    area=np.s_[40:-40,40:-40]
    assert np.median(np.abs(warped[area][...,0]-(xx[area]+12)))<1
    assert np.median(np.abs(warped[area][...,1]-yy[area]))<1
    assert np.median(confidence[area])>.7
    assert np.mean(confidence[:,-8:])<.2


def test_colour_fit_recovers_known_exposure_shift(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).parents[1]/'scripts'))
    module=importlib.import_module('gnt_splat_correction')
    ramp=np.tile(np.linspace(30,180,960,dtype=np.float32),(540,1))
    source=np.stack([ramp,ramp,ramp],axis=2).astype(np.uint8)
    target=(source.astype(np.float32)*1.1+8).astype(np.uint8)
    corrected=module.apply_colour(source,module.colour_fit(source,target))
    assert np.mean(np.abs(corrected-target))<1
