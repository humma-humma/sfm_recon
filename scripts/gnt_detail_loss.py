"""Contiguous ray patches and a small SSIM penalty for the bounded GNT trial."""
import numpy as np


def patch_indices(height, width, rng, side=16):
    if side>min(height,width) or side<7:
        raise ValueError('Patch must fit the image and the 7-pixel structural window')
    y = int(rng.integers(height-side+1))
    x = int(rng.integers(width-side+1))
    return ((y+np.arange(side))[:,None]*width+x+np.arange(side)[None,:]).ravel()


def structural_loss(prediction, target, side=16):
    """Mean DSSIM on valid 7x7 uniform windows, RGB range [0,1]."""
    import torch.nn.functional as functional
    a = prediction.reshape(-1,side,side,3).permute(0,3,1,2)
    b = target.reshape(-1,side,side,3).permute(0,3,1,2)
    pool = lambda value: functional.avg_pool2d(value,7,stride=1)
    ma,mb = pool(a),pool(b)
    va,vb = (pool(a*a)-ma*ma).clamp_min(0),(pool(b*b)-mb*mb).clamp_min(0)
    covariance = pool(a*b)-ma*mb
    score = ((2*ma*mb+.01**2)*(2*covariance+.03**2))/((ma*ma+mb*mb+.01**2)*(va+vb+.03**2))
    return ((1-score)/2).mean()
