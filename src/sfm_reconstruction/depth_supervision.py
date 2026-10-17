"""Measured-depth constraints inspired by uncertainty-aware depth-prior work.

This is an adaptation to Splatfacto, not a reproduction of Dense Depth Priors.
"""
import cv2
import numpy as np


def reliable_depth(depth, min_depth=.1, max_depth=8.):
    """Reject invalid measurements and a one-pixel neighborhood of depth edges."""
    valid = np.isfinite(depth) & (depth > min_depth) & (depth < max_depth)
    clean = np.where(valid, depth, 0).astype(np.float32)
    kernel = np.ones((3, 3), np.uint8)
    spread = cv2.dilate(clean, kernel)-cv2.erode(clean, kernel)
    return valid & (spread < .05+.03*clean)


def relative_depth_loss(predicted, measured, weights, beta=.1):
    """Robust relative error; zero-weight/invalid depths contribute no gradient."""
    import torch
    valid = torch.isfinite(measured) & (measured > .1) & (measured < 8) & (weights > 0)
    if not bool(valid.any()):
        return predicted.sum()*0
    error = (predicted[valid]-measured[valid])/measured[valid].clamp_min(.5)
    penalty = torch.where(error.abs() < beta, .5*error.square()/beta, error.abs()-.5*beta)
    confidence = weights[valid]
    return (penalty*confidence).sum()/confidence.sum().clamp_min(1e-8)
