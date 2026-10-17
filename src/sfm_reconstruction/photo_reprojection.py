"""Visibility-checked source-photo reprojection in COLMAP camera coordinates."""
from __future__ import annotations
import cv2
import numpy as np


def project(depth, target_pose, source_pose, target_k, source_k):
    """Return source pixel maps and source-camera Z for target depth samples."""
    y, x = np.indices(depth.shape, dtype=np.float32)
    fx, fy, cx, cy = target_k[:4]
    depth = np.where(np.isfinite(depth) & (depth > 0), depth, 0)
    camera = np.stack(((x-cx)*depth/fx, (y-cy)*depth/fy, depth), axis=-1)
    rt, tt = target_pose
    rs, ts = source_pose
    world = (camera-tt) @ rt
    other = world @ rs.T + ts
    z = other[..., 2]
    denominator = np.where(z > 1e-6, z, 1)
    return ((source_k[0]*other[..., 0]/denominator+source_k[2]).astype(np.float32),
            (source_k[1]*other[..., 1]/denominator+source_k[3]).astype(np.float32), z)


def photo_render(base, depth, pose, intrinsics, sources):
    """Blend agreeing training photographs; leave unsupported pixels as base.

    Sources contain RGB, optical-Z depth, COLMAP (R,t), intrinsics and a scalar
    camera-proximity weight. No target RGB participates in rendering.
    """
    colors, weights = [], []
    for source in sources:
        x, y, z = project(depth, pose, source['pose'], intrinsics, source['k'])
        sd = cv2.remap(source['depth'], x, y, cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT)
        rgb = cv2.remap(source['rgb'], x, y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
        h, w = source['depth'].shape
        valid = ((depth > 0) & np.isfinite(depth) & (z > 0) & (sd > 0) &
                 (x >= 1) & (x < w-2) & (y >= 1) & (y < h-2))
        tolerance = .03 + .03*np.maximum(z, 0)
        valid &= np.isfinite(sd) & np.isfinite(z)
        agreement = np.where(valid, np.abs(sd-z)/tolerance, np.inf)
        weight = valid * np.exp(-2*agreement**2) * (agreement < 1) * source['weight']
        colors.append(rgb)
        weights.append(weight.astype(np.float32))
    if not sources:
        return base.copy(), np.zeros(depth.shape, np.float32)
    colors, weights = np.stack(colors), np.stack(weights)
    # Colour agreement suppresses occlusion errors that pass the depth check.
    denominator = weights.sum(axis=0)
    estimate = (colors*weights[..., None]).sum(axis=0)/np.maximum(denominator[..., None], 1e-8)
    difference = np.mean(np.abs(colors-estimate), axis=-1)
    weights *= np.exp(-(difference/.12)**2)
    count = (weights > .05).sum(axis=0)
    denominator = weights.sum(axis=0)
    estimate = (colors*weights[..., None]).sum(axis=0)/np.maximum(denominator[..., None], 1e-8)
    mask = ((count >= 2) & (denominator > .15)).astype(np.float32)
    # Erode before feathering so unsupported pixels never acquire photo content.
    mask = cv2.GaussianBlur(cv2.erode(mask, np.ones((3, 3), np.uint8)), (5, 5), 0)*mask
    mask *= .9
    return np.clip(base*(1-mask[..., None])+estimate*mask[..., None], 0, 1), mask
