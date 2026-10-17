import numpy as np
from sfm_reconstruction.photo_reprojection import photo_render, project


def test_identity_projection_and_photo_detail():
    depth = np.ones((12, 16), np.float32)*2
    pose = (np.eye(3), np.zeros(3))
    k = (10., 10., 8., 6.)
    x, y, z = project(depth, pose, pose, k, k)
    yy, xx = np.indices(depth.shape)
    np.testing.assert_allclose(x, xx, atol=1e-6)
    np.testing.assert_allclose(y, yy, atol=1e-6)
    np.testing.assert_allclose(z, depth)
    source = dict(depth=depth, pose=pose, k=k, rgb=np.ones((12, 16, 3), np.float32), weight=1.)
    rendered, mask = photo_render(np.zeros_like(source['rgb']), depth, pose, k, [source, source])
    assert rendered[6, 8, 0] > .85 and mask[6, 8] > .85


def test_occluded_and_single_source_pixels_keep_baseline():
    depth = np.ones((12, 16), np.float32)*2
    pose = (np.eye(3), np.zeros(3))
    k = (10., 10., 8., 6.)
    base = np.ones((12, 16, 3), np.float32)*.2
    source = dict(depth=depth*.5, pose=pose, k=k, rgb=np.ones_like(base), weight=1.)
    for sources in ([source, source], [dict(source, depth=depth)]):
        rendered, mask = photo_render(base, depth, pose, k, sources)
        np.testing.assert_allclose(rendered, base)
        assert not mask.any()


def test_translation_changes_projection_by_focal_times_baseline_over_depth():
    depth = np.ones((3, 4), np.float32)*2
    target = (np.eye(3), np.zeros(3))
    source = (np.eye(3), np.array([.2, 0, 0]))
    x, _, _ = project(depth, target, source, (10, 10, 2, 1), (10, 10, 2, 1))
    np.testing.assert_allclose(x, np.indices(depth.shape)[1]+1, atol=1e-6)


def test_invalid_depth_does_not_contaminate_fallback():
    depth = np.full((12, 16), np.nan, np.float32)
    pose = (np.eye(3), np.zeros(3))
    k = (10., 10., 8., 6.)
    base = np.ones((12, 16, 3), np.float32)*.2
    source = dict(depth=np.ones_like(depth)*2, pose=pose, k=k, rgb=np.ones_like(base), weight=1.)
    rendered, mask = photo_render(base, depth, pose, k, [source, source])
    np.testing.assert_allclose(rendered, base)
    assert not mask.any()
