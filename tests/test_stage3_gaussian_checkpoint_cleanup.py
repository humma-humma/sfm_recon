import numpy as np
import pytest

from sfm_reconstruction.stage3_gaussian_checkpoint_cleanup import (
    cleanup_mask,
    prune_checkpoint,
)


def test_cleanup_mask_requires_every_support_gate():
    keep = cleanup_mask(
        np.asarray([0.5, 0.05, 0.5, 0.5, 0.5]),
        np.asarray([0.01, 0.01, 0.03, 0.01, 0.01]),
        np.asarray([0.01, 0.01, 0.01, 0.06, 0.01]),
        np.asarray([3, 3, 3, 3, 1]),
        min_opacity=0.1,
        max_gaussian_scale=0.02,
        max_seed_distance=0.05,
        min_camera_support=2,
    )

    np.testing.assert_array_equal(keep, [True, False, False, False, False])


def test_prune_checkpoint_updates_gaussians_and_optimizer_state():
    torch = pytest.importorskip("torch")
    checkpoint = {
        "pipeline": {
            "_model.gauss_params.means": torch.arange(12).reshape(4, 3),
            "_model.gauss_params.opacities": torch.arange(4).reshape(4, 1),
            "unrelated": torch.arange(3),
        },
        "optimizers": {"means": {"state": {0: {"exp_avg": torch.arange(12).reshape(4, 3)}}}},
    }

    result = prune_checkpoint(checkpoint, np.asarray([True, False, True, False]))

    assert result["pipeline"]["_model.gauss_params.means"].shape == (2, 3)
    assert result["pipeline"]["unrelated"].shape == (3,)
    assert result["optimizers"]["means"]["state"][0]["exp_avg"].shape == (2, 3)


def test_cleanup_mask_rejects_invalid_thresholds():
    values = np.ones(2)
    try:
        cleanup_mask(
            values,
            values,
            values,
            values,
            min_opacity=1.1,
            max_gaussian_scale=1.0,
            max_seed_distance=1.0,
            min_camera_support=1,
        )
    except ValueError:
        pass
    else:
        raise AssertionError("expected invalid opacity threshold to fail")
