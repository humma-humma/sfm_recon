"""Protect spatial ray ordering and differentiable structural supervision."""
import importlib
from pathlib import Path
import numpy as np
import pytest


@pytest.fixture
def loss_module(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).parents[1]/'scripts'))
    return importlib.import_module('gnt_detail_loss')


def test_patch_is_contiguous_in_image_coordinates(loss_module):
    for seed in range(30):
        indices = loss_module.patch_indices(19,23,np.random.default_rng(seed)).reshape(16,16)
        assert indices.min()>=0 and indices.max()<19*23
        np.testing.assert_array_equal(np.diff(indices,axis=1),np.ones((16,15),dtype=int))
        np.testing.assert_array_equal(np.diff(indices,axis=0),np.full((15,16),23))
        assert len(np.unique(indices))==256
        assert np.all(indices//23==(indices[:,0]//23)[:,None])


def test_structural_loss_zero_for_identical_patch_and_detects_blur(loss_module):
    import torch
    torch.set_num_threads(2)
    # A step edge gives a reference with structure rather than random texture.
    target = torch.zeros(16,16,3);target[:,8:]=1.
    assert abs(loss_module.structural_loss(target,target).item())<1e-6
    blurred = torch.nn.functional.avg_pool2d(target.permute(2,0,1)[None],3,stride=1,padding=1)
    prediction = blurred[0].permute(1,2,0).reshape(-1,3).detach().requires_grad_()
    loss = loss_module.structural_loss(prediction,target)
    assert torch.isfinite(loss) and loss>0
    loss.backward()
    assert prediction.grad is not None and torch.isfinite(prediction.grad).all()
    assert prediction.grad.abs().sum()>0


def test_patch_rejects_invalid_size(loss_module):
    with pytest.raises(ValueError):
        loss_module.patch_indices(8,8,np.random.default_rng(0),side=16)
