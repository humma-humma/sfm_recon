import numpy as np
import pytest
from sfm_reconstruction.depth_supervision import reliable_depth, relative_depth_loss


def test_invalid_depth_and_surface_boundary_are_excluded():
    depth=np.ones((12,12),np.float32)
    depth[:,6:]=2
    depth[2,2]=np.nan
    mask=reliable_depth(depth)
    assert not mask[:,5:7].any()
    assert not mask[1:4,1:4].any()
    assert mask[9,2] and mask[9,9]


def test_depth_gradient_moves_predictions_toward_observations_only_where_reliable():
    torch=pytest.importorskip('torch')
    predicted=torch.tensor([.3,3.,.4],requires_grad=True)
    measured=torch.tensor([1.,1.,float('nan')])
    weights=torch.tensor([1.,0.,1.])
    loss=relative_depth_loss(predicted,measured,weights)
    loss.backward()
    assert torch.isfinite(loss)
    assert predicted.grad[0]<0
    assert predicted.grad[1]==0 and predicted.grad[2]==0


def test_empty_mask_preserves_zero_gradient():
    torch=pytest.importorskip('torch')
    prediction=torch.tensor([.2],requires_grad=True)
    loss=relative_depth_loss(prediction,torch.tensor([0.]),torch.tensor([0.]))
    loss.backward()
    assert loss.item()==0 and prediction.grad.item()==0


def test_pruning_retains_unknown_occluded_and_surface_supported_points():
    from sfm_reconstruction.stage3_gaussian_depth_cleanup import depth_keep_mask
    surface=np.array([0,0,1,0,0])
    conflicts=np.array([0,2,5,3,8])
    keep=depth_keep_mask(np.ones(5,dtype=bool),surface,conflicts,min_conflicts=3)
    np.testing.assert_array_equal(keep,[True,True,True,False,False])
