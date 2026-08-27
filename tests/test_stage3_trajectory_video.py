import numpy as np
import pytest

from sfm_reconstruction.stage3_trajectory_video import frame_prefix_lengths


def test_frame_prefix_lengths_covers_full_trajectory_monotonically():
    lengths = frame_prefix_lengths(10, 4)

    np.testing.assert_array_equal(lengths, [3, 5, 8, 10])
    assert np.all(np.diff(lengths) >= 0)


def test_frame_prefix_lengths_rejects_empty_inputs():
    with pytest.raises(ValueError, match="positive"):
        frame_prefix_lengths(0, 4)
    with pytest.raises(ValueError, match="positive"):
        frame_prefix_lengths(4, 0)
