import numpy as np
import pytest

from sfm_reconstruction.demo_video import camera_orbit_basis, orbit_front, trim_point_table
from sfm_reconstruction.open3d_viewer import PlyTable
from sfm_reconstruction.models import Pose


def test_trim_point_table_filters_percentile_outlier():
    table = PlyTable(
        ["x", "y", "z", "red"],
        np.array(
            [
                [0.0, 0.0, 0.0, 255.0],
                [1.0, 1.0, 1.0, 128.0],
                [2.0, 2.0, 2.0, 64.0],
                [100.0, 100.0, 100.0, 0.0],
            ],
            dtype=np.float64,
        ),
    )

    trimmed = trim_point_table(table, 10.0)

    assert trimmed.values.shape[0] == 2
    assert trimmed.values[:, :3].tolist() == [[1.0, 1.0, 1.0], [2.0, 2.0, 2.0]]


def test_trim_point_table_rejects_invalid_percentile():
    table = PlyTable(["x", "y", "z"], np.zeros((1, 3), dtype=np.float64))

    with pytest.raises(ValueError, match="trim-percentile"):
        trim_point_table(table, 50.0)


def test_orbit_front_keeps_unit_length_and_rotates():
    first = orbit_front(0, 4, np.array([1.0, 0.0, -0.5]))
    second = orbit_front(1, 4, np.array([1.0, 0.0, -0.5]))

    assert np.isclose(np.linalg.norm(first), 1.0)
    assert np.isclose(np.linalg.norm(second), 1.0)
    assert not np.allclose(first, second)


def test_orbit_front_rotates_about_requested_up_axis():
    first = orbit_front(0, 4, np.array([1.0, 0.0, 0.0]), up=np.array([0.0, 1.0, 0.0]))
    second = orbit_front(1, 4, np.array([1.0, 0.0, 0.0]), up=np.array([0.0, 1.0, 0.0]))

    assert np.allclose(first, [1.0, 0.0, 0.0])
    assert np.allclose(second, [0.0, 0.0, -1.0], atol=1e-8)


def test_camera_orbit_basis_uses_camera_down_and_first_center():
    poses = {
        "00000.jpg": Pose(np.eye(3), np.array([0.0, 0.0, -2.0])),
        "00001.jpg": Pose(np.eye(3), np.array([-2.0, 0.0, 0.0])),
    }

    up, front = camera_orbit_basis(poses, np.zeros(3))

    assert np.allclose(up, [0.0, -1.0, 0.0])
    assert np.allclose(front, [0.0, 0.0, 1.0])
