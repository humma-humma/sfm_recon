import numpy as np

from sfm_reconstruction.stage3_gaussian_submaps import (
    SubmapSpec,
    parse_submap_spec,
    select_local_seed_indices,
)


def test_parse_submap_spec_supports_finite_and_open_end():
    assert parse_submap_spec("stairs:40:64") == SubmapSpec("stairs", 40.0, 64.0)
    assert parse_submap_spec("final:92:end") == SubmapSpec("final", 92.0, None)


def test_parse_submap_spec_rejects_invalid_ranges():
    for value in ("bad", ":0:1", "bad:-1:2", "bad:2:2", "bad:3:2"):
        try:
            parse_submap_spec(value)
        except ValueError:
            pass
        else:
            raise AssertionError(f"expected {value!r} to fail")


def test_select_local_seed_indices_keeps_only_multiview_points():
    counts = np.asarray([0, 1, 2, 4, 1, 3, 2], dtype=np.uint16)

    selected = select_local_seed_indices(counts, max_points=3, min_views=2)

    assert len(selected) == 3
    assert np.all(counts[selected] >= 2)
    assert np.all(np.diff(selected) > 0)


def test_select_local_seed_indices_does_not_pad_with_unsupported_points():
    counts = np.asarray([0, 2, 0, 0], dtype=np.uint16)

    selected = select_local_seed_indices(counts, max_points=4, min_views=2)

    np.testing.assert_array_equal(selected, [1])
