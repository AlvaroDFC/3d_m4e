"""Tests for the matplotlib-free symbol builders in multibody_3d.plotting.plotting_helper."""
import numpy as np
import pytest

from multibody_3d.plotting import plotting_helper as helper


def _units(n_frames=5, n=2, seed=0):
    v = np.random.default_rng(seed).normal(size=(n_frames, n, 3))
    return v / np.linalg.norm(v, axis=-1, keepdims=True)


def test_perpendicular_frame_is_orthonormal():
    unit = _units()
    u, v = helper.perpendicular_frame(unit, helper.reference_axes(unit))
    np.testing.assert_allclose(np.linalg.norm(u, axis=-1), 1.0)
    np.testing.assert_allclose(np.linalg.norm(v, axis=-1), 1.0)
    np.testing.assert_allclose(np.sum(u * unit, axis=-1), 0.0, atol=1e-12)
    np.testing.assert_allclose(np.sum(v * unit, axis=-1), 0.0, atol=1e-12)
    np.testing.assert_allclose(np.sum(u * v, axis=-1), 0.0, atol=1e-12)


def test_reference_axis_avoids_the_direction_the_element_points_along():
    along_z = np.tile([0.0, 0.0, 1.0], (4, 1, 1))
    assert helper.reference_axes(along_z)[0] != 2
    in_xy = np.array([[[np.cos(t), np.sin(t), 0.0]] for t in np.linspace(0.0, 3.0, 6)])
    assert helper.reference_axes(in_xy)[0] == 2   # only z never lines up with the element


def test_reference_axis_ignores_non_finite_frames():
    unit = np.tile([0.0, 0.0, 1.0], (4, 1, 1))
    unit[-1] = np.nan
    assert helper.reference_axes(unit)[0] != 2


def test_symbols_of_coincident_points_stay_finite():
    a = np.zeros((2, 1, 3))
    u = np.tile([1.0, 0.0, 0.0], (2, 1, 1))
    v = np.tile([0.0, 1.0, 0.0], (2, 1, 1))
    assert np.all(np.isfinite(helper.coil_segments(a, a, u, v, 0.1)))
    assert np.all(np.isfinite(helper.dashpot_segments(a, a, u, 0.1)))


@pytest.mark.parametrize("turns, steps", [(3, 4), (7, 10)])
def test_coil_has_one_segment_per_vertex_gap(turns, steps):
    unit = _units(n_frames=2, n=3)
    a, b = np.zeros((2, 3, 3)), 2.0 * unit
    u, v = helper.perpendicular_frame(unit, helper.reference_axes(unit))
    segs = helper.coil_segments(a, b, u, v, 0.1, turns=turns, steps=steps)
    assert segs.shape == (2, 3 * (turns * steps + 2), 2, 3)


def test_arrow_is_a_shaft_with_four_barbs_folded_back_from_the_tip():
    unit   = _units(n_frames=3, n=2)
    origin = np.random.default_rng(1).normal(size=(3, 2, 3))
    vec    = 0.8 * unit
    u, v   = helper.perpendicular_frame(unit, helper.reference_axes(unit))
    segs   = helper.arrow_segments(origin, vec, u, v, head=0.1)

    assert segs.shape == (3, 2 * 5, 2, 3)
    arrow = segs.reshape(3, 2, 5, 2, 3)                      # frame, arrow, segment, start/end, xyz
    np.testing.assert_allclose(arrow[:, :, 0, 0], origin)
    np.testing.assert_allclose(arrow[:, :, 0, 1], origin + vec)
    for k in range(1, 5):
        np.testing.assert_allclose(arrow[:, :, k, 0], origin + vec)
    back = arrow[:, :, 1:, 1] - arrow[:, :, 1:, 0]
    assert np.all(np.sum(back * unit[:, :, None, :], axis=-1) < 0)


def test_zero_length_arrows_stay_finite():
    zero = np.zeros((2, 1, 3))
    u = np.tile([1.0, 0.0, 0.0], (2, 1, 1))
    v = np.tile([0.0, 1.0, 0.0], (2, 1, 1))
    assert np.all(np.isfinite(helper.arrow_segments(zero, zero, u, v, 0.1)))
