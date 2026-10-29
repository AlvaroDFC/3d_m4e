"""Tests for the snapshot / animation functions in multibody_3d.plotting (Agg backend)."""
import matplotlib

matplotlib.use("Agg")

import shutil  # noqa: E402
from types import SimpleNamespace  # noqa: E402

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pytest  # noqa: E402
import sympy as sym  # noqa: E402
from matplotlib.animation import FuncAnimation  # noqa: E402
from mpl_toolkits.mplot3d import proj3d  # noqa: E402

from multibody_3d import (  # noqa: E402
    MbdSystem3D, compute_forces_states, compute_kinematics_states, plotting,
)
from multibody_3d.plotting import plotting_main  # noqa: E402

pytestmark = pytest.mark.filterwarnings("ignore:Animation was deleted without rendering anything")

N_FRAMES = 4


@pytest.fixture(autouse=True)
def _close_figures():
    yield
    plt.close("all")


def _rot_z(angle):
    c, s = np.cos(angle), np.sin(angle)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def _chain(n_frames=N_FRAMES, with_points=True):
    """Ground -> body 1 -> body 2 with synthetic kinematics (geometry mapping only)."""
    rng = np.random.default_rng(0)
    axis = np.array([0.0, 0.0, 1.0])
    mbd = SimpleNamespace(joint_system=SimpleNamespace(joints=[
        SimpleNamespace(parent=0, child=1, axis_u_vec=axis),
        SimpleNamespace(parent=1, child=2, axis_u_vec=axis),
    ]))
    angles = np.linspace(0.0, 1.0, n_frames)
    kin = SimpleNamespace(
        ts=np.linspace(0.0, 1.0, n_frames),
        r_cg=rng.normal(size=(n_frames, 2, 3)),
        R_cg=np.array([[_rot_z(a), _rot_z(2.0 * a)] for a in angles]),
        r_joint=rng.normal(size=(n_frames, 2, 3)),
        r_pts=rng.normal(size=(n_frames, 2, 3)) if with_points else np.zeros((n_frames, 0, 3)),
        point_body_ids=(2, 2) if with_points else (),
        pt_body_slices={2: slice(0, 2)} if with_points else {},
        r_gr=rng.normal(size=(1, 3)) if with_points else np.zeros((0, 3)),
    )
    return mbd, kin


def _all_frames(mbd, kin, **kw):
    n = kin.r_cg.shape[0]
    return plotting_main._scene_geometry(mbd, kin, np.arange(n), **kw)


def _with_forces(mbd, **elements):
    empty = dict(
        tension_springs=[], tension_dampers=[], torsion_springs=[], torsion_dampers=[],
        cg_forces=[], point_forces=[],
    )
    mbd.forces_def = SimpleNamespace(**{**empty, **elements})
    return mbd


def _arrow_system(n_frames=N_FRAMES):
    """_chain with a CG force on body 1 (Fx 1 -> 4 N) and a point force on body-2 pt 1 (Fy 2 N)."""
    mbd, kin = _chain(n_frames)
    _with_forces(
        mbd,
        cg_forces=[SimpleNamespace(body_id=1)],
        point_forces=[SimpleNamespace(body_id=2, point_idx=1)],
    )
    cg = np.zeros((n_frames, 2, 6))
    cg[:, 0, 0] = np.linspace(1.0, 4.0, n_frames)
    pf = np.zeros((n_frames, 1, 3))
    pf[:, 0, 1] = 2.0
    return mbd, kin, SimpleNamespace(cg=cg, point_forces=pf)


def _tension(body_id_a, pt_idx_a, body_id_b, pt_idx_b):
    return SimpleNamespace(
        body_id_a=body_id_a, pt_idx_a=pt_idx_a, body_id_b=body_id_b, pt_idx_b=pt_idx_b,
    )


def _scene_size(kin):
    pts = np.concatenate([
        kin.r_cg.reshape(-1, 3), kin.r_joint.reshape(-1, 3),
        kin.r_pts.reshape(-1, 3), kin.r_gr.reshape(-1, 3),
    ])
    return float(np.ptp(pts, axis=0).max())


def _system_size(kin):
    moving = np.concatenate([kin.r_cg, kin.r_joint, kin.r_pts], axis=1)
    return float(np.ptp(moving, axis=1).max())


def _off_axis(points, a, b):
    """Distance of *points* ``(F, m, 3)`` from the line through *a* and *b* ``(F, 3)``."""
    d   = (b - a) / np.linalg.norm(b - a, axis=-1, keepdims=True)
    rel = points - a[:, None]
    return np.linalg.norm(rel - np.sum(rel * d[:, None], axis=-1, keepdims=True) * d[:, None], axis=-1)


# ---------------------------------------------------------------------------
# _scene_geometry: pure numpy
# ---------------------------------------------------------------------------

def test_links_follow_the_joint_graph():
    mbd, kin = _chain()
    g = _all_frames(mbd, kin)
    links = g.layers["links"]
    assert links.segments.shape == (N_FRAMES, 3, 2, 3)
    assert links.colors == (1, 1, 2)
    assert links.width == 4.0
    segs = links.segments
    # ground -> 1: joint to child CG only; 1 -> 2: parent CG to joint, then joint to child CG
    np.testing.assert_allclose(segs[:, 0, 0], kin.r_joint[:, 0])
    np.testing.assert_allclose(segs[:, 0, 1], kin.r_cg[:, 0])
    np.testing.assert_allclose(segs[:, 1, 0], kin.r_cg[:, 0])
    np.testing.assert_allclose(segs[:, 1, 1], kin.r_joint[:, 1])
    np.testing.assert_allclose(segs[:, 2, 0], kin.r_joint[:, 1])
    np.testing.assert_allclose(segs[:, 2, 1], kin.r_cg[:, 1])


def test_body_point_dashes_run_from_the_owner_cg():
    mbd, kin = _chain()
    dashes = _all_frames(mbd, kin).layers["dashes"]
    assert dashes.colors == (2, 2)
    assert dashes.width == 2.5
    np.testing.assert_allclose(dashes.segments[:, :, 0], kin.r_cg[:, [1, 1]])
    np.testing.assert_allclose(dashes.segments[:, :, 1], kin.r_pts)


def test_triads_are_off_by_default_and_follow_the_body_axes_when_enabled():
    mbd, kin = _chain()
    assert "triads" not in _all_frames(mbd, kin).layers

    triads = _all_frames(mbd, kin, triad_scale=0.5, show_triads=True).layers["triads"].segments
    assert triads.shape == (N_FRAMES, 6, 2, 3)
    for b in range(2):
        for k in range(3):
            seg = triads[:, 3 * b + k]
            np.testing.assert_allclose(seg[:, 0], kin.r_cg[:, b])
            np.testing.assert_allclose(seg[:, 1] - seg[:, 0], 0.5 * kin.R_cg[:, b, :, k])


def test_geometry_of_a_frame_subset_matches_the_full_geometry():
    mbd, kin, forces = _arrow_system()
    _with_forces(
        mbd,
        tension_springs=[_tension(0, 0, 2, 1)],
        tension_dampers=[_tension(0, 0, 2, 0)],
        torsion_springs=[SimpleNamespace(joint_idx=1)],
        torsion_dampers=[SimpleNamespace(joint_idx=0)],
        cg_forces=mbd.forces_def.cg_forces,
        point_forces=mbd.forces_def.point_forces,
    )
    full = _all_frames(mbd, kin, show_triads=True, forces=forces)
    sub  = plotting_main._scene_geometry(mbd, kin, [2], show_triads=True, forces=forces)
    assert set(full.layers) == {
        "links", "dashes", "triads", "springs", "dampers", "torsion_springs", "torsion_dampers",
        "cg_forces", "point_forces",
    }
    for name in full.layers:
        np.testing.assert_allclose(sub.layers[name].segments[0], full.layers[name].segments[2])
    np.testing.assert_allclose(sub.ts, full.ts[[2]])
    assert sub.bounds == full.bounds   # bounds span every frame, not only the chosen ones


def test_default_bounds_are_a_cube_enclosing_every_frame():
    mbd, kin = _chain()
    limits = np.reshape(_all_frames(mbd, kin).bounds, (3, 2))
    assert np.ptp(limits, axis=1) == pytest.approx(np.full(3, np.ptp(limits[0])))
    for pts in (kin.r_cg, kin.r_joint, kin.r_pts, kin.r_gr):
        flat = pts.reshape(-1, 3)
        assert np.all(flat >= limits[:, 0]) and np.all(flat <= limits[:, 1])


def test_bounds_ignore_non_finite_states():
    mbd, kin = _chain()
    kin.r_cg[-1] = np.inf   # e.g. an unsaved solver step
    g = plotting_main._scene_geometry(mbd, kin, [0])
    assert np.all(np.isfinite(g.bounds))


def test_geometry_rejects_kinematics_of_another_system():
    mbd, kin = _chain()
    mbd.joint_system.joints.pop()
    with pytest.raises(ValueError, match="joints"):
        plotting_main._scene_geometry(mbd, kin, [0])


def test_geometry_rejects_malformed_bounds():
    mbd, kin = _chain()
    with pytest.raises(ValueError, match="bounds"):
        plotting_main._scene_geometry(mbd, kin, [0], bounds=(0, 1, 0, 1))


# ---------------------------------------------------------------------------
# _scene_geometry: springs, dampers and torsion symbols
# ---------------------------------------------------------------------------

def test_springs_need_forces_def_and_can_be_disabled():
    mbd, kin = _chain()
    assert set(_all_frames(mbd, kin).layers) == {"links", "dashes"}   # no forces_def at all
    _with_forces(mbd, tension_springs=[_tension(0, 0, 2, 1)])
    assert "springs" in _all_frames(mbd, kin).layers
    assert "springs" not in _all_frames(mbd, kin, show_springs=False).layers


def test_tension_spring_winds_around_the_axis_between_its_points():
    mbd, kin = _chain()
    _with_forces(mbd, tension_springs=[_tension(0, 0, 2, 1)])   # ground pt 0 <-> body 2 pt 1
    layer = _all_frames(mbd, kin).layers["springs"]
    segs  = layer.segments
    a = np.broadcast_to(kin.r_gr[0], (N_FRAMES, 3))
    b = kin.r_pts[:, 1]

    assert layer.label == "Tension spring"
    np.testing.assert_allclose(segs[:, 0, 0], a)
    np.testing.assert_allclose(segs[:, -1, 1], b)
    np.testing.assert_allclose(segs[:, :-1, 1], segs[:, 1:, 0])   # one connected polyline
    assert _off_axis(segs[:, :, 0], a, b).max() == pytest.approx(0.03 * _scene_size(kin))


def test_tension_damper_is_a_dashpot_between_its_points():
    mbd, kin = _chain()
    _with_forces(mbd, tension_dampers=[_tension(0, 0, 2, 0)])
    layer = _all_frames(mbd, kin).layers["dampers"]
    segs  = layer.segments
    a = np.broadcast_to(kin.r_gr[0], (N_FRAMES, 3))
    b = kin.r_pts[:, 0]

    assert segs.shape == (N_FRAMES, 6, 2, 3)
    assert layer.colors == ("m",) and layer.label == "Tension damper"
    np.testing.assert_allclose(segs[:, 0, 0], a)    # lead wire starts at a
    np.testing.assert_allclose(segs[:, -1, 1], b)   # rod ends at b
    off = _off_axis(segs.reshape(N_FRAMES, -1, 3), a, b)
    assert off.max() == pytest.approx(0.03 * _scene_size(kin))


def test_torsion_symbols_are_rings_normal_to_the_joint_axis():
    mbd, kin = _chain()
    mbd.joint_system.joints[1].axis_u_vec = np.array([1.0, 0.0, 0.0])   # in the frame of body 1
    _with_forces(
        mbd,
        torsion_springs=[SimpleNamespace(joint_idx=1)],
        torsion_dampers=[SimpleNamespace(joint_idx=1)],
    )
    layers = _all_frames(mbd, kin).layers
    centre = kin.r_joint[:, 1]
    axis   = kin.R_cg[:, 0] @ np.array([1.0, 0.0, 0.0])                 # joint axis in the world
    ring   = 0.04 * _scene_size(kin)

    springs = layers["torsion_springs"].segments                        # two rings of 32 segments
    assert springs.shape == (N_FRAMES, 64, 2, 3)
    for part, radius in ((slice(0, 32), ring), (slice(32, 64), 0.75 * ring)):
        rel = springs[:, part, 0] - centre[:, None]
        np.testing.assert_allclose(np.linalg.norm(rel, axis=-1), radius)
        np.testing.assert_allclose(np.sum(rel * axis[:, None], axis=-1), 0.0, atol=1e-12)

    dampers = layers["torsion_dampers"].segments                        # ring + small dashpot
    assert dampers.shape == (N_FRAMES, 36, 2, 3)
    rel = dampers[:, :32, 0] - centre[:, None]
    np.testing.assert_allclose(np.linalg.norm(rel, axis=-1), 1.25 * ring)
    rel = dampers[:, 32:].reshape(N_FRAMES, -1, 3) - centre[:, None]
    np.testing.assert_allclose(np.sum(rel * axis[:, None], axis=-1), 0.0, atol=1e-12)


def test_torsion_symbol_of_a_symbolic_axis_is_skipped_with_a_warning():
    mbd, kin = _chain()
    mbd.joint_system.joints[1].axis_u_vec = sym.Matrix([0, 0, sym.Symbol("s")])
    _with_forces(mbd, torsion_springs=[SimpleNamespace(joint_idx=1)])
    with pytest.warns(UserWarning, match="symbolic axis"):
        layers = _all_frames(mbd, kin).layers
    assert "torsion_springs" not in layers


# ---------------------------------------------------------------------------
# _scene_geometry: force arrows
# ---------------------------------------------------------------------------

def test_force_arrows_start_at_their_points_and_scale_to_the_longest_force():
    mbd, kin, forces = _arrow_system()
    g      = _all_frames(mbd, kin, forces=forces)
    longest = 0.2 * _system_size(kin)

    cg = g.layers["cg_forces"]
    assert cg.segments.shape == (N_FRAMES, 5, 2, 3)         # shaft + four barbs
    assert cg.colors == ("r",) and cg.label.startswith("CG force") and "4" in cg.label
    shaft = cg.segments[:, 0]
    np.testing.assert_allclose(shaft[:, 0], kin.r_cg[:, 0])
    np.testing.assert_allclose(
        shaft[:, 1] - shaft[:, 0],
        np.outer(np.linspace(1.0, 4.0, N_FRAMES) / 4.0 * longest, [1.0, 0.0, 0.0]),
        atol=1e-12,
    )
    np.testing.assert_allclose(cg.segments[:, 1:, 0], shaft[:, None, 1] * np.ones((1, 4, 1)))   # barbs start at the tip

    pt = g.layers["point_forces"]
    assert pt.colors == ("b",) and pt.label.startswith("Point force")
    np.testing.assert_allclose(pt.segments[:, 0, 0], kin.r_pts[:, 1])
    np.testing.assert_allclose(
        pt.segments[:, 0, 1] - pt.segments[:, 0, 0],
        np.tile([0.0, 0.5 * longest, 0.0], (N_FRAMES, 1)), atol=1e-12,
    )


def test_force_arrow_scale_ignores_travel_and_distant_ground_points():
    mbd, kin, forces = _arrow_system()
    original = _all_frames(mbd, kin, forces=forces, force_scale=0.15)
    travel = np.arange(N_FRAMES)[:, None, None] * np.array([100.0, -50.0, 25.0])
    for positions in (kin.r_cg, kin.r_joint, kin.r_pts):
        positions += travel
    kin.r_gr += 10000.0
    translated = _all_frames(mbd, kin, forces=forces, force_scale=0.15)
    for name in ("cg_forces", "point_forces"):
        before = original.layers[name].segments
        after = translated.layers[name].segments
        np.testing.assert_allclose(
            after[:, :, 1] - after[:, :, 0], before[:, :, 1] - before[:, :, 0],
            atol=1e-12,
        )
    shaft = translated.layers["cg_forces"].segments[-1, 0]
    assert np.linalg.norm(shaft[1] - shaft[0]) == pytest.approx(0.15 * _system_size(kin))


def test_force_arrow_scale_handles_non_finite_and_zero_size_geometry():
    mbd, kin, forces = _arrow_system()
    kin.r_cg[-1] = np.inf
    geometry = plotting_main._scene_geometry(mbd, kin, [0], forces=forces)
    assert np.all(np.isfinite(geometry.layers["cg_forces"].segments))
    for positions in (kin.r_cg, kin.r_joint, kin.r_pts):
        positions[:] = 0.0
    shaft = _all_frames(mbd, kin, forces=forces).layers["cg_forces"].segments[-1, 0]
    assert np.linalg.norm(shaft[1] - shaft[0]) == pytest.approx(0.2)


def test_zero_forces_and_disabled_arrows_draw_nothing():
    mbd, kin, forces = _arrow_system()
    zero = SimpleNamespace(cg=np.zeros_like(forces.cg), point_forces=np.zeros_like(forces.point_forces))
    for layers in (
        _all_frames(mbd, kin, forces=zero).layers,
        _all_frames(mbd, kin, forces=forces, show_forces=False).layers,
        _all_frames(mbd, kin).layers,
    ):
        assert "cg_forces" not in layers and "point_forces" not in layers


def test_force_arrows_ignore_non_finite_forces():
    mbd, kin, forces = _arrow_system()
    forces.cg[-1] = np.nan     # e.g. an unsaved solver step
    g = _all_frames(mbd, kin, forces=forces)
    shaft = g.layers["cg_forces"].segments[:, 0]
    assert np.all(np.isfinite(g.bounds)) and np.all(np.isfinite(shaft))
    np.testing.assert_allclose(shaft[-1, 1], shaft[-1, 0])   # zero-length arrow


def test_bounds_include_the_arrow_tips():
    mbd, kin, forces = _arrow_system()
    g      = _all_frames(mbd, kin, forces=forces, force_scale=2.0)
    limits = np.reshape(g.bounds, (3, 2))
    for layer in (g.layers["cg_forces"], g.layers["point_forces"]):
        tips = layer.segments[:, 0, 1]
        assert np.all(tips >= limits[:, 0]) and np.all(tips <= limits[:, 1])


def test_forces_must_cover_the_frames_of_kin():
    mbd, kin, forces = _arrow_system()
    short = SimpleNamespace(cg=forces.cg[:2], point_forces=forces.point_forces[:2])
    with pytest.raises(ValueError, match="frames"):
        _all_frames(mbd, kin, forces=short)


# ---------------------------------------------------------------------------
# _render: create once, update in place
# ---------------------------------------------------------------------------

def test_render_updates_artists_in_place():
    mbd, kin = _chain()
    g = _all_frames(mbd, kin)
    scene = plotting_main._render(g, 0)
    counts = (len(scene.ax.lines), len(scene.ax.collections))

    again = plotting_main._render(g, 3, scene=scene)

    assert again is scene
    assert (len(scene.ax.lines), len(scene.ax.collections)) == counts
    np.testing.assert_allclose(np.ravel(scene.cg[0].get_data_3d()), kin.r_cg[3, 0])
    np.testing.assert_allclose(scene.joints.get_data_3d(), kin.r_joint[3].T)
    np.testing.assert_allclose(scene.pts[2].get_data_3d(), kin.r_pts[3].T)
    assert scene.ax.get_title() == "t = 1.000 s"


@pytest.mark.parametrize("provided_axes", [False, True])
def test_render_keeps_legend_outside_plotting_box(provided_axes):
    mbd, kin, forces = _arrow_system()
    geom = _all_frames(mbd, kin, forces=forces)
    ax = plt.figure().add_subplot(projection="3d") if provided_axes else None
    scene = plotting_main._render(geom, 0, ax=ax)
    for frame in (0, 3):
        plotting_main._render(geom, frame, scene=scene)
        scene.fig.canvas.draw()
        renderer = scene.fig.canvas.get_renderer()
        legend_box = scene.ax.get_legend().get_window_extent(renderer)
        axes_box = scene.ax.get_window_extent(renderer)
        assert legend_box.x0 > axes_box.x1
        if not provided_axes:
            figure_box = scene.fig.bbox
            assert legend_box.x1 <= figure_box.x1
            assert legend_box.y0 >= figure_box.y0
            assert legend_box.y1 <= figure_box.y1


@pytest.mark.parametrize("provided_axes", [False, True])
def test_render_keeps_markers_above_lines_after_drawing_and_updating(provided_axes):
    mbd, kin, forces = _arrow_system()
    geom = _all_frames(mbd, kin, forces=forces, show_triads=True)
    ax = plt.figure().add_subplot(projection="3d") if provided_axes else None
    scene = plotting_main._render(geom, 0, ax=ax)
    for frame, view in ((0, (30, -60)), (3, (-15, 120))):
        plotting_main._render(geom, frame, scene=scene)
        scene.ax.view_init(*view)
        scene.fig.canvas.draw()
        line_order = max(collection.get_zorder() for collection in scene.collections.values())
        joint_order = scene.joints.get_zorder()
        point_orders = [line.get_zorder() for line in scene.pts.values()]
        cg_orders = [line.get_zorder() for line in scene.cg]
        assert min(cg_orders) > max(point_orders)
        assert min(point_orders) > joint_order > line_order


def test_render_without_points_skips_point_artists():
    mbd, kin = _chain(with_points=False)
    scene = plotting_main._render(_all_frames(mbd, kin), 0)
    assert scene.pts == {} and "dashes" not in scene.collections
    _, labels = scene.ax.get_legend_handles_labels()
    assert "Body points" not in labels and "GR" not in labels
    assert "Joints" in labels


def test_render_creates_one_collection_per_layer_and_labels_the_elements():
    mbd, kin = _chain()
    _with_forces(
        mbd,
        tension_springs=[_tension(0, 0, 2, 1)],
        tension_dampers=[_tension(0, 0, 2, 0)],
        torsion_springs=[SimpleNamespace(joint_idx=1)],
        torsion_dampers=[SimpleNamespace(joint_idx=0)],
    )
    g = _all_frames(mbd, kin, show_triads=True)
    scene = plotting_main._render(g, 0)
    assert list(scene.collections) == list(g.layers)
    _, labels = scene.ax.get_legend_handles_labels()
    assert {"Tension spring", "Tension damper", "Torsion spring", "Torsion damper"} <= set(labels)

    count = len(scene.ax.collections)
    plotting_main._render(g, 2, scene=scene)
    assert len(scene.ax.collections) == count


# ---------------------------------------------------------------------------
# plot_snapshot
# ---------------------------------------------------------------------------

def test_snapshot_returns_3d_axes_and_resolves_negative_frames():
    mbd, kin = _chain()
    _, ax = plotting.plot_snapshot(mbd, kin)
    assert ax.name == "3d"
    assert ax.get_title() == "t = 1.000 s"
    assert plotting.plot_snapshot(mbd, kin, frame=0)[1].get_title() == "t = 0.000 s"


def test_snapshot_point_labels_are_opt_in_and_use_local_point_indices():
    mbd, kin = _chain()
    kin.pt_body_slices = {1: slice(0, 1), 2: slice(1, 2)}
    kin.point_body_ids = (1, 2)
    _, default_ax = plotting.plot_snapshot(mbd, kin)
    assert not default_ax.texts

    fig, ax = plotting.plot_snapshot(mbd, kin, frame=-2, show_labels=True)
    expected = {
        r"$J_{0}$": kin.r_joint[-2, 0], r"$J_{1}$": kin.r_joint[-2, 1],
        r"$BD_{0}^{1}$": kin.r_pts[-2, 0], r"$BD_{0}^{2}$": kin.r_pts[-2, 1],
        "$GR_0$": kin.r_gr[0],
    }
    assert {text.get_text().strip() for text in ax.texts} == set(expected)
    for text in ax.texts:
        np.testing.assert_allclose(text.get_position_3d(), expected[text.get_text().strip()])
    fig.canvas.draw()


def test_snapshot_point_labels_without_declared_points():
    mbd, kin = _chain(with_points=False)
    _, ax = plotting.plot_snapshot(mbd, kin, show_labels=True)
    assert {text.get_text().strip() for text in ax.texts} == {r"$J_{0}$", r"$J_{1}$"}


def test_snapshot_joint_labels_clear_markers_after_camera_zoom_and_dpi_changes(monkeypatch):
    mbd, kin = _chain()
    rendered_bounds = {}
    original_draw = matplotlib.text.Text.draw

    def record_bounds(text, renderer):
        label = text.get_text().strip()
        if label.startswith("$J_{"):
            rendered_bounds[label] = text.get_window_extent(renderer)
        return original_draw(text, renderer)

    monkeypatch.setattr(matplotlib.text.Text, "draw", record_bounds)
    fig, ax = plotting.plot_snapshot(mbd, kin, show_labels=True)
    joints = next(line for line in ax.lines if line.get_label() == "Joints")
    for dpi, view, limit in ((72, (30, -60), 5), (144, (-15, 120), 100)):
        fig.set_dpi(dpi)
        ax.view_init(*view)
        ax.set(xlim=(-limit, limit), ylim=(-limit, limit), zlim=(-limit, limit))
        fig.canvas.draw()
        assert len(rendered_bounds) == kin.r_joint.shape[1]
        marker_radius = 0.5 * joints.get_markersize() * dpi / 72.0
        for joint_idx, position in enumerate(kin.r_joint[-1]):
            projected = proj3d.proj_transform(*position, ax.get_proj())
            marker = ax.transData.transform(projected[:2])
            label_bounds = rendered_bounds[f"$J_{{{joint_idx}}}$"]
            assert label_bounds.x0 > marker[0] + marker_radius
            assert label_bounds.y0 > marker[1] + marker_radius
            np.testing.assert_allclose(
                [label_bounds.x0, label_bounds.y0] - marker, np.full(2, 8.0 * dpi / 72.0),
                atol=1e-8,
            )


def test_snapshot_frame_out_of_range_raises():
    mbd, kin = _chain()
    with pytest.raises(IndexError):
        plotting.plot_snapshot(mbd, kin, frame=N_FRAMES)


def test_snapshot_draws_on_a_given_3d_axes():
    mbd, kin = _chain()
    fig = plt.figure()
    ax = fig.add_subplot(projection="3d")
    out_fig, out_ax = plotting.plot_snapshot(mbd, kin, ax=ax)
    assert out_ax is ax and out_fig is fig


def test_snapshot_rejects_2d_axes():
    mbd, kin = _chain()
    _, ax = plt.subplots()
    with pytest.raises(ValueError, match="3-D"):
        plotting.plot_snapshot(mbd, kin, ax=ax)


def test_snapshot_honours_explicit_bounds_and_view():
    mbd, kin = _chain()
    _, ax = plotting.plot_snapshot(mbd, kin, bounds=(-1, 1, -2, 2, -3, 3), view=(15, 40))
    assert ax.get_xlim() == (-1, 1)
    assert ax.get_ylim() == (-2, 2)
    assert ax.get_zlim() == (-3, 3)
    assert (ax.elev, ax.azim) == (15, 40)


# ---------------------------------------------------------------------------
# animate_multibody
# ---------------------------------------------------------------------------

def test_animation_is_returned_and_leaves_the_figure_open():
    mbd, kin = _chain()
    anim = plotting.animate_multibody(mbd, kin)
    assert isinstance(anim, FuncAnimation)
    assert len(plt.get_fignums()) == 1


@pytest.mark.parametrize("camera_mode", [None, "track", "global"])
def test_animation_camera_tracks_by_default_and_global_preserves_fixed_bounds(camera_mode):
    mbd, kin = _chain()
    travel = np.arange(N_FRAMES)[:, None, None] * np.array([100.0, -50.0, 25.0])
    for positions in (kin.r_cg, kin.r_joint, kin.r_pts):
        positions[:] = positions[[0]] + travel
    kin.r_gr[:] = 10000.0
    global_bounds = np.reshape(_all_frames(mbd, kin).bounds, (3, 2))
    options = {} if camera_mode is None else {"camera_mode": camera_mode}
    anim = plotting.animate_multibody(mbd, kin, frame_step=2, view=(15, 40), **options)
    widths = []
    for animation_frame, state_frame in enumerate(range(0, N_FRAMES, 2)):
        scene = anim._func(animation_frame)
        limits = np.array([scene.ax.get_xlim(), scene.ax.get_ylim(), scene.ax.get_zlim()])
        widths.append(np.ptp(limits, axis=1))
        if camera_mode == "global":
            np.testing.assert_allclose(limits, global_bounds)
        else:
            moving = np.concatenate([
                kin.r_cg[state_frame], kin.r_joint[state_frame], kin.r_pts[state_frame],
            ])
            np.testing.assert_allclose(limits.mean(axis=1), 0.5 * (moving.min(0) + moving.max(0)))
            assert np.all(moving >= limits[:, 0]) and np.all(moving <= limits[:, 1])
            assert widths[-1].max() < 2.0 * _system_size(kin)
        assert (scene.ax.elev, scene.ax.azim) == (15, 40)
    np.testing.assert_allclose(widths[0], widths[-1])


@pytest.mark.parametrize("camera_mode", ["track", "global"])
def test_animation_explicit_bounds_override_camera_tracking(camera_mode):
    mbd, kin = _chain()
    bounds = (-1, 1, -2, 2, -3, 3)
    anim = plotting.animate_multibody(mbd, kin, bounds=bounds, camera_mode=camera_mode)
    for frame in (0, N_FRAMES - 1):
        scene = anim._func(frame)
        limits = np.array([scene.ax.get_xlim(), scene.ax.get_ylim(), scene.ax.get_zlim()])
        np.testing.assert_allclose(limits.ravel(), bounds)


def test_animation_rejects_unknown_camera_mode_before_creating_a_figure():
    mbd, kin = _chain()
    with pytest.raises(ValueError, match="camera_mode"):
        plotting.animate_multibody(mbd, kin, camera_mode="invalid")
    assert not plt.get_fignums()


def test_tracking_geometry_includes_force_arrows_and_triads_without_zoom_changes():
    mbd, kin, forces = _arrow_system()
    geom = _all_frames(mbd, kin, forces=forces, show_triads=True, camera_mode="track")
    limits = np.asarray(geom.bounds).reshape(N_FRAMES, 3, 2)
    np.testing.assert_allclose(
        np.ptp(limits, axis=2), np.broadcast_to(np.ptp(limits[0], axis=1), (N_FRAMES, 3)),
    )
    for name in ("cg_forces", "point_forces", "triads"):
        positions = geom.layers[name].segments.reshape(N_FRAMES, -1, 3)
        assert np.all(positions >= limits[:, None, :, 0])
        assert np.all(positions <= limits[:, None, :, 1])
    triads = geom.layers["triads"].segments
    np.testing.assert_allclose(
        np.linalg.norm(triads[:, :, 1] - triads[:, :, 0], axis=-1), 0.1 * _system_size(kin),
    )
    subset = plotting_main._scene_geometry(
        mbd, kin, [2], forces=forces, show_triads=True, camera_mode="track",
    )
    np.testing.assert_allclose(subset.bounds[0], geom.bounds[2])


def test_tracking_symbol_sizes_ignore_translation_and_distant_ground_points():
    mbd, kin, forces = _arrow_system()
    mbd.forces_def.tension_springs = [_tension(2, 0, 2, 1)]
    original = _all_frames(mbd, kin, forces=forces, show_triads=True, camera_mode="track")
    travel = np.arange(N_FRAMES)[:, None, None] * np.array([100.0, -50.0, 25.0])
    for positions in (kin.r_cg, kin.r_joint, kin.r_pts):
        positions += travel
    kin.r_gr += 10000.0
    moved = _all_frames(mbd, kin, forces=forces, show_triads=True, camera_mode="track")
    for name in original.layers:
        np.testing.assert_allclose(
            moved.layers[name].segments - travel[:, :, None], original.layers[name].segments,
            atol=1e-12,
        )
    np.testing.assert_allclose(
        np.asarray(moved.bounds).reshape(N_FRAMES, 3, 2) - travel.reshape(N_FRAMES, 3, 1),
        np.asarray(original.bounds).reshape(N_FRAMES, 3, 2), atol=1e-12,
    )


def test_tracking_bounds_handle_missing_points_and_non_finite_frames():
    mbd, kin = _chain(with_points=False)
    kin.r_cg[-1] = np.inf
    kin.r_joint[-1] = np.nan
    geom = _all_frames(mbd, kin, camera_mode="track")
    assert np.all(np.isfinite(geom.bounds))
    np.testing.assert_allclose(geom.bounds[-1], geom.bounds[-2])
    kin.r_cg[:] = 0.0
    kin.r_joint[:] = 0.0
    limits = np.asarray(_all_frames(mbd, kin, camera_mode="track").bounds).reshape(N_FRAMES, 3, 2)
    assert np.all(np.isfinite(limits)) and np.all(np.ptp(limits, axis=2) > 0.0)


def test_animation_saves_a_gif_with_the_selected_frames(tmp_path):
    from PIL import Image
    mbd, kin = _chain(n_frames=5)
    path = tmp_path / "anim.gif"
    plotting.animate_multibody(mbd, kin, frame_step=2, fps=5, save_path=path)
    with Image.open(path) as gif:
        assert gif.n_frames == 3   # frames 0, 2, 4


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg not installed")
def test_animation_saves_an_mp4(tmp_path):
    mbd, kin = _chain()
    path = tmp_path / "anim.mp4"
    plotting.animate_multibody(mbd, kin, fps=5, save_path=path)
    assert path.stat().st_size > 0


def test_animation_rejects_unknown_save_extension(tmp_path):
    mbd, kin = _chain()
    with pytest.raises(ValueError, match="gif"):
        plotting.animate_multibody(mbd, kin, save_path=tmp_path / "anim.txt")
    assert not plt.get_fignums()   # failed before any figure was created


def test_animation_rejects_non_positive_frame_step():
    mbd, kin = _chain()
    with pytest.raises(ValueError, match="frame_step"):
        plotting.animate_multibody(mbd, kin, frame_step=0)


# ---------------------------------------------------------------------------
# Real system: interface with compute_kinematics_states / compute_forces_states
# ---------------------------------------------------------------------------

REAL_MNV = np.array([0.0, 0.0, 0.0, 0.0, 0.25])                  # q (2), qd (2), a
REAL_Q   = np.array([[0.0, 0.0], [0.3, -0.2], [0.6, -0.4]])
REAL_TS  = np.linspace(0.0, 1.0, 3)


@pytest.fixture(scope="module")
def real_system():
    """Hanging 2-body revolute chain with a ground point, a body-2 point and force elements."""
    a  = sym.Symbol("a", real=True)
    z3 = [0.0, 0.0, 0.0]
    data = {
        "NBodies": 2,
        "joints": [(0, 1), (1, 2)],
        "types": ["R", "R"],
        "parent_cg_to_joint": [z3, [0.0, -0.5, 0.0]],
        "joint_to_child_cg":  [[0.0, -0.5, 0.0], [0.0, -0.5, 0.0]],
        "axis_u":  [[0, 0, 1], [0, 0, 1]],
        "axis_u1": [None, None],
        "axis_u2": [None, None],
    }
    mbd = MbdSystem3D(
        data=data,
        points_sym={"a": a},
        Initial_Points={"GR": [z3], "BD": {2: [[a, 0.0, 0.0]]}},
        Force={
            "CG": {1: {"force": [3.0, 0.0, 0.0]}},
            "PointsBD": [(2, 0, [0.0, 2.0, 0.0])],
            "TensionSpring": [(0, 0, 2, 0, 10.0, 1.0)],
            "TensionDamper": [(0, 0, 2, 0, 0.5)],
            "TorsionSpring": [(1, 5.0, 0.0)],
            "TorsionDamper": [(1, 0.2)],
        },
    )
    kin = compute_kinematics_states(mbd, REAL_TS, REAL_Q, REAL_MNV, with_velocities=False)
    return mbd, kin


@pytest.fixture(scope="module")
def real_forces(real_system):
    mbd, _ = real_system
    return compute_forces_states(
        mbd, REAL_TS, np.hstack([REAL_Q, np.zeros_like(REAL_Q)]), REAL_MNV,
    )


def test_snapshot_of_a_real_system_without_velocities(real_system):
    mbd, kin = real_system
    _, ax = plotting.plot_snapshot(mbd, kin, frame=1)
    _, labels = ax.get_legend_handles_labels()
    assert {"CG body 1", "CG body 2", "Joints", "Body points", "GR"} <= set(labels)
    assert {"Tension spring", "Tension damper", "Torsion spring", "Torsion damper"} <= set(labels)
    assert ax.get_title() == "t = 0.500 s"
    assert not any(label.startswith(("CG force", "Point force")) for label in labels)   # no forces given


def test_snapshot_draws_the_force_arrows_of_a_real_system(real_system, real_forces):
    mbd, kin = real_system
    _, ax = plotting.plot_snapshot(mbd, kin, frame=1, forces=real_forces)
    _, labels = ax.get_legend_handles_labels()
    assert any(label.startswith("CG force") and "3" in label for label in labels)
    assert any(label.startswith("Point force") for label in labels)


def test_animation_of_a_real_system(real_system, real_forces, tmp_path):
    mbd, kin = real_system
    plotting.animate_multibody(mbd, kin, forces=real_forces, fps=5, save_path=tmp_path / "real.gif")
    assert (tmp_path / "real.gif").stat().st_size > 0
