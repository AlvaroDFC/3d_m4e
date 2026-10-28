"""plotting_main.py

Matplotlib snapshot and animation of a 3-D multibody system.

Both functions take the system and an already-computed ``KinematicsResult``
(``MbdSystem3D.compute_kinematics`` or ``compute_kinematics_states``); neither
integrates nor evaluates kinematics, and neither calls ``plt.show()`` or
``plt.close()``, so the caller decides whether to display or save.

The drawing is split in two private functions.  ``_scene_geometry`` turns the
kinematics into plain numpy arrays (points and line-segment layers: links, body
points, springs, dampers, body axes) with no matplotlib involved.  ``_render``
creates the artists once and moves them to a frame, so ``plot_snapshot`` and
``animate_multibody`` share one drawing path and an animation only updates
artists in place.
"""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import NamedTuple, Optional

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FFMpegWriter, FuncAnimation, PillowWriter
from mpl_toolkits.mplot3d.art3d import Line3DCollection

from .plotting_helper import (
    arrow_segments,
    coil_segments,
    dashpot_segments,
    perpendicular_frame,
    reference_axes,
    ring_segments,
    tangent_dashpot_segments,
    unit_direction,
)

__all__ = ["plot_snapshot", "animate_multibody"]

_BODY_COLORS = matplotlib.colormaps["tab10"]
_AXIS_COLORS = ("tab:red", "tab:green", "tab:blue")   # body x, y, z axes


class _Layer(NamedTuple):
    segments: np.ndarray      # (F, n, 2, 3) line segments [start, end]
    colors:   tuple           # per segment, or one for all: colour name or int body id
    width:    float
    style:    str             # matplotlib linestyle
    label:    Optional[str]   # legend entry


class _Geometry(NamedTuple):
    ts:        np.ndarray   # (F,) time of each selected frame
    cg:        np.ndarray   # (F, NB, 3)
    joints:    np.ndarray   # (F, NJ, 3)
    pts:       np.ndarray   # (F, n_pt, 3) declared body points
    pt_slices: dict         # body id -> slice into pts
    ground:    np.ndarray   # (n_gr, 3) constant
    layers:    dict         # name -> _Layer, in drawing order
    bounds:    tuple        # (xmin, xmax, ymin, ymax, zmin, zmax)


class _Scene(NamedTuple):
    fig:         object
    ax:          object
    cg:          list   # one Line3D per body
    joints:      object # Line3D
    pts:         dict   # body id -> Line3D
    collections: dict   # layer name -> Line3DCollection


def _scene_geometry(
    mbd, kin, frames, *, triad_scale=None, bounds=None, show_triads=False, show_springs=True,
    forces=None, show_forces=True, force_scale=0.2,
) -> _Geometry:
    """Points and line-segment layers of the chosen *frames* as numpy arrays.

    Bounds, symbol orientations and the force-arrow scale use every frame of
    *kin*, so all snapshots and the animation of one result look alike.
    """
    frames = np.asarray(frames, dtype=int)
    joints = mbd.joint_system.joints
    n_all, n_body = kin.r_cg.shape[:2]
    if kin.r_joint.shape[1] != len(joints):
        raise ValueError(
            f"kin has {kin.r_joint.shape[1]} joints but the system has {len(joints)}."
        )

    r_cg, r_joint, r_pts = kin.r_cg[frames], kin.r_joint[frames], kin.r_pts[frames]

    # Links: parent CG -> joint (none for a ground parent) and joint -> child CG
    starts, ends, owners = [], [], []
    for j, joint in enumerate(joints):
        if joint.parent != 0:
            starts.append(r_cg[:, joint.parent - 1])
            ends.append(r_joint[:, j])
            owners.append(int(joint.parent))
        starts.append(r_joint[:, j])
        ends.append(r_cg[:, joint.child - 1])
        owners.append(int(joint.child))
    links  = np.stack([np.stack(starts, axis=1), np.stack(ends, axis=1)], axis=2)
    layers = {"links": _Layer(links, tuple(owners), 2.5, "-", None)}

    pt_bodies = tuple(int(b) for b in kin.point_body_ids)
    if pt_bodies:
        cg_of_pt = r_cg[:, np.asarray(pt_bodies, dtype=int) - 1]       # (F, n_pt, 3)
        layers["dashes"] = _Layer(np.stack([cg_of_pt, r_pts], axis=2), pt_bodies, 1.0, "--", None)

    ground = np.asarray(kin.r_gr, dtype=float).reshape(-1, 3)
    every  = np.concatenate([
        kin.r_cg.reshape(-1, 3), kin.r_joint.reshape(-1, 3),
        kin.r_pts.reshape(-1, 3), ground,
    ])
    every  = every[np.isfinite(every).all(axis=1)]                     # skip unsaved / failed steps
    lo, hi = (every.min(axis=0), every.max(axis=0)) if len(every) else (-np.ones(3), np.ones(3))
    span   = float((hi - lo).max()) or 1.0
    axis_len = 0.1 * span if triad_scale is None else float(triad_scale)

    if show_triads:
        R      = kin.R_cg[frames]                                      # (F, NB, 3, 3)
        origin = np.broadcast_to(r_cg[:, :, None, :], R.shape)
        tip    = origin + axis_len * np.swapaxes(R, -1, -2)            # row k of R^T = body axis k
        triads = np.stack([origin, tip], axis=3).reshape(len(frames), 3 * n_body, 2, 3)
        layers["triads"] = _Layer(triads, _AXIS_COLORS * n_body, 1.5, "-", None)

    fdef = getattr(mbd, "forces_def", None) if show_springs else None
    if fdef is not None:
        size = 0.03 * span      # coil radius and dashpot half-width
        ring = 0.04 * span      # torsion symbol radius

        def endpoint(body_id, pt_idx):                                 # (n_all, 3), every frame
            if body_id == 0:
                return np.broadcast_to(ground[pt_idx], (n_all, 3))
            return kin.r_pts[:, kin.pt_body_slices[body_id].start + pt_idx]

        def tension_frame(defs):                                       # a, b, u, v of the chosen frames
            a_all = np.stack([endpoint(d.body_id_a, d.pt_idx_a) for d in defs], axis=1)
            b_all = np.stack([endpoint(d.body_id_b, d.pt_idx_b) for d in defs], axis=1)
            ref   = reference_axes(unit_direction(a_all, b_all))
            a, b  = a_all[frames], b_all[frames]
            return (a, b, *perpendicular_frame(unit_direction(a, b), ref))

        def torsion_frame(defs):                                       # centre, e1, e2 of the chosen frames
            ids, axes = [], []
            for d in defs:
                joint = joints[d.joint_idx]
                try:
                    local = np.asarray(joint.axis_u_vec, dtype=float).ravel()
                except TypeError:
                    warnings.warn(
                        f"Joint {d.joint_idx} has a symbolic axis; its torsion symbol is not drawn."
                    )
                    continue
                local = local / np.linalg.norm(local)
                ids.append(d.joint_idx)
                axes.append(
                    np.broadcast_to(local, (n_all, 3)) if joint.parent == 0
                    else kin.R_cg[:, joint.parent - 1] @ local         # parent frame -> world
                )
            if not ids:
                return None
            axis_all = np.stack(axes, axis=1)                          # (n_all, m, 3)
            e1, e2   = perpendicular_frame(axis_all[frames], reference_axes(axis_all))
            return r_joint[:, ids], e1, e2

        if fdef.tension_springs:
            a, b, u, v = tension_frame(fdef.tension_springs)
            layers["springs"] = _Layer(
                coil_segments(a, b, u, v, size), ("k",), 1.2, "-", "Tension spring",
            )
        if fdef.tension_dampers:
            a, b, u, _ = tension_frame(fdef.tension_dampers)
            layers["dampers"] = _Layer(
                dashpot_segments(a, b, u, size), ("m",), 1.2, "-", "Tension damper",
            )
        frame = torsion_frame(fdef.torsion_springs) if fdef.torsion_springs else None
        if frame is not None:
            c, e1, e2 = frame
            segs = np.concatenate(
                [ring_segments(c, e1, e2, ring), ring_segments(c, e1, e2, 0.75 * ring)], axis=1,
            )
            layers["torsion_springs"] = _Layer(segs, ("k",), 1.2, "-", "Torsion spring")
        frame = torsion_frame(fdef.torsion_dampers) if fdef.torsion_dampers else None
        if frame is not None:
            c, e1, e2 = frame
            segs = np.concatenate([
                ring_segments(c, e1, e2, 1.25 * ring),
                tangent_dashpot_segments(c, e1, e2, 1.25 * ring),
            ], axis=1)
            layers["torsion_dampers"] = _Layer(segs, ("m",), 1.2, "-", "Torsion damper")

    if forces is not None and show_forces and getattr(mbd, "forces_def", None) is not None:
        fd = mbd.forces_def
        if forces.cg.shape[0] != n_all:
            raise ValueError(f"forces has {forces.cg.shape[0]} frames but kin has {n_all}.")
        arrows = []                                                    # name, label, colour, origins, vectors (every frame)
        if fd.cg_forces:
            arrows.append((
                "cg_forces", "CG force", "r",
                np.stack([kin.r_cg[:, d.body_id - 1] for d in fd.cg_forces], axis=1),
                np.stack([forces.cg[:, d.body_id - 1, :3] for d in fd.cg_forces], axis=1),
            ))
        if fd.point_forces:
            arrows.append((
                "point_forces", "Point force", "b",
                np.stack([
                    kin.r_pts[:, kin.pt_body_slices[d.body_id].start + d.point_idx]
                    for d in fd.point_forces
                ], axis=1),
                np.asarray(forces.point_forces),
            ))
        arrows = [(*a[:4], np.nan_to_num(a[4], nan=0.0, posinf=0.0, neginf=0.0)) for a in arrows]
        f_max  = max((float(np.linalg.norm(a[4], axis=-1).max()) for a in arrows), default=0.0)
        if f_max > 0.0:
            gain = force_scale * span / f_max      # the longest arrow spans force_scale of the scene
            tips = []
            for name, label, colour, org, vec in arrows:
                vec  = gain * vec
                unit = unit_direction(np.zeros_like(vec), vec)
                u, v = perpendicular_frame(unit[frames], reference_axes(unit))
                layers[name] = _Layer(
                    arrow_segments(org[frames], vec[frames], u, v, 0.05 * span),
                    (colour,), 2.0, "-", f"{label} (longest: {f_max:.3g})",
                )
                tips.append((org + vec).reshape(-1, 3))
            tips = np.concatenate(tips)
            tips = tips[np.isfinite(tips).all(axis=1)]
            if len(tips):
                lo, hi = np.minimum(lo, tips.min(axis=0)), np.maximum(hi, tips.max(axis=0))

    if bounds is None:
        half   = 1.05 * 0.5 * (float((hi - lo).max()) or 1.0) + axis_len
        centre = 0.5 * (lo + hi)
        bounds = np.stack([centre - half, centre + half], axis=1).ravel()
    elif len(bounds) != 6:
        raise ValueError("bounds must be (xmin, xmax, ymin, ymax, zmin, zmax).")
    bounds = tuple(float(v) for v in bounds)

    return _Geometry(
        ts=np.asarray(kin.ts, dtype=float)[frames], cg=r_cg, joints=r_joint,
        pts=r_pts, pt_slices=dict(kin.pt_body_slices), ground=ground,
        layers=layers, bounds=bounds,
    )


def _render(geom, k, *, scene=None, ax=None, view=None) -> _Scene:
    """Create the artists on the first call (``scene is None``) and move them to frame *k*.

    Pass the returned scene back on later calls to update the artists in place.
    """
    if scene is None:
        if ax is None:
            ax = plt.figure(figsize=(8, 6)).add_subplot(projection="3d")
        elif not hasattr(ax, "get_zlim"):
            raise ValueError("ax must be a 3-D axes (projection='3d').")

        n_body = geom.cg.shape[1]
        colors = [_BODY_COLORS(b % _BODY_COLORS.N) for b in range(n_body)]
        blank  = ([0.0], [0.0], [0.0])   # replaced with frame data below

        cg = [
            ax.plot(*blank, ls="none", marker="D", ms=7, mfc=colors[b], mec="k",
                    label=f"CG body {b + 1}")[0]
            for b in range(n_body)
        ]
        joints = ax.plot(*blank, ls="none", marker="o", ms=7, mfc="w", mec="k", label="Joints")[0]
        pts = {
            b: ax.plot(*blank, ls="none", marker="x", color=colors[b - 1],
                       **({"label": "Body points"} if i == 0 else {}))[0]
            for i, b in enumerate(geom.pt_slices)
        }
        collections = {}
        for name, layer in geom.layers.items():
            collections[name] = Line3DCollection(
                layer.segments[k],
                colors=[colors[c - 1] if isinstance(c, int) else c for c in layer.colors],
                linewidths=layer.width, linestyles=layer.style,
                **({"label": layer.label} if layer.label else {}),
            )
            ax.add_collection3d(collections[name])
        if len(geom.ground):
            ax.plot(*geom.ground.T, ls="none", marker="s", ms=8, color="k", label="GR")

        ax.set_xlim(*geom.bounds[0:2])
        ax.set_ylim(*geom.bounds[2:4])
        ax.set_zlim(*geom.bounds[4:6])
        ax.set_box_aspect(np.ptp(np.reshape(geom.bounds, (3, 2)), axis=1))
        ax.set(xlabel="X", ylabel="Y", zlabel="Z")
        ax.legend(loc="upper left", fontsize="small")
        if view is not None:
            ax.view_init(*view)
        scene = _Scene(ax.figure, ax, cg, joints, pts, collections)

    for b, line in enumerate(scene.cg):
        line.set_data_3d(*geom.cg[k, b][:, None])
    scene.joints.set_data_3d(*geom.joints[k].T)
    for b, line in scene.pts.items():
        line.set_data_3d(*geom.pts[k, geom.pt_slices[b]].T)
    for name, collection in scene.collections.items():
        collection.set_segments(geom.layers[name].segments[k])
    scene.ax.set_title(f"t = {geom.ts[k]:.3f} s")
    return scene


def plot_snapshot(
    mbd, kin, frame=-1, *, ax=None, bounds=None, show_triads=False, show_springs=True,
    forces=None, show_forces=True, force_scale=0.2, triad_scale=None, view=None,
):
    """Draw one configuration of a 3-D multibody system.

    Shows the body CGs (coloured by body), the links CG - joint - CG, the
    joints, the declared body points (dashed from their CG), the ground
    points and the tension / torsion springs and dampers of ``mbd.forces_def``;
    optionally the body axes at each CG and, given *forces*, the CG and point
    forces as arrows.

    Parameters
    ----------
    mbd : MbdSystem3D
        Supplies the joint graph (parent and child of every joint) and the
        force elements.
    kin : KinematicsResult
        From :meth:`MbdSystem3D.compute_kinematics` or
        :func:`compute_kinematics_states`; the velocity fields are not used.
    frame : int, optional
        Index into the frames of *kin*; negative values count from the end.
    ax : Axes3D, optional
        3-D axes to draw on; a new figure is created if omitted.
    bounds : tuple of 6 floats, optional
        ``(xmin, xmax, ymin, ymax, zmin, zmax)``.  Default: a cube spanning
        every frame of *kin* (and the arrow tips), so snapshots of one result
        share the same view.
    show_triads : bool, optional
        Draw the body x / y / z axes (red / green / blue) at each CG.
    show_springs : bool, optional
        Draw the springs and dampers: tension ones as a coil (black) and a
        dashpot (magenta) between their points, torsion ones as rings around
        the joint (a plain ring pair for springs, a ring with a dashpot for
        dampers) in the plane normal to the joint axis.  Torsion symbols of a
        joint with a symbolic axis are skipped with a warning.
    forces : ForcesEvalResult, optional
        From :func:`compute_forces_states` for the same states as *kin*.  Draws
        the CG forces (red) and the point forces (blue) of ``mbd.forces_def``
        as arrows from their point of application.
    show_forces : bool, optional
        Set to False to hide the arrows although *forces* is given.
    force_scale : float, optional
        Length of the longest arrow (over all frames and force elements) as a
        fraction of the scene size; the legend gives the force it stands for.
    triad_scale : float, optional
        Length of the body axes; default 10 % of the scene size.
    view : tuple of 2 floats, optional
        Camera ``(elev, azim)`` in degrees; default keeps the axes' own view.

    Returns
    -------
    fig, ax
    """
    k     = range(kin.r_cg.shape[0])[frame]   # IndexError when out of range
    geom  = _scene_geometry(
        mbd, kin, [k], triad_scale=triad_scale, bounds=bounds,
        show_triads=show_triads, show_springs=show_springs,
        forces=forces, show_forces=show_forces, force_scale=force_scale,
    )
    scene = _render(geom, 0, ax=ax, view=view)
    return scene.fig, scene.ax


def animate_multibody(
    mbd, kin, *, frame_step=1, fps=20, loop=False, save_path=None,
    ax=None, bounds=None, show_triads=False, show_springs=True,
    forces=None, show_forces=True, force_scale=0.2, triad_scale=None, view=None,
):
    """Animate a 3-D multibody system over the frames of *kin*.

    The first frame is drawn exactly as :func:`plot_snapshot` does; later
    frames only move the existing artists.  The figure is neither shown nor
    closed; keep a reference to the returned animation while it is displayed.

    Parameters
    ----------
    mbd, kin, ax, bounds, show_triads, show_springs, forces, show_forces,
    force_scale, triad_scale, view
        As in :func:`plot_snapshot`.
    frame_step : int, optional
        Draw every ``frame_step``-th frame of *kin*.
    fps : float, optional
        Frames per second for playback and saving.
    loop : bool, optional
        Repeat the animation when it reaches the last frame.
    save_path : str or Path, optional
        Write the animation here; ``.gif`` (Pillow) or ``.mp4`` (ffmpeg).

    Returns
    -------
    matplotlib.animation.FuncAnimation
    """
    if frame_step < 1:
        raise ValueError("frame_step must be >= 1.")
    suffix = None if save_path is None else Path(save_path).suffix.lower()
    if suffix not in (None, ".gif", ".mp4"):
        raise ValueError(f"save_path must end in .gif or .mp4, got {suffix!r}.")

    frames = np.arange(0, kin.r_cg.shape[0], frame_step)
    geom   = _scene_geometry(
        mbd, kin, frames, triad_scale=triad_scale, bounds=bounds,
        show_triads=show_triads, show_springs=show_springs,
        forces=forces, show_forces=show_forces, force_scale=force_scale,
    )
    scene  = _render(geom, 0, ax=ax, view=view)
    anim   = FuncAnimation(
        scene.fig, lambda k: _render(geom, k, scene=scene),
        frames=len(frames), interval=1000.0 / fps, blit=False, repeat=loop,
    )
    if suffix is not None:
        writer = PillowWriter(fps=fps) if suffix == ".gif" else FFMpegWriter(fps=fps)
        anim.save(save_path, writer=writer)
    return anim
