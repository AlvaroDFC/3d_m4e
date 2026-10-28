"""kinematics_runtime_3d.py

Postprocessing helper: CG, joint, ground-point and declared body-point
(``Initial_Points["BD"]``) kinematics from an already-integrated solution
or from any array of states.

Not part of the ODE right-hand side; call after
:meth:`MbdSystem3D.integrate` (mirrors ``compute_energy_3d`` in
``mass_runtime_3d.py``).  Does **not** require ``body_inertia`` — only
geometry / kinematics are used, so this works on any constructed
``MbdSystem3D`` regardless of whether mass/force data was declared.

The jitted kernel and the lambdified point evaluators are cached on the system
(``MbdSystem3D._runtime_cache``); the kernel is rebuilt only when the body or
point parameter values change.
"""

from __future__ import annotations

from typing import Callable, NamedTuple, Optional, TYPE_CHECKING

import numpy as np

try:
    import jax
    import jax.numpy as jnp
    jax.config.update("jax_enable_x64", True)
    _JAX_AVAILABLE = True
except ImportError:  # pragma: no cover
    _JAX_AVAILABLE = False

if not _JAX_AVAILABLE:
    raise ImportError("JAX is required for kinematics_runtime_3d.")

if TYPE_CHECKING:
    from .mbd_system_3d import MbdSystem3D


# ---------------------------------------------------------------------------
# Output container
# ---------------------------------------------------------------------------

class KinematicsResult(NamedTuple):
    """World-frame kinematics of every state of an integrated solution.

    Produced by :func:`compute_kinematics_3d` (called via
    :meth:`MbdSystem3D.compute_kinematics`).  All arrays are plain
    ``numpy`` (already pulled off the JAX device).

    Attributes
    ----------
    ts : ndarray, shape ``(n_steps,)``
        Time points; mirrors ``sol.ts``.
    r_cg : ndarray, shape ``(n_steps, NBodies, 3)``
        World-frame CG (translation) positions.
    R_cg : ndarray, shape ``(n_steps, NBodies, 3, 3)``
        World-frame CG rotation matrices (unambiguous source of
        orientation regardless of joint type).
    euler_cg : ndarray, shape ``(n_steps, NBodies, 3)``
        Convenience Euler-angle view of ``R_cg``: intrinsic Z-Y-X
        (yaw-pitch-roll) Tait-Bryan angles ``[roll, pitch, yaw]`` in
        radians.  Not gimbal-lock safe near pitch = +/-90 deg; use
        ``R_cg`` directly for anything requiring robustness.
    v_cg : ndarray, shape ``(n_steps, NBodies, 3)`` or None
        World-frame CG linear velocity.
    omega_cg : ndarray, shape ``(n_steps, NBodies, 3)`` or None
        World-frame CG angular velocity.
    r_pts : ndarray, shape ``(n_steps, n_body_pts, 3)``
        World-frame positions of every declared body point
        (``Initial_Points["BD"]``), stacked ``(body_id ASC, point_idx ASC)``.
        ``n_body_pts`` is 0 when no body points are declared.
    v_pts : ndarray, shape ``(n_steps, n_body_pts, 3)`` or None
        World-frame velocities of the same body points.
    point_body_ids : tuple[int, ...]
        Owning body id for each row of ``r_pts`` / ``v_pts``.
    pt_body_slices : dict[int, slice]
        Slice into ``r_pts`` / ``v_pts`` for each body id.
    r_joint : ndarray, shape ``(n_steps, NJoints, 3)``
        World-frame joint location, in the joint declaration order of
        ``MbdSystem3D.joint_system.joints``: the parent-attached point
        ``r_parent + A_parent @ parent_cg_to_joint``.  For P/C joints the
        child-attached point differs by the slider travel.
    r_gr : ndarray, shape ``(n_gr, 3)``
        World-frame ground points (``Initial_Points["GR"]``).  Constant over
        the solution, so it has no time axis.

    The velocity fields are *None* when computed with
    ``with_velocities=False``.
    """
    ts:       "np.ndarray"
    r_cg:     "np.ndarray"
    R_cg:     "np.ndarray"
    euler_cg: "np.ndarray"
    v_cg:     Optional["np.ndarray"]
    omega_cg: Optional["np.ndarray"]
    r_pts:    "np.ndarray"
    v_pts:    Optional["np.ndarray"]
    point_body_ids: tuple
    pt_body_slices: dict
    r_joint:  "np.ndarray"
    r_gr:     "np.ndarray"


# ---------------------------------------------------------------------------
# Euler-angle convenience view
# ---------------------------------------------------------------------------

def _rotmat_to_euler_xyz(R: "jnp.ndarray") -> "jnp.ndarray":
    """Extract intrinsic Z-Y-X (yaw-pitch-roll) Euler angles from *R*.

    Assumes ``R = Rz(yaw) @ Ry(pitch) @ Rx(roll)``.  Returns
    ``[roll, pitch, yaw]`` in radians.  Convenience view only — breaks
    down (gimbal lock) at ``pitch = +/-90 deg``; ``R`` itself is always
    the robust representation.
    """
    pitch = -jnp.arcsin(jnp.clip(R[2, 0], -1.0, 1.0))
    roll  = jnp.arctan2(R[2, 1], R[2, 2])
    yaw   = jnp.arctan2(R[1, 0], R[0, 0])
    return jnp.stack([roll, pitch, yaw])


# ---------------------------------------------------------------------------
# Cached evaluators (rebuilt only when their inputs change)
# ---------------------------------------------------------------------------

class _PointFns(NamedTuple):
    body_ids:       tuple   # owning body id per flattened body point
    body_fns:       list    # lambdified r_local, one per body point
    pt_body_slices: dict
    gr_fns:         list    # lambdified r_local of the ground points


class _KinKernel(NamedTuple):
    param_key: tuple        # body / point parameter blocks the kernel was built for
    fn:        Callable     # jitted + vmapped: states (n, W) -> kinematics arrays
    r_gr:      "np.ndarray"


def _get_point_fns(mbd: "MbdSystem3D") -> _PointFns:
    """Lambdified ``r_local`` evaluators; they only depend on the symbolic points."""
    cached = mbd._runtime_cache.get("kinematics_point_fns")
    if cached is not None:
        return cached
    try:
        from .points_3d import _flatten_body_points, _make_r_local_fn   # noqa: PLC0415
    except Exception:  # pragma: no cover
        from points_3d import _flatten_body_points, _make_r_local_fn

    if mbd.sym_points is None:
        cached = _PointFns((), [], {}, [])
    else:
        body_syms   = list(mbd.body_data_sym.values())
        points_syms = list(mbd.points_sym.values())
        # Same flattening / ordering as the force evaluators (points_3d.py).
        body_ids, body_fns, slices = _flatten_body_points(
            mbd.sym_points, body_syms, points_syms,
        )
        gr_fns = [
            _make_r_local_fn(rec.r_local, body_syms, points_syms)
            for rec in mbd.sym_points.ground_points
        ]
        cached = _PointFns(tuple(body_ids), body_fns, slices, gr_fns)
    mbd._runtime_cache["kinematics_point_fns"] = cached
    return cached


def _get_kernel(mbd: "MbdSystem3D", mainNumVars, with_velocities: bool) -> _KinKernel:
    """Jitted, batched kinematics kernel; rebuilt only if body/point parameters change."""
    try:
        from .runtime_context_3d import build_runtime_context   # noqa: PLC0415
    except Exception:  # pragma: no cover
        from runtime_context_3d import build_runtime_context

    ctx    = build_runtime_context(mbd, mainNumVars)
    key    = (ctx.bp_np.tobytes(), ctx.pp_np.tobytes())
    slot   = ("kinematics_kernel", with_velocities)
    cached = mbd._runtime_cache.get(slot)
    if cached is not None and cached.param_key == key:
        return cached

    try:
        from .velocity_transformation_3d import (   # noqa: PLC0415
            build_cache_jax,
            build_rate_cache_jax,
        )
    except Exception:  # pragma: no cover
        from velocity_transformation_3d import build_cache_jax, build_rate_cache_jax

    pf = _get_point_fns(mbd)
    NB, NJ, n_qi    = ctx.NB, mbd.NJoints, ctx.n_qi
    pos_kw, rate_kw = ctx.pos_cache_kwargs, ctx.rate_topo_kwargs
    body_ids_0idx   = tuple(b - 1 for b in pf.body_ids)
    n_body_pts      = len(body_ids_0idx)

    def _eval(fns):
        if not fns:
            return np.zeros((0, 3), dtype=float)
        return np.stack([fn(*ctx.bp_np, *ctx.pp_np) for fn in fns])

    r_locals_jax = jnp.asarray(_eval(pf.body_fns), dtype=jnp.float64)   # (n_body_pts, 3)
    r_gr         = _eval(pf.gr_fns)                                      # constant, world frame

    def _kin_at(y):
        q_int = y[:n_qi]
        A_abs, r_abs, rJ, U, _ = build_cache_jax(q_int, **pos_kw)

        r_cg    = jnp.stack([r_abs[b + 1].ravel() for b in range(NB)])  # (NB,3)
        R_cg    = jnp.stack([A_abs[b + 1]          for b in range(NB)])  # (NB,3,3)
        euler   = jax.vmap(_rotmat_to_euler_xyz)(R_cg)                   # (NB,3)
        r_joint = jnp.stack([rJ[j].ravel()         for j in range(NJ)])  # (NJ,3)

        rhos = [A_abs[b0 + 1] @ r_locals_jax[i] for i, b0 in enumerate(body_ids_0idx)]
        if n_body_pts:
            r_pts = jnp.stack(
                [r_abs[b0 + 1].ravel() + rho for b0, rho in zip(body_ids_0idx, rhos)]
            )
        else:
            r_pts = jnp.zeros((0, 3), dtype=jnp.float64)

        if not with_velocities:
            return r_cg, R_cg, euler, r_joint, r_pts, None, None, None

        qd = y[n_qi:]
        omega_abs, v_abs, _, _ = build_rate_cache_jax(
            q_int, qd, A_abs=A_abs, r_abs=r_abs, rJ=rJ, U=U, **rate_kw,
        )
        v_cg  = jnp.stack([v_abs[b + 1].ravel()     for b in range(NB)])  # (NB,3)
        omega = jnp.stack([omega_abs[b + 1].ravel() for b in range(NB)])  # (NB,3)
        if n_body_pts:
            v_pts = jnp.stack([
                v_abs[b0 + 1].ravel() + jnp.cross(omega_abs[b0 + 1].ravel(), rho)
                for b0, rho in zip(body_ids_0idx, rhos)
            ])
        else:
            v_pts = jnp.zeros((0, 3), dtype=jnp.float64)

        return r_cg, R_cg, euler, r_joint, r_pts, v_cg, omega, v_pts

    kern = _KinKernel(key, jax.jit(jax.vmap(_kin_at)), r_gr)
    mbd._runtime_cache[slot] = kern
    return kern


def _as_internal_states(mbd: "MbdSystem3D", ys, with_velocities: bool) -> "np.ndarray":
    """Return state rows in the internal layout, mapping from the user layout if needed."""
    ys = np.atleast_2d(np.asarray(ys, dtype=float))
    n_qi, n_qu, n_qd = mbd.total_cfg_dof, mbd.total_user_dof, mbd.total_dof
    width = ys.shape[1]

    if width in (n_qi, n_qi + n_qd):
        out = ys   # also the user layout when no Euler S/F joint is present
    elif width in (n_qu, n_qu + n_qd):
        q_int = np.stack([mbd.map_q_user_to_q_int(row[:n_qu]) for row in ys])
        out   = np.hstack([q_int, ys[:, n_qu:]])
    else:
        raise ValueError(
            f"State width {width} not recognised: expected [q_int, qd] ({n_qi + n_qd}) "
            f"or [q_user, qd] ({n_qu + n_qd}); without velocities also q_int ({n_qi}) "
            f"or q_user ({n_qu})."
        )
    if with_velocities and out.shape[1] != n_qi + n_qd:
        raise ValueError(
            f"with_velocities=True needs the qd columns (width {n_qi + n_qd} or "
            f"{n_qu + n_qd}); got {width}."
        )
    return out


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def compute_kinematics_3d(
    mbd: "MbdSystem3D", sol, mainNumVars, *, with_velocities: bool = True,
) -> KinematicsResult:
    """Compute kinematics at every saved state of an integrated solution.

    Thin wrapper around :func:`compute_kinematics_states` using ``sol.ts``
    and ``sol.ys`` (result of :meth:`MbdSystem3D.integrate`).
    """
    return compute_kinematics_states(
        mbd, sol.ts, sol.ys, mainNumVars, with_velocities=with_velocities,
    )


def compute_kinematics_states(
    mbd: "MbdSystem3D", ts, ys, mainNumVars, *, with_velocities: bool = True,
) -> KinematicsResult:
    """Compute CG, joint, ground and body-point kinematics for a set of states.

    Postprocessing operation — not part of the ODE right-hand side.
    Re-evaluates position (and optionally rate) kinematics
    (``build_cache_jax`` / ``build_rate_cache_jax``) at every state in *ys*.
    Works on any states, so a snapshot of an initial configuration does not
    need an integration.

    Parameters
    ----------
    mbd : MbdSystem3D
    ts : array_like, shape ``(n_steps,)``
        Time of each state; copied into the result.
    ys : array_like, shape ``(n_steps, W)``
        State rows: the internal layout ``[q_int, qd]`` (e.g. ``sol.ys``) or
        the user layout ``[q_user, qd]``, which is mapped through
        :meth:`MbdSystem3D.map_q_user_to_q_int`.  The layout is told apart by
        the row width.  The ``qd`` columns may be omitted when
        ``with_velocities`` is *False*.  Without Euler S/F joints both layouts
        coincide and rows are used as given (quaternions must be unit-norm).
    mainNumVars : array_like, shape ``(len(mainSymVars),)``
        The same user-facing vector passed to :meth:`MbdSystem3D.integrate`;
        supplies the constant body/force/point parameter blocks (assumed
        unchanged for all states).
    with_velocities : bool, optional
        If *False*, skip the rate kinematics; ``v_cg``, ``omega_cg`` and
        ``v_pts`` are then *None*.  Default *True*.

    Returns
    -------
    KinematicsResult
    """
    ts = np.atleast_1d(np.array(ts, dtype=float))
    ys = _as_internal_states(mbd, ys, with_velocities)
    if ts.shape[0] != ys.shape[0]:
        raise ValueError(f"ts has {ts.shape[0]} entries but ys has {ys.shape[0]} rows.")

    kern = _get_kernel(mbd, mainNumVars, with_velocities)
    pf   = _get_point_fns(mbd)

    (r_cg_all, R_cg_all, euler_all, r_joint_all, r_pts_all,
     v_cg_all, omega_all, v_pts_all) = kern.fn(jnp.asarray(ys, dtype=jnp.float64))

    def _to_np(a):
        return None if a is None else np.array(a)

    return KinematicsResult(
        ts=ts,
        r_cg=np.array(r_cg_all),
        R_cg=np.array(R_cg_all),
        euler_cg=np.array(euler_all),
        v_cg=_to_np(v_cg_all),
        omega_cg=_to_np(omega_all),
        r_pts=np.array(r_pts_all),
        v_pts=_to_np(v_pts_all),
        point_body_ids=pf.body_ids,
        pt_body_slices=dict(pf.pt_body_slices),
        r_joint=np.array(r_joint_all),
        r_gr=kern.r_gr.copy(),
    )


# ---------------------------------------------------------------------------
# CSV export
# ---------------------------------------------------------------------------

def save_trajectory_csv(
    kin: KinematicsResult,
    filepath,
    *,
    body_names=None,
) -> None:
    """Write CG position and linear-velocity trajectories to a CSV file.

    Columns: ``time, body_index, body_name, pos_x, pos_y, pos_z,
    vel_x, vel_y, vel_z``.  One row per (time-step, body) pair, ordered
    by time then body index (0-based).

    Parameters
    ----------
    kin : KinematicsResult
        Output of :func:`compute_kinematics_3d` (or
        :meth:`MbdSystem3D.compute_kinematics`).
    filepath : str or path-like
        Destination CSV file path.
    body_names : list[str] or None, optional
        Label for each body, length ``NBodies`` in 1-based order.
        Defaults to ``["body1", "body2", …]`` when *None*.
    """
    import csv  # noqa: PLC0415

    if kin.v_cg is None:
        raise ValueError(
            "save_trajectory_csv requires velocities; compute the kinematics "
            "with with_velocities=True."
        )
    n_steps, NB, _ = kin.r_cg.shape
    if body_names is None:
        body_names = [f"body{b}" for b in range(1, NB + 1)]
    if len(body_names) != NB:
        raise ValueError(
            f"body_names has {len(body_names)} entries but system has {NB} bodies."
        )

    with open(filepath, "w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow([
            "time", "body_index", "body_name",
            "pos_x", "pos_y", "pos_z",
            "vel_x", "vel_y", "vel_z",
        ])
        for i in range(n_steps):
            t = kin.ts[i]
            for b in range(NB):
                px, py, pz = kin.r_cg[i, b]
                vx, vy, vz = kin.v_cg[i, b]
                writer.writerow([
                    f"{t:.8f}",
                    b,
                    body_names[b],
                    f"{px:.8f}", f"{py:.8f}", f"{pz:.8f}",
                    f"{vx:.8f}", f"{vy:.8f}", f"{vz:.8f}",
                ])
