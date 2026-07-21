"""kinematics_runtime_3d.py

Postprocessing helper: CG and declared body-point (``Initial_Points["BD"]``)
trajectories from an already-integrated solution.

Not part of the ODE right-hand side; call after
:meth:`MbdSystem3D.integrate` (mirrors ``compute_energy_3d`` in
``mass_runtime_3d.py``).  Does **not** require ``body_inertia`` — only
geometry / kinematics are used, so this works on any constructed
``MbdSystem3D`` regardless of whether mass/force data was declared.
"""

from __future__ import annotations

from typing import NamedTuple, TYPE_CHECKING

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
    """CG and body-point trajectories from an integrated solution.

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
    v_cg : ndarray, shape ``(n_steps, NBodies, 3)``
        World-frame CG linear velocity.
    omega_cg : ndarray, shape ``(n_steps, NBodies, 3)``
        World-frame CG angular velocity.
    r_pts : ndarray, shape ``(n_steps, n_body_pts, 3)``
        World-frame positions of every declared body point
        (``Initial_Points["BD"]``), stacked ``(body_id ASC, point_idx ASC)``.
        ``n_body_pts`` is 0 when no body points are declared.
    v_pts : ndarray, shape ``(n_steps, n_body_pts, 3)``
        World-frame velocities of the same body points.
    point_body_ids : tuple[int, ...]
        Owning body id for each row of ``r_pts`` / ``v_pts``.
    pt_body_slices : dict[int, slice]
        Slice into ``r_pts`` / ``v_pts`` for each body id (mirrors
        ``PointsRuntimeSpec.pt_body_slices``).
    """
    ts:       "np.ndarray"
    r_cg:     "np.ndarray"
    R_cg:     "np.ndarray"
    euler_cg: "np.ndarray"
    v_cg:     "np.ndarray"
    omega_cg: "np.ndarray"
    r_pts:    "np.ndarray"
    v_pts:    "np.ndarray"
    point_body_ids: tuple
    pt_body_slices: dict


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
# Main entry point
# ---------------------------------------------------------------------------

def compute_kinematics_3d(mbd: "MbdSystem3D", sol, mainNumVars) -> KinematicsResult:
    """Compute CG and declared body-point trajectories from an integrated solution.

    Postprocessing operation — not part of the ODE right-hand side.
    Re-evaluates position/rate kinematics (``build_cache_jax`` /
    ``build_rate_cache_jax``) at every saved state in *sol* to recover CG
    translation/rotation and linear/angular velocity for every body, plus
    world-frame position/velocity of every point declared in
    ``Initial_Points["BD"]``.

    Parameters
    ----------
    mbd : MbdSystem3D
    sol : diffrax.Solution
        Result of :meth:`MbdSystem3D.integrate` (or :func:`integrate_3d`).
        Uses ``sol.ts`` and ``sol.ys``.
    mainNumVars : array_like, shape ``(len(mainSymVars),)``
        The same user-facing vector passed to :meth:`MbdSystem3D.integrate`;
        supplies the constant body/force/point parameter blocks (assumed
        unchanged throughout the integration).

    Returns
    -------
    KinematicsResult
    """
    # ── Deferred imports ─────────────────────────────────────────────────
    try:
        from .velocity_transformation_3d import (   # noqa: PLC0415
            build_cache_jax,
            build_rate_cache_jax,
        )
        from .points_3d import _flatten_body_points                # noqa: PLC0415
        from .runtime_context_3d import build_runtime_context      # noqa: PLC0415
        from .integrator_3d import _normalize_q_int_jax            # noqa: PLC0415
    except Exception:  # pragma: no cover
        from velocity_transformation_3d import build_cache_jax, build_rate_cache_jax
        from points_3d import _flatten_body_points
        from runtime_context_3d import build_runtime_context
        from integrator_3d import _normalize_q_int_jax

    ctx = build_runtime_context(mbd, mainNumVars)
    NB        = ctx.NB
    n_qi      = ctx.n_qi
    per_joint = mbd.coords.per_joint

    # ── Flatten declared body points (body_id ASC, point_idx ASC) ───────
    # Shared with make_points_evaluator_mainint (points_3d.py) so both stay
    # in sync on ordering / lambdification without duplicating the logic.
    body_sym_list   = list(mbd.body_data_sym.values())
    points_sym_list = list(mbd.points_sym.values())

    if mbd.sym_points is not None:
        body_ids_flat, r_local_fns, pt_body_slices = _flatten_body_points(
            mbd.sym_points, body_sym_list, points_sym_list,
        )
    else:
        body_ids_flat, r_local_fns, pt_body_slices = [], [], {}

    n_body_pts = len(body_ids_flat)
    body_ids_0idx = tuple(b - 1 for b in body_ids_flat)

    if r_local_fns:
        r_locals_np = np.stack(
            [fn(*ctx.bp_np, *ctx.pp_np) for fn in r_local_fns]
        )   # (n_body_pts, 3)
    else:
        r_locals_np = np.zeros((0, 3), dtype=float)
    r_locals_jax = jnp.asarray(r_locals_np, dtype=jnp.float64)

    # ── Per-state JAX kernel ─────────────────────────────────────────────
    @jax.jit
    def _kin_at(y):
        q_int = y[:n_qi]
        qd    = y[n_qi:]
        # Suppress quaternion norm drift accumulated by the numeric integrator
        q_int = _normalize_q_int_jax(q_int, per_joint)
        A_abs, r_abs, rJ, U, _ = build_cache_jax(q_int, **ctx.pos_cache_kwargs)
        omega_abs, v_abs, _, _ = build_rate_cache_jax(
            q_int, qd, A_abs=A_abs, r_abs=r_abs, rJ=rJ, U=U,
            **ctx.rate_topo_kwargs,
        )

        r_cg  = jnp.stack([r_abs[b + 1].ravel()     for b in range(NB)])  # (NB,3)
        R_cg  = jnp.stack([A_abs[b + 1]              for b in range(NB)])  # (NB,3,3)
        v_cg  = jnp.stack([v_abs[b + 1].ravel()      for b in range(NB)])  # (NB,3)
        omega = jnp.stack([omega_abs[b + 1].ravel()  for b in range(NB)])  # (NB,3)
        euler = jax.vmap(_rotmat_to_euler_xyz)(R_cg)                       # (NB,3)

        if n_body_pts:
            r_pts_list = []
            v_pts_list = []
            for i, b0 in enumerate(body_ids_0idx):
                A_b  = A_abs[b0 + 1]
                r_b  = r_abs[b0 + 1].ravel()
                v_b  = v_abs[b0 + 1].ravel()
                om_b = omega_abs[b0 + 1].ravel()
                rl   = r_locals_jax[i]
                rho  = A_b @ rl
                r_pts_list.append(r_b + rho)
                v_pts_list.append(v_b + jnp.cross(om_b, rho))
            r_pts = jnp.stack(r_pts_list)
            v_pts = jnp.stack(v_pts_list)
        else:
            r_pts = jnp.zeros((0, 3), dtype=jnp.float64)
            v_pts = jnp.zeros((0, 3), dtype=jnp.float64)

        return r_cg, R_cg, euler, v_cg, omega, r_pts, v_pts

    (r_cg_all, R_cg_all, euler_all, v_cg_all, omega_all,
     r_pts_all, v_pts_all) = jax.vmap(_kin_at)(
        jnp.asarray(sol.ys, dtype=jnp.float64)
    )

    return KinematicsResult(
        ts=np.array(sol.ts),
        r_cg=np.array(r_cg_all),
        R_cg=np.array(R_cg_all),
        euler_cg=np.array(euler_all),
        v_cg=np.array(v_cg_all),
        omega_cg=np.array(omega_all),
        r_pts=np.array(r_pts_all),
        v_pts=np.array(v_pts_all),
        point_body_ids=tuple(body_ids_flat),
        pt_body_slices=pt_body_slices,
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
