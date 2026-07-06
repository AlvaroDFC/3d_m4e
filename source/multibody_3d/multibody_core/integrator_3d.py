"""integrator_3d.py

Pure-JAX ODE right-hand side and diffrax integration for 3D multibody dynamics.

The entire ODE solve — including the adaptive step-size loop — is JIT-compiled by
``jax.jit`` and dispatched to whichever JAX backend is active (CPU / GPU / TPU).
No NumPy operations appear inside the integration loop.

Requires
--------
diffrax >= 0.7  (pip install diffrax)

State vector
------------
``y = [q_int (total_cfg_dof), qd (total_dof)]``.

For S / F joints ``len(q_int) != len(qd)`` because quaternion position uses 4
(or 7) components but only 3 (or 6) speed coordinates.  The mapping is handled
by :func:`_compute_q_int_dot_jax`.

Quaternion kinematics
---------------------
::

    q_dot = 0.5 * Omega(omega) @ q

where ``omega`` is the relative angular velocity in the parent frame
(``= qd[speed_slice]`` for S / F joints).

Equations of motion
-------------------
::

    M(q) qdd = B(q)^T (f_ext - gamma - M_body(q) Bdot(q,qd) qd)
    gamma[6b+3:6b+6] = omega_b x (J_world_b @ omega_b)  (gyroscopic)
"""

from __future__ import annotations

import numpy as np
import jax
import jax.numpy as jnp
from time import time

try:
    import diffrax
except ImportError as _e:
    diffrax          = None  # type: ignore[assignment]
    _DIFFRAX_ERR_MSG = str(_e)
else:
    _DIFFRAX_ERR_MSG = ""


# ---------------------------------------------------------------------------
# JAX quaternion helpers (fully JIT-traceable)
# ---------------------------------------------------------------------------

def _omega_matrix_jax(omega: jnp.ndarray) -> jnp.ndarray:
    """4x4 quaternion kinematics matrix — JAX version, JIT-traceable.

    ::

        Omega = [[ 0,  -ox, -oy, -oz],
                 [ox,   0,   oz, -oy],
                 [oy,  -oz,  0,   ox],
                 [oz,   oy,  -ox, 0  ]]

    Used as ``q_dot = 0.5 * Omega @ q``.
    """
    ox, oy, oz = omega[0], omega[1], omega[2]
    z = jnp.zeros_like(ox)
    return jnp.stack([
        jnp.stack([ z,  -ox,  -oy,  -oz]),
        jnp.stack([ox,    z,   oz,  -oy]),
        jnp.stack([oy,  -oz,    z,   ox]),
        jnp.stack([oz,   oy,  -ox,    z]),
    ])


def _normalize_q_int_jax(
    q_int: jnp.ndarray,
    per_joint: list,
) -> jnp.ndarray:
    """Return *q_int* with all S/F quaternion blocks unit-normalised.

    Built by concatenating per-joint segments.  The Python ``for`` loop
    unrolls at JAX trace time — the compiled kernel has no dynamic branching.
    """
    segments: list = []
    for pj in per_joint:
        code   = pj["type"].value
        int_sl = pj["int_slice"]
        if code in ("R", "P", "U", "C"):
            segments.append(q_int[int_sl])
        elif code == "S":
            e = q_int[int_sl]
            segments.append(e / jnp.linalg.norm(e))
        elif code == "F":
            cs    = int_sl.start
            trans = q_int[cs: cs + 3]
            quat  = q_int[cs + 3: cs + 7]
            segments.append(jnp.concatenate([trans, quat / jnp.linalg.norm(quat)]))
    return jnp.concatenate(segments)


def _compute_q_int_dot_jax(
    q_int:     jnp.ndarray,
    qd:        jnp.ndarray,
    per_joint: list,
) -> jnp.ndarray:
    """Map *qd* (total_dof) to *q_int_dot* (total_cfg_dof) — pure JAX.

    Built by concatenating per-joint segments:

    * **R / P / U / C** — direct copy segment (``qd[speed_slice]``).
    * **S** — 4-element quaternion kinematic segment.
    * **F** — 7-element segment: 3 translational (direct) + 4 quaternion.
    """
    segments: list = []
    for pj in per_joint:
        code     = pj["type"].value
        int_sl   = pj["int_slice"]
        speed_sl = pj["speed_slice"]
        if code in ("R", "P", "U", "C"):
            segments.append(qd[speed_sl])
        elif code == "S":
            e     = q_int[int_sl]
            omega = qd[speed_sl]
            segments.append(0.5 * _omega_matrix_jax(omega) @ e)
        elif code == "F":
            cs    = int_sl.start
            ss    = speed_sl.start
            e     = q_int[cs + 3: cs + 7]
            omega = qd[ss + 3: ss + 6]
            segments.append(jnp.concatenate([
                qd[ss: ss + 3],
                0.5 * _omega_matrix_jax(omega) @ e,
            ]))
    return jnp.concatenate(segments)


# ---------------------------------------------------------------------------
# diffrax vector field
# ---------------------------------------------------------------------------

def _make_eom_vector_field(mbd, const_body, const_force, const_points):
    """Build the diffrax vector-field closure ``(t, y, args) -> dy/dt``.

    Constant parameter blocks are frozen as JAX arrays inside the closure so
    they are baked into the compiled kernel as literals.  When ``eom_func`` or
    ``forces_func`` was built for parameterised geometry (body params enter
    kinematics symbolically), their ``.freeze()`` method is called here to
    pre-evaluate geometry once and return a fully ``@jax.jit``-traceable
    function compatible with diffrax.
    """
    NB        = mbd.NBodies
    n_qi      = mbd.total_cfg_dof
    total_dof = mbd.total_dof
    per_joint = mbd.coords.per_joint

    _cb = jnp.asarray(const_body,   dtype=jnp.float64)
    _cf = jnp.asarray(const_force,  dtype=jnp.float64)
    _cp = jnp.asarray(const_points, dtype=jnp.float64)

    # Freeze parameterised evaluators so they accept traced JAX arrays
    bp_np = np.asarray(const_body,   dtype=float)
    fp_np = np.asarray(const_force,  dtype=float)
    pp_np = np.asarray(const_points, dtype=float)

    _eom_func = (
        mbd.eom_func.freeze(bp_np)
        if hasattr(mbd.eom_func, "freeze")
        else mbd.eom_func
    )
    _forces_func = None
    if mbd.forces_func is not None:
        _forces_func = (
            mbd.forces_func.freeze(bp_np, fp_np, pp_np)
            if hasattr(mbd.forces_func, "freeze")
            else mbd.forces_func
        )

    # Static (Python-int) row ranges for extracting J_world_b from M_body
    _J_slices = [(6 * b + 3, 6 * b + 6) for b in range(NB)]

    def _vf(t, y, args):
        q_int = y[:n_qi]
        qd    = y[n_qi:]

        # Drift suppression: normalise quaternion blocks each evaluation
        q_int   = _normalize_q_int_jax(q_int, per_joint)
        mainint = jnp.concatenate([q_int, qd, _cb, _cf, _cp])

        # Single-pass kinematics via frozen (fully traceable) evaluator
        eom    = _eom_func(mainint)
        B      = eom.B       # (6*NB, total_dof)
        Bdot   = eom.Bdot    # (6*NB, total_dof)
        M_body = eom.M_body  # (6*NB, 6*NB)
        M      = eom.M       # (total_dof, total_dof)

        # External wrenches: (NB, 6) ravel -> (6*NB,)
        if _forces_func is not None:
            f_ext = _forces_func(mainint).total.ravel()
        else:
            f_ext = jnp.zeros(6 * NB, dtype=jnp.float64)

        # Gyroscopic correction: omega_b x (J_world_b @ omega_b)
        # Reshape B to body blocks; loop unrolls at trace time for small NB
        B_blocks     = B.reshape(NB, 6, total_dof)
        omega_bodies = B_blocks[:, 3:6, :] @ qd                       # (NB, 3)
        J_world_all  = jnp.stack(
            [M_body[r0:r1, r0:r1] for r0, r1 in _J_slices]
        )                                                                # (NB, 3, 3)
        J_omega  = jnp.einsum('bij,bj->bi', J_world_all, omega_bodies) # (NB, 3)
        gyro_ang = jnp.cross(omega_bodies, J_omega)                    # (NB, 3)
        # Assemble gyro as (NB, 6) = [zeros(3) | gyro_ang] then ravel
        gyro = (
            jnp.zeros((NB, 6), dtype=jnp.float64)
            .at[:, 3:].set(gyro_ang)
            .ravel()
        )                                                                # (6*NB,)

        # Solve EOM: M qdd = B^T (f - gyro - M_body Bdot qd)
        rhs = B.T @ (f_ext - gyro - M_body @ (Bdot @ qd))
        qdd = jnp.linalg.solve(M, rhs)

        q_int_dot = _compute_q_int_dot_jax(q_int, qd, per_joint)
        return jnp.concatenate([q_int_dot, qdd])

    return _vf


# ---------------------------------------------------------------------------
# Public integration factory
# ---------------------------------------------------------------------------

_SOLVERS = {
    "Dopri5":   lambda: diffrax.Dopri5()   if diffrax else None,
    "Dopri8":   lambda: diffrax.Dopri8()   if diffrax else None,
    "Tsit5":    lambda: diffrax.Tsit5()    if diffrax else None,
    "Kvaerno3": lambda: diffrax.Kvaerno3() if diffrax else None,
    "Kvaerno5": lambda: diffrax.Kvaerno5() if diffrax else None,
}


def integrate_3d(
    mbd,
    mainNumVars,
    *,
    tspan,
    dt=None,
    rtol:      float = 1e-8,
    atol:      float = 1e-8,
    algorithm: str   = "Dopri5",
    max_steps: int   = 100_000,
):
    """Numerically integrate the 3D multibody equations of motion via diffrax.

    The complete solve — adaptive step controller, RK stages, linear algebra —
    is compiled by ``jax.jit`` and runs entirely on the active JAX device
    (CPU / GPU / TPU).  No NumPy operations appear inside the integration loop.

    Parameters
    ----------
    mbd : MbdSystem3D
        System with ``body_inertia`` declared (``eom_func`` must not be *None*).
    mainNumVars : array_like, shape ``(len(mainSymVars),)``
        Initial user-facing variable vector
        ``[q_user, qd, body_params, force_params, point_params]``.
        Body / force / point parameter slices are constant throughout.
    tspan : float or (float, float)
        End time (start = 0) or ``(t_start, t_end)``.
    dt : float or None
        Output time step.  *None* saves only the final state.
    rtol, atol : float
        Adaptive step-size tolerances.
    algorithm : str
        diffrax solver name:

        * ``"Dopri5"``  — explicit, 4th-order, FSAL (default; equivalent to RK45)
        * ``"Dopri8"``  — explicit, 8th-order (equivalent to DOP853)
        * ``"Tsit5"``   — explicit, 5th-order Tsitouras
        * ``"Kvaerno3"`` / ``"Kvaerno5"`` — implicit, for stiff systems
    max_steps : int
        Maximum number of internal solver steps (default 100 000).

    Returns
    -------
    diffrax.Solution
        ``sol.ts``  — shape ``(n_out,)`` saved time points.
        ``sol.ys``  — shape ``(n_out, total_cfg_dof + total_dof)``; axis 0 is
        time so ``sol.ys[-1, :total_cfg_dof]`` is the final ``q_int``.
        ``sol.result`` — ``diffrax.RESULTS.successful`` on convergence.

    Raises
    ------
    ImportError
        If diffrax is not installed.
    RuntimeError
        If ``mbd.eom_func`` is *None*.
    """
    if diffrax is None:
        raise ImportError(
            "diffrax is required for JAX-native integration.  "
            "Install it with:  pip install diffrax\n"
            f"Original error: {_DIFFRAX_ERR_MSG}"
        )
    if mbd.eom_func is None:
        raise RuntimeError(
            "integrate_3d() requires body_inertia to be declared.  "
            "eom_func is None."
        )

    if isinstance(tspan, (int, float)):
        tspan = (0.0, float(tspan))
    else:
        tspan = (float(tspan[0]), float(tspan[1]))
    t0_span, t1_span = tspan

    # Build internal representation
    arr     = mbd._validate_mainNumVars_shape(mainNumVars)
    mainint = mbd._build_mainNumVars_int(arr)
    mainint_jax = jnp.asarray(mainint, dtype=jnp.float64)

    # Initial ODE state: [q_int, qd] as JAX arrays
    y0 = jnp.concatenate([
        mainint_jax[mbd._slc_q_int],
        mainint_jax[mbd._slc_qd_int],
    ])

    # Constant parameter blocks (stay on the device, baked into the closure)
    const_body   = np.array(mainint[mbd._slc_body_int],   dtype=float)
    const_force  = np.array(mainint[mbd._slc_force_int],  dtype=float)
    const_points = np.array(mainint[mbd._slc_points_int], dtype=float)

    # Build diffrax term and solver
    vf   = _make_eom_vector_field(mbd, const_body, const_force, const_points)
    term = diffrax.ODETerm(vf)

    if algorithm not in _SOLVERS:
        print(f"Unknown algorithm '{algorithm}'; falling back to Dopri5.")
    solver = _SOLVERS.get(algorithm, _SOLVERS["Dopri5"])()

    controller = diffrax.PIDController(rtol=rtol, atol=atol)
    dt0        = float(dt) if dt is not None else (t1_span - t0_span) * 0.01

    if dt is not None:
        # Ensure final time is included; small nudge avoids fp rounding exclusion
        t_eval = jnp.arange(t0_span, t1_span + dt * 0.5, float(dt),
                            dtype=jnp.float64)
        saveat = diffrax.SaveAt(ts=t_eval)
    else:
        saveat = diffrax.SaveAt(ts=jnp.array([t1_span], dtype=jnp.float64))

    # JIT-compile the full diffeqsolve; y0 is the only traced argument
    @jax.jit
    def _solve(y0):
        return diffrax.diffeqsolve(
            term, solver,
            t0=t0_span, t1=t1_span, dt0=dt0,
            y0=y0, saveat=saveat,
            stepsize_controller=controller,
            max_steps=max_steps,
        )

    # Warm up eom_func / forces_func JIT kernels before the timed run
    _ = mbd.eom_func(mainint_jax)
    if mbd.forces_func is not None:
        _ = mbd.forces_func(mainint_jax)

    t0_wall = time()
    sol     = _solve(y0)
    sol.ys.block_until_ready()          # block until GPU finishes
    elapsed = time() - t0_wall

    n_steps = int(sol.stats["num_steps"])
    success = bool(sol.result == diffrax.RESULTS.successful)
    print(
        f"Integrated {t1_span - t0_span:.2f} s  in {elapsed:.3f} s  "
        f"({n_steps} steps, {'success' if success else 'FAILED'})"
    )
    return sol

