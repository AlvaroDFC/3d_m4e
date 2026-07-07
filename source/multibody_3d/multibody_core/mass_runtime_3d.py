"""mass_runtime_3d.py

JAX runtime evaluators for the system mass matrix and EOM kernel.

Public factories
----------------
``make_mass_evaluator_mainint``
    Standalone mass-matrix evaluator ``f(mainNumVars_int) -> MassEvalResult``.
    Useful for one-off checks (positive-definiteness, modal analysis).

``make_eom_evaluator_mainint``
    Combined EOM kernel ``f(mainNumVars_int) -> EomKernelResult``.
    Computes B, Bdot, M_body, and M in **a single kinematics pass** —
    preferred in integrator loops where all four quantities are needed.

``compute_energy_3d``
    Postprocessing helper ``f(mbd, sol, mainNumVars) -> EnergyResult``.
    Computes kinetic/potential energy time series from an already-integrated
    solution.  Not part of the integration loop itself — call after
    :meth:`MbdSystem3D.integrate`.

Design
------
* Both factories follow the two-branch (constant / parameterised geometry)
  pattern of ``make_B_evaluator_mainint``.
* ``J_body`` is a numeric ``jnp.ndarray`` frozen in the JIT closure.  The
  world-frame transform ``A @ J_body @ A.T`` is evaluated inside the JIT
  kernel from the runtime rotation matrix ``A_abs[b]``, keeping it opaque
  at the JAX trace level.
* ``mass`` is resolved to a Python ``float`` at construction time.
  Symbolic masses referencing ``body_data_sym`` are not yet supported.
"""

from __future__ import annotations

from typing import Any, Dict, List, NamedTuple, TYPE_CHECKING

import numpy as np
import sympy as sym

try:
    import jax
    import jax.numpy as jnp
    jax.config.update("jax_enable_x64", True)
    _JAX_AVAILABLE = True
except ImportError:  # pragma: no cover
    _JAX_AVAILABLE = False

if not _JAX_AVAILABLE:
    raise ImportError("JAX is required for mass_runtime_3d.")

if TYPE_CHECKING:
    from .mbd_system_3d import MbdSystem3D


# ---------------------------------------------------------------------------
# Output container
# ---------------------------------------------------------------------------

class MassEvalResult(NamedTuple):
    """Result of the standalone mass-matrix evaluator.

    Attributes
    ----------
    M : jnp.ndarray, shape ``(total_dof, total_dof)``
        Symmetric positive-definite generalised mass matrix
        ``B^T M_body B`` evaluated at the current configuration.
    """
    M: "jnp.ndarray"


class EomKernelResult(NamedTuple):
    """Kinematic and inertia quantities computed in one integrator timestep.

    All four arrays share a single ``build_cache_jax`` call.

    Attributes
    ----------
    B : jnp.ndarray, shape ``(6*NBodies, total_dof)``
        Velocity-transformation matrix at the current configuration.
    Bdot : jnp.ndarray, shape ``(6*NBodies, total_dof)``
        Time derivative of B at the current configuration and velocity.
    M_body : jnp.ndarray, shape ``(6*NBodies, 6*NBodies)``
        Block-diagonal world-frame spatial mass matrix.
    M : jnp.ndarray, shape ``(total_dof, total_dof)``
        Generalised mass matrix ``B^T M_body B``.
    """
    B:      "jnp.ndarray"
    Bdot:   "jnp.ndarray"
    M_body: "jnp.ndarray"
    M:      "jnp.ndarray"


class EnergyResult(NamedTuple):
    """Kinetic / potential energy time series from an integrated solution.

    Produced by :func:`compute_energy_3d` (called via
    :meth:`MbdSystem3D.compute_energy`).  All arrays are plain ``numpy``
    (already pulled off the JAX device).

    Attributes
    ----------
    ts : ndarray, shape ``(n_steps,)``
        Time points; mirrors ``sol.ts``.
    KE, PE, E_total : ndarray, shape ``(n_steps,)``
        System-level kinetic, potential, and total mechanical energy.
    KE_body, PE_body : ndarray, shape ``(n_steps, NBodies)``
        Per-body kinetic and potential energy.
    """
    ts:      "np.ndarray"
    KE:      "np.ndarray"
    PE:      "np.ndarray"
    E_total: "np.ndarray"
    KE_body: "np.ndarray"
    PE_body: "np.ndarray"


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def make_mass_evaluator_mainint(
    body_inertia: Dict[int, Dict[str, Any]],
    params,                # NumericModelParams
    slc_q_int:    slice,
    slc_qd_int:   slice,
    slc_body_int: slice,
    extractor,             # GeometryExtractor
    NBodies: int,
    body_sym_list: list = (),
) -> "callable":
    """Build a JAX mass-matrix evaluator that accepts ``mainNumVars_int``.

    Delegates to :func:`make_eom_evaluator_mainint` and returns only the
    ``M`` field, eliminating duplicated kinematics code.

    The returned callable has signature::

        mass_func(mainNumVars_int: array_like) -> MassEvalResult
    """
    eom_eval = make_eom_evaluator_mainint(
        body_inertia, params,
        slc_q_int, slc_qd_int, slc_body_int,
        extractor, NBodies, body_sym_list,
    )

    def _mass_only(mainNumVars_int):
        return MassEvalResult(M=eom_eval(mainNumVars_int).M)

    return _mass_only


# ---------------------------------------------------------------------------
# Combined EOM kernel factory
# ---------------------------------------------------------------------------

def make_eom_evaluator_mainint(
    body_inertia: Dict[int, Dict[str, Any]],
    params,                # NumericModelParams
    slc_q_int:    slice,
    slc_qd_int:   slice,
    slc_body_int: slice,
    extractor,             # GeometryExtractor
    NBodies: int,
    body_sym_list: list = (),
) -> "callable":
    """Build a combined JAX EOM kernel evaluator accepting ``mainNumVars_int``.

    Returns B, Bdot, M_body, and M in **a single kinematics pass**, avoiding
    the three separate ``build_cache_jax`` calls that occur when
    ``B_func``, ``Bdot_func``, and ``mass_func`` are called independently
    in an integrator loop.

    The returned callable has signature::

        eom_func(mainNumVars_int: array_like) -> EomKernelResult

    Parameters are identical to :func:`make_mass_evaluator_mainint` except
    that ``slc_qd_int`` is required (needed to extract *qd* for Bdot).
    """
    # ── Deferred imports ──────────────────────────────────────────────────
    try:
        from .velocity_transformation_3d import (   # noqa: PLC0415
            build_cache_jax,
            build_rate_cache_jax,
            _convert_geometry_to_jax,
            _np_geom_to_jax,
        )
        from ._velocity_transformation_helper import (   # noqa: PLC0415
            _assemble_B_recursive_jax,
            _assemble_Bdot_recursive_jax,
        )
    except Exception:  # pragma: no cover
        from velocity_transformation_3d import (
            build_cache_jax,
            build_rate_cache_jax,
            _convert_geometry_to_jax,
            _np_geom_to_jax,
        )
        from _velocity_transformation_helper import (
            _assemble_B_recursive_jax,
            _assemble_Bdot_recursive_jax,
        )

    NB        = NBodies
    total_dof = params.total_dof

    # ── Slice offsets (static Python ints, not JAX-traced) ───────────────
    qi_start = slc_q_int.start
    n_qi     = slc_q_int.stop  - slc_q_int.start
    qd_start = slc_qd_int.start
    n_qd     = slc_qd_int.stop - slc_qd_int.start
    bp_start = slc_body_int.start
    n_bp     = slc_body_int.stop - slc_body_int.start

    # ── Validate and freeze mass / inertia to JAX arrays ─────────────────
    _masses_jax:   List["jnp.ndarray"] = []
    _J_bodies_jax: List["jnp.ndarray"] = []

    for b in range(1, NB + 1):
        if b not in body_inertia:
            raise KeyError(f"body_inertia missing entry for body {b}.")
        entry  = body_inertia[b]
        m_expr = sym.sympify(entry["mass"])
        if m_expr.free_symbols:
            raise ValueError(
                f"body_inertia[{b}]['mass'] contains free symbols "
                f"{m_expr.free_symbols}. Symbolic masses are not supported."
            )
        _masses_jax.append(jnp.asarray(float(m_expr), dtype=jnp.float64))
        J_raw = np.asarray(entry["J"], dtype=float)
        if J_raw.shape != (3, 3):
            raise ValueError(
                f"body_inertia[{b}]['J'] must be (3, 3), got {J_raw.shape}."
            )
        if not np.all(np.isfinite(J_raw)):
            raise ValueError(
                f"body_inertia[{b}]['J'] contains non-finite values."
            )
        _J_bodies_jax.append(jnp.asarray(J_raw, dtype=jnp.float64))

    # ── Static topology (closed over in JIT) ─────────────────────────────
    _n_bodies    = params.n_bodies
    _n_joints    = params.n_joints
    _parent      = params.parent
    _child       = params.child
    _codes       = params.code
    _cfg_slices  = params.cfg_slices
    _col_slices  = params.col_slices
    _body_paths  = params.body_paths
    _joint_paths = params.joint_paths

    # ── Inner JAX kernel ─────────────────────────────────────────────────

    def _compute_eom(q_int, qd, p2j, j2c, u, u1, u2) -> "EomKernelResult":
        """Single-pass EOM kernel (pure JAX, called inside @jax.jit)."""
        # 1. Position-level kinematics — one call gives A_abs, r_abs, rJ, U
        A_abs, r_abs, rJ, U, _ = build_cache_jax(
            q_int,
            n_bodies=_n_bodies, n_joints=_n_joints,
            parent=_parent, child=_child, codes=_codes,
            cfg_slices=_cfg_slices, p2j=p2j, j2c=j2c,
            u=u, u1=u1, u2=u2,
        )

        # 2. Rate-level kinematics — reuses position cache, no second pass
        omega_abs, v_abs, vJ, Udot = build_rate_cache_jax(
            q_int, qd,
            A_abs=A_abs, r_abs=r_abs, rJ=rJ, U=U,
            n_bodies=_n_bodies, n_joints=_n_joints,
            parent=_parent, child=_child, codes=_codes,
            col_slices=_col_slices,
        )

        # 3. B and Bdot from shared position / rate cache
        B    = _assemble_B_recursive_jax(
            r_abs, rJ, U,
            n_bodies=_n_bodies, total_dof=total_dof,
            codes=_codes, body_paths=_body_paths,
            joint_paths=_joint_paths, col_slices=_col_slices,
        )
        Bdot = _assemble_Bdot_recursive_jax(
            r_abs, rJ, U, v_abs, vJ, Udot,
            n_bodies=_n_bodies, total_dof=total_dof,
            codes=_codes, body_paths=_body_paths,
            joint_paths=_joint_paths, col_slices=_col_slices,
        )

        # 4. M_body from A_abs already in hand
        I3 = jnp.eye(3, dtype=jnp.float64)
        Z3 = jnp.zeros((3, 3), dtype=jnp.float64)
        blocks = []
        for b_idx in range(_n_bodies):
            m_b = _masses_jax[b_idx]
            J_b = _J_bodies_jax[b_idx]
            A_b = A_abs[b_idx + 1]
            J_world_b = A_b @ J_b @ A_b.T
            top = jnp.concatenate([m_b * I3, Z3], axis=1)
            bot = jnp.concatenate([Z3, J_world_b], axis=1)
            blocks.append(jnp.concatenate([top, bot], axis=0))
        M_body = jax.scipy.linalg.block_diag(*blocks)

        # 5. Generalised mass matrix
        M = B.T @ M_body @ B

        return EomKernelResult(B=B, Bdot=Bdot, M_body=M_body, M=M)

    # ── Two branches: constant vs parameterised geometry ─────────────────

    if not extractor.has_dynamic:
        kw   = _convert_geometry_to_jax(params)
        _p2j = kw["p2j"]
        _j2c = kw["j2c"]
        _u   = kw["u"]
        _u1  = kw["u1"]
        _u2  = kw["u2"]

        @jax.jit
        def _eval_eom(mainNumVars_int):
            v     = jnp.asarray(mainNumVars_int, dtype=jnp.float64)
            q_int = v[qi_start: qi_start + n_qi]
            qd    = v[qd_start: qd_start + n_qd]
            return _compute_eom(q_int, qd, _p2j, _j2c, _u, _u1, _u2)

        return _eval_eom

    else:
        @jax.jit
        def _jit_eom(q_int, qd, p2j, j2c, u, u1, u2):
            return _compute_eom(q_int, qd, p2j, j2c, u, u1, u2)

        def _eval_eom(mainNumVars_int):
            v    = np.asarray(mainNumVars_int, dtype=float)
            q    = jnp.asarray(v[qi_start: qi_start + n_qi], dtype=jnp.float64)
            qd   = jnp.asarray(v[qd_start: qd_start + n_qd], dtype=jnp.float64)
            bp   = v[bp_start: bp_start + n_bp]
            geom = extractor.evaluate(bp)
            return _jit_eom(q, qd, *_np_geom_to_jax(*geom))

        def _freeze(body_params_np):
            """Return a ``@jax.jit`` evaluator with geometry frozen for *body_params_np*.

            Fully JAX-traceable: the extractor call (requires concrete values)
            runs once here at freeze time rather than at every eval.
            """
            geom = extractor.evaluate(np.asarray(body_params_np, dtype=float))
            _p2j_f, _j2c_f, _u_f, _u1_f, _u2_f = _np_geom_to_jax(*geom)

            @jax.jit
            def _frozen(mainNumVars_int):
                v     = jnp.asarray(mainNumVars_int, dtype=jnp.float64)
                q_int = v[qi_start: qi_start + n_qi]
                qd    = v[qd_start: qd_start + n_qd]
                return _compute_eom(q_int, qd, _p2j_f, _j2c_f, _u_f, _u1_f, _u2_f)

            return _frozen

        _eval_eom.freeze = _freeze
        return _eval_eom


# ---------------------------------------------------------------------------
# Energy postprocessing (not part of the integration loop)
# ---------------------------------------------------------------------------

_BCACHE_KEYS = frozenset({
    "n_bodies", "n_joints", "parent", "child", "codes",
    "cfg_slices", "p2j", "j2c", "u", "u1", "u2",
})


def compute_energy_3d(mbd: "MbdSystem3D", sol, mainNumVars) -> EnergyResult:
    """Compute kinetic / potential energy time series from an integrated solution.

    This is a postprocessing operation, not part of the ODE right-hand side:
    it re-evaluates ``mbd.eom_func`` and the position kinematics at every
    saved state in *sol* to recover generalised mass and CG positions, then
    combines them with gravity to obtain per-step, per-body mechanical energy.

    Parameters
    ----------
    mbd : MbdSystem3D
        System with ``body_inertia`` declared (``eom_func`` must not be
        *None*).
    sol : diffrax.Solution
        Result of :meth:`MbdSystem3D.integrate` (or :func:`integrate_3d`).
        Uses ``sol.ts`` and ``sol.ys``.
    mainNumVars : array_like, shape ``(len(mainSymVars),)``
        The same user-facing vector passed to :meth:`MbdSystem3D.integrate`;
        supplies the constant body/force/point parameter blocks (assumed
        unchanged throughout the integration).

    Returns
    -------
    EnergyResult

    Raises
    ------
    RuntimeError
        If ``mbd.eom_func`` is *None* or ``mbd.body_inertia`` is empty.
    """
    if mbd.eom_func is None:
        raise RuntimeError(
            "compute_energy_3d() requires body_inertia to be declared.  "
            "eom_func is None."
        )
    if not mbd.body_inertia:
        raise RuntimeError(
            "compute_energy_3d() requires mbd.body_inertia to be non-empty."
        )

    # ── Deferred imports (mirrors make_eom_evaluator_mainint's pattern) ─────
    try:
        from .velocity_transformation_3d import (   # noqa: PLC0415
            build_cache_jax,
            _convert_geometry_to_jax,
            _convert_topology_to_jax,
            _np_geom_to_jax,
        )
    except Exception:  # pragma: no cover
        from velocity_transformation_3d import (
            build_cache_jax,
            _convert_geometry_to_jax,
            _convert_topology_to_jax,
            _np_geom_to_jax,
        )

    NB   = mbd.NBodies
    n_qi = mbd.total_cfg_dof

    # ── Gravity vector and per-body applied fraction ────────────────────────
    gd = mbd.forces_def.gravity if mbd.forces_def is not None else None
    g_vec_np = np.asarray(gd.g_vec, dtype=float) if gd is not None else np.zeros(3)
    g_app_np = np.asarray(gd.g_app, dtype=float) if gd is not None else np.ones(NB)
    g_vec_jax = jnp.asarray(g_vec_np, dtype=jnp.float64)

    # ── Per-body weight = g_app * mass, ordered by body id ──────────────────
    masses_np   = np.array([mbd.body_inertia[b]["mass"] for b in range(1, NB + 1)])
    weights_jax = jnp.asarray(g_app_np * masses_np, dtype=jnp.float64)

    # ── Constant parameter blocks (same ones used during integration) ──────
    arr   = mbd._validate_mainNumVars_shape(mainNumVars)
    mint0 = mbd._build_mainNumVars_int(arr)
    bp_np = np.array(mint0[mbd._slc_body_int],   dtype=float)
    fp_np = np.array(mint0[mbd._slc_force_int],  dtype=float)
    pp_np = np.array(mint0[mbd._slc_points_int], dtype=float)

    # ── Freeze EOM evaluator (bakes geometry into the JIT closure) ──────────
    eom_e = (
        mbd.eom_func.freeze(bp_np)
        if hasattr(mbd.eom_func, "freeze")
        else mbd.eom_func
    )

    # ── Kinematic cache kwargs (constant or parameterised geometry) ─────────
    if mbd._geom_extractor.has_dynamic:
        p2j_e, j2c_e, u_e, u1_e, u2_e = _np_geom_to_jax(
            *mbd._geom_extractor.evaluate(bp_np)
        )
        ckw = {k: v for k, v in _convert_topology_to_jax(mbd._numeric_params).items()
               if k in _BCACHE_KEYS}
        ckw.update(p2j=p2j_e, j2c=j2c_e, u=u_e, u1=u1_e, u2=u2_e)
    else:
        ckw = {k: v for k, v in _convert_geometry_to_jax(mbd._numeric_params).items()
               if k in _BCACHE_KEYS}

    cb_e = jnp.asarray(bp_np, dtype=jnp.float64)
    cf_e = jnp.asarray(fp_np, dtype=jnp.float64)
    cp_e = jnp.asarray(pp_np, dtype=jnp.float64)

    @jax.jit
    def _energy_at(y):
        q_int = y[:n_qi]
        qd    = y[n_qi:]
        # Kinetic energy: T = ½ qd^T M(q) qd  (M = B^T M_body B)
        mainint_y = jnp.concatenate([q_int, qd, cb_e, cf_e, cp_e])
        eom_res   = eom_e(mainint_y)
        M_gen     = eom_res.M
        KE        = 0.5 * (qd @ M_gen @ qd)
        # Potential energy: V = -Σ_b (g_app_b · m_b) · (g_vec · r_cg_b)
        _, r_abs, _, _, _ = build_cache_jax(q_int, **ckw)
        r_cg = jnp.stack([r_abs[b + 1].ravel() for b in range(NB)])  # (NB, 3)
        PE   = -jnp.dot(weights_jax, r_cg @ g_vec_jax)
        # Per-body energies
        B      = eom_res.B       # (6*NB, total_dof)
        M_body = eom_res.M_body  # (6*NB, 6*NB)
        KE_b   = jnp.stack([
            0.5 * (B[6*b:6*b+6, :] @ qd) @ M_body[6*b:6*b+6, 6*b:6*b+6] @ (B[6*b:6*b+6, :] @ qd)
            for b in range(NB)
        ])
        PE_b = -weights_jax * (r_cg @ g_vec_jax)   # (NB,)
        return KE, PE, KE_b, PE_b

    KE_arr, PE_arr, KE_body, PE_body = jax.vmap(_energy_at)(
        jnp.asarray(sol.ys, dtype=jnp.float64)
    )
    KE_arr  = np.array(KE_arr)
    PE_arr  = np.array(PE_arr)
    E_total = KE_arr + PE_arr
    KE_body = np.array(KE_body)   # (n_steps, NB)
    PE_body = np.array(PE_body)   # (n_steps, NB)
    ts      = np.array(sol.ts)

    return EnergyResult(
        ts=ts, KE=KE_arr, PE=PE_arr, E_total=E_total,
        KE_body=KE_body, PE_body=PE_body,
    )
