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

from typing import Any, Dict, List, NamedTuple

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
