"""forces_runtime_3d.py

Persistent JAX force evaluators for 3D multibody systems.

Mirrors the architecture of ``make_B_evaluator_mainint`` and
``make_points_evaluator_mainint`` in the existing codebase.  The built
callable accepts a full ``mainNumVars_int`` vector (same layout as every
other runtime evaluator in the system) and returns a :class:`ForcesEvalResult`
with per-category per-body 6-DOF wrenches and potential energy.

Design principles
-----------------
* **Single compiled path**: one ``@jax.jit`` closure per system, built once.
* **Reuse existing JAX backends**: ``build_cache_jax`` and (optionally)
  ``build_rate_cache_jax`` from ``velocity_transformation_3d`` provide body
  kinematics.  No separate kinematics engine is implemented here.
* **Constitutive lambdas built once**: every symbolic constitutive scalar
  (spring stiffness, damping coefficient, natural length, equilibrium angle,
  mass, force components) is lambdified over ``force_sym_list`` and
  ``points_sym_list`` at construction time.  At runtime the values are
  extracted from the appropriate slices of ``mainNumVars_int`` in numpy,
  then converted to a single JAX array (``const_vals``) that is passed into
  the JIT boundary.
* **r_local evaluated outside JIT**: body-point local-frame coordinates may
  depend on ``points_sym`` parameters.  They are lambdified once (same as in
  ``make_points_evaluator_mainint``) and evaluated in numpy before calling the
  JIT kernel.
* **Static-shape friendly**: every output array has shape ``(NBodies, 6)``
  regardless of which categories are present.  Absent categories return zeros.

Wrench layout
-------------
Each row of a ``(NBodies, 6)`` wrench array is
``[Fx, Fy, Fz, Mx, My, Mz]`` in the world frame about the body CG.

const_vals layout (per element, in declaration order)
------------------------------------------------------
* ``CG``           : [Fx, Fy, Fz, Mx, My, Mz] × n_cg
* ``PointsBD``     : [Fx, Fy, Fz, Mx_free, My_free, Mz_free] × n_pbd
* ``TensionSpring``: [k, L0] × n_ts
* ``TensionDamper``: [c] × n_td
* ``TorsionSpring``: [k, theta_eq] × n_ss
* ``TorsionDamper``: [c] × n_sd
* ``Gravity``      : [mass_b1, mass_b2, ...] per declared body
"""

from __future__ import annotations

from typing import Any, Dict, List, NamedTuple, Optional, Tuple

import numpy as np
import sympy as sym

try:
    import jax
    import jax.numpy as jnp
    jax.config.update("jax_enable_x64", True)
    _JAX_AVAILABLE = True
except ImportError:
    _JAX_AVAILABLE = False

if not _JAX_AVAILABLE:
    raise ImportError("JAX is required for forces_runtime_3d.")

try:
    from .force_definition_3d import ForcesDefinition3D
    from .points_3d import SymbolicPointsCache3D, PointRecord3D
    from .joint_coordinate_3d import CoordBundle
except Exception:  # pragma: no cover
    from force_definition_3d import ForcesDefinition3D
    from points_3d import SymbolicPointsCache3D, PointRecord3D
    from joint_coordinate_3d import CoordBundle


# ---------------------------------------------------------------------------
# Output container
# ---------------------------------------------------------------------------

class ForcesEvalResult(NamedTuple):
    """Per-body 6-DOF wrenches and spring potential energy.

    All wrench arrays have shape ``(NBodies, 6)`` in world-frame coordinates.
    Row ``b-1`` corresponds to body *b* (1-indexed).

    Attributes
    ----------
    cg : jnp.ndarray, (NBodies, 6)
    points_bd : jnp.ndarray, (NBodies, 6)
    tension_spring : jnp.ndarray, (NBodies, 6)
    tension_damper : jnp.ndarray, (NBodies, 6)
    torsion_spring : jnp.ndarray, (NBodies, 6)
    torsion_damper : jnp.ndarray, (NBodies, 6)
    gravity : jnp.ndarray, (NBodies, 6)
    total : jnp.ndarray, (NBodies, 6)
    spring_potential_energy : jnp.ndarray, scalar
    """
    cg:                    "jnp.ndarray"
    points_bd:             "jnp.ndarray"
    tension_spring:        "jnp.ndarray"
    tension_damper:        "jnp.ndarray"
    torsion_spring:        "jnp.ndarray"
    torsion_damper:        "jnp.ndarray"
    gravity:               "jnp.ndarray"
    total:                 "jnp.ndarray"
    spring_potential_energy: "jnp.ndarray"


# ---------------------------------------------------------------------------
# Build-time helpers
# ---------------------------------------------------------------------------

def _make_const_fn(expr: Any, force_sym_list: list, points_sym_list: list):
    """Return ``fn(fp, pp) -> float`` for a scalar constitutive expression.

    Parameters
    ----------
    expr : numeric or sympy.Expr
        The constitutive scalar (e.g. spring stiffness ``k``).
    force_sym_list : list[sym.Symbol]
        Ordered force parameter symbols.
    points_sym_list : list[sym.Symbol]
        Ordered point parameter symbols.

    Returns
    -------
    callable
        ``fn(fp: np.ndarray, pp: np.ndarray) -> float``
        where *fp* is the force-params slice and *pp* is the point-params slice.
    """
    e = sym.sympify(expr)
    if not e.free_symbols:
        # Pure numeric constant — ignore both parameter arrays.
        val = float(e)
        return lambda fp, pp, _v=val: _v

    all_syms = force_sym_list + points_sym_list
    raw = sym.lambdify(all_syms, e, modules="numpy")

    def fn(fp, pp, _f=raw, _nfp=len(force_sym_list), _npp=len(points_sym_list)):
        fp_vals = list(fp[:_nfp]) if _nfp else []
        pp_vals = list(pp[:_npp]) if _npp else []
        return float(_f(*fp_vals, *pp_vals))

    return fn


def _make_vec3_fn(vec: Tuple[Any, Any, Any], force_sym_list: list, points_sym_list: list):
    """Return ``fn(fp, pp) -> np.ndarray(3,)`` for a 3-component vector."""
    fns = [_make_const_fn(c, force_sym_list, points_sym_list) for c in vec]
    def fn(fp, pp, _fns=fns):
        return np.array([f(fp, pp) for f in _fns], dtype=float)
    return fn


def _get_point_record(
    body_id: int, pt_idx: int, sym_points: SymbolicPointsCache3D, label: str
) -> PointRecord3D:
    if body_id == 0:
        if pt_idx >= len(sym_points.ground_points):
            raise ValueError(
                f"{label}: ground pt_idx={pt_idx} out of range "
                f"({len(sym_points.ground_points)} declared)."
            )
        return sym_points.ground_points[pt_idx]
    if body_id not in sym_points.body_points:
        raise ValueError(f"{label}: no body_points for body {body_id}.")
    pts = sym_points.body_points[body_id]
    if pt_idx >= len(pts):
        raise ValueError(
            f"{label}: body {body_id} pt_idx={pt_idx} out of range "
            f"({len(pts)} declared)."
        )
    return pts[pt_idx]


def _make_r_local_fn(
    r_local_sym: "sym.Matrix",
    body_sym_list: list,
    points_sym_list: list,
):
    """Return ``fn(bp, pp) -> np.ndarray(3,)`` for a symbolic r_local.

    Lambdified over ``body_sym_list + points_sym_list`` so that r_local
    expressions that reference body-geometry symbols (e.g. link lengths)
    are resolved at runtime from the body-params slice.
    """
    all_syms = body_sym_list + points_sym_list
    components = [r_local_sym[i, 0] for i in range(3)]
    raw = sym.lambdify(all_syms, components, modules="numpy")
    n_bp = len(body_sym_list)
    n_pp = len(points_sym_list)
    def fn(bp, pp, _f=raw, _nbp=n_bp, _npp=n_pp):
        bp_vals = list(bp[:_nbp]) if _nbp else []
        pp_vals = list(pp[:_npp]) if _npp else []
        return np.asarray(_f(*bp_vals, *pp_vals), dtype=float).ravel()
    return fn


# ---------------------------------------------------------------------------
# Inner JAX assembly (called inside the JIT boundary)
# ---------------------------------------------------------------------------

def _cross3(a: "jnp.ndarray", b: "jnp.ndarray") -> "jnp.ndarray":
    """Cross product of two (3,) JAX arrays."""
    return jnp.cross(a, b)


def _wrench6(f3: "jnp.ndarray", m3: "jnp.ndarray") -> "jnp.ndarray":
    """Stack (3,) force + (3,) moment into a (6,) wrench."""
    return jnp.concatenate([f3, m3])


def _point_rabs_rho(
    body_id: int,
    rl_idx: int,
    r_locals: "jnp.ndarray",
    A_abs: list,
    r_abs: list,
) -> "Tuple[jnp.ndarray, jnp.ndarray]":
    """Return ``(r_abs_point, rho)`` for a point in JAX.

    Called inside the JIT boundary.  ``body_id`` is a Python constant.
    """
    r_loc = r_locals[rl_idx]       # (3,)
    if body_id == 0:
        return r_loc, jnp.zeros(3)
    A_b = A_abs[body_id]           # (3, 3)
    r_b = r_abs[body_id].ravel()   # (3,)
    rho = (A_b @ r_loc.reshape(3, 1)).ravel()
    return r_b + rho, rho


# ---------------------------------------------------------------------------
# Main factory
# ---------------------------------------------------------------------------

_RATE_KEYS = frozenset({"n_bodies", "n_joints", "parent", "child", "codes", "col_slices"})


def make_forces_evaluator_mainint(
    forces_def: ForcesDefinition3D,
    sym_points: Optional[SymbolicPointsCache3D],
    coords: CoordBundle,
    force_sym_list: list,     # ordered list of force symbols (force_points_sym.values())
    points_sym_list: list,    # ordered list of point symbols (points_sym.values())
    params,                   # NumericModelParams
    slc_q_int: slice,
    slc_qd_int: slice,
    slc_body_int: slice,
    slc_force_int: slice,
    slc_points_int: slice,
    extractor,                # GeometryExtractor
    NBodies: int,
    body_sym_list: list = (),  # ordered list of body_data_sym symbols
) -> "callable":
    """Build a persistent JAX force evaluator that accepts ``mainNumVars_int``.

    The returned callable has signature::

        forces_func(mainNumVars_int: array_like) -> ForcesEvalResult

    Parameters
    ----------
    forces_def : ForcesDefinition3D
        Parsed force definitions.
    sym_points : SymbolicPointsCache3D or None
        Symbolic points cache.  Required when *forces_def* contains point-
        based or tension elements.
    coords : CoordBundle
        Symbolic coordinate bundle (used for torsion element joint metadata).
    force_sym_list : list[sym.Symbol]
        Force parameter symbols in mainNumVars layout order.
    points_sym_list : list[sym.Symbol]
        Point parameter symbols in mainNumVars layout order.
    params : NumericModelParams
    slc_q_int, slc_qd_int, slc_body_int, slc_force_int, slc_points_int : slice
        Slices into ``mainNumVars_int``.
    extractor : GeometryExtractor
    NBodies : int

    Returns
    -------
    callable
        ``f(mainNumVars_int) -> ForcesEvalResult``
    """
    # --- deferred imports (avoids circular import at module load) ----------
    try:
        from .velocity_transformation_3d import (   # noqa: PLC0415
            build_cache_jax,
            build_rate_cache_jax,
            _convert_geometry_to_jax,
            _convert_topology_to_jax,
            _np_geom_to_jax,
        )
    except Exception:  # pragma: no cover
        from velocity_transformation_3d import (
            build_cache_jax,
            build_rate_cache_jax,
            _convert_geometry_to_jax,
            _convert_topology_to_jax,
            _np_geom_to_jax,
        )

    NB = NBodies

    # ── Slice offsets (static ints) ──────────────────────────────────────
    qi_start  = slc_q_int.start;    n_qi  = slc_q_int.stop  - slc_q_int.start
    qd_start  = slc_qd_int.start;   n_qd  = slc_qd_int.stop - slc_qd_int.start
    bp_start  = slc_body_int.start;  n_bp  = slc_body_int.stop - slc_body_int.start
    fp_start  = slc_force_int.start; n_fp  = slc_force_int.stop - slc_force_int.start
    pp_start  = slc_points_int.start; n_pp = slc_points_int.stop - slc_points_int.start

    # ── Build-time data collection ───────────────────────────────────────
    _const_fns: List       = []   # fn(fp, pp) -> float
    _r_local_fns: List     = []   # fn(*pp_vals) -> np.ndarray(3,)
    _r_local_body_ids: List = []  # int: body_id for each r_local entry (0 = ground)
    cv_cursor = 0                 # current const_vals cursor

    # — Per-category static metadata —
    # CG forces
    _cg_meta: List[Tuple[int, int]] = []   # (body_0idx, cv_start)
    for d in forces_def.cg_forces:
        _cg_meta.append((d.body_id - 1, cv_cursor))
        for c in list(d.force_vec) + list(d.moment_vec):
            _const_fns.append(_make_const_fn(c, force_sym_list, points_sym_list))
        cv_cursor += 6

    # PointsBD forces
    _pbd_meta: List[Tuple[int, int, int]] = []  # (body_0idx, cv_start, rl_idx)
    for i, d in enumerate(forces_def.point_forces):
        rl_idx = len(_r_local_fns)
        rec = _get_point_record(d.body_id, d.point_idx, sym_points,
                                f"PointsBD[{i}]")
        _r_local_fns.append(_make_r_local_fn(rec.r_local, list(body_sym_list), points_sym_list))
        _r_local_body_ids.append(d.body_id)
        _pbd_meta.append((d.body_id - 1, cv_cursor, rl_idx))
        for c in list(d.force_vec) + list(d.moment_vec):
            _const_fns.append(_make_const_fn(c, force_sym_list, points_sym_list))
        cv_cursor += 6

    # TensionSpring
    _ts_meta: List[Tuple[int, int, int, int, int]] = []
    # (body_a_id, body_b_id, rl_a_idx, rl_b_idx, cv_start)
    for i, d in enumerate(forces_def.tension_springs):
        rl_a = len(_r_local_fns)
        rec_a = _get_point_record(d.body_id_a, d.pt_idx_a, sym_points,
                                  f"TensionSpring[{i}] A")
        _r_local_fns.append(_make_r_local_fn(rec_a.r_local, list(body_sym_list), points_sym_list))
        _r_local_body_ids.append(d.body_id_a)

        rl_b = len(_r_local_fns)
        rec_b = _get_point_record(d.body_id_b, d.pt_idx_b, sym_points,
                                  f"TensionSpring[{i}] B")
        _r_local_fns.append(_make_r_local_fn(rec_b.r_local, list(body_sym_list), points_sym_list))
        _r_local_body_ids.append(d.body_id_b)

        _ts_meta.append((d.body_id_a, d.body_id_b, rl_a, rl_b, cv_cursor))
        _const_fns.append(_make_const_fn(d.k,  force_sym_list, points_sym_list))
        _const_fns.append(_make_const_fn(d.L0, force_sym_list, points_sym_list))
        cv_cursor += 2

    # TensionDamper
    _td_meta: List[Tuple[int, int, int, int, int]] = []
    for i, d in enumerate(forces_def.tension_dampers):
        rl_a = len(_r_local_fns)
        rec_a = _get_point_record(d.body_id_a, d.pt_idx_a, sym_points,
                                  f"TensionDamper[{i}] A")
        _r_local_fns.append(_make_r_local_fn(rec_a.r_local, list(body_sym_list), points_sym_list))
        _r_local_body_ids.append(d.body_id_a)

        rl_b = len(_r_local_fns)
        rec_b = _get_point_record(d.body_id_b, d.pt_idx_b, sym_points,
                                  f"TensionDamper[{i}] B")
        _r_local_fns.append(_make_r_local_fn(rec_b.r_local, list(body_sym_list), points_sym_list))
        _r_local_body_ids.append(d.body_id_b)

        _td_meta.append((d.body_id_a, d.body_id_b, rl_a, rl_b, cv_cursor))
        _const_fns.append(_make_const_fn(d.c, force_sym_list, points_sym_list))
        cv_cursor += 1

    # TorsionSpring: (child, parent, j_idx, q_col, qd_col, cv_start)
    _ss_meta: List[Tuple[int, int, int, int, int, int]] = []
    for d in forces_def.torsion_springs:
        pj = coords.per_joint[d.joint_idx]
        _ss_meta.append((
            pj["child"],
            pj["parent"],
            d.joint_idx,
            pj["int_slice"].start,   # index into q_int for this joint angle
            pj["speed_slice"].start,  # index into qd for this joint speed (unused in spring)
            cv_cursor,
        ))
        _const_fns.append(_make_const_fn(d.k,        force_sym_list, points_sym_list))
        _const_fns.append(_make_const_fn(d.theta_eq, force_sym_list, points_sym_list))
        cv_cursor += 2

    # TorsionDamper: (child, parent, j_idx, qd_col, cv_start)
    _sd_meta: List[Tuple[int, int, int, int, int]] = []
    for d in forces_def.torsion_dampers:
        pj = coords.per_joint[d.joint_idx]
        _sd_meta.append((
            pj["child"],
            pj["parent"],
            d.joint_idx,
            pj["speed_slice"].start,
            cv_cursor,
        ))
        _const_fns.append(_make_const_fn(d.c, force_sym_list, points_sym_list))
        cv_cursor += 1

    # Gravity: ordered list of (body_0idx, cv_start)
    _grav_g: Optional[Tuple[float, float, float]] = None
    _grav_meta: List[Tuple[int, int]] = []
    if forces_def.gravity is not None:
        gd = forces_def.gravity
        _grav_g = (float(sym.sympify(gd.g_vec[0])),
                   float(sym.sympify(gd.g_vec[1])),
                   float(sym.sympify(gd.g_vec[2])))
        for body_id in sorted(gd.mass.keys()):
            _grav_meta.append((body_id - 1, cv_cursor))
            _const_fns.append(_make_const_fn(gd.mass[body_id],
                                             force_sym_list, points_sym_list))
            cv_cursor += 1

    n_cv = cv_cursor
    n_rl = len(_r_local_fns)
    _needs_rate = bool(forces_def.tension_dampers)
    _needs_cache = bool(
        forces_def.point_forces
        or forces_def.tension_springs
        or forces_def.tension_dampers
        or forces_def.torsion_springs
        or forces_def.torsion_dampers
    )

    # Capture gravity vector as JAX array for JIT closure
    _g_jax = jnp.array(_grav_g if _grav_g is not None else [0., 0., 0.],
                        dtype=jnp.float64)

    # ── Build the inner JAX assembly function (traceable) ─────────────────
    def _assemble(q_int, qd, r_locals, const_vals, A_abs, r_abs, U,
                  omega_abs, v_abs):
        """Pure-JAX force assembly.  Called inside jit boundary."""
        z6  = jnp.zeros((NB, 6))
        z3  = jnp.zeros(3)
        pe  = jnp.zeros(())

        w_cg   = z6
        w_pbd  = z6
        w_ts   = z6
        w_td   = z6
        w_ss   = z6
        w_sd   = z6
        w_grav = z6

        # CG forces
        for b0, cv0 in _cg_meta:
            f6 = const_vals[cv0:cv0 + 6]
            w_cg = w_cg.at[b0].add(f6)

        # PointsBD forces
        for b0, cv0, rl_i in _pbd_meta:
            f3      = const_vals[cv0:cv0 + 3]
            m_free  = const_vals[cv0 + 3:cv0 + 6]
            _, rho  = _point_rabs_rho(b0 + 1, rl_i, r_locals, A_abs, r_abs)
            m_r     = _cross3(rho, f3)
            w_pbd   = w_pbd.at[b0].add(_wrench6(f3, m_r + m_free))

        # TensionSpring
        for ba, bb, rl_a, rl_b, cv0 in _ts_meta:
            k_val   = const_vals[cv0]
            L0_val  = const_vals[cv0 + 1]
            r_pa, rho_a = _point_rabs_rho(ba, rl_a, r_locals, A_abs, r_abs)
            r_pb, rho_b = _point_rabs_rho(bb, rl_b, r_locals, A_abs, r_abs)
            d_vec   = r_pb - r_pa
            L       = jnp.linalg.norm(d_vec)
            e       = d_vec / L
            F_mag   = k_val * (L - L0_val)
            f_on_a  =  F_mag * e
            f_on_b  = -F_mag * e
            if ba > 0:
                b0a = ba - 1
                m_a = _cross3(rho_a, f_on_a)
                w_ts = w_ts.at[b0a].add(_wrench6(f_on_a, m_a))
            if bb > 0:
                b0b = bb - 1
                m_b = _cross3(rho_b, f_on_b)
                w_ts = w_ts.at[b0b].add(_wrench6(f_on_b, m_b))
            pe = pe + jnp.array(0.5, dtype=jnp.float64) * k_val * (L - L0_val) ** 2

        # TensionDamper
        for ba, bb, rl_a, rl_b, cv0 in _td_meta:
            c_val   = const_vals[cv0]
            r_pa, rho_a = _point_rabs_rho(ba, rl_a, r_locals, A_abs, r_abs)
            r_pb, rho_b = _point_rabs_rho(bb, rl_b, r_locals, A_abs, r_abs)
            d_vec   = r_pb - r_pa
            L       = jnp.linalg.norm(d_vec)
            e       = d_vec / L
            # Point velocities
            if ba == 0:
                v_pa = z3
            else:
                v_cg_a  = v_abs[ba].ravel()
                om_a    = omega_abs[ba].ravel()
                v_pa    = v_cg_a + _cross3(om_a, rho_a)
            if bb == 0:
                v_pb = z3
            else:
                v_cg_b  = v_abs[bb].ravel()
                om_b    = omega_abs[bb].ravel()
                v_pb    = v_cg_b + _cross3(om_b, rho_b)
            L_dot   = jnp.dot(e, v_pb - v_pa)
            F_mag   = c_val * L_dot
            f_on_a  =  F_mag * e
            f_on_b  = -F_mag * e
            if ba > 0:
                b0a = ba - 1
                m_a = _cross3(rho_a, f_on_a)
                w_td = w_td.at[b0a].add(_wrench6(f_on_a, m_a))
            if bb > 0:
                b0b = bb - 1
                m_b = _cross3(rho_b, f_on_b)
                w_td = w_td.at[b0b].add(_wrench6(f_on_b, m_b))

        # TorsionSpring
        for child, parent, j_idx, q_col, _, cv0 in _ss_meta:
            k_val      = const_vals[cv0]
            theta_eq   = const_vals[cv0 + 1]
            theta      = q_int[q_col]
            u_j        = U[j_idx].ravel()          # (3,)
            tau        = -k_val * (theta - theta_eq)
            M_child    = tau * u_j
            w_ss = w_ss.at[child - 1].add(_wrench6(z3, M_child))
            if parent > 0:
                w_ss = w_ss.at[parent - 1].add(_wrench6(z3, -M_child))
            pe = pe + jnp.array(0.5, dtype=jnp.float64) * k_val * (theta - theta_eq) ** 2

        # TorsionDamper
        for child, parent, j_idx, qd_col, cv0 in _sd_meta:
            c_val      = const_vals[cv0]
            theta_dot  = qd[qd_col]
            u_j        = U[j_idx].ravel()
            tau        = -c_val * theta_dot
            M_child    = tau * u_j
            w_sd = w_sd.at[child - 1].add(_wrench6(z3, M_child))
            if parent > 0:
                w_sd = w_sd.at[parent - 1].add(_wrench6(z3, -M_child))

        # Gravity
        for b0, cv0 in _grav_meta:
            mass  = const_vals[cv0]
            f_g   = mass * _g_jax
            w_grav = w_grav.at[b0].add(_wrench6(f_g, z3))

        total = w_cg + w_pbd + w_ts + w_td + w_ss + w_sd + w_grav
        return ForcesEvalResult(
            cg=w_cg, points_bd=w_pbd,
            tension_spring=w_ts, tension_damper=w_td,
            torsion_spring=w_ss, torsion_damper=w_sd,
            gravity=w_grav, total=total,
            spring_potential_energy=pe,
        )

    # ── Numpy helper: evaluate all lambdas → const_vals ───────────────────
    def _eval_const_vals(fp, pp):
        if not _const_fns:
            return np.zeros(0, dtype=float)
        return np.array([fn(fp, pp) for fn in _const_fns], dtype=float)

    def _eval_r_locals(bp, pp):
        if not _r_local_fns:
            return np.zeros((0, 3), dtype=float)
        return np.stack([fn(bp, pp) for fn in _r_local_fns])   # (n_rl, 3)

    # ── Build JIT kernel (constant vs. parameterized geometry) ────────────
    if not extractor.has_dynamic:
        # ---- constant geometry: bake into JIT closure ----
        full_kw  = _convert_geometry_to_jax(params)
        cache_kw = {k: full_kw[k] for k in
                    ("n_bodies", "n_joints", "parent", "child", "codes",
                     "cfg_slices", "p2j", "j2c", "u", "u1", "u2")}
        rate_topo_kw = {k: full_kw[k] for k in _RATE_KEYS}

        if _needs_cache:
            @jax.jit
            def _jit_forces(q_int, qd, r_locals, const_vals):
                A_abs, r_abs, rJ, U, _ = build_cache_jax(q_int, **cache_kw)
                if _needs_rate:
                    omega_abs, v_abs, _, _ = build_rate_cache_jax(
                        q_int, qd,
                        A_abs=A_abs, r_abs=r_abs, rJ=rJ, U=U,
                        **rate_topo_kw,
                    )
                else:
                    omega_abs = v_abs = None
                return _assemble(q_int, qd, r_locals, const_vals,
                                 A_abs, r_abs, U, omega_abs, v_abs)
        else:
            # CG-only and/or Gravity — no kinematics needed
            @jax.jit
            def _jit_forces(q_int, qd, r_locals, const_vals):
                return _assemble(q_int, qd, r_locals, const_vals,
                                 None, None, None, None, None)

        def _evaluate(mainNumVars_int):
            v   = np.asarray(mainNumVars_int, dtype=float)
            q   = jnp.asarray(v[qi_start:qi_start + n_qi], dtype=jnp.float64)
            qd  = jnp.asarray(v[qd_start:qd_start + n_qd], dtype=jnp.float64)
            fp  = v[fp_start:fp_start + n_fp]
            pp  = v[pp_start:pp_start + n_pp]
            bp  = v[bp_start:bp_start + n_bp]
            cv  = jnp.asarray(_eval_const_vals(fp, pp), dtype=jnp.float64)
            rl  = jnp.asarray(_eval_r_locals(bp, pp), dtype=jnp.float64)
            return _jit_forces(q, qd, rl, cv)

    else:
        # ---- parameterized geometry ----
        topo_kw      = _convert_topology_to_jax(params)
        rate_topo_kw = {k: topo_kw[k] for k in _RATE_KEYS}
        cache_topo   = {k: topo_kw[k] for k in
                        ("n_bodies", "n_joints", "parent", "child", "codes",
                         "cfg_slices")}

        if _needs_cache:
            @jax.jit
            def _jit_forces(q_int, qd, r_locals, const_vals, p2j, j2c, u, u1, u2):
                A_abs, r_abs, rJ, U, _ = build_cache_jax(
                    q_int, p2j=p2j, j2c=j2c, u=u, u1=u1, u2=u2, **cache_topo,
                )
                if _needs_rate:
                    omega_abs, v_abs, _, _ = build_rate_cache_jax(
                        q_int, qd,
                        A_abs=A_abs, r_abs=r_abs, rJ=rJ, U=U,
                        **rate_topo_kw,
                    )
                else:
                    omega_abs = v_abs = None
                return _assemble(q_int, qd, r_locals, const_vals,
                                 A_abs, r_abs, U, omega_abs, v_abs)
        else:
            @jax.jit
            def _jit_forces(q_int, qd, r_locals, const_vals, p2j, j2c, u, u1, u2):
                return _assemble(q_int, qd, r_locals, const_vals,
                                 None, None, None, None, None)

        def _evaluate(mainNumVars_int):
            v   = np.asarray(mainNumVars_int, dtype=float)
            q   = jnp.asarray(v[qi_start:qi_start + n_qi], dtype=jnp.float64)
            qd  = jnp.asarray(v[qd_start:qd_start + n_qd], dtype=jnp.float64)
            bp  = v[bp_start:bp_start + n_bp]
            fp  = v[fp_start:fp_start + n_fp]
            pp  = v[pp_start:pp_start + n_pp]
            geom = extractor.evaluate(bp)
            cv  = jnp.asarray(_eval_const_vals(fp, pp), dtype=jnp.float64)
            rl  = jnp.asarray(_eval_r_locals(bp, pp), dtype=jnp.float64)
            return _jit_forces(q, qd, rl, cv, *_np_geom_to_jax(*geom))

    return _evaluate
