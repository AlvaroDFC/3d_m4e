"""forces_symbolic_3d.py

Symbolic 3D force layer: assembles per-body 6D wrenches from parsed force
definitions, the symbolic points cache, and the VT kinematics caches.

Wrench layout
-------------
Each per-body wrench is a ``(6, 1)`` :class:`sym.Matrix` in world-frame
coordinates, ordered ``[Fx, Fy, Fz, Mx, My, Mz]`` (force then moment,
both about the body CG).

Symbolic-opacity guarantee
--------------------------
This module **never** calls ``expand``, ``simplify``, ``trigsimp``, or any
other aggressive SymPy transform.  All expressions are assembled from:

* Direct sum / difference of :class:`sym.Matrix` objects (element-wise
  SymPy arithmetic — stays unevaluated unless both operands are numeric).
* Cross products via :meth:`sym.Matrix.cross` (returns a new
  :class:`sym.Matrix` without expansion).
* Scalar products of a SymPy scalar with a :class:`sym.Matrix` column
  (element-wise, stays symbolic).
* ``sym.sqrt`` applied to a SymPy scalar expression (kept unevaluated).

Force categories
----------------
CG
    Direct force + free moment applied at body CG, world frame.

PointsBD
    Force + optional free moment at a declared body point.
    Reduced to (force, moment-about-CG) via ``sym_force_reduction_at_point``.

TensionSpring
    Equal-and-opposite endpoint forces along the AB unit vector.
    Magnitude:  ``k * (L - L0)``  where ``L = ||r_B - r_A||``.
    Potential energy accumulated: ``0.5 * k * (L - L0)**2``.

TensionDamper
    Equal-and-opposite endpoint forces along the AB unit vector.
    Magnitude:  ``c * dL/dt``  where ``dL/dt = e_AB · (v_B - v_A)``.
    Requires a :class:`KinematicsRateCache3D` (pass via *rate_cache*).

TorsionSpring
    Pure moment on child and parent about the joint rotation axis.
    Torque on child:  ``-k * (theta - theta_eq)``  (restoring).
    Potential energy accumulated: ``0.5 * k * (theta - theta_eq)**2``.

TorsionDamper
    Pure moment on child and parent about the joint rotation axis.
    Torque on child:  ``-c * theta_dot``  (dissipative).

Gravity
    Body-CG force ``g_app[b-1] * m_b * g_vec`` applied to each declared body.
    No moment (gravity acts at CG).  Per-body mass is read from
    ``body_inertia`` passed to the builder.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, TYPE_CHECKING

import sympy as sym

if TYPE_CHECKING:
    from .velocity_transformation_3d import KinematicsCache3D, KinematicsRateCache3D
    from .joint_coordinate_3d import CoordBundle
    from .points_3d import SymbolicPointsCache3D, PointRecord3D
    from .force_definition_3d import ForcesDefinition3D

try:
    from ._velocity_transformation_helper import skew
except Exception:  # pragma: no cover
    from _velocity_transformation_helper import skew

try:
    from .points_3d import sym_force_reduction_at_point
except Exception:  # pragma: no cover
    from points_3d import sym_force_reduction_at_point


# ---------------------------------------------------------------------------
# Output container
# ---------------------------------------------------------------------------

@dataclass
class SymbolicForcesCache3D:
    """All symbolic per-body wrenches for a 3D multibody system.

    Produced by :func:`build_forces_symbolic` from a parsed
    :class:`~force_definition_3d.ForcesDefinition3D`.

    Attributes
    ----------
    wrench_by_category : dict[str, list[sym.Matrix]]
        Per-category wrench tables.  Each value is a list of length
        *NBodies*; entry ``wrench_by_category[cat][b-1]`` is the ``(6, 1)``
        world-frame wrench on body *b* from category *cat*.
        Categories present: ``"CG"``, ``"PointsBD"``, ``"TensionSpring"``,
        ``"TensionDamper"``, ``"TorsionSpring"``, ``"TorsionDamper"``,
        ``"Gravity"``.  A category key is absent when no elements of that
        type are defined.
    total_wrench : list[sym.Matrix]
        Sum over all categories; ``total_wrench[b-1]`` is the ``(6, 1)``
        net wrench on body *b*.
    spring_potential_energy : sym.Expr
        Total potential energy from all spring elements
        (tension springs + torsion springs).
    NBodies : int
        Number of bodies (excluding ground).
    """

    wrench_by_category: Dict[str, List[sym.Matrix]]
    total_wrench:       List[sym.Matrix]
    spring_potential_energy: Any   # sym.Expr (0 if no springs)
    NBodies: int


# ---------------------------------------------------------------------------
# Module-private helpers
# ---------------------------------------------------------------------------

def _zeros_wrenches(NBodies: int) -> List[sym.Matrix]:
    """Return a list of ``NBodies`` zero ``(6, 1)`` wrenches."""
    return [sym.zeros(6, 1) for _ in range(NBodies)]


def _wrench6(f: sym.Matrix, m: sym.Matrix) -> sym.Matrix:
    """Stack a (3,1) force and (3,1) moment into a (6,1) wrench."""
    f3 = sym.Matrix(f).reshape(3, 1)
    m3 = sym.Matrix(m).reshape(3, 1)
    return f3.col_join(m3)


def _get_point_rec(
    body_id: int,
    pt_idx:  int,
    sym_points: "SymbolicPointsCache3D",
    label:   str,
) -> "PointRecord3D":
    """Look up a :class:`PointRecord3D` by body id and point index."""
    if body_id == 0:
        pts = sym_points.ground_points
        if pt_idx >= len(pts):
            raise ValueError(
                f"{label}: ground point index {pt_idx} is out of range "
                f"(only {len(pts)} ground point(s) declared)."
            )
        return pts[pt_idx]
    else:
        if body_id not in sym_points.body_points:
            raise ValueError(
                f"{label}: body {body_id} has no declared points "
                "in Initial_Points[\"BD\"]."
            )
        pts = sym_points.body_points[body_id]
        if pt_idx >= len(pts):
            raise ValueError(
                f"{label}: body {body_id} has only {len(pts)} point(s); "
                f"pt_idx={pt_idx} is out of range."
            )
        return pts[pt_idx]


def _r_abs_mat(rec: "PointRecord3D") -> sym.Matrix:
    """Return the absolute position of *rec* as a ``(3, 1)`` :class:`sym.Matrix`."""
    return sym.Matrix(rec.r_abs).reshape(3, 1)


def _point_velocity(
    body_id:  int,
    rho_abs:  Any,
    rate_cache: "KinematicsRateCache3D",
) -> sym.Matrix:
    """World-frame velocity of a body-attached (or ground) point.

    Parameters
    ----------
    body_id : int
        0 = ground; 1..NBodies = moving body.
    rho_abs : sym.Matrix or sym.MatrixExpr
        CG-relative moment arm in the world frame.  Ignored for ground.
    rate_cache : KinematicsRateCache3D

    Returns
    -------
    sym.Matrix, shape ``(3, 1)``
    """
    if body_id == 0:
        return sym.zeros(3, 1)
    v_cg  = sym.Matrix(rate_cache.v_abs[body_id]).reshape(3, 1)
    omega = sym.Matrix(rate_cache.omega_abs[body_id]).reshape(3, 1)
    rho   = sym.Matrix(rho_abs).reshape(3, 1)
    # v_P = v_CG + omega × rho  (cross product: no expansion)
    return v_cg + omega.cross(rho)


# ---------------------------------------------------------------------------
# Per-category builders
# ---------------------------------------------------------------------------


def _build_cg_wrenches(
    forces_def: "ForcesDefinition3D",
    NBodies: int,
) -> List[sym.Matrix]:
    """Assemble per-body wrenches from CG force/moment definitions."""
    wrenches = _zeros_wrenches(NBodies)
    for d in forces_def.cg_forces:
        b = d.body_id
        f = sym.Matrix(list(d.force_vec)).reshape(3, 1)
        m = sym.Matrix(list(d.moment_vec)).reshape(3, 1)
        wrenches[b - 1] = wrenches[b - 1] + _wrench6(f, m)
    return wrenches


def _build_point_force_wrenches(
    forces_def: "ForcesDefinition3D",
    sym_points: "SymbolicPointsCache3D",
    NBodies: int,
) -> List[sym.Matrix]:
    """Assemble per-body wrenches from body-point force definitions."""
    wrenches = _zeros_wrenches(NBodies)
    for i, d in enumerate(forces_def.point_forces):
        label = f"PointsBD[{i}]"
        rec = _get_point_rec(d.body_id, d.point_idx, sym_points, label)
        f_vec = sym.Matrix(list(d.force_vec)).reshape(3, 1)
        m_free = sym.Matrix(list(d.moment_vec)).reshape(3, 1)
        # Force reduction: f_eq = f_vec, m_from_r = rho × f
        f_eq, m_from_r = sym_force_reduction_at_point(rec, f_vec)
        w = _wrench6(f_eq, m_from_r + m_free)
        wrenches[d.body_id - 1] = wrenches[d.body_id - 1] + w
    return wrenches


def _build_tension_spring_wrenches(
    forces_def: "ForcesDefinition3D",
    sym_points: "SymbolicPointsCache3D",
    NBodies: int,
) -> "tuple[List[sym.Matrix], Any]":
    """Assemble per-body wrenches and potential energy from tension springs.

    Returns
    -------
    wrenches : list[sym.Matrix]
    pe_total : sym.Expr
    """
    wrenches = _zeros_wrenches(NBodies)
    pe_total: Any = sym.Integer(0)

    for i, d in enumerate(forces_def.tension_springs):
        label = f"TensionSpring[{i}]"
        rec_a = _get_point_rec(d.body_id_a, d.pt_idx_a, sym_points, label + " A")
        rec_b = _get_point_rec(d.body_id_b, d.pt_idx_b, sym_points, label + " B")

        r_a = _r_abs_mat(rec_a)
        r_b = _r_abs_mat(rec_b)
        d_vec = r_b - r_a                          # (3,1) sym.Matrix

        # Scalar length (involves sqrt; kept unevaluated by SymPy)
        L_sq  = (d_vec.T * d_vec)[0, 0]
        L     = sym.sqrt(L_sq)

        # Unit vector A→B
        e_vec = d_vec / L                          # (3,1) sym.Matrix

        # Constitutive law (may be symbolic)
        F_mag = d.k * (L - d.L0)                  # scalar sym.Expr

        # Equal-and-opposite forces along e_vec
        f_on_b = -F_mag * e_vec                   # force on body B
        f_on_a =  F_mag * e_vec                   # force on body A

        # Reduce to wrenches and accumulate
        if d.body_id_a != 0:
            f_eq_a, m_eq_a = sym_force_reduction_at_point(rec_a, f_on_a)
            wrenches[d.body_id_a - 1] = (
                wrenches[d.body_id_a - 1] + _wrench6(f_eq_a, m_eq_a)
            )
        if d.body_id_b != 0:
            f_eq_b, m_eq_b = sym_force_reduction_at_point(rec_b, f_on_b)
            wrenches[d.body_id_b - 1] = (
                wrenches[d.body_id_b - 1] + _wrench6(f_eq_b, m_eq_b)
            )

        # Potential energy: 0.5 * k * (L - L0)^2
        pe_total = pe_total + sym.Rational(1, 2) * d.k * (L - d.L0) ** 2

    return wrenches, pe_total


def _build_tension_damper_wrenches(
    forces_def: "ForcesDefinition3D",
    sym_points: "SymbolicPointsCache3D",
    rate_cache: "KinematicsRateCache3D",
    NBodies: int,
) -> List[sym.Matrix]:
    """Assemble per-body wrenches from tension damper definitions.

    Requires *rate_cache* (``v_abs``, ``omega_abs``) for point velocity
    computation.  The extension rate is

        ``dL/dt = e_AB · (v_B - v_A)``

    where ``e_AB = (r_B - r_A) / ||r_B - r_A||`` and ``v_P`` is the
    world-frame velocity of the attachment point on body *b*:

        ``v_P = v_cg(b) + omega(b) × rho_P``
    """
    wrenches = _zeros_wrenches(NBodies)

    for i, d in enumerate(forces_def.tension_dampers):
        label = f"TensionDamper[{i}]"
        rec_a = _get_point_rec(d.body_id_a, d.pt_idx_a, sym_points, label + " A")
        rec_b = _get_point_rec(d.body_id_b, d.pt_idx_b, sym_points, label + " B")

        r_a = _r_abs_mat(rec_a)
        r_b = _r_abs_mat(rec_b)
        d_vec = r_b - r_a
        L_sq  = (d_vec.T * d_vec)[0, 0]
        L     = sym.sqrt(L_sq)
        e_vec = d_vec / L

        # Point velocities
        rho_a = sym.zeros(3, 1) if rec_a.rho_abs is None else rec_a.rho_abs
        rho_b = sym.zeros(3, 1) if rec_b.rho_abs is None else rec_b.rho_abs
        v_a = _point_velocity(d.body_id_a, rho_a, rate_cache)
        v_b = _point_velocity(d.body_id_b, rho_b, rate_cache)

        # Extension rate: scalar
        v_rel = v_b - v_a
        L_dot  = (e_vec.T * v_rel)[0, 0]

        F_mag  = d.c * L_dot
        f_on_b = -F_mag * e_vec
        f_on_a =  F_mag * e_vec

        if d.body_id_a != 0:
            f_eq_a, m_eq_a = sym_force_reduction_at_point(rec_a, f_on_a)
            wrenches[d.body_id_a - 1] = (
                wrenches[d.body_id_a - 1] + _wrench6(f_eq_a, m_eq_a)
            )
        if d.body_id_b != 0:
            f_eq_b, m_eq_b = sym_force_reduction_at_point(rec_b, f_on_b)
            wrenches[d.body_id_b - 1] = (
                wrenches[d.body_id_b - 1] + _wrench6(f_eq_b, m_eq_b)
            )

    return wrenches


def _build_torsion_spring_wrenches(
    forces_def: "ForcesDefinition3D",
    coords: "CoordBundle",
    pos_cache: "KinematicsCache3D",
    NBodies: int,
) -> "tuple[List[sym.Matrix], Any]":
    """Assemble per-body wrenches and PE from torsion spring definitions.

    The joint rotation axis ``u`` is taken from
    ``pos_cache.U[joint_idx]`` (already expressed in the world frame).
    For a revolute joint this is a ``(3, 1)`` unit vector.

    Torque convention (restoring on child):
        ``tau = -k * (theta - theta_eq)``
        ``M_child  = tau * u``
        ``M_parent = -tau * u``  (reaction, Newton 3rd law)

    Ground (parent = 0) receives no wrench.
    """
    wrenches = _zeros_wrenches(NBodies)
    pe_total: Any = sym.Integer(0)

    for d in forces_def.torsion_springs:
        pj    = coords.per_joint[d.joint_idx]
        child  = pj["child"]
        parent = pj["parent"]

        # Joint angle (single symbol for R joint)
        theta = pj["q_user"][0]

        # Joint axis in global frame
        u = sym.Matrix(pos_cache.U[d.joint_idx]).reshape(3, 1)

        # Restoring torque scalar (positive = child twisted past eq)
        tau = -d.k * (theta - d.theta_eq)

        M_child = tau * u                 # (3,1) sym.Matrix

        # Child body wrench (pure moment, no force)
        wrenches[child - 1] = (
            wrenches[child - 1] + _wrench6(sym.zeros(3, 1), M_child)
        )
        # Parent reaction (skip ground)
        if parent != 0:
            wrenches[parent - 1] = (
                wrenches[parent - 1] + _wrench6(sym.zeros(3, 1), -M_child)
            )

        pe_total = pe_total + sym.Rational(1, 2) * d.k * (theta - d.theta_eq) ** 2

    return wrenches, pe_total


def _build_torsion_damper_wrenches(
    forces_def: "ForcesDefinition3D",
    coords: "CoordBundle",
    pos_cache: "KinematicsCache3D",
    NBodies: int,
) -> List[sym.Matrix]:
    """Assemble per-body wrenches from torsion damper definitions.

    Torque convention (dissipative on child):
        ``tau = -c * theta_dot``
        ``M_child  = tau * u``
        ``M_parent = -tau * u``
    """
    wrenches = _zeros_wrenches(NBodies)

    for d in forces_def.torsion_dampers:
        pj    = coords.per_joint[d.joint_idx]
        child  = pj["child"]
        parent = pj["parent"]

        # Joint angular speed (single symbol for R joint)
        theta_dot = pj["qd_user"][0]

        u = sym.Matrix(pos_cache.U[d.joint_idx]).reshape(3, 1)

        tau     = -d.c * theta_dot
        M_child = tau * u

        wrenches[child - 1] = (
            wrenches[child - 1] + _wrench6(sym.zeros(3, 1), M_child)
        )
        if parent != 0:
            wrenches[parent - 1] = (
                wrenches[parent - 1] + _wrench6(sym.zeros(3, 1), -M_child)
            )

    return wrenches


def _build_gravity_wrenches(
    forces_def: "ForcesDefinition3D",
    pos_cache: "KinematicsCache3D",
    NBodies: int,
    body_inertia: dict = {},
) -> List[sym.Matrix]:
    """Assemble per-body wrenches from the gravity definition.

    Gravity acts at each body CG, so there is no moment contribution.
    ``F_gravity(b) = g_app[b-1] * mass(b) * g_vec``
    where ``mass(b)`` is taken from *body_inertia*.
    """
    wrenches = _zeros_wrenches(NBodies)
    gd = forces_def.gravity
    if gd is None:
        return wrenches

    g_col = sym.Matrix(list(gd.g_vec)).reshape(3, 1)
    zeros3 = sym.zeros(3, 1)

    for b in range(1, NBodies + 1):
        mass    = sym.sympify(body_inertia.get(b, {}).get("mass", 0))
        g_app_b = gd.g_app[b - 1] if b - 1 < len(gd.g_app) else 1.0
        if mass == 0:
            continue
        f_grav = g_app_b * mass * g_col
        wrenches[b - 1] = wrenches[b - 1] + _wrench6(f_grav, zeros3)

    return wrenches


# ---------------------------------------------------------------------------
# Main builder
# ---------------------------------------------------------------------------

def build_forces_symbolic(
    forces_def: "ForcesDefinition3D",
    sym_points: Optional["SymbolicPointsCache3D"],
    coords: "CoordBundle",
    pos_cache: "KinematicsCache3D",
    rate_cache: Optional["KinematicsRateCache3D"],
    NBodies: int,
    body_inertia: dict = {},
) -> SymbolicForcesCache3D:
    """Build the full symbolic 3D force cache.

    Iterates over each force category in *forces_def*, assembles per-body
    ``(6, 1)`` wrenches in world coordinates, sums them into
    ``total_wrench``, and accumulates potential-energy terms.

    Parameters
    ----------
    forces_def : ForcesDefinition3D
        Parsed force definitions.
    sym_points : SymbolicPointsCache3D or None
        Symbolic point cache.  Required when *forces_def* contains
        ``PointsBD``, ``TensionSpring``, or ``TensionDamper`` entries.
        May be *None* if only ``CG``, ``Gravity``, torsion, or empty.
    coords : CoordBundle
        Symbolic coordinates (used for torsion element joint angles/speeds).
    pos_cache : KinematicsCache3D
        Position-level symbolic kinematics (used for joint axes ``U``).
    rate_cache : KinematicsRateCache3D or None
        Rate kinematics (needed only for ``TensionDamper`` entries).
    NBodies : int
        Number of bodies (excluding ground).
    body_inertia : dict, optional
        Per-body inertia dict ``{body_id: {"mass": ..., "J": ...}}``.  Mass
        values are used to assemble the gravity wrench.

    Returns
    -------
    SymbolicForcesCache3D

    Raises
    ------
    ValueError
        If ``PointsBD`` or tension elements are requested but *sym_points*
        is *None*, or if *rate_cache* is *None* when tension dampers are
        present.
    """
    # --- Guard: sym_points required when point-based forces exist ----------
    needs_sym_pts = bool(
        forces_def.point_forces
        or forces_def.tension_springs
        or forces_def.tension_dampers
    )
    if needs_sym_pts and sym_points is None:
        raise ValueError(
            "build_forces_symbolic: sym_points is None but the force "
            "definition includes PointsBD or tension elements that require "
            "declared Initial_Points."
        )

    if forces_def.tension_dampers and rate_cache is None:
        raise ValueError(
            "build_forces_symbolic: rate_cache is None but TensionDamper "
            "elements are defined.  Pass a KinematicsRateCache3D."
        )

    # --- Per-category assembly --------------------------------------------
    wrench_by_category: Dict[str, List[sym.Matrix]] = {}
    pe_total: Any = sym.Integer(0)

    # CG
    if forces_def.cg_forces:
        wrench_by_category["CG"] = _build_cg_wrenches(forces_def, NBodies)

    # PointsBD
    if forces_def.point_forces:
        wrench_by_category["PointsBD"] = _build_point_force_wrenches(
            forces_def, sym_points, NBodies
        )

    # TensionSpring
    if forces_def.tension_springs:
        w_ts, pe_ts = _build_tension_spring_wrenches(
            forces_def, sym_points, NBodies
        )
        wrench_by_category["TensionSpring"] = w_ts
        pe_total = pe_total + pe_ts

    # TensionDamper
    if forces_def.tension_dampers:
        wrench_by_category["TensionDamper"] = _build_tension_damper_wrenches(
            forces_def, sym_points, rate_cache, NBodies
        )

    # TorsionSpring
    if forces_def.torsion_springs:
        w_ss, pe_ss = _build_torsion_spring_wrenches(
            forces_def, coords, pos_cache, NBodies
        )
        wrench_by_category["TorsionSpring"] = w_ss
        pe_total = pe_total + pe_ss

    # TorsionDamper
    if forces_def.torsion_dampers:
        wrench_by_category["TorsionDamper"] = _build_torsion_damper_wrenches(
            forces_def, coords, pos_cache, NBodies
        )

    # Gravity
    if forces_def.gravity is not None:
        wrench_by_category["Gravity"] = _build_gravity_wrenches(
            forces_def, pos_cache, NBodies, body_inertia=body_inertia
        )

    # --- Total wrench (sum across categories) ----------------------------
    total_wrench: List[sym.Matrix] = [sym.zeros(6, 1) for _ in range(NBodies)]
    for cat_wrenches in wrench_by_category.values():
        for b_idx in range(NBodies):
            total_wrench[b_idx] = total_wrench[b_idx] + cat_wrenches[b_idx]

    return SymbolicForcesCache3D(
        wrench_by_category=wrench_by_category,
        total_wrench=total_wrench,
        spring_potential_energy=pe_total,
        NBodies=NBodies,
    )
