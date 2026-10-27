"""points_3d.py

Symbolic 3-D points layer.

Parses ``Initial_Points`` and builds opaque symbolic position expressions for
every declared point, using a pre-built ``KinematicsCache3D`` as the sole
kinematic source.

Point-definition contract
-------------------------
``Initial_Points`` is a two-key dict:

``"GR"``
    List of world-frame 3-D points.  Each entry is ``[x, y, z]`` where
    components may be numeric or SymPy expressions.  Ground points are
    already in the global frame so no rotation is applied.

``"BD"``
    ``dict[int, list[[x, y, z]]]`` keyed by integer body id (1-based,
    excluding ground).  Coordinates are expressed in the body's own local
    frame at the CG origin.

Any free SymPy symbols that appear in the entries must also be registered in
the example module's ``points_sym`` dict so they participate in the canonical
variable bookkeeping of ``MbdSystem3D``.

Symbolic-structure guarantee
----------------------------
All rotation–translation products are built with ``MatMul(..., evaluate=False)``
so the expression tree remains opaque and mirrors the convention established in
``KinematicsCache3D``.  No ``expand``, ``simplify``, ``trigsimp``, or similar
aggressive transforms are called during construction.

Indexing
--------
* ``SymbolicPointsCache3D.ground_points``  — list indexed by declaration order.
* ``SymbolicPointsCache3D.body_points[b]`` — list indexed by declaration order
  within body *b*.
* ``SymbolicPointsCache3D.cg_points[b]``   — one synthetic CG record per body.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, TYPE_CHECKING

import numpy as np
import sympy as sym
from sympy import MatMul

if TYPE_CHECKING:  # avoids a runtime import cycle
    from .velocity_transformation_3d import KinematicsCache3D


# ---------------------------------------------------------------------------
# Public dataclasses
# ---------------------------------------------------------------------------

@dataclass(frozen=True, slots=True)
class PointRecord3D:
    """Symbolic representation of a single 3-D point.

    All matrix expressions are stored in non-expanded form.

    Attributes
    ----------
    body_id : int
        Owning body (0 = ground, 1..NBodies for body-attached points).
    point_idx : int
        Zero-based index within the body's (or ground's) declaration list.
        The sentinel value ``-1`` marks a synthetic CG point.
    r_local : sym.Matrix
        3×1 column vector in the body's local frame (CG origin) for body
        points, or in the world frame for ground points.  Components may be
        SymPy expressions.
    r_abs : sym.MatrixBase or sym.MatrixExpr
        Absolute (world-frame) position, 3×1.  Stored as an opaque
        ``MatAdd`` / ``MatMul`` expression.
    rho_abs : sym.MatrixBase or sym.MatrixExpr or None
        CG-relative moment arm in the global frame, 3×1::

            rho_abs = A_abs_body @ r_local    (body points)
            rho_abs = zeros(3,1)              (CG points)
            rho_abs = None                    (ground points)

        Stored without expansion.
    """
    body_id:   int
    point_idx: int
    r_local:   sym.Matrix
    r_abs:     Any   # sym.MatrixBase | sym.MatrixExpr
    rho_abs:   Any   # sym.MatrixBase | sym.MatrixExpr | None


@dataclass(frozen=True, slots=True)
class SymbolicPointsCache3D:
    """Container for all symbolic 3-D point expressions produced by
    :func:`build_points_symbolic`.

    Attributes
    ----------
    ground_points : list[PointRecord3D]
        World-frame points (body_id = 0), in declaration order.
    body_points : dict[int, list[PointRecord3D]]
        Body-local points keyed by body id (1..NBodies), values in
        declaration order within each body.
    cg_points : dict[int, PointRecord3D]
        One synthetic CG point per body (body_id 1..NBodies).
        ``r_local = zeros(3,1)``,
        ``r_abs   = cache.r_abs[body_id]`` (opaque symbolic position),
        ``rho_abs = zeros(3,1)``.
        Useful as a canonical reference for force assembly.
    """
    ground_points: List[PointRecord3D]
    body_points:   Dict[int, List[PointRecord3D]]
    cg_points:     Dict[int, PointRecord3D]


# ---------------------------------------------------------------------------
# Private helper
# ---------------------------------------------------------------------------

def _parse_point_vec(raw: list) -> sym.Matrix:
    """Convert a 3-element ``[x, y, z]`` list to a SymPy 3×1 column vector.

    Each component is passed through ``sym.sympify`` so that plain Python
    numbers and pre-existing SymPy objects are both accepted without copying.

    Raises
    ------
    ValueError
        If *raw* does not have exactly 3 elements.
    """
    if len(raw) != 3:
        raise ValueError(
            f"Each point must have exactly 3 components [x, y, z]; got {len(raw)}."
        )
    return sym.Matrix([[sym.sympify(v)] for v in raw])


# ---------------------------------------------------------------------------
# Public builder
# ---------------------------------------------------------------------------

def build_points_symbolic(
    Initial_Points: dict,
    cache: "KinematicsCache3D",
    NBodies: int,
) -> SymbolicPointsCache3D:
    """Build symbolic position expressions for all declared 3-D points.

    Uses ``cache.A_abs`` and ``cache.r_abs`` as the sole kinematic source of
    truth.  All rotation–translation products are kept in opaque,
    non-expanded form (``MatMul(..., evaluate=False)``), matching the
    symbolic style of the VT position cache.

    Parameters
    ----------
    Initial_Points : dict
        Point-definition dictionary (see module docstring for the contract).
    cache : KinematicsCache3D
        Position-level symbolic cache, already built by
        ``VelocityTransformation3D.build_cache_symbolic``.
    NBodies : int
        Number of bodies (excluding ground).  Used to validate body ids in
        ``Initial_Points["BD"]``.

    Returns
    -------
    SymbolicPointsCache3D

    Notes
    -----
    * ``expand``, ``simplify``, ``trigsimp``, and similar transforms are
      never called; symbolic structure is preserved by construction.
    * The returned object is immutable (frozen dataclass) and safe to cache
      on ``MbdSystem3D``.
    """
    # ── Ground points (world-frame, no rotation) ─────────────────────────
    ground_points: List[PointRecord3D] = []
    for idx, raw in enumerate(Initial_Points.get("GR", [])):
        r_local = _parse_point_vec(raw)
        ground_points.append(PointRecord3D(
            body_id=0,
            point_idx=idx,
            r_local=r_local,
            r_abs=r_local,   # world coords — no transform needed
            rho_abs=None,
        ))

    # ── Body-local points ────────────────────────────────────────────────
    body_points: Dict[int, List[PointRecord3D]] = {}
    for body_id, raw_list in Initial_Points.get("BD", {}).items():
        body_id = int(body_id)
        if body_id < 1 or body_id > NBodies:
            raise ValueError(
                f"Initial_Points['BD'] contains invalid body id {body_id}; "
                f"valid range is 1..{NBodies}."
            )
        A_b = cache.A_abs[body_id]   # opaque MatMul chain
        r_b = cache.r_abs[body_id]   # opaque MatAdd / MatMul

        pts: List[PointRecord3D] = []
        for idx, raw in enumerate(raw_list):
            r_local = _parse_point_vec(raw)
            # rho_abs = A_b @ r_local  — one level of opaque MatMul
            rho_abs = MatMul(A_b, r_local, evaluate=False)
            # r_abs = r_cg + rho_abs  — MatAdd, not expanded
            r_abs   = r_b + rho_abs
            pts.append(PointRecord3D(
                body_id=body_id,
                point_idx=idx,
                r_local=r_local,
                r_abs=r_abs,
                rho_abs=rho_abs,
            ))
        body_points[body_id] = pts

    # ── Synthetic CG points (one per body) ───────────────────────────────
    # rho_abs = 0  (arm from CG to itself);  r_abs = cache.r_abs[b] (opaque)
    zeros3 = sym.zeros(3, 1)
    cg_points: Dict[int, PointRecord3D] = {
        b: PointRecord3D(
            body_id=b,
            point_idx=-1,          # sentinel: not a user-declared point
            r_local=zeros3,
            r_abs=cache.r_abs[b],  # reuse opaque cache entry directly
            rho_abs=zeros3,
        )
        for b in range(1, NBodies + 1)
    }

    return SymbolicPointsCache3D(
        ground_points=ground_points,
        body_points=body_points,
        cg_points=cg_points,
    )


# ---------------------------------------------------------------------------
# Symbolic force-reduction helper
# ---------------------------------------------------------------------------

def sym_force_reduction_at_point(
    rec: PointRecord3D,
    force_vec: sym.Matrix,
) -> "tuple[sym.Matrix, sym.Matrix]":
    """Reduce a force applied at a body point to force + moment about the CG.

    Uses the CG-relative moment arm ``rec.rho_abs`` already stored in the
    :class:`PointRecord3D` (symbolic, in the world/global frame).

    The symbolic expressions are built with ordinary SymPy ``cross`` (i.e.
    ``rho.cross(f)``), which keeps the result as a 3×1 :class:`sym.Matrix`
    without invoking any expansion or simplification.

    Parameters
    ----------
    rec : PointRecord3D
        A body-attached point record (``rho_abs`` must not be ``None``).
    force_vec : sym.Matrix, shape ``(3, 1)`` or ``(3,)``
        Symbolic applied force vector, expressed in the world frame.

    Returns
    -------
    f_eq : sym.Matrix, shape ``(3, 1)``
        Equivalent body force (identical to *force_vec*, re-shaped to column).
    m_eq : sym.Matrix, shape ``(3, 1)``
        Equivalent moment about the body CG:  ``rho_abs × force_vec``.

    Raises
    ------
    ValueError
        If *rec* is a ground point (``rho_abs`` is ``None``).
    TypeError
        If *rec* is not a :class:`PointRecord3D`.

    Notes
    -----
    * No ``expand``, ``simplify``, or trigonometric transforms are applied.
    * The cross product is computed symbolically via :meth:`sym.Matrix.cross`,
      which returns a new unevaluated :class:`sym.Matrix`.

    Example
    -------
    >>> import sympy as sym
    >>> # Suppose body-4 point has rho_abs = [d4, 0, 0] (already built)
    >>> Fx, Fy, Fz = sym.symbols('Fx Fy Fz')
    >>> f_sym = sym.Matrix([Fx, Fy, Fz])
    >>> f_eq, m_eq = sym_force_reduction_at_point(rec, f_sym)
    >>> m_eq  # = [0, d4*Fz, -d4*Fy] (cross product)
    """
    if not isinstance(rec, PointRecord3D):
        raise TypeError(f"Expected PointRecord3D, got {type(rec).__name__}.")
    if rec.rho_abs is None:
        raise ValueError(
            "sym_force_reduction_at_point requires a body-attached point "
            "(rho_abs is None for ground points)."
        )
    f_col  = sym.Matrix(force_vec).reshape(3, 1)
    rho_col = sym.Matrix(rec.rho_abs).reshape(3, 1)
    # rho × f, returned as a 3×1 Matrix (not expanded)
    m_eq = rho_col.cross(f_col)
    return f_col, m_eq


# does this make sense from the AD stand point? --> To separate sympy scope from JAX scope it does
def _make_r_local_fn(r_local_sym: "sym.Matrix", body_sym_list: list, points_sym_list: list):
    """Return a callable ``fn(*body_sym_vals, *point_sym_vals) -> np.ndarray(3,)``.

    Uses :func:`sympy.lambdify` so both numeric and symbolic ``r_local``
    expressions are handled uniformly.  The returned function always produces
    a ``float64`` numpy array of shape ``(3,)``.

    ``body_sym_list`` must contain all body-geometry symbols (e.g. ``R``, ``L``)
    that may appear in point expressions; they are prepended to
    ``points_sym_list`` in the lambdify argument list.
    """
    components = [r_local_sym[i, 0] for i in range(3)]
    raw_fn = sym.lambdify(body_sym_list + points_sym_list, components, modules="numpy")
    def fn(*args, _f=raw_fn):
        return np.asarray(_f(*args), dtype=float).ravel()
    return fn


def _flatten_body_points(
    sym_points: SymbolicPointsCache3D,
    body_sym_list: list,
    points_sym_list: list,
) -> "tuple[List[int], List[Any], Dict[int, slice]]":
    """Flatten declared body-attached points into parallel lists.

    Order: ``body_id`` ASC, then ``point_idx`` ASC within each body — the
    canonical ordering shared by every consumer of
    ``SymbolicPointsCache3D.body_points`` (kinematics postprocessing, force
    evaluators, ...).

    Parameters
    ----------
    sym_points : SymbolicPointsCache3D
    body_sym_list : list[sym.Symbol]
        Ordered body-geometry symbols (see :func:`_make_r_local_fn`).
    points_sym_list : list[sym.Symbol]
        Ordered point-parameter symbols.

    Returns
    -------
    body_ids_flat : list[int]
        Owning body id (1-based) for each flattened point row.
    r_local_fns : list[callable]
        Lambdified ``r_local`` functions (see :func:`_make_r_local_fn`),
        one per row, in the same order as *body_ids_flat*.
    pt_body_slices : dict[int, slice]
        Row-range slice into the flattened arrays for each body id.
    """
    body_ids_flat:  List[int] = []
    r_local_fns:    List[Any] = []
    pt_body_slices: Dict[int, "slice"] = {}
    cursor = 0

    _body_sym_list = list(body_sym_list)
    for body_id in sorted(sym_points.body_points):
        recs = sym_points.body_points[body_id]
        for rec in recs:
            body_ids_flat.append(body_id)
            r_local_fns.append(_make_r_local_fn(rec.r_local, _body_sym_list, points_sym_list))
        pt_body_slices[body_id] = slice(cursor, cursor + len(recs))
        cursor += len(recs)

    return body_ids_flat, r_local_fns, pt_body_slices
