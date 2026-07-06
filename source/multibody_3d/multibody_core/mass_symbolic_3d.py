"""mass_symbolic_3d.py

Symbolic mass / inertia layer for 3D multibody systems.

Opacity guarantee
-----------------
Each body's world-frame inertia tensor is stored as an **unevaluated**
``MatMul`` expression::

    J_world_b = MatMul(A_abs_b, J_body_b, Transpose(A_abs_b), evaluate=False)

This mirrors the opacity policy of the kinematics layer:

* ``A_abs[b]`` is a chain of opaque ``MatMul(..., MatrixSymbol(...), ...,
  evaluate=False)`` objects produced by
  :meth:`VelocityTransformation3D.build_cache_symbolic`.
* ``J_body_b`` is wrapped as ``sym.ImmutableMatrix`` so it participates
  in the ``MatrixExpr`` hierarchy without triggering element-wise expansion.

Calls to ``expand``, ``simplify``, ``trigsimp``, or any other aggressive
SymPy transform are deliberately absent from this module.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, TYPE_CHECKING

import sympy as sym

if TYPE_CHECKING:
    from .velocity_transformation_3d import KinematicsCache3D


# ---------------------------------------------------------------------------
# Data containers
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class BodyInertiaRecord:
    """Symbolic inertia data for a single body.

    Attributes
    ----------
    body_id : int
        1-based body index (1..NBodies).
    mass : sym.Expr
        Scalar mass (``sym.sympify`` of the user-supplied value; may be
        numeric or symbolic).
    J_body : sym.ImmutableMatrix
        (3, 3) inertia tensor expressed in the body's own reference frame.
        Stored as :class:`sympy.ImmutableMatrix` so it participates in the
        ``MatrixExpr`` hierarchy without triggering copying or evaluation.
    J_world : sym.MatMul
        (3, 3) inertia tensor in the world frame.  Stored as the opaque
        ``MatMul(A_abs_b, J_body_b, Transpose(A_abs_b), evaluate=False)``
        expression — never expanded.
    """

    body_id: int
    mass:    Any                    # sym.Expr (numeric or symbolic)
    J_body:  sym.ImmutableMatrix    # (3, 3) body frame
    J_world: Any                    # sym.MatMul — opaque A J_b A^T


@dataclass
class SymbolicMassCache3D:
    """Symbolic mass / inertia data for all moving bodies.

    Produced by :func:`build_mass_symbolic`.

    Attributes
    ----------
    body_records : list[BodyInertiaRecord]
        Per-body records ordered body 1 … NBodies.
        ``body_records[b-1]`` corresponds to body *b*.
    NBodies : int
        Number of moving bodies (excluding ground).
    """

    body_records: List[BodyInertiaRecord]
    NBodies:      int

    # ------------------------------------------------------------------
    # Convenience accessors
    # ------------------------------------------------------------------

    def mass(self, body_id: int) -> Any:
        """Symbolic scalar mass for *body_id* (1-based)."""
        return self.body_records[body_id - 1].mass

    def J_body(self, body_id: int) -> sym.ImmutableMatrix:
        """Body-frame (3, 3) inertia tensor for *body_id* (1-based)."""
        return self.body_records[body_id - 1].J_body

    def J_world(self, body_id: int) -> Any:
        """Opaque world-frame inertia expression for *body_id* (1-based).

        Returns
        -------
        sym.MatMul
            ``MatMul(A_abs_b, J_body_b, Transpose(A_abs_b), evaluate=False)``
        """
        return self.body_records[body_id - 1].J_world


# ---------------------------------------------------------------------------
# Builder
# ---------------------------------------------------------------------------

def build_mass_symbolic(
    body_inertia: Dict[int, Dict[str, Any]],
    pos_cache: "KinematicsCache3D",
    NBodies: int,
) -> SymbolicMassCache3D:
    """Build symbolic mass / inertia records from user-supplied data.

    Parameters
    ----------
    body_inertia : dict[int, dict]
        Keys are 1-based body ids (1..NBodies).  Each value must contain:

        ``"mass"``
            Scalar mass — a ``float``, integer, or :class:`sympy.Expr`.
        ``"J"``
            (3, 3) body-frame inertia tensor — nested list, numpy array,
            or :class:`sympy.Matrix`.  Entries may be numeric or symbolic.

    pos_cache : KinematicsCache3D
        Position-level symbolic cache produced by
        :meth:`VelocityTransformation3D.build_cache_symbolic`.
        Provides the opaque ``A_abs[b]`` rotation chains used to form the
        world-frame expression.
    NBodies : int
        Number of moving bodies (excluding ground).

    Returns
    -------
    SymbolicMassCache3D

    Raises
    ------
    KeyError
        If *body_inertia* is missing an entry for any body 1..NBodies.
    ValueError
        If ``"J"`` does not have shape (3, 3).
    """
    records: List[BodyInertiaRecord] = []

    for b in range(1, NBodies + 1):
        if b not in body_inertia:
            raise KeyError(
                f"body_inertia missing entry for body {b}. "
                f"Provide a dict with keys 'mass' and 'J' for every body."
            )
        entry = body_inertia[b]

        # ── mass ──────────────────────────────────────────────────────────
        mass_sym = sym.sympify(entry["mass"])

        # ── body-frame inertia tensor ─────────────────────────────────────
        J_body_imm = sym.ImmutableMatrix(sym.Matrix(entry["J"]))
        if J_body_imm.shape != (3, 3):
            raise ValueError(
                f"body_inertia[{b}]['J'] must be (3, 3), got {J_body_imm.shape}."
            )

        # ── world-frame inertia (opaque MatMul) ───────────────────────────
        A_b = pos_cache.A_abs[b]    # MatrixExpr chain from kinematics layer
        # A_b.T produces Transpose(A_b) — a MatrixExpr, not expanded
        J_world = sym.MatMul(A_b, J_body_imm, A_b.T, evaluate=False)

        records.append(BodyInertiaRecord(
            body_id=b,
            mass=mass_sym,
            J_body=J_body_imm,
            J_world=J_world,
        ))

    return SymbolicMassCache3D(body_records=records, NBodies=NBodies)
