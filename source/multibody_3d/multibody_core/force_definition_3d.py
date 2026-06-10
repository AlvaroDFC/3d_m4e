"""force_definition_3d.py

3D force-definition contract and internal normalization layer.

Defines typed records for each supported force family, a grouped container,
and a parser that converts the user-facing ``Force`` dictionary into validated
internal objects.

User-facing ``Force`` dictionary
---------------------------------
All keys are optional.  Unrecognized keys raise ``ValueError``.

``"CG"`` : dict[int, dict]
    Direct force + moment at body CG, in the world frame.
    Key is the integer body id (1-based).  Value must contain the key
    ``"force"`` mapping to a 3-element sequence (numeric or SymPy); the
    optional key ``"moment"`` maps to a 3-element sequence (defaults to
    ``[0, 0, 0]``).

    Example::

        "CG": {
            1: {"force": [Fx, 0, 0], "moment": [0, 0, Mz]},
            2: {"force": [0, Fy, 0]},           # moment defaults to [0,0,0]
        }

``"PointsBD"`` : list
    Force (and optional free moment) applied at a declared body point.
    Each entry may be a tuple or a dict:

    * Tuple: ``(body_id, pt_idx, force_vec)``
      or     ``(body_id, pt_idx, force_vec, moment_vec)``
    * Dict:  ``{"body": body_id, "pt": pt_idx, "force": [...], "moment": [...]}``

    ``body_id`` is 1-based; ``pt_idx`` is the 0-based index into the
    corresponding entry in ``Initial_Points["BD"][body_id]``.

    Example::

        "PointsBD": [
            (2, 0, [Fx, 0, 0]),                 # body 2, point 0, no free moment
            (3, 0, [0, Fy, 0], [0, 0, Mz]),     # body 3, point 0, with free moment
        ]

``"TensionSpring"`` : list
    Linear tension spring between two body (or ground) points.
    Entry tuple: ``(body_id_A, pt_idx_A, body_id_B, pt_idx_B, k, L0)``

    ``body_id = 0`` selects a ground point from ``Initial_Points["GR"]``.
    ``k``  is the spring stiffness (numeric or SymPy).
    ``L0`` is the natural length (numeric or SymPy).

    Example::

        "TensionSpring": [
            (2, 0, 3, 0, k, L0),    # body-2 pt-0  ↔  body-3 pt-0
            (0, 0, 4, 0, k2, L0),   # ground pt-0  ↔  body-4 pt-0
        ]

``"TensionDamper"`` : list
    Linear viscous tension damper between two body (or ground) points.
    Entry tuple: ``(body_id_A, pt_idx_A, body_id_B, pt_idx_B, c)``

    Example::

        "TensionDamper": [
            (2, 0, 3, 0, c),
        ]

``"TorsionSpring"`` : list
    Torsional spring at a revolute joint DOF.
    Entry tuple: ``(joint_idx, k, theta_eq)``

    ``joint_idx`` is the 0-based index into the ``joints`` list and must
    reference a revolute joint (type ``"R"``).
    ``theta_eq``  is the equilibrium angle (numeric or SymPy).

    Example::

        "TorsionSpring": [
            (1, k_t, 0.0),    # joint index 1, stiffness k_t, equilibrium at 0
        ]

``"TorsionDamper"`` : list
    Torsional viscous damper at a revolute joint DOF.
    Entry tuple: ``(joint_idx, c)``

    Example::

        "TorsionDamper": [
            (1, c_t),
        ]

``"Gravity"`` : dict
    Uniform gravitational field applied at each body CG.
    Must contain exactly two keys:

    ``"g_vec"`` — 3-element sequence ``[gx, gy, gz]`` (numeric or SymPy),
    typically ``[0, 0, -9.81]``.

    ``"mass"``  — either a list of body masses (1-indexed, so
    ``mass[i-1]`` is the mass of body *i*), or a dict
    ``{body_id: mass}`` mapping 1-based body ids to mass values
    (numeric or SymPy).

    Example::

        "Gravity": {
            "g_vec": [0, 0, -9.81],
            "mass":  {1: m1, 2: m2, 3: m3, 4: m4},
        }
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import sympy as sym


# ---------------------------------------------------------------------------
# Internal leaf records  (one per force element)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class CGForceDef:
    """Direct force and moment applied at the CG of a body, in the world frame.

    Attributes
    ----------
    body_id : int
        1-based body index.
    force_vec : tuple[Any, Any, Any]
        (Fx, Fy, Fz) components; numeric or SymPy expressions.
    moment_vec : tuple[Any, Any, Any]
        (Mx, My, Mz) components; numeric or SymPy expressions.
    """

    body_id:    int
    force_vec:  Tuple[Any, Any, Any]
    moment_vec: Tuple[Any, Any, Any]


@dataclass(frozen=True)
class PointForceDef:
    """Force and optional free moment applied at a declared body point.

    Attributes
    ----------
    body_id : int
        1-based body index.
    point_idx : int
        0-based index into ``Initial_Points["BD"][body_id]``.
    force_vec : tuple[Any, Any, Any]
        (Fx, Fy, Fz) in the world frame; numeric or SymPy.
    moment_vec : tuple[Any, Any, Any]
        (Mx, My, Mz) free moment in the world frame; defaults to (0, 0, 0).
    """

    body_id:    int
    point_idx:  int
    force_vec:  Tuple[Any, Any, Any]
    moment_vec: Tuple[Any, Any, Any]


@dataclass(frozen=True)
class TensionSpringDef:
    """Linear tension spring connecting two body (or ground) points.

    Attributes
    ----------
    body_id_a, body_id_b : int
        Body ids for the two attachment points.  0 = ground.
    pt_idx_a, pt_idx_b : int
        0-based point indices within their respective bodies (or ground list).
    k : Any
        Spring stiffness; numeric or SymPy.
    L0 : Any
        Natural (unstretched) length; numeric or SymPy.
    """

    body_id_a: int
    pt_idx_a:  int
    body_id_b: int
    pt_idx_b:  int
    k:         Any
    L0:        Any


@dataclass(frozen=True)
class TensionDamperDef:
    """Linear viscous tension damper connecting two body (or ground) points.

    Attributes
    ----------
    body_id_a, body_id_b : int
        Body ids for the two attachment points.  0 = ground.
    pt_idx_a, pt_idx_b : int
        0-based point indices within their respective bodies (or ground list).
    c : Any
        Damping coefficient; numeric or SymPy.
    """

    body_id_a: int
    pt_idx_a:  int
    body_id_b: int
    pt_idx_b:  int
    c:         Any


@dataclass(frozen=True)
class TorsionSpringDef:
    """Torsional spring at a revolute joint DOF.

    Attributes
    ----------
    joint_idx : int
        0-based index into the ``joints`` list.  Must be a revolute joint.
    k : Any
        Torsional stiffness; numeric or SymPy.
    theta_eq : Any
        Equilibrium angle; numeric or SymPy.
    """

    joint_idx: int
    k:         Any
    theta_eq:  Any


@dataclass(frozen=True)
class TorsionDamperDef:
    """Torsional viscous damper at a revolute joint DOF.

    Attributes
    ----------
    joint_idx : int
        0-based index into the ``joints`` list.  Must be a revolute joint.
    c : Any
        Torsional damping coefficient; numeric or SymPy.
    """

    joint_idx: int
    c:         Any


@dataclass(frozen=True)
class GravityDef:
    """Uniform gravitational field applied at each body CG.

    Attributes
    ----------
    g_vec : tuple[Any, Any, Any]
        Gravity acceleration vector (gx, gy, gz) in the world frame.
    mass : dict[int, Any]
        Map from 1-based body id to mass value (numeric or SymPy).
        Bodies absent from the dict are assigned zero mass.
    """

    g_vec: Tuple[Any, Any, Any]
    mass:  Dict[int, Any]  # body_id (1-based) → mass


# ---------------------------------------------------------------------------
# Grouped container
# ---------------------------------------------------------------------------

@dataclass
class ForcesDefinition3D:
    """All force elements for a 3-D multibody system, in parsed form.

    Built by :func:`parse_force_dict` from the user-facing ``Force`` dict.
    All lists preserve the user's declaration order.

    Attributes
    ----------
    cg_forces : list[CGForceDef]
    point_forces : list[PointForceDef]
    tension_springs : list[TensionSpringDef]
    tension_dampers : list[TensionDamperDef]
    torsion_springs : list[TorsionSpringDef]
    torsion_dampers : list[TorsionDamperDef]
    """

    cg_forces:       List[CGForceDef]       = field(default_factory=list)
    point_forces:    List[PointForceDef]    = field(default_factory=list)
    tension_springs: List[TensionSpringDef] = field(default_factory=list)
    tension_dampers: List[TensionDamperDef] = field(default_factory=list)
    torsion_springs: List[TorsionSpringDef] = field(default_factory=list)
    torsion_dampers: List[TorsionDamperDef] = field(default_factory=list)
    gravity:         Optional[GravityDef]   = field(default=None)

    # ── Public query helpers ──────────────────────────────────────────────────────────

    def is_empty(self) -> bool:
        """Return *True* if no force elements are defined."""
        return not any([
            self.cg_forces,
            self.point_forces,
            self.tension_springs,
            self.tension_dampers,
            self.torsion_springs,
            self.torsion_dampers,
            self.gravity is not None,
        ])

    def collect_symbols(self) -> List[sym.Symbol]:
        """Return all free SymPy symbols appearing in force parameters.

        Symbols are returned in first-seen declaration order; duplicates are
        dropped.  This list is derived by scanning the parsed records; it is
        **not** the same as (though it should match) the user-declared
        ``force_sym`` dict.

        Returns
        -------
        list[sympy.Symbol]
        """
        seen: Dict[sym.Symbol, None] = {}  # ordered set via insertion-ordered dict

        def _scan(expr: Any) -> None:
            if isinstance(expr, sym.Basic):
                for s in expr.free_symbols:
                    seen.setdefault(s, None)

        def _scan_vec(vec: Sequence[Any]) -> None:
            for comp in vec:
                _scan(comp)

        for d in self.cg_forces:
            _scan_vec(d.force_vec)
            _scan_vec(d.moment_vec)

        for d in self.point_forces:
            _scan_vec(d.force_vec)
            _scan_vec(d.moment_vec)

        for d in self.tension_springs:
            _scan(d.k)
            _scan(d.L0)

        for d in self.tension_dampers:
            _scan(d.c)

        for d in self.torsion_springs:
            _scan(d.k)
            _scan(d.theta_eq)

        for d in self.torsion_dampers:
            _scan(d.c)

        if self.gravity is not None:
            _scan_vec(self.gravity.g_vec)
            for mass_val in self.gravity.mass.values():
                _scan(mass_val)

        return list(seen)


# ---------------------------------------------------------------------------
# Parser
# ---------------------------------------------------------------------------

_KNOWN_KEYS = frozenset({
    "CG",
    "PointsBD",
    "TensionSpring",
    "TensionDamper",
    "TorsionSpring",
    "TorsionDamper",
    "Gravity",
})

_ZEROS3: Tuple[int, int, int] = (0, 0, 0)


def parse_force_dict(
    force_dict: Dict,
    *,
    n_bodies: Optional[int] = None,
    joint_types: Optional[List[str]] = None,
) -> ForcesDefinition3D:
    """Parse the user-facing ``Force`` dictionary into a :class:`ForcesDefinition3D`.

    Parameters
    ----------
    force_dict : dict
        User-declared force dictionary.  See module docstring for the full
        contract.
    n_bodies : int, optional
        If given, body-id references are validated to lie in ``[0, n_bodies]``.
    joint_types : list[str], optional
        If given, torsion-element joint indices are validated against this
        list.  Each entry is a joint-type string, e.g. ``"R"``, ``"F"``.

    Returns
    -------
    ForcesDefinition3D

    Raises
    ------
    TypeError
        If *force_dict* is not a ``dict``.
    ValueError
        On unrecognized keys, malformed entries, or out-of-range references.
    """
    if not isinstance(force_dict, dict):
        raise TypeError(
            f"Force must be a dict; got {type(force_dict).__name__!r}."
        )

    unknown = set(force_dict.keys()) - _KNOWN_KEYS
    if unknown:
        raise ValueError(
            f"Unrecognized Force key(s): {sorted(unknown)}. "
            f"Allowed keys: {sorted(_KNOWN_KEYS)}."
        )

    fd = ForcesDefinition3D()

    # ── CG forces ────────────────────────────────────────────────────────────
    for raw_id, spec in force_dict.get("CG", {}).items():
        body_id = int(raw_id)
        _check_body_id(body_id, n_bodies, allow_ground=False, label="CG")
        if not isinstance(spec, dict) or "force" not in spec:
            raise ValueError(
                f"Force['CG'][{body_id}] must be a dict with key 'force'; "
                f"got {spec!r}."
            )
        fvec = _as3(spec["force"],                    f"Force['CG'][{body_id}]['force']")
        mvec = _as3(spec.get("moment", _ZEROS3),      f"Force['CG'][{body_id}]['moment']")
        fd.cg_forces.append(CGForceDef(body_id, fvec, mvec))

    # ── Body-point forces ─────────────────────────────────────────────────────
    for i, entry in enumerate(force_dict.get("PointsBD", [])):
        label = f"Force['PointsBD'][{i}]"
        if isinstance(entry, dict):
            body_id  = int(entry["body"])
            pt_idx   = int(entry["pt"])
            fvec     = _as3(entry["force"],                f"{label}['force']")
            mvec     = _as3(entry.get("moment", _ZEROS3),  f"{label}['moment']")
        elif isinstance(entry, (tuple, list)) and len(entry) in (3, 4):
            body_id = int(entry[0])
            pt_idx  = int(entry[1])
            fvec    = _as3(entry[2], f"{label}[2]")
            mvec    = _as3(entry[3], f"{label}[3]") if len(entry) == 4 else _ZEROS3
        else:
            raise ValueError(
                f"{label}: expected a 3- or 4-element tuple, or a dict; "
                f"got {entry!r}."
            )
        _check_body_id(body_id, n_bodies, allow_ground=False, label=label)
        fd.point_forces.append(PointForceDef(body_id, pt_idx, fvec, mvec))

    # ── Tension springs ───────────────────────────────────────────────────────
    for i, entry in enumerate(force_dict.get("TensionSpring", [])):
        label = f"Force['TensionSpring'][{i}]"
        t = tuple(entry)
        if len(t) != 6:
            raise ValueError(
                f"{label}: expected 6-tuple "
                f"(body_A, pt_A, body_B, pt_B, k, L0); got length {len(t)}."
            )
        ba, pa, bb, pb = int(t[0]), int(t[1]), int(t[2]), int(t[3])
        k, L0 = t[4], t[5]
        _check_body_id(ba, n_bodies, allow_ground=True, label=f"{label} body_A")
        _check_body_id(bb, n_bodies, allow_ground=True, label=f"{label} body_B")
        fd.tension_springs.append(TensionSpringDef(ba, pa, bb, pb, k, L0))

    # ── Tension dampers ───────────────────────────────────────────────────────
    for i, entry in enumerate(force_dict.get("TensionDamper", [])):
        label = f"Force['TensionDamper'][{i}]"
        t = tuple(entry)
        if len(t) != 5:
            raise ValueError(
                f"{label}: expected 5-tuple "
                f"(body_A, pt_A, body_B, pt_B, c); got length {len(t)}."
            )
        ba, pa, bb, pb = int(t[0]), int(t[1]), int(t[2]), int(t[3])
        c = t[4]
        _check_body_id(ba, n_bodies, allow_ground=True, label=f"{label} body_A")
        _check_body_id(bb, n_bodies, allow_ground=True, label=f"{label} body_B")
        fd.tension_dampers.append(TensionDamperDef(ba, pa, bb, pb, c))

    # ── Torsion springs ───────────────────────────────────────────────────────
    for i, entry in enumerate(force_dict.get("TorsionSpring", [])):
        label = f"Force['TorsionSpring'][{i}]"
        t = tuple(entry)
        if len(t) != 3:
            raise ValueError(
                f"{label}: expected 3-tuple (joint_idx, k, theta_eq); "
                f"got length {len(t)}."
            )
        jidx, k, theta_eq = int(t[0]), t[1], t[2]
        _check_joint_idx(jidx, joint_types, label, allowed_types=("R",))
        fd.torsion_springs.append(TorsionSpringDef(jidx, k, theta_eq))

    # ── Torsion dampers ───────────────────────────────────────────────────────
    for i, entry in enumerate(force_dict.get("TorsionDamper", [])):
        label = f"Force['TorsionDamper'][{i}]"
        t = tuple(entry)
        if len(t) != 2:
            raise ValueError(
                f"{label}: expected 2-tuple (joint_idx, c); "
                f"got length {len(t)}."
            )
        jidx, c = int(t[0]), t[1]
        _check_joint_idx(jidx, joint_types, label, allowed_types=("R",))
        fd.torsion_dampers.append(TorsionDamperDef(jidx, c))
    # ── Gravity ──────────────────────────────────────────────────────────────────
    if "Gravity" in force_dict:
        gspec = force_dict["Gravity"]
        if not isinstance(gspec, dict):
            raise ValueError(
                f"Force['Gravity'] must be a dict with keys 'g_vec' and 'mass'; "
                f"got {type(gspec).__name__!r}."
            )
        if "g_vec" not in gspec or "mass" not in gspec:
            raise ValueError(
                "Force['Gravity'] must contain both 'g_vec' and 'mass' keys."
            )
        g_vec = _as3(gspec["g_vec"], "Force['Gravity']['g_vec']")
        raw_mass = gspec["mass"]
        if isinstance(raw_mass, dict):
            mass_map: Dict[int, Any] = {int(k): v for k, v in raw_mass.items()}
        elif hasattr(raw_mass, "__iter__"):
            # list/tuple — 0-indexed → body id = index+1
            mass_map = {i + 1: v for i, v in enumerate(raw_mass)}
        else:
            raise ValueError(
                "Force['Gravity']['mass'] must be a dict {body_id: mass} "
                "or a list [m1, m2, ...]; "
                f"got {type(raw_mass).__name__!r}."
            )
        # Validate body ids against n_bodies
        for bid in mass_map:
            _check_body_id(bid, n_bodies, allow_ground=False, label="Force['Gravity']['mass']")
        fd.gravity = GravityDef(g_vec, mass_map)
    return fd


# ---------------------------------------------------------------------------
# Validation helpers (module-private)
# ---------------------------------------------------------------------------

def _as3(seq: Sequence, label: str) -> Tuple[Any, Any, Any]:
    """Coerce *seq* to a 3-tuple, raising ``ValueError`` on wrong length."""
    try:
        a, b, c = seq
    except (TypeError, ValueError):
        raise ValueError(
            f"{label} must be a 3-element sequence; got {seq!r}."
        ) from None
    return (a, b, c)


def _check_body_id(
    body_id: int,
    n_bodies: Optional[int],
    allow_ground: bool,
    label: str,
) -> None:
    min_id = 0 if allow_ground else 1
    if body_id < min_id:
        detail = "ground (id=0) is not allowed here" if not allow_ground else "negative id"
        raise ValueError(
            f"{label}: body_id={body_id} is invalid ({detail})."
        )
    if n_bodies is not None and body_id > n_bodies:
        raise ValueError(
            f"{label}: body_id={body_id} exceeds n_bodies={n_bodies}."
        )


def _check_joint_idx(
    joint_idx: int,
    joint_types: Optional[List[str]],
    label: str,
    allowed_types: Tuple[str, ...],
) -> None:
    if joint_types is None:
        return
    if joint_idx < 0 or joint_idx >= len(joint_types):
        raise ValueError(
            f"{label}: joint_idx={joint_idx} out of range "
            f"[0, {len(joint_types) - 1}]."
        )
    jtype = joint_types[joint_idx]
    if jtype not in allowed_types:
        raise ValueError(
            f"{label}: joint_idx={joint_idx} has type {jtype!r}; "
            f"torsion elements require a revolute joint "
            f"(allowed types: {allowed_types})."
        )
