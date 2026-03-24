# source/multibody/Symvars/joint_coords_3d.py
"""
Symbolic generalized-coordinate generation for JointSystem3D.

This module generates ONLY generalized coordinates (q) and speeds (qd) as
`sympy.Symbol` objects (no functions of time, no qdd).

Notes
-----
For spherical (S) and floating (F) joints, later kinematics will typically
interpret the child's angular velocity as:

    omega_child = omega_parent + omega_rel

This module does not implement any kinematics; it only creates symbols and
bookkeeping consistent with the system topology and sys.col_slice.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import sympy as sym

from .joint_system_3d import JointSystem3D, JointType


@dataclass(frozen=True, slots=True)
class CoordBundle:
    """Container for system-wide and per-joint symbolic coordinates.

    Attributes
    ----------
    q_user : sym.Matrix
        User-facing configuration coordinates.  S/F size depends on ``rot_param``.
    qd_user : sym.Matrix
        Velocity-level coordinates (DOF-sized, same dim as ``nu``).  For all
        joint types the naming uses speed suffixes (e.g. ``wx, wy, wz`` for S)
        regardless of the position parameterisation.
    q_int : sym.Matrix
        Internal configuration coordinates.  S always 4 (quat), F always 7 (3+quat).
    nu : sym.Matrix
        Generalized speeds for B multiplication (DOF-sized: S=3, F=6).
    names_user, names_d_user : list[str]
        Symbol name lists for ``q_user`` and ``qd_user``.
    names_int : list[str]
        Symbol name list for ``q_int``.
    names_nu : list[str]
        Symbol name list for ``nu``.
    per_joint : list[dict]
        Per-joint metadata with keys ``user_slice``, ``int_slice``,
        ``speed_slice``, ``q_user``, ``qd_user``, ``q_int``, ``nu``, etc.
    """
    q_user: sym.Matrix
    qd_user: sym.Matrix
    q_int: sym.Matrix
    qd_int: sym.Matrix
    names_user: list[str]
    names_d_user: list[str]
    names_int: list[str]
    names_d_int: list[str]
    per_joint: list[dict[str, Any]]

    # Backward-compatible aliases ------------------------------------------

    @property
    def q(self) -> sym.Matrix:
        """Alias for ``q_user`` (backward compatibility)."""
        return self.q_user

    @property
    def qd(self) -> sym.Matrix:
        """Alias for ``qd_user`` (backward compatibility)."""
        return self.qd_user

    @property
    def names(self) -> list[str]:
        """Alias for ``names_user`` (backward compatibility)."""
        return self.names_user

    @property
    def names_d(self) -> list[str]:
        """Alias for ``names_d_user`` (backward compatibility)."""
        return self.names_d_user


# Keyed by *code* (e.g. "R"), not by JointType member name, to be robust to enums like
# JointType.Revolute = "R".
_SUFFIXES_BY_CODE: dict[str, tuple[str, ...]] = {
    "R": ("theta",),
    "P": ("s",),
    "S": ("phi", "theta", "psi"),
    "U": ("alpha", "beta"),
    "C": ("theta", "s"),  # rotation then translation
    "F": ("x", "y", "z", "phi", "theta", "psi"),
}

# User suffixes when rot_param == "quat" (S/F only)
_USER_SUFFIXES_QUAT: dict[str, tuple[str, ...]] = {
    "S": ("e0", "e1", "e2", "e3"),
    "F": ("x", "y", "z", "e0", "e1", "e2", "e3"),
}

# Internal configuration suffixes (always quaternion for S/F)
_INT_SUFFIXES: dict[str, tuple[str, ...]] = {
    "R": ("theta",),
    "P": ("s",),
    "S": ("e0", "e1", "e2", "e3"),
    "U": ("alpha", "beta"),
    "C": ("theta", "s"),
    "F": ("x", "y", "z", "e0", "e1", "e2", "e3"),
}

# Speed / generalized-velocity suffixes (DOF-sized)
_SPEED_SUFFIXES: dict[str, tuple[str, ...]] = {
    "R": ("theta",),
    "P": ("s",),
    "S": ("wx", "wy", "wz"),
    "U": ("alpha", "beta"),
    "C": ("theta", "s"),
    "F": ("vx", "vy", "vz", "wx", "wy", "wz"),
}


def _get_user_suffixes(code: str, rot_param: str | None) -> tuple[str, ...]:
    """Return user-facing coordinate suffixes, accounting for rot_param."""
    if code in ("S", "F") and rot_param == "quat":
        return _USER_SUFFIXES_QUAT[code]
    return _SUFFIXES_BY_CODE[code]


def _slice_len(s: slice) -> int:
    if not isinstance(s, slice):
        raise ValueError(f"Expected slice in sys.col_slice, got {type(s)!r}.")
    if s.step not in (None, 1):
        raise ValueError(f"sys.col_slice step must be None or 1; got {s.step!r}.")
    if s.start is None or s.stop is None:
        raise ValueError(f"sys.col_slice entries must have start/stop set; got {s!r}.")
    if not isinstance(s.start, int) or not isinstance(s.stop, int):
        raise ValueError(f"sys.col_slice start/stop must be int; got {s!r}.")
    if s.stop < s.start:
        raise ValueError(f"sys.col_slice stop must be >= start; got {s!r}.")
    return s.stop - s.start


def _resolve_joint_type(raw: Any) -> JointType:
    """
    Convert a raw joint type to a JointType enum member.

    Supports:
      - raw already a JointType
      - raw is the enum's value (e.g. "R")
      - raw is the enum's name (e.g. "Revolute" or "R" depending on implementation)
    """
    if isinstance(raw, JointType):
        return raw

    # Try value-based construction: JointType("R") -> JointType.Revolute (for example)
    try:
        return JointType(raw)  # type: ignore[arg-type]
    except Exception:
        pass

    # Fallback: match by name or stringified value
    raw_s = str(raw)
    for m in JointType:
        if m.name == raw_s:
            return m
        try:
            if str(m.value) == raw_s:
                return m
        except Exception:
            continue

    raise ValueError(f"Unrecognized joint type {raw!r} (cannot resolve to JointType).")


def _joint_type_and_code(j: Any) -> tuple[JointType, str]:
    raw = getattr(j, "type", None)
    if raw is None:
        raw = getattr(j, "joint_type", None)
    if raw is None:
        raise ValueError("Joint is missing a 'type' (or 'joint_type') attribute.")

    jt = _resolve_joint_type(raw)

    # Prefer using the enum value as the code when it's a 1-letter string like "R".
    code: str
    try:
        if isinstance(jt.value, str) and jt.value in _SUFFIXES_BY_CODE:
            code = jt.value
        elif jt.name in _SUFFIXES_BY_CODE:
            code = jt.name
        else:
            # Last resort: stringify value
            code = str(jt.value)
    except Exception:
        code = jt.name

    if code not in _SUFFIXES_BY_CODE:
        raise ValueError(
            f"Unsupported JointType/code for coordinate generation: jt={jt!r}, code={code!r}."
        )

    return jt, code


def _joint_dof(j: Any) -> int:
    if hasattr(j, "dof") and callable(getattr(j, "dof")):
        dof = int(j.dof())
        if dof <= 0:
            raise ValueError(f"Joint.dof() must be positive; got {dof}.")
        return dof

    _, code = _joint_type_and_code(j)
    return len(_SUFFIXES_BY_CODE[code])

def build_joint_coordinates(sys: JointSystem3D, *, prefix: str = "q") -> CoordBundle:
    """Build symbolic coordinates and speeds for a JointSystem3D.

    Generates four symbol vectors
    -----------------------------
    * **q_user** – user-facing configuration (S/F size depends on ``rot_param``).
    * **qd_user** – velocity-level coordinates, DOF-sized (same dim as ``nu``).
      Always uses speed suffixes (e.g. ``wx, wy, wz`` for S, ``vx, vy, vz, wx, wy, wz``
      for F) regardless of the position parameterisation (Euler or quaternion).
    * **q_int** – internal configuration (S always 4 quat, F always 3+4 quat).
    * **nu** – generalized speeds / B-column velocities (DOF-sized: S=3, F=6).

    Naming
    ------
    ``q_user``:  ``{prefix}{child}_{suffix}``
    ``qd_user``: ``{prefix_d}{child}_{suffix}``
    ``q_int``:   ``{prefix}i{child}_{suffix}``   (always-quaternion for S/F)
    ``nu``:      ``nu{child}_{suffix}``

    User suffixes per joint type (euler default / quat override)
    ------------------------------------------------------------
    R: [theta]           | P: [s]           | U: [alpha, beta]
    C: [theta, s]
    S euler: [phi, theta, psi]          S quat: [e0, e1, e2, e3]
    F euler: [x,y,z, phi,theta,psi]     F quat: [x,y,z, e0,e1,e2,e3]

    Raises ValueError if sys.total_dof / sys.col_slice / joint DOFs are inconsistent.
    """
    ########## VALIDATION ##########
    if not hasattr(sys, "joints"):
        raise ValueError("sys must have attribute 'joints'.")

    joints      = list(sys.joints)
    n_joints    = len(joints)

    if not hasattr(sys, "col_slice"):
        raise ValueError("sys must have attribute 'col_slice' (list[slice]).")
    if len(sys.col_slice) != n_joints:
        raise ValueError(
            f"len(sys.col_slice) must match number of joints: "
            f"{len(sys.col_slice)} != {n_joints}."
        )

    dofs = [_joint_dof(j) for j in joints]
    expected_total_dof = sum(dofs)

    if not hasattr(sys, "total_dof"):
        raise ValueError("sys must have attribute 'total_dof'.")
    if int(sys.total_dof) != expected_total_dof:
        raise ValueError(
            f"sys.total_dof mismatch: sys.total_dof={sys.total_dof} "
            f"but sum(joint.dof())={expected_total_dof}."
        )

    # Validate each slice length equals its joint DOF and total matches sys.total_dof.
    slice_lengths: list[int] = []
    for i, (slc, dof) in enumerate(zip(sys.col_slice, dofs, strict=True)):
        n = _slice_len(slc)
        slice_lengths.append(n)
        if n != dof:
            j = joints[i]
            child = getattr(j, "child", None)
            raise ValueError(
                f"sys.col_slice[{i}] length mismatch for child={child}: "
                f"slice={slc} has length {n}, but joint.dof()={dof}."
            )
    if sum(slice_lengths) != int(sys.total_dof):
        raise ValueError(
            f"sum(len(sys.col_slice[i])) must equal sys.total_dof: "
            f"{sum(slice_lengths)} != {sys.total_dof}."
        )


    ########### CONSTRUCTION ##########
    prefix_d    = "qd" if prefix == "q" else f"{prefix}d"
    prefix_int  = f"{prefix}i"
    prefix_nu   = f"{prefix_d}i"

    q_user_syms:  list[sym.Symbol] = []
    qd_user_syms: list[sym.Symbol] = []
    q_int_syms:   list[sym.Symbol] = []
    qd_int_syms:      list[sym.Symbol] = []

    names_user:   list[str] = []
    names_d_user: list[str] = []
    names_int:    list[str] = []
    names_d_int:     list[str] = []
    per_joint: list[dict[str, Any]] = []

    for j_idx, j in enumerate(joints):
        jt, code    = _joint_type_and_code(j)

        child       = int(getattr(j, "child"))
        parent      = int(getattr(j, "parent"))
        rot_param   = getattr(j, "rot_param", None)

        base_q   = f"{prefix}{child}"
        base_qd  = f"{prefix_d}{child}"
        base_int = f"{prefix_int}{child}"
        base_nu  = f"{prefix_nu}{child}"

        # --- user configuration ---
        user_sufs  = _get_user_suffixes(code, rot_param)
        speed_sufs = _SPEED_SUFFIXES[code]

        loc_names_user   = [f"{base_q}_{s}" for s in user_sufs]
        loc_names_d_user = [f"{base_qd}_{s}" for s in speed_sufs]
        loc_q_user  = sym.Matrix([sym.Symbol(nm, real=True) for nm in loc_names_user])
        loc_qd_user = sym.Matrix([sym.Symbol(nm, real=True) for nm in loc_names_d_user])

        # --- internal configuration (always quaternion for S/F) ---
        int_sufs      = _INT_SUFFIXES[code]
        loc_names_int = [f"{base_int}_{s}" for s in int_sufs]
        loc_q_int     = sym.Matrix([sym.Symbol(nm, real=True) for nm in loc_names_int])

        # --- generalized speeds (DOF-sized) ---
        loc_names_d_int = [f"{base_nu}_{s}" for s in speed_sufs]
        loc_qd_int   = sym.Matrix([sym.Symbol(nm, real=True) for nm in loc_names_d_int])

        # accumulate
        q_user_syms.extend(list(loc_q_user))
        qd_user_syms.extend(list(loc_qd_user))
        q_int_syms.extend(list(loc_q_int))
        qd_int_syms.extend(list(loc_qd_int))

        names_user.extend(loc_names_user)
        names_d_user.extend(loc_names_d_user)
        names_int.extend(loc_names_int)
        names_d_int.extend(loc_names_d_int)

        per_joint.append(
            {
                "joint_index": j_idx,
                "child": child,
                "parent": parent,
                "type": jt,
                "rot_param": rot_param,
                # slices
                "user_slice":  sys.user_col_slice[j_idx],
                "int_slice":   sys.cfg_col_slice[j_idx],
                "speed_slice": sys.col_slice[j_idx],
                # backward-compat alias
                "slice": sys.col_slice[j_idx],
                # symbolic vectors
                "q_user":  loc_q_user,
                "qd_user": loc_qd_user,
                "q_int":   loc_q_int,
                "qd_int":  loc_qd_int,
                # backward-compat aliases
                "q":  loc_q_user,
                "qd": loc_qd_user,
                # name lists
                "names_user":   loc_names_user,
                "names_d_user": loc_names_d_user,
                "names_int":    loc_names_int,
                "names_d_int":  loc_names_d_int,
                "names":   loc_names_user,
                "names_d": loc_names_d_user,
            }
        )

    q_user  = sym.Matrix(q_user_syms)
    qd_user = sym.Matrix(qd_user_syms)
    q_int   = sym.Matrix(q_int_syms)
    qd_int  = sym.Matrix(qd_int_syms)

    # Shape sanity checks
    if q_user.shape != (int(sys.total_user_dof), 1):
        raise ValueError(
            f"q_user shape mismatch: got {q_user.shape}, "
            f"expected {(sys.total_user_dof, 1)}."
        )
    if q_int.shape != (int(sys.total_cfg_dof), 1):
        raise ValueError(
            f"q_int shape mismatch: got {q_int.shape}, "
            f"expected {(sys.total_cfg_dof, 1)}."
        )
    if qd_int.shape != (int(sys.total_dof), 1):
        raise ValueError(
            f"qd_int shape mismatch: got {qd_int.shape}, "
            f"expected {(sys.total_dof, 1)}."
        )

    return CoordBundle(
        q_user=q_user,
        qd_user=qd_user,
        q_int=q_int,
        qd_int=qd_int,
        names_user=names_user,
        names_d_user=names_d_user,
        names_int=names_int,
        names_d_int=names_d_int,
        per_joint=per_joint,
    )


# --------------- demo / smoke test ---------------

def _demo_coord_bundles():
    """Demo: S joint with 'euler' vs 'quat' user coords; q_int always quaternion."""
    z3 = [0.0, 0.0, 0.0]
    data_euler = {
        "NBodies": 1,
        "joints": [(0, 1)],
        "types": ["S"],
        "parent_cg_to_joint": [z3],
        "joint_to_child_cg": [z3],
        "axis_u": [None],
        "rot_param": ["euler"],
    }
    sys_e = JointSystem3D.from_data(data_euler)
    bundle_e = build_joint_coordinates(sys_e)

    data_quat = dict(data_euler, rot_param=["quat"])
    sys_q = JointSystem3D.from_data(data_quat)
    bundle_q = build_joint_coordinates(sys_q)

    print(f"Euler: q_user={bundle_e.q_user.shape}, q_int={bundle_e.q_int.shape}, qd_int={bundle_e.qd_int.shape}")
    print(f"Quat:  q_user={bundle_q.q_user.shape}, q_int={bundle_q.q_int.shape}, qd_int={bundle_q.qd_int.shape}")
    # Euler: q_user=(3, 1), q_int=(4, 1), qd_int=(3, 1)
    # Quat:  q_user=(4, 1), q_int=(4, 1), qd_int=(3, 1)
    return bundle_e, bundle_q