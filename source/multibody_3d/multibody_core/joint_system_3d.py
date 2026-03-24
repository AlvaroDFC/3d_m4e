# source/multibody/joints_system_3d.py
"""
joints_system_3d.py

3D Joint data model + topology/indexing utilities for joint-coordinate multibody systems.
This module intentionally does NOT implement B, Bdot, or kinematics.

Key conventions
---------------
- Bodies: ground=0, bodies=1..NBodies (1-based).
- Joint axes are expressed in the PARENT body frame.
- parent_cg_to_joint: vector from parent CG to joint location, expressed in parent frame.
- joint_to_child_cg: vector from joint location to child CG, expressed in child frame.

Note on S and F joints
----------------------
Spherical ('S') and Floating ('F') joints will later use RELATIVE angular velocities:
    omega_child = omega_parent + omega_rel
This affects later kinematics/Bdot, not topology. Documented here for clarity.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import sympy as sym

from .topology_3d import (
    TreeIndex,
    build_adjacency,
    compute_Btrack,
    compute_root_to_leaf_joint_paths,
    validate_tree,
    to_display_value,
)


class JointType(Enum):
    REVOLUTE    = "R"       # 1 dof rotation about u
    PRISMATIC   = "P"       # 1 dof translation along u
    SPHERICAL   = "S"       # 3 dof rotation
    UNIVERSAL   = "U"       # 2 dof rotation about u1, u2
    CYLINDRICAL = "C"       # 2 dof: rotation + translation about same axis u
    FLOATING    = "F"       # 6 dof


_DOF_MAP: Dict[JointType, int] = {
    JointType.REVOLUTE:     1,
    JointType.PRISMATIC:    1,
    JointType.SPHERICAL:    3,
    JointType.UNIVERSAL:    2,
    JointType.CYLINDRICAL:  2,
    JointType.FLOATING:     6,
}

# Internal configuration sizes (always quaternion for S/F)
_CFG_DOF_MAP: Dict[JointType, int] = {
    JointType.REVOLUTE:     1,
    JointType.PRISMATIC:    1,
    JointType.SPHERICAL:    4,   # unit quaternion [e0,e1,e2,e3]
    JointType.UNIVERSAL:    2,
    JointType.CYLINDRICAL:  2,
    JointType.FLOATING:     7,   # [x,y,z, e0,e1,e2,e3]
}


def _to_vec3(v: Any, *, ctx: str) -> sym.Matrix:
    """Convert v (list/tuple/np array/sym Matrix) to sympy Matrix shape (3,1)."""
    if v is None:
        raise ValueError(f"{ctx}: expected a 3-vector, got None.")
    M = sym.Matrix(v)
    if M.shape == (1, 3):
        M = M.T
    if M.shape != (3, 1):
        raise ValueError(f"{ctx}: expected shape (3,1) or (1,3), got {M.shape}.")
    return M


def _maybe_vec3(v: Any, *, ctx: str) -> Optional[sym.Matrix]:
    """Like _to_vec3 but allows None."""
    if v is None:
        return None
    return _to_vec3(v, ctx=ctx)


def _normalize_axis(u: sym.Matrix, *, ctx: str) -> sym.Matrix:
    """Return u normalized to unit length; raise on zero length or NaNs."""
    if u.has(sym.nan):
        raise ValueError(f"{ctx}: axis contains NaN.")
    n2 = sym.simplify(u.dot(u))
    if n2 == 0:
        raise ValueError(f"{ctx}: axis has zero length.")
    n = sym.sqrt(n2)
    return sym.Matrix(u) / n


def _axes_noncollinear(u1: sym.Matrix, u2: sym.Matrix, *, ctx: str) -> None:
    """Validate u1 and u2 are not collinear (cross != 0)."""
    c = sym.Matrix(u1).cross(sym.Matrix(u2))
    c_s = sym.Matrix([sym.simplify(ci) for ci in c])
    if all(ci == 0 for ci in c_s):
        raise ValueError(f"{ctx}: universal axes are collinear (u1 x u2 == 0). Provide non-collinear axes.")


# --------------- Euler / Quaternion conversion utilities ---------------

def euler_to_quat(phi: float, theta: float, psi: float) -> np.ndarray:
    """Convert ZYX Euler angles to unit quaternion ``[e0, e1, e2, e3]``.

    Convention
    ----------
    ZYX intrinsic (body-fixed) rotation sequence:

        R = Rz(psi) · Ry(theta) · Rx(phi)

    * phi   – roll  (rotation about X)
    * theta – pitch (rotation about Y)
    * psi   – yaw   (rotation about Z)

    Quaternion layout: ``[e0, e1, e2, e3] = [w, x, y, z]``.
    Output is always normalized to unit length.
    """
    cp, sp = np.cos(phi / 2), np.sin(phi / 2)
    ct, st = np.cos(theta / 2), np.sin(theta / 2)
    cs, ss = np.cos(psi / 2), np.sin(psi / 2)

    e0 = cp * ct * cs + sp * st * ss
    e1 = sp * ct * cs - cp * st * ss
    e2 = cp * st * cs + sp * ct * ss
    e3 = cp * ct * ss - sp * st * cs

    q = np.array([e0, e1, e2, e3])
    return q / np.linalg.norm(q)


def quat_to_euler(q) -> np.ndarray:
    """Convert unit quaternion ``[e0, e1, e2, e3]`` to ZYX Euler angles.

    Returns ``[phi, theta, psi]`` in radians.
    Gimbal-lock handling: ``theta`` is clamped to ``[-pi/2, pi/2]`` via
    ``np.clip`` before ``arcsin``.
    """
    q = np.asarray(q, dtype=float).ravel()
    q = q / np.linalg.norm(q)
    e0, e1, e2, e3 = q

    # Roll (phi)
    sinr_cosp = 2.0 * (e0 * e1 + e2 * e3)
    cosr_cosp = 1.0 - 2.0 * (e1 ** 2 + e2 ** 2)
    phi = np.arctan2(sinr_cosp, cosr_cosp)

    # Pitch (theta)
    sinp = np.clip(2.0 * (e0 * e2 - e3 * e1), -1.0, 1.0)
    theta = np.arcsin(sinp)

    # Yaw (psi)
    siny_cosp = 2.0 * (e0 * e3 + e1 * e2)
    cosy_cosp = 1.0 - 2.0 * (e2 ** 2 + e3 ** 2)
    psi = np.arctan2(siny_cosp, cosy_cosp)

    return np.array([phi, theta, psi])


@dataclass
class Joint3D:
    parent: int
    child: int
    type: JointType

    parent_cg_to_joint: Any
    joint_to_child_cg: Any

    axis_u: Optional[Any]   = None
    axis_u1: Optional[Any]  = None
    axis_u2: Optional[Any]  = None

    rot_param: Optional[str] = None  # "euler" or "quat" (S/F only; default "euler")

    # internal normalized matrices
    _pcg2j: sym.Matrix          = field(init=False, repr=False)
    _j2ccg: sym.Matrix          = field(init=False, repr=False)
    _u: Optional[sym.Matrix]    = field(init=False, default=None, repr=False)
    _u1: Optional[sym.Matrix]   = field(init=False, default=None, repr=False)
    _u2: Optional[sym.Matrix]   = field(init=False, default=None, repr=False)

    def __post_init__(self) -> None:
        self._pcg2j = _to_vec3(self.parent_cg_to_joint, ctx=f"Joint(child={self.child}) parent_cg_to_joint")
        self._j2ccg = _to_vec3(self.joint_to_child_cg, ctx=f"Joint(child={self.child}) joint_to_child_cg")

        self._u     = _maybe_vec3(self.axis_u, ctx=f"Joint(child={self.child}) axis_u")
        self._u1    = _maybe_vec3(self.axis_u1, ctx=f"Joint(child={self.child}) axis_u1")
        self._u2    = _maybe_vec3(self.axis_u2, ctx=f"Joint(child={self.child}) axis_u2")

        # Normalize axes if present (actual requirement enforcement happens in validate()).
        if self._u is not None:
            self._u = _normalize_axis(self._u, ctx=f"Joint(child={self.child}) axis_u")
        if self._u1 is not None:
            self._u1 = _normalize_axis(self._u1, ctx=f"Joint(child={self.child}) axis_u1")
        if self._u2 is not None:
            self._u2 = _normalize_axis(self._u2, ctx=f"Joint(child={self.child}) axis_u2")

        # Default rot_param for S/F joints
        if self.rot_param is None and self.type in (JointType.SPHERICAL, JointType.FLOATING):
            self.rot_param = "euler"

    @property
    def parent_cg_to_joint_vec(self) -> sym.Matrix:
        return self._pcg2j

    @property
    def joint_to_child_cg_vec(self) -> sym.Matrix:
        return self._j2ccg

    @property
    def axis_u_vec(self) -> Optional[sym.Matrix]:
        return self._u

    @property
    def axis_u1_vec(self) -> Optional[sym.Matrix]:
        return self._u1

    @property
    def axis_u2_vec(self) -> Optional[sym.Matrix]:
        return self._u2

    def dof(self) -> int:
        return _DOF_MAP[self.type]

    def cfg_dof(self) -> int:
        """Internal configuration DOF (quaternion for S/F): R=1,P=1,U=2,C=2,S=4,F=7."""
        return _CFG_DOF_MAP[self.type]

    def user_cfg_dof(self) -> int:
        """User-facing configuration size (depends on rot_param for S/F)."""
        if self.type == JointType.SPHERICAL:
            return 4 if self.rot_param == "quat" else 3
        if self.type == JointType.FLOATING:
            return 7 if self.rot_param == "quat" else 6
        return _DOF_MAP[self.type]

    def validate(self, NBodies: int) -> None:
        """
        Validate joint fields with descriptive errors.

        Notes:
        - Axis vectors are defined in the PARENT frame.
        - For U joints: u1 and u2 must be non-collinear (orthonormal preferred).
        """
        if not (0 <= self.parent <= NBodies):
            raise ValueError(
                f"Joint(child={self.child}): parent={self.parent} out of range [0..{NBodies}]."
            )
        if not (1 <= self.child <= NBodies):
            raise ValueError(
                f"Joint(child={self.child}): child={self.child} out of range [1..{NBodies}]."
            )
        if self.parent == self.child:
            raise ValueError(f"Joint(child={self.child}): self-parenting detected (parent==child).")

        # axis requirements
        if self.type in (JointType.REVOLUTE, JointType.PRISMATIC, JointType.CYLINDRICAL):
            if self._u is None:
                raise ValueError(f"Joint(child={self.child}, type={self.type.value}): axis_u is required.")
        if self.type == JointType.UNIVERSAL:
            if self._u1 is None or self._u2 is None:
                raise ValueError(f"Joint(child={self.child}, type=U): axis_u1 and axis_u2 are required.")
            _axes_noncollinear(self._u1, self._u2, ctx=f"Joint(child={self.child}, type=U)")

        # rot_param validation for S/F
        if self.type in (JointType.SPHERICAL, JointType.FLOATING):
            if self.rot_param not in ("euler", "quat"):
                raise ValueError(
                    f"Joint(child={self.child}, type={self.type.value}): "
                    f"rot_param must be 'euler' or 'quat', got {self.rot_param!r}."
                )

    def as_dict(self) -> Dict[str, Any]:
        """Small convenience serializer for debugging/logging."""
        def _v(M: Optional[sym.Matrix]) -> Optional[List[Any]]:
            return None if M is None else [M[i, 0] for i in range(3)]

        return {
            "parent": self.parent,
            "child": self.child,
            "type": self.type.value,
            "parent_cg_to_joint": _v(self._pcg2j),
            "joint_to_child_cg": _v(self._j2ccg),
            "axis_u": _v(self._u),
            "axis_u1": _v(self._u1),
            "axis_u2": _v(self._u2),
            "dof": self.dof(),
            "cfg_dof": self.cfg_dof(),
            "rot_param": self.rot_param,
        }


@dataclass
class JointSystem3D:
    """
    Container for 3D joints + topology cache and column indexing.

    After construction, the system provides:
    - joints sorted by child index
    - per-joint DOF column indexing: col_start, col_slice
    - topology cache: adjacency, parent_of_body, joint_of_body, roots, body_paths, joint_paths, Btrack
    """
    NBodies: int
    joints: List[Joint3D]

    # DOF indexing (0-based columns)
    col_start: List[int]    = field(init=False)
    col_slice: List[slice]  = field(init=False)
    total_dof: int          = field(init=False)

    # Internal configuration indexing (quaternion for S/F)
    cfg_col_start: List[int]    = field(init=False)
    cfg_col_slice: List[slice]  = field(init=False)
    total_cfg_dof: int          = field(init=False)

    # User-facing configuration indexing (depends on rot_param)
    user_col_start: List[int]   = field(init=False)
    user_col_slice: List[slice] = field(init=False)
    total_user_dof: int         = field(init=False)

    # topology cache
    adjacency: Dict[int, List[int]] = field(init=False)
    parent_body_of_body: List[int]  = field(init=False)
    parent_joint_of_body: List[int] = field(init=False)
    roots: List[int]                = field(init=False)
    body_paths: List[List[int]]     = field(init=False)
    joint_paths: List[List[int]]    = field(init=False)
    Btrack: Any                     = field(init=False)  # numpy bool array

    @classmethod
    def from_data(cls, data: Mapping[str, Any]) -> "JointSystem3D":
        """
        Build JointSystem3D from a raw dict.

        Required keys:
            NBodies, joints, types, parent_cg_to_joint, joint_to_child_cg, axis_u
        Optional keys:
            axis_u1, axis_u2 (for universal joints)
        """
        required    = ["NBodies", "joints", "types", "parent_cg_to_joint", "joint_to_child_cg", "axis_u"]
        missing     = [k for k in required if k not in data]

        if missing:
            raise ValueError(f"JointSystem3D.from_data: missing required keys: {missing}")

        # Initialize body and joint vectors
        NBodies = int(data["NBodies"])
        edges   = list(map(tuple, data["joints"]))
        types   = list(data["types"])
        pcg2j   = list(data["parent_cg_to_joint"])
        j2ccg   = list(data["joint_to_child_cg"])

        # Initialize axis vectors for specific joint types
        axis_u      = list(data.get("axis_u", [None] * len(edges)))
        axis_u1     = list(data.get("axis_u1", [None] * len(edges)))
        axis_u2     = list(data.get("axis_u2", [None] * len(edges)))
        rot_params  = list(data.get("rot_param", [None] * len(edges)))

        n = len(edges)
        for name, arr in [
            ("types", types),
            ("parent_cg_to_joint", pcg2j),
            ("joint_to_child_cg", j2ccg),
            ("axis_u", axis_u),
            ("axis_u1", axis_u1),
            ("axis_u2", axis_u2),
            ("rot_param", rot_params),
        ]:
            if len(arr) != n:
                raise ValueError(f"JointSystem3D.from_data: '{name}' length {len(arr)} != number of joints {n}.")

        joints: List[Joint3D] = []
        for i, (p, c) in enumerate(edges):
            try:
                jt = JointType(types[i])
            except Exception as e:
                raise ValueError(f"Joint {i}: invalid joint type '{types[i]}'.") from e

            joints.append(
                Joint3D(
                    parent=int(p),
                    child=int(c),
                    type=jt,
                    parent_cg_to_joint=pcg2j[i],
                    joint_to_child_cg=j2ccg[i],
                    axis_u=axis_u[i],
                    axis_u1=axis_u1[i],
                    axis_u2=axis_u2[i],
                    rot_param=rot_params[i],
                )
            )

        sys = cls(NBodies=NBodies, joints=joints)
        sys._build()
        return sys

    def _build(self) -> None:
        # per-joint validation first (ranges + axes)
        for j in self.joints:
            j.validate(self.NBodies)

        # sort by child id (enables stable indexing and joint_of_body mapping)
        self.joints.sort(key=lambda x: x.child)

        # topology validation on sorted edges (joint index is list index)
        edges = [(j.parent, j.child) for j in self.joints]
        tree_idx: TreeIndex = validate_tree(edges, self.NBodies, ground_id=0)

        self.parent_body_of_body    = tree_idx.parent_body_of_body
        self.parent_joint_of_body   = tree_idx.parent_joint_of_body

        self.adjacency  = build_adjacency(edges, self.NBodies, ground_id=0)
        self.roots      = list(self.adjacency.get(0, []))

        self.body_paths, self.joint_paths = compute_root_to_leaf_joint_paths(
            self.adjacency, tree_idx.child_to_joint, root=0
        )

        # DOF column indexing
        self.col_start  = []
        self.col_slice  = []
        col             = 0
        for ji, j in enumerate(self.joints):
            self.col_start.append(col)
            self.col_slice.append(slice(col, col + j.dof()))
            col += j.dof()
        self.total_dof = col

        # Internal configuration indexing (quaternion for S/F)
        self.cfg_col_start  = []
        self.cfg_col_slice  = []
        cfg_col             = 0
        for ji, j in enumerate(self.joints):
            self.cfg_col_start.append(cfg_col)
            self.cfg_col_slice.append(slice(cfg_col, cfg_col + j.cfg_dof()))
            cfg_col += j.cfg_dof()
        self.total_cfg_dof = cfg_col

        # User-facing configuration indexing
        self.user_col_start = []
        self.user_col_slice = []
        user_col            = 0
        for ji, j in enumerate(self.joints):
            self.user_col_start.append(user_col)
            self.user_col_slice.append(slice(user_col, user_col + j.user_cfg_dof()))
            user_col += j.user_cfg_dof()
        self.total_user_dof = user_col

        # Btrack: (NBodies+1, NJoints)
        self.Btrack = compute_Btrack(
            self.parent_joint_of_body,
            self.parent_body_of_body,
            self.NBodies,
            len(self.joints),
            ground_id=0,
        )

    def get_paths(self) -> Tuple[List[List[int]], List[List[int]]]:
        """Return (body_paths, joint_paths)."""
        return self.body_paths, self.joint_paths

    # --------------- coordinate mapping (user ↔ internal) ---------------

    def map_q_user_to_q_int(self, q_user_np) -> np.ndarray:
        """Map user-facing configuration to internal quaternion-based configuration.

        For R/P/U/C joints, entries are copied directly.
        For S joints:
            euler  → 3 Euler angles ``[phi,theta,psi]`` converted to 4-quaternion.
            quat   → 4 quaternion entries copied (and normalized).
        For F joints:
            Translation ``[x,y,z]`` always copied (3 entries).
            euler  → 3 Euler angles converted to 4-quaternion.
            quat   → 4 quaternion entries copied (and normalized).

        Parameters
        ----------
        q_user_np : array_like, length ``total_user_dof``

        Returns
        -------
        np.ndarray, length ``total_cfg_dof``
        """
        q_user = np.asarray(q_user_np, dtype=float).ravel()
        if q_user.shape[0] != self.total_user_dof:
            raise ValueError(
                f"q_user length mismatch: expected {self.total_user_dof}, "
                f"got {q_user.shape[0]}."
            )
        q_int = np.zeros(self.total_cfg_dof)
        for ji, j in enumerate(self.joints):
            u_sl = self.user_col_slice[ji]
            c_sl = self.cfg_col_slice[ji]
            u_vals = q_user[u_sl]

            if j.type in (JointType.REVOLUTE, JointType.PRISMATIC,
                          JointType.UNIVERSAL, JointType.CYLINDRICAL):
                q_int[c_sl] = u_vals

            elif j.type == JointType.SPHERICAL:
                if j.rot_param == "quat":
                    q_int[c_sl] = u_vals / np.linalg.norm(u_vals)
                else:
                    q_int[c_sl] = euler_to_quat(u_vals[0], u_vals[1], u_vals[2])

            elif j.type == JointType.FLOATING:
                q_int[c_sl.start:c_sl.start + 3] = u_vals[:3]
                if j.rot_param == "quat":
                    qv = u_vals[3:7]
                    q_int[c_sl.start + 3:c_sl.start + 7] = qv / np.linalg.norm(qv)
                else:
                    q_int[c_sl.start + 3:c_sl.start + 7] = euler_to_quat(
                        u_vals[3], u_vals[4], u_vals[5]
                    )
        return q_int

    def map_q_int_to_q_user(self, q_int_np) -> np.ndarray:
        """Map internal quaternion-based configuration to user-facing coordinates.

        Parameters
        ----------
        q_int_np : array_like, length ``total_cfg_dof``

        Returns
        -------
        np.ndarray, length ``total_user_dof``
        """
        q_int = np.asarray(q_int_np, dtype=float).ravel()
        if q_int.shape[0] != self.total_cfg_dof:
            raise ValueError(
                f"q_int length mismatch: expected {self.total_cfg_dof}, "
                f"got {q_int.shape[0]}."
            )
        q_user = np.zeros(self.total_user_dof)
        for ji, j in enumerate(self.joints):
            u_sl = self.user_col_slice[ji]
            c_sl = self.cfg_col_slice[ji]
            c_vals = q_int[c_sl]

            if j.type in (JointType.REVOLUTE, JointType.PRISMATIC,
                          JointType.UNIVERSAL, JointType.CYLINDRICAL):
                q_user[u_sl] = c_vals

            elif j.type == JointType.SPHERICAL:
                if j.rot_param == "quat":
                    q_user[u_sl] = c_vals / np.linalg.norm(c_vals)
                else:
                    q_user[u_sl] = quat_to_euler(c_vals)

            elif j.type == JointType.FLOATING:
                q_user[u_sl.start:u_sl.start + 3] = c_vals[:3]
                if j.rot_param == "quat":
                    qv = c_vals[3:7]
                    q_user[u_sl.start + 3:u_sl.start + 7] = qv / np.linalg.norm(qv)
                else:
                    q_user[u_sl.start + 3:u_sl.start + 6] = quat_to_euler(c_vals[3:7])
        return q_user

    def summary_table(self,precision: int = 3):
        """Return a pandas DataFrame summary (imported lazily)."""
        import pandas as pd

        def v3(M: sym.Matrix) -> List[Any]:
            return [M[i, 0] for i in range(3)]

        rows = []
        for idx, j in enumerate(self.joints):
            rows.append(
                {
                    "joint_index": idx,
                    "parent": j.parent,
                    "child": j.child,
                    "type": j.type.value,
                    "dof": j.dof(),
                    "pcg2j_parent": v3(j.parent_cg_to_joint_vec),
                    "j2ccg_child": v3(j.joint_to_child_cg_vec),
                    "axis_u": None if j.axis_u_vec is None else v3(j.axis_u_vec),
                    "axis_u1": None if j.axis_u1_vec is None else v3(j.axis_u1_vec),
                    "axis_u2": None if j.axis_u2_vec is None else v3(j.axis_u2_vec),
                    "col_slice": str(self.col_slice[idx]),
                    "cfg_col_slice": str(self.cfg_col_slice[idx]),
                    "rot_param": j.rot_param,
                }
            )

        df = pd.DataFrame(rows)
        df = df.map(lambda v: to_display_value(v, nd=precision))

        print(df.to_string(index=False))
