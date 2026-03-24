# source/multibody/velocity_transformation_3d.py
"""
velocity_transformation_3d.py

Topology + indexing utilities for the 3D velocity transformation (B) assembly
and the symbolic rate-kinematics layer that supports Bdot assembly.

This module provides:
- joint DOF bookkeeping
- per-joint slices into the system generalized coordinate vector and B columns
- per-body row slices (6 rows per body)
- a root-to-leaf write schedule (k, j) to assemble B blocks in a stable order,
  while preventing duplicate writes across overlapping root-to-leaf paths
- symbolic position-level kinematics cache (KinematicsCache3D)
- symbolic first-order rate kinematics cache (KinematicsRateCache3D)
- block-kinematics helpers (BlockKinematics3D, BlockRateKinematics3D)

Conventions
-----------
- Ground body id = 0
- Bodies are 1..NBodies
- Joints are stored in JointSystem3D.joints and are assumed sorted by child id
  (JointSystem3D already enforces this).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable, Iterator, List, Optional, Sequence, Tuple, TYPE_CHECKING

import numpy as np
import sympy as sym
from sympy import Identity, MatMul, MatrixSymbol, ZeroMatrix

try:
    # package-style imports
    from .joint_system_3d import JointSystem3D, JointType
    from .topology_3d import build_adjacency, compute_root_to_leaf_joint_paths, validate_tree
    from ._velocity_transformation_helper import skew, _axis_angle_rotation, _A_from_quaternion_sym, RootToLeafPaths, _type_code
except Exception:  # pragma: no cover
    # script-style fallback (matches some existing files in the repo)
    from .joint_system_3d import JointSystem3D, JointType
    from .topology_3d import build_adjacency, compute_root_to_leaf_joint_paths, validate_tree


BodyId      = int
JointIndex  = int
WritePair   = Tuple[BodyId, JointIndex]


@dataclass
class BlockKinematics3D:
    """Kinematic quantities for a single (body k, joint j) B block.

    Attributes
    ----------
    body_id : int
        Body index *k* (1..NBodies).
    joint_index : int
        Joint index *j* (0..NJoints-1).
    d_kj : sym.Matrix
        3×1 vector from joint *j* to body-*k* CG, expressed in the global frame.
    U_j : sym.Matrix
        3×m joint axis / basis in the global frame (m depends on joint type).
    """

    body_id:     int
    joint_index: int
    d_kj:        sym.Matrix
    U_j:         sym.Matrix


@dataclass(frozen=True, slots=True)
class KinematicsCache3D:
    """Symbolic kinematics cache for a 3D joint-coordinate multibody system.

    All matrix products are stored as unevaluated ``MatMul`` expressions so that
    symbolic structure is preserved (no expansion).

    Attributes
    ----------
    A_abs : list[sym.MatrixExpr]
        Absolute rotation matrices, length ``NBodies + 1``.
        ``A_abs[0] = Identity(3)``.
    A_u1 : list[sym.MatrixExpr | sym.MatrixBase]
        Universal joint first rotation axis relative to the parent body, length ``NJoints``.
    r_abs : list[sym.MatrixBase]
        Absolute CG position vectors (3 x 1), length ``NBodies + 1``.
        ``r_abs[0] = zeros(3, 1)``.
    rJ : list[sym.MatrixBase]
        Joint global-frame position vectors (3 x 1), length ``NJoints``.
    U : list[sym.MatrixExpr | sym.MatrixBase]
        Joint axis / basis expressed in the global frame, length ``NJoints``.
        Shape depends on joint type: R/P/C → 3 x 1, U → 3 x 2, S → 3 x 3, F → 3 x 6.
    Arel : list[sym.MatrixSymbol]
        Opaque relative rotation symbols, length ``NJoints``.
    parent_of_body : list[int]
        ``parent_of_body[b]`` = parent body id.  Index 0 is ground (0).
    joint_of_body : list[int]
        ``joint_of_body[b]`` = joint index connecting parent → b.  Index 0 = -1.
    """
    A_abs:          List[Any]
    A_u1:           List[Any]
    r_abs:          List[Any]
    rJ:             List[Any]
    U:              List[Any]
    Arel:           List[Any]
    parent_of_body: List[int]
    joint_of_body:  List[int]


@dataclass(frozen=True, slots=True)
class KinematicsRateCache3D:
    """Symbolic first-order rate kinematics cache for a 3D multibody system.

    All quantities are expressed in the global (inertial) frame and computed
    via a single topological pass consistent with :class:`KinematicsCache3D`.

    Attributes
    ----------
    omega_abs : list[Any]
        Absolute angular velocity vectors (3 x 1), length ``NBodies + 1``.
        ``omega_abs[0] = zeros(3, 1)`` (ground).
    v_abs : list[Any]
        Absolute CG linear velocity vectors (3 x 1), length ``NBodies + 1``.
        ``v_abs[0] = zeros(3, 1)`` (ground).
    vJ : list[Any]
        Absolute linear velocity of each joint point (3 x 1), length ``NJoints``.
    Udot : list[Any]
        Time derivative of each joint axis / basis in the global frame,
        length ``NJoints``.  Shape mirrors :attr:`KinematicsCache3D.U`.
    """

    omega_abs: List[Any]
    v_abs:     List[Any]
    vJ:        List[Any]
    Udot:      List[Any]


@dataclass
class BlockRateKinematics3D:
    """Rate-kinematic quantities for a single (body k, joint j) Bdot block.

    Attributes
    ----------
    body_id : int
        Body index *k* (1..NBodies).
    joint_index : int
        Joint index *j* (0..NJoints-1).
    d_dot_kj : sym.Matrix
        3×1 time derivative of the position vector from joint *j* to body-*k* CG,
        in the global frame.  Equal to ``v_abs[k] - vJ[j]``.
    U_dot_j : sym.Matrix
        3×m time derivative of the joint axis / basis in the global frame.
    """

    body_id:    int
    joint_index: int
    d_dot_kj:   sym.Matrix
    U_dot_j:    sym.Matrix


class VelocityTransformation3D:
    """
    Topology/indexing layer for 3D velocity transformation assembly.

    Parameters
    ----------
    joint_system:
        A JointSystem3D describing a rooted tree with ground=0 and bodies 1..NBodies.

    Attributes (selected)
    ---------------------
    NBodies, NJoints:
        Number of bodies (excluding ground) and number of joints.

    joint_dof_by_code:
        Joint DOF per type code: R=1,P=1,U=2,C=2,S=3,F=6.

    q_slices[j], col_slices[j]:
        slice objects into the system coordinate vector / B columns for joint j.

    Btrack:
        Boolean write-tracker matrix of shape (NBodies+1, NJoints). Mutated by
        iter_write_pairs_root_to_leaf() to prevent duplicate writes of block (k,j).

    paths:
        Root-to-leaf joint paths (list of joint-index sequences).
    """

    # Required DOF mapping (by 1-letter code)
    _DOF_BY_CODE = {
        "R": 1,
        "P": 1,
        "U": 2,
        "C": 2,
        "S": 3,
        "F": 6,
    }

    def __init__(self, joint_system: JointSystem3D):
        self.joint_system: JointSystem3D = joint_system

        self.NBodies: int = int(getattr(joint_system, "NBodies"))
        self.NJoints: int = len(getattr(joint_system, "joints"))

        # DOF bookkeeping
        self.joint_dof_by_code: dict[str, int] = dict(self._DOF_BY_CODE)

        self.joint_dof: List[int] = []
        for j in self.joint_system.joints:
            code = _type_code(j.type)
            try:
                self.joint_dof.append(self.joint_dof_by_code[code])
            except KeyError as e:
                raise ValueError(f"Unsupported joint type code {code!r}.") from e

        # Slices into B columns (speed DOF)
        self.col_slices: List[slice]    = list(getattr(joint_system, "col_slice"))
        # Slices into internal configuration vector q_int (quaternion for S/F)
        self.q_slices: List[slice]      = list(getattr(joint_system, "cfg_col_slice"))
        self.total_dof: int             = int(getattr(joint_system, "total_dof"))
        self.total_cfg_dof: int         = int(getattr(joint_system, "total_cfg_dof"))

        # Root-to-leaf traversal structure
        self.body_paths: List[List[int]]    = getattr(joint_system, "body_paths")
        self.joint_paths: List[List[int]]   = getattr(joint_system, "joint_paths")

        # Expose joint-index paths as requested
        self.paths: List[List[int]]         = self.joint_paths

        # Write-tracker for (k,j) blocks
        self.Btrack: np.ndarray             = np.zeros((self.NBodies + 1, self.NJoints), dtype=bool)

    def reset_Btrack(self) -> None:
        """Reset the internal (k,j) write tracker to all False."""
        self.Btrack.fill(False)

    def iter_write_pairs_root_to_leaf(self) -> Iterator[WritePair]:
        """
        Yield (k, j) pairs in root-to-leaf order for later B block assembly.

        Schedule
        --------
        For each root-to-leaf path:
          for each joint j on that path (in path order):
            for each downstream body k after that joint on the same path:
              if not Btrack[k, j]:
                yield (k, j) and set Btrack[k, j] = True

        This method does not compute any kinematics; it only emits a stable
        write order and prevents duplicates across overlapping paths.
        """
        for body_path, joint_path in zip(self.body_paths, self.joint_paths):
            # body_path: [b0, b1, ...]
            # joint_path: [j(b0), j(b1), ...] aligned indices
            for i, j in enumerate(joint_path):
                for k in body_path[i:]:
                    if not self.Btrack[k, j]:
                        self.Btrack[k, j] = True
                        yield (k, j)

    # -------------------- B block helpers --------------------

    def _get_block_kinematics(
        self,
        cache: KinematicsCache3D,
        k: int,
        j: int,
    ) -> BlockKinematics3D:
        """Extract pre-computed kinematic quantities for block (k, j) from the cache.

        Parameters
        ----------
        cache : KinematicsCache3D
            Symbolic kinematics cache built by :meth:`build_cache_symbolic`.
        k : int
            Body index (1..NBodies).
        j : int
            Joint index (0..NJoints-1).

        Returns
        -------
        BlockKinematics3D
            Contains ``d_kj = r_abs[k] - rJ[j]`` and ``U_j`` as explicit
            ``sym.Matrix`` objects ready for :meth:`_block_B`.
        """
        d_kj = sym.Matrix(cache.r_abs[k] - cache.rJ[j])
        U_j  = sym.Matrix(cache.U[j])
        return BlockKinematics3D(body_id=k, joint_index=j, d_kj=d_kj, U_j=U_j)

    # -------------------- S/F Udot helpers --------------------

    @staticmethod
    def _udot_spherical(omega_p: sym.Matrix, U_j: sym.Matrix) -> sym.Matrix:
        """Time derivative of the S-joint basis in the global frame.

        The spherical basis is the parent-frame identity rotated into global::

            U_j = A_p                 (3x3)
            Udot_j = skew(omega_p) * A_p = skew(omega_p) * U_j

        Parameters
        ----------
        omega_p : sym.Matrix (3,1)
            Absolute angular velocity of the parent body.
        U_j : sym.Matrix (3,3)
            Current joint basis (= A_p evaluated symbolically).

        Returns
        -------
        sym.Matrix (3,3)
        """
        return skew(omega_p) * U_j

    @staticmethod
    def _udot_floating(omega_p: sym.Matrix, U_j: sym.Matrix) -> sym.Matrix:
        """Time derivative of the F-joint basis in the global frame.

        The floating basis is 3×6 (translation | rotation columns), both
        spanned by A_p::

            U_j = [A_p | A_p]              (3x6)
            Udot_j = skew(omega_p) * U_j   (3x6)

        Parameters
        ----------
        omega_p : sym.Matrix (3,1)
            Absolute angular velocity of the parent body.
        U_j : sym.Matrix (3,6)
            Current joint basis.

        Returns
        -------
        sym.Matrix (3,6)
        """
        return skew(omega_p) * U_j

    # -------------------- rate cache --------------------

    def build_rate_cache_symbolic(
        self,
        q: sym.Matrix,
        qd: sym.Matrix,
        cache: Optional[KinematicsCache3D] = None,
    ) -> KinematicsRateCache3D:
        """Build a symbolic first-order rate kinematics cache.

        Parameters
        ----------
        q : sym.Matrix, shape ``(total_cfg_dof, 1)``
            Internal configuration vector consistent with ``self.q_slices``.
        qd : sym.Matrix, shape ``(total_dof, 1)``
            Generalized speed vector consistent with ``self.col_slices``.
        cache : KinematicsCache3D, optional
            Position-level cache.  Built automatically when *None*.

        Returns
        -------
        KinematicsRateCache3D
            Contains ``omega_abs``, ``v_abs``, ``vJ``, ``Udot``.

        Notes
        -----
        * The loop follows the same topological order as
          :meth:`build_cache_symbolic`: joints sorted by child guarantees
          parents are processed before children.
        * All quantities are expressed in the global (inertial) frame.
        * Ground (body 0) has zero velocity and zero angular velocity.
        """
        q  = sym.Matrix(q)
        qd = sym.Matrix(qd)

        if q.shape != (self.total_cfg_dof, 1):
            raise ValueError(
                f"q shape mismatch: expected ({self.total_cfg_dof}, 1), got {q.shape}."
            )
        if qd.shape != (self.total_dof, 1):
            raise ValueError(
                f"qd shape mismatch: expected ({self.total_dof}, 1), got {qd.shape}."
            )

        if cache is None:
            cache = self.build_cache_symbolic(q)

        joints = self.joint_system.joints
        NB     = self.NBodies
        NJ     = self.NJoints

        # Ground initial conditions
        omega_abs: List[Any] = [None] * (NB + 1)
        v_abs:     List[Any] = [None] * (NB + 1)
        omega_abs[0] = sym.zeros(3, 1)
        v_abs[0]     = sym.zeros(3, 1)

        vJ:   List[Any] = [None] * NJ
        Udot: List[Any] = [None] * NJ

        for j_idx, jnt in enumerate(joints):
            p    = jnt.parent
            c    = jnt.child
            code = _type_code(jnt.type)

            omega_p = omega_abs[p]          # already computed
            v_p     = v_abs[p]

            # Generalised speed entries for this joint
            col_sl  = self.col_slices[j_idx]    # speed DOF slice
            cfg_sl  = self.q_slices[j_idx]      # cfg DOF slice (for C/F translation)

            # ----------------------------------------------------------------
            # Joint-point velocity: vJ = v_p + skew(omega_p) * (rJ - r_p)
            # ----------------------------------------------------------------
            r_p  = sym.Matrix(cache.r_abs[p])
            rJ_j = sym.Matrix(cache.rJ[j_idx])
            vJ[j_idx] = v_p + skew(omega_p) * (rJ_j - r_p)

            # ----------------------------------------------------------------
            # Axes in global frame (explicit sym.Matrix)
            # ----------------------------------------------------------------
            if code in ("R", "P", "C"):
                U_j = sym.Matrix(cache.U[j_idx])   # 3×1

            # ----------------------------------------------------------------
            # Per-type: child angular velocity, child CG velocity, Udot
            # ----------------------------------------------------------------
            if code == "R":
                qd_j        = qd[col_sl.start, 0]          # θ̇
                omega_c     = omega_p + U_j * qd_j
                v_c         = vJ[j_idx] + skew(omega_c) * sym.Matrix(
                                  cache.r_abs[c] - cache.rJ[j_idx])
                Udot[j_idx] = skew(omega_p) * U_j

            elif code == "P":
                qd_j        = qd[col_sl.start, 0]          # ṡ
                omega_c     = omega_p                       # no rotation
                v_c         = vJ[j_idx] + U_j * qd_j + skew(omega_c) * sym.Matrix(
                                  cache.r_abs[c] - cache.rJ[j_idx])
                Udot[j_idx] = skew(omega_p) * U_j

            elif code == "U":
                # U_j is 3×2 (explicit matrix from cache)
                U2 = sym.Matrix(cache.U[j_idx])             # 3×2
                u1_g = U2[:, 0]                             # 3×1
                u2_g = U2[:, 1]                             # 3×1

                qd1 = qd[col_sl.start, 0]
                qd2 = qd[col_sl.start + 1, 0]

                # Corrected axis transport (see module docstring)
                u1_dot = skew(omega_p) * u1_g
                omega_after_u1 = omega_p + qd1 * u1_g
                u2_dot = skew(omega_after_u1) * u2_g

                Udot_u = sym.zeros(3, 2)
                Udot_u[:, 0] = u1_dot
                Udot_u[:, 1] = u2_dot
                Udot[j_idx] = Udot_u

                omega_c = omega_p + qd1 * u1_g + qd2 * u2_g
                v_c     = vJ[j_idx] + skew(omega_c) * sym.Matrix(
                              cache.r_abs[c] - cache.rJ[j_idx])

            elif code == "C":
                qd_rot  = qd[col_sl.start, 0]              # θ̇ (rotation)
                qd_trans = qd[col_sl.start + 1, 0]         # ṡ (translation)
                omega_c  = omega_p + U_j * qd_rot
                v_c      = vJ[j_idx] + U_j * qd_trans + skew(omega_c) * sym.Matrix(
                               cache.r_abs[c] - cache.rJ[j_idx])
                Udot[j_idx] = skew(omega_p) * U_j

            elif code == "S":
                # U_j = A_p  (3×3), basis = 3×3
                U_j_s   = sym.Matrix(cache.U[j_idx])           # 3×3
                qd_s    = sym.Matrix([qd[col_sl.start + i, 0] for i in range(3)])  # ω_rel
                omega_c = omega_p + U_j_s * qd_s
                v_c     = vJ[j_idx] + skew(omega_c) * sym.Matrix(
                              cache.r_abs[c] - cache.rJ[j_idx])
                Udot[j_idx] = self._udot_spherical(omega_p, U_j_s)

            elif code == "F":
                # U_j = [A_p | A_p]  (3×6)
                U_j_f   = sym.Matrix(cache.U[j_idx])           # 3×6
                # translational DOFs are the first 3 speed entries
                qd_t    = sym.Matrix([qd[col_sl.start + i, 0] for i in range(3)])
                # rotational DOFs are the last 3 speed entries
                qd_r    = sym.Matrix([qd[col_sl.start + 3 + i, 0] for i in range(3)])
                A_p     = sym.Matrix(cache.A_abs[p])
                omega_c = omega_p + A_p * qd_r
                v_c     = vJ[j_idx] + A_p * qd_t + skew(omega_c) * sym.Matrix(
                              cache.r_abs[c] - cache.rJ[j_idx])
                Udot[j_idx] = self._udot_floating(omega_p, U_j_f)

            else:
                raise ValueError(
                    f"Unsupported joint code {code!r} in build_rate_cache_symbolic."
                )

            omega_abs[c] = omega_c
            v_abs[c]     = v_c

        return KinematicsRateCache3D(
            omega_abs=omega_abs,
            v_abs=v_abs,
            vJ=vJ,
            Udot=Udot,
        )

    def _get_block_rate_kinematics(
        self,
        cache: KinematicsCache3D,
        rate_cache: KinematicsRateCache3D,
        k: int,
        j: int,
    ) -> BlockRateKinematics3D:
        """Extract rate-kinematic quantities for block (k, j) from the rate cache.

        Parameters
        ----------
        cache : KinematicsCache3D
            Position-level cache (unused here but provided for API consistency).
        rate_cache : KinematicsRateCache3D
            Rate cache built by :meth:`build_rate_cache_symbolic`.
        k : int
            Body index (1..NBodies).
        j : int
            Joint index (0..NJoints-1).

        Returns
        -------
        BlockRateKinematics3D
            Contains ``d_dot_kj = v_abs[k] - vJ[j]`` and ``U_dot_j`` as
            explicit ``sym.Matrix`` objects, ready for Bdot block assembly.
        """
        d_dot_kj = sym.Matrix(rate_cache.v_abs[k] - rate_cache.vJ[j])
        U_dot_j  = sym.Matrix(rate_cache.Udot[j])
        return BlockRateKinematics3D(
            body_id=k, joint_index=j, d_dot_kj=d_dot_kj, U_dot_j=U_dot_j
        )

    def _block_B(
        self,
        joint: "Joint3D",
        d_kj: sym.Matrix,
        U_j: sym.Matrix,
    ) -> sym.Matrix:
        """Return the 6 x m symbolic B block for body k / joint j.

        Parameters
        ----------
        joint : Joint3D
            The joint object (used only for its type).
        d_kj : sympy.Matrix (3, 1)
            Position vector from joint j to body k CG, in the global frame.
        U_j : sympy.Matrix (3, m)
            Joint axis / basis expressed in the global frame (from the cache).

        Returns
        -------
        sympy.Matrix (6, m)
            The velocity-transformation block::

                R:  [[-skew(d)*u], [u]]           6x1
                P:  [[u], [0_3x1]]                6x1
                S:  [[-skew(d)], [I]]             6x3
                U:  [[-skew(d)*U], [U]]           6x2
                C:  [[-skew(d)*u, u], [u, 0]]     6x2
                F:  [[I, -skew(d)], [0, I]]       6x6
        """
        code    = _type_code(joint.type)
        d_tilde = skew(d_kj)
        I3      = sym.eye(3)
        Z3      = sym.zeros(3)

        if code == "R":
            # U_j is 3x1
            return sym.Matrix.vstack(-d_tilde * U_j, U_j)

        if code == "P":
            # U_j is 3x1
            return sym.Matrix.vstack(U_j, sym.zeros(3, 1))

        if code == "S":
            # 6x3: [[-skew(d)], [I]]
            return sym.Matrix.vstack(-d_tilde, I3)

        if code == "U":
            # U_j is 3x2
            return sym.Matrix.vstack(-d_tilde * U_j, U_j)

        if code == "C":
            # U_j is 3x1 (single axis); DOFs = [rotation, translation]
            u   = U_j
            top = sym.Matrix.hstack(-d_tilde * u, u)
            bot = sym.Matrix.hstack(u, sym.zeros(3, 1))
            return sym.Matrix.vstack(top, bot)

        if code == "F":
            # 6x6: [[I, -skew(d)], [0, I]]
            top = sym.Matrix.hstack(I3, -d_tilde)
            bot = sym.Matrix.hstack(Z3, I3)
            return sym.Matrix.vstack(top, bot)

        raise ValueError(f"Unsupported joint code {code!r} in _block_B.")

    # -------------------- symbolic B assembly --------------------

    def assemble_B_symbolic(
        self,
        q: sym.Matrix,
        cache: Optional[KinematicsCache3D] = None,
    ) -> sym.Matrix:
        """Assemble the full symbolic velocity-transformation matrix B.

        Parameters
        ----------
        q : sympy.Matrix (total_cfg_dof, 1)
            Internal generalized-coordinate vector (quaternion for S/F).
        cache : KinematicsCache3D, optional
            Pre-built kinematics cache.  Built automatically when *None*.

        Returns
        -------
        sympy.Matrix, shape ``(6*NBodies, total_dof)``
            Body *k* (1..NBodies) occupies rows ``6*(k-1) .. 6*(k-1)+5``.
        """
        if cache is None:
            cache = self.build_cache_symbolic(q)

        # Unpack and initialize
        NB      = self.NBodies
        n_rows  = 6 * NB
        B       = sym.zeros(n_rows, self.total_dof)
        joints  = self.joint_system.joints

        # Reset Btrack so iter_write_pairs_root_to_leaf starts clean.
        self.reset_Btrack()

        for k, j in self.iter_write_pairs_root_to_leaf():
            bk      = self._get_block_kinematics(cache, k, j)
            block   = self._block_B(joints[j], bk.d_kj, bk.U_j)

            # Row slice for body k (body k=1 starts at row 0)
            r0 = 6 * (k - 1)
            c0 = self.col_slices[j].start
            c1 = self.col_slices[j].stop

            B[r0:r0 + 6, c0:c1] = block

        return B

    # -------------------- Bdot block helpers --------------------

    def _block_Bdot(
        self,
        joint: "Joint3D",
        d_kj: sym.Matrix,
        d_dot_kj: sym.Matrix,
        U_j: sym.Matrix,
        U_dot_j: sym.Matrix,
    ) -> sym.Matrix:
        """Return the 6 x m symbolic Bdot block for body k / joint j.

        Parameters
        ----------
        joint : Joint3D
            The joint object (used only for its type).
        d_kj : sympy.Matrix (3, 1)
            Position vector from joint j to body k CG, in the global frame.
        d_dot_kj : sympy.Matrix (3, 1)
            Time derivative of d_kj.
        U_j : sympy.Matrix (3, m)
            Joint axis / basis in the global frame.
        U_dot_j : sympy.Matrix (3, m)
            Time derivative of U_j.

        Returns
        -------
        sympy.Matrix (6, m)
        """
        code = _type_code(joint.type)

        if code == "R":
            # 6x1
            top = -skew(d_dot_kj) * U_j - skew(d_kj) * U_dot_j
            return sym.Matrix.vstack(top, U_dot_j)

        if code == "P":
            # 6x1
            return sym.Matrix.vstack(U_dot_j, sym.zeros(3, 1))

        if code == "U":
            # 6x2
            top = -skew(d_dot_kj) * U_j - skew(d_kj) * U_dot_j
            return sym.Matrix.vstack(top, U_dot_j)

        if code == "C":
            # U_j is 3x1 (single axis); DOFs = [rotation, translation]
            u     = U_j
            u_dot = U_dot_j
            top = sym.Matrix.hstack(
                -skew(d_dot_kj) * u - skew(d_kj) * u_dot, u_dot
            )
            bot = sym.Matrix.hstack(u_dot, sym.zeros(3, 1))
            return sym.Matrix.vstack(top, bot)

        if code == "S":
            # B = vstack(-skew(d_kj), I) → Bdot = vstack(-skew(d_dot_kj), 0)
            return sym.Matrix.vstack(-skew(d_dot_kj), sym.zeros(3, 3))

        if code == "F":
            # B = [[I, -skew(d)], [0, I]]
            # Bdot = [[0, -skew(d_dot)], [0, 0]]
            Z3 = sym.zeros(3)
            top = sym.Matrix.hstack(Z3, -skew(d_dot_kj))
            bot = sym.Matrix.hstack(Z3, Z3)
            return sym.Matrix.vstack(top, bot)

        raise ValueError(f"Unsupported joint code {code!r} in _block_Bdot.")

    # -------------------- symbolic Bdot assembly --------------------

    def assemble_Bdot_symbolic(
        self,
        q: sym.Matrix,
        qd: sym.Matrix,
        cache: Optional[KinematicsCache3D] = None,
        rate_cache: Optional[KinematicsRateCache3D] = None,
    ) -> sym.Matrix:
        """Assemble the full symbolic Bdot matrix from per-block formulas.

        Parameters
        ----------
        q : sympy.Matrix (total_cfg_dof, 1)
            Internal generalized-coordinate vector.
        qd : sympy.Matrix (total_dof, 1)
            Generalized speed vector.
        cache : KinematicsCache3D, optional
            Position-level cache.  Built automatically when *None*.
        rate_cache : KinematicsRateCache3D, optional
            Rate cache.  Built automatically when *None*.

        Returns
        -------
        sympy.Matrix, shape ``(6*NBodies, total_dof)``
        """
        if cache is None:
            cache = self.build_cache_symbolic(q)
        if rate_cache is None:
            rate_cache = self.build_rate_cache_symbolic(q, qd, cache=cache)

        NB     = self.NBodies
        n_rows = 6 * NB
        Bdot   = sym.zeros(n_rows, self.total_dof)
        joints = self.joint_system.joints

        self.reset_Btrack()

        for k, j in self.iter_write_pairs_root_to_leaf():
            bk  = self._get_block_kinematics(cache, k, j)
            brk = self._get_block_rate_kinematics(cache, rate_cache, k, j)

            block = self._block_Bdot(
                joints[j],
                bk.d_kj,
                brk.d_dot_kj,
                bk.U_j,
                brk.U_dot_j,
            )

            r0 = 6 * (k - 1)
            c0 = self.col_slices[j].start
            c1 = self.col_slices[j].stop

            Bdot[r0:r0 + 6, c0:c1] = block

        return Bdot

    # -------------------- compilation --------------------

    def compile_B_lambdified(self, q_syms: sym.Matrix) -> callable:
        """Compile the velocity-transformation matrix **B** to a fast NumPy callable.

        Steps
        -----
        1. Build ``B_symbolic(q_syms)`` using opaque ``Arel`` symbols.
        2. Substitute each ``Arel[j]`` with an explicit 3×3 rotation matrix
           derived from the internal configuration coordinates.
        3. ``sympy.cse`` + ``lambdify`` to NumPy.

        Parameters
        ----------
        q_syms : sympy.Matrix (total_cfg_dof, 1)
            Column vector of symbolic *internal* configuration coordinates
            (quaternions for S/F joints).

        Returns
        -------
        callable
            ``B_func(q_int_np) -> numpy.ndarray`` where *q_int_np* is a 1-D
            array of length ``total_cfg_dof``.  The returned array has shape
            ``(6*NBodies, total_dof)``.

        Rotation parameterization per joint type
        -----------------------------------------
        R (Revolute)   : axis-angle about ``u_local``, angle = θ.
        P (Prismatic)  : Identity (no rotation).
        U (Universal)  : ``R(u1, θ₁) · R(u2, θ₂)``.
        C (Cylindrical): axis-angle about ``u_local``, angle = θ (first DOF).
        S (Spherical)  : unit quaternion ``[e0,e1,e2,e3]`` → polynomial ``R(q)``.
        F (Floating)   : q_int = ``[x,y,z, e0,e1,e2,e3]``.
                         First 3 entries are translation (parent frame).
                         Last 4 entries are unit quaternion rotation.
        """
        q_syms = sym.Matrix(q_syms)
        if q_syms.shape != (self.total_cfg_dof, 1):
            raise ValueError(
                f"q_syms shape mismatch: expected ({self.total_cfg_dof}, 1), "
                f"got {q_syms.shape}."
            )

        # 1. Symbolic B with opaque Arel
        cache = self.build_cache_symbolic(q_syms)
        B_sym = self.assemble_B_symbolic(q_syms, cache=cache)

        # 2. Build explicit Arel rotation matrices and substitution dict
        joints          = self.joint_system.joints
        subs_dict: dict = {}

        for j_idx, jnt in enumerate(joints):
            code        = _type_code(jnt.type)
            sl          = self.q_slices[j_idx]
            Arel_sym    = cache.Arel[j_idx]

            if code == "R":
                theta       = q_syms[sl.start, 0]
                Arel_expl   = _axis_angle_rotation(jnt.axis_u_vec, theta)

            elif code == "P":
                Arel_expl = sym.eye(3)

            elif code == "U":
                theta1  = q_syms[sl.start, 0]
                theta2  = q_syms[sl.start + 1, 0]

                R1      = _axis_angle_rotation(jnt.axis_u1_vec, theta1)
                R2      = _axis_angle_rotation(jnt.axis_u2_vec, theta2)

                # substitute A_u1 first
                A_u1_sym = cache.A_u1[j_idx]
                for r in range(3):
                    for c in range(3):
                        subs_dict[A_u1_sym[r, c]] = R1[r, c]

                # substitute Arel
                Arel_expl = R1 * R2

            elif code == "C":
                theta       = q_syms[sl.start, 0]
                Arel_expl   = _axis_angle_rotation(jnt.axis_u_vec, theta)

            elif code == "S":
                # q_int slice has 4 quaternion entries [e0,e1,e2,e3]
                e0          = q_syms[sl.start, 0]
                e           = q_syms[sl.start+1:sl.start+4, 0]
                Arel_expl   = _A_from_quaternion_sym(e0, e)

            elif code == "F":
                # q_int slice: [x,y,z, e0,e1,e2,e3]; quaternion at offset +3
                e0          = q_syms[sl.start + 3, 0]
                e           = q_syms[sl.start+4:sl.start+7, 0]
                Arel_expl   = _A_from_quaternion_sym(e0, e)

            else:
                raise ValueError(
                    f"Unsupported joint code {code!r} in compile_B_lambdified."
                )

            # Element-wise substitution for the 3x3 MatrixSymbol
            for r in range(3):
                for c in range(3):
                    subs_dict[Arel_sym[r, c]] = Arel_expl[r, c]

        B_explicit = B_sym.subs(subs_dict)

        # 3 & 4. Lambdify with CSE to numpy
        q_flat = [q_syms[i, 0] for i in range(self.total_cfg_dof)]
        _B_raw = sym.lambdify(q_flat, B_explicit, modules="numpy", cse=True)

        n_rows = 6 * self.NBodies
        n_cols = self.total_dof

        def B_func(q_int_np):
            """Evaluate B at a numeric q_int vector (1-D array of length total_cfg_dof)."""
            q_int_np = np.asarray(q_int_np, dtype=float).ravel()
            with np.errstate(invalid="ignore", divide="ignore"):
                raw = _B_raw(*q_int_np)
            return np.asarray(raw, dtype=float).reshape(n_rows, n_cols)

        return B_func

    def compile_Bdot_lambdified(
        self,
        q_syms: sym.Matrix,
        qd_syms: sym.Matrix,
    ) -> callable:
        """Compile the time-derivative **Bdot** to a fast NumPy callable.

        Mirrors :meth:`compile_B_lambdified` but for the Bdot matrix.

        Parameters
        ----------
        q_syms : sympy.Matrix (total_cfg_dof, 1)
            Symbolic internal configuration coordinates.
        qd_syms : sympy.Matrix (total_dof, 1)
            Symbolic generalized speed coordinates.

        Returns
        -------
        callable
            ``Bdot_func(q_int_np, qd_np) -> numpy.ndarray`` where
            *q_int_np* has length ``total_cfg_dof`` and *qd_np* has length
            ``total_dof``.  The returned array has shape
            ``(6*NBodies, total_dof)``.
        """
        q_syms  = sym.Matrix(q_syms)
        qd_syms = sym.Matrix(qd_syms)

        if q_syms.shape != (self.total_cfg_dof, 1):
            raise ValueError(
                f"q_syms shape mismatch: expected ({self.total_cfg_dof}, 1), "
                f"got {q_syms.shape}."
            )
        if qd_syms.shape != (self.total_dof, 1):
            raise ValueError(
                f"qd_syms shape mismatch: expected ({self.total_dof}, 1), "
                f"got {qd_syms.shape}."
            )

        # 1. Symbolic Bdot with opaque Arel
        cache      = self.build_cache_symbolic(q_syms)
        rate_cache = self.build_rate_cache_symbolic(q_syms, qd_syms, cache=cache)
        Bdot_sym   = self.assemble_Bdot_symbolic(
            q_syms, qd_syms, cache=cache, rate_cache=rate_cache,
        )

        # 2. Build explicit Arel rotation matrices and substitution dict
        joints          = self.joint_system.joints
        subs_dict: dict = {}

        for j_idx, jnt in enumerate(joints):
            code        = _type_code(jnt.type)
            sl          = self.q_slices[j_idx]
            Arel_sym    = cache.Arel[j_idx]

            if code == "R":
                theta       = q_syms[sl.start, 0]
                Arel_expl   = _axis_angle_rotation(jnt.axis_u_vec, theta)

            elif code == "P":
                Arel_expl = sym.eye(3)

            elif code == "U":
                theta1  = q_syms[sl.start, 0]
                theta2  = q_syms[sl.start + 1, 0]

                R1      = _axis_angle_rotation(jnt.axis_u1_vec, theta1)
                R2      = _axis_angle_rotation(jnt.axis_u2_vec, theta2)

                A_u1_sym = cache.A_u1[j_idx]
                for r in range(3):
                    for c in range(3):
                        subs_dict[A_u1_sym[r, c]] = R1[r, c]

                Arel_expl = R1 * R2

            elif code == "C":
                theta       = q_syms[sl.start, 0]
                Arel_expl   = _axis_angle_rotation(jnt.axis_u_vec, theta)

            elif code == "S":
                e0          = q_syms[sl.start, 0]
                e           = q_syms[sl.start+1:sl.start+4, 0]
                Arel_expl   = _A_from_quaternion_sym(e0, e)

            elif code == "F":
                e0          = q_syms[sl.start + 3, 0]
                e           = q_syms[sl.start+4:sl.start+7, 0]
                Arel_expl   = _A_from_quaternion_sym(e0, e)

            else:
                raise ValueError(
                    f"Unsupported joint code {code!r} in compile_Bdot_lambdified."
                )

            for r in range(3):
                for c in range(3):
                    subs_dict[Arel_sym[r, c]] = Arel_expl[r, c]

        Bdot_explicit = Bdot_sym.subs(subs_dict)

        # 3. Lambdify over concatenated [q_syms, qd_syms]
        q_flat  = [q_syms[i, 0]  for i in range(self.total_cfg_dof)]
        qd_flat = [qd_syms[i, 0] for i in range(self.total_dof)]
        all_flat = q_flat + qd_flat

        _Bdot_raw = sym.lambdify(all_flat, Bdot_explicit, modules="numpy", cse=True)

        n_rows = 6 * self.NBodies
        n_cols = self.total_dof

        def Bdot_func(q_int_np, qd_np):
            """Evaluate Bdot at numeric q_int and qd vectors."""
            q_int_np = np.asarray(q_int_np, dtype=float).ravel()
            qd_np    = np.asarray(qd_np, dtype=float).ravel()
            args     = np.concatenate([q_int_np, qd_np])
            with np.errstate(invalid="ignore", divide="ignore"):
                raw = _Bdot_raw(*args)
            return np.asarray(raw, dtype=float).reshape(n_rows, n_cols)

        return Bdot_func

    # -------------------- symbolic kinematics cache --------------------

    def build_cache_symbolic(self, q: sym.Matrix) -> KinematicsCache3D:
        """Build a symbolic kinematics cache (no B assembly).

        Parameters
        ----------
        q : sympy.Matrix
            Internal configuration vector, shape ``(total_cfg_dof, 1)``.
            Must be consistent with ``self.q_slices`` (cfg_col_slice).

        Returns
        -------
        KinematicsCache3D
            Symbolic cache with ``A_abs``, ``r_abs``, ``rJ``, ``U``, ``Arel``.

        Notes
        -----
        * Relative rotations are *opaque*: ``Arel[j] = MatrixSymbol(...)``.
        * All products use ``MatMul(..., evaluate=False)`` to suppress expansion.
        * Prismatic / cylindrical translation terms use the translational DOF
          extracted from *q* via ``self.q_slices``.
        """
        q = sym.Matrix(q)
        if q.shape != (self.total_cfg_dof, 1):
            raise ValueError(
                f"q shape mismatch: expected ({self.total_cfg_dof}, 1), got {q.shape}."
            )

        joints  = self.joint_system.joints
        NB      = self.NBodies
        NJ      = self.NJoints
        I3      = Identity(3)

        # parent / joint-of-body arrays from joint_system
        parent_of_body: List[int]   = list(self.joint_system.parent_body_of_body)
        joint_of_body: List[int]    = list(self.joint_system.parent_joint_of_body)

        # Opaque relative rotations
        Arel: List[MatrixSymbol] = [
            MatrixSymbol(f"Arel_{j}", 3, 3) for j in range(NJ)
        ]

        # Absolute rotations & positions (indexed by body id 0..NBodies)
        A_abs: List[Any]    = [None] * (NB + 1)
        A_u1:  List[Any]    = [None] * NJ
        r_abs: List[Any]    = [None] * (NB + 1)
        A_abs[0]            = I3
        r_abs[0]            = sym.zeros(3, 1)

        # Joint quantities (indexed by joint index 0..NJ-1)
        rJ: List[Any]   = [None] * NJ
        U: List[Any]    = [None] * NJ

        # Process joints in topological order (sorted by child ensures parent done first
        # because in a rooted tree child > parent when joints sorted by child).
        for j_idx, jnt in enumerate(joints):
            p       = jnt.parent
            c       = jnt.child
            code    = _type_code(jnt.type)

            A_p     = A_abs[p]                                   # already computed
            r_p     = r_abs[p]

            # Local geometry vectors (already sym.Matrix(3,1) via Joint3D.__post_init__)
            p2j     = jnt.parent_cg_to_joint_vec                 # parent frame
            j2c     = jnt.joint_to_child_cg_vec                  # child frame

            # ---- absolute rotation: A_abs[child] = A_p * Arel[j] ----
            A_abs[c]    = MatMul(A_p, Arel[j_idx], evaluate=False)

            # ---- joint global point: rJ = r_p + A_p * p2j ----
            rJ[j_idx]   = r_p + MatMul(A_p, p2j, evaluate=False)

            # ---- translation term for prismatic / cylindrical ----
            trans_term = sym.zeros(3, 1)
            if code == "P":
                u_local     = jnt.axis_u_vec
                s_val       = q[self.q_slices[j_idx].start, 0]
                trans_term  = MatMul(A_p, u_local * s_val, evaluate=False)

            elif code == "C":
                u_local     = jnt.axis_u_vec
                # Cylindrical: DOFs are [theta, s]; translational is the 2nd
                s_val       = q[self.q_slices[j_idx].start + 1, 0]
                trans_term  = MatMul(A_p, u_local * s_val, evaluate=False)
                
            elif code == "F":
                # Floating: first 3 DOFs are translational (x, y, z) in parent frame
                sl          = self.q_slices[j_idx]
                t_vec       = sym.Matrix([q[sl.start + i, 0] for i in range(3)])
                trans_term  = MatMul(A_p, t_vec, evaluate=False)

            # ---- child CG: r_abs[c] = rJ + A_c * j2c + trans_term ----
            r_abs[c] = rJ[j_idx] + MatMul(A_abs[c], j2c, evaluate=False) + trans_term

            # ---- axis / basis in global frame ----
            if code in ("R", "P", "C"):
                U[j_idx]    = MatMul(A_p, jnt.axis_u_vec, evaluate=False)

            elif code == "U":
                A_u1[j_idx] = MatrixSymbol(f"A_u1_{j_idx}", 3, 3)

                u1_g        = MatMul(A_p, jnt.axis_u1_vec, evaluate=False)
                u2_g        = MatMul(A_p, A_u1[j_idx], jnt.axis_u2_vec, evaluate=False)

                Uj          = sym.zeros(3, 2)
                Uj[:, 0]    = sym.Matrix(u1_g)
                Uj[:, 1]    = sym.Matrix(u2_g)
                U[j_idx]    = Uj

            elif code == "S":
                # Basis = parent frame columns
                U[j_idx]    = MatMul(A_p, sym.eye(3), evaluate=False)

            elif code == "F":
                # 3x6: [A_p | A_p] — first 3 cols translation basis, last 3 rotation
                U[j_idx]    = MatMul(
                                    A_p,
                                    sym.Matrix.hstack(sym.eye(3), sym.eye(3)),
                                    evaluate=False,
                )
            else:
                raise ValueError(f"Unsupported joint code {code!r} in cache builder.")

        return KinematicsCache3D(
            A_abs=A_abs,
            A_u1=A_u1,
            r_abs=r_abs,
            rJ=rJ,
            U=U,
            Arel=Arel,
            parent_of_body=parent_of_body,
            joint_of_body=joint_of_body,
        )

################### Section for pytest ###############################
# ------------------------- minimal test/demo -------------------------

def _build_tiny_3_body_chain_system() -> JointSystem3D:
    """
    Build a 3-body chain: 0->1->2->3 (3 joints).

    Types: R, P, S (axes provided where required).
    Geometry vectors are zeros because this test is topology-only.
    """
    z3 = [0.0, 0.0, 0.0]
    data = {
        "NBodies": 3,
        "joints": [(0, 1), (1, 2), (2, 3)],
        "types": ["R", "P", "S"],
        "parent_cg_to_joint": [z3, z3, z3],
        "joint_to_child_cg": [z3, z3, z3],
        "axis_u": [[0.0, 0.0, 1.0], [0.0, 0.0, 1.0], None],  # S does not require axis_u
        "axis_u1": [None, None, None],
        "axis_u2": [None, None, None],
    }
    return JointSystem3D.from_data(data)


def demo_write_pairs_chain() -> List[WritePair]:
    """
    Demonstrate yielded (k,j) pairs for a 3-body chain.

    Expected order:
      (1,0), (2,0), (3,0), (2,1), (3,1), (3,2)
    """
    sys = _build_tiny_3_body_chain_system()
    vt = VelocityTransformation3D(sys)
    return list(vt.iter_write_pairs_root_to_leaf())


def _demo_quaternion_B():
    """Demo: compile B for the 3-body chain using quaternion q_int.

    Joint layout: R(1 cfg) + P(1 cfg) + S(4 cfg) → total_cfg_dof = 6
    Speed DOF:    R(1) + P(1) + S(3) → total_dof = 5
    B shape:      (6*3, 5) = (18, 5)
    """
    js  = _build_tiny_3_body_chain_system()
    vt  = VelocityTransformation3D(js)

    q_int_syms = sym.Matrix(
        [sym.Symbol(f"qi{i}", real=True) for i in range(vt.total_cfg_dof)]
    )
    B_func = vt.compile_B_lambdified(q_int_syms)

    # Build numeric q_int with identity quaternion for S joint
    q_int_np = np.zeros(vt.total_cfg_dof)
    sl_s = vt.q_slices[2]          # cfg slice for the S joint (index 2)
    q_int_np[sl_s.start] = 1.0     # e0 = 1 → identity rotation

    B = B_func(q_int_np)
    expected_shape = (6 * vt.NBodies, vt.total_dof)
    assert B.shape == expected_shape, f"Expected {expected_shape}, got {B.shape}"
    print(f"B shape: {B.shape}  \u2713")
    return B