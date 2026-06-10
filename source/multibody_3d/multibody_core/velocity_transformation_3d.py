# source/multibody_3d/multibody_core/velocity_transformation_3d.py
"""
velocity_transformation_3d.py

Topology + indexing utilities for the 3D velocity transformation (B) assembly
and the symbolic/numeric rate-kinematics layers that support B and Bdot assembly.

Architecture (four kinematics layers)
-------------------------------------
**Layer 1 – Raw Kinematics** (position-level cache)
    ``build_cache_symbolic`` → ``KinematicsCache3D``
    Computes absolute rotations, CG positions, joint positions, and joint
    axes/bases for every body and joint in the tree.

**Layer 2 – Block Kinematics**
    ``_get_block_kinematics`` → ``BlockKinematics3D``
    Extracts per-block quantities ``d_kj = r_abs[k] - rJ[j]`` and ``U_j``.

**Layer 3 – Rate Kinematics** (first-order velocities)
    ``build_rate_cache_symbolic`` → ``KinematicsRateCache3D``
    Computes absolute angular velocities, CG velocities, joint-point
    velocities, and time-derivatives of joint axes.

**Layer 4 – Block-Rate Kinematics**
    ``_get_block_rate_kinematics`` → ``BlockRateKinematics3D``
    Extracts per-block rate quantities ``d_dot_kj`` and ``U_dot_j``.

Downstream consumers:
    ``_block_B`` / ``_block_Bdot`` — per-block formulas
    ``build_B_blocks_symbolic`` / ``build_Bdot_blocks_symbolic`` — block dicts
    ``assemble_B_symbolic`` / ``assemble_Bdot_symbolic`` — full matrix assembly
    ``compile_B_lambdified`` / ``compile_Bdot_lambdified`` — sympy → numpy (validation / symbolic export)
    ``build_B_evaluator_jax`` / ``build_Bdot_evaluator_jax`` — **preferred runtime path** (JIT-compiled JAX evaluators)

Conventions
-----------
- Ground body id = 0
- Bodies are 1..NBodies
- Joints are stored in JointSystem3D.joints and are assumed sorted by child id
  (JointSystem3D already enforces this).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterator, List, Optional

import numpy as np
import sympy as sym
from sympy import Identity, MatMul, MatrixSymbol

try:
    # package-style imports
    from .joint_system_3d import JointSystem3D, JointType
    from .topology_3d import build_adjacency, compute_root_to_leaf_joint_paths, validate_tree
    from ._velocity_transformation_helper import (
        skew, _axis_angle_rotation, _A_from_quaternion_sym, _type_code,
        _block_B_sym, _block_Bdot_sym,
        _get_block_kinematics, _get_block_rate_kinematics,
        BodyId, JointIndex, WritePair,
        BlockKinematics3D, BlockRateKinematics3D,
        SymbolicBBlock, SymbolicBdotBlock,
        NumericModelParams,
        GeometryExtractor, build_geometry_extractor,
        _skew_jax, _axis_angle_rotation_jax, _quaternion_to_rotation_jax,
        _assemble_B_recursive_jax, _assemble_Bdot_recursive_jax,
    )
    from ._velocity_transformation_inspector import BlockInspector
except Exception:  # pragma: no cover
    # script-style fallback (matches some existing files in the repo)
    from .joint_system_3d import JointSystem3D, JointType
    from .topology_3d import build_adjacency, compute_root_to_leaf_joint_paths, validate_tree
    from ._velocity_transformation_helper import (
        skew, _axis_angle_rotation, _A_from_quaternion_sym, _type_code,
        _block_B_sym, _block_Bdot_sym,
        _get_block_kinematics, _get_block_rate_kinematics,
        BodyId, JointIndex, WritePair,
        BlockKinematics3D, BlockRateKinematics3D,
        SymbolicBBlock, SymbolicBdotBlock,
        NumericModelParams,
        GeometryExtractor, build_geometry_extractor,
        _skew_jax, _axis_angle_rotation_jax, _quaternion_to_rotation_jax,
        _assemble_B_recursive_jax, _assemble_Bdot_recursive_jax,
    )
    from ._velocity_transformation_inspector import BlockInspector

try:
    import jax as _jax_mod
    _jax_mod.config.update("jax_enable_x64", True)
    import jax
    import jax.numpy as jnp
except ImportError:
    pass  # JAX functions will raise NameError if called without JAX


# ==================== Symbolic Cache Dataclasses ============================
# KinematicsCache3D (Layer 1) and KinematicsRateCache3D (Layer 3) live here
# because they are direct output types of the symbolic cache builders and
# tightly coupled to VelocityTransformation3D's orchestration logic.
# BlockKinematics3D / BlockRateKinematics3D (layers 2 & 4) and the block
# output containers live in _velocity_transformation_helper.py.

# -- Layer 1 --

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


# -- Layer 3 --

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


# (BlockKinematics3D / BlockRateKinematics3D / SymbolicBBlock /
#  SymbolicBdotBlock / NumericModelParams are imported from
#  _velocity_transformation_helper; BlockInspector from
#  _velocity_transformation_inspector.)


# =============================================================================


# =============================================================================

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

    Method organisation
    -------------------
    **Construction / topology**
        ``__init__``, ``reset_Btrack``, ``iter_write_pairs_root_to_leaf``

    **Symbolic cache builders** (Layers 1–4)
        ``build_cache_symbolic``, ``_get_block_kinematics``,
        ``build_rate_cache_symbolic``, ``_get_block_rate_kinematics``

    **Symbolic block assembly**
        ``_block_B``, ``_block_Bdot``,
        ``build_B_blocks_symbolic``, ``build_Bdot_blocks_symbolic``,
        ``assemble_B_from_blocks``, ``assemble_Bdot_from_blocks``,
        ``print_B_blocks``, ``print_Bdot_blocks``

    **Symbolic full assembly**
        ``assemble_B_symbolic``, ``assemble_Bdot_symbolic``

    **JAX runtime wrappers** (preferred runtime)
        ``build_numeric_params``,
        ``build_B_evaluator_jax``, ``build_Bdot_evaluator_jax``
        (all accept ``mainNumVars_int``)

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

    # ==================== Symbolic Cache Builders ==============================
    # Layers 1–4: position-level cache, block extraction, rate cache,
    # and block-rate extraction.  All outputs are SymPy expressions.

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
        * Relative rotations are *opaque* (not expanded): ``Arel[j] = MatrixSymbol(...)``.
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
                rJ[j_idx]   = MatMul(A_p, t_vec, evaluate=False)

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

    # ---- rate kinematics ----

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
                Udot[j_idx] = skew(omega_p) * U_j_s

            elif code == "F":
                # U_j = [A_p | A_p]  (3×6)
                U_j_f   = sym.Matrix(cache.U[j_idx])           # 3×6
                # translational DOFs are the first 3 speed entries
                qd_t    = sym.Matrix([qd[col_sl.start + i, 0] for i in range(3)])
                # rotational DOFs are the last 3 speed entries
                qd_r    = sym.Matrix([qd[col_sl.start + 3 + i, 0] for i in range(3)])
                A_p     = sym.Matrix(cache.A_abs[p])
                omega_c = omega_p + A_p * qd_r
                vJ[j_idx]   = A_p * qd_t # Overwrite vJ for F-joint to make the joint be coincident with the child CG
                v_c         = vJ[j_idx] # + skew(omega_c) * sym.Matrix(cache.r_abs[c] - cache.rJ[j_idx]) NOTE: commented out, but to be checked.
                Udot[j_idx] = skew(omega_p) * U_j_f

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

    # ==================== Symbolic Block Assembly ==============================
    # Per-block formulas (now in helper as _block_B_sym / _block_Bdot_sym),
    # block-dict builders, scatter assemblers, and block-printing helpers.

    # ---- block dicts ----

    def build_B_blocks_symbolic(
        self,
        q: sym.Matrix,
        cache: Optional[KinematicsCache3D] = None,
    ) -> dict[WritePair, SymbolicBBlock]:
        """Build all symbolic B blocks indexed by ``(body_id, joint_index)``.

        Each block retains the compact kinematic ingredients (``d_kj``, ``U_j``)
        alongside the 6×m block matrix, enabling symbolic inspection without
        having to assemble the full system matrix.

        Parameters
        ----------
        q : sympy.Matrix (total_cfg_dof, 1)
            Internal configuration vector.
        cache : KinematicsCache3D, optional
            Pre-built kinematics cache.

        Returns
        -------
        dict[(int, int), SymbolicBBlock]
            Mapping from ``(k, j)`` write-pair to the corresponding block.
        """
        if cache is None:
            cache = self.build_cache_symbolic(q)

        joints = self.joint_system.joints
        blocks: dict[WritePair, SymbolicBBlock] = {}

        self.reset_Btrack()
        for k, j in self.iter_write_pairs_root_to_leaf():
            code = _type_code(joints[j].type)
            bk   = _get_block_kinematics(cache, k, j)
            mat  = _block_B_sym(code, bk.d_kj, bk.U_j)
            r0   = 6 * (k - 1)
            blocks[(k, j)] = SymbolicBBlock(
                body_id=k,
                joint_index=j,
                joint_type=code,
                row_slice=slice(r0, r0 + 6),
                col_slice=self.col_slices[j],
                d_kj=bk.d_kj,
                U_j=bk.U_j,
                matrix=mat,
            )
        return blocks

    def build_Bdot_blocks_symbolic(
        self,
        q: sym.Matrix,
        qd: sym.Matrix,
        cache: Optional[KinematicsCache3D] = None,
        rate_cache: Optional[KinematicsRateCache3D] = None,
    ) -> dict[WritePair, SymbolicBdotBlock]:
        """Build all symbolic Bdot blocks indexed by ``(body_id, joint_index)``.

        Each block retains compact kinematic ingredients alongside the 6×m
        Bdot block matrix.

        Parameters
        ----------
        q : sympy.Matrix (total_cfg_dof, 1)
            Internal configuration vector.
        qd : sympy.Matrix (total_dof, 1)
            Generalized speed vector.
        cache : KinematicsCache3D, optional
            Pre-built kinematics cache.
        rate_cache : KinematicsRateCache3D, optional
            Pre-built rate cache.

        Returns
        -------
        dict[(int, int), SymbolicBdotBlock]
            Mapping from ``(k, j)`` write-pair to the corresponding Bdot block.
        """
        if cache is None:
            cache = self.build_cache_symbolic(q)
        if rate_cache is None:
            rate_cache = self.build_rate_cache_symbolic(q, qd, cache=cache)

        joints = self.joint_system.joints
        blocks: dict[WritePair, SymbolicBdotBlock] = {}

        self.reset_Btrack()
        for k, j in self.iter_write_pairs_root_to_leaf():
            code = _type_code(joints[j].type)
            bk   = _get_block_kinematics(cache, k, j)
            brk  = _get_block_rate_kinematics(cache, rate_cache, k, j)
            mat  = _block_Bdot_sym(code, bk.d_kj, brk.d_dot_kj, bk.U_j, brk.U_dot_j)
            r0   = 6 * (k - 1)
            blocks[(k, j)] = SymbolicBdotBlock(
                body_id=k,
                joint_index=j,
                joint_type=code,
                row_slice=slice(r0, r0 + 6),
                col_slice=self.col_slices[j],
                d_kj=bk.d_kj,
                U_j=bk.U_j,
                d_dot_kj=brk.d_dot_kj,
                U_dot_j=brk.U_dot_j,
                matrix=mat,
            )
        return blocks

    # ---- assembly from block dicts ------------------------------------------

    def assemble_B_from_blocks(
        self,
        blocks: dict[WritePair, SymbolicBBlock],
    ) -> sym.Matrix:
        """Scatter pre-built B blocks into the full system matrix.

        Parameters
        ----------
        blocks : dict[(int, int), SymbolicBBlock]
            As returned by :meth:`build_B_blocks_symbolic`.
        """
        B = sym.zeros(6 * self.NBodies, self.total_dof)
        for blk in blocks.values():
            B[blk.row_slice, blk.col_slice.start:blk.col_slice.stop] = blk.matrix
        return B

    def assemble_Bdot_from_blocks(
        self,
        blocks: dict[WritePair, SymbolicBdotBlock],
    ) -> sym.Matrix:
        """Scatter pre-built Bdot blocks into the full system matrix.

        Parameters
        ----------
        blocks : dict[(int, int), SymbolicBdotBlock]
            As returned by :meth:`build_Bdot_blocks_symbolic`.
        """
        Bdot = sym.zeros(6 * self.NBodies, self.total_dof)
        for blk in blocks.values():
            Bdot[blk.row_slice, blk.col_slice.start:blk.col_slice.stop] = blk.matrix
        return Bdot

    # ---- block inspection / printing ----

    @staticmethod
    def print_B_blocks(
        blocks: "dict[WritePair, SymbolicBBlock]",
        *,
        simplify: bool = False,
        show_matrix: bool = True,
    ) -> None:
        """Print a human-readable summary of all symbolic B blocks.

        Delegates to :class:`BlockInspector`.  For richer options (e.g.
        ``show_matrix=False`` or ``simplify=True``) call
        ``BlockInspector.display_B_blocks`` directly.
        """
        BlockInspector.display_B_blocks(
            blocks, simplify=simplify, show_matrix=show_matrix,
        )

    @staticmethod
    def print_Bdot_blocks(
        blocks: "dict[WritePair, SymbolicBdotBlock]",
        *,
        simplify: bool = False,
        show_matrix: bool = True,
    ) -> None:
        """Print a human-readable summary of all symbolic Bdot blocks.

        Delegates to :class:`BlockInspector`.
        """
        BlockInspector.display_Bdot_blocks(
            blocks, simplify=simplify, show_matrix=show_matrix,
        )

    # ==================== Symbolic Full Assembly ================================

    def assemble_B_symbolic(
        self,
        q: sym.Matrix,
        cache: Optional[KinematicsCache3D] = None,
    ) -> sym.Matrix:
        """Assemble the full symbolic velocity-transformation matrix B.

        Delegates to :meth:`build_B_blocks_symbolic` and
        :meth:`assemble_B_from_blocks`.

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
        blocks = self.build_B_blocks_symbolic(q, cache=cache)
        return self.assemble_B_from_blocks(blocks)

    def assemble_Bdot_symbolic(
        self,
        q: sym.Matrix,
        qd: sym.Matrix,
        cache: Optional[KinematicsCache3D] = None,
        rate_cache: Optional[KinematicsRateCache3D] = None,
    ) -> sym.Matrix:
        """Assemble the full symbolic Bdot matrix from per-block formulas.

        Delegates to :meth:`build_Bdot_blocks_symbolic` and
        :meth:`assemble_Bdot_from_blocks`.

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
        blocks = self.build_Bdot_blocks_symbolic(
            q, qd, cache=cache, rate_cache=rate_cache,
        )
        return self.assemble_Bdot_from_blocks(blocks)

    # ==================== Lambdified (validation / symbolic export) ===========
    # TODO: keep this method for now (remove before code deployment) 
    def compile_B_lambdified(self, q_syms: sym.Matrix) -> callable:
        """Compile B to a fast NumPy callable via SymPy lambdify (reference path).

        Parameters
        ----------
        q_syms : sympy.Matrix (total_cfg_dof, 1)
            Symbolic internal configuration coordinates.

        Returns
        -------
        callable
            ``B_func(q_int_np) -> numpy.ndarray``, shape ``(6*NBodies, total_dof)``.
        """
        q_syms = sym.Matrix(q_syms)
        if q_syms.shape != (self.total_cfg_dof, 1):
            raise ValueError(
                f"q_syms shape mismatch: expected ({self.total_cfg_dof}, 1), "
                f"got {q_syms.shape}."
            )
        cache  = self.build_cache_symbolic(q_syms)
        B_sym  = self.assemble_B_symbolic(q_syms, cache=cache)
        joints = self.joint_system.joints
        subs_dict: dict = {}
        for j_idx, jnt in enumerate(joints):
            code     = _type_code(jnt.type)
            sl       = self.q_slices[j_idx]
            Arel_sym = cache.Arel[j_idx]
            if code == "R":
                Arel_expl = _axis_angle_rotation(jnt.axis_u_vec, q_syms[sl.start, 0])
            elif code == "P":
                Arel_expl = sym.eye(3)
            elif code == "U":
                R1 = _axis_angle_rotation(jnt.axis_u1_vec, q_syms[sl.start, 0])
                R2 = _axis_angle_rotation(jnt.axis_u2_vec, q_syms[sl.start + 1, 0])
                A_u1_sym = cache.A_u1[j_idx]
                for r in range(3):
                    for c in range(3):
                        subs_dict[A_u1_sym[r, c]] = R1[r, c]
                Arel_expl = R1 * R2
            elif code == "C":
                Arel_expl = _axis_angle_rotation(jnt.axis_u_vec, q_syms[sl.start, 0])
            elif code == "S":
                Arel_expl = _A_from_quaternion_sym(
                    q_syms[sl.start, 0], q_syms[sl.start + 1:sl.start + 4, 0])
            elif code == "F":
                Arel_expl = _A_from_quaternion_sym(
                    q_syms[sl.start + 3, 0], q_syms[sl.start + 4:sl.start + 7, 0])
            else:
                raise ValueError(f"Unsupported joint code {code!r} in compile_B_lambdified.")
            for r in range(3):
                for c in range(3):
                    subs_dict[Arel_sym[r, c]] = Arel_expl[r, c]
        B_explicit = B_sym.subs(subs_dict)
        q_flat = [q_syms[i, 0] for i in range(self.total_cfg_dof)]
        _B_raw = sym.lambdify(q_flat, B_explicit, modules="numpy", cse=True)
        n_rows = 6 * self.NBodies
        n_cols = self.total_dof

        def B_func(q_int_np):
            q_int_np = np.asarray(q_int_np, dtype=float).ravel()
            with np.errstate(invalid="ignore", divide="ignore"):
                raw = _B_raw(*q_int_np)
            return np.asarray(raw, dtype=float).reshape(n_rows, n_cols)

        return B_func
    # TODO: keep this method for now (remove before code deployment)
    def compile_Bdot_lambdified(
        self, q_syms: sym.Matrix, qd_syms: sym.Matrix,
    ) -> callable:
        """Compile Bdot to a fast NumPy callable via SymPy lambdify (reference path).

        Parameters
        ----------
        q_syms : sympy.Matrix (total_cfg_dof, 1)
        qd_syms : sympy.Matrix (total_dof, 1)

        Returns
        -------
        callable
            ``Bdot_func(q_int_np, qd_np) -> numpy.ndarray``,
            shape ``(6*NBodies, total_dof)``.
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
        cache      = self.build_cache_symbolic(q_syms)
        rate_cache = self.build_rate_cache_symbolic(q_syms, qd_syms, cache=cache)
        Bdot_sym   = self.assemble_Bdot_symbolic(
            q_syms, qd_syms, cache=cache, rate_cache=rate_cache,
        )
        joints    = self.joint_system.joints
        subs_dict: dict = {}
        for j_idx, jnt in enumerate(joints):
            code     = _type_code(jnt.type)
            sl       = self.q_slices[j_idx]
            Arel_sym = cache.Arel[j_idx]
            if code == "R":
                Arel_expl = _axis_angle_rotation(jnt.axis_u_vec, q_syms[sl.start, 0])
            elif code == "P":
                Arel_expl = sym.eye(3)
            elif code == "U":
                R1 = _axis_angle_rotation(jnt.axis_u1_vec, q_syms[sl.start, 0])
                R2 = _axis_angle_rotation(jnt.axis_u2_vec, q_syms[sl.start + 1, 0])
                A_u1_sym = cache.A_u1[j_idx]
                for r in range(3):
                    for c in range(3):
                        subs_dict[A_u1_sym[r, c]] = R1[r, c]
                Arel_expl = R1 * R2
            elif code == "C":
                Arel_expl = _axis_angle_rotation(jnt.axis_u_vec, q_syms[sl.start, 0])
            elif code == "S":
                Arel_expl = _A_from_quaternion_sym(
                    q_syms[sl.start, 0], q_syms[sl.start + 1:sl.start + 4, 0])
            elif code == "F":
                Arel_expl = _A_from_quaternion_sym(
                    q_syms[sl.start + 3, 0], q_syms[sl.start + 4:sl.start + 7, 0])
            else:
                raise ValueError(f"Unsupported joint code {code!r} in compile_Bdot_lambdified.")
            for r in range(3):
                for c in range(3):
                    subs_dict[Arel_sym[r, c]] = Arel_expl[r, c]
        Bdot_explicit = Bdot_sym.subs(subs_dict)
        q_flat   = [q_syms[i, 0]  for i in range(self.total_cfg_dof)]
        qd_flat  = [qd_syms[i, 0] for i in range(self.total_dof)]
        _Bdot_raw = sym.lambdify(q_flat + qd_flat, Bdot_explicit, modules="numpy", cse=True)
        n_rows = 6 * self.NBodies
        n_cols = self.total_dof

        def Bdot_func(q_int_np, qd_np):
            q_int_np = np.asarray(q_int_np, dtype=float).ravel()
            qd_np    = np.asarray(qd_np, dtype=float).ravel()
            args     = np.concatenate([q_int_np, qd_np])
            with np.errstate(invalid="ignore", divide="ignore"):
                raw = _Bdot_raw(*args)
            return np.asarray(raw, dtype=float).reshape(n_rows, n_cols)

        return Bdot_func

    # ==================== JAX Runtime Wrappers ==================================
    # ``build_numeric_params`` converts joint-system geometry to NumPy once
    # and returns an immutable ``NumericModelParams`` that is consumed by
    # the JAX evaluators below and by ``compile_B_lambdified``.
    # ``build_B/Bdot_evaluator_jax`` -- JIT-compiled closures for repeated use

    def build_numeric_params(self) -> NumericModelParams:
        """Build the static runtime specification for this joint system.

        Stores topology metadata and raw geometry vectors (as ``sym.Matrix``)
        from the joint objects.  Geometry that depends on ``body_data_sym``
        symbols is preserved in symbolic form; fully-numeric geometry is also
        stored as a sympy matrix (numeric evaluation happens downstream).

        The returned object is immutable and is built once at
        ``MbdSystem3D`` construction time via ``__post_init__``.
        """
        joints = self.joint_system.joints

        return NumericModelParams(
            n_bodies=self.NBodies,
            n_joints=self.NJoints,
            total_dof=self.total_dof,
            total_cfg_dof=self.total_cfg_dof,
            parent=[j.parent for j in joints],
            child=[j.child for j in joints],
            code=[_type_code(j.type) for j in joints],
            p2j=[j.parent_cg_to_joint_vec for j in joints],
            j2c=[j.joint_to_child_cg_vec  for j in joints],
            u=[j.axis_u_vec               for j in joints],
            u1=[j.axis_u1_vec             for j in joints],
            u2=[j.axis_u2_vec             for j in joints],
            col_slices=list(self.col_slices),
            cfg_slices=list(self.q_slices),
            body_paths=self.body_paths,
            joint_paths=self.joint_paths,
        )

    def build_geometry_extractor(
        self, body_sym_list, *, params=None,
    ) -> "GeometryExtractor":
        """Build a geometry extractor for evaluating parameterized joint geometry.

        Parameters
        ----------
        body_sym_list : list[sym.Symbol]
            Ordered body-data symbols from ``MbdSystem3D.body_data_sym``.
        params : NumericModelParams, optional
            Uses ``self.build_numeric_params()`` when *None*.

        Returns
        -------
        GeometryExtractor
        """
        if params is None:
            params = self.build_numeric_params()
        return build_geometry_extractor(params, body_sym_list)

    # NOTE: constant geometry evaluator. May remove in the future
    def build_B_evaluator_jax(
        self,
        params: Optional["NumericModelParams"] = None,
    ) -> callable:
        """Return a ``jax.jit``-compiled callable ``f(mainNumVars_int) -> B``.

        At the :class:`VelocityTransformation3D` level ``mainNumVars_int``
        is ``[q_int, qd]`` (``qd`` is accepted but ignored for B).
        Topology and constant geometry are baked into the XLA computation.

        Parameters
        ----------
        params : NumericModelParams, optional

        Returns
        -------
        callable
            ``f(mainNumVars_int: jnp.ndarray) -> jnp.ndarray``
            where ``mainNumVars_int`` has length
            ``total_cfg_dof + total_dof``.
        """
        if params is None:
            params = self.build_numeric_params()
        kw   = _convert_geometry_to_jax(params)
        n_qi = self.total_cfg_dof

        @jax.jit
        def _evaluate(mainNumVars_int):
            v     = jnp.asarray(mainNumVars_int, dtype=jnp.float64)
            q_int = _slice_q_int(v, 0, n_qi)
            return _evaluate_B_jax(q_int, **kw)

        return _evaluate

    # NOTE: constant geometry evaluator. May remove in the future
    def build_Bdot_evaluator_jax(
        self,
        params: Optional["NumericModelParams"] = None,
    ) -> callable:
        """Return a ``jax.jit``-compiled callable ``f(mainNumVars_int) -> Bdot``.

        At the :class:`VelocityTransformation3D` level ``mainNumVars_int``
        is ``[q_int, qd]``.

        Parameters
        ----------
        params : NumericModelParams, optional

        Returns
        -------
        callable
            ``f(mainNumVars_int: jnp.ndarray) -> jnp.ndarray``
            where ``mainNumVars_int`` has length
            ``total_cfg_dof + total_dof``.
        """
        if params is None:
            params = self.build_numeric_params()
        kw   = _convert_geometry_to_jax(params)
        n_qi = self.total_cfg_dof
        n_qd = self.total_dof

        @jax.jit
        def _evaluate(mainNumVars_int):
            v     = jnp.asarray(mainNumVars_int, dtype=jnp.float64)
            q_int = _slice_q_int(v, 0, n_qi)
            qd    = _slice_qd(v, n_qi, n_qd)
            return _evaluate_Bdot_jax(q_int, qd, **kw)

        return _evaluate


# ===========================================================================
# JAX Cache Builders and High-Level Evaluators
# ===========================================================================
# Moved from _velocity_transformation_helper.py so that they live alongside
# the class that wraps them.  The underlying block-formula and assembly
# helpers (_skew_jax, _block_B_jax, _assemble_B_recursive_jax, etc.) remain
# in _velocity_transformation_helper.py and are imported above.

def build_cache_jax(
    q_int,
    *,
    n_bodies,
    n_joints,
    parent,
    child,
    codes,
    cfg_slices,
    p2j,
    j2c,
    u,
    u1,
    u2,
):
    """Build position-level kinematics cache (pure JAX).

    Returns ``(A_abs, r_abs, rJ, U, R1_cache)`` where ``R1_cache[j]`` stores
    the first-axis rotation for universal joints (needed by rate cache) and
    *None* for other types.
    """
    q = q_int.ravel()
    NB, NJ = n_bodies, n_joints
    I3 = jnp.eye(3)
    z3 = jnp.zeros((3, 1))

    A_abs = [None] * (NB + 1)
    r_abs = [None] * (NB + 1)
    A_abs[0] = I3
    r_abs[0] = z3

    rJ = [None] * NJ
    U  = [None] * NJ
    R1_cache = [None] * NJ  # stored for universal joints

    for j in range(NJ):
        p_id   = parent[j]
        c_id   = child[j]
        code   = codes[j]
        A_p    = A_abs[p_id]
        r_p    = r_abs[p_id]
        sl     = cfg_slices[j]

        # ---- relative rotation ----
        if code == "R":
            Arel_j = _axis_angle_rotation_jax(u[j], q[sl.start])
        elif code == "P":
            Arel_j = I3
        elif code == "U":
            R1 = _axis_angle_rotation_jax(u1[j], q[sl.start])
            R2 = _axis_angle_rotation_jax(u2[j], q[sl.start + 1])
            Arel_j = R1 @ R2
            R1_cache[j] = R1
        elif code == "C":
            Arel_j = _axis_angle_rotation_jax(u[j], q[sl.start])
        elif code == "S":
            Arel_j = _quaternion_to_rotation_jax(q[sl.start], q[sl.start + 1:sl.start + 4])
        elif code == "F":
            Arel_j = _quaternion_to_rotation_jax(q[sl.start + 3], q[sl.start + 4:sl.start + 7])
        else:
            raise ValueError(f"Unsupported joint code {code!r}.")

        # ---- absolute rotation ----
        A_abs[c_id] = A_p @ Arel_j

        # ---- joint position ----
        rJ[j] = r_p + A_p @ p2j[j]

        # ---- prismatic / cylindrical / floating translation ----
        if code == "P":
            trans = A_p @ (u[j] * q[sl.start])
        elif code == "C":
            trans = A_p @ (u[j] * q[sl.start + 1])
        elif code == "F":
            rJ[j] = A_p @ q[sl.start:sl.start + 3].reshape(3, 1)
            trans = z3  # already incorporated into rJ for F-joint
        else:
            trans = z3

        # ---- child CG position ----
        r_abs[c_id] = rJ[j] + A_abs[c_id] @ j2c[j] + trans

        # ---- global axes / basis ----
        if code in ("R", "P", "C"):
            U[j] = A_p @ u[j]                                         # (3,1)
        elif code == "U":
            U[j] = jnp.hstack([A_p @ u1[j], A_p @ R1 @ u2[j]])       # (3,2)
        elif code == "S":
            U[j] = A_p                                                 # (3,3)
        elif code == "F":
            U[j] = jnp.hstack([A_p, A_p])                             # (3,6)

    return A_abs, r_abs, rJ, U, R1_cache


def build_rate_cache_jax(
    q_int,
    qd,
    *,
    A_abs,
    r_abs,
    rJ,
    U,
    n_bodies,
    n_joints,
    parent,
    child,
    codes,
    col_slices,
):
    """Build first-order rate kinematics cache (pure JAX).

    Returns ``(omega_abs, v_abs, vJ, Udot)``.
    """
    qd_arr = qd.ravel()
    NB, NJ = n_bodies, n_joints
    z3 = jnp.zeros((3, 1))

    omega_abs = [None] * (NB + 1)
    v_abs     = [None] * (NB + 1)
    omega_abs[0] = z3
    v_abs[0]     = z3

    vJ   = [None] * NJ
    Udot = [None] * NJ

    for j in range(NJ):
        p_id    = parent[j]
        c_id    = child[j]
        code    = codes[j]
        col     = col_slices[j]
        omega_p = omega_abs[p_id]
        v_p     = v_abs[p_id]

        # ---- joint-point velocity ----
        vJ[j] = v_p + _skew_jax(omega_p) @ (rJ[j] - r_abs[p_id])

        if code == "R":
            U_j    = U[j]
            qd_j   = qd_arr[col.start]
            omega_c = omega_p + U_j * qd_j
            v_c     = vJ[j] + _skew_jax(omega_c) @ (r_abs[c_id] - rJ[j])
            Udot[j] = _skew_jax(omega_p) @ U_j

        elif code == "P":
            U_j    = U[j]
            qd_j   = qd_arr[col.start]
            omega_c = omega_p
            v_c     = vJ[j] + U_j * qd_j + _skew_jax(omega_c) @ (r_abs[c_id] - rJ[j])
            Udot[j] = _skew_jax(omega_p) @ U_j

        elif code == "U":
            U2     = U[j]
            u1_g   = U2[:, 0:1]
            u2_g   = U2[:, 1:2]
            qd1    = qd_arr[col.start]
            qd2    = qd_arr[col.start + 1]
            u1_dot = _skew_jax(omega_p) @ u1_g
            omega_after_u1 = omega_p + qd1 * u1_g
            u2_dot = _skew_jax(omega_after_u1) @ u2_g
            Udot[j] = jnp.hstack([u1_dot, u2_dot])
            omega_c = omega_p + qd1 * u1_g + qd2 * u2_g
            v_c     = vJ[j] + _skew_jax(omega_c) @ (r_abs[c_id] - rJ[j])

        elif code == "C":
            U_j     = U[j]
            qd_rot  = qd_arr[col.start]
            qd_tr   = qd_arr[col.start + 1]
            omega_c = omega_p + U_j * qd_rot
            v_c     = vJ[j] + U_j * qd_tr + _skew_jax(omega_c) @ (r_abs[c_id] - rJ[j])
            Udot[j] = _skew_jax(omega_p) @ U_j

        elif code == "S":
            U_j_s  = U[j]
            qd_s   = qd_arr[col.start:col.start + 3].reshape(3, 1)
            omega_c = omega_p + U_j_s @ qd_s
            v_c     = vJ[j] + _skew_jax(omega_c) @ (r_abs[c_id] - rJ[j])
            Udot[j] = _skew_jax(omega_p) @ U_j_s

        elif code == "F":
            A_p_j  = A_abs[p_id]
            qd_t   = qd_arr[col.start:col.start + 3].reshape(3, 1)
            qd_r   = qd_arr[col.start + 3:col.start + 6].reshape(3, 1)
            omega_c = omega_p + A_p_j @ qd_r
            vJ[j]   = A_p_j @ qd_t  # Overwrite vJ for F-joint: joint coincident with child CG
            v_c     = vJ[j]
            Udot[j] = _skew_jax(omega_p) @ U[j]

        else:
            raise ValueError(f"Unsupported joint code {code!r}.")

        omega_abs[c_id] = omega_c
        v_abs[c_id]     = v_c

    return omega_abs, v_abs, vJ, Udot


def _evaluate_B_jax(
    q_int,
    *,
    n_bodies,
    n_joints,
    total_dof,
    parent,
    child,
    codes,
    cfg_slices,
    col_slices,
    p2j,
    j2c,
    u,
    u1,
    u2,
    body_paths,
    joint_paths,
):
    """Private JAX kernel: evaluate B given ``q_int`` and baked geometry.

    This is the innermost traceable function; all JIT closures call this
    directly.  Prefer :meth:`VelocityTransformation3D.build_B_evaluator_jax`
    when calling from Python code that has a full ``mainNumVars_int`` vector.
    """
    A_abs, r_abs, rJ, U, _ = build_cache_jax(
        q_int,
        n_bodies=n_bodies, n_joints=n_joints,
        parent=parent, child=child, codes=codes,
        cfg_slices=cfg_slices, p2j=p2j, j2c=j2c,
        u=u, u1=u1, u2=u2,
    )
    return _assemble_B_recursive_jax(
        r_abs, rJ, U,
        n_bodies=n_bodies, total_dof=total_dof,
        codes=codes, body_paths=body_paths,
        joint_paths=joint_paths, col_slices=col_slices,
    )


def _evaluate_Bdot_jax(
    q_int,
    qd,
    *,
    n_bodies,
    n_joints,
    total_dof,
    parent,
    child,
    codes,
    cfg_slices,
    col_slices,
    p2j,
    j2c,
    u,
    u1,
    u2,
    body_paths,
    joint_paths,
):
    """Private JAX kernel: evaluate Bdot given ``q_int``, ``qd`` and baked geometry.

    This is the innermost traceable function; all JIT closures call this
    directly.  Prefer :meth:`VelocityTransformation3D.build_Bdot_evaluator_jax`
    when calling from Python code that has a full ``mainNumVars_int`` vector.
    """
    A_abs, r_abs, rJ, U, _ = build_cache_jax(
        q_int,
        n_bodies=n_bodies, n_joints=n_joints,
        parent=parent, child=child, codes=codes,
        cfg_slices=cfg_slices, p2j=p2j, j2c=j2c,
        u=u, u1=u1, u2=u2,
    )
    omega_abs, v_abs, vJ, Udot = build_rate_cache_jax(
        q_int, qd,
        A_abs=A_abs, r_abs=r_abs, rJ=rJ, U=U,
        n_bodies=n_bodies, n_joints=n_joints,
        parent=parent, child=child, codes=codes,
        col_slices=col_slices,
    )
    return _assemble_Bdot_recursive_jax(
        r_abs, rJ, U, v_abs, vJ, Udot,
        n_bodies=n_bodies, total_dof=total_dof,
        codes=codes, body_paths=body_paths,
        joint_paths=joint_paths, col_slices=col_slices,
    )


def _sym_to_jax(v, fallback):
    """Convert a raw geometry entry (``sym.Matrix`` or ``None``) to a JAX array.

    For constant (fully-numeric) sympy matrices the conversion goes through
    numpy.  For ``None`` the provided *fallback* JAX array is returned.
    Parameterized entries (containing free symbols) will raise a
    ``TypeError`` here; use :class:`GeometryExtractor` for those.
    """
    if v is None:
        return fallback
    if isinstance(v, sym.Matrix):
        return jnp.asarray(np.array(v.tolist(), dtype=float))
    # Already a numpy / JAX array (legacy or pre-converted).
    return jnp.asarray(v)


def _convert_topology_to_jax(params):
    """Extract static topology kwargs from *params* (no geometry arrays).

    Returns a dict suitable for unpacking into ``_evaluate_B_jax`` alongside
    separately-provided geometry arrays.
    """
    return dict(
        n_bodies=params.n_bodies,
        n_joints=params.n_joints,
        total_dof=params.total_dof,
        parent=params.parent,
        child=params.child,
        codes=params.code,
        cfg_slices=params.cfg_slices,
        col_slices=params.col_slices,
        body_paths=params.body_paths,
        joint_paths=params.joint_paths,
    )


def _np_geom_to_jax(p2j, j2c, u, u1, u2):
    """Convert extractor output (lists of ``ndarray | None``) to JAX arrays."""
    z31 = jnp.zeros((3, 1))
    _j = lambda v: jnp.asarray(v) if v is not None else z31
    return (
        [_j(v) for v in p2j],
        [_j(v) for v in j2c],
        [_j(v) for v in u],
        [_j(v) for v in u1],
        [_j(v) for v in u2],
    )


def _convert_geometry_to_jax(params):
    """Convert ``NumericModelParams`` geometry vectors to JAX arrays.

    Geometry entries are raw ``sym.Matrix | None`` values.  Constant matrices
    are converted via numpy; ``None`` entries are replaced with a zero (3,1)
    column.  Parameterized entries (containing free symbols) will raise;
    use :class:`GeometryExtractor` for those.

    Returns a dict of keyword arguments suitable for unpacking into
    ``_evaluate_B_jax`` / ``_evaluate_Bdot_jax``.
    """
    z31 = jnp.zeros((3, 1))
    kw = _convert_topology_to_jax(params)
    kw.update(
        p2j=[_sym_to_jax(v, z31) for v in params.p2j],
        j2c=[_sym_to_jax(v, z31) for v in params.j2c],
        u=[_sym_to_jax(v, z31)   for v in params.u],
        u1=[_sym_to_jax(v, z31)  for v in params.u1],
        u2=[_sym_to_jax(v, z31)  for v in params.u2],
    )
    return kw


# ---------------------------------------------------------------------------
# mainNumVars_int slice helpers (JAX-traceable)
# ---------------------------------------------------------------------------

def _slice_q_int(v: "jnp.ndarray", qi_start: int, n_qi: int) -> "jnp.ndarray":
    """Extract the ``q_int`` segment from a ``mainNumVars_int`` vector.

    Parameters
    ----------
    v : jnp.ndarray
        Flat ``mainNumVars_int`` array.
    qi_start : int
        Index of the first ``q_int`` element (always 0 in canonical layout).
    n_qi : int
        Length of ``q_int`` (``total_cfg_dof``).
    """
    return v[qi_start : qi_start + n_qi]


def _slice_qd(v: "jnp.ndarray", qd_start: int, n_qd: int) -> "jnp.ndarray":
    """Extract the ``qd`` segment from a ``mainNumVars_int`` vector.

    Parameters
    ----------
    v : jnp.ndarray
        Flat ``mainNumVars_int`` array.
    qd_start : int
        Index of the first ``qd`` element (``total_cfg_dof`` in canonical layout).
    n_qd : int
        Length of ``qd`` (``total_dof``).
    """
    return v[qd_start : qd_start + n_qd]


# ---------------------------------------------------------------------------
# make_B/Bdot_evaluator_mainint — public factory functions
# ---------------------------------------------------------------------------


def make_B_evaluator_mainint(params, slc_q_int, slc_qd_int, extractor, slc_body_int):
    """Return a callable ``f(mainNumVars_int) -> B``.

    When geometry is fully constant the evaluator is a single JIT closure
    with geometry baked in.  When geometry depends on body-data parameters
    the body slice is extracted at runtime (numpy), evaluated via
    *extractor*, converted to JAX arrays, and passed to a JIT-compiled
    assembly kernel.

    Parameters
    ----------
    params : NumericModelParams
    slc_q_int : slice
    slc_qd_int : slice
    extractor : GeometryExtractor
    slc_body_int : slice
    """
    qi_start = slc_q_int.start
    n_qi     = slc_q_int.stop - slc_q_int.start

    if not extractor.has_dynamic:
        # ---- constant geometry: bake into JIT closure ----
        kw = _convert_geometry_to_jax(params)

        @jax.jit
        def _evaluate(mainNumVars_int):
            v     = jnp.asarray(mainNumVars_int, dtype=jnp.float64)
            q_int = _slice_q_int(v, qi_start, n_qi)
            return _evaluate_B_jax(q_int, **kw)

        return _evaluate

    # ---- parameterized geometry: evaluate at runtime ----
    topo_kw  = _convert_topology_to_jax(params)
    bp_start = slc_body_int.start
    n_bp     = slc_body_int.stop - slc_body_int.start

    @jax.jit
    def _jit_B(q_int, p2j, j2c, u, u1, u2):
        return _evaluate_B_jax(
            q_int, p2j=p2j, j2c=j2c, u=u, u1=u1, u2=u2, **topo_kw,
        )

    def _evaluate(mainNumVars_int):
        v     = np.asarray(mainNumVars_int, dtype=float)
        q_int = v[qi_start : qi_start + n_qi]
        bp    = v[bp_start : bp_start + n_bp]
        geom  = extractor.evaluate(bp)
        return _jit_B(
            jnp.asarray(q_int, dtype=jnp.float64),
            *_np_geom_to_jax(*geom),
        )

    return _evaluate


def make_Bdot_evaluator_mainint(params, slc_q_int, slc_qd_int, extractor, slc_body_int):
    """Return a callable ``f(mainNumVars_int) -> Bdot``.

    Same constant/parameterized branching as
    :func:`make_B_evaluator_mainint`.

    Parameters
    ----------
    params : NumericModelParams
    slc_q_int  : slice
    slc_qd_int : slice
    extractor  : GeometryExtractor
    slc_body_int : slice
    """
    qi_start = slc_q_int.start
    n_qi     = slc_q_int.stop - slc_q_int.start
    qd_start = slc_qd_int.start
    n_qd     = slc_qd_int.stop - slc_qd_int.start

    if not extractor.has_dynamic:
        kw = _convert_geometry_to_jax(params)

        @jax.jit
        def _evaluate(mainNumVars_int):
            v     = jnp.asarray(mainNumVars_int, dtype=jnp.float64)
            q_int = _slice_q_int(v, qi_start, n_qi)
            qd    = _slice_qd(v, qd_start, n_qd)
            return _evaluate_Bdot_jax(q_int, qd, **kw)

        return _evaluate

    topo_kw  = _convert_topology_to_jax(params)
    bp_start = slc_body_int.start
    n_bp     = slc_body_int.stop - slc_body_int.start

    @jax.jit
    def _jit_Bdot(q_int, qd, p2j, j2c, u, u1, u2):
        return _evaluate_Bdot_jax(
            q_int, qd, p2j=p2j, j2c=j2c, u=u, u1=u1, u2=u2, **topo_kw,
        )

    def _evaluate(mainNumVars_int):
        v     = np.asarray(mainNumVars_int, dtype=float)
        q_int = v[qi_start : qi_start + n_qi]
        qd    = v[qd_start : qd_start + n_qd]
        bp    = v[bp_start : bp_start + n_bp]
        geom  = extractor.evaluate(bp)
        return _jit_Bdot(
            jnp.asarray(q_int, dtype=jnp.float64),
            jnp.asarray(qd, dtype=jnp.float64),
            *_np_geom_to_jax(*geom),
        )

    return _evaluate


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
    """Demo: evaluate B for the 3-body chain via the JAX backend.

    Joint layout: R(1 cfg) + P(1 cfg) + S(4 cfg) → total_cfg_dof = 6
    Speed DOF:    R(1) + P(1) + S(3) → total_dof = 5
    B shape:      (6*3, 5) = (18, 5)
    """
    js  = _build_tiny_3_body_chain_system()
    vt  = VelocityTransformation3D(js)

    # Build numeric q_int with identity quaternion for S joint
    mainNumVars_int = np.zeros(vt.total_cfg_dof + vt.total_dof)
    sl_s = vt.q_slices[2]          # cfg slice for the S joint (index 2)
    mainNumVars_int[sl_s.start] = 1.0     # e0 = 1 → identity rotation

    B_fn = vt.build_B_evaluator_jax()
    B    = np.asarray(B_fn(mainNumVars_int))
    expected_shape = (6 * vt.NBodies, vt.total_dof)
    assert B.shape == expected_shape, f"Expected {expected_shape}, got {B.shape}"
    print(f"B shape: {B.shape}  \u2713")
    return B