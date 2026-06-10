"""_velocity_transformation_helper.py

Pure, stateless helpers shared by the symbolic and JAX runtime paths:

**Symbolic helpers** (no JAX dependency)
    ``skew``                   — 3×3 skew-symmetric matrix (SymPy)
    ``_axis_angle_rotation``   — Rodrigues rotation (SymPy)
    ``_A_from_quaternion_sym`` — unit-quaternion rotation matrix (SymPy)
    ``_type_code``             — joint-type → 1-letter code
    ``RootToLeafPaths``        — lightweight dataclass for traversal paths

**JAX runtime backend** (requires ``jax``; lazy-imported)
    Rotation / algebra helpers, block formulas (B and Bdot), and recursive
    assemblers.  The higher-level cache builders (``build_cache_jax``,
    ``build_rate_cache_jax``) and JIT factories (``make_B_evaluator_mainint``,
    ``make_Bdot_evaluator_mainint``) live in ``velocity_transformation_3d.py``.
"""
import numpy as np
import sympy as sym
from dataclasses import dataclass
from typing import Any, List, Optional, Tuple
from .joint_system_3d import JointType

@dataclass(frozen=True, slots=True)
class RootToLeafPaths:
    """Root-to-leaf paths cache."""
    body_paths: List[List[int]]   # [[b1,b2,...], ...] excluding ground
    joint_paths: List[List[int]]  # [[j(b1),j(b2),...], ...] aligned with body_paths

def skew(v: sym.Matrix) -> sym.Matrix:
    """Return the 3x3 skew-symmetric matrix of a 3x1 vector."""
    if v.shape != (3, 1):
        raise ValueError(f"skew: expected (3,1), got {v.shape}.")
    x, y, z = v[0, 0], v[1, 0], v[2, 0]
    return sym.Matrix([
        [   0, -z,  y],
        [   z,  0, -x],
        [  -y,  x,  0],
    ])

def _type_code(jt: JointType) -> str:
    """Return the 1-letter joint code ('R','P','S','U','C','F')."""
    # Prefer enum value if it is the one-letter code
    v = getattr(jt, "value", jt)
    if isinstance(v, str) and len(v) == 1:
        return v
    # Fallback: enum name if it matches
    n = getattr(jt, "name", str(v))
    if isinstance(n, str) and len(n) == 1:
        return n
    return str(v)


def _axis_angle_rotation(u: sym.Matrix, theta: sym.Expr) -> sym.Matrix:
    """Rodrigues rotation matrix: rotation by *theta* about unit axis *u* (3x1).

    .. math::

        R = I + \\sin\\theta\\, K + (1 - \\cos\\theta)\\, K^2

    where :math:`K = \\mathrm{skew}(u)`.
    """
    K = skew(u)
    return sym.eye(3) + sym.sin(theta) * K + (1 - sym.cos(theta)) * K * K


def _A_from_quaternion_sym(e0, e) -> sym.Matrix:
    """Rotation matrix from unit quaternion ``[e0,e1,e2,e3] = [w,x,y,z]``.

    Returns a 3×3 sympy Matrix (polynomial in quaternion components).
    """
    e = sym.Matrix(e)
    A = (2*e0**2 - 1) * sym.eye(3) + 2 * (e * e.T + e0 * skew(e))

    return A


# ===========================================================================
# Type Aliases and Transport Containers
# ===========================================================================
# Lightweight type aliases and frozen dataclasses shared between the main
# orchestration module, the JAX backend, and the inspector module.
# They carry no behaviour and impose no dependency on SymPy or JAX.

BodyId     = int
JointIndex = int
WritePair  = Tuple[BodyId, JointIndex]


# -- Symbolic block-extraction containers (Layers 2 & 4) ---------------------

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
        3\u00d71 vector from joint *j* to body-*k* CG, expressed in the global frame.
    U_j : sym.Matrix
        3\u00d7m joint axis / basis in the global frame (m depends on joint type).
    """

    body_id:     int
    joint_index: int
    d_kj:        sym.Matrix
    U_j:         sym.Matrix


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
        3\u00d71 time derivative of the position vector from joint *j* to body-*k* CG,
        in the global frame.
    U_dot_j : sym.Matrix
        3\u00d7m time derivative of the joint axis / basis.
    """

    body_id:     int
    joint_index: int
    d_dot_kj:    sym.Matrix
    U_dot_j:     sym.Matrix


# -- Symbolic block output containers ----------------------------------------

@dataclass(frozen=True, slots=True)
class SymbolicBBlock:
    """Symbolic B block for a single (body *k*, joint *j*) pair.

    Stores both the compact kinematic ingredients and the assembled 6\u00d7m block
    matrix so that downstream code can inspect either representation.

    Symbolic parameters (geometry vectors, rotation symbols, joint coordinates)
    flow through ``d_kj`` and ``U_j`` from the position-level cache; they
    remain unevaluated inside ``matrix`` until the caller decides to expand or
    substitute.
    """
    body_id:     int
    joint_index: int
    joint_type:  str           # 1-letter code: R, P, U, C, S, F
    row_slice:   slice         # rows in full B  (6 rows per body)
    col_slice:   slice         # cols in full B  (DOF columns for this joint)
    d_kj:        sym.Matrix    # 3\u00d71 position vector (global)
    U_j:         sym.Matrix    # 3\u00d7m joint axis / basis (global)
    matrix:      sym.Matrix    # 6\u00d7m assembled block


@dataclass(frozen=True, slots=True)
class SymbolicBdotBlock:
    """Symbolic Bdot block for a single (body *k*, joint *j*) pair.

    Stores position-level and rate-level kinematic ingredients together with
    the assembled 6\u00d7m Bdot block matrix.
    """
    body_id:     int
    joint_index: int
    joint_type:  str
    row_slice:   slice
    col_slice:   slice
    d_kj:        sym.Matrix    # 3\u00d71 position vector (global)
    U_j:         sym.Matrix    # 3\u00d7m joint axis / basis (global)
    d_dot_kj:    sym.Matrix    # 3\u00d71 time derivative of d_kj
    U_dot_j:     sym.Matrix    # 3\u00d7m time derivative of U_j
    matrix:      sym.Matrix    # 6\u00d7m assembled Bdot block


# -- Numeric model parameters -------------------------------------------------

@dataclass(frozen=True, slots=True)
class NumericModelParams:
    """Static runtime specification extracted from ``JointSystem3D``.

    Stores immutable topology and bookkeeping metadata.  Geometry vectors
    (``p2j``, ``j2c``, ``u``, ``u1``, ``u2``) are kept as raw
    ``sym.Matrix | None`` values — exactly as they appear on the joint
    objects — so that parameterized entries (containing ``body_data_sym``
    symbols) are preserved rather than frozen to floats.

    For purely-constant systems all geometry vectors are numeric sympy
    matrices and can be converted to JAX arrays on demand.  For
    parameterized systems the conversion step must substitute numeric
    values for the symbolic parameters first (implemented in a later
    phase).

    Built once via :meth:`VelocityTransformation3D.build_numeric_params`;
    reused across many evaluation calls without reconstruction.
    """
    n_bodies:      int
    n_joints:      int
    total_dof:     int
    total_cfg_dof: int
    parent:        List[int]                        # parent body id per joint
    child:         List[int]                        # child body id per joint
    code:          List[str]                        # 1-letter type code per joint
    p2j:           List[Any]                        # parent_cg_to_joint: sym.Matrix (3,1)
    j2c:           List[Any]                        # joint_to_child_cg:  sym.Matrix (3,1)
    u:             List[Optional[Any]]              # axis_u:  sym.Matrix (3,1) or None
    u1:            List[Optional[Any]]              # axis_u1: sym.Matrix (3,1) or None
    u2:            List[Optional[Any]]              # axis_u2: sym.Matrix (3,1) or None
    col_slices:    List[slice]                      # speed-DOF slices into B cols
    cfg_slices:    List[slice]                      # cfg-DOF slices into q_int
    body_paths:    List[List[int]]                  # root-to-leaf body paths
    joint_paths:   List[List[int]]                  # root-to-leaf joint paths


# -- Geometry extractor -------------------------------------------------------

@dataclass(frozen=True)
class GeometryExtractor:
    """Evaluates joint geometry vectors given numeric body-parameter values.

    Constant (symbol-free) entries are pre-converted to ``np.ndarray``.
    Parameterized entries (containing ``body_data_sym`` symbols) are
    lambdified once at construction and evaluated on each call.

    Attributes
    ----------
    _entries : tuple
        5-tuple ``(p2j, j2c, u, u1, u2)`` of per-joint lists.  Each
        element is ``np.ndarray(3,1)``, ``None``, or a callable
        ``f(body_params_np) -> np.ndarray(3,1)``.
    n_body_params : int
        Expected length of the body-parameter vector.
    has_dynamic : bool
        *True* if any entry depends on ``body_data_sym`` symbols.
    """

    _entries: tuple
    n_body_params: int
    has_dynamic: bool

    def evaluate(self, body_params_np=None):
        """Return ``(p2j, j2c, u, u1, u2)`` as a tuple of five lists.

        Parameters
        ----------
        body_params_np : array_like or None
            Numeric body-parameter values, shape ``(n_body_params,)``.
            Required when ``has_dynamic`` is *True*.

        Returns
        -------
        tuple of 5 lists
            Each list has one entry per joint: ``np.ndarray(3,1)`` or ``None``.
        """
        if self.has_dynamic:
            if body_params_np is None:
                raise ValueError(
                    "body_params_np is required when geometry has dynamic parameters."
                )
            bp = np.asarray(body_params_np, dtype=float).ravel()
            if bp.shape[0] != self.n_body_params:
                raise ValueError(
                    f"body_params length mismatch: expected {self.n_body_params}, "
                    f"got {bp.shape[0]}."
                )
        else:
            bp = None
        return tuple(
            [e(bp) if callable(e) else e for e in field_list]
            for field_list in self._entries
        )


def build_geometry_extractor(params, body_sym_list):
    """Build a :class:`GeometryExtractor` from the symbolic geometry in *params*.

    Parameters
    ----------
    params : NumericModelParams
        Static runtime spec whose ``p2j``, ``j2c``, ``u``, ``u1``, ``u2``
        entries are ``sym.Matrix(3,1)`` or ``None``.
    body_sym_list : list[sym.Symbol]
        Ordered body-data symbols matching the body_params slice of
        ``mainNumVars_int``.

    Returns
    -------
    GeometryExtractor
    """
    body_sym_set = frozenset(body_sym_list)
    has_dynamic = False

    def _prepare(v):
        nonlocal has_dynamic
        if v is None:
            return None
        free = v.free_symbols
        if not free:
            return np.array(v.tolist(), dtype=float).reshape(3, 1)
        unknown = free - body_sym_set
        if unknown:
            raise ValueError(
                f"Geometry vector contains symbols not in body_data_sym: {unknown}"
            )
        has_dynamic = True
        # TODO: this is very suspicious - why are we lambdifying here?
        fn = sym.lambdify(body_sym_list, v, modules='numpy')

        def _eval(bp, _fn=fn):
            return np.array(_fn(*bp), dtype=float).reshape(3, 1)

        return _eval

    entries = tuple(
        [_prepare(v) for v in getattr(params, name)]
        for name in ('p2j', 'j2c', 'u', 'u1', 'u2')
    )

    return GeometryExtractor(
        _entries=entries,
        n_body_params=len(body_sym_list),
        has_dynamic=has_dynamic,
    )


# ===========================================================================
# Symbolic Assembly Helpers
# ===========================================================================
# Pure functions moved from VelocityTransformation3D.  These have no
# dependency on ``self`` and operate only on symbolic quantities.

def _get_block_kinematics(cache, k: int, j: int) -> "BlockKinematics3D":
    """Extract position-level kinematic quantities for block (k, j).

    Parameters
    ----------
    cache : KinematicsCache3D
        Position-level cache built by ``build_cache_symbolic``.
    k : int
        Body index (1..NBodies).
    j : int
        Joint index (0..NJoints-1).

    Returns
    -------
    BlockKinematics3D
        ``d_kj = r_abs[k] - rJ[j]`` and ``U_j`` as explicit ``sym.Matrix``.
    """
    d_kj = sym.Matrix(cache.r_abs[k] - cache.rJ[j])
    U_j  = sym.Matrix(cache.U[j])
    return BlockKinematics3D(body_id=k, joint_index=j, d_kj=d_kj, U_j=U_j)


def _get_block_rate_kinematics(
    cache, rate_cache, k: int, j: int
) -> "BlockRateKinematics3D":
    """Extract rate-kinematic quantities for block (k, j).

    Parameters
    ----------
    cache : KinematicsCache3D
        Position-level cache (not used in computation; present for API symmetry
        with ``_get_block_kinematics``).
    rate_cache : KinematicsRateCache3D
        Rate cache built by ``build_rate_cache_symbolic``.
    k : int
        Body index (1..NBodies).
    j : int
        Joint index (0..NJoints-1).

    Returns
    -------
    BlockRateKinematics3D
        ``d_dot_kj = v_abs[k] - vJ[j]`` and ``U_dot_j`` as explicit ``sym.Matrix``.
    """
    d_dot_kj = sym.Matrix(rate_cache.v_abs[k] - rate_cache.vJ[j])
    U_dot_j  = sym.Matrix(rate_cache.Udot[j])
    return BlockRateKinematics3D(
        body_id=k, joint_index=j, d_dot_kj=d_dot_kj, U_dot_j=U_dot_j
    )


def _block_B_sym(
    code: str,
    d_kj: sym.Matrix,
    U_j: sym.Matrix,
) -> sym.Matrix:
    """Return the 6\u00d7m symbolic B block for joint type *code*.

    Parameters
    ----------
    code : str
        1-letter joint type code (``'R'``, ``'P'``, ``'U'``, ``'C'``,
        ``'S'``, ``'F'``).
    d_kj : sym.Matrix (3, 1)
        Position vector from joint *j* to body *k* CG in the global frame.
    U_j : sym.Matrix (3, m)
        Joint axis / basis in the global frame.

    Returns
    -------
    sym.Matrix (6, m)::

        R:  [[-skew(d)\u00b7u],  [u]]           6\u00d71
        P:  [[u],           [0]]           6\u00d71
        S:  [[-skew(d)],    [I]]           6\u00d73
        U:  [[-skew(d)\u00b7U],  [U]]           6\u00d72
        C:  [[-skew(d)\u00b7u, u], [u, 0]]      6\u00d72
        F:  [[I, -skew(d)], [0, I]]        6\u00d76
    """
    d_tilde = skew(d_kj)
    I3      = sym.eye(3)
    Z3      = sym.zeros(3)

    if code == "R":
        return sym.Matrix.vstack(-d_tilde * U_j, U_j)

    if code == "P":
        return sym.Matrix.vstack(U_j, sym.zeros(3, 1))

    if code == "S":
        return sym.Matrix.vstack(-d_tilde, I3)

    if code == "U":
        return sym.Matrix.vstack(-d_tilde * U_j, U_j)

    if code == "C":
        u   = U_j
        top = sym.Matrix.hstack(-d_tilde * u, u)
        bot = sym.Matrix.hstack(u, sym.zeros(3, 1))
        return sym.Matrix.vstack(top, bot)

    if code == "F":
        top = sym.Matrix.hstack(I3, -d_tilde)
        bot = sym.Matrix.hstack(Z3, I3)
        return sym.Matrix.vstack(top, bot)

    raise ValueError(f"Unsupported joint code {code!r} in _block_B_sym.")


def _block_Bdot_sym(
    code: str,
    d_kj: sym.Matrix,
    d_dot_kj: sym.Matrix,
    U_j: sym.Matrix,
    U_dot_j: sym.Matrix,
) -> sym.Matrix:
    """Return the 6\u00d7m symbolic Bdot block for joint type *code*.

    Parameters
    ----------
    code : str
        1-letter joint type code.
    d_kj : sym.Matrix (3, 1)
        Position vector from joint *j* to body *k* CG in the global frame.
    d_dot_kj : sym.Matrix (3, 1)
        Time derivative of *d_kj*.
    U_j : sym.Matrix (3, m)
        Joint axis / basis in the global frame.
    U_dot_j : sym.Matrix (3, m)
        Time derivative of *U_j*.

    Returns
    -------
    sym.Matrix (6, m)
    """
    if code == "R":
        top = -skew(d_dot_kj) * U_j - skew(d_kj) * U_dot_j
        return sym.Matrix.vstack(top, U_dot_j)

    if code == "P":
        return sym.Matrix.vstack(U_dot_j, sym.zeros(3, 1))

    if code == "U":
        top = -skew(d_dot_kj) * U_j - skew(d_kj) * U_dot_j
        return sym.Matrix.vstack(top, U_dot_j)

    if code == "C":
        u     = U_j
        u_dot = U_dot_j
        top   = sym.Matrix.hstack(-skew(d_dot_kj) * u - skew(d_kj) * u_dot, u_dot)
        bot   = sym.Matrix.hstack(u_dot, sym.zeros(3, 1))
        return sym.Matrix.vstack(top, bot)

    if code == "S":
        return sym.Matrix.vstack(-skew(d_dot_kj), sym.zeros(3, 3))

    if code == "F":
        Z3  = sym.zeros(3)
        top = sym.Matrix.hstack(Z3, -skew(d_dot_kj))
        bot = sym.Matrix.hstack(Z3, Z3)
        return sym.Matrix.vstack(top, bot)

    raise ValueError(f"Unsupported joint code {code!r} in _block_Bdot_sym.")


# ===========================================================================
# JAX Runtime Backend
# ===========================================================================
#
# Pure, stateless JAX implementations of all kinematic helpers, block
# formulas, cache builders, recursive assemblers, and JIT-compiled factory
# functions.  Merged from the former ``_jax_backend.py`` module.
#
# The try/except below keeps this module importable in environments without
# JAX.  Any call to a JAX function when JAX is not installed will raise
# a NameError at call time.

try:
    import jax as _jax_mod
    _jax_mod.config.update("jax_enable_x64", True)
    import jax
    import jax.numpy as jnp
except ImportError:
    pass  # JAX functions will raise NameError if called without JAX


# ==================== JAX Rotation / Algebra Helpers =========================

def _skew_jax(v):
    """3x3 skew-symmetric matrix of a (3,) or (3,1) vector."""
    v = v.ravel()
    x, y, z = v[0], v[1], v[2]
    return jnp.array([[0., -z, y],
                       [z,  0., -x],
                       [-y, x,  0.]])


def _axis_angle_rotation_jax(u, theta):
    """Rodrigues rotation: R = I + sin(theta) K + (1-cos(theta)) K^2."""
    K = _skew_jax(u)
    s = jnp.sin(theta)
    c = jnp.cos(theta)
    return jnp.eye(3) + s * K + (1. - c) * (K @ K)


def _quaternion_to_rotation_jax(e0, e):
    """Unit quaternion [w,x,y,z] -> 3x3 rotation matrix."""
    e = e.ravel()
    return (2. * e0 * e0 - 1.) * jnp.eye(3) + 2. * (jnp.outer(e, e) + e0 * _skew_jax(e))


# ==================== Block Formulas (B) =====================================

def _block_B_jax(code, d_kj, U_j):
    """Compute 6xm B block for one (body, joint) pair."""
    dt = _skew_jax(d_kj)
    if code == "R":
        return jnp.vstack([-dt @ U_j, U_j])                          # (6,1)
    if code == "P":
        return jnp.vstack([U_j, jnp.zeros((3, 1))])                  # (6,1)
    if code == "U":
        return jnp.vstack([-dt @ U_j, U_j])                          # (6,2)
    if code == "C":
        u = U_j
        return jnp.vstack([jnp.hstack([-dt @ u, u]),
                            jnp.hstack([u, jnp.zeros((3, 1))])])     # (6,2)
    if code == "S":
        return jnp.vstack([-dt, jnp.eye(3)])                          # (6,3)
    if code == "F":
        I3 = jnp.eye(3)
        Z3 = jnp.zeros((3, 3))
        return jnp.vstack([jnp.hstack([I3, -dt]),
                            jnp.hstack([Z3, I3])])                    # (6,6)
    raise ValueError(f"Unsupported joint code {code!r}.")


def _update_B_block_jax(code, prev_block, delta, U_j):
    """Incrementally update a B block for the next downstream body."""
    sd = _skew_jax(delta)
    if code in ("R", "U"):
        return prev_block.at[:3, :].set(prev_block[:3, :] - sd @ U_j)
    if code == "P":
        return prev_block  # d_kj-independent
    if code == "C":
        return prev_block.at[0:3, 0:1].set(prev_block[0:3, 0:1] - sd @ U_j)
    if code == "S":
        return prev_block.at[:3, :].set(prev_block[:3, :] - sd)
    if code == "F":
        return prev_block.at[0:3, 3:6].set(prev_block[0:3, 3:6] - sd)
    raise ValueError(f"Unsupported joint code {code!r}.")


# ==================== Block Formulas (Bdot) ==================================

def _block_Bdot_jax(code, d_kj, d_dot_kj, U_j, U_dot_j):
    """Compute 6xm Bdot block for one (body, joint) pair."""
    dd = _skew_jax(d_dot_kj)
    dt = _skew_jax(d_kj)
    if code == "R":
        return jnp.vstack([-dd @ U_j - dt @ U_dot_j, U_dot_j])       # (6,1)
    if code == "P":
        return jnp.vstack([U_dot_j, jnp.zeros((3, 1))])              # (6,1)
    if code == "U":
        return jnp.vstack([-dd @ U_j - dt @ U_dot_j, U_dot_j])       # (6,2)
    if code == "C":
        u, ud = U_j, U_dot_j
        top = jnp.hstack([-dd @ u - dt @ ud, ud])
        bot = jnp.hstack([ud, jnp.zeros((3, 1))])
        return jnp.vstack([top, bot])                                  # (6,2)
    if code == "S":
        return jnp.vstack([-dd, jnp.zeros((3, 3))])                   # (6,3)
    if code == "F":
        Z3 = jnp.zeros((3, 3))
        return jnp.vstack([jnp.hstack([Z3, -dd]),
                            jnp.hstack([Z3, Z3])])                    # (6,6)
    raise ValueError(f"Unsupported joint code {code!r}.")


def _update_Bdot_block_jax(code, prev_block, delta, delta_dot, U_j, U_dot_j):
    """Incrementally update a Bdot block for the next downstream body."""
    sdd = _skew_jax(delta_dot)
    if code in ("R", "U"):
        top = prev_block[:3, :] - sdd @ U_j - _skew_jax(delta) @ U_dot_j
        return prev_block.at[:3, :].set(top)
    if code == "P":
        return prev_block
    if code == "C":
        val = prev_block[0:3, 0:1] - sdd @ U_j - _skew_jax(delta) @ U_dot_j
        return prev_block.at[0:3, 0:1].set(val)
    if code == "S":
        return prev_block.at[:3, :].set(prev_block[:3, :] - sdd)
    if code == "F":
        return prev_block.at[0:3, 3:6].set(prev_block[0:3, 3:6] - sdd)
    raise ValueError(f"Unsupported joint code {code!r}.")


# ==================== Recursive Assembly =====================================

def _assemble_B_recursive_jax(
    r_abs,
    rJ,
    U,
    *,
    n_bodies,
    total_dof,
    codes,
    body_paths,
    joint_paths,
    col_slices,
):
    """Recursive downstream B assembly (pure JAX, functional array updates)."""
    n_rows = 6 * n_bodies
    B = jnp.zeros((n_rows, total_dof))
    Btrack = set()

    for body_path, joint_path in zip(body_paths, joint_paths):
        for i, j in enumerate(joint_path):
            code = codes[j]
            U_j  = U[j]
            prev_k     = None
            prev_d_kj  = None
            prev_block = None

            for k in body_path[i:]:
                if (k, j) in Btrack:
                    continue
                Btrack.add((k, j))

                if prev_k is None:
                    d_kj  = r_abs[k] - rJ[j]
                    block = _block_B_jax(code, d_kj, U_j)
                else:
                    delta = r_abs[k] - r_abs[prev_k]
                    d_kj  = prev_d_kj + delta
                    block = _update_B_block_jax(code, prev_block, delta, U_j)

                r0 = 6 * (k - 1)
                cs = col_slices[j]
                B = B.at[r0:r0 + 6, cs.start:cs.stop].set(block)

                prev_k     = k
                prev_d_kj  = d_kj
                prev_block = block

    return B


def _assemble_Bdot_recursive_jax(
    r_abs,
    rJ,
    U,
    v_abs,
    vJ,
    Udot,
    *,
    n_bodies,
    total_dof,
    codes,
    body_paths,
    joint_paths,
    col_slices,
):
    """Recursive downstream Bdot assembly (pure JAX, functional array updates)."""
    n_rows = 6 * n_bodies
    Bdot = jnp.zeros((n_rows, total_dof))
    Btrack = set()

    for body_path, joint_path in zip(body_paths, joint_paths):
        for i, j in enumerate(joint_path):
            code    = codes[j]
            U_j     = U[j]
            U_dot_j = Udot[j]
            prev_k        = None
            prev_d_kj     = None
            prev_d_dot_kj = None
            prev_block    = None

            for k in body_path[i:]:
                if (k, j) in Btrack:
                    continue
                Btrack.add((k, j))

                if prev_k is None:
                    d_kj     = r_abs[k] - rJ[j]
                    d_dot_kj = v_abs[k] - vJ[j]
                    block    = _block_Bdot_jax(code, d_kj, d_dot_kj, U_j, U_dot_j)
                else:
                    delta     = r_abs[k] - r_abs[prev_k]
                    delta_dot = v_abs[k] - v_abs[prev_k]
                    d_kj      = prev_d_kj + delta
                    d_dot_kj  = prev_d_dot_kj + delta_dot
                    block     = _update_Bdot_block_jax(
                        code, prev_block, delta, delta_dot, U_j, U_dot_j,
                    )

                r0 = 6 * (k - 1)
                cs = col_slices[j]
                Bdot = Bdot.at[r0:r0 + 6, cs.start:cs.stop].set(block)

                prev_k        = k
                prev_d_kj     = d_kj
                prev_d_dot_kj = d_dot_kj
                prev_block    = block

    return Bdot
