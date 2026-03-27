# source/multibody_3d/multibody_core/mbd_system_3d.py
"""
Thin orchestrator façade for 3D multibody kinematics (Phase 1).

Owns ``JointSystem3D``, ``CoordBundle``, and ``VelocityTransformation3D``
and exposes a single-object API for topology, coordinates, and B / Bdot
assembly / evaluation.

Not included (Phase 2+): forces, mass/inertia, EOM assembly, integration,
points/markers, energy.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from types import ModuleType
from typing import Optional

import numpy as np
import sympy as sym

try:
    # package-style imports
    from .joint_system_3d import JointSystem3D
    from .joint_coordinate_3d import CoordBundle, build_joint_coordinates
    from .velocity_transformation_3d import (
        VelocityTransformation3D,
        KinematicsCache3D,
        KinematicsRateCache3D,
        NumericModelParams,
    )
except Exception:  # pragma: no cover
    from joint_system_3d import JointSystem3D
    from joint_coordinate_3d import CoordBundle, build_joint_coordinates
    from velocity_transformation_3d import (
        VelocityTransformation3D,
        KinematicsCache3D,
        KinematicsRateCache3D,
        NumericModelParams,
    )


@dataclass
class MbdSystem3D:
    """Single entry-point for 3D multibody kinematics (Phase 1).

    Construction builds three owned objects in a fixed sequence:

    1. ``joint_system`` — topology, DOF bookkeeping, coordinate mapping.
    2. ``coords``       — symbolic coordinate bundle (user + internal).
    3. ``vt``           — velocity-transformation engine (B / Bdot).

    All downstream methods delegate to these objects; no kinematics logic
    is duplicated here.

    Parameters
    ----------
    data : dict
        Raw geometry dictionary with the keys expected by
        :meth:`JointSystem3D.from_data`.
    """

    # ── Init fields ──────────────────────────────────────────────────────────

    data: dict

    # ── Derived (built in __post_init__) ─────────────────────────────────────

    joint_system: JointSystem3D              = field(init=False, repr=False)
    coords:       CoordBundle                = field(init=False, repr=False)
    vt:           VelocityTransformation3D   = field(init=False, repr=False)
    _numeric_params: Optional[NumericModelParams] = field(
        init=False, repr=False, default=None,
    )

    def __post_init__(self) -> None:
        if not isinstance(self.data, dict):
            raise TypeError(
                f"MbdSystem3D expects a dict, got {type(self.data).__name__}."
            )
        self.joint_system = JointSystem3D.from_data(self.data)
        self.coords       = build_joint_coordinates(self.joint_system)
        self.vt           = VelocityTransformation3D(self.joint_system)

    # ── Private shape validators ─────────────────────────────────────────────

    def _validate_q_user_shape(self, q_user_np) -> np.ndarray:
        """Flatten and check that *q_user_np* has length ``total_user_dof``."""
        arr = np.asarray(q_user_np, dtype=float).ravel()
        if arr.shape[0] != self.total_user_dof:
            raise ValueError(
                f"q_user length mismatch: expected {self.total_user_dof}, "
                f"got {arr.shape[0]}."
            )
        return arr

    def _validate_q_int_shape(self, q_int_np) -> np.ndarray:
        """Flatten and check that *q_int_np* has length ``total_cfg_dof``."""
        arr = np.asarray(q_int_np, dtype=float).ravel()
        if arr.shape[0] != self.total_cfg_dof:
            raise ValueError(
                f"q_int length mismatch: expected {self.total_cfg_dof}, "
                f"got {arr.shape[0]}."
            )
        return arr

    def _validate_qd_shape(self, qd_np) -> np.ndarray:
        """Flatten and check that *qd_np* has length ``total_dof``."""
        arr = np.asarray(qd_np, dtype=float).ravel()
        if arr.shape[0] != self.total_dof:
            raise ValueError(
                f"qd length mismatch: expected {self.total_dof}, "
                f"got {arr.shape[0]}."
            )
        return arr

    def _validate_params(self, params: NumericModelParams) -> None:
        """Check that *params* dimensions match the current system."""
        if params.total_dof != self.total_dof:
            raise ValueError(
                f"NumericModelParams.total_dof={params.total_dof} does not "
                f"match system total_dof={self.total_dof}."
            )
        if params.total_cfg_dof != self.total_cfg_dof:
            raise ValueError(
                f"NumericModelParams.total_cfg_dof={params.total_cfg_dof} does "
                f"not match system total_cfg_dof={self.total_cfg_dof}."
            )

    # ── Alternate constructors ───────────────────────────────────────────────

    @classmethod
    def from_data(cls, data: dict) -> "MbdSystem3D":
        """Construct from a raw geometry dictionary.

        Equivalent to ``MbdSystem3D(data)``; mirrors the
        ``JointSystem3D.from_data`` naming convention.
        """
        return cls(data=data)

    @classmethod
    def from_example(cls, ex: ModuleType) -> "MbdSystem3D":
        """Construct from an example module that exposes a ``data`` dict.

        Parameters
        ----------
        ex : module
            Any module with a module-level ``data`` attribute (e.g.
            ``import example4; MbdSystem3D.from_example(example4)``).
        """
        if not hasattr(ex, "data"):
            raise AttributeError(
                f"Module {ex.__name__!r} has no 'data' attribute."
            )
        return cls(data=ex.data)

    # ── Topology / sizing properties ─────────────────────────────────────────

    @property
    def NBodies(self) -> int:
        """Number of bodies (excluding ground)."""
        return self.joint_system.NBodies

    @property
    def NJoints(self) -> int:
        """Number of joints."""
        return len(self.joint_system.joints)

    @property
    def total_dof(self) -> int:
        """Total speed-level DOF (columns of B)."""
        return self.joint_system.total_dof

    @property
    def total_cfg_dof(self) -> int:
        """Total internal configuration DOF (quaternion for S/F)."""
        return self.joint_system.total_cfg_dof

    @property
    def total_user_dof(self) -> int:
        """Total user-facing configuration DOF."""
        return self.joint_system.total_user_dof

    # ── Coordinate access (delegates to coords) ─────────────────────────────

    @property
    def q_user(self) -> sym.Matrix:
        """User-facing symbolic configuration vector."""
        return self.coords.q_user

    @property
    def qd_user(self) -> sym.Matrix:
        """Generalized-speed vector with user-friendly symbol names.

        Speed coordinates have **no** user/internal split: both ``qd_user``
        and ``qd_int`` are DOF-sized.  They differ only in symbol naming
        style.  Use ``qd_int`` when passing directly to B/Bdot evaluators.
        """
        return self.coords.qd_user

    @property
    def q_int(self) -> sym.Matrix:
        """Internal symbolic configuration vector.

        S/F joints always use quaternion entries here (4 or 7 components).
        This is the vector consumed by B/Bdot evaluators.  Use ``q_user``
        when working in Euler-angle space.
        """
        return self.coords.q_int

    @property
    def qd_int(self) -> sym.Matrix:
        """Generalized-speed vector with internal symbol names (DOF-sized).

        Identical dimension to ``qd_user``; preferred when building symbolic
        B/Bdot expressions or passing to lambdified/JAX evaluators.
        """
        return self.coords.qd_int

    # ── Coordinate mapping (delegates to joint_system) ───────────────────────

    def map_q_user_to_q_int(self, q_user_np) -> np.ndarray:
        """Map user-facing config to internal config (Euler → quaternion)."""
        q_user_np = self._validate_q_user_shape(q_user_np)
        return self.joint_system.map_q_user_to_q_int(q_user_np)

    def map_q_int_to_q_user(self, q_int_np) -> np.ndarray:
        """Map internal config to user-facing config (quaternion → Euler)."""
        q_int_np = self._validate_q_int_shape(q_int_np)
        return self.joint_system.map_q_int_to_q_user(q_int_np)

    # ── Symbolic assembly (delegates to vt) ──────────────────────────────────

    def assemble_B_symbolic(
        self,
        *,
        cache: Optional[KinematicsCache3D] = None,
    ) -> sym.Matrix:
        """Assemble the full symbolic B matrix.

        Uses ``self.q_int`` automatically.

        Returns
        -------
        sympy.Matrix, shape ``(6*NBodies, total_dof)``
        """
        return self.vt.assemble_B_symbolic(self.coords.q_int, cache=cache)

    def assemble_Bdot_symbolic(
        self,
        *,
        cache: Optional[KinematicsCache3D] = None,
        rate_cache: Optional[KinematicsRateCache3D] = None,
    ) -> sym.Matrix:
        """Assemble the full symbolic Bdot matrix.

        Uses ``self.q_int`` and ``self.qd_int`` automatically.

        Returns
        -------
        sympy.Matrix, shape ``(6*NBodies, total_dof)``
        """
        return self.vt.assemble_Bdot_symbolic(
            self.coords.q_int,
            self.coords.qd_int,
            cache=cache,
            rate_cache=rate_cache,
        )

    # ── Lambdified compilation (delegates to vt) ────────────────────────────

    def compile_B_lambdified(self) -> callable:
        """Compile B to a fast NumPy callable.

        Returns
        -------
        callable
            ``f(q_int_np) -> np.ndarray``
        """
        return self.vt.compile_B_lambdified(self.coords.q_int)

    def compile_Bdot_lambdified(self) -> callable:
        """Compile Bdot to a fast NumPy callable.

        Returns
        -------
        callable
            ``f(q_int_np, qd_np) -> np.ndarray``
        """
        return self.vt.compile_Bdot_lambdified(
            self.coords.q_int, self.coords.qd_int,
        )

    # ── Numeric params (delegates to vt, cached) ────────────────────────────

    def build_numeric_params(self, *, force: bool = False) -> NumericModelParams:
        """Extract constant geometry as immutable NumPy arrays.

        The result is cached after the first call.  Pass ``force=True`` to
        discard the cache and rebuild (e.g. after mutating ``data`` and
        reconstructing the system).

        Parameters
        ----------
        force : bool, optional
            If *True*, rebuild even if a cached result exists.

        Returns
        -------
        NumericModelParams
        """
        if force or self._numeric_params is None:
            self._numeric_params = self.vt.build_numeric_params()
        return self._numeric_params

    # ── JAX evaluation (delegates to vt) ─────────────────────────────────────

    def evaluate_B_jax(self, q_int_np, *, params=None):
        """Evaluate B using the JAX backend (eager, not JIT).

        Parameters
        ----------
        q_int_np : array_like, shape ``(total_cfg_dof,)``
        params : NumericModelParams, optional
            Falls back to cached params if *None*.

        Returns
        -------
        jnp.ndarray, shape ``(6*NBodies, total_dof)``
        """
        q_int_np = self._validate_q_int_shape(q_int_np)
        if params is None:
            params = self.build_numeric_params()
        self._validate_params(params)
        return self.vt.evaluate_B_jax(q_int_np, params=params)

    def evaluate_Bdot_jax(self, q_int_np, qd_np, *, params=None):
        """Evaluate Bdot using the JAX backend (eager, not JIT).

        Parameters
        ----------
        q_int_np : array_like, shape ``(total_cfg_dof,)``
        qd_np : array_like, shape ``(total_dof,)``
        params : NumericModelParams, optional
            Falls back to cached params if *None*.

        Returns
        -------
        jnp.ndarray, shape ``(6*NBodies, total_dof)``
        """
        q_int_np = self._validate_q_int_shape(q_int_np)
        qd_np = self._validate_qd_shape(qd_np)
        if params is None:
            params = self.build_numeric_params()
        self._validate_params(params)
        return self.vt.evaluate_Bdot_jax(q_int_np, qd_np, params=params)

    def build_B_evaluator_jax(self, *, params=None) -> callable:
        """Return a JIT-compiled ``f(q_int) -> B``.

        Parameters
        ----------
        params : NumericModelParams, optional
            Falls back to cached params if *None*.

        Returns
        -------
        callable
            ``f(q_int: jnp.ndarray) -> jnp.ndarray``
        """
        if params is None:
            params = self.build_numeric_params()
        self._validate_params(params)
        return self.vt.build_B_evaluator_jax(params=params)

    def build_Bdot_evaluator_jax(self, *, params=None) -> callable:
        """Return a JIT-compiled ``f(q_int, qd) -> Bdot``.

        Parameters
        ----------
        params : NumericModelParams, optional
            Falls back to cached params if *None*.

        Returns
        -------
        callable
            ``f(q_int: jnp.ndarray, qd: jnp.ndarray) -> jnp.ndarray``
        """
        if params is None:
            params = self.build_numeric_params()
        self._validate_params(params)
        return self.vt.build_Bdot_evaluator_jax(params=params)

    # ── User-facing runtime helpers ─────────────────────────────────────────

    def evaluate_B_from_user_q(self, q_user_np, *, params=None):
        """Evaluate B from a **user-facing** configuration vector.

        Internally maps ``q_user → q_int`` (e.g. Euler angles → quaternion)
        then evaluates via the JAX backend.

        Parameters
        ----------
        q_user_np : array_like, shape ``(total_user_dof,)``
            User-facing configuration (Euler angles for S/F joints).
        params : NumericModelParams, optional
            Falls back to cached params if *None*.

        Returns
        -------
        jnp.ndarray, shape ``(6*NBodies, total_dof)``
        """
        q_user_np = self._validate_q_user_shape(q_user_np)
        q_int_np = self.joint_system.map_q_user_to_q_int(q_user_np)
        if params is None:
            params = self.build_numeric_params()
        self._validate_params(params)
        return self.vt.evaluate_B_jax(q_int_np, params=params)

    def evaluate_Bdot_from_user_state(self, q_user_np, qd_np, *, params=None):
        """Evaluate Bdot from a **user-facing** configuration and speed vector.

        Internally maps ``q_user → q_int``.  The speed vector ``qd`` is
        **not** remapped — it must already be the DOF-sized generalized-speed
        vector expected by the velocity-transformation layer.

        Parameters
        ----------
        q_user_np : array_like, shape ``(total_user_dof,)``
            User-facing configuration (Euler angles for S/F joints).
        qd_np : array_like, shape ``(total_dof,)``
            Generalized speeds (DOF-sized, same basis as columns of B).
        params : NumericModelParams, optional
            Falls back to cached params if *None*.

        Returns
        -------
        jnp.ndarray, shape ``(6*NBodies, total_dof)``
        """
        q_user_np = self._validate_q_user_shape(q_user_np)
        qd_np = self._validate_qd_shape(qd_np)
        q_int_np = self.joint_system.map_q_user_to_q_int(q_user_np)
        if params is None:
            params = self.build_numeric_params()
        self._validate_params(params)
        return self.vt.evaluate_Bdot_jax(q_int_np, qd_np, params=params)

    def build_B_evaluator_from_user_q(self, *, params=None) -> callable:
        """Return a callable ``f(q_user) -> B`` that accepts user-facing config.

        The returned function maps ``q_user → q_int`` internally and then
        evaluates the JIT-compiled B evaluator.

        Parameters
        ----------
        params : NumericModelParams, optional
            Falls back to cached params if *None*.

        Returns
        -------
        callable
            ``f(q_user_np: array_like) -> jnp.ndarray``
        """
        if params is None:
            params = self.build_numeric_params()
        self._validate_params(params)
        jit_fn = self.vt.build_B_evaluator_jax(params=params)
        js = self.joint_system
        _check_q_user = self._validate_q_user_shape

        def _eval_B_user(q_user_np):
            return jit_fn(js.map_q_user_to_q_int(_check_q_user(q_user_np)))

        return _eval_B_user

    def build_Bdot_evaluator_from_user_state(self, *, params=None) -> callable:
        """Return a callable ``f(q_user, qd) -> Bdot`` that accepts user-facing config.

        The returned function maps ``q_user → q_int`` internally.  The speed
        vector ``qd`` is passed through unchanged (DOF-sized).

        Parameters
        ----------
        params : NumericModelParams, optional
            Falls back to cached params if *None*.

        Returns
        -------
        callable
            ``f(q_user_np: array_like, qd_np: array_like) -> jnp.ndarray``
        """
        if params is None:
            params = self.build_numeric_params()
        self._validate_params(params)
        jit_fn = self.vt.build_Bdot_evaluator_jax(params=params)
        js = self.joint_system
        _check_q_user = self._validate_q_user_shape
        _check_qd = self._validate_qd_shape

        def _eval_Bdot_user(q_user_np, qd_np):
            return jit_fn(
                js.map_q_user_to_q_int(_check_q_user(q_user_np)),
                _check_qd(qd_np),
            )

        return _eval_Bdot_user

    # ── Ergonomic helpers ────────────────────────────────────────────────────

    def summary_table(self, precision: int = 3):
        """Print a summary table of joint information (delegates to joint_system)."""
        return self.joint_system.summary_table(precision=precision)

    def __repr__(self) -> str:
        return (
            f"MbdSystem3D(NBodies={self.NBodies}, NJoints={self.NJoints}, "
            f"total_dof={self.total_dof}, total_cfg_dof={self.total_cfg_dof}, "
            f"total_user_dof={self.total_user_dof})"
        )
