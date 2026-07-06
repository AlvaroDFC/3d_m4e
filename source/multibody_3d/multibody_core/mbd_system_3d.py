# source/multibody_3d/multibody_core/mbd_system_3d.py
"""
Orchestrator façade for 3D multibody kinematics and forces.

Owns ``JointSystem3D``, ``CoordBundle``, ``VelocityTransformation3D``,
the symbolic / numeric points cache, and (when declared) the full force
layer: parsed definitions, symbolic wrenches, and persistent JAX evaluators.

Not included: EOM assembly, numerical integration.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from types import ModuleType
from typing import Callable, Optional

import numpy as np
import sympy as sym

# package-style imports
from .joint_system_3d import JointSystem3D
from .joint_coordinate_3d import CoordBundle, build_joint_coordinates
from .velocity_transformation_3d import (
    VelocityTransformation3D,
    KinematicsCache3D,
    KinematicsRateCache3D,
    NumericModelParams,
    make_B_evaluator_mainint,
    make_Bdot_evaluator_mainint,
)
from .points_3d import (
    PointRecord3D,
    SymbolicPointsCache3D,
    PointsEvalResult,
    PointsRuntimeSpec,
    build_points_symbolic,
    make_points_evaluator_mainint,
)
from .force_definition_3d import (
    ForcesDefinition3D,
    CGForceDef,
    PointForceDef,
    TensionSpringDef,
    TensionDamperDef,
    TorsionSpringDef,
    TorsionDamperDef,
    parse_force_dict,
)
from .forces_symbolic_3d import (
    SymbolicForcesCache3D,
    build_forces_symbolic,
)
from .forces_runtime_3d import (
    ForcesEvalResult,
    make_forces_evaluator_mainint,
    )
from .mass_symbolic_3d import (
    SymbolicMassCache3D,
    build_mass_symbolic,
)
from .mass_runtime_3d import (
    MassEvalResult,
    make_mass_evaluator_mainint,
    EomKernelResult,
    make_eom_evaluator_mainint,
)


@dataclass
class MbdSystem3D:
    """Single entry-point for 3D multibody kinematics and forces.

    Construction builds the following owned objects in a fixed sequence:

    1. ``joint_system``  — topology, DOF bookkeeping, coordinate mapping.
    2. ``coords``        — symbolic coordinate bundle (user + internal).
    3. ``vt``            — velocity-transformation engine (B / Bdot).
    4. ``sym_points``    — symbolic point records (when ``Initial_Points`` is
       non-empty or force elements reference body points).
    5. ``forces_def``    — parsed force definitions (when ``Force`` is
       non-empty).
    6. ``sym_forces``    — symbolic per-body wrench cache, one ``(6,1)``
       expression per body per category, assembled from ``forces_def``.
    7. ``forces_func``   — JIT-compiled JAX callable that evaluates all
       wrenches and spring PE from a ``mainNumVars_int`` vector.

    Force ownership model
    ---------------------
    * **Force definitions** live in ``self.forces_def``
      (:class:`~force_definition_3d.ForcesDefinition3D`).  Parsed from the
      user-supplied ``Force`` dict; contains typed records for CG forces,
      body-point forces, tension/torsion elements, and gravity.

    * **Symbolic force objects** live in ``self.sym_forces``
      (:class:`~forces_symbolic_3d.SymbolicForcesCache3D`).  Provides
      ``wrench_by_category``, ``total_wrench`` (list of ``(6,1)``
      :class:`sympy.Matrix` per body), and ``spring_potential_energy``.
      Convenience properties :attr:`sym_total_wrench` and
      :attr:`sym_spring_pe` expose the most common entries directly.

    * **JAX force evaluators** live in ``self.forces_func``.  A single
      ``@jax.jit`` closure compiled once at construction, reused across all
      evaluations.  Accepts ``mainNumVars_int`` and returns a
      :class:`~forces_runtime_3d.ForcesEvalResult`.

    * **Total forces** are exposed via
      :meth:`evaluate_forces` (returns the full per-category result) and
      :meth:`evaluate_total_wrench` (returns only the ``(NBodies, 6)``
      sum). The symbolic total is available via :attr:`sym_total_wrench`.

    All setup is fully automatic in ``__post_init__``; no user calls are
    needed beyond constructing the object or calling
    :meth:`from_example`.

    Parameters
    ----------
    data : dict
        Raw geometry dictionary with the keys expected by
        :meth:`JointSystem3D.from_data`.
    """

    # ── Init fields ──────────────────────────────────────────────────────────

    data: dict
    force_points_sym: dict = field(default_factory=dict)
    body_data_sym: dict    = field(default_factory=dict)
    #: User-facing force-element dictionary.  Supported top-level keys:
    #: ``"CG"``, ``"PointsBD"``, ``"TensionSpring"``, ``"TensionDamper"``,
    #: ``"TorsionSpring"``, ``"TorsionDamper"``.
    #: Parsed immediately into :attr:`forces_def` during ``__post_init__``.
    Force: dict            = field(default_factory=dict)
    #: Flat ordered dict of symbolic point-component parameters.
    #: Keys are user-chosen names; values are SymPy symbols that appear as
    #: components in ``Initial_Points`` entries.  User-declared order is
    #: preserved and propagated into ``mainSymVars`` / ``mainSymVars_int``.
    points_sym: dict       = field(default_factory=dict)
    #: Per-body mass and body-frame inertia tensor.
    #: Keys are 1-based body ids; values are dicts with:
    #:
    #:   ``"mass"`` — scalar (float or sym.Expr).
    #:
    #:   ``"J"``    — (3, 3) array-like or sym.Matrix expressed in the
    #:                body's own reference frame.
    #:
    #: When non-empty, :attr:`sym_mass` and :attr:`mass_func` are built
    #: automatically during ``__post_init__``.
    body_inertia: dict     = field(default_factory=dict)
    #: Point-definition dictionary with two sub-keys:
    #:
    #: ``"GR"`` — list of world-frame 3-D points ``[x, y, z]`` (numeric or
    #:            symbolic).
    #:
    #: ``"BD"`` — ``dict[int, list[[x, y, z]]]`` of body-local points keyed
    #:            by integer body id (1-based, excluding ground).  Coordinates
    #:            are expressed in the body's own reference frame at the CG.
    #:
    #: Any free SymPy symbols contained in the entries must also appear in
    #: ``points_sym`` so that they participate in the canonical variable
    #: bookkeeping.
    Initial_Points: dict   = field(default_factory=dict)

    # ── Derived (built in __post_init__) ─────────────────────────────────────

    joint_system: JointSystem3D              = field(init=False, repr=False)
    coords:       CoordBundle                = field(init=False, repr=False)
    vt:           VelocityTransformation3D   = field(init=False, repr=False)
    _numeric_params: Optional[NumericModelParams] = field(
        init=False, repr=False, default=None,
    )
    B_func:    Optional[Callable] = field(init=False, repr=False, default=None)
    Bdot_func: Optional[Callable] = field(init=False, repr=False, default=None)
    _geom_extractor: object        = field(init=False, repr=False, default=None)
    #: Symbolic 3-D points cache, built on construction when
    #: ``Initial_Points`` is non-empty.  ``None`` otherwise.
    sym_points: Optional[SymbolicPointsCache3D] = field(
        init=False, repr=False, default=None,
    )
    #: JIT-compiled points callable ``f(mainNumVars_int) -> PointsEvalResult``.
    #: Built when ``Initial_Points`` is non-empty; ``None`` otherwise.
    points_func: Optional[Callable] = field(init=False, repr=False, default=None)
    #: Static layout metadata companion to ``points_func``.
    _points_spec: Optional[object] = field(init=False, repr=False, default=None)
    #: Parsed force elements produced from :attr:`Force` by
    #: :func:`force_definition_3d.parse_force_dict`.
    #: ``None`` when :attr:`Force` is empty.
    forces_def: Optional[ForcesDefinition3D] = field(
        init=False, repr=False, default=None,
    )
    #: Symbolic per-body wrench cache produced from :attr:`forces_def` and
    #: the position-level kinematics cache.  Built whenever :attr:`Force`
    #: is non-empty; ``None`` otherwise.
    sym_forces: Optional[SymbolicForcesCache3D] = field(
        init=False, repr=False, default=None,
    )
    #: JIT-compiled force callable ``f(mainNumVars_int) -> ForcesEvalResult``.
    #: Built when :attr:`Force` is non-empty; ``None`` otherwise.
    forces_func: Optional[Callable] = field(
        init=False, repr=False, default=None,
    )
    #: Symbolic mass / inertia cache (one :class:`~mass_symbolic_3d.BodyInertiaRecord`
    #: per body).  Built when :attr:`body_inertia` is non-empty; ``None`` otherwise.
    sym_mass: Optional[SymbolicMassCache3D] = field(
        init=False, repr=False, default=None,
    )
    #: JIT-compiled mass-matrix callable
    #: ``f(mainNumVars_int) -> MassEvalResult``.
    #: Built when :attr:`body_inertia` is non-empty; ``None`` otherwise.
    mass_func: Optional[Callable] = field(
        init=False, repr=False, default=None,
    )
    #: JIT-compiled combined EOM kernel callable
    #: ``f(mainNumVars_int) -> EomKernelResult``.
    #: Returns B, Bdot, M_body, M in a single kinematics pass.
    #: Built when :attr:`body_inertia` is non-empty; ``None`` otherwise.
    eom_func: Optional[Callable] = field(
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
        # Parse Force dictionary early so validation fires before any
        # expensive symbolic or JAX work.
        if self.Force:
            self.forces_def = parse_force_dict(
                self.Force,
                n_bodies=self.joint_system.NBodies,
                joint_types=self.data.get("types"),
            )
        self.mainSymVars_int = (list(self.coords.q_int) +
                                list(self.coords.qd_int) +
                                list(self.body_data_sym.values()) +
                                list(self.force_points_sym.values()) +
                                list(self.points_sym.values()))
        self.mainSymVars = (list(self.coords.q_user) +
                            list(self.coords.qd_user) +
                            list(self.body_data_sym.values()) +
                            list(self.force_points_sym.values()) +
                            list(self.points_sym.values()))
        self._build_slice_metadata()
        # Build symbolic points cache and JAX point evaluator if Initial_Points was supplied.
        # TODO: initial points and forces defnition may have different symbolic parameters. Are both allowed?
        if self.Initial_Points:
            _pos_cache = self.vt.build_cache_symbolic(self.coords.q_int)
            self.sym_points = build_points_symbolic(
                self.Initial_Points, _pos_cache, self.NBodies,
            )
        elif self.forces_def is not None or self.body_inertia:
            # No Initial_Points, but Force or body_inertia is declared:
            # build pos_cache for torsion axes / world-frame inertia;
            # sym_points gets only CG records.
            _pos_cache = self.vt.build_cache_symbolic(self.coords.q_int)
            self.sym_points = build_points_symbolic({}, _pos_cache, self.NBodies)
        else:
            _pos_cache = None
        # Build runtime state once: frozen geometry params + JIT-compiled evaluators.
        self._numeric_params = self.vt.build_numeric_params()
        self._geom_extractor = self.vt.build_geometry_extractor(
            list(self.body_data_sym.values()), params=self._numeric_params,
        )
        self.B_func    = make_B_evaluator_mainint(
            self._numeric_params,
            self._slc_q_int, self._slc_qd_int,
            self._geom_extractor, self._slc_body_int,
        )
        self.Bdot_func = make_Bdot_evaluator_mainint(
            self._numeric_params,
            self._slc_q_int, self._slc_qd_int,
            self._geom_extractor, self._slc_body_int,
        )

        # Build symbolic points cache and JAX point evaluator if Initial_Points was supplied.
        # TODO: initial points and forces defnition may have different symbolic parameters. Are both allowed?
        if self.Initial_Points:
            _pos_cache = self.vt.build_cache_symbolic(self.coords.q_int)
            self.sym_points = build_points_symbolic(
                self.Initial_Points, _pos_cache, self.NBodies,
            )
        elif self.forces_def is not None:
            # No Initial_Points, but Force is declared: build pos_cache
            # for torsion axes; sym_points gets only CG records.
            _pos_cache = self.vt.build_cache_symbolic(self.coords.q_int)
            self.sym_points = build_points_symbolic({}, _pos_cache, self.NBodies)
        else:
            _pos_cache = None
        # Build JAX point evaluator if Initial_Points was supplied.
        if not self.sym_points and self.Initial_Points:
            self.points_func, self._points_spec = make_points_evaluator_mainint(
                self.sym_points,
                list(self.points_sym.values()),
                self._numeric_params,
                self._slc_q_int,
                self._geom_extractor,
                self._slc_body_int,
                self._slc_points_int,
                body_sym_list=list(self.body_data_sym.values()),
            )
        # Build symbolic force cache if Force was declared.
        if self.forces_def is not None and _pos_cache is not None:
            # Build rate cache only when tension dampers need point velocities.
            _rate_cache = None
            if self.forces_def.tension_dampers:
                _rate_cache = self.vt.build_rate_cache_symbolic(
                    self.coords.q_int,
                    self.coords.qd_int,
                    cache=_pos_cache,
                )
            self.sym_forces = build_forces_symbolic(
                self.forces_def,
                self.sym_points,   # may be minimal (CG-only) or full
                self.coords,
                _pos_cache,
                _rate_cache,
                self.NBodies,
            )
            # Build JAX force evaluator from the parsed force definitions.
            self.forces_func = make_forces_evaluator_mainint(
                self.forces_def,
                self.sym_points,
                self.coords,
                list(self.force_points_sym.values()),
                list(self.points_sym.values()),
                self._numeric_params,
                self._slc_q_int,
                self._slc_qd_int,
                self._slc_body_int,
                self._slc_force_int,
                self._slc_points_int,
                self._geom_extractor,
                self.NBodies,
                body_sym_list=list(self.body_data_sym.values()),
            )
        # Build symbolic mass cache and JAX mass evaluator if body_inertia
        # was declared.
        if self.body_inertia and _pos_cache is not None:
            self.sym_mass = build_mass_symbolic(
                self.body_inertia, _pos_cache, self.NBodies,
            )
            self.mass_func = make_mass_evaluator_mainint(
                self.body_inertia,
                self._numeric_params,
                self._slc_q_int,
                self._slc_qd_int,
                self._slc_body_int,
                self._geom_extractor,
                self.NBodies,
                body_sym_list=list(self.body_data_sym.values()),
            )
            self.eom_func = make_eom_evaluator_mainint(
                self.body_inertia,
                self._numeric_params,
                self._slc_q_int,
                self._slc_qd_int,
                self._slc_body_int,
                self._geom_extractor,
                self.NBodies,
                body_sym_list=list(self.body_data_sym.values()),
            )

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

    # ── mainNumVars helpers ──────────────────────────────────────────────────

    def _build_slice_metadata(self) -> None:
        """Compute and cache slice objects for ``mainNumVars`` and ``mainNumVars_int``.

        Called once from ``__post_init__`` after the symbolic lists are built.
        All slices are derived from existing DOF metadata; nothing is hardcoded.

        User-facing slices (index into ``mainNumVars`` / ``mainSymVars``)
        ──────────────────────────────────────────────────────────────────
        ``_slc_q_user``  — configuration coordinates (length ``total_user_dof``)
        ``_slc_qd``      — generalized speeds       (length ``total_dof``)
        ``_slc_body``    — body-data parameters     (length ``len(body_data_sym)``)
        ``_slc_force``   — force-point parameters   (length ``len(force_points_sym)``)

        Internal slices (index into ``mainNumVars_int`` / ``mainSymVars_int``)
        ───────────────────────────────────────────────────────────────────────
        ``_slc_q_int``    — internal configuration  (length ``total_cfg_dof``)
        ``_slc_qd_int``   — generalized speeds      (same length as ``_slc_qd``)
        ``_slc_body_int`` — body-data parameters    (same symbols as ``_slc_body``)
        ``_slc_force_int``  — force-point parameters  (same symbols as ``_slc_force``)

        Internal slices — point symbols
        ────────────────────────────────
        ``_slc_points``    — point-component parameters (length ``len(points_sym)``)
        ``_slc_points_int``— point-component parameters (same symbols as ``_slc_points``)
        """
        n_qu = self.total_user_dof
        n_qi = self.total_cfg_dof
        n_qd = self.total_dof
        n_bp = len(self.body_data_sym)
        n_fp = len(self.force_points_sym)
        n_pp = len(self.points_sym)

        # User-facing (mainNumVars / mainSymVars)
        self._slc_q_user    = slice(0,                       n_qu)
        self._slc_qd        = slice(n_qu,                    n_qu + n_qd)
        self._slc_body      = slice(n_qu + n_qd,             n_qu + n_qd + n_bp)
        self._slc_force     = slice(n_qu + n_qd + n_bp,      n_qu + n_qd + n_bp + n_fp)
        self._slc_points    = slice(n_qu + n_qd + n_bp + n_fp, n_qu + n_qd + n_bp + n_fp + n_pp)

        # Internal (mainNumVars_int / mainSymVars_int)
        self._slc_q_int     = slice(0,                       n_qi)
        self._slc_qd_int    = slice(n_qi,                    n_qi + n_qd)
        self._slc_body_int  = slice(n_qi + n_qd,             n_qi + n_qd + n_bp)
        self._slc_force_int = slice(n_qi + n_qd + n_bp,      n_qi + n_qd + n_bp + n_fp)
        self._slc_points_int= slice(n_qi + n_qd + n_bp + n_fp, n_qi + n_qd + n_bp + n_fp + n_pp)

        # Sanity checks
        n_main     = n_qu + n_qd + n_bp + n_fp + n_pp
        n_main_int = n_qi + n_qd + n_bp + n_fp + n_pp
        if len(self.mainSymVars) != n_main:
            raise AssertionError(
                f"mainSymVars length {len(self.mainSymVars)} != expected {n_main}"
            )
        if len(self.mainSymVars_int) != n_main_int:
            raise AssertionError(
                f"mainSymVars_int length {len(self.mainSymVars_int)} "
                f"!= expected {n_main_int}"
            )
        if self.mainSymVars[self._slc_body] != self.mainSymVars_int[self._slc_body_int]:
            raise AssertionError(
                "body_data_sym order mismatch between mainSymVars and mainSymVars_int"
            )
        if self.mainSymVars[self._slc_force] != self.mainSymVars_int[self._slc_force_int]:
            raise AssertionError(
                "force_points_sym order mismatch between mainSymVars and mainSymVars_int"
            )
        if self.mainSymVars[self._slc_points] != self.mainSymVars_int[self._slc_points_int]:
            raise AssertionError(
                "points_sym order mismatch between mainSymVars and mainSymVars_int"
            )

    def _validate_mainNumVars_shape(self, mainNumVars) -> np.ndarray:
        """Flatten and check that *mainNumVars* has the expected length.

        Expected layout: ``[q_user, qd, body_params, force_params]``
        matching ``self.mainSymVars``.

        Returns
        -------
        np.ndarray, shape ``(len(mainSymVars),)``
        """
        arr = np.asarray(mainNumVars, dtype=float).ravel()
        expected = self._slc_points.stop
        if arr.shape[0] != expected:
            n_qu = self._slc_q_user.stop
            n_qd = self._slc_qd.stop  - self._slc_qd.start
            n_bp = self._slc_body.stop  - self._slc_body.start
            n_fp = self._slc_force.stop - self._slc_force.start
            n_pp = self._slc_points.stop - self._slc_points.start
            raise ValueError(
                f"mainNumVars length mismatch: expected {expected}, "
                f"got {arr.shape[0]}. "
                f"Expected layout: [q_user({n_qu}), qd({n_qd}), "
                f"body_params({n_bp}), force_params({n_fp}), "
                f"point_params({n_pp})]."
            )
        return arr

    def _split_mainNumVars(
        self, mainNumVars: np.ndarray
    ) -> "tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]":
        """Split a validated *mainNumVars* vector into its four components.

        Parameters
        ----------
        mainNumVars : np.ndarray, shape ``(len(mainSymVars),)``
            Validated (already flattened) user-facing variable vector.

        Returns
        -------
        q_user_np, qd_np, body_params_np, force_params_np, point_params_np : np.ndarray
        """
        # TODO: is this used?
        return (
            mainNumVars[self._slc_q_user],
            mainNumVars[self._slc_qd],
            mainNumVars[self._slc_body],
            mainNumVars[self._slc_force],
            mainNumVars[self._slc_points],
        )

    # TODO: this may be unnecessary as it should already exist?
    def _build_mainNumVars_int(self, mainNumVars: np.ndarray) -> np.ndarray:
        """Build the internal variable vector from a validated *mainNumVars*.

        Maps ``q_user → q_int`` via the joint-system mapping; ``qd``,
        ``body_params``, and ``force_params`` are copied unchanged.

        The returned array is **transient** — it is never stored as object state.

        Returns
        -------
        np.ndarray, shape ``(len(mainSymVars_int),)``
            Ordered as ``[q_int, qd, body_params, force_params]``.
        """
       
        q_int_np = self.joint_system.map_q_user_to_q_int(
            mainNumVars[self._slc_q_user]
        )
        return np.concatenate([
            q_int_np,
            mainNumVars[self._slc_qd],
            mainNumVars[self._slc_body],
            mainNumVars[self._slc_force],
            mainNumVars[self._slc_points],
        ])

    # TODO: this may be unnecessary as it should already exist?
    def _extract_q_int_qd(
        self, mainNumVars
    ) -> "tuple[np.ndarray, np.ndarray]":
        """Validate *mainNumVars* and return ``(q_int_np, qd_np)``.

        Common first step shared by all runtime evaluator paths:
        validates shape, extracts ``q_user`` and ``qd`` slices, and maps
        ``q_user -> q_int`` via the joint-system mapping.

        Parameters
        ----------
        mainNumVars : array_like, shape ``(len(mainSymVars),)``

        Returns
        -------
        q_int_np : np.ndarray, shape ``(total_cfg_dof,)``
        qd_np : np.ndarray, shape ``(total_dof,)``
        """
        # TODO: is this necessary?
        arr = self._validate_mainNumVars_shape(mainNumVars)
        q_int_np = self.joint_system.map_q_user_to_q_int(arr[self._slc_q_user])
        return q_int_np, arr[self._slc_qd]

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
            raise AttributeError(f"Module {ex.__name__!r} has no 'data' attribute.")

        # Merge legacy force_points_sym with the new force_sym dict
        # (both optional; new force_sym takes precedence on key collision).
        fp_sym  = getattr(ex, "force_points_sym", {})
        f_sym   = getattr(ex, "force_sym", {})
        merged_force_sym = {**fp_sym, **f_sym}
        return cls(
            data=ex.data,
            body_data_sym=getattr(ex, "body_data_sym", {}),
            force_points_sym=merged_force_sym,
            points_sym=getattr(ex, "points_sym", {}),
            Initial_Points=getattr(ex, "Initial_Points", {}),
            Force=getattr(ex, "Force", {}),
            body_inertia=getattr(ex, "body_inertia", {}),
        )

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

    # ── Symbolic force access ────────────────────────────────────────────────

    @property
    def sym_total_wrench(self) -> "Optional[list]":
        """Symbolic total wrench per body (sum of all force categories).

        Returns
        -------
        list[sym.Matrix] or None
            Length-``NBodies`` list of ``(6, 1)`` :class:`sympy.Matrix`
            objects (world-frame ``[Fx, Fy, Fz, Mx, My, Mz]`` about CG),
            or *None* when no ``Force`` dictionary was declared.
        """
        return self.sym_forces.total_wrench if self.sym_forces is not None else None

    @property
    def sym_spring_pe(self):
        """Symbolic total spring potential energy.

        Returns
        -------
        sym.Expr or None
            Sum of ``0.5*k*(L-L0)^2`` for every tension spring and
            ``0.5*k*(theta-theta_eq)^2`` for every torsion spring, or
            *None* when no ``Force`` dictionary was declared.
        """
        return (
            self.sym_forces.spring_potential_energy
            if self.sym_forces is not None else None
        )

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

    # ── Numeric params (delegates to vt, cached) ────────────────────────────

    def build_numeric_params(self, *, force: bool = False) -> NumericModelParams:
        """Return the cached ``NumericModelParams`` (built automatically at construction).

        Pass ``force=True`` to rebuild the geometry snapshot and regenerate
        ``B_func`` / ``Bdot_func`` (e.g. after reconstructing the system).

        Parameters
        ----------
        force : bool, optional
            If *True*, discard the cache and rebuild everything.

        Returns
        -------
        NumericModelParams
        """
        if force:
            self._numeric_params = self.vt.build_numeric_params()
            self._geom_extractor = self.vt.build_geometry_extractor(
                list(self.body_data_sym.values()), params=self._numeric_params,
            )
            self.B_func    = make_B_evaluator_mainint(
                self._numeric_params,
                self._slc_q_int, self._slc_qd_int,
                self._geom_extractor, self._slc_body_int,
            )
            self.Bdot_func = make_Bdot_evaluator_mainint(
                self._numeric_params,
                self._slc_q_int, self._slc_qd_int,
                self._geom_extractor, self._slc_body_int,
            )
        return self._numeric_params

    # ── Internal JAX backend (delegates to vt, accepts internal coords) ──────

    # ── Public runtime API ───────────────────────────────────────────────────

    def evaluate_B(self, mainNumVars):
        """Evaluate the velocity-transformation matrix B.

        Parameters
        ----------
        mainNumVars : array_like, shape ``(len(mainSymVars),)``
            User-facing variable vector ordered as
            ``[q_user, qd, body_params, force_params]``, matching
            ``self.mainSymVars``.

        Returns
        -------
        jnp.ndarray, shape ``(6*NBodies, total_dof)``
        """
        arr = self._validate_mainNumVars_shape(mainNumVars)
        return self.B_func(self._build_mainNumVars_int(arr))

    def evaluate_Bdot(self, mainNumVars):
        """Evaluate the time-derivative Bdot of the velocity-transformation matrix.

        Parameters
        ----------
        mainNumVars : array_like, shape ``(len(mainSymVars),)``
            User-facing variable vector ordered as
            ``[q_user, qd, body_params, force_params]``, matching
            ``self.mainSymVars``.

        Returns
        -------
        jnp.ndarray, shape ``(6*NBodies, total_dof)``
        """
        arr = self._validate_mainNumVars_shape(mainNumVars)
        return self.Bdot_func(self._build_mainNumVars_int(arr))

    def evaluate_points(self, mainNumVars) -> "PointsEvalResult":
        """Evaluate absolute positions and CG-relative arms for all declared points.

        Accepts the same user-facing ``mainNumVars`` vector as
        :meth:`evaluate_B` and :meth:`evaluate_Bdot`, converts it to the
        internal representation, and delegates to the compiled
        ``points_func`` backend.

        Parameters
        ----------
        mainNumVars : array_like, shape ``(len(mainSymVars),)``
            User-facing variable vector
            ``[q_user, qd, body_params, force_params, point_params]``.

        Returns
        -------
        PointsEvalResult
            Named tuple with four JAX arrays:

            * ``r_abs_body``   — ``(n_body_pts, 3)`` absolute positions of
              every user-declared body-attached point (body_id ASC,
              point_idx ASC within each body).
            * ``rho_abs_body`` — ``(n_body_pts, 3)`` CG-relative moment arms
              in the world frame.
            * ``r_abs_cg``     — ``(NBodies, 3)`` absolute CG positions, one
              row per body (1-based order).
            * ``r_abs_gr``     — ``(n_gr, 3)`` world-frame ground-point
              positions.

        Use ``self._points_spec.pt_body_slices[body_id]`` to extract the
        rows belonging to a specific body from ``r_abs_body`` /
        ``rho_abs_body``.

        Raises
        ------
        RuntimeError
            If no ``Initial_Points`` were declared (``points_func`` is None).
        """
        if self.points_func is None:
            raise RuntimeError(
                "evaluate_points() requires Initial_Points to be declared "
                "in the example module.  points_func is None."
            )
        arr = self._validate_mainNumVars_shape(mainNumVars)
        return self.points_func(self._build_mainNumVars_int(arr))

    def evaluate_forces(self, mainNumVars) -> "ForcesEvalResult":
        """Evaluate per-body 6-DOF wrenches and spring potential energy.

        Accepts the same user-facing ``mainNumVars`` vector as
        :meth:`evaluate_B`, converts it to the internal representation, and
        delegates to the compiled ``forces_func`` JAX backend.

        Parameters
        ----------
        mainNumVars : array_like, shape ``(len(mainSymVars),)``
            User-facing variable vector
            ``[q_user, qd, body_params, force_params, point_params]``.

        Returns
        -------
        ForcesEvalResult
            Named tuple with per-body ``(NBodies, 6)`` wrench arrays for each
            force category (``cg``, ``points_bd``, ``tension_spring``,
            ``tension_damper``, ``torsion_spring``, ``torsion_damper``,
            ``gravity``, ``total``) and a scalar
            ``spring_potential_energy``.

        Raises
        ------
        RuntimeError
            If no ``Force`` dictionary was declared (``forces_func`` is None).
        """
        if self.forces_func is None:
            raise RuntimeError(
                "evaluate_forces() requires a Force dictionary to be declared "
                "in the example module.  forces_func is None."
            )
        arr = self._validate_mainNumVars_shape(mainNumVars)
        return self.forces_func(self._build_mainNumVars_int(arr))

    def evaluate_mass_matrix(self, mainNumVars) -> "MassEvalResult":
        """Evaluate the generalised mass matrix M = B^T M_body B.

        Parameters
        ----------
        mainNumVars : array_like, shape ``(len(mainSymVars),)``
            User-facing variable vector
            ``[q_user, qd, body_params, force_params, point_params]``.

        Returns
        -------
        MassEvalResult
            Named tuple with one field:

            * ``M`` — ``jnp.ndarray`` of shape ``(total_dof, total_dof)``,
              the symmetric positive-definite generalised mass matrix.

        Raises
        ------
        RuntimeError
            If no ``body_inertia`` dictionary was declared (``mass_func`` is
            None).
        """
        if self.mass_func is None:
            raise RuntimeError(
                "evaluate_mass_matrix() requires body_inertia to be declared. "
                "mass_func is None."
            )
        arr = self._validate_mainNumVars_shape(mainNumVars)
        return self.mass_func(self._build_mainNumVars_int(arr))

    def evaluate_eom_kernel(self, mainNumVars) -> "EomKernelResult":
        """Evaluate B, Bdot, M_body, and M in a single kinematics pass.

        Preferred over calling :meth:`evaluate_B`, :meth:`evaluate_Bdot`,
        and :meth:`evaluate_mass_matrix` separately in integrator loops,
        since all four quantities share one ``build_cache_jax`` call.

        Parameters
        ----------
        mainNumVars : array_like, shape ``(len(mainSymVars),)``

        Returns
        -------
        EomKernelResult
            Named tuple with fields ``B``, ``Bdot``, ``M_body``, ``M``.

        Raises
        ------
        RuntimeError
            If no ``body_inertia`` was declared (``eom_func`` is None).
        """
        if self.eom_func is None:
            raise RuntimeError(
                "evaluate_eom_kernel() requires body_inertia to be declared. "
                "eom_func is None."
            )
        arr = self._validate_mainNumVars_shape(mainNumVars)
        return self.eom_func(self._build_mainNumVars_int(arr))

    def evaluate_generalized_forces(self, mainNumVars):
        """Evaluate the generalised force vector Q_gen = B^T f_wrench.

        Combines :meth:`evaluate_B` and :meth:`evaluate_total_wrench`
        into the generalised force vector used on the right-hand side of
        the equations of motion::

            M(q) q̈ = Q_gen(q, q̇)

        Parameters
        ----------
        mainNumVars : array_like, shape ``(len(mainSymVars),)``
            User-facing variable vector
            ``[q_user, qd, body_params, force_params, point_params]``.

        Returns
        -------
        jnp.ndarray, shape ``(total_dof,)``
            Generalised force vector.

        Raises
        ------
        RuntimeError
            If no ``Force`` dictionary was declared (``forces_func`` is None).
        """
        if self.forces_func is None:
            raise RuntimeError(
                "evaluate_generalized_forces() requires a Force dictionary to "
                "be declared.  forces_func is None."
            )
        arr     = self._validate_mainNumVars_shape(mainNumVars)
        mnv_int = self._build_mainNumVars_int(arr)
        B       = self.B_func(mnv_int)              # (6*NBodies, total_dof)
        f_total = self.forces_func(mnv_int).total   # (NBodies, 6)
        return B.T @ f_total.ravel()                # (total_dof,)

    def evaluate_total_wrench(self, mainNumVars):
        """Evaluate the total (summed) per-body wrench array.

        Convenience wrapper around :meth:`evaluate_forces` that returns only
        the ``total`` field — the ``(NBodies, 6)`` JAX array of
        world-frame ``[Fx, Fy, Fz, Mx, My, Mz]`` contributions from **all**
        active force categories combined.  Intended for downstream RHS
        construction where only the net wrench is needed.

        Parameters
        ----------
        mainNumVars : array_like, shape ``(len(mainSymVars),)``
            User-facing variable vector
            ``[q_user, qd, body_params, force_params, point_params]``.

        Returns
        -------
        jnp.ndarray, shape ``(NBodies, 6)``
            Net world-frame wrench on each body (force then moment about CG).

        Raises
        ------
        RuntimeError
            If no ``Force`` dictionary was declared (``forces_func`` is None).
        """
        return self.evaluate_forces(mainNumVars).total

    def sym_force_reduction_at_point(self, rec, force_vec):
        """Reduce a symbolic force at a body point to force + moment about CG.

        Thin façade over :func:`points_3d.sym_force_reduction_at_point`;
        see that function for full documentation.

        Parameters
        ----------
        rec : PointRecord3D
            A body-attached point record from ``self.sym_points``.
        force_vec : sym.Matrix, shape ``(3, 1)`` or ``(3,)``
            Symbolic applied force vector in the world frame.

        Returns
        -------
        f_eq : sym.Matrix, shape ``(3, 1)``
        m_eq : sym.Matrix, shape ``(3, 1)``
            Moment about body CG: ``rho_abs × force_vec`` (unevaluated).
        """
        from .points_3d import sym_force_reduction_at_point as _sym_fr  # noqa: PLC0415
        return _sym_fr(rec, force_vec)

    # ── Ergonomic helpers ────────────────────────────────────────────────────

    def integrate(
        self,
        mainNumVars,
        *,
        tspan,
        dt=None,
        rtol: float = 1e-6,
        atol: float = 1e-6,
        algorithm: str = "Dopri5",
        max_steps: int = 500_000,
    ):
        """Numerically integrate the equations of motion.

        Delegates to :func:`integrator_3d.integrate_3d`.  The system must
        have ``body_inertia`` declared so that ``eom_func`` is compiled.

        Parameters
        ----------
        mainNumVars : array_like, shape ``(len(mainSymVars),)``
            Initial user-facing variable vector
            ``[q_user, qd, body_params, force_params, point_params]``.
        tspan : float or (float, float)
            End time (start = 0) or ``(t_start, t_end)``.
        dt : float or None
            Output time step.  *None* uses the solver's adaptive grid.
        rtol, atol : float
            ODE solver tolerances.
        algorithm : str
            Diffrax solver name (``"Dopri5"``, ``"Dopri8"``, ``"Tsit5"``, …).
        max_steps : int
            Maximum number of internal solver steps before raising an error.

        Returns
        -------
        diffrax.Solution
            ``sol.ts`` — time vector, ``sol.ys`` — state history
            (shape ``(n_steps, total_cfg_dof + total_dof)``).
            ``sol.result == 0`` indicates success.
        """
        from .integrator_3d import integrate_3d  # noqa: PLC0415
        return integrate_3d(
            self, mainNumVars,
            tspan=tspan, dt=dt,
            rtol=rtol, atol=atol,
            algorithm=algorithm,
            max_steps=max_steps,
        )

    def summary_table(self, precision: int = 3):
        """Print a summary table of joint information (delegates to joint_system)."""
        return self.joint_system.summary_table(precision=precision)

    def __repr__(self) -> str:
        force_info = (
            f", forces={len(self.forces_def.cg_forces + self.forces_def.point_forces + self.forces_def.tension_springs + self.forces_def.tension_dampers + self.forces_def.torsion_springs + self.forces_def.torsion_dampers)}el"
            if self.forces_def is not None else ""
        )
        mass_info = (
            f", mass=True"
            if self.mass_func is not None else ""
        )
        return (
            f"MbdSystem3D(NBodies={self.NBodies}, NJoints={self.NJoints}, "
            f"total_dof={self.total_dof}, total_cfg_dof={self.total_cfg_dof}, "
            f"total_user_dof={self.total_user_dof}{force_info}{mass_info})"
        )
