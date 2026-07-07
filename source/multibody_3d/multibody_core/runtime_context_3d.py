"""runtime_context_3d.py

Shared frozen-parameter context for JAX postprocessing of an integrated
solution (energy, kinematics, ...).

Body / force / point parameters and geometry are constant for the whole
duration of one :meth:`MbdSystem3D.integrate` call (they only change across
separate integrate calls, e.g. in a design-exploration or optimization
loop).  Rather than have every postprocessing consumer independently
extract the constant parameter blocks, freeze ``eom_func`` / ``forces_func``,
and rebuild the ``build_cache_jax`` / ``build_rate_cache_jax`` keyword
dictionaries, this module builds that frozen context once per
integrate/postprocess cycle and hands it to consumers such as
``compute_energy_3d`` (mass_runtime_3d.py) and ``compute_kinematics_3d``
(kinematics_runtime_3d.py).

Not part of the ODE right-hand side; see ``integrator_3d.py`` for that.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, TYPE_CHECKING

import numpy as np

try:
    import jax.numpy as jnp
    _JAX_AVAILABLE = True
except ImportError:  # pragma: no cover
    _JAX_AVAILABLE = False

if not _JAX_AVAILABLE:
    raise ImportError("JAX is required for runtime_context_3d.")

if TYPE_CHECKING:
    from .mbd_system_3d import MbdSystem3D


# ---------------------------------------------------------------------------
# Keyword filters for build_cache_jax / build_rate_cache_jax
# ---------------------------------------------------------------------------

_POS_CACHE_KEYS = frozenset({
    "n_bodies", "n_joints", "parent", "child", "codes",
    "cfg_slices", "p2j", "j2c", "u", "u1", "u2",
})
_RATE_TOPO_KEYS = frozenset({
    "n_bodies", "n_joints", "parent", "child", "codes", "col_slices",
})


# ---------------------------------------------------------------------------
# Context container
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class RuntimeContext:
    """Frozen parameter / geometry context shared by postprocessing evaluators.

    Attributes
    ----------
    n_qi : int
        Length of the internal configuration vector ``q_int`` — used to
        split a diffrax state ``y = [q_int, qd]``.
    NB : int
        Number of bodies.
    bp_np, fp_np, pp_np : np.ndarray
        Constant body / force / point parameter blocks (numpy, 1-D).
    cb, cf, cp : jnp.ndarray
        Same blocks as JAX arrays, ready to concatenate into a
        ``mainNumVars_int`` vector via :meth:`build_mainint`.
    eom_e : callable or None
        Frozen EOM evaluator ``f(mainint) -> EomKernelResult``, or *None*
        when ``body_inertia`` was not declared on the system.
    forces_e : callable or None
        Frozen forces evaluator ``f(mainint, t=0.0) -> ForcesEvalResult``,
        or *None* when no ``Force`` dictionary was declared.
    pos_cache_kwargs : dict
        Keyword arguments for
        :func:`velocity_transformation_3d.build_cache_jax`.
    rate_topo_kwargs : dict
        Topology-only keyword arguments for
        :func:`velocity_transformation_3d.build_rate_cache_jax`
        (``A_abs``/``r_abs``/``rJ``/``U`` are supplied per-call from the
        position cache, not stored here).
    is_time_dependent : bool
        Whether the declared forces reference the reserved time symbol.
        Always *False* until time-dependent forces are implemented.
    """
    n_qi: int
    NB:   int
    bp_np: "np.ndarray"
    fp_np: "np.ndarray"
    pp_np: "np.ndarray"
    cb: "jnp.ndarray"
    cf: "jnp.ndarray"
    cp: "jnp.ndarray"
    eom_e:    Optional[Any]
    forces_e: Optional[Any]
    pos_cache_kwargs: Dict[str, Any]
    rate_topo_kwargs: Dict[str, Any]
    is_time_dependent: bool = False

    def build_mainint(self, y: "jnp.ndarray") -> "jnp.ndarray":
        """Concatenate a diffrax state ``y = [q_int, qd]`` with the frozen
        body/force/point parameter blocks into a full ``mainNumVars_int``.
        """
        q_int = y[:self.n_qi]
        qd    = y[self.n_qi:]
        return jnp.concatenate([q_int, qd, self.cb, self.cf, self.cp])


# ---------------------------------------------------------------------------
# Builder
# ---------------------------------------------------------------------------

def build_runtime_context(mbd: "MbdSystem3D", mainNumVars) -> RuntimeContext:
    """Freeze body/force/point parameters and geometry for one integrate/postprocess cycle.

    Parameters
    ----------
    mbd : MbdSystem3D
    mainNumVars : array_like, shape ``(len(mainSymVars),)``
        The same vector passed to :meth:`MbdSystem3D.integrate`; supplies
        the constant body/force/point parameter blocks (assumed unchanged
        for the whole solve).

    Returns
    -------
    RuntimeContext
    """
    # ── Deferred imports (mirrors the pattern used elsewhere in *_runtime_3d.py) ─
    try:
        from .velocity_transformation_3d import (   # noqa: PLC0415
            _convert_geometry_to_jax,
            _convert_topology_to_jax,
            _np_geom_to_jax,
        )
    except Exception:  # pragma: no cover
        from velocity_transformation_3d import (
            _convert_geometry_to_jax,
            _convert_topology_to_jax,
            _np_geom_to_jax,
        )

    arr   = mbd._validate_mainNumVars_shape(mainNumVars)
    mint0 = mbd._build_mainNumVars_int(arr)
    bp_np = np.array(mint0[mbd._slc_body_int],   dtype=float)
    fp_np = np.array(mint0[mbd._slc_force_int],  dtype=float)
    pp_np = np.array(mint0[mbd._slc_points_int], dtype=float)

    eom_e = None
    if mbd.eom_func is not None:
        eom_e = (
            mbd.eom_func.freeze(bp_np)
            if hasattr(mbd.eom_func, "freeze")
            else mbd.eom_func
        )

    forces_e = None
    if mbd.forces_func is not None:
        forces_e = (
            mbd.forces_func.freeze(bp_np, fp_np, pp_np)
            if hasattr(mbd.forces_func, "freeze")
            else mbd.forces_func
        )

    if mbd._geom_extractor.has_dynamic:
        p2j_e, j2c_e, u_e, u1_e, u2_e = _np_geom_to_jax(
            *mbd._geom_extractor.evaluate(bp_np)
        )
        topo = _convert_topology_to_jax(mbd._numeric_params)
        pos_cache_kwargs = {k: v for k, v in topo.items() if k in _POS_CACHE_KEYS}
        pos_cache_kwargs.update(p2j=p2j_e, j2c=j2c_e, u=u_e, u1=u1_e, u2=u2_e)
        rate_topo_kwargs = {k: v for k, v in topo.items() if k in _RATE_TOPO_KEYS}
    else:
        geom = _convert_geometry_to_jax(mbd._numeric_params)
        pos_cache_kwargs = {k: v for k, v in geom.items() if k in _POS_CACHE_KEYS}
        rate_topo_kwargs = {k: v for k, v in geom.items() if k in _RATE_TOPO_KEYS}

    is_time_dependent = bool(
        mbd.forces_def is not None
        and getattr(mbd.forces_def, "is_time_dependent", False)
    )

    return RuntimeContext(
        n_qi=mbd.total_cfg_dof,
        NB=mbd.NBodies,
        bp_np=bp_np, fp_np=fp_np, pp_np=pp_np,
        cb=jnp.asarray(bp_np, dtype=jnp.float64),
        cf=jnp.asarray(fp_np, dtype=jnp.float64),
        cp=jnp.asarray(pp_np, dtype=jnp.float64),
        eom_e=eom_e,
        forces_e=forces_e,
        pos_cache_kwargs=pos_cache_kwargs,
        rate_topo_kwargs=rate_topo_kwargs,
        is_time_dependent=is_time_dependent,
    )
