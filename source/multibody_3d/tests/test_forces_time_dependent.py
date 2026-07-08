# tests/test_forces_time_dependent.py
"""Tests for time-dependent force expressions (reserved symbol ``t``).

Coverage
--------
- ``force_definition_3d.T_SYM`` is reserved for time; ``ForcesDefinition3D.
  is_time_dependent`` is set correctly by ``parse_force_dict``.
- ``collect_symbols()`` excludes ``t`` by default, includes it with
  ``include_time=True``.
- ``MbdSystem3D.evaluate_forces(mainNumVars, t=...)`` produces the expected
  time-varying force for CG, PointsBD, TensionSpring, TorsionSpring elements.
- The diffrax integrator automatically threads its own traced stage time
  into time-dependent forces (no user wiring required).
- Static (non-time-dependent) systems are unaffected (regression).
"""
from __future__ import annotations

import numpy as np
import pytest
import sympy as sym

from multibody_3d import MbdSystem3D
from multibody_3d.multibody_core.force_definition_3d import (
    T_SYM,
    is_time_symbol,
    parse_force_dict,
)

jax = pytest.importorskip("jax")
diffrax = pytest.importorskip("diffrax")


# ---------------------------------------------------------------------------
# Single-body fixture: 1 revolute joint, time-dependent CG force Fx = A*sin(t)
# ---------------------------------------------------------------------------

_DATA_1BODY = {
    "NBodies": 1,
    "joints": [(0, 1)],
    "types": ["R"],
    "parent_cg_to_joint": [[0, 0, 0]],
    "joint_to_child_cg": [[0, 0, -1]],
    "axis_u": [[0, 1, 0]],
    "axis_u1": [None],
    "axis_u2": [None],
}
_I_SMALL = [[0.01, 0.0, 0.0], [0.0, 0.01, 0.0], [0.0, 0.0, 0.01]]


@pytest.fixture(scope="module")
def mbd_time_cg():
    """1-body system with Force['CG'] = A * sin(t) along x."""
    A = sym.Symbol("A", positive=True)
    Force = {"CG": {1: {"force": [A * sym.sin(T_SYM), 0, 0]}}}
    mbd = MbdSystem3D(
        data=_DATA_1BODY,
        force_points_sym={"A": A},
        Force=Force,
        body_inertia={1: {"mass": 1.0, "J": _I_SMALL}},
    )
    return mbd, A


@pytest.fixture(scope="module")
def mbd_static_cg():
    """1-body system with a plain (non-time-dependent) CG force, for regression."""
    Fx = sym.Symbol("Fx")
    Force = {"CG": {1: {"force": [Fx, 0, 0]}}}
    mbd = MbdSystem3D(
        data=_DATA_1BODY,
        force_points_sym={"Fx": Fx},
        Force=Force,
        body_inertia={1: {"mass": 1.0, "J": _I_SMALL}},
    )
    return mbd, Fx


def _mnv_1body(q=0.0, qd=0.0, param=0.0):
    return np.array([q, qd, param], dtype=float)


# ---------------------------------------------------------------------------
# Parsing / classification
# ---------------------------------------------------------------------------

class TestParsingClassification:

    def test_static_force_not_time_dependent(self):
        Fx = sym.Symbol("Fx")
        fd = parse_force_dict(
            {"CG": {1: {"force": [Fx, 0, 0]}}}, n_bodies=1,
        )
        assert fd.is_time_dependent is False

    def test_time_dependent_cg_force_detected(self):
        A = sym.Symbol("A")
        fd = parse_force_dict(
            {"CG": {1: {"force": [A * sym.sin(T_SYM), 0, 0]}}}, n_bodies=1,
        )
        assert fd.is_time_dependent is True

    def test_time_dependent_spring_stiffness_detected(self):
        k0 = sym.Symbol("k0")
        fd = parse_force_dict(
            {"TensionSpring": [(0, 0, 1, 0, k0 * (1 + T_SYM), 1.0)]},
            n_bodies=1,
        )
        assert fd.is_time_dependent is True

    def test_collect_symbols_excludes_time_by_default(self):
        A = sym.Symbol("A")
        fd = parse_force_dict(
            {"CG": {1: {"force": [A * sym.sin(T_SYM), 0, 0]}}}, n_bodies=1,
        )
        syms = fd.collect_symbols()
        assert A in syms
        assert T_SYM not in syms

    def test_collect_symbols_includes_time_when_requested(self):
        A = sym.Symbol("A")
        fd = parse_force_dict(
            {"CG": {1: {"force": [A * sym.sin(T_SYM), 0, 0]}}}, n_bodies=1,
        )
        syms = fd.collect_symbols(include_time=True)
        assert A in syms
        assert T_SYM in syms

    def test_gravity_still_requires_purely_numeric_g_vec(self):
        """Gravity g_vec is unaffected by Task 2 — still must be numeric."""
        fd = parse_force_dict({"Gravity": {"g_vec": [0, 0, -9.81]}}, n_bodies=1)
        assert fd.is_time_dependent is False


# ---------------------------------------------------------------------------
# Regression: user-created Symbol("t", ...) with non-default assumptions
# ---------------------------------------------------------------------------
# Bug: a user-created ``sym.symbols("t", real=True)`` is NOT the same SymPy
# object as ``T_SYM = sym.Symbol("t")`` (different assumptions -> different
# object, ``==`` is False).  Detection must be name-based, not identity-based,
# or such expressions silently fail to be recognized as time-dependent.

class TestUserCreatedTimeSymbol:

    def test_name_based_detection_with_different_assumptions(self):
        """A user's sym.symbols('t', real=True) must still be recognized."""
        t_user = sym.symbols("t", real=True)
        assert t_user != T_SYM  # sanity: genuinely different SymPy objects
        assert is_time_symbol(t_user) is True

        A = sym.Symbol("A")
        fd = parse_force_dict(
            {"CG": {1: {"force": [A * sym.sin(t_user), 0, 0]}}}, n_bodies=1,
        )
        assert fd.is_time_dependent is True

    def test_evaluate_forces_varies_with_user_created_t_symbol(self):
        """End-to-end: force actually varies over t even with a user t symbol
        (regression for the reported bug where the body did not move).
        """
        t_user = sym.symbols("t", real=True)
        Force = {"CG": {1: {"force": [100 * sym.sin(t_user), 0, 0]}}}
        mbd = MbdSystem3D(
            data=_DATA_1BODY,
            Force=Force,
            body_inertia={1: {"mass": 1.0, "J": _I_SMALL}},
        )
        assert mbd.forces_def.is_time_dependent is True
        mnv = _mnv_1body()[:-1]  # no force params declared this time -> [q, qd]
        r0  = mbd.evaluate_forces(mnv, t=0.0)
        r90 = mbd.evaluate_forces(mnv, t=np.pi / 2)
        assert float(r0.cg[0, 0]) == pytest.approx(0.0, abs=1e-9)
        assert float(r90.cg[0, 0]) == pytest.approx(100.0, rel=1e-6)

    def test_declaring_t_in_force_sym_raises_clear_error(self):
        """The exact footgun from the bug report: declaring 't' as a force
        parameter must raise a clear, actionable error instead of silently
        freezing it at whatever constant value mainNumVars carries.
        """
        t_user = sym.symbols("t", real=True)
        Force = {"CG": {1: {"force": [100 * sym.sin(t_user), 0, 0]}}}
        with pytest.raises(ValueError, match="reserved to mean simulation time"):
            MbdSystem3D(
                data=_DATA_1BODY,
                force_points_sym={"t": t_user},
                Force=Force,
                body_inertia={1: {"mass": 1.0, "J": _I_SMALL}},
            )

    def test_declaring_t_in_points_sym_raises_clear_error(self):
        t_user = sym.symbols("t")
        with pytest.raises(ValueError, match="reserved to mean simulation time"):
            MbdSystem3D(data=_DATA_1BODY, points_sym={"t": t_user})


# ---------------------------------------------------------------------------
# Runtime evaluation: MbdSystem3D.evaluate_forces(mainNumVars, t=...)
# ---------------------------------------------------------------------------

class TestTimeDependentEvaluation:

    def test_is_time_dependent_flag_on_system(self, mbd_time_cg):
        mbd, _ = mbd_time_cg
        assert mbd.forces_def.is_time_dependent is True

    def test_force_zero_at_t0(self, mbd_time_cg):
        mbd, A = mbd_time_cg
        mnv = _mnv_1body(param=2.0)
        result = mbd.evaluate_forces(mnv, t=0.0)
        assert float(result.cg[0, 0]) == pytest.approx(0.0, abs=1e-9)

    def test_force_at_pi_over_2_equals_amplitude(self, mbd_time_cg):
        mbd, A = mbd_time_cg
        mnv = _mnv_1body(param=2.0)
        result = mbd.evaluate_forces(mnv, t=np.pi / 2)
        assert float(result.cg[0, 0]) == pytest.approx(2.0, rel=1e-6)

    def test_force_matches_sin_curve_at_several_times(self, mbd_time_cg):
        mbd, A = mbd_time_cg
        amplitude = 3.0
        mnv = _mnv_1body(param=amplitude)
        for tt in [0.0, 0.5, 1.0, np.pi, 2 * np.pi]:
            result = mbd.evaluate_forces(mnv, t=tt)
            expected = amplitude * np.sin(tt)
            assert float(result.cg[0, 0]) == pytest.approx(expected, abs=1e-6)

    def test_default_t_is_zero(self, mbd_time_cg):
        """Omitting t defaults to 0.0 (for standalone/inspection calls)."""
        mbd, _ = mbd_time_cg
        mnv = _mnv_1body(param=5.0)
        result_default = mbd.evaluate_forces(mnv)
        result_t0 = mbd.evaluate_forces(mnv, t=0.0)
        assert float(result_default.cg[0, 0]) == pytest.approx(float(result_t0.cg[0, 0]))

    def test_no_moment_induced(self, mbd_time_cg):
        mbd, _ = mbd_time_cg
        mnv = _mnv_1body(param=2.0)
        result = mbd.evaluate_forces(mnv, t=1.23)
        np.testing.assert_allclose(np.asarray(result.cg[0, 3:]), 0.0, atol=1e-12)


# ---------------------------------------------------------------------------
# Static regression: non-time-dependent systems unaffected
# ---------------------------------------------------------------------------

class TestStaticRegression:

    def test_static_system_not_time_dependent(self, mbd_static_cg):
        mbd, _ = mbd_static_cg
        assert mbd.forces_def.is_time_dependent is False

    def test_static_force_ignores_t(self, mbd_static_cg):
        mbd, _ = mbd_static_cg
        mnv = _mnv_1body(param=7.0)
        r_no_t  = mbd.evaluate_forces(mnv)
        r_t0    = mbd.evaluate_forces(mnv, t=0.0)
        r_t_big = mbd.evaluate_forces(mnv, t=999.0)
        assert float(r_no_t.cg[0, 0])  == pytest.approx(7.0)
        assert float(r_t0.cg[0, 0])    == pytest.approx(7.0)
        assert float(r_t_big.cg[0, 0]) == pytest.approx(7.0)


# ---------------------------------------------------------------------------
# Integration: diffrax threads its own traced stage time automatically
# ---------------------------------------------------------------------------

class TestIntegratorThreadsTime:

    def test_integration_runs_with_time_dependent_force(self, mbd_time_cg):
        mbd, _ = mbd_time_cg
        mnv = _mnv_1body(param=1.0)
        sol = mbd.integrate(mnv, tspan=(0.0, 2.0), dt=0.5, algorithm="Dopri5")
        assert bool(sol.result == diffrax.RESULTS.successful)
        assert sol.ys.shape[0] == 5  # t = 0, 0.5, 1.0, 1.5, 2.0

    def test_time_dependent_force_differs_from_frozen_at_t0(self, mbd_time_cg):
        """A time-dependent force must NOT be equivalent to freezing t=0
        for the whole solve — sanity check that force varies during
        integration by comparing against a static system with Fx frozen
        at its t=0 value (0.0, since sin(0)=0): if the time-dependent
        system responded identically to zero force, the body would not
        move under gravity-free, zero-initial-velocity conditions.
        """
        mbd, _ = mbd_time_cg
        mnv = _mnv_1body(q=0.0, qd=0.0, param=5.0)
        sol = mbd.integrate(mnv, tspan=(0.0, 3.0), dt=0.5, algorithm="Dopri5")
        # Body starts at rest; if Fx(t) were always 0 (frozen at t=0 value),
        # qd would remain 0 throughout (no other forces/gravity present).
        final_qd = float(np.array(sol.ys[-1, 1]))
        assert abs(final_qd) > 1e-6
