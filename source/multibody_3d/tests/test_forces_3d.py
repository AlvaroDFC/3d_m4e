# tests/test_forces_3d.py
"""Focused tests for the 3D force architecture in MbdSystem3D.

Coverage
--------
Goal 3  — force-related symbols are in canonical variable vectors in user order.
Goal 4  — symbolic force expressions remain opaque (not expanded).
Goal 5  — JAX force evaluators are created automatically in __post_init__.
Goal 6  — public numeric force evaluation works from mainNumVars.
Goal 7  — CG, PointsBD, TensionSpring, TorsionSpring, Gravity, total.
Goal 8  — changing numeric values of symbolic coefficients changes results.
"""
from __future__ import annotations

import math
from types import SimpleNamespace

import numpy as np
import pytest
import sympy as sym

from multibody_3d import MbdSystem3D, ForcesEvalResult, SymbolicForcesCache3D

jax = pytest.importorskip("jax")
import jax.numpy as jnp  # noqa: E402


# ---------------------------------------------------------------------------
# Shared fixture: 2-body revolute chain with all 5 force categories
# ---------------------------------------------------------------------------
#
# Topology
# --------
#   Ground (0) → Body 1 (revolute, axis z)
#   Body 1     → Body 2 (revolute, axis z)
#
# Geometry (all numeric → constant-geometry extractor branch)
#   p2j[0] = [1, 0, 0],  j2c[0] = [-1, 0, 0]   body-1 CG at origin
#   p2j[1] = [2, 0, 0],  j2c[1] = [-1, 0, 0]   body-2 CG at [1,0,0] rest pos
#
# Points (declared in Initial_Points)
#   GR[0]  = [0, 0, 0]                            world origin anchor
#   BD[1][0] = [a, 0, 0]  (symbolic; spring attachment on body 1)
#   BD[2][0] = [b, 0, 0]  (symbolic; spring/point-force attachment on body 2)
#
# Forces
#   CG           body 1 ← [Fx, 0, 0] force + [0, 0, Mz] moment
#   PointsBD     body 2, pt 0 ← [0, Fy, 0] force  (induces CG moment)
#   TorsionSpring joint 0 (body 1 child), stiffness k_t, eq = 0
#   Gravity      body 1 mass m1, body 2 mass m2, g = [0, -9.81, 0]
#
# Note: TensionSpring between GR[0] and BD[2][0] is tested in a separate
# fixture to keep this one minimal.
#
# Symbolic params
#   body_data_sym : {}  (no symbolic geometry in this fixture)
#   force_sym     : {Fx, Fy, Mz, k_t, m1, m2}
#   points_sym    : {a, b}

_Z3 = [0.0, 0.0, 0.0]


@pytest.fixture(scope="module")
def mbd_force_rr():
    """2-body RR chain with CG, PointsBD, TorsionSpring, and Gravity forces."""
    Fx, Fy, Mz = sym.symbols("Fx Fy Mz", real=True)
    k_t         = sym.Symbol("k_t", positive=True)
    m1, m2      = sym.symbols("m1 m2", positive=True)
    a, b        = sym.symbols("a b", real=True)

    data = {
        "NBodies": 2,
        "joints": [(0, 1), (1, 2)],
        "types": ["R", "R"],
        "parent_cg_to_joint": [[1.0, 0.0, 0.0], [2.0, 0.0, 0.0]],
        "joint_to_child_cg":  [[-1.0, 0.0, 0.0], [-1.0, 0.0, 0.0]],
        "axis_u":  [[0, 0, 1], [0, 0, 1]],
        "axis_u1": [None, None],
        "axis_u2": [None, None],
    }
    force_sym = {"Fx": Fx, "Fy": Fy, "Mz": Mz, "k_t": k_t, "m1": m1, "m2": m2}
    points_sym = {"a": a, "b": b}
    Initial_Points = {
        "GR": [[0.0, 0.0, 0.0]],
        "BD": {
            1: [[a, 0.0, 0.0]],
            2: [[b, 0.0, 0.0]],
        },
    }
    Force = {
        "CG": {1: {"force": [Fx, 0, 0], "moment": [0, 0, Mz]}},
        "PointsBD": [(2, 0, [0, Fy, 0])],
        "TorsionSpring": [(0, k_t, 0.0)],
        "Gravity": {"g_vec": [0.0, -9.81, 0.0], "mass": {1: m1, 2: m2}},
    }
    return MbdSystem3D(
        data=data,
        force_points_sym=force_sym,
        points_sym=points_sym,
        Initial_Points=Initial_Points,
        Force=Force,
    ), dict(Fx=Fx, Fy=Fy, Mz=Mz, k_t=k_t, m1=m1, m2=m2, a=a, b=b)


def _mnv_rr(mbd, q1, q2, qd1, qd2,
            Fx=0.0, Fy=0.0, Mz=0.0, k_t=1.0, m1=1.0, m2=1.0,
            a=1.0, b=1.0):
    """Assemble mainNumVars for the 2-body RR fixture (theta, theta, dtheta, dtheta, …)."""
    q_user = [q1, q2]
    qd     = [qd1, qd2]
    fp     = [Fx, Fy, Mz, k_t, m1, m2]   # matches force_sym insertion order
    pp     = [a, b]                         # matches points_sym insertion order
    return np.array(q_user + qd + fp + pp, dtype=float)


# ---------------------------------------------------------------------------
# Goal 3 — force symbols in canonical variable vectors (user order preserved)
# ---------------------------------------------------------------------------

class TestForceSymbolsInVectors:

    def test_force_syms_in_mainSymVars(self, mbd_force_rr):
        mbd, syms = mbd_force_rr
        for s in syms.values():
            assert s in list(mbd.mainSymVars), f"{s} missing from mainSymVars"

    def test_force_syms_in_mainSymVars_int(self, mbd_force_rr):
        mbd, syms = mbd_force_rr
        for s in syms.values():
            assert s in list(mbd.mainSymVars_int), f"{s} missing from mainSymVars_int"

    def test_force_params_order_preserved(self, mbd_force_rr):
        """force_sym insertion order must match the _slc_force slice."""
        mbd, syms = mbd_force_rr
        slc = mbd._slc_force
        syms_at_slice = list(mbd.mainSymVars)[slc]
        expected = [syms["Fx"], syms["Fy"], syms["Mz"],
                    syms["k_t"], syms["m1"], syms["m2"]]
        assert syms_at_slice == expected

    def test_point_params_order_preserved(self, mbd_force_rr):
        """points_sym insertion order must match the _slc_points slice."""
        mbd, syms = mbd_force_rr
        slc = mbd._slc_points
        syms_at_slice = list(mbd.mainSymVars)[slc]
        assert syms_at_slice == [syms["a"], syms["b"]]

    def test_mainSymVars_length(self, mbd_force_rr):
        mbd, _ = mbd_force_rr
        expected = (mbd.total_user_dof   # q1, q2
                    + mbd.total_dof       # qd1, qd2
                    + 6                   # Fx, Fy, Mz, k_t, m1, m2
                    + 2)                  # a, b
        assert len(mbd.mainSymVars) == expected


# ---------------------------------------------------------------------------
# Goal 4 — symbolic force expressions remain opaque
# ---------------------------------------------------------------------------

class TestSymbolicOpacity:

    def test_sym_forces_not_none(self, mbd_force_rr):
        mbd, _ = mbd_force_rr
        assert mbd.sym_forces is not None
        assert isinstance(mbd.sym_forces, SymbolicForcesCache3D)

    def test_sym_total_wrench_property(self, mbd_force_rr):
        mbd, _ = mbd_force_rr
        tw = mbd.sym_total_wrench
        assert tw is not None
        assert len(tw) == mbd.NBodies
        for w in tw:
            assert w.shape == (6, 1)

    def test_cg_wrench_contains_Fx_symbol(self, mbd_force_rr):
        mbd, syms = mbd_force_rr
        Fx = syms["Fx"]
        cg_cat = mbd.sym_forces.wrench_by_category["CG"]
        # Body 1 CG wrench [Fx, 0, 0, 0, 0, Mz]
        assert Fx in cg_cat[0][0, 0].free_symbols

    def test_cg_wrench_contains_Mz_symbol(self, mbd_force_rr):
        mbd, syms = mbd_force_rr
        Mz = syms["Mz"]
        cg_cat = mbd.sym_forces.wrench_by_category["CG"]
        assert Mz in cg_cat[0][5, 0].free_symbols

    def test_torsion_spring_wrench_contains_k_t(self, mbd_force_rr):
        mbd, syms = mbd_force_rr
        k_t = syms["k_t"]
        ts_cat = mbd.sym_forces.wrench_by_category["TorsionSpring"]
        # Body 1 (child of joint 0) gets the torsion moment
        w1 = ts_cat[0]
        all_syms = w1.free_symbols
        assert k_t in all_syms, "k_t must appear in body-1 torsion-spring wrench"

    def test_sym_spring_pe_contains_k_t(self, mbd_force_rr):
        mbd, syms = mbd_force_rr
        pe = mbd.sym_spring_pe
        assert pe is not None
        assert syms["k_t"] in pe.free_symbols

    def test_gravity_wrench_contains_mass_symbols(self, mbd_force_rr):
        mbd, syms = mbd_force_rr
        grav_cat = mbd.sym_forces.wrench_by_category["Gravity"]
        assert syms["m1"] in grav_cat[0].free_symbols
        assert syms["m2"] in grav_cat[1].free_symbols

    def test_points_bd_wrench_contains_Fy(self, mbd_force_rr):
        mbd, syms = mbd_force_rr
        Fy = syms["Fy"]
        pbd_cat = mbd.sym_forces.wrench_by_category["PointsBD"]
        assert Fy in pbd_cat[1].free_symbols   # body 2 (index 1)

    def test_no_expand_in_cg_wrench(self, mbd_force_rr):
        """Symbolic CG wrench at body 1 must not have been trigsimp/expanded."""
        mbd, syms = mbd_force_rr
        Fx = syms["Fx"]
        cg_w = mbd.sym_forces.wrench_by_category["CG"][0]
        # The expression for Fx-component should literally BE Fx, not some expanded form
        expr = cg_w[0, 0]
        assert expr == Fx, f"Expected raw symbol Fx, got: {expr}"


# ---------------------------------------------------------------------------
# Goal 5 — JAX evaluators created automatically in __post_init__
# ---------------------------------------------------------------------------

class TestEvaluatorSetup:

    def test_forces_func_built(self, mbd_force_rr):
        mbd, _ = mbd_force_rr
        assert mbd.forces_func is not None
        assert callable(mbd.forces_func)

    def test_forces_def_built(self, mbd_force_rr):
        from multibody_3d import ForcesDefinition3D
        mbd, _ = mbd_force_rr
        assert mbd.forces_def is not None
        assert isinstance(mbd.forces_def, ForcesDefinition3D)

    def test_forces_func_none_without_Force_dict(self):
        """A system with no Force dict must have forces_func = None."""
        data = {
            "NBodies": 1, "joints": [(0, 1)], "types": ["R"],
            "parent_cg_to_joint": [_Z3], "joint_to_child_cg": [_Z3],
            "axis_u": [[0, 0, 1]], "axis_u1": [None], "axis_u2": [None],
        }
        mbd = MbdSystem3D(data=data)
        assert mbd.forces_func is None
        assert mbd.forces_def is None
        assert mbd.sym_forces is None

    def test_evaluate_forces_raises_without_Force(self):
        data = {
            "NBodies": 1, "joints": [(0, 1)], "types": ["R"],
            "parent_cg_to_joint": [_Z3], "joint_to_child_cg": [_Z3],
            "axis_u": [[0, 0, 1]], "axis_u1": [None], "axis_u2": [None],
        }
        mbd = MbdSystem3D(data=data)
        with pytest.raises(RuntimeError, match="forces_func is None"):
            mbd.evaluate_forces(np.zeros(2))


# ---------------------------------------------------------------------------
# Goal 6 — public numeric force evaluation from mainNumVars
# ---------------------------------------------------------------------------

class TestNumericEvaluation:

    def test_evaluate_forces_returns_ForcesEvalResult(self, mbd_force_rr):
        mbd, _ = mbd_force_rr
        mnv = _mnv_rr(mbd, 0.0, 0.0, 0.0, 0.0)
        result = mbd.evaluate_forces(mnv)
        assert isinstance(result, ForcesEvalResult)

    def test_evaluate_forces_output_shapes(self, mbd_force_rr):
        mbd, _ = mbd_force_rr
        mnv    = _mnv_rr(mbd, 0.0, 0.0, 0.0, 0.0)
        result = mbd.evaluate_forces(mnv)
        NB = mbd.NBodies
        for arr in (result.cg, result.points_bd,
                    result.torsion_spring, result.gravity, result.total):
            assert arr.shape == (NB, 6), f"Shape mismatch: {arr.shape}"

    def test_evaluate_total_wrench_matches_total(self, mbd_force_rr):
        mbd, _ = mbd_force_rr
        mnv = _mnv_rr(mbd, 0.1, -0.2, 0.3, 0.4, Fx=2.0, Fy=3.0, m1=5.0)
        total_via_result = mbd.evaluate_forces(mnv).total
        total_via_method = mbd.evaluate_total_wrench(mnv)
        np.testing.assert_allclose(
            np.asarray(total_via_result),
            np.asarray(total_via_method),
            atol=0,
        )

    def test_wrong_length_raises(self, mbd_force_rr):
        mbd, _ = mbd_force_rr
        with pytest.raises(ValueError, match="mainNumVars length mismatch"):
            mbd.evaluate_forces(np.zeros(3))


# ---------------------------------------------------------------------------
# Goal 7a — CG force contribution
# ---------------------------------------------------------------------------

class TestCGForce:

    def test_cg_force_Fx_on_body1(self, mbd_force_rr):
        """Body 1 must receive exactly Fx in its x-column."""
        mbd, _ = mbd_force_rr
        Fx_val = 7.0
        mnv    = _mnv_rr(mbd, 0.0, 0.0, 0.0, 0.0, Fx=Fx_val)
        result = mbd.evaluate_forces(mnv)
        assert float(result.cg[0, 0]) == pytest.approx(Fx_val)

    def test_cg_moment_Mz_on_body1(self, mbd_force_rr):
        """Body 1 must receive exactly Mz in its z-moment column."""
        mbd, _ = mbd_force_rr
        Mz_val = 3.5
        mnv    = _mnv_rr(mbd, 0.0, 0.0, 0.0, 0.0, Mz=Mz_val)
        result = mbd.evaluate_forces(mnv)
        assert float(result.cg[0, 5]) == pytest.approx(Mz_val)

    def test_cg_zero_on_body2(self, mbd_force_rr):
        """No CG force was declared on body 2; its CG wrench must be zero."""
        mbd, _ = mbd_force_rr
        mnv    = _mnv_rr(mbd, 0.0, 0.0, 0.0, 0.0, Fx=5.0)
        result = mbd.evaluate_forces(mnv)
        np.testing.assert_allclose(np.asarray(result.cg[1]), 0.0, atol=1e-14)


# ---------------------------------------------------------------------------
# Goal 7b — PointsBD: force + induced CG moment
# ---------------------------------------------------------------------------

class TestPointsBDForce:

    def test_points_bd_Fy_on_body2(self, mbd_force_rr):
        """Body 2 must receive exactly Fy in its y-force column."""
        mbd, _ = mbd_force_rr
        Fy_val = 5.0
        mnv    = _mnv_rr(mbd, 0.0, 0.0, 0.0, 0.0, Fy=Fy_val, b=1.5)
        result = mbd.evaluate_forces(mnv)
        assert float(result.points_bd[1, 1]) == pytest.approx(Fy_val)

    def test_points_bd_induces_moment_on_body2(self, mbd_force_rr):
        """[0,Fy,0] at rho=[b,0,0] → moment = rho × F = [0,0,-Fy*b]."""
        mbd, _ = mbd_force_rr
        Fy_val, b_val = 4.0, 2.0
        mnv    = _mnv_rr(mbd, 0.0, 0.0, 0.0, 0.0, Fy=Fy_val, b=b_val)
        result = mbd.evaluate_forces(mnv)
        # moment z = rho × F at zero angle: [b,0,0] × [0,Fy,0] = [0,0, b*Fy]
        # sign: rho cross F → [0*0-0*Fy, 0*0-b*0, b*Fy-0*0] = [0, 0, b*Fy]
        expected_Mz = b_val * Fy_val
        assert float(result.points_bd[1, 5]) == pytest.approx(expected_Mz, rel=1e-9)

    def test_points_bd_zero_on_body1(self, mbd_force_rr):
        """No PointsBD force was declared on body 1."""
        mbd, _ = mbd_force_rr
        mnv    = _mnv_rr(mbd, 0.0, 0.0, 0.0, 0.0, Fy=5.0)
        result = mbd.evaluate_forces(mnv)
        np.testing.assert_allclose(np.asarray(result.points_bd[0]), 0.0, atol=1e-14)


# ---------------------------------------------------------------------------
# Goal 7c — TorsionSpring
# ---------------------------------------------------------------------------

class TestTorsionSpring:

    def test_torsion_spring_zero_at_zero_angle(self, mbd_force_rr):
        """At θ=0 (eq=0): torsion moment must be zero."""
        mbd, _ = mbd_force_rr
        mnv    = _mnv_rr(mbd, 0.0, 0.0, 0.0, 0.0, k_t=2.0)
        result = mbd.evaluate_forces(mnv)
        np.testing.assert_allclose(np.asarray(result.torsion_spring), 0.0, atol=1e-14)

    def test_torsion_spring_moment_at_nonzero_angle(self, mbd_force_rr):
        """At θ=π/4 (child=body1), torque = -k_t * θ about z-axis."""
        mbd, _ = mbd_force_rr
        theta  = math.pi / 4
        k_t_val = 2.0
        mnv    = _mnv_rr(mbd, theta, 0.0, 0.0, 0.0, k_t=k_t_val)
        result = mbd.evaluate_forces(mnv)
        # Restoring torque on child (body 1) about z: -k_t * theta
        expected_Mz = -k_t_val * theta
        assert float(result.torsion_spring[0, 5]) == pytest.approx(expected_Mz, rel=1e-9)

    def test_torsion_spring_pe_at_nonzero_angle(self, mbd_force_rr):
        """PE = 0.5 * k_t * theta^2."""
        mbd, _ = mbd_force_rr
        theta   = math.pi / 6
        k_t_val = 3.0
        mnv     = _mnv_rr(mbd, theta, 0.0, 0.0, 0.0, k_t=k_t_val)
        result  = mbd.evaluate_forces(mnv)
        expected = 0.5 * k_t_val * theta ** 2
        assert float(result.spring_potential_energy) == pytest.approx(expected, rel=1e-9)


# ---------------------------------------------------------------------------
# Goal 7d — Gravity
# ---------------------------------------------------------------------------

class TestGravity:

    def test_gravity_Fy_on_body1(self, mbd_force_rr):
        """Body 1 gravity force = m1 * g_y in the y-column."""
        mbd, _ = mbd_force_rr
        m1_val = 5.0
        mnv    = _mnv_rr(mbd, 0.0, 0.0, 0.0, 0.0, m1=m1_val)
        result = mbd.evaluate_forces(mnv)
        expected = m1_val * (-9.81)
        assert float(result.gravity[0, 1]) == pytest.approx(expected, rel=1e-9)

    def test_gravity_Fy_on_body2(self, mbd_force_rr):
        mbd, _ = mbd_force_rr
        m2_val = 3.0
        mnv    = _mnv_rr(mbd, 0.0, 0.0, 0.0, 0.0, m2=m2_val)
        result = mbd.evaluate_forces(mnv)
        expected = m2_val * (-9.81)
        assert float(result.gravity[1, 1]) == pytest.approx(expected, rel=1e-9)

    def test_gravity_no_moment(self, mbd_force_rr):
        """Gravity acts at CG → no induced moment."""
        mbd, _ = mbd_force_rr
        mnv    = _mnv_rr(mbd, 0.0, 0.0, 0.0, 0.0, m1=2.0, m2=3.0)
        result = mbd.evaluate_forces(mnv)
        for b in range(mbd.NBodies):
            np.testing.assert_allclose(
                np.asarray(result.gravity[b, 3:]), 0.0, atol=1e-14,
                err_msg=f"body {b+1} gravity moment must be zero",
            )


# ---------------------------------------------------------------------------
# Goal 7e — TensionSpring (separate fixture with numeric-only geometry)
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def mbd_tension_spring():
    """
    1-body system: floating joint (constant rotation = identity).
    One tension spring from GR[0]=[0,0,0] to BD[1][0]=[L0, 0, 0].

    At rest (no motion), spring is at natural length → F = 0.
    Stretch the attachment point to [2*L0, 0, 0] → F = k*(2*L0 - L0) = k*L0.
    """
    k  = sym.Symbol("k",  positive=True)
    L0 = sym.Symbol("L0", positive=True)

    data = {
        "NBodies": 1,
        "joints": [(0, 1)],
        "types": ["R"],
        "parent_cg_to_joint": [[1.0, 0.0, 0.0]],
        "joint_to_child_cg":  [[-1.0, 0.0, 0.0]],
        "axis_u": [[0, 0, 1]], "axis_u1": [None], "axis_u2": [None],
    }
    s = sym.Symbol("s", real=True)       # attachment x-offset in body frame
    return MbdSystem3D(
        data=data,
        force_points_sym={"k": k, "L0": L0},
        points_sym={"s": s},
        Initial_Points={
            "GR": [[0.0, 0.0, 0.0]],
            "BD": {1: [[s, 0.0, 0.0]]},
        },
        Force={
            "TensionSpring": [(0, 0, 1, 0, k, L0)],   # GR[0] ↔ BD[1][0]
        },
    ), dict(k=k, L0=L0, s=s)


class TestTensionSpring:

    def test_tension_spring_zero_at_natural_length(self, mbd_tension_spring):
        """Spring force is zero when |r_B - r_A| == L0."""
        mbd, syms = mbd_tension_spring
        # At θ=0, body-1 CG = [0,0,0], attachment at [s,0,0] in body frame
        # → absolute position = [s,0,0].  GR[0] = [0,0,0].
        # With s = L0_val the distance equals L0 → force = 0.
        L0_val, k_val = 2.0, 3.0
        # q=0, qd=0, k=3, L0=2, s=2 (natural length)
        mnv = np.array([0.0, 0.0, k_val, L0_val, L0_val])
        result = mbd.evaluate_forces(mnv)
        np.testing.assert_allclose(
            np.asarray(result.tension_spring), 0.0, atol=1e-10,
        )

    def test_tension_spring_nonzero_when_stretched(self, mbd_tension_spring):
        """Stretched spring must produce non-zero force on body 1."""
        mbd, _ = mbd_tension_spring
        L0_val, k_val, s_val = 1.0, 2.0, 3.0   # stretched: |s| - L0 = 2
        mnv    = np.array([0.0, 0.0, k_val, L0_val, s_val])
        result = mbd.evaluate_forces(mnv)
        # Force on body 1: along (GR - BD) direction = -x direction
        # magnitude = k*(s-L0) = 2*(3-1) = 4.0
        Fx_on_body1 = float(result.tension_spring[0, 0])
        assert Fx_on_body1 == pytest.approx(-k_val * (s_val - L0_val), rel=1e-9)

    def test_tension_spring_pe(self, mbd_tension_spring):
        """PE = 0.5 * k * (L - L0)^2."""
        mbd, _ = mbd_tension_spring
        L0_val, k_val, s_val = 1.0, 2.0, 3.0
        mnv    = np.array([0.0, 0.0, k_val, L0_val, s_val])
        result = mbd.evaluate_forces(mnv)
        expected = 0.5 * k_val * (s_val - L0_val) ** 2
        assert float(result.spring_potential_energy) == pytest.approx(expected, rel=1e-9)


# ---------------------------------------------------------------------------
# Goal 7f — total force aggregation
# ---------------------------------------------------------------------------

class TestTotalForce:

    def test_total_is_sum_of_categories(self, mbd_force_rr):
        """total wrench must equal sum of all non-zero category arrays."""
        mbd, _ = mbd_force_rr
        mnv    = _mnv_rr(mbd, 0.2, -0.1, 0.0, 0.0,
                          Fx=3.0, Fy=2.0, Mz=1.0, k_t=1.5, m1=4.0, m2=2.0)
        result = mbd.evaluate_forces(mnv)
        manual_total = (
            np.asarray(result.cg)
            + np.asarray(result.points_bd)
            + np.asarray(result.torsion_spring)
            + np.asarray(result.gravity)
        )
        np.testing.assert_allclose(
            np.asarray(result.total), manual_total, atol=1e-12,
        )

    def test_evaluate_total_wrench_shape(self, mbd_force_rr):
        mbd, _ = mbd_force_rr
        mnv    = _mnv_rr(mbd, 0.0, 0.0, 0.0, 0.0)
        total  = mbd.evaluate_total_wrench(mnv)
        assert total.shape == (mbd.NBodies, 6)


# ---------------------------------------------------------------------------
# Goal 8 — changing symbolic coefficient values changes the result
# ---------------------------------------------------------------------------

class TestParametricSensitivity:

    def test_Fx_scales_cg_force(self, mbd_force_rr):
        """Doubling Fx must double body-1 CG x-force."""
        mbd, _ = mbd_force_rr
        r1 = float(mbd.evaluate_forces(_mnv_rr(mbd, 0, 0, 0, 0, Fx=2.0)).cg[0, 0])
        r2 = float(mbd.evaluate_forces(_mnv_rr(mbd, 0, 0, 0, 0, Fx=4.0)).cg[0, 0])
        assert r2 == pytest.approx(2.0 * r1)

    def test_k_t_scales_torsion_moment(self, mbd_force_rr):
        """Doubling k_t must double the torsion restoring moment (linear)."""
        mbd, _ = mbd_force_rr
        theta = math.pi / 8
        m1_rr = float(mbd.evaluate_forces(_mnv_rr(mbd, theta, 0, 0, 0, k_t=1.0))
                      .torsion_spring[0, 5])
        m2_rr = float(mbd.evaluate_forces(_mnv_rr(mbd, theta, 0, 0, 0, k_t=2.0))
                      .torsion_spring[0, 5])
        assert m2_rr == pytest.approx(2.0 * m1_rr, rel=1e-9)

    def test_m1_scales_gravity_force(self, mbd_force_rr):
        """Doubling m1 must double body-1 gravity force."""
        mbd, _ = mbd_force_rr
        g1 = float(mbd.evaluate_forces(_mnv_rr(mbd, 0, 0, 0, 0, m1=3.0)).gravity[0, 1])
        g2 = float(mbd.evaluate_forces(_mnv_rr(mbd, 0, 0, 0, 0, m1=6.0)).gravity[0, 1])
        assert g2 == pytest.approx(2.0 * g1, rel=1e-9)

    def test_Fy_scales_points_bd_force(self, mbd_force_rr):
        """Doubling Fy must double body-2 PointsBD y-force."""
        mbd, _ = mbd_force_rr
        f1 = float(mbd.evaluate_forces(_mnv_rr(mbd, 0, 0, 0, 0, Fy=2.0)).points_bd[1, 1])
        f2 = float(mbd.evaluate_forces(_mnv_rr(mbd, 0, 0, 0, 0, Fy=4.0)).points_bd[1, 1])
        assert f2 == pytest.approx(2.0 * f1, rel=1e-9)

    def test_result_invariant_to_kinematics_changes(self, mbd_force_rr):
        """Force params not affecting kinematics; changing only k_t must not alter gravity."""
        mbd, _ = mbd_force_rr
        g_low  = np.asarray(mbd.evaluate_forces(_mnv_rr(mbd, 0, 0, 0, 0, k_t=1.0, m1=5.0)).gravity)
        g_high = np.asarray(mbd.evaluate_forces(_mnv_rr(mbd, 0, 0, 0, 0, k_t=99.0, m1=5.0)).gravity)
        np.testing.assert_allclose(g_low, g_high, atol=1e-14)
