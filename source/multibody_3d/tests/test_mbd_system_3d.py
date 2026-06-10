# tests/test_mbd_system_3d.py
"""Smoke tests for the MbdSystem3D orchestrator façade.

These tests verify **wiring** — that the façade correctly delegates to and
composes JointSystem3D, CoordBundle, and VelocityTransformation3D — not the
correctness of symbolic kinematics (which is covered elsewhere).
"""
from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from multibody_3d import MbdSystem3D, JointSystem3D, build_joint_coordinates
from multibody_3d import VelocityTransformation3D, NumericModelParams, CoordBundle

jax = pytest.importorskip("jax")


# ---------------------------------------------------------------------------
# Shared data fixtures
# ---------------------------------------------------------------------------

Z3 = [0.0, 0.0, 0.0]


@pytest.fixture
def data_RR():
    """2-body chain: two revolute joints (no quat complexity)."""
    return {
        "NBodies": 2,
        "joints": [(0, 1), (1, 2)],
        "types": ["R", "R"],
        "parent_cg_to_joint": [Z3, [0, 1, 0]],
        "joint_to_child_cg": [Z3, [0, -1, 0]],
        "axis_u": [[0, 0, 1], [0, 0, 1]],
        "axis_u1": [None, None],
        "axis_u2": [None, None],
    }


@pytest.fixture
def data_SR():
    """2-body chain: spherical + revolute (quat internal coords)."""
    return {
        "NBodies": 2,
        "joints": [(0, 1), (1, 2)],
        "types": ["S", "R"],
        "parent_cg_to_joint": [Z3, [0.5, 0.5, 0]],
        "joint_to_child_cg": [[0.5, 0, 0], [0, -0.5, 0]],
        "axis_u": [None, [0, 0, 1]],
        "axis_u1": [None, None],
        "axis_u2": [None, None],
        "rot_param": ["quat", None],
    }


@pytest.fixture
def data_FR():
    """2-body chain: floating + revolute."""
    return {
        "NBodies": 2,
        "joints": [(0, 1), (1, 2)],
        "types": ["F", "R"],
        "parent_cg_to_joint": [Z3, [1.0, 0, 0]],
        "joint_to_child_cg": [Z3, [-0.5, 0, 0]],
        "axis_u": [None, [0, 1, 0]],
        "axis_u1": [None, None],
        "axis_u2": [None, None],
        "rot_param": ["quat", None],
    }


# ---------------------------------------------------------------------------
# Construction tests
# ---------------------------------------------------------------------------

class TestConstruction:

    def test_from_data(self, data_RR):
        mbd = MbdSystem3D.from_data(data_RR)
        assert isinstance(mbd, MbdSystem3D)
        assert mbd.data is data_RR

    def test_direct_constructor(self, data_RR):
        mbd = MbdSystem3D(data_RR)
        assert isinstance(mbd.joint_system, JointSystem3D)
        assert isinstance(mbd.coords, CoordBundle)
        assert isinstance(mbd.vt, VelocityTransformation3D)

    def test_from_example(self, data_SR):
        fake_module = SimpleNamespace(data=data_SR, __name__="fake_example")
        mbd = MbdSystem3D.from_example(fake_module)
        assert mbd.NBodies == 2

    def test_from_example_missing_data(self):
        fake_module = SimpleNamespace(__name__="no_data_mod")
        with pytest.raises(AttributeError, match="no 'data' attribute"):
            MbdSystem3D.from_example(fake_module)

    def test_data_must_be_dict(self):
        with pytest.raises(TypeError, match="expects a dict"):
            MbdSystem3D(data=[1, 2, 3])

    def test_repr(self, data_RR):
        mbd = MbdSystem3D(data_RR)
        r = repr(mbd)
        assert "MbdSystem3D" in r
        assert "NBodies=2" in r


# ---------------------------------------------------------------------------
# Property tests
# ---------------------------------------------------------------------------

class TestProperties:

    def test_sizing_RR(self, data_RR):
        mbd = MbdSystem3D(data_RR)
        assert mbd.NBodies == 2
        assert mbd.NJoints == 2
        assert mbd.total_dof == 2
        assert mbd.total_cfg_dof == 2
        assert mbd.total_user_dof == 2

    def test_sizing_SR(self, data_SR):
        mbd = MbdSystem3D(data_SR)
        assert mbd.NBodies == 2
        assert mbd.NJoints == 2
        assert mbd.total_dof == 4       # S=3 + R=1
        assert mbd.total_cfg_dof == 5   # S=4 (quat) + R=1
        assert mbd.total_user_dof == 5  # rot_param='quat' → user=4+1

    def test_sizing_FR(self, data_FR):
        mbd = MbdSystem3D(data_FR)
        assert mbd.total_dof == 7       # F=6 + R=1
        assert mbd.total_cfg_dof == 8   # F=7 + R=1

    def test_properties_match_joint_system(self, data_SR):
        mbd = MbdSystem3D(data_SR)
        assert mbd.NBodies == mbd.joint_system.NBodies
        assert mbd.total_dof == mbd.joint_system.total_dof
        assert mbd.total_cfg_dof == mbd.joint_system.total_cfg_dof
        assert mbd.total_user_dof == mbd.joint_system.total_user_dof

    def test_symbolic_vectors_match_coords(self, data_SR):
        mbd = MbdSystem3D(data_SR)
        assert mbd.q_user == mbd.coords.q_user
        assert mbd.qd_user == mbd.coords.qd_user
        assert mbd.q_int == mbd.coords.q_int
        assert mbd.qd_int == mbd.coords.qd_int

    def test_symbolic_vector_shapes(self, data_SR):
        mbd = MbdSystem3D(data_SR)
        assert mbd.q_int.shape == (mbd.total_cfg_dof, 1)
        assert mbd.qd_int.shape == (mbd.total_dof, 1)
        assert mbd.q_user.shape == (mbd.total_user_dof, 1)
        assert mbd.qd_user.shape == (mbd.total_dof, 1)


# ---------------------------------------------------------------------------
# Coordinate mapping tests
# ---------------------------------------------------------------------------

class TestCoordinateMapping:

    def test_roundtrip_RR(self, data_RR):
        """R-only: user and internal coords are identical."""
        mbd = MbdSystem3D(data_RR)
        q_user = np.array([0.3, -0.7])
        q_int = mbd.map_q_user_to_q_int(q_user)
        np.testing.assert_allclose(q_int, q_user)
        q_back = mbd.map_q_int_to_q_user(q_int)
        np.testing.assert_allclose(q_back, q_user)

    def test_SR_mapping_changes_length(self, data_SR):
        """S (quat) + R: user has 5 entries, internal has 5 (quat passes through)."""
        mbd = MbdSystem3D(data_SR)
        # Unit quaternion [1,0,0,0] for S + angle for R
        q_user = np.array([1.0, 0.0, 0.0, 0.0, 0.5])
        q_int = mbd.map_q_user_to_q_int(q_user)
        assert q_int.shape[0] == mbd.total_cfg_dof

    def test_shape_mismatch_q_user(self, data_RR):
        mbd = MbdSystem3D(data_RR)
        with pytest.raises(ValueError, match="q_user length mismatch"):
            mbd.map_q_user_to_q_int(np.zeros(99))

    def test_shape_mismatch_q_int(self, data_RR):
        mbd = MbdSystem3D(data_RR)
        with pytest.raises(ValueError, match="q_int length mismatch"):
            mbd.map_q_int_to_q_user(np.zeros(99))


# ---------------------------------------------------------------------------
# Numeric params tests
# ---------------------------------------------------------------------------

class TestNumericParams:

    def test_returns_correct_type(self, data_RR):
        mbd = MbdSystem3D(data_RR)
        params = mbd.build_numeric_params()
        assert isinstance(params, NumericModelParams)

    def test_caching(self, data_RR):
        mbd = MbdSystem3D(data_RR)
        p1 = mbd.build_numeric_params()
        p2 = mbd.build_numeric_params()
        assert p1 is p2

    def test_force_rebuild(self, data_RR):
        mbd = MbdSystem3D(data_RR)
        p1 = mbd.build_numeric_params()
        p3 = mbd.build_numeric_params(force=True)
        assert p1 is not p3

    def test_params_dimensions(self, data_SR):
        mbd = MbdSystem3D(data_SR)
        params = mbd.build_numeric_params()
        assert params.total_dof == mbd.total_dof
        assert params.total_cfg_dof == mbd.total_cfg_dof


# ---------------------------------------------------------------------------
# Eager evaluation tests (public API)
# ---------------------------------------------------------------------------

class TestEagerEvaluation:

    def test_B_shape_RR(self, data_RR):
        mbd = MbdSystem3D(data_RR)
        q_user = np.array([0.1, 0.2])
        qd     = np.array([1.0, -0.5])
        B = mbd.evaluate_B(np.concatenate([q_user, qd]))
        assert B.shape == (6 * mbd.NBodies, mbd.total_dof)

    def test_Bdot_shape_RR(self, data_RR):
        mbd = MbdSystem3D(data_RR)
        q_user = np.array([0.1, 0.2])
        qd     = np.array([1.0, -0.5])
        Bd = mbd.evaluate_Bdot(np.concatenate([q_user, qd]))
        assert Bd.shape == (6 * mbd.NBodies, mbd.total_dof)

    def test_B_shape_SR(self, data_SR):
        mbd = MbdSystem3D(data_SR)
        q_user = np.array([1.0, 0.0, 0.0, 0.0, 0.3])  # unit quat + R angle
        qd     = np.zeros(mbd.total_dof)
        B = mbd.evaluate_B(np.concatenate([q_user, qd]))
        assert B.shape == (6 * mbd.NBodies, mbd.total_dof)

    def test_wrong_length_B_raises(self, data_RR):
        mbd = MbdSystem3D(data_RR)
        with pytest.raises(ValueError, match="mainNumVars length mismatch"):
            mbd.evaluate_B(np.zeros(99))

    def test_wrong_length_Bdot_raises(self, data_RR):
        mbd = MbdSystem3D(data_RR)
        with pytest.raises(ValueError, match="mainNumVars length mismatch"):
            mbd.evaluate_Bdot(np.zeros(99))


# ---------------------------------------------------------------------------
# JIT evaluator tests
# ---------------------------------------------------------------------------

class TestBFuncAttributes:
    """Verify B_func / Bdot_func are compiled at construction and consistent."""

    def test_B_func_is_callable(self, data_SR):
        mbd = MbdSystem3D(data_SR)
        assert callable(mbd.B_func)

    def test_Bdot_func_is_callable(self, data_SR):
        mbd = MbdSystem3D(data_SR)
        assert callable(mbd.Bdot_func)

    def test_B_func_matches_evaluate_B(self, data_SR):
        """B_func(mainNumVars_int) must equal evaluate_B(mainNumVars)."""
        mbd    = MbdSystem3D(data_SR)
        q_user = np.array([1.0, 0.0, 0.0, 0.0, 0.3])
        qd     = np.array([0.1, -0.2, 0.3, 0.5])
        mnv    = np.concatenate([q_user, qd])
        B_pub  = mbd.evaluate_B(mnv)
        arr    = mbd._validate_mainNumVars_shape(mnv)
        B_raw  = np.asarray(mbd.B_func(mbd._build_mainNumVars_int(arr)))
        np.testing.assert_allclose(B_raw, B_pub, atol=0)

    def test_Bdot_func_matches_evaluate_Bdot(self, data_SR):
        """Bdot_func(mainNumVars_int) must equal evaluate_Bdot(mainNumVars)."""
        mbd    = MbdSystem3D(data_SR)
        q_user = np.array([1.0, 0.0, 0.0, 0.0, 0.3])
        qd     = np.array([0.1, -0.2, 0.3, 0.5])
        mnv    = np.concatenate([q_user, qd])
        Bd_pub = mbd.evaluate_Bdot(mnv)
        arr    = mbd._validate_mainNumVars_shape(mnv)
        Bd_raw = np.asarray(mbd.Bdot_func(mbd._build_mainNumVars_int(arr)))
        np.testing.assert_allclose(Bd_raw, Bd_pub, atol=0)


# ---------------------------------------------------------------------------
# mainNumVars API tests
# ---------------------------------------------------------------------------

class TestMainNumVars:
    """Verify the mainNumVars-based public runtime API end-to-end."""

    # ---- slice metadata ----

    def test_slice_metadata_RR(self, data_RR):
        """Slices cover the full mainNumVars length with no gaps."""
        mbd = MbdSystem3D(data_RR)
        assert mbd._slc_q_user.start == 0
        assert mbd._slc_q_user.stop  == mbd.total_user_dof
        assert mbd._slc_qd.start     == mbd.total_user_dof
        assert mbd._slc_qd.stop      == mbd.total_user_dof + mbd.total_dof
        # no body/force params
        assert mbd._slc_body.start   == mbd._slc_body.stop
        assert mbd._slc_force.start  == mbd._slc_force.stop
        assert mbd._slc_force.stop   == len(mbd.mainSymVars)

    def test_slice_metadata_SR(self, data_SR):
        mbd = MbdSystem3D(data_SR)
        assert mbd._slc_force.stop == len(mbd.mainSymVars)
        assert mbd._slc_force_int.stop == len(mbd.mainSymVars_int)

    # ---- wrong-length input ----

    def test_wrong_length_evaluate_B(self, data_RR):
        mbd = MbdSystem3D(data_RR)
        with pytest.raises(ValueError, match="mainNumVars length mismatch"):
            mbd.evaluate_B(np.zeros(99))

    def test_wrong_length_evaluate_Bdot(self, data_RR):
        mbd = MbdSystem3D(data_RR)
        with pytest.raises(ValueError, match="mainNumVars length mismatch"):
            mbd.evaluate_Bdot(np.zeros(99))

    # ---- _extract_q_int_qd ----

    def test_extract_q_int_qd_RR(self, data_RR):
        """R-only: q_user == q_int; qd extracted unchanged."""
        mbd    = MbdSystem3D(data_RR)
        q_user = np.array([0.3, -0.7])
        qd     = np.array([1.5,  2.1])
        mnv    = np.concatenate([q_user, qd])
        q_int_out, qd_out = mbd._extract_q_int_qd(mnv)
        np.testing.assert_allclose(q_int_out, q_user)
        np.testing.assert_allclose(qd_out, qd)

    def test_extract_q_int_qd_SR(self, data_SR):
        """S(quat)+R: unit quaternion passes through; qd unchanged."""
        mbd    = MbdSystem3D(data_SR)
        q_user = np.array([1.0, 0.0, 0.0, 0.0, 0.5])
        qd     = np.array([0.1, -0.2, 0.3, 0.4])
        mnv    = np.concatenate([q_user, qd])
        q_int_out, qd_out = mbd._extract_q_int_qd(mnv)
        assert q_int_out.shape[0] == mbd.total_cfg_dof
        np.testing.assert_allclose(qd_out, qd)

    # ---- body/force passthrough via _build_mainNumVars_int ----

    def test_build_mainNumVars_int_body_force_passthrough(self):
        """Body/force numeric values must survive the conversion unchanged."""
        import sympy as sym
        body_syms  = {"m": sym.Symbol("m"), "L": sym.Symbol("L")}
        force_syms = {"Fx": sym.Symbol("Fx")}
        data = {
            "NBodies": 1, "joints": [(0, 1)], "types": ["R"],
            "parent_cg_to_joint": [[0.0, 0.0, 0.0]],
            "joint_to_child_cg":  [[0.0, 0.0, 0.0]],
            "axis_u": [[0, 0, 1]], "axis_u1": [None], "axis_u2": [None],
        }
        mbd = MbdSystem3D(
            data=data,
            body_data_sym=body_syms,
            force_points_sym=force_syms,
        )
        q_user     = np.array([0.4])
        qd         = np.array([1.2])
        body_vals  = np.array([10.0, 2.5])  # m=10, L=2.5
        force_vals = np.array([3.7])         # Fx=3.7
        mnv = np.concatenate([q_user, qd, body_vals, force_vals])

        mnv_int = mbd._build_mainNumVars_int(mnv)
        # internal layout: [q_int, qd, body, force]
        qi = mbd.total_cfg_dof
        qd_len = mbd.total_dof
        np.testing.assert_allclose(mnv_int[qi : qi + qd_len], qd)
        np.testing.assert_allclose(mnv_int[qi + qd_len : qi + qd_len + 2], body_vals)
        np.testing.assert_allclose(mnv_int[-1], force_vals[0])
        # total length must match mainSymVars_int
        assert len(mnv_int) == len(mbd.mainSymVars_int)

    # ---- evaluate_B / Bdot match internal path ----

    def test_evaluate_B_matches_B_func(self, data_RR):
        """evaluate_B(mainNumVars) == B_func(mainNumVars_int)."""
        mbd    = MbdSystem3D(data_RR)
        q_user = np.array([0.1, 0.2])
        qd     = np.array([1.0, -0.5])
        mnv    = np.concatenate([q_user, qd])
        B_pub  = mbd.evaluate_B(mnv)
        arr    = mbd._validate_mainNumVars_shape(mnv)
        B_raw  = np.asarray(mbd.B_func(mbd._build_mainNumVars_int(arr)))
        np.testing.assert_allclose(B_pub, B_raw, atol=1e-12)

    def test_evaluate_Bdot_matches_Bdot_func(self, data_SR):
        """evaluate_Bdot(mainNumVars) == Bdot_func(mainNumVars_int)."""
        mbd    = MbdSystem3D(data_SR)
        q_user = np.array([1.0, 0.0, 0.0, 0.0, 0.3])
        qd     = np.array([0.1, -0.2, 0.3, 0.5])
        mnv    = np.concatenate([q_user, qd])
        Bd_pub = mbd.evaluate_Bdot(mnv)
        arr    = mbd._validate_mainNumVars_shape(mnv)
        Bd_raw = np.asarray(mbd.Bdot_func(mbd._build_mainNumVars_int(arr)))
        np.testing.assert_allclose(Bd_pub, Bd_raw, atol=1e-12)

    # ---- JIT callable builders ----

    def test_B_func_is_reentrant(self, data_SR):
        """Calling B_func twice with the same mainNumVars_int gives identical output."""
        mbd    = MbdSystem3D(data_SR)
        q_user = np.array([1.0, 0.0, 0.0, 0.0, 0.3])
        qd     = np.array([0.1, -0.2, 0.3, 0.5])
        mnv    = np.concatenate([q_user, qd])
        arr    = mbd._validate_mainNumVars_shape(mnv)
        mnv_int = mbd._build_mainNumVars_int(arr)
        B1 = np.asarray(mbd.B_func(mnv_int))
        B2 = np.asarray(mbd.B_func(mnv_int))
        np.testing.assert_array_equal(B1, B2)

    def test_Bdot_func_is_reentrant(self, data_SR):
        """Calling Bdot_func twice with the same mainNumVars_int gives identical output."""
        mbd    = MbdSystem3D(data_SR)
        q_user = np.array([1.0, 0.0, 0.0, 0.0, 0.3])
        qd     = np.array([0.1, -0.2, 0.3, 0.5])
        mnv    = np.concatenate([q_user, qd])
        arr    = mbd._validate_mainNumVars_shape(mnv)
        mnv_int = mbd._build_mainNumVars_int(arr)
        Bd1 = np.asarray(mbd.Bdot_func(mnv_int))
        Bd2 = np.asarray(mbd.Bdot_func(mnv_int))
        np.testing.assert_array_equal(Bd1, Bd2)



# ---------------------------------------------------------------------------
# Summary table smoke test
# ---------------------------------------------------------------------------

class TestSummaryTable:

    def test_runs_without_error(self, data_RR):
        mbd = MbdSystem3D(data_RR)
        mbd.summary_table()  # should not raise


# ---------------------------------------------------------------------------
# Variable-geometry runtime tests
# ---------------------------------------------------------------------------
# A single fixture: one revolute joint whose geometry (p2j, j2c, axis_u)
# depends on a symbolic parameter L.  This covers all 8 required checks
# with minimal overhead.

import sympy as sym  # noqa: E402 (already imported at top of file in practice)


@pytest.fixture
def sym_L_system():
    """
    1-body system: one revolute joint with symbolic geometry parameter L.

    Layout
    ------
    - Joint 0→1, type R, axis z
    - parent_cg_to_joint = [L, 0, 0]  (symbolic)
    - joint_to_child_cg  = [-L, 0, 0] (symbolic)
    - axis_u             = [0, 0, 1]

    body_data_sym = {"L": L}
    force_points_sym = {"Fx": Fx}  (unused by B/Bdot, tests force tail)

    mainSymVars = [q_theta, qd_theta, L_val, Fx_val]
    """
    L  = sym.Symbol("L",  real=True, positive=True)
    Fx = sym.Symbol("Fx", real=True)

    data = {
        "NBodies": 1,
        "joints": [(0, 1)],
        "types": ["R"],
        "parent_cg_to_joint": [sym.Matrix([L, 0, 0])],
        "joint_to_child_cg":  [sym.Matrix([-L, 0, 0])],
        "axis_u":  [[0, 0, 1]],
        "axis_u1": [None],
        "axis_u2": [None],
    }
    body_data_sym    = {"L": L}
    force_points_sym = {"Fx": Fx}
    return MbdSystem3D(
        data=data,
        body_data_sym=body_data_sym,
        force_points_sym=force_points_sym,
    )


def _mnv(mbd, q_user, qd, body_vals, force_vals=()):
    """Assemble a mainNumVars vector for *mbd*."""
    return np.concatenate([
        np.asarray(q_user, dtype=float),
        np.asarray(qd, dtype=float),
        np.asarray(body_vals, dtype=float),
        np.asarray(force_vals, dtype=float),
    ])


class TestVariableGeometryRuntime:
    """Verify the automatic runtime backend for symbolic-geometry systems."""

    # 1. __post_init__ creates runtime artifacts automatically -------------------

    def test_B_func_built_at_construction(self, sym_L_system):
        mbd = sym_L_system
        assert callable(mbd.B_func), "B_func must be callable after __post_init__"

    def test_Bdot_func_built_at_construction(self, sym_L_system):
        mbd = sym_L_system
        assert callable(mbd.Bdot_func), "Bdot_func must be callable after __post_init__"

    def test_numeric_params_built_at_construction(self, sym_L_system):
        mbd = sym_L_system
        assert isinstance(mbd._numeric_params, NumericModelParams)

    # 2. evaluate_B/Bdot need no manual setup in the caller ---------------------

    def test_evaluate_B_no_manual_setup(self, sym_L_system):
        """evaluate_B must work straight after construction with no extra calls."""
        mbd = sym_L_system
        mnv = _mnv(mbd, [0.3], [1.2], [0.5], [0.0])
        B = mbd.evaluate_B(mnv)
        assert B.shape == (6 * mbd.NBodies, mbd.total_dof)

    def test_evaluate_Bdot_no_manual_setup(self, sym_L_system):
        mbd = sym_L_system
        mnv = _mnv(mbd, [0.3], [1.2], [0.5], [0.0])
        Bd = mbd.evaluate_Bdot(mnv)
        assert Bd.shape == (6 * mbd.NBodies, mbd.total_dof)

    # 3. mainNumVars → mainNumVars_int conversion is correct --------------------

    def test_build_mainNumVars_int_layout(self, sym_L_system):
        """[q_user, qd, body, force] → [q_int, qd, body, force] with R-joint identity."""
        mbd = sym_L_system
        q_user, qd   = [0.7], [2.1]
        body, force  = [1.5], [9.9]
        mnv    = _mnv(mbd, q_user, qd, body, force)
        arr    = mbd._validate_mainNumVars_shape(mnv)
        mnv_int = mbd._build_mainNumVars_int(arr)

        # R-joint: q_user == q_int
        assert mnv_int[0] == pytest.approx(q_user[0])
        # qd preserved
        assert mnv_int[mbd.total_cfg_dof] == pytest.approx(qd[0])
        # body param preserved
        n_qi = mbd.total_cfg_dof
        n_qd = mbd.total_dof
        assert mnv_int[n_qi + n_qd] == pytest.approx(body[0])
        # force param preserved
        assert mnv_int[n_qi + n_qd + 1] == pytest.approx(force[0])
        # length matches mainSymVars_int
        assert len(mnv_int) == len(mbd.mainSymVars_int)

    # 4. B_func / Bdot_func accept full mainNumVars_int -------------------------

    def test_B_func_accepts_mainNumVars_int(self, sym_L_system):
        mbd = sym_L_system
        mnv    = _mnv(mbd, [0.0], [0.0], [1.0], [0.0])
        arr    = mbd._validate_mainNumVars_shape(mnv)
        mnv_int = mbd._build_mainNumVars_int(arr)
        B = mbd.B_func(mnv_int)   # must not raise
        assert np.asarray(B).shape == (6 * mbd.NBodies, mbd.total_dof)

    def test_Bdot_func_accepts_mainNumVars_int(self, sym_L_system):
        mbd = sym_L_system
        mnv    = _mnv(mbd, [0.0], [1.0], [1.0], [0.0])
        arr    = mbd._validate_mainNumVars_shape(mnv)
        mnv_int = mbd._build_mainNumVars_int(arr)
        Bd = mbd.Bdot_func(mnv_int)
        assert np.asarray(Bd).shape == (6 * mbd.NBodies, mbd.total_dof)

    # 5. Body-data tail is preserved in order ----------------------------------

    def test_body_params_reach_geometry_layer(self, sym_L_system):
        """B at L=1.0 must differ from B at L=2.0 (geometry scales with L)."""
        mbd = sym_L_system
        q, qd = [0.0], [0.0]
        B1 = np.asarray(mbd.evaluate_B(_mnv(mbd, q, qd, [1.0], [0.0])))
        B2 = np.asarray(mbd.evaluate_B(_mnv(mbd, q, qd, [2.0], [0.0])))
        assert not np.allclose(B1, B2), "B must depend on the body geometry parameter L"

    # 6. Changing body-data values changes B/Bdot ------------------------------

    def test_B_scales_with_L(self, sym_L_system):
        """For a revolute joint, the d_kj column of B scales linearly with L."""
        mbd = sym_L_system
        q, qd = [0.0], [0.0]   # identity rotation, zero speed

        B1 = np.asarray(mbd.evaluate_B(_mnv(mbd, q, qd, [1.0], [0.0])))
        B2 = np.asarray(mbd.evaluate_B(_mnv(mbd, q, qd, [2.0], [0.0])))

        # d_kj = (p2j + j2c) = [L,0,0] + [-L,0,0] = [0,0,0] for any L,
        # but the *joint* position rJ = r_p + A_p*p2j = [L,0,0] while
        # r_cg = [0,0,0], so d_kj = r_cg - rJ = [-L,0,0].
        # The translational rows of B contain skew(d_kj)*u = L * something.
        # Doubling L must therefore double the translational part of B.
        np.testing.assert_allclose(B2[:3], 2 * B1[:3], atol=1e-10)

    def test_Bdot_changes_with_qd_and_L(self, sym_L_system):
        """Bdot at (qd=1, L=1) must differ from Bdot at (qd=1, L=2)."""
        mbd = sym_L_system
        q = [0.0]
        Bd1 = np.asarray(mbd.evaluate_Bdot(_mnv(mbd, q, [1.0], [1.0], [0.0])))
        Bd2 = np.asarray(mbd.evaluate_Bdot(_mnv(mbd, q, [1.0], [2.0], [0.0])))
        assert not np.allclose(Bd1, Bd2)

    # 7. Force-parameter tail is accepted even though B/Bdot ignores it --------

    def test_force_tail_accepted(self, sym_L_system):
        """Different force values must not change B or Bdot."""
        mbd = sym_L_system
        q, qd, body = [0.3], [0.5], [1.0]
        B_f0 = np.asarray(mbd.evaluate_B(_mnv(mbd, q, qd, body, [0.0])))
        B_f1 = np.asarray(mbd.evaluate_B(_mnv(mbd, q, qd, body, [99.9])))
        np.testing.assert_array_equal(B_f0, B_f1)

        Bd_f0 = np.asarray(mbd.evaluate_Bdot(_mnv(mbd, q, qd, body, [0.0])))
        Bd_f1 = np.asarray(mbd.evaluate_Bdot(_mnv(mbd, q, qd, body, [99.9])))
        np.testing.assert_array_equal(Bd_f0, Bd_f1)

    # 8. Wrong-length inputs raise clear errors --------------------------------

    def test_wrong_length_evaluate_B_raises(self, sym_L_system):
        mbd = sym_L_system
        with pytest.raises(ValueError, match="mainNumVars length mismatch"):
            mbd.evaluate_B(np.zeros(99))

    def test_wrong_length_evaluate_Bdot_raises(self, sym_L_system):
        mbd = sym_L_system
        with pytest.raises(ValueError, match="mainNumVars length mismatch"):
            mbd.evaluate_Bdot(np.zeros(1))

    def test_missing_body_params_raises(self, sym_L_system):
        """Omitting body/force tail must be caught by length validation."""
        mbd = sym_L_system
        # only [q_user, qd] — missing body+force tail
        short = np.concatenate([np.array([0.3]), np.array([1.0])])
        with pytest.raises(ValueError, match="mainNumVars length mismatch"):
            mbd.evaluate_B(short)


# ---------------------------------------------------------------------------
# Symbolic points layer tests
# ---------------------------------------------------------------------------
# Fixture: 1-body revolute system with one point symbol ``a``.
#
# Geometry (all numeric so the geometry extractor is constant):
#   parent_cg_to_joint = [1, 0, 0]
#   joint_to_child_cg  = [-1, 0, 0]
#   axis_u             = [0, 0, 1]
#
# At θ=0: A_abs[1] = I, body-1 CG = [0, 0, 0].
# Body point [a, 0, 0] maps to r_abs = [a, 0, 0], rho = [a, 0, 0].
#
# mainNumVars = [θ, θ̇, a]   (length 3)

@pytest.fixture
def mbd_with_points():
    """1-body revolute system with one symbolic point parameter 'a'."""
    a = sym.Symbol("a", real=True)
    data = {
        "NBodies": 1,
        "joints": [(0, 1)],
        "types": ["R"],
        "parent_cg_to_joint": [[1.0, 0.0, 0.0]],
        "joint_to_child_cg":  [[-1.0, 0.0, 0.0]],
        "axis_u":  [[0, 0, 1]],
        "axis_u1": [None],
        "axis_u2": [None],
    }
    mbd = MbdSystem3D(
        data=data,
        points_sym={"a": a},
        Initial_Points={
            "GR": [[0.0, 0.0, 0.0]],
            "BD": {1: [[a, 0.0, 0.0]]},
        },
    )
    return mbd, a


class TestSymbolicPoints:
    """Validate the 3D point layer: symbolic cache, JAX evaluators, force reduction."""

    # ---- 1. Point symbols appear in mainSymVars ---------------------------------

    def test_points_sym_in_mainSymVars(self, mbd_with_points):
        mbd, a = mbd_with_points
        assert a in list(mbd.mainSymVars)
        assert a in list(mbd.mainSymVars_int)

    def test_points_sym_at_expected_slice(self, mbd_with_points):
        """The symbol 'a' sits exactly at _slc_points in mainSymVars."""
        mbd, a = mbd_with_points
        slc = mbd._slc_points
        assert list(mbd.mainSymVars)[slc.start] is a

    def test_points_sym_int_at_expected_slice(self, mbd_with_points):
        mbd, a = mbd_with_points
        slc = mbd._slc_points_int
        assert list(mbd.mainSymVars_int)[slc.start] is a

    # ---- 2. User-declared order is preserved ------------------------------------

    def test_points_sym_order_preserved(self, mbd_with_points):
        """Symbols at the points slice must match points_sym value order."""
        mbd, a = mbd_with_points
        slc = mbd._slc_points
        syms_in_vector = list(mbd.mainSymVars)[slc]
        assert syms_in_vector == [a]

    # ---- 3. sym_points built automatically in __post_init__ --------------------

    def test_sym_points_not_none(self, mbd_with_points):
        mbd, _ = mbd_with_points
        assert mbd.sym_points is not None

    def test_sym_points_correct_type(self, mbd_with_points):
        from multibody_3d import SymbolicPointsCache3D
        mbd, _ = mbd_with_points
        assert isinstance(mbd.sym_points, SymbolicPointsCache3D)

    def test_sym_points_has_correct_entries(self, mbd_with_points):
        mbd, _ = mbd_with_points
        assert len(mbd.sym_points.ground_points) == 1
        assert 1 in mbd.sym_points.body_points
        assert len(mbd.sym_points.body_points[1]) == 1

    # ---- 4. Symbolic expression opacity ----------------------------------------

    def test_sym_rho_is_matmul(self, mbd_with_points):
        """rho_abs (A_abs @ r_local) must remain un-expanded MatMul."""
        from sympy import MatMul
        mbd, _ = mbd_with_points
        rec = mbd.sym_points.body_points[1][0]
        assert isinstance(rec.rho_abs, MatMul)

    def test_sym_r_abs_contains_point_symbol(self, mbd_with_points):
        """r_abs for a body point must carry the free symbol 'a'."""
        mbd, a = mbd_with_points
        rec = mbd.sym_points.body_points[1][0]
        assert a in rec.r_abs.free_symbols

    # ---- 5. JAX evaluator built during __post_init__ ---------------------------

    def test_points_func_built(self, mbd_with_points):
        mbd, _ = mbd_with_points
        assert mbd.points_func is not None
        assert callable(mbd.points_func)

    def test_points_spec_built(self, mbd_with_points):
        from multibody_3d import PointsRuntimeSpec
        mbd, _ = mbd_with_points
        assert mbd._points_spec is not None
        assert isinstance(mbd._points_spec, PointsRuntimeSpec)

    # ---- 6. Public evaluate_points(mainNumVars) ----------------------------------

    def test_evaluate_points_returns_correct_type(self, mbd_with_points):
        from multibody_3d import PointsEvalResult
        mbd, _ = mbd_with_points
        result = mbd.evaluate_points(np.array([0.0, 0.0, 1.5]))
        assert isinstance(result, PointsEvalResult)

    def test_evaluate_points_output_shapes(self, mbd_with_points):
        mbd, _ = mbd_with_points
        result = mbd.evaluate_points(np.array([0.0, 0.0, 1.5]))
        assert result.r_abs_body.shape  == (1, 3)
        assert result.rho_abs_body.shape == (1, 3)
        assert result.r_abs_cg.shape    == (mbd.NBodies, 3)
        assert result.r_abs_gr.shape    == (1, 3)

    def test_evaluate_points_gr_at_world_origin(self, mbd_with_points):
        mbd, _ = mbd_with_points
        result = mbd.evaluate_points(np.array([0.0, 0.0, 1.5]))
        np.testing.assert_allclose(
            np.asarray(result.r_abs_gr[0]), [0.0, 0.0, 0.0], atol=1e-12
        )

    def test_evaluate_points_cg_at_zero_angle(self, mbd_with_points):
        """At θ=0: body-1 CG = p2j + j2c = [1,0,0] + [-1,0,0] = [0,0,0]."""
        mbd, _ = mbd_with_points
        result = mbd.evaluate_points(np.array([0.0, 0.0, 1.5]))
        np.testing.assert_allclose(
            np.asarray(result.r_abs_cg[0]), [0.0, 0.0, 0.0], atol=1e-10
        )

    def test_evaluate_points_body_coords_at_zero_angle(self, mbd_with_points):
        """At θ=0 and a=1.5: body-point absolute position = [1.5, 0, 0]."""
        mbd, _ = mbd_with_points
        result = mbd.evaluate_points(np.array([0.0, 0.0, 1.5]))
        np.testing.assert_allclose(
            np.asarray(result.r_abs_body[0]), [1.5, 0.0, 0.0], atol=1e-10
        )

    def test_evaluate_points_rho_at_zero_angle(self, mbd_with_points):
        """At θ=0 and a=1.5: CG-relative arm rho = [1.5, 0, 0]."""
        mbd, _ = mbd_with_points
        result = mbd.evaluate_points(np.array([0.0, 0.0, 1.5]))
        np.testing.assert_allclose(
            np.asarray(result.rho_abs_body[0]), [1.5, 0.0, 0.0], atol=1e-10
        )

    # ---- 7. Internal path via points_func(mainNumVars_int) ---------------------

    def test_evaluate_points_internal_path(self, mbd_with_points):
        mbd, _ = mbd_with_points
        mnv = np.array([0.0, 0.0, 1.5])
        arr = mbd._validate_mainNumVars_shape(mnv)
        mnv_int = mbd._build_mainNumVars_int(arr)
        result = mbd.points_func(mnv_int)
        np.testing.assert_allclose(
            np.asarray(result.r_abs_body[0]), [1.5, 0.0, 0.0], atol=1e-10
        )

    # ---- 8. Changing point-symbol value changes coordinates --------------------

    def test_point_sym_value_changes_result(self, mbd_with_points):
        """Doubling 'a' must double the x-coordinate of the body point."""
        mbd, _ = mbd_with_points
        r1 = float(np.asarray(mbd.evaluate_points(np.array([0.0, 0.0, 1.5])).r_abs_body[0, 0]))
        r2 = float(np.asarray(mbd.evaluate_points(np.array([0.0, 0.0, 3.0])).r_abs_body[0, 0]))
        assert r2 == pytest.approx(2.0 * r1)

    # ---- 9. No points_func / sym_points when Initial_Points is empty -----------

    def test_no_points_func_when_no_initial_points(self, data_RR):
        mbd = MbdSystem3D(data_RR)
        assert mbd.points_func is None
        assert mbd.sym_points is None

    # ---- 10. Symbolic force reduction -------------------------------------------

    def test_sym_force_reduction_has_point_symbol(self, mbd_with_points):
        from multibody_3d import sym_force_reduction_at_point
        mbd, a = mbd_with_points
        rec = mbd.sym_points.body_points[1][0]
        f_vec = sym.Matrix([0, 0, 1])
        _, m_eq = sym_force_reduction_at_point(rec, f_vec)
        assert a in m_eq.free_symbols

    def test_sym_force_reduction_gr_raises(self, mbd_with_points):
        """sym_force_reduction_at_point must refuse GR points (no body frame)."""
        from multibody_3d import sym_force_reduction_at_point
        mbd, _ = mbd_with_points
        gr_rec = mbd.sym_points.ground_points[0]
        with pytest.raises(ValueError):
            sym_force_reduction_at_point(gr_rec, sym.Matrix([1, 0, 0]))

    def test_facade_sym_force_reduction(self, mbd_with_points):
        """MbdSystem3D.sym_force_reduction_at_point delegates correctly."""
        mbd, a = mbd_with_points
        rec = mbd.sym_points.body_points[1][0]
        f_vec = sym.Matrix([0, 0, 1])
        _, m_eq = mbd.sym_force_reduction_at_point(rec, f_vec)
        assert a in m_eq.free_symbols

    # ---- 11. Numeric force reduction via PointsEvalResult.reduce_force_at -----

    def test_numeric_force_reduction_correctness(self, mbd_with_points):
        """
        rho = [1.5, 0, 0], F = [0, 0, 1]
        moment = rho × F = [0*1-0*0, 0*0-1.5*1, 1.5*0-0*0] = [0, -1.5, 0]
        """
        mbd, _ = mbd_with_points
        result = mbd.evaluate_points(np.array([0.0, 0.0, 1.5]))
        f_eq, m_eq = result.reduce_force_at(0, [0.0, 0.0, 1.0])
        np.testing.assert_allclose(
            np.asarray(f_eq).ravel(), [0.0, 0.0, 1.0], atol=1e-12
        )
        np.testing.assert_allclose(
            np.asarray(m_eq).ravel(), [0.0, -1.5, 0.0], atol=1e-10
        )
