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
# JAX evaluation tests (internal coordinates)
# ---------------------------------------------------------------------------

class TestJaxEvaluation:

    def test_B_shape_RR(self, data_RR):
        mbd = MbdSystem3D(data_RR)
        q_int = np.array([0.1, 0.2])
        B = mbd.evaluate_B_jax(q_int)
        assert B.shape == (6 * mbd.NBodies, mbd.total_dof)

    def test_Bdot_shape_RR(self, data_RR):
        mbd = MbdSystem3D(data_RR)
        q_int = np.array([0.1, 0.2])
        qd = np.array([1.0, -0.5])
        Bd = mbd.evaluate_Bdot_jax(q_int, qd)
        assert Bd.shape == (6 * mbd.NBodies, mbd.total_dof)

    def test_B_shape_SR(self, data_SR):
        mbd = MbdSystem3D(data_SR)
        q_int = np.array([1.0, 0.0, 0.0, 0.0, 0.3])  # unit quat + R angle
        B = mbd.evaluate_B_jax(q_int)
        assert B.shape == (6 * mbd.NBodies, mbd.total_dof)

    def test_shape_mismatch_q_int(self, data_RR):
        mbd = MbdSystem3D(data_RR)
        with pytest.raises(ValueError, match="q_int length mismatch"):
            mbd.evaluate_B_jax(np.zeros(99))

    def test_shape_mismatch_qd(self, data_RR):
        mbd = MbdSystem3D(data_RR)
        q_int = np.zeros(mbd.total_cfg_dof)
        with pytest.raises(ValueError, match="qd length mismatch"):
            mbd.evaluate_Bdot_jax(q_int, np.zeros(99))


# ---------------------------------------------------------------------------
# JIT evaluator tests
# ---------------------------------------------------------------------------

class TestJitEvaluators:

    def test_B_jit_matches_eager(self, data_SR):
        mbd = MbdSystem3D(data_SR)
        q_int = np.array([1.0, 0.0, 0.0, 0.0, 0.3])
        B_eager = mbd.evaluate_B_jax(q_int)
        B_fn = mbd.build_B_evaluator_jax()
        B_jit = B_fn(q_int)
        np.testing.assert_allclose(B_jit, B_eager, atol=1e-12)

    def test_Bdot_jit_matches_eager(self, data_SR):
        mbd = MbdSystem3D(data_SR)
        q_int = np.array([1.0, 0.0, 0.0, 0.0, 0.3])
        qd = np.array([0.1, -0.2, 0.3, 0.5])
        Bd_eager = mbd.evaluate_Bdot_jax(q_int, qd)
        Bd_fn = mbd.build_Bdot_evaluator_jax()
        Bd_jit = Bd_fn(q_int, qd)
        np.testing.assert_allclose(Bd_jit, Bd_eager, atol=1e-12)


# ---------------------------------------------------------------------------
# User-facing helper tests
# ---------------------------------------------------------------------------

class TestUserFacingHelpers:

    def test_evaluate_B_from_user_q_RR(self, data_RR):
        """R-only: user_q == q_int, so results should match."""
        mbd = MbdSystem3D(data_RR)
        q = np.array([0.1, 0.2])
        B_user = mbd.evaluate_B_from_user_q(q)
        B_int = mbd.evaluate_B_jax(q)
        np.testing.assert_allclose(B_user, B_int, atol=1e-12)

    def test_evaluate_Bdot_from_user_state_RR(self, data_RR):
        mbd = MbdSystem3D(data_RR)
        q = np.array([0.1, 0.2])
        qd = np.array([1.0, -0.5])
        Bd_user = mbd.evaluate_Bdot_from_user_state(q, qd)
        Bd_int = mbd.evaluate_Bdot_jax(q, qd)
        np.testing.assert_allclose(Bd_user, Bd_int, atol=1e-12)

    def test_evaluate_B_from_user_q_SR(self, data_SR):
        """S(quat) + R: user coords pass through quat, so same result."""
        mbd = MbdSystem3D(data_SR)
        q_user = np.array([1.0, 0.0, 0.0, 0.0, 0.3])
        B = mbd.evaluate_B_from_user_q(q_user)
        assert B.shape == (6 * mbd.NBodies, mbd.total_dof)

    def test_user_q_shape_mismatch(self, data_RR):
        mbd = MbdSystem3D(data_RR)
        with pytest.raises(ValueError, match="q_user length mismatch"):
            mbd.evaluate_B_from_user_q(np.zeros(99))

    def test_user_qd_shape_mismatch(self, data_RR):
        mbd = MbdSystem3D(data_RR)
        q = np.zeros(mbd.total_user_dof)
        with pytest.raises(ValueError, match="qd length mismatch"):
            mbd.evaluate_Bdot_from_user_state(q, np.zeros(99))

    def test_build_B_evaluator_from_user_q(self, data_SR):
        mbd = MbdSystem3D(data_SR)
        q_user = np.array([1.0, 0.0, 0.0, 0.0, 0.3])
        fn = mbd.build_B_evaluator_from_user_q()
        B = fn(q_user)
        B_ref = mbd.evaluate_B_from_user_q(q_user)
        np.testing.assert_allclose(B, B_ref, atol=1e-12)

    def test_build_Bdot_evaluator_from_user_state(self, data_SR):
        mbd = MbdSystem3D(data_SR)
        q_user = np.array([1.0, 0.0, 0.0, 0.0, 0.3])
        qd = np.array([0.1, -0.2, 0.3, 0.5])
        fn = mbd.build_Bdot_evaluator_from_user_state()
        Bd = fn(q_user, qd)
        Bd_ref = mbd.evaluate_Bdot_from_user_state(q_user, qd)
        np.testing.assert_allclose(Bd, Bd_ref, atol=1e-12)

    def test_wrapped_evaluator_validates_shape(self, data_RR):
        mbd = MbdSystem3D(data_RR)
        fn = mbd.build_B_evaluator_from_user_q()
        with pytest.raises(ValueError, match="q_user length mismatch"):
            fn(np.zeros(99))


# ---------------------------------------------------------------------------
# Summary table smoke test
# ---------------------------------------------------------------------------

class TestSummaryTable:

    def test_runs_without_error(self, data_RR):
        mbd = MbdSystem3D(data_RR)
        mbd.summary_table()  # should not raise
