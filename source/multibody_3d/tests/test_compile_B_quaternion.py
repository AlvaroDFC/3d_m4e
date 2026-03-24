# source/multibody_3d/tests/test_compile_B_quaternion.py
"""
Tests for quaternion-based compile_B_lambdified().

Covers:
- Case A: S/F joints with rot_param="quat" (quaternion user coordinates).
- Case B: S/F joints with rot_param="euler" (euler user coords, internal quaternion).
- Shape, finiteness, known-value checks at identity quaternion.
- Mapping between user and internal coordinates.

In both cases the compiled B_func always takes q_int (which contains
quaternion entries for S/F joints).  The only difference between euler
and quat rot_param is the user-facing coordinate vector q_user and the
mapping functions map_q_user_to_q_int / map_q_int_to_q_user.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
import sympy as sym

from multibody_3d import JointSystem3D, VelocityTransformation3D
from multibody_3d.multibody_core.joint_system_3d import euler_to_quat, quat_to_euler


# --------------- helpers ---------------

def _z3():
    return [0.0, 0.0, 0.0]


def _make_system(data: dict) -> JointSystem3D:
    return JointSystem3D.from_data(data)


def _make_q_syms(n: int) -> sym.Matrix:
    return sym.Matrix(sym.symbols(f"q0:{n}"))


# ============================================================
# Case B fixtures (euler S/F — default rot_param)
# ============================================================

@pytest.fixture()
def spherical_euler():
    """0 -> 1, S with euler rot_param (default)."""
    data = {
        "NBodies": 1,
        "joints": [(0, 1)],
        "types": ["S"],
        "parent_cg_to_joint": [[0.0, 0.0, 0.0]],
        "joint_to_child_cg": [[0.5, 0.0, 0.0]],
        "axis_u": [None],
        "axis_u1": [None],
        "axis_u2": [None],
    }
    sys = _make_system(data)
    vt = VelocityTransformation3D(sys)
    q = _make_q_syms(vt.total_cfg_dof)
    B_func = vt.compile_B_lambdified(q)
    return vt, B_func, sys


@pytest.fixture()
def floating_euler():
    """0 -> 1, F with euler rot_param (default)."""
    data = {
        "NBodies": 1,
        "joints": [(0, 1)],
        "types": ["F"],
        "parent_cg_to_joint": [[0.0, 0.0, 0.0]],
        "joint_to_child_cg": [[0.5, 0.0, 0.0]],
        "axis_u": [None],
        "axis_u1": [None],
        "axis_u2": [None],
    }
    sys = _make_system(data)
    vt = VelocityTransformation3D(sys)
    q = _make_q_syms(vt.total_cfg_dof)
    B_func = vt.compile_B_lambdified(q)
    return vt, B_func, sys


@pytest.fixture()
def chain_rps_euler():
    """3-body chain: 0->1(R) ->2(P) ->3(S), S is euler (default)."""
    data = {
        "NBodies": 3,
        "joints": [(0, 1), (1, 2), (2, 3)],
        "types": ["R", "P", "S"],
        "parent_cg_to_joint": [[1.0, 0.0, 0.0], [0.5, 0.0, 0.0], [0.3, 0.0, 0.0]],
        "joint_to_child_cg": [[0.5, 0.0, 0.0], [0.3, 0.0, 0.0], [0.2, 0.0, 0.0]],
        "axis_u": [[0.0, 0.0, 1.0], [1.0, 0.0, 0.0], None],
        "axis_u1": [None, None, None],
        "axis_u2": [None, None, None],
    }
    sys = _make_system(data)
    vt = VelocityTransformation3D(sys)
    q = _make_q_syms(vt.total_cfg_dof)
    B_func = vt.compile_B_lambdified(q)
    return vt, B_func, sys


# ============================================================
# Case A fixtures (quaternion S/F — rot_param="quat")
# ============================================================

@pytest.fixture()
def spherical_quat():
    """0 -> 1, S with quaternion rot_param."""
    data = {
        "NBodies": 1,
        "joints": [(0, 1)],
        "types": ["S"],
        "parent_cg_to_joint": [[0.0, 0.0, 0.0]],
        "joint_to_child_cg": [[0.5, 0.0, 0.0]],
        "axis_u": [None],
        "axis_u1": [None],
        "axis_u2": [None],
        "rot_param": ["quat"],
    }
    sys = _make_system(data)
    vt = VelocityTransformation3D(sys)
    q = _make_q_syms(vt.total_cfg_dof)
    B_func = vt.compile_B_lambdified(q)
    return vt, B_func, sys


@pytest.fixture()
def floating_quat():
    """0 -> 1, F with quaternion rot_param."""
    data = {
        "NBodies": 1,
        "joints": [(0, 1)],
        "types": ["F"],
        "parent_cg_to_joint": [[0.0, 0.0, 0.0]],
        "joint_to_child_cg": [[0.5, 0.0, 0.0]],
        "axis_u": [None],
        "axis_u1": [None],
        "axis_u2": [None],
        "rot_param": ["quat"],
    }
    sys = _make_system(data)
    vt = VelocityTransformation3D(sys)
    q = _make_q_syms(vt.total_cfg_dof)
    B_func = vt.compile_B_lambdified(q)
    return vt, B_func, sys


@pytest.fixture()
def chain_rps_quat():
    """3-body chain: 0->1(R) ->2(P) ->3(S), S is quaternion."""
    data = {
        "NBodies": 3,
        "joints": [(0, 1), (1, 2), (2, 3)],
        "types": ["R", "P", "S"],
        "parent_cg_to_joint": [[1.0, 0.0, 0.0], [0.5, 0.0, 0.0], [0.3, 0.0, 0.0]],
        "joint_to_child_cg": [[0.5, 0.0, 0.0], [0.3, 0.0, 0.0], [0.2, 0.0, 0.0]],
        "axis_u": [[0.0, 0.0, 1.0], [1.0, 0.0, 0.0], None],
        "axis_u1": [None, None, None],
        "axis_u2": [None, None, None],
        "rot_param": [None, None, "quat"],
    }
    sys = _make_system(data)
    vt = VelocityTransformation3D(sys)
    q = _make_q_syms(vt.total_cfg_dof)
    B_func = vt.compile_B_lambdified(q)
    return vt, B_func, sys


# ============================================================
# Shape tests — both euler and quat rot_param
# ============================================================

class TestShape:
    def test_shape_spherical_euler(self, spherical_euler):
        vt, B_func, _ = spherical_euler
        q_int = np.zeros(vt.total_cfg_dof)  # 4
        q_int[0] = 1.0  # identity quat e0
        B = B_func(q_int)
        assert B.shape == (6, vt.total_dof)

    def test_shape_floating_euler(self, floating_euler):
        vt, B_func, _ = floating_euler
        q_int = np.zeros(vt.total_cfg_dof)  # 7
        q_int[3] = 1.0  # identity quat e0
        B = B_func(q_int)
        assert B.shape == (6, vt.total_dof)

    def test_shape_chain_rps_euler(self, chain_rps_euler):
        vt, B_func, _ = chain_rps_euler
        q_int = np.zeros(vt.total_cfg_dof)  # 6
        sl = vt.q_slices[2]  # S joint
        q_int[sl.start] = 1.0
        B = B_func(q_int)
        assert B.shape == (18, vt.total_dof)

    def test_shape_spherical_quat(self, spherical_quat):
        vt, B_func, _ = spherical_quat
        q = np.array([1.0, 0.0, 0.0, 0.0])  # identity quaternion
        B = B_func(q)
        assert B.shape == (6, vt.total_dof)

    def test_shape_floating_quat(self, floating_quat):
        vt, B_func, _ = floating_quat
        q = np.array([0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0])
        B = B_func(q)
        assert B.shape == (6, vt.total_dof)

    def test_shape_chain_rps_quat(self, chain_rps_quat):
        vt, B_func, _ = chain_rps_quat
        # total_cfg_dof = 1(R) + 1(P) + 4(S_quat) = 6
        q_int = np.zeros(vt.total_cfg_dof)
        q_int[2] = 1.0  # S quaternion e0
        B = B_func(q_int)
        assert B.shape == (18, vt.total_dof)


# ============================================================
# Finite (no NaN / Inf) at random q, both cases
# ============================================================

class TestFinite:
    @pytest.mark.parametrize("seed", [0, 1, 2])
    def test_finite_spherical_quat(self, spherical_quat, seed):
        vt, B_func, _ = spherical_quat
        rng = np.random.default_rng(seed)
        quat = rng.standard_normal(4)
        quat = quat / np.linalg.norm(quat)  # normalize
        B = B_func(quat)
        assert np.all(np.isfinite(B))

    @pytest.mark.parametrize("seed", [0, 1])
    def test_finite_floating_quat(self, floating_quat, seed):
        vt, B_func, _ = floating_quat
        rng = np.random.default_rng(seed)
        q = np.zeros(7)
        q[:3] = rng.standard_normal(3) * 0.5
        quat = rng.standard_normal(4)
        q[3:7] = quat / np.linalg.norm(quat)
        B = B_func(q)
        assert np.all(np.isfinite(B))

    @pytest.mark.parametrize("seed", [0, 1])
    def test_finite_chain_rps_euler(self, chain_rps_euler, seed):
        """Euler S — B_func still takes q_int with embedded quaternion."""
        vt, B_func, _ = chain_rps_euler
        rng = np.random.default_rng(seed)
        q_int = np.zeros(vt.total_cfg_dof)
        q_int[:2] = rng.standard_normal(2) * 0.5
        quat = rng.standard_normal(4)
        quat = quat / np.linalg.norm(quat)
        sl = vt.q_slices[2]
        q_int[sl.start:sl.stop] = quat
        B = B_func(q_int)
        assert np.all(np.isfinite(B))


# ============================================================
# Known values at identity quaternion
# ============================================================

class TestKnownValuesIdentityQuat:
    def test_spherical_identity_quat_caseA(self, spherical_quat):
        """S at identity quat → Arel=I, d=j2c=[0.5,0,0].

        B block = [[-skew(d)], [I]]; shape 6x3.
        """
        vt, B_func, _ = spherical_quat
        q = np.array([1.0, 0.0, 0.0, 0.0])  # identity
        B = B_func(q)
        block = B[0:6, :]
        # d = [0.5, 0, 0], skew(d) * e1 = [0, 0, 0]
        # -skew([0.5,0,0]) = [[0, 0, 0], [0, 0, 0.5], [0, -0.5, 0]]
        expected = np.array([
            [0,    0,    0],
            [0,    0,    0.5],
            [0,   -0.5,  0],
            [1,    0,    0],
            [0,    1,    0],
            [0,    0,    1],
        ], dtype=float)
        np.testing.assert_allclose(block, expected, atol=1e-14)

    def test_spherical_identity_quat_caseB(self, spherical_euler):
        """Same test for euler S — B_func takes q_int with identity quat."""
        vt, B_func, _ = spherical_euler
        q_int = np.zeros(vt.total_cfg_dof)
        q_int[0] = 1.0  # identity quaternion e0
        B = B_func(q_int)
        block = B[0:6, :]
        expected = np.array([
            [0,    0,    0],
            [0,    0,    0.5],
            [0,   -0.5,  0],
            [1,    0,    0],
            [0,    1,    0],
            [0,    0,    1],
        ], dtype=float)
        np.testing.assert_allclose(block, expected, atol=1e-14)

    def test_floating_identity_quat_caseA(self, floating_quat):
        """F at identity quat → Arel=I, d=j2c=[0.5,0,0].

        B block = [[I, -skew(d)], [0, I]]; shape 6x6.
        """
        vt, B_func, _ = floating_quat
        q = np.array([0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0])
        B = B_func(q)
        block = B[0:6, :]
        expected = np.array([
            [1, 0, 0,  0,     0,    0],
            [0, 1, 0,  0,     0,    0.5],
            [0, 0, 1,  0,    -0.5,  0],
            [0, 0, 0,  1,     0,    0],
            [0, 0, 0,  0,     1,    0],
            [0, 0, 0,  0,     0,    1],
        ], dtype=float)
        np.testing.assert_allclose(block, expected, atol=1e-14)

    def test_floating_identity_quat_caseB(self, floating_euler):
        """Same test for euler F — B_func takes q_int with identity quat."""
        vt, B_func, _ = floating_euler
        q_int = np.zeros(vt.total_cfg_dof)  # 7
        q_int[3] = 1.0  # identity quaternion e0
        B = B_func(q_int)
        block = B[0:6, :]
        expected = np.array([
            [1, 0, 0,  0,     0,    0],
            [0, 1, 0,  0,     0,    0.5],
            [0, 0, 1,  0,    -0.5,  0],
            [0, 0, 0,  1,     0,    0],
            [0, 0, 0,  0,     1,    0],
            [0, 0, 0,  0,     0,    1],
        ], dtype=float)
        np.testing.assert_allclose(block, expected, atol=1e-14)


# ============================================================
# Nonzero quaternion rotation
# ============================================================

class TestNonzeroQuat:
    def test_spherical_90deg_z_rotation(self, spherical_quat):
        """90° rotation about z: q = [cos45, 0, 0, sin45].

        Arel = Rz(pi/2), d = Arel * j2c = Rz(90)*[0.5,0,0] = [0, 0.5, 0].
        """
        vt, B_func, _ = spherical_quat
        c = math.cos(math.pi / 4)
        s = math.sin(math.pi / 4)
        q = np.array([c, 0.0, 0.0, s])  # [w, x, y, z]
        B = B_func(q)

        # d = R_z(90°) * [0.5, 0, 0] = [0, 0.5, 0]
        # -skew([0, 0.5, 0]) = [[0, 0, -0.5], [0, 0, 0], [0.5, 0, 0]]
        expected_top = np.array([
            [0,    0,   -0.5],
            [0,    0,    0],
            [0.5,  0,    0],
        ])
        np.testing.assert_allclose(B[0:3, :], expected_top, atol=1e-12)
        np.testing.assert_allclose(B[3:6, :], np.eye(3), atol=1e-12)


# ============================================================
# Idempotency
# ============================================================

class TestIdempotency:
    def test_same_q_same_B_caseA(self, chain_rps_quat):
        vt, B_func, _ = chain_rps_quat
        q_int = np.zeros(vt.total_cfg_dof)
        q_int[0] = 0.3   # revolute
        q_int[2] = 1.0   # S quat e0
        B1 = B_func(q_int)
        B2 = B_func(q_int)
        np.testing.assert_array_equal(B1, B2)

    def test_same_q_same_B_caseB(self, chain_rps_euler):
        vt, B_func, _ = chain_rps_euler
        q_int = np.zeros(vt.total_cfg_dof)
        q_int[0] = 0.3
        sl = vt.q_slices[2]
        q_int[sl.start] = 1.0  # identity quat
        B1 = B_func(q_int)
        B2 = B_func(q_int)
        np.testing.assert_array_equal(B1, B2)


# ============================================================
# total_cfg_dof vs total_dof consistency
# ============================================================

class TestDofConsistency:
    def test_euler_S_cfg_dof_gt_dof(self, spherical_euler):
        """For euler S, total_cfg_dof == 4 > total_dof == 3."""
        vt, _, _ = spherical_euler
        assert vt.total_cfg_dof == 4
        assert vt.total_dof == 3

    def test_quat_S_cfg_dof_gt_dof(self, spherical_quat):
        """For quaternion S, total_cfg_dof == 4 > total_dof == 3."""
        vt, _, _ = spherical_quat
        assert vt.total_cfg_dof == 4
        assert vt.total_dof == 3

    def test_floating_quat_cfg_dof(self, floating_quat):
        """For quaternion F, total_cfg_dof == 7 > total_dof == 6."""
        vt, _, _ = floating_quat
        assert vt.total_cfg_dof == 7
        assert vt.total_dof == 6

    def test_chain_quat_cfg_dof(self, chain_rps_quat):
        """Chain R(1)+P(1)+S_quat(4) → total_cfg_dof=6, total_dof=5."""
        vt, _, _ = chain_rps_quat
        assert vt.total_cfg_dof == 6
        assert vt.total_dof == 5


# ============================================================
# User ↔ internal coordinate mapping
# ============================================================

class TestMapping:
    def test_euler_to_quat_at_zero(self):
        """euler_to_quat(0,0,0) → [1,0,0,0]."""
        q = euler_to_quat(0.0, 0.0, 0.0)
        np.testing.assert_allclose(q, [1, 0, 0, 0], atol=1e-15)

    def test_quat_to_euler_at_identity(self):
        """quat_to_euler([1,0,0,0]) → [0,0,0]."""
        e = quat_to_euler([1, 0, 0, 0])
        np.testing.assert_allclose(e, [0, 0, 0], atol=1e-15)

    def test_euler_roundtrip(self):
        """euler → quat → euler roundtrip at small angles."""
        angles = np.array([0.1, 0.2, 0.3])
        q = euler_to_quat(*angles)
        assert len(q) == 4
        back = quat_to_euler(q)
        np.testing.assert_allclose(back, angles, atol=1e-12)

    def test_map_q_user_to_q_int_euler_S(self, spherical_euler):
        """map_q_user_to_q_int for euler S: 3 euler → 4 quat."""
        _, _, sys = spherical_euler
        q_user = np.array([0.1, 0.2, 0.3])
        q_int = sys.map_q_user_to_q_int(q_user)
        assert len(q_int) == sys.total_cfg_dof  # 4
        # Should be a unit quaternion
        np.testing.assert_allclose(np.linalg.norm(q_int), 1.0, atol=1e-14)

    def test_map_q_user_to_q_int_quat_S(self, spherical_quat):
        """map_q_user_to_q_int for quat S: identity mapping."""
        _, _, sys = spherical_quat
        quat = np.array([0.5, 0.5, 0.5, 0.5])  # normalized
        q_int = sys.map_q_user_to_q_int(quat)
        np.testing.assert_array_equal(q_int, quat)

    def test_map_roundtrip_euler_S(self, spherical_euler):
        """user → int → user roundtrip for euler S."""
        _, _, sys = spherical_euler
        q_user = np.array([0.1, 0.2, 0.3])
        q_int = sys.map_q_user_to_q_int(q_user)
        q_user_back = sys.map_q_int_to_q_user(q_int)
        np.testing.assert_allclose(q_user_back, q_user, atol=1e-12)

    def test_map_roundtrip_euler_F(self, floating_euler):
        """user → int → user roundtrip for euler F."""
        _, _, sys = floating_euler
        q_user = np.array([1.0, 2.0, 3.0, 0.1, 0.2, 0.3])
        q_int = sys.map_q_user_to_q_int(q_user)
        assert len(q_int) == 7
        q_user_back = sys.map_q_int_to_q_user(q_int)
        np.testing.assert_allclose(q_user_back, q_user, atol=1e-12)
