# source/multibody_3d/tests/test_compile_B_lambdified.py
"""
Numeric tests for VelocityTransformation3D.compile_B_lambdified().

Each test builds a joint system, compiles B, evaluates at random (and zero) q,
and checks structural properties + selected numeric values.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
import sympy as sym

from multibody_3d import JointSystem3D, VelocityTransformation3D


# --------------- helpers ---------------

def _make_q_syms(n: int) -> sym.Matrix:
    return sym.Matrix(sym.symbols(f"q0:{n}"))


def _make_system(data: dict) -> JointSystem3D:
    return JointSystem3D.from_data(data)


def _z3():
    return [0.0, 0.0, 0.0]


_IDENTITY_QUAT = [1.0, 0.0, 0.0, 0.0]


# --------------- fixtures ---------------

@pytest.fixture()
def single_revolute():
    """0 -> 1, R about z, nonzero geometry vectors."""
    data = {
        "NBodies": 1,
        "joints": [(0, 1)],
        "types": ["R"],
        "parent_cg_to_joint": [[1.0, 0.0, 0.0]],
        "joint_to_child_cg": [[0.5, 0.0, 0.0]],
        "axis_u": [[0.0, 0.0, 1.0]],
        "axis_u1": [None],
        "axis_u2": [None],
    }
    sys = _make_system(data)
    vt = VelocityTransformation3D(sys)
    q = _make_q_syms(vt.total_cfg_dof)
    B_func = vt.compile_B_lambdified(q)
    return vt, B_func


@pytest.fixture()
def single_prismatic():
    """0 -> 1, P along z."""
    data = {
        "NBodies": 1,
        "joints": [(0, 1)],
        "types": ["P"],
        "parent_cg_to_joint": [[0.0, 0.0, 0.0]],
        "joint_to_child_cg": [[0.0, 0.0, 0.0]],
        "axis_u": [[0.0, 0.0, 1.0]],
        "axis_u1": [None],
        "axis_u2": [None],
    }
    sys = _make_system(data)
    vt = VelocityTransformation3D(sys)
    q = _make_q_syms(vt.total_cfg_dof)
    B_func = vt.compile_B_lambdified(q)
    return vt, B_func


@pytest.fixture()
def chain_rps():
    """3-body chain: 0->1(R) ->2(P) ->3(S)."""
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
    return vt, B_func


@pytest.fixture()
def single_universal():
    """0 -> 1, U with u1=z, u2=x."""
    data = {
        "NBodies": 1,
        "joints": [(0, 1)],
        "types": ["U"],
        "parent_cg_to_joint": [[0.0, 0.0, 0.0]],
        "joint_to_child_cg": [[1.0, 0.0, 0.0]],
        "axis_u": [None],
        "axis_u1": [[0.0, 0.0, 1.0]],
        "axis_u2": [[1.0, 0.0, 0.0]],
    }
    sys = _make_system(data)
    vt = VelocityTransformation3D(sys)
    q = _make_q_syms(vt.total_cfg_dof)
    B_func = vt.compile_B_lambdified(q)
    return vt, B_func


@pytest.fixture()
def single_cylindrical():
    """0 -> 1, C about z."""
    data = {
        "NBodies": 1,
        "joints": [(0, 1)],
        "types": ["C"],
        "parent_cg_to_joint": [[0.0, 0.0, 0.0]],
        "joint_to_child_cg": [[0.5, 0.0, 0.0]],
        "axis_u": [[0.0, 0.0, 1.0]],
        "axis_u1": [None],
        "axis_u2": [None],
    }
    sys = _make_system(data)
    vt = VelocityTransformation3D(sys)
    q = _make_q_syms(vt.total_cfg_dof)
    B_func = vt.compile_B_lambdified(q)
    return vt, B_func


@pytest.fixture()
def single_floating():
    """0 -> 1, F."""
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
    return vt, B_func


# ============================================================
# Shape & structural tests
# ============================================================

class TestShapeAndStructure:
    def test_shape_single_revolute(self, single_revolute):
        vt, B_func = single_revolute
        B = B_func(np.array([0.0]))
        assert B.shape == (6, 1)  # 6*1 x 1 (no ground)

    def test_shape_chain_rps(self, chain_rps):
        vt, B_func = chain_rps
        q_int = np.zeros(vt.total_cfg_dof)
        sl = vt.q_slices[2]  # S joint cfg slice
        q_int[sl.start] = 1.0  # e0 = 1 (identity quat)
        B = B_func(q_int)
        assert B.shape == (18, 5)  # 6*3 x (1+1+3), no ground

    def test_shape_universal(self, single_universal):
        vt, B_func = single_universal
        B = B_func(np.zeros(2))
        assert B.shape == (6, 2)  # 6*1 x 2

    def test_shape_cylindrical(self, single_cylindrical):
        vt, B_func = single_cylindrical
        B = B_func(np.zeros(2))
        assert B.shape == (6, 2)  # 6*1 x 2

    def test_shape_floating(self, single_floating):
        vt, B_func = single_floating
        q_int = np.zeros(vt.total_cfg_dof)  # 7
        q_int[3] = 1.0  # e0 = 1 (identity quat)
        B = B_func(q_int)
        assert B.shape == (6, 6)  # 6*1 x 6


# ============================================================
# Finite (no NaN / Inf) at random q
# ============================================================

class TestFinite:
    @pytest.mark.parametrize("seed", [0, 1, 2])
    def test_finite_chain_rps(self, chain_rps, seed):
        vt, B_func = chain_rps
        rng = np.random.default_rng(seed)
        q_int = np.zeros(vt.total_cfg_dof)
        q_int[:2] = rng.standard_normal(2) * 0.5  # R + P
        quat = rng.standard_normal(4)
        quat = quat / np.linalg.norm(quat)
        sl = vt.q_slices[2]
        q_int[sl.start:sl.stop] = quat
        B = B_func(q_int)
        assert np.all(np.isfinite(B))

    @pytest.mark.parametrize("seed", [10, 11])
    def test_finite_universal(self, single_universal, seed):
        vt, B_func = single_universal
        rng = np.random.default_rng(seed)
        q = rng.standard_normal(vt.total_cfg_dof) * 0.5
        B = B_func(q)
        assert np.all(np.isfinite(B))

    @pytest.mark.parametrize("seed", [20, 21])
    def test_finite_floating(self, single_floating, seed):
        vt, B_func = single_floating
        rng = np.random.default_rng(seed)
        q_int = np.zeros(vt.total_cfg_dof)
        q_int[:3] = rng.standard_normal(3) * 0.5
        quat = rng.standard_normal(4)
        quat = quat / np.linalg.norm(quat)
        q_int[3:7] = quat
        B = B_func(q_int)
        assert np.all(np.isfinite(B))


# ============================================================
# Known-value tests at q = 0
# ============================================================

class TestKnownValuesAtZero:
    def test_revolute_at_zero(self, single_revolute):
        """At q=0  R(z,0)=I so d = j2c = [0.5,0,0], u = [0,0,1].

        B block = [[-skew(d)*u], [u]] = [[0], [0.5], [0], [0], [0], [1]].
        """
        vt, B_func = single_revolute
        B = B_func(np.array([0.0]))
        body_block = B[0:6, 0]  # Body 1 now at rows 0:6
        expected = np.array([0.0, 0.5, 0.0, 0.0, 0.0, 1.0])
        np.testing.assert_allclose(body_block, expected, atol=1e-14)

    def test_prismatic_at_zero(self, single_prismatic):
        """P along z at q=0.  B block = [[u],[0]] = [[0,0,1,0,0,0]]^T."""
        vt, B_func = single_prismatic
        B = B_func(np.array([0.0]))
        body_block = B[0:6, 0]
        expected = np.array([0.0, 0.0, 1.0, 0.0, 0.0, 0.0])
        np.testing.assert_allclose(body_block, expected, atol=1e-14)

    def test_cylindrical_at_zero(self, single_cylindrical):
        """C about z at q=[0,0].  DOF0 = rotation col, DOF1 = translation col.

        d = j2c = [0.5,0,0], u=[0,0,1].
        Rot col: [[-skew(d)*u],[u]] = [[0,0.5,0,0,0,1]].T
        Trans col: [[u],[0]] = [[0,0,1,0,0,0]].T
        """
        vt, B_func = single_cylindrical
        B = B_func(np.array([0.0, 0.0]))
        block = B[0:6, :]
        expected_rot = np.array([0.0, 0.5, 0.0, 0.0, 0.0, 1.0])
        expected_tra = np.array([0.0, 0.0, 1.0, 0.0, 0.0, 0.0])
        np.testing.assert_allclose(block[:, 0], expected_rot, atol=1e-14)
        np.testing.assert_allclose(block[:, 1], expected_tra, atol=1e-14)

    def test_floating_at_zero(self, single_floating):
        """F at q=zeros.  Block = [[I, -skew(d)],[0, I]] with d=[0.5,0,0].

        -skew([0.5,0,0]) = [[0, 0, 0], [0, 0, 0.5], [0, -0.5, 0]].
        """
        vt, B_func = single_floating
        q_int = np.zeros(vt.total_cfg_dof)  # 7
        q_int[3] = 1.0  # e0 = 1 (identity quat)
        B = B_func(q_int)
        block = B[0:6, :]
        expected = np.array([
            [1, 0, 0,  0,     0,    0  ],
            [0, 1, 0,  0,     0,    0.5],
            [0, 0, 1,  0,    -0.5,  0  ],
            [0, 0, 0,  1,     0,    0  ],
            [0, 0, 0,  0,     1,    0  ],
            [0, 0, 0,  0,     0,    1  ],
        ], dtype=float)
        np.testing.assert_allclose(block, expected, atol=1e-14)


# ============================================================
# Revolute: known value at nonzero theta
# ============================================================

class TestRevoluteNonzero:
    def test_revolute_at_pi_over_4(self, single_revolute):
        """Revolute about z, theta=pi/4.  R_z(pi/4) * [0.5,0,0] = [c/2, s/2, 0]."""
        vt, B_func = single_revolute
        theta = math.pi / 4
        B = B_func(np.array([theta]))

        c, s = math.cos(theta), math.sin(theta)
        # d = A_abs[1]*j2c = R_z(theta)*[0.5,0,0] = [0.5c, 0.5s, 0]
        d = np.array([0.5 * c, 0.5 * s, 0.0])
        u = np.array([0.0, 0.0, 1.0])

        # -skew(d)*u
        # skew(d)*u = [d[1]*u[2]-d[2]*u[1], d[2]*u[0]-d[0]*u[2], d[0]*u[1]-d[1]*u[0]]
        #           = [0.5s, -0.5c, 0]
        expected_top = np.array([-0.5 * s, 0.5 * c, 0.0])
        expected = np.concatenate([expected_top, u])

        body_block = B[0:6, 0]
        np.testing.assert_allclose(body_block, expected, atol=1e-12)


# ============================================================
# Small-angle stability for spherical (exp-coordinates near 0)
# ============================================================

class TestSmallAngle:
    def test_spherical_identity_quat(self, chain_rps):
        """Evaluate with identity quaternion — must not produce NaN."""
        vt, B_func = chain_rps
        q_int = np.zeros(vt.total_cfg_dof)
        sl = vt.q_slices[2]
        q_int[sl.start] = 1.0
        B = B_func(q_int)
        assert np.all(np.isfinite(B))

    def test_spherical_near_identity_quat(self, chain_rps):
        """Quaternion close to identity must give finite B."""
        vt, B_func = chain_rps
        q_int = np.zeros(vt.total_cfg_dof)
        sl = vt.q_slices[2]
        quat = np.array([1.0, 1e-18, 1e-18, 1e-18])
        quat = quat / np.linalg.norm(quat)
        q_int[sl.start:sl.stop] = quat
        B = B_func(q_int)
        assert np.all(np.isfinite(B))


# ============================================================
# Idempotency: same q gives same B
# ============================================================

class TestIdempotency:
    def test_same_q_same_B(self, chain_rps):
        vt, B_func = chain_rps
        rng = np.random.default_rng(99)
        q_int = np.zeros(vt.total_cfg_dof)
        q_int[:2] = rng.standard_normal(2) * 0.3
        quat = rng.standard_normal(4)
        quat = quat / np.linalg.norm(quat)
        sl = vt.q_slices[2]
        q_int[sl.start:sl.stop] = quat
        B1 = B_func(q_int)
        B2 = B_func(q_int)
        np.testing.assert_array_equal(B1, B2)
