# source/multibody_3d/tests/test_velocity_transformation_3d_bdot.py
"""
Numeric tests for compile_Bdot_lambdified() and related Bdot assembly.

Covers:
  - shape checks for B and Bdot
  - zero-speed sanity (Bdot == 0 when qd == 0 for applicable joint types)
  - universal-joint axis-transport regression
  - finite-difference consistency  (dB/dt ≈ Bdot)
  - mixed-joint chain coverage
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


def _make_qd_syms(n: int) -> sym.Matrix:
    return sym.Matrix(sym.symbols(f"qd0:{n}"))


def _make_system(data: dict) -> JointSystem3D:
    return JointSystem3D.from_data(data)


def _z3():
    return [0.0, 0.0, 0.0]


_IDENTITY_QUAT = [1.0, 0.0, 0.0, 0.0]


def _compile_both(vt):
    """Return (B_func, Bdot_func) compiled from fresh symbolic vectors."""
    q  = _make_q_syms(vt.total_cfg_dof)
    qd = _make_qd_syms(vt.total_dof)
    B_func    = vt.compile_B_lambdified(q)
    Bdot_func = vt.compile_Bdot_lambdified(q, qd)
    return B_func, Bdot_func


def _identity_q_int(vt):
    """Return a q_int vector at the reference config (zero angles, identity quats)."""
    q = np.zeros(vt.total_cfg_dof)
    for j_idx, jnt in enumerate(vt.joint_system.joints):
        code = jnt.type.value
        sl = vt.q_slices[j_idx]
        if code == "S":
            q[sl.start] = 1.0  # e0 = 1
        elif code == "F":
            q[sl.start + 3] = 1.0  # e0 = 1 (after x,y,z)
    return q


# --------------- fixtures ---------------

@pytest.fixture()
def single_revolute():
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
    B_func, Bdot_func = _compile_both(vt)
    return vt, B_func, Bdot_func


@pytest.fixture()
def single_prismatic():
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
    B_func, Bdot_func = _compile_both(vt)
    return vt, B_func, Bdot_func


@pytest.fixture()
def single_universal():
    """0 -> 1, U with u1=z, u2=x, nonzero geometry."""
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
    B_func, Bdot_func = _compile_both(vt)
    return vt, B_func, Bdot_func


@pytest.fixture()
def single_cylindrical():
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
    B_func, Bdot_func = _compile_both(vt)
    return vt, B_func, Bdot_func


@pytest.fixture()
def single_spherical():
    data = {
        "NBodies": 1,
        "joints": [(0, 1)],
        "types": ["S"],
        "parent_cg_to_joint": [[1.0, 0.0, 0.0]],
        "joint_to_child_cg": [[0.5, 0.0, 0.0]],
        "axis_u": [None],
        "axis_u1": [None],
        "axis_u2": [None],
    }
    sys = _make_system(data)
    vt = VelocityTransformation3D(sys)
    B_func, Bdot_func = _compile_both(vt)
    return vt, B_func, Bdot_func


@pytest.fixture()
def single_floating():
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
    B_func, Bdot_func = _compile_both(vt)
    return vt, B_func, Bdot_func


@pytest.fixture()
def chain_rps():
    """3-body chain: 0 -> 1(R) -> 2(P) -> 3(S) with nonzero geometry."""
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
    B_func, Bdot_func = _compile_both(vt)
    return vt, B_func, Bdot_func


@pytest.fixture()
def chain_ruc():
    """3-body chain: 0 -> 1(R) -> 2(U) -> 3(C).

    All joints have cfg DOF == speed DOF, so finite-difference is straightforward.
    """
    data = {
        "NBodies": 3,
        "joints": [(0, 1), (1, 2), (2, 3)],
        "types": ["R", "U", "C"],
        "parent_cg_to_joint": [[0.5, 0.0, 0.0], [0.4, 0.1, 0.0], [0.3, 0.0, 0.0]],
        "joint_to_child_cg": [[0.3, 0.0, 0.0], [0.2, 0.0, 0.1], [0.4, 0.0, 0.0]],
        "axis_u": [[0.0, 0.0, 1.0], None, [0.0, 1.0, 0.0]],
        "axis_u1": [None, [0.0, 0.0, 1.0], None],
        "axis_u2": [None, [1.0, 0.0, 0.0], None],
    }
    sys = _make_system(data)
    vt = VelocityTransformation3D(sys)
    B_func, Bdot_func = _compile_both(vt)
    return vt, B_func, Bdot_func


# ============================================================
# Shape checks
# ============================================================

class TestShape:
    """B and Bdot must have shape (6*NBodies, total_dof)."""

    def test_shape_revolute(self, single_revolute):
        vt, B_func, Bdot_func = single_revolute
        q = np.zeros(vt.total_cfg_dof)
        qd = np.zeros(vt.total_dof)
        assert B_func(q).shape == (6, 1)
        assert Bdot_func(q, qd).shape == (6, 1)

    def test_shape_prismatic(self, single_prismatic):
        vt, B_func, Bdot_func = single_prismatic
        q = np.zeros(vt.total_cfg_dof)
        qd = np.zeros(vt.total_dof)
        assert B_func(q).shape == (6, 1)
        assert Bdot_func(q, qd).shape == (6, 1)

    def test_shape_universal(self, single_universal):
        vt, B_func, Bdot_func = single_universal
        q = np.zeros(vt.total_cfg_dof)
        qd = np.zeros(vt.total_dof)
        assert B_func(q).shape == (6, 2)
        assert Bdot_func(q, qd).shape == (6, 2)

    def test_shape_cylindrical(self, single_cylindrical):
        vt, B_func, Bdot_func = single_cylindrical
        q = np.zeros(vt.total_cfg_dof)
        qd = np.zeros(vt.total_dof)
        assert B_func(q).shape == (6, 2)
        assert Bdot_func(q, qd).shape == (6, 2)

    def test_shape_spherical(self, single_spherical):
        vt, B_func, Bdot_func = single_spherical
        q = _identity_q_int(vt)
        qd = np.zeros(vt.total_dof)
        assert B_func(q).shape == (6, 3)
        assert Bdot_func(q, qd).shape == (6, 3)

    def test_shape_floating(self, single_floating):
        vt, B_func, Bdot_func = single_floating
        q = _identity_q_int(vt)
        qd = np.zeros(vt.total_dof)
        assert B_func(q).shape == (6, 6)
        assert Bdot_func(q, qd).shape == (6, 6)

    def test_shape_chain_rps(self, chain_rps):
        vt, B_func, Bdot_func = chain_rps
        q = _identity_q_int(vt)
        qd = np.zeros(vt.total_dof)
        expected = (6 * vt.NBodies, vt.total_dof)
        assert B_func(q).shape == expected
        assert Bdot_func(q, qd).shape == expected

    def test_shape_chain_ruc(self, chain_ruc):
        vt, B_func, Bdot_func = chain_ruc
        q = np.zeros(vt.total_cfg_dof)
        qd = np.zeros(vt.total_dof)
        expected = (6 * vt.NBodies, vt.total_dof)
        assert B_func(q).shape == expected
        assert Bdot_func(q, qd).shape == expected


# ============================================================
# Zero-speed sanity: Bdot(q, qd=0) == 0
# ============================================================

class TestZeroSpeed:
    """When all generalized speeds are zero, Bdot must vanish for R, P, U, C joints."""

    def test_revolute_zero_speed(self, single_revolute):
        vt, _, Bdot_func = single_revolute
        q = np.array([0.3])
        qd = np.zeros(vt.total_dof)
        np.testing.assert_allclose(Bdot_func(q, qd), 0.0, atol=1e-14)

    def test_prismatic_zero_speed(self, single_prismatic):
        vt, _, Bdot_func = single_prismatic
        q = np.array([0.5])
        qd = np.zeros(vt.total_dof)
        np.testing.assert_allclose(Bdot_func(q, qd), 0.0, atol=1e-14)

    def test_universal_zero_speed(self, single_universal):
        vt, _, Bdot_func = single_universal
        q = np.array([0.4, 0.2])
        qd = np.zeros(vt.total_dof)
        np.testing.assert_allclose(Bdot_func(q, qd), 0.0, atol=1e-14)

    def test_cylindrical_zero_speed(self, single_cylindrical):
        vt, _, Bdot_func = single_cylindrical
        q = np.array([0.6, 0.1])
        qd = np.zeros(vt.total_dof)
        np.testing.assert_allclose(Bdot_func(q, qd), 0.0, atol=1e-14)

    def test_spherical_zero_speed(self, single_spherical):
        vt, _, Bdot_func = single_spherical
        q = _identity_q_int(vt)
        qd = np.zeros(vt.total_dof)
        np.testing.assert_allclose(Bdot_func(q, qd), 0.0, atol=1e-14)

    def test_floating_zero_speed(self, single_floating):
        vt, _, Bdot_func = single_floating
        q = _identity_q_int(vt)
        qd = np.zeros(vt.total_dof)
        np.testing.assert_allclose(Bdot_func(q, qd), 0.0, atol=1e-14)

    def test_chain_ruc_zero_speed(self, chain_ruc):
        """Mixed chain at random q, but qd=0 → Bdot=0."""
        vt, _, Bdot_func = chain_ruc
        rng = np.random.default_rng(42)
        q = rng.standard_normal(vt.total_cfg_dof) * 0.3
        qd = np.zeros(vt.total_dof)
        np.testing.assert_allclose(Bdot_func(q, qd), 0.0, atol=1e-14)


# ============================================================
# Universal-joint axis transport regression
# ============================================================

class TestUniversalAxisTransport:
    """Verify that Udot[:,1] (second universal axis derivative)
    depends on the first universal rate qd1, not just on parent transport.

    If the implementation incorrectly uses only omega_parent to transport u2
    (instead of omega_parent + qd1*u1), this test will fail.
    """

    def test_second_axis_depends_on_first_rate(self, single_universal):
        vt, _, Bdot_func = single_universal
        q = np.array([0.0, 0.0])  # reference config

        # Case A: only qd2 active
        qd_a = np.array([0.0, 1.0])
        Bdot_a = Bdot_func(q, qd_a)

        # Case B: both qd1 and qd2 active
        qd_b = np.array([1.0, 1.0])
        Bdot_b = Bdot_func(q, qd_b)

        # If Udot[:,1] did NOT depend on qd1, Bdot_b == Bdot_a
        # since the only difference is qd1.
        # The correct implementation couples qd1 into u2_dot, so they differ.
        assert not np.allclose(Bdot_a, Bdot_b, atol=1e-14), (
            "Bdot is identical with and without first universal rate — "
            "axis-transport of u2 likely ignores qd1."
        )

    def test_first_axis_independent_of_second_rate(self, single_universal):
        """At reference config (ground is parent), Udot[:,0] = skew(omega_p)*u1 = 0
        regardless of qd2.  Varying qd2 should NOT affect the u1 column of Bdot.
        """
        vt, _, Bdot_func = single_universal
        q = np.array([0.0, 0.0])

        Bdot_0 = Bdot_func(q, np.array([1.0, 0.0]))
        Bdot_1 = Bdot_func(q, np.array([1.0, 2.0]))

        # Column 0 (u1 axis) should be the same in both
        np.testing.assert_allclose(Bdot_0[:, 0], Bdot_1[:, 0], atol=1e-14)


# ============================================================
# Finite-difference consistency
# ============================================================

class TestFiniteDifference:
    """Numerically verify  (B(q + eps*qd) - B(q)) / eps ≈ Bdot(q, qd).

    Only used for joint types where cfg DOF == speed DOF (R, P, U, C)
    so that qdot_cfg == qd and the finite difference is straightforward.
    """

    @staticmethod
    def _fd_Bdot(B_func, q, qd, eps=1e-7):
        """Forward finite-difference approximation of dB/dt."""
        B0 = B_func(q)
        B1 = B_func(q + eps * qd)
        return (B1 - B0) / eps

    def test_revolute_fd(self, single_revolute):
        vt, B_func, Bdot_func = single_revolute
        rng = np.random.default_rng(100)
        q = rng.standard_normal(vt.total_cfg_dof) * 0.5
        qd = rng.standard_normal(vt.total_dof) * 0.3

        Bdot_fd = self._fd_Bdot(B_func, q, qd)
        Bdot_an = Bdot_func(q, qd)
        np.testing.assert_allclose(Bdot_an, Bdot_fd, atol=1e-5)

    def test_prismatic_fd(self, single_prismatic):
        vt, B_func, Bdot_func = single_prismatic
        rng = np.random.default_rng(101)
        q = rng.standard_normal(vt.total_cfg_dof) * 0.5
        qd = rng.standard_normal(vt.total_dof) * 0.3

        Bdot_fd = self._fd_Bdot(B_func, q, qd)
        Bdot_an = Bdot_func(q, qd)
        np.testing.assert_allclose(Bdot_an, Bdot_fd, atol=1e-5)

    def test_universal_fd(self, single_universal):
        vt, B_func, Bdot_func = single_universal
        rng = np.random.default_rng(102)
        q = rng.standard_normal(vt.total_cfg_dof) * 0.5
        qd = rng.standard_normal(vt.total_dof) * 0.3

        Bdot_fd = self._fd_Bdot(B_func, q, qd)
        Bdot_an = Bdot_func(q, qd)
        np.testing.assert_allclose(Bdot_an, Bdot_fd, atol=1e-5)

    def test_cylindrical_fd(self, single_cylindrical):
        vt, B_func, Bdot_func = single_cylindrical
        rng = np.random.default_rng(103)
        q = rng.standard_normal(vt.total_cfg_dof) * 0.5
        qd = rng.standard_normal(vt.total_dof) * 0.3

        Bdot_fd = self._fd_Bdot(B_func, q, qd)
        Bdot_an = Bdot_func(q, qd)
        np.testing.assert_allclose(Bdot_an, Bdot_fd, atol=1e-5)

    def test_chain_ruc_fd(self, chain_ruc):
        """Mixed chain R+U+C: all joints have cfg DOF == speed DOF."""
        vt, B_func, Bdot_func = chain_ruc
        rng = np.random.default_rng(104)
        q = rng.standard_normal(vt.total_cfg_dof) * 0.4
        qd = rng.standard_normal(vt.total_dof) * 0.3

        Bdot_fd = self._fd_Bdot(B_func, q, qd)
        Bdot_an = Bdot_func(q, qd)
        np.testing.assert_allclose(Bdot_an, Bdot_fd, atol=1e-5)

    @pytest.mark.parametrize("seed", [200, 201, 202])
    def test_chain_ruc_fd_multi_seed(self, chain_ruc, seed):
        """Repeatability across seeds for the R+U+C chain."""
        vt, B_func, Bdot_func = chain_ruc
        rng = np.random.default_rng(seed)
        q = rng.standard_normal(vt.total_cfg_dof) * 0.5
        qd = rng.standard_normal(vt.total_dof) * 0.4

        Bdot_fd = self._fd_Bdot(B_func, q, qd)
        Bdot_an = Bdot_func(q, qd)
        np.testing.assert_allclose(Bdot_an, Bdot_fd, atol=1e-5)


# ============================================================
# Finite results at random state
# ============================================================

class TestFinite:
    """Bdot must be finite (no NaN / Inf) at random configurations."""

    @pytest.mark.parametrize("seed", [0, 1])
    def test_chain_rps_finite(self, chain_rps, seed):
        vt, _, Bdot_func = chain_rps
        rng = np.random.default_rng(seed)
        q = _identity_q_int(vt)
        q[:2] = rng.standard_normal(2) * 0.4
        quat = rng.standard_normal(4)
        quat /= np.linalg.norm(quat)
        sl = vt.q_slices[2]
        q[sl.start:sl.stop] = quat
        qd = rng.standard_normal(vt.total_dof) * 0.3
        Bdot = Bdot_func(q, qd)
        assert np.all(np.isfinite(Bdot))

    @pytest.mark.parametrize("seed", [10, 11])
    def test_floating_finite(self, single_floating, seed):
        vt, _, Bdot_func = single_floating
        rng = np.random.default_rng(seed)
        q = _identity_q_int(vt)
        q[:3] = rng.standard_normal(3) * 0.5
        quat = rng.standard_normal(4)
        quat /= np.linalg.norm(quat)
        q[3:7] = quat
        qd = rng.standard_normal(vt.total_dof) * 0.3
        Bdot = Bdot_func(q, qd)
        assert np.all(np.isfinite(Bdot))


# ============================================================
# Idempotency: same inputs give same Bdot
# ============================================================

class TestIdempotency:
    def test_same_inputs_same_output(self, chain_ruc):
        vt, _, Bdot_func = chain_ruc
        rng = np.random.default_rng(77)
        q = rng.standard_normal(vt.total_cfg_dof) * 0.3
        qd = rng.standard_normal(vt.total_dof) * 0.2
        Bdot_1 = Bdot_func(q, qd)
        Bdot_2 = Bdot_func(q, qd)
        np.testing.assert_array_equal(Bdot_1, Bdot_2)
