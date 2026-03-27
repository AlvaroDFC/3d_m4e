"""Consistency tests for the JAX B / Bdot evaluators.

Compares ``evaluate_B_jax`` and ``evaluate_Bdot_jax`` against the
lambdified reference implementation (``compile_B_lambdified`` /
``compile_Bdot_lambdified``) across several topologies and joint-type
combinations.

Also verifies JIT compilation and basic autodiff readiness.
"""
from __future__ import annotations

import numpy as np
import sympy as sym
import pytest

from source.multibody_3d.multibody_core.joint_system_3d import JointSystem3D
from source.multibody_3d.multibody_core.velocity_transformation_3d import (
    VelocityTransformation3D,
)
from source.multibody_3d.multibody_core._velocity_transformation_helper import _type_code

jax = pytest.importorskip("jax")
jnp = pytest.importorskip("jax.numpy")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

Z3 = [0.0, 0.0, 0.0]


def _make_system(n_bodies, joints, types, p2j, j2c, u, u1=None, u2=None):
    if u1 is None:
        u1 = [None] * len(joints)
    if u2 is None:
        u2 = [None] * len(joints)
    data = {
        "NBodies": n_bodies,
        "joints": joints,
        "types": types,
        "parent_cg_to_joint": p2j,
        "joint_to_child_cg": j2c,
        "axis_u": u,
        "axis_u1": u1,
        "axis_u2": u2,
    }
    return JointSystem3D.from_data(data)


def _random_q_int(vt: VelocityTransformation3D, rng: np.random.Generator):
    """Generate a valid random q_int (unit quaternions for S/F joints)."""
    q = rng.standard_normal(vt.total_cfg_dof)
    for j_idx, jnt in enumerate(vt.joint_system.joints):
        code = _type_code(jnt.type)
        sl = vt.q_slices[j_idx]
        if code == "S":
            quat = q[sl.start:sl.start + 4]
            q[sl.start:sl.start + 4] = quat / np.linalg.norm(quat)
        elif code == "F":
            quat = q[sl.start + 3:sl.start + 7]
            q[sl.start + 3:sl.start + 7] = quat / np.linalg.norm(quat)
    return q


def _random_qd(vt: VelocityTransformation3D, rng: np.random.Generator):
    return rng.standard_normal(vt.total_dof)


def _build_B_ref(vt: VelocityTransformation3D):
    """Build a lambdified B reference callable."""
    q_syms = sym.Matrix(
        [sym.Symbol(f"qi{i}", real=True) for i in range(vt.total_cfg_dof)]
    )
    return vt.compile_B_lambdified(q_syms)


def _build_Bdot_ref(vt: VelocityTransformation3D):
    """Build a lambdified Bdot reference callable."""
    q_syms = sym.Matrix(
        [sym.Symbol(f"qi{i}", real=True) for i in range(vt.total_cfg_dof)]
    )
    qd_syms = sym.Matrix(
        [sym.Symbol(f"qdi{i}", real=True) for i in range(vt.total_dof)]
    )
    return vt.compile_Bdot_lambdified(q_syms, qd_syms)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def chain_RP():
    """2-body chain: R -> P with non-trivial geometry."""
    js = _make_system(
        n_bodies=2,
        joints=[(0, 1), (1, 2)],
        types=["R", "P"],
        p2j=[[0, 0, 1], [0, 1, 0]],
        j2c=[[0, 0, -1], [0, -1, 0]],
        u=[[0, 0, 1], [1, 0, 0]],
    )
    return VelocityTransformation3D(js)


@pytest.fixture
def chain_RPS():
    """3-body chain: R -> P -> S with zero geometry."""
    js = _make_system(
        n_bodies=3,
        joints=[(0, 1), (1, 2), (2, 3)],
        types=["R", "P", "S"],
        p2j=[Z3, Z3, Z3],
        j2c=[Z3, Z3, Z3],
        u=[[0, 0, 1], [0, 0, 1], None],
    )
    return VelocityTransformation3D(js)


@pytest.fixture
def chain_UCF():
    """3-body chain: U -> C -> F."""
    js = _make_system(
        n_bodies=3,
        joints=[(0, 1), (1, 2), (2, 3)],
        types=["U", "C", "F"],
        p2j=[[1, 0, 0], [0, 1, 0], Z3],
        j2c=[[-1, 0, 0], [0, -1, 0], Z3],
        u=[None, [0, 0, 1], None],
        u1=[[1, 0, 0], None, None],
        u2=[[0, 1, 0], None, None],
    )
    return VelocityTransformation3D(js)


@pytest.fixture
def branch_RR():
    """Branching: body 1 <- ground -> body 2, both R joints."""
    js = _make_system(
        n_bodies=2,
        joints=[(0, 1), (0, 2)],
        types=["R", "R"],
        p2j=[[0, 0, 1], [0, 1, 0]],
        j2c=[[0, 0, -0.5], [0, -0.5, 0]],
        u=[[0, 0, 1], [0, 1, 0]],
    )
    return VelocityTransformation3D(js)


ALL_FIXTURE_NAMES = ["chain_RP", "chain_RPS", "chain_UCF", "branch_RR"]


# ---------------------------------------------------------------------------
# B consistency tests
# ---------------------------------------------------------------------------

class TestEvaluateBJaxMatchesLambdified:
    """evaluate_B_jax must match compile_B_lambdified at random configurations."""

    RTOL = 1e-10
    ATOL = 1e-10

    @pytest.mark.parametrize("fixture_name", ALL_FIXTURE_NAMES)
    def test_random_configs(self, fixture_name, request):
        vt = request.getfixturevalue(fixture_name)
        B_ref_func = _build_B_ref(vt)
        rng = np.random.default_rng(42)

        for _ in range(5):
            q = _random_q_int(vt, rng)
            B_ref = B_ref_func(q)
            B_jax = np.asarray(vt.evaluate_B_jax(q))
            np.testing.assert_allclose(
                B_jax, B_ref, rtol=self.RTOL, atol=self.ATOL,
                err_msg=f"B mismatch on {fixture_name}",
            )


class TestEvaluateBJaxShape:
    """Basic shape and identity-config checks."""

    def test_shape(self, chain_RPS):
        vt = chain_RPS
        q = np.zeros(vt.total_cfg_dof)
        sl_s = vt.q_slices[2]
        q[sl_s.start] = 1.0
        B = np.asarray(vt.evaluate_B_jax(q))
        assert B.shape == (6 * vt.NBodies, vt.total_dof)


# ---------------------------------------------------------------------------
# Bdot consistency tests
# ---------------------------------------------------------------------------

class TestEvaluateBdotJaxMatchesLambdified:
    """evaluate_Bdot_jax must match compile_Bdot_lambdified at random configurations."""

    RTOL = 1e-10
    ATOL = 1e-10

    @pytest.mark.parametrize("fixture_name", ALL_FIXTURE_NAMES)
    def test_random_configs(self, fixture_name, request):
        vt = request.getfixturevalue(fixture_name)
        Bdot_ref_func = _build_Bdot_ref(vt)
        rng = np.random.default_rng(99)

        for _ in range(3):
            q  = _random_q_int(vt, rng)
            qd = _random_qd(vt, rng)
            Bdot_ref = Bdot_ref_func(q, qd)
            Bdot_jax = np.asarray(vt.evaluate_Bdot_jax(q, qd))
            np.testing.assert_allclose(
                Bdot_jax, Bdot_ref, rtol=self.RTOL, atol=self.ATOL,
                err_msg=f"Bdot mismatch on {fixture_name}",
            )


class TestEvaluateBdotJaxZeroSpeed:
    """At qd=0, Bdot must be zero (all rates vanish)."""

    def test_zero_speed(self, chain_RP):
        vt = chain_RP
        q  = np.zeros(vt.total_cfg_dof)
        qd = np.zeros(vt.total_dof)
        Bdot = np.asarray(vt.evaluate_Bdot_jax(q, qd))
        np.testing.assert_allclose(Bdot, 0.0, atol=1e-15)


# ---------------------------------------------------------------------------
# JIT compilation
# ---------------------------------------------------------------------------

class TestJITCompilation:
    """Verify that the JIT-compiled evaluators work and match."""

    RTOL = 1e-10
    ATOL = 1e-10

    @pytest.mark.parametrize("fixture_name", ALL_FIXTURE_NAMES)
    def test_jit_B(self, fixture_name, request):
        vt = request.getfixturevalue(fixture_name)
        params = vt.build_numeric_params()
        B_eval = vt.build_B_evaluator_jax(params)
        B_ref  = _build_B_ref(vt)

        rng = np.random.default_rng(7)
        q = _random_q_int(vt, rng)

        B_jit1 = np.asarray(B_eval(q))
        B_jit2 = np.asarray(B_eval(q))
        B_lam  = B_ref(q)
        np.testing.assert_allclose(B_jit1, B_lam,  rtol=self.RTOL, atol=self.ATOL)
        np.testing.assert_allclose(B_jit2, B_jit1, rtol=0, atol=0)

    @pytest.mark.parametrize("fixture_name", ALL_FIXTURE_NAMES)
    def test_jit_Bdot(self, fixture_name, request):
        vt = request.getfixturevalue(fixture_name)
        params = vt.build_numeric_params()
        Bdot_eval = vt.build_Bdot_evaluator_jax(params)
        Bdot_ref  = _build_Bdot_ref(vt)

        rng = np.random.default_rng(11)
        q  = _random_q_int(vt, rng)
        qd = _random_qd(vt, rng)

        Bdot_jit = np.asarray(Bdot_eval(q, qd))
        Bdot_lam = Bdot_ref(q, qd)
        np.testing.assert_allclose(Bdot_jit, Bdot_lam, rtol=self.RTOL, atol=self.ATOL)


# ---------------------------------------------------------------------------
# Autodiff smoke test
# ---------------------------------------------------------------------------

class TestAutodiff:
    """Verify JAX autodiff runs without error on B evaluator."""

    def test_jacobian_B_runs(self, chain_RP):
        """jax.jacobian(B)(q) should return a tensor without error."""
        vt = chain_RP
        params = vt.build_numeric_params()
        B_eval = vt.build_B_evaluator_jax(params)

        q = jnp.zeros(vt.total_cfg_dof)
        J = jax.jacobian(B_eval)(q)
        assert J.shape == (6 * vt.NBodies, vt.total_dof, vt.total_cfg_dof)

    def test_jacobian_Bdot_runs(self, chain_RP):
        """jax.jacobian(Bdot)(q, qd) should return without error."""
        vt = chain_RP
        params = vt.build_numeric_params()
        Bdot_eval = vt.build_Bdot_evaluator_jax(params)

        q  = jnp.zeros(vt.total_cfg_dof)
        qd = jnp.zeros(vt.total_dof)
        # Jacobian w.r.t. first argument (q_int)
        J = jax.jacobian(Bdot_eval, argnums=0)(q, qd)
        assert J.shape == (6 * vt.NBodies, vt.total_dof, vt.total_cfg_dof)

