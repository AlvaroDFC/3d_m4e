# source/multibody_3d/tests/test_kinematics_cache_symbolic.py
"""
Tests for VelocityTransformation3D.build_cache_symbolic().

Verifies shapes, types, and structural properties of the symbolic
kinematics cache (no numeric evaluation).
"""
import pytest
import sympy as sym
from sympy import Identity, MatMul, MatrixSymbol

try:
    from multibody_3d import JointSystem3D, VelocityTransformation3D
    from multibody_3d.multibody_core.velocity_transformation_3d import KinematicsCache3D
except Exception:  # pragma: no cover
    import sys as _sys
    _sys.path.insert(0, ".")
    from source.multibody_3d import JointSystem3D, VelocityTransformation3D
    from source.multibody_3d.multibody_core.velocity_transformation_3d import KinematicsCache3D


def _make(data: dict) -> VelocityTransformation3D:
    return VelocityTransformation3D(JointSystem3D.from_data(data))


def _sym_q(n: int) -> sym.Matrix:
    return sym.Matrix([sym.Symbol(f"q{i}", real=True) for i in range(n)])


# ------------------------------------------------------------------ #
#  3-body chain:  0 -R-> 1 -P-> 2 -S-> 3
# ------------------------------------------------------------------ #
def _chain_RPS():
    z3 = [0, 0, 0]
    return {
        "NBodies": 3,
        "joints": [(0, 1), (1, 2), (2, 3)],
        "types": ["R", "P", "S"],
        "parent_cg_to_joint": [z3, z3, z3],
        "joint_to_child_cg": [z3, z3, z3],
        "axis_u": [[0, 0, 1], [0, 0, 1], None],
        "axis_u1": [None, None, None],
        "axis_u2": [None, None, None],
    }


class TestChainRPS:
    def setup_method(self):
        self.vt = _make(_chain_RPS())
        self.q = _sym_q(self.vt.total_cfg_dof)  # 1+1+4 = 6 (cfg)
        self.cache = self.vt.build_cache_symbolic(self.q)

    def test_return_type(self):
        assert isinstance(self.cache, KinematicsCache3D)

    def test_A_abs_lengths(self):
        assert len(self.cache.A_abs) == self.vt.NBodies + 1

    def test_r_abs_lengths(self):
        assert len(self.cache.r_abs) == self.vt.NBodies + 1

    def test_rJ_length(self):
        assert len(self.cache.rJ) == self.vt.NJoints

    def test_U_length(self):
        assert len(self.cache.U) == self.vt.NJoints

    def test_Arel_length(self):
        assert len(self.cache.Arel) == self.vt.NJoints

    def test_ground_A_is_identity(self):
        assert self.cache.A_abs[0] == Identity(3)

    def test_ground_r_is_zero(self):
        assert self.cache.r_abs[0] == sym.zeros(3, 1)

    def test_A_abs_shapes(self):
        for b in range(1, self.vt.NBodies + 1):
            A = self.cache.A_abs[b]
            assert A.shape == (3, 3), f"A_abs[{b}].shape = {A.shape}"

    def test_r_abs_shapes(self):
        for b in range(1, self.vt.NBodies + 1):
            r = self.cache.r_abs[b]
            assert r.shape == (3, 1), f"r_abs[{b}].shape = {r.shape}"

    def test_rJ_shapes(self):
        for j in range(self.vt.NJoints):
            assert self.cache.rJ[j].shape == (3, 1), f"rJ[{j}].shape = {self.cache.rJ[j].shape}"

    def test_U_shapes(self):
        expected = {0: (3, 1), 1: (3, 1), 2: (3, 3)}  # R, P, S
        for j, shape in expected.items():
            assert self.cache.U[j].shape == shape, f"U[{j}].shape = {self.cache.U[j].shape}"

    def test_Arel_are_MatrixSymbol(self):
        for j, ar in enumerate(self.cache.Arel):
            assert isinstance(ar, MatrixSymbol), f"Arel[{j}] is {type(ar)}"
            assert ar.shape == (3, 3)

    def test_A_abs_contains_MatMul(self):
        """Non-ground absolute rotations must be unevaluated MatMul products."""
        for b in range(1, self.vt.NBodies + 1):
            A = self.cache.A_abs[b]
            assert A.has(MatMul) or isinstance(A, MatMul), (
                f"A_abs[{b}] should contain MatMul, got {type(A)}"
            )

    def test_parent_of_body(self):
        assert self.cache.parent_of_body[0] == 0
        assert self.cache.parent_of_body[1] == 0
        assert self.cache.parent_of_body[2] == 1
        assert self.cache.parent_of_body[3] == 2

    def test_joint_of_body(self):
        assert self.cache.joint_of_body[0] == -1
        assert self.cache.joint_of_body[1] == 0
        assert self.cache.joint_of_body[2] == 1
        assert self.cache.joint_of_body[3] == 2


# ------------------------------------------------------------------ #
#  Single-body tests for each joint type
# ------------------------------------------------------------------ #
_SINGLE_BODY_CASES = [
    ("R", 1, (3, 1), {"axis_u": [0, 0, 1]}),
    ("P", 1, (3, 1), {"axis_u": [1, 0, 0]}),
    ("C", 2, (3, 1), {"axis_u": [0, 1, 0]}),
    ("U", 2, (3, 2), {"axis_u1": [1, 0, 0], "axis_u2": [0, 1, 0]}),
    ("S", 4, (3, 3), {}),
    ("F", 7, (3, 6), {}),
]


@pytest.mark.parametrize("jtype,ndof,u_shape,axes", _SINGLE_BODY_CASES,
                         ids=[c[0] for c in _SINGLE_BODY_CASES])
def test_single_body_shapes(jtype, ndof, u_shape, axes):
    data = {
        "NBodies": 1,
        "joints": [(0, 1)],
        "types": [jtype],
        "parent_cg_to_joint": [[0, 0, 0]],
        "joint_to_child_cg": [[0, 0, 0]],
        "axis_u": [axes.get("axis_u")],
        "axis_u1": [axes.get("axis_u1")],
        "axis_u2": [axes.get("axis_u2")],
    }
    vt = _make(data)
    q = _sym_q(ndof)
    cache = vt.build_cache_symbolic(q)

    assert cache.A_abs[1].shape == (3, 3)
    assert cache.r_abs[1].shape == (3, 1)
    assert cache.rJ[0].shape == (3, 1)
    assert cache.U[0].shape == u_shape, f"U[0].shape={cache.U[0].shape}, expected {u_shape}"


# ------------------------------------------------------------------ #
#  Branching tree: ground->1, ground->2  (two roots)
# ------------------------------------------------------------------ #
def test_branching_tree():
    z3 = [0, 0, 0]
    data = {
        "NBodies": 2,
        "joints": [(0, 1), (0, 2)],
        "types": ["R", "R"],
        "parent_cg_to_joint": [z3, z3],
        "joint_to_child_cg": [z3, z3],
        "axis_u": [[0, 0, 1], [1, 0, 0]],
        "axis_u1": [None, None],
        "axis_u2": [None, None],
    }
    vt = _make(data)
    q = _sym_q(2)
    cache = vt.build_cache_symbolic(q)

    # Both children have same parent (ground) Identity
    for b in (1, 2):
        A = cache.A_abs[b]
        assert A.shape == (3, 3)
        # Should be I3 * Arel_j  ->  MatMul(Identity(3), Arel_j)
        assert isinstance(A, MatMul)


# ------------------------------------------------------------------ #
#  q shape mismatch raises ValueError
# ------------------------------------------------------------------ #
def test_q_shape_mismatch_raises():
    vt = _make(_chain_RPS())
    bad_q = sym.Matrix([sym.Symbol("x")])
    with pytest.raises(ValueError, match="q shape mismatch"):
        vt.build_cache_symbolic(bad_q)
