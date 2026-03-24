# source/multibody_3d/tests/test_assemble_B_symbolic.py
"""Tests for VelocityTransformation3D.assemble_B_symbolic()."""
import pytest
import sympy as sym

try:
    from multibody_3d import JointSystem3D, VelocityTransformation3D
    from multibody_3d.multibody_core.velocity_transformation_3d import skew
except Exception:  # pragma: no cover
    import sys as _sys
    _sys.path.insert(0, ".")
    from source.multibody_3d import JointSystem3D, VelocityTransformation3D
    from source.multibody_3d.multibody_core.velocity_transformation_3d import skew


def _make(data: dict) -> VelocityTransformation3D:
    return VelocityTransformation3D(JointSystem3D.from_data(data))


def _sym_q(n: int) -> sym.Matrix:
    return sym.Matrix([sym.Symbol(f"q{i}", real=True) for i in range(n)])


# ------------------------------------------------------------------ #
#  2-body revolute chain: 0 -R-> 1 -R-> 2
# ------------------------------------------------------------------ #
def _chain_RR():
    return {
        "NBodies": 2,
        "joints": [(0, 1), (1, 2)],
        "types": ["R", "R"],
        "parent_cg_to_joint": [[0, 0, 0], [1, 0, 0]],
        "joint_to_child_cg": [[1, 0, 0], [1, 0, 0]],
        "axis_u": [[0, 0, 1], [0, 0, 1]],
        "axis_u1": [None, None],
        "axis_u2": [None, None],
    }


class TestAssembleBShape:
    def setup_method(self):
        self.vt = _make(_chain_RR())
        self.q = _sym_q(self.vt.total_dof)
        self.B = self.vt.assemble_B_symbolic(self.q)

    def test_shape(self):
        NB = self.vt.NBodies
        expected = (6 * NB, self.vt.total_dof)
        assert self.B.shape == expected

    def test_body1_joint0_block_nonzero(self):
        """Body 1 should have a nonzero block from joint 0 (its own joint)."""
        block = self.B[0:6, 0:1]
        assert block != sym.zeros(6, 1)

    def test_body2_joint0_block_nonzero(self):
        """Body 2 is downstream of joint 0, so B[body2, joint0_cols] must be nonzero."""
        block = self.B[6:12, 0:1]
        assert block != sym.zeros(6, 1)

    def test_body2_joint1_block_nonzero(self):
        """Body 2's own joint (joint 1) must produce a nonzero block."""
        block = self.B[6:12, 1:2]
        assert block != sym.zeros(6, 1)

    def test_body1_joint1_block_zero(self):
        """Body 1 is NOT downstream of joint 1, so that block stays zero."""
        block = self.B[0:6, 1:2]
        assert block == sym.zeros(6, 1)


# ------------------------------------------------------------------ #
#  Idempotent: calling twice produces same result
# ------------------------------------------------------------------ #
def test_assemble_twice_same_result():
    vt = _make(_chain_RR())
    q = _sym_q(vt.total_dof)
    B1 = vt.assemble_B_symbolic(q)
    B2 = vt.assemble_B_symbolic(q)
    assert B1 == B2


# ------------------------------------------------------------------ #
#  Passing pre-built cache produces same result
# ------------------------------------------------------------------ #
def test_assemble_with_cache():
    vt = _make(_chain_RR())
    q = _sym_q(vt.total_dof)
    cache = vt.build_cache_symbolic(q)
    B1 = vt.assemble_B_symbolic(q)
    B2 = vt.assemble_B_symbolic(q, cache=cache)
    assert B1 == B2


# ------------------------------------------------------------------ #
#  3-body chain R-P-S: shapes + Btrack pattern
# ------------------------------------------------------------------ #
def test_chain_RPS_shape_and_pattern():
    z3 = [0, 0, 0]
    data = {
        "NBodies": 3,
        "joints": [(0, 1), (1, 2), (2, 3)],
        "types": ["R", "P", "S"],
        "parent_cg_to_joint": [z3, z3, z3],
        "joint_to_child_cg": [z3, z3, z3],
        "axis_u": [[0, 0, 1], [0, 0, 1], None],
        "axis_u1": [None, None, None],
        "axis_u2": [None, None, None],
    }
    vt = _make(data)
    q = _sym_q(vt.total_cfg_dof)  # 1+1+4 = 6 (cfg)
    B = vt.assemble_B_symbolic(q)

    # Shape: 6*3 x 5 = 18 x 5 (no ground)
    assert B.shape == (18, 5)

    # Body 3 is downstream of all 3 joints -> all column groups nonzero
    # Joint 0 cols: 0:1, Joint 1 cols: 1:2, Joint 2 cols: 2:5
    assert B[12:18, 0:1] != sym.zeros(6, 1)   # body3, joint0 (R)
    assert B[12:18, 1:2] != sym.zeros(6, 1)   # body3, joint1 (P)
    assert B[12:18, 2:5] != sym.zeros(6, 3)   # body3, joint2 (S)

    # Body 1 is NOT downstream of joint 1 or 2
    assert B[0:6, 1:2] == sym.zeros(6, 1)
    assert B[0:6, 2:5] == sym.zeros(6, 3)


# ------------------------------------------------------------------ #
#  Branching: ground->1, ground->2  (independent branches)
# ------------------------------------------------------------------ #
def test_branching_independent_bodies():
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
    B = vt.assemble_B_symbolic(q)

    assert B.shape == (12, 2)  # 6*2 x 2 (no ground)

    # Body 1 affected only by joint 0 (col 0)
    assert B[0:6, 0:1] != sym.zeros(6, 1)
    assert B[0:6, 1:2] == sym.zeros(6, 1)

    # Body 2 affected only by joint 1 (col 1)
    assert B[6:12, 0:1] == sym.zeros(6, 1)
    assert B[6:12, 1:2] != sym.zeros(6, 1)
