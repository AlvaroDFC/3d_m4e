# source/multibody_3d/tests/test_block_B.py
"""Tests for skew() and VelocityTransformation3D._block_B()."""
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


# ------------------------------------------------------------------ #
#  skew()
# ------------------------------------------------------------------ #
class TestSkew:
    def test_shape(self):
        v = sym.Matrix([1, 2, 3])
        assert skew(v).shape == (3, 3)

    def test_antisymmetric(self):
        a, b, c = sym.symbols("a b c")
        v = sym.Matrix([a, b, c])
        S = skew(v)
        assert S + S.T == sym.zeros(3)

    def test_cross_product(self):
        """skew(v) * w == v x w."""
        v = sym.Matrix([1, 2, 3])
        w = sym.Matrix([4, 5, 6])
        assert skew(v) * w == v.cross(w)

    def test_bad_shape_raises(self):
        with pytest.raises(ValueError, match="expected"):
            skew(sym.Matrix([1, 2]))


# ------------------------------------------------------------------ #
#  _block_B shapes per joint type
# ------------------------------------------------------------------ #
_CASES = [
    ("R", 1, (6, 1), {"axis_u": [0, 0, 1]}),
    ("P", 1, (6, 1), {"axis_u": [1, 0, 0]}),
    ("C", 2, (6, 2), {"axis_u": [0, 1, 0]}),
    ("U", 2, (6, 2), {"axis_u1": [1, 0, 0], "axis_u2": [0, 1, 0]}),
    ("S", 4, (6, 3), {}),
    ("F", 7, (6, 6), {}),
]


@pytest.mark.parametrize("jtype,ndof,expected_shape,axes", _CASES,
                         ids=[c[0] for c in _CASES])
def test_block_B_shape(jtype, ndof, expected_shape, axes):
    z3 = [0, 0, 0]
    data = {
        "NBodies": 1,
        "joints": [(0, 1)],
        "types": [jtype],
        "parent_cg_to_joint": [z3],
        "joint_to_child_cg": [z3],
        "axis_u": [axes.get("axis_u")],
        "axis_u1": [axes.get("axis_u1")],
        "axis_u2": [axes.get("axis_u2")],
    }
    vt = _make(data)
    joint = vt.joint_system.joints[0]

    # Build a symbolic d vector and U from cache
    q = sym.Matrix([sym.Symbol(f"q{i}") for i in range(ndof)])
    cache = vt.build_cache_symbolic(q)
    d = sym.Matrix(sym.symbols("dx dy dz"))
    U_j = cache.U[0]

    # For types with opaque MatMul U, convert to explicit matrix for block
    U_explicit = sym.Matrix(U_j) if not isinstance(U_j, sym.Matrix) else U_j

    block = vt._block_B(joint, d, U_explicit)
    assert block.shape == expected_shape, f"{jtype}: got {block.shape}, expected {expected_shape}"


def test_spherical_block_content():
    """For S joint, block should be [[-skew(d)], [I]]."""
    z3 = [0, 0, 0]
    data = {
        "NBodies": 1,
        "joints": [(0, 1)],
        "types": ["S"],
        "parent_cg_to_joint": [z3],
        "joint_to_child_cg": [z3],
        "axis_u": [None],
        "axis_u1": [None],
        "axis_u2": [None],
    }
    vt = _make(data)
    joint = vt.joint_system.joints[0]
    d = sym.Matrix(sym.symbols("dx dy dz"))

    block = vt._block_B(joint, d, sym.eye(3))  # U_j unused for S
    expected = sym.Matrix.vstack(-skew(d), sym.eye(3))
    assert block == expected


# ------------------------------------------------------------------ #
#  Revolute block content sanity
# ------------------------------------------------------------------ #
def test_revolute_block_content():
    """For R joint with u=[0,0,1], block should be [[-skew(d)*u],[u]]."""
    z3 = [0, 0, 0]
    data = {
        "NBodies": 1,
        "joints": [(0, 1)],
        "types": ["R"],
        "parent_cg_to_joint": [z3],
        "joint_to_child_cg": [z3],
        "axis_u": [[0, 0, 1]],
        "axis_u1": [None],
        "axis_u2": [None],
    }
    vt = _make(data)
    joint = vt.joint_system.joints[0]
    u = sym.Matrix([0, 0, 1])
    d = sym.Matrix(sym.symbols("dx dy dz"))

    block = vt._block_B(joint, d, u)

    expected_top = -skew(d) * u
    expected_bot = u
    expected = sym.Matrix.vstack(expected_top, expected_bot)
    assert block == expected


# ------------------------------------------------------------------ #
#  Prismatic block content sanity
# ------------------------------------------------------------------ #
def test_prismatic_block_content():
    """For P joint with u=[1,0,0], block should be [[u],[0]]."""
    z3 = [0, 0, 0]
    data = {
        "NBodies": 1,
        "joints": [(0, 1)],
        "types": ["P"],
        "parent_cg_to_joint": [z3],
        "joint_to_child_cg": [z3],
        "axis_u": [[1, 0, 0]],
        "axis_u1": [None],
        "axis_u2": [None],
    }
    vt = _make(data)
    joint = vt.joint_system.joints[0]
    u = sym.Matrix([1, 0, 0])
    d = sym.Matrix(sym.symbols("dx dy dz"))

    block = vt._block_B(joint, d, u)
    expected = sym.Matrix.vstack(u, sym.zeros(3, 1))
    assert block == expected


# ------------------------------------------------------------------ #
#  Floating block content sanity
# ------------------------------------------------------------------ #
def test_floating_block_content():
    """For F joint, block should be [[I, -skew(d)], [0, I]]."""
    z3 = [0, 0, 0]
    data = {
        "NBodies": 1,
        "joints": [(0, 1)],
        "types": ["F"],
        "parent_cg_to_joint": [z3],
        "joint_to_child_cg": [z3],
        "axis_u": [None],
        "axis_u1": [None],
        "axis_u2": [None],
    }
    vt = _make(data)
    joint = vt.joint_system.joints[0]
    d = sym.Matrix(sym.symbols("dx dy dz"))
    # U_j unused for F (block ignores it), pass dummy
    U_dummy = sym.eye(3)

    block = vt._block_B(joint, d, U_dummy)
    I3 = sym.eye(3)
    S = skew(d)
    expected = sym.Matrix.vstack(
        sym.Matrix.hstack(I3, -S),
        sym.Matrix.hstack(sym.zeros(3), I3),
    )
    assert block == expected
