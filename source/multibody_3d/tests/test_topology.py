# tests/test_joint_system_3d.py
import pytest
import sympy as sym

from multibody_3d import JointSystem3D


def test_valid_small_tree_double_pendulum():
    data = {
        "NBodies": 2,
        "joints": [(0, 1), (1, 2)],
        "types": ["R", "R"],
        "parent_cg_to_joint": [[0, 0, 0], [0, 0, 0]],
        "joint_to_child_cg": [[0, 0, 0], [0, 0, 0]],
        "axis_u": [[0, 0, 1], [0, 0, 1]],
        "axis_u1": [None, None],
        "axis_u2": [None, None],
    }
    sys = JointSystem3D.from_data(data)

    assert sys.total_dof == 2
    assert sys.col_start == [0, 1]
    assert [s.start for s in sys.col_slice] == [0, 1]
    assert [s.stop for s in sys.col_slice] == [1, 2]

    # topology
    assert sys.parent_body_of_body[1] == 0
    assert sys.parent_body_of_body[2] == 1
    assert sys.roots == [1]
    assert sys.body_paths == [[1, 2]]
    assert sys.joint_paths == [[0, 1]]

    # Btrack: body 1 affected by joint0; body2 affected by joint0 and joint1
    assert sys.Btrack[1, 0] == True
    assert sys.Btrack[1, 1] == False
    assert sys.Btrack[2, 0] == True
    assert sys.Btrack[2, 1] == True


def test_invalid_duplicate_child():
    data = {
        "NBodies": 2,
        "joints": [(0, 1), (0, 1)],  # duplicate child=1, missing child=2
        "types": ["R", "R"],
        "parent_cg_to_joint": [[0, 0, 0], [0, 0, 0]],
        "joint_to_child_cg": [[0, 0, 0], [0, 0, 0]],
        "axis_u": [[0, 0, 1], [0, 0, 1]],
        "axis_u1": [None, None],
        "axis_u2": [None, None],
    }
    with pytest.raises(ValueError) as e:
        JointSystem3D.from_data(data)
    assert "Duplicate child body" in str(e.value) or "Missing child body" in str(e.value)


def test_invalid_cycle():
    # 2-body cycle: 2->1 and 1->2
    data = {
        "NBodies": 2,
        "joints": [(2, 1), (1, 2)],
        "types": ["R", "R"],
        "parent_cg_to_joint": [[0, 0, 0], [0, 0, 0]],
        "joint_to_child_cg": [[0, 0, 0], [0, 0, 0]],
        "axis_u": [[0, 0, 1], [0, 0, 1]],
        "axis_u1": [None, None],
        "axis_u2": [None, None],
    }
    with pytest.raises(ValueError) as e:
        JointSystem3D.from_data(data)
    assert "Cycle detected" in str(e.value)


def test_invalid_parent_out_of_range():
    data = {
        "NBodies": 1,
        "joints": [(2, 1)],  # parent=2 out of range
        "types": ["R"],
        "parent_cg_to_joint": [[0, 0, 0]],
        "joint_to_child_cg": [[0, 0, 0]],
        "axis_u": [[0, 0, 1]],
        "axis_u1": [None],
        "axis_u2": [None],
    }
    with pytest.raises(ValueError) as e:
        JointSystem3D.from_data(data)
    assert "out of range" in str(e.value)


def test_universal_axes_collinear_rejected():
    data = {
        "NBodies": 1,
        "joints": [(0, 1)],
        "types": ["U"],
        "parent_cg_to_joint": [[0, 0, 0]],
        "joint_to_child_cg": [[0, 0, 0]],
        "axis_u": [None],
        "axis_u1": [[1, 0, 0]],
        "axis_u2": [[2, 0, 0]],  # collinear with u1
    }
    with pytest.raises(ValueError) as e:
        JointSystem3D.from_data(data)
    assert "collinear" in str(e.value)