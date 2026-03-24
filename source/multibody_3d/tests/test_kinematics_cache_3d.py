# source/multibody/tests/test_kinematics_cache_3d.py
import numpy as np

try:
    # package import (expected in src-layout)
    from multibody_3d import VelocityTransformationKinematics3D
except Exception:  # pragma: no cover
    from kinematics_cache_3d import VelocityTransformationKinematics3D  # type: ignore

try:
    from multibody_3d import JointSystem3D
except Exception:  # pragma: no cover
    import sys
    sys.path.append("/mnt/data")
    from joint_system_3d import JointSystem3D  # type: ignore


def _make_sys(data: dict) -> JointSystem3D:
    return JointSystem3D.from_data(data)


def test_two_link_revolute_chain_positions_and_shapes():
    # ground -> 1 -> 2, both revolute about +z.
    data = {
        "NBodies": 2,
        "joints": [(0, 1), (1, 2)],
        "types": ["R", "R"],
        # vectors in local frames
        "parent_cg_to_joint": [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]],
        "joint_to_child_cg": [[1.0, 0.0, 0.0], [1.0, 0.0, 0.0]],
        "axis_u": [[0.0, 0.0, 1.0], [0.0, 0.0, 1.0]],
        "axis_u1": [None, None],
        "axis_u2": [None, None],
    }

    sys = _make_sys(data)
    kin = VelocityTransformationKinematics3D(sys)

    # theta1=theta2=0
    cache = kin.build_cache([0.0, 0.0])

    assert cache.A[0].shape == (3, 3)
    assert np.allclose(cache.A[0], np.eye(3))
    assert np.allclose(cache.r[0], np.zeros(3))

    assert cache.NBodies == 2
    assert cache.NJoints == 2
    assert len(cache.A) == 3
    assert len(cache.r) == 3
    assert len(cache.rJ) == 2
    assert len(cache.U) == 2

    # Straight along x: r1=[1,0,0], rJ1=[2,0,0], r2=[3,0,0]
    assert np.allclose(cache.r[1], np.array([1.0, 0.0, 0.0]))
    assert np.allclose(cache.rJ[0], np.array([0.0, 0.0, 0.0]))
    assert np.allclose(cache.rJ[1], np.array([2.0, 0.0, 0.0]))
    assert np.allclose(cache.r[2], np.array([3.0, 0.0, 0.0]))

    # U shapes for revolute
    assert cache.U[0].shape == (3, 1)
    assert cache.U[1].shape == (3, 1)

    # theta1=90deg, theta2=0 => chain points along +y
    cache2 = kin.build_cache([np.pi / 2.0, 0.0])
    assert np.allclose(cache2.r[1], np.array([0.0, 1.0, 0.0]), atol=1e-12)
    assert np.allclose(cache2.r[2], np.array([0.0, 3.0, 0.0]), atol=1e-12)


def test_U_shapes_for_each_joint_type_single_body():
    # Each case is a single joint: ground -> 1.
    cases = [
        ("R", 1, (3, 1), {"axis_u": [0.0, 0.0, 1.0]}),
        ("P", 1, (3, 1), {"axis_u": [1.0, 0.0, 0.0]}),
        ("C", 2, (3, 1), {"axis_u": [0.0, 1.0, 0.0]}),
        ("U", 2, (3, 2), {"axis_u1": [1.0, 0.0, 0.0], "axis_u2": [0.0, 1.0, 0.0]}),
        ("S", 3, (3, 3), {}),
        ("F", 6, (3, 6), {}),
    ]

    for jtype, ndof, Ushape, axes in cases:
        data = {
            "NBodies": 1,
            "joints": [(0, 1)],
            "types": [jtype],
            "parent_cg_to_joint": [[0.0, 0.0, 0.0]],
            "joint_to_child_cg": [[0.0, 0.0, 0.0]],
            "axis_u": [axes.get("axis_u", None)],  # required key in from_data
            "axis_u1": [axes.get("axis_u1") if "axis_u1" in axes else None],
            "axis_u2": [axes.get("axis_u2") if "axis_u2" in axes else None],
        }

        sys = _make_sys(data)
        kin = VelocityTransformationKinematics3D(sys)

        q = np.zeros(ndof)
        cache = kin.build_cache(q)

        assert cache.U[0].shape == Ushape
        assert cache.A[1].shape == (3, 3)
        assert cache.r[1].shape == (3,)