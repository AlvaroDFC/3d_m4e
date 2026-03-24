# tests/test_joint_coords_3d.py
from __future__ import annotations

import inspect
from types import SimpleNamespace
from typing import Any

from multibody_3d import build_joint_coordinates


def _jt(code: str):
    from multibody_3d import JointType
    try:
        return JointType(code)
    except Exception:
        for m in JointType:
            if m.name == code:
                return m
            try:
                if str(m.value) == code:
                    return m
            except Exception:
                continue
        raise


def _dof_from_code(code: str) -> int:
    return {"R": 1, "P": 1, "S": 3, "U": 2, "C": 2, "F": 6}[code]


def _instantiate_dataclass_like(cls: type, **seed: Any) -> Any:
    sig = inspect.signature(cls)
    kwargs: dict[str, Any] = {}
    for pname, p in sig.parameters.items():
        if pname in seed:
            kwargs[pname] = seed[pname]
            continue
        if p.default is not inspect._empty:
            continue

        low = pname.lower()
        if "axis" in low:
            kwargs[pname] = (0.0, 0.0, 1.0)
        elif ("parent" in low and "joint" in low) or ("cg_to_joint" in low) or ("com_to_joint" in low):
            kwargs[pname] = (0.0, 0.0, 0.0)
        elif ("joint" in low and "child" in low) or ("joint_to_cg" in low) or ("joint_to_com" in low):
            kwargs[pname] = (0.0, 0.0, 0.0)
        else:
            kwargs[pname] = None
    return cls(**kwargs)


def _make_test_system():
    try:
        from multibody_3d import Joint3D, JointSystem3D  # type: ignore
        have_real = True
    except Exception:
        Joint3D = None  # type: ignore
        JointSystem3D = None  # type: ignore
        have_real = False

    # Types: [C, R, S] with children 1,2,3
    codes = ["C", "R", "S"]
    j_types = [_jt(c) for c in codes]
    parents = [0, 1, 2]
    childs = [1, 2, 3]

    if have_real and Joint3D is not None:
        joints = [
            _instantiate_dataclass_like(Joint3D, parent=p, child=c, type=jt)
            for (p, c, jt) in zip(parents, childs, j_types, strict=True)
        ]
    else:
        class _Joint:
            def __init__(self, parent: int, child: int, jt: Any, code: str):
                self.parent = parent
                self.child = child
                self.type = jt
                self._code = code

            def dof(self) -> int:
                return _dof_from_code(self._code)

        joints = [_Joint(p, c, jt, code) for (p, c, jt, code) in zip(parents, childs, j_types, codes, strict=True)]

    joints = sorted(joints, key=lambda j: j.child)
    dofs = [int(j.dof()) for j in joints]

    col_slice = []
    k = 0
    for d in dofs:
        col_slice.append(slice(k, k + d))
        k += d
    total_dof = k

    # Internal config DOFs (quaternion for S/F)
    _cfg_dof_map = {"R": 1, "P": 1, "S": 4, "U": 2, "C": 2, "F": 7}
    cfg_dofs = [_cfg_dof_map[c] for c in codes]
    cfg_col_slice = []
    ck = 0
    for d in cfg_dofs:
        cfg_col_slice.append(slice(ck, ck + d))
        ck += d

    # User config DOFs (euler default for S/F)
    user_dofs = dofs  # same as speed DOF when rot_param="euler"
    user_col_slice = list(col_slice)

    if have_real and JointSystem3D is not None:
        try:
            sys = _instantiate_dataclass_like(JointSystem3D, NBodies=3, joints=joints)
        except Exception:
            sys = SimpleNamespace(NBodies=3, joints=joints)
    else:
        sys = SimpleNamespace(NBodies=3, joints=joints)

    sys.col_slice = col_slice
    sys.total_dof = total_dof
    sys.cfg_col_slice = cfg_col_slice
    sys.total_cfg_dof = ck
    sys.user_col_slice = user_col_slice
    sys.total_user_dof = total_dof
    return sys


def test_build_joint_coords_3d_shapes_names_and_slices():
    sys = _make_test_system()
    bundle = build_joint_coordinates(sys)

    assert bundle.q.shape == (sys.total_dof, 1)
    assert bundle.qd.shape == (sys.total_dof, 1)

    assert bundle.names == ["q1_theta", "q1_s", "q2_theta", "q3_phi", "q3_theta", "q3_psi"]
    assert bundle.names_d == ["qd1_theta", "qd1_sd", "qd2_theta", "qd3_phi", "qd3_theta", "qd3_psi"]

    assert len(bundle.per_joint) == len(sys.joints) == len(sys.col_slice)
    for i, pj in enumerate(bundle.per_joint):
        assert pj["joint_index"] == i
        assert pj["slice"] == sys.col_slice[i]
        slc = pj["slice"]
        assert bundle.names[slc.start:slc.stop] == pj["names"]
        assert bundle.names_d[slc.start:slc.stop] == pj["names_d"]

    # Cylindrical ordering must be [theta, s]
    assert bundle.per_joint[0]["names"] == ["q1_theta", "q1_s"]
    assert bundle.per_joint[0]["names_d"] == ["qd1_theta", "qd1_sd"]