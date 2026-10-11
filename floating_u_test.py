# Floating-base Universal joint example: bodyA is a fully free (F-joint to
# ground) floating body; bodyB is attached to bodyA via a UNIVERSAL joint.
# Used for a cross-validation conservation test (momentum/energy) against
# SEA-Stack/Chrono -- see the sibling SEA-Stack repo's
# demos/floating_u_test/ and demos/u_joint_test/floating_u_m4e.py.
import numpy as np

data = {
    "NBodies": 2,
    "joints": [(0, 1), (1, 2)],
    "types": ["F", "U"],
    "parent_cg_to_joint": [[0, 0, 0], [0.5, 0, 0]],
    "joint_to_child_cg": [[0, 0, 0], [0.5, 0, 0]],
    "axis_u": [None, None],
    "axis_u1": [None, [1, 0, 0]],
    "axis_u2": [None, [0, 1, 0]],
    "rot_param": ["quat", None],
}

# lambda*I (spherical) inertia -- matches demos/u_joint_test/floating_u_m4e.py
body_inertia = {
    1: {"mass": 1.0, "J": (0.6 * np.eye(3)).tolist()},
    2: {"mass": 1.0, "J": (0.5 * np.eye(3)).tolist()},
}
Force = {"Gravity": {"g_vec": [0.0, 0.0, -0.1]}}
