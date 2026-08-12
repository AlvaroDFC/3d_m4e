# Example 2 - Define a simple double pendulum system 
data = {
        "NBodies": 2,
        "joints": [(0, 1), (1, 2)],
        "types": ["S", "R"],
        "parent_cg_to_joint": [[0, 0, 0], [0.5, 0.5, -0.5]],
        "joint_to_child_cg": [[0.5, 0.5, -0.5], [-0.5, -0.5, -0.5]],
        "axis_u": [None, [0, 1, 0]],
        "axis_u1": [None, None],
        "axis_u2": [None, None],
        "rot_param": ['quat', None],
    }

# Per-body masses [kg] and body-frame inertia tensors [kg·m²].
body_inertia = {
    1: {
        "mass": 3.0,
        "J": [[1.2,  0.15, 0.  ],
              [0.15, 0.9,  0.  ],
              [0.,   0.,   0.6 ]],
    },
    2: {
        "mass": 1.5,
        "J": [[0.6,   0.075, 0.  ],
              [0.075, 0.45,  0.  ],
              [0.,    0.,    0.3 ]],
    },
}

Force = {
    "Gravity": {
        "g_vec": [0, 0, -9.81],
        "g_app": [1, 1],
    }
}
