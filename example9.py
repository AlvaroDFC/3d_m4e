# Example 9 - Define a simple double pendulum system
L = 1.0 # Length of the main rotor blades
data = {
        "NBodies": 2,
        "joints": [(0, 1), (1, 2)],
        "types": ["R", "R"],
        "parent_cg_to_joint": [[0, 0, 0], [0, 0, -L]],
        "joint_to_child_cg": [[0, 0, -L], [0, 0, -L]],
        "axis_u": [[0, 1, 0], [1, 0, 0]],
        "axis_u1": [None, None],
        "axis_u2": [None, None],
        "rot_param": [None, None]
    }
