# Example 1 - Define a simple double pendulum system
L = 1.0 # Length of the main rotor blades
data = {
        "NBodies": 3,
        "joints": [(0, 1), (1, 2),(1,3)],
        "types": ["R", "R", "R"],
        "parent_cg_to_joint": [[0, 0, 0], [0, -L, 0.0], [0, L, 0.0]],
        "joint_to_child_cg": [[0, 0, 0], [0, 0, 0.1], [0, 0, -0.1]],
        "axis_u": [[1, 0, 0], [0, 0, 1], [0, 0, -1]],
        "axis_u1": [None, None, None],
        "axis_u2": [None, None, None],
        "rot_param": [None, None, None]
    }
