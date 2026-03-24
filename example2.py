# Example 2 - Define a simple double pendulum system 
data = {
        "NBodies": 2,
        "joints": [(0, 1), (1, 2)],
        "types": ["S", "R"],
        "parent_cg_to_joint": [[0, 0, 0], [0.5, 0.5, -0.5]],
        "joint_to_child_cg": [[0.5, 0.5, -0.5], [-0.5, -0.5, -0.5]],
        "axis_u": [None, [1, 1, 0]],
        "axis_u1": [None, None],
        "axis_u2": [None, None],
        "rot_param": ['quat', None],
    }