import sympy as sym

sqrt = sym.sqrt

data = {
    "NBodies": 4,

    # edges: parent -> child
    "joints": [
        (0, 1),
        (1, 2),
        (1, 3),
        (3, 4),
    ],

    "types": ["R", "R", "P", "R"],

    # parent CG -> joint, in parent frame
    "parent_cg_to_joint": [
        [0, 0, 0],                 # ground -> joint(0-1), assumed zero
        [-0.2, -0.2, 1.0],         # s1j1_l
        [0.5, 0.0, 0.5],           # s1j2_l
        [0.0, 0.0, 0.7],           # s3j4_l
    ],

    # joint -> child CG, in child frame
    "joint_to_child_cg": [
        [0.5, 0.5, 0.5],           # s1_l
        [-0.5, -0.5, 0.0],         # s2_l
        [0.0, 1.0, 0.5],           # s3_l
        [0.1, 0.0, 0.5],           # s4_l
    ],

    # axis in parent frame, required for R/P/C joints
    "axis_u": [
        [-1/sqrt(3), -1/sqrt(3),  1/sqrt(3)],   # u0_l
        [ 1/sqrt(3),  1/sqrt(3),  1/sqrt(3)],   # u1_l
        [ 1,          0,          0],           # u2_l
        [ 0,          2/sqrt(5),  1/sqrt(5)],   # u3_l
    ],
    "axis_u1": [None, None, None, None],  # not used for R/P joints
    "axis_u2": [None, None, None, None],  # not used
    "rot_param": [None, None, None, None]  # not used for R/P joints
}