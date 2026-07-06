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
Force = {
    "Gravity": {
        "g_vec": [0, 0, -9.81],
        "mass": {1: 1.0, 2: 1.0},
    },
}

# Body-frame principal inertia tensors [kg·m²] — full 3×3 to allow non-diagonal entries.
# For a uniform rod of mass 1 kg, length 2 m: Ixx=Iyy≈mL²/12=0.333, Izz≈0 (thin rod).
J_body_imm = {
    1: [[0.333, 1.,    3.   ],
        [1.,    0.333, 0.   ],
        [3.,    0.,    0.01 ]],
    2: [[0.333, 1.33,    0.33   ],
        [1.33,    0.333, 0.   ],
        [0.33,    0.,    0.01 ]],
}