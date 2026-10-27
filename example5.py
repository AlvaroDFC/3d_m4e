import numpy as np
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

# Placeholder mass [kg] and isotropic inertia [kg·m²] for every body; replace with real values.
body_inertia = {b: {"mass": 1.0, "J": (0.01 * np.eye(3)).tolist()} for b in range(1, 5)}

Initial_Points = {
    "GR": [],
    "BD": {1: [[0.0, 0.0, 0.0]], 4: [[0.0, 0.0, 0.0]]},
}

Force = {
    "TensionSpring": [(1, 0, 4, 0, 1.0, 3.5)],
    "TorsionSpring": [(1, 0.2, 0.0)],
}

# ── Initial conditions and numerical parameters ───────────────────────────────
q_user_np = np.array([np.pi/6, 0.3, 2.0, 1.1])
qd_np     = np.array([0.7, -0.3, 0.4, 1.2])

body_data_sym = {}
force_sym     = {}
points_sym    = {}

body_params   = np.ones(len(body_data_sym))
force_params  = np.ones(len(force_sym))
points_params = np.ones(len(points_sym))

# ── Simulation ─────────────────────────────────────
tspan             = 10.0
TimeStep          = 0.05
integrator_kwargs = {"rtol": 1e-12, "atol": 1e-12, "algorithm": "Dopri8",
                     "max_steps": 10_000_000, "dtmax": 0.01}