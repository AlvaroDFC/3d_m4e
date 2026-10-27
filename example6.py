# Universa; joint example

import numpy as np
import sympy as sym

sqrt = sym.sqrt

data = {
    "NBodies": 2,

    "joints": [
        (0, 1),
        (1, 2),
    ],

    "types": ["R", "U"],

    # parent CG -> joint, in parent frame
    "parent_cg_to_joint": [
        [0, 0, 0],            # ground -> joint(0-1), assumed zero
        [-0.2, -0.2, 1.0],    # s1j1_l
    ],

    # joint -> child CG, in child frame
    "joint_to_child_cg": [
        [0.5, 0.5, 0.5],      # s1_l
        [-0.5, -0.5, 0.0],    # s2_l
    ],

    # axis_u is required by from_data(), but only used for R/P/C joints
    "axis_u": [
        [-1/sqrt(3), -1/sqrt(3),  1/sqrt(3)],   # u1_l for R joint
        None,                                   # U joint does not use axis_u
    ],

    # universal joint axes
    "axis_u1": [
        None,
        [1/sqrt(3), 1/sqrt(3), 1/sqrt(3)],      # u2_l
    ],

    "axis_u2": [
        None,
        [1, 0, -1],                              # u3_l
    ],
    "rot_param": [None, None]  # not used for R/P/U joints
}

# Placeholder mass [kg] and isotropic inertia [kg·m²] for every body; replace with real values.
body_inertia = {b: {"mass": 1.0, "J": (0.01 * np.eye(3)).tolist()} for b in range(1, 3)}

# ── Initial conditions and numerical parameters ───────────────────────────────
# R: theta | U: alpha, beta.
q_user_np = np.array([np.pi/6, 0.3, 2.0])
qd_np     = np.array([0.7, -0.4, 1.1])

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