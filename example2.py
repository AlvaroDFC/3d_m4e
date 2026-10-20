# Example 2 - Define a simple double pendulum system 
import numpy as np

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

Initial_Points = {
    "GR": [],
    "BD": {1: [[0.0, 0.0, 0.0]], 2: [[0.0, 0.0, 0.0]]},
}

Force = {
    "Gravity": {
        "g_vec": [0, 0, -9.81],
        "g_app": [1, 1],
    },
    "TensionSpring": [(1, 0, 2, 0, 3.0, 0.6)],
    "TorsionSpring": [(1, 0.5, 0.0)],
}

# ── Initial conditions and numerical parameters ───────────────────────────────
# S joint (quat): e0..e3, then the revolute angle.
q_user_np = np.array([0.9659, 0.0, 0.0, 0.2588, -np.pi/12])
qd_np     = np.array([1.0, -2.0, 1.0, 1.0])

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
