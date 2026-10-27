# Example 1 - Define a simple double pendulum system
import numpy as np

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

# Placeholder mass [kg] and isotropic inertia [kg·m²] for every body; replace with real values.
body_inertia = {b: {"mass": 1.0, "J": (0.01 * np.eye(3)).tolist()} for b in range(1, 4)}

Initial_Points = {
    "GR": [],
    "BD": {2: [[0.0, 0.0, 0.0]], 3: [[0.0, 0.0, 0.0]]},
}

Force = {
    "TensionSpring": [(2, 0, 3, 0, 2.0, 1.6)],
    "TorsionSpring": [(1, 0.3, 0.0)],
}

# ── Initial conditions and numerical parameters ───────────────────────────────
q_user_np = np.array([0.1, 0.2, 0.3])
qd_np     = np.array([1.0, 3.0, 3.0])

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
