# Floating-base Universal joint example: bodyA is a fully free (F-joint to
# ground) floating body; bodyB is attached to bodyA via a UNIVERSAL joint.
# Used for a cross-validation conservation test (momentum/energy) against
# SEA-Stack/Chrono -- see the sibling SEA-Stack repo's
# demos/floating_u_test/ and demos/u_joint_test/floating_u_m4e.py.
import numpy as np

data = {
    "NBodies": 2,
    "joints": [(0, 1), (1, 2)],
    "types": ["F", "U"],
    "parent_cg_to_joint": [[0, 0, 0], [0.5, 0, 0]],
    "joint_to_child_cg": [[0, 0, 0], [0.5, 0, 0]],
    "axis_u": [None, None],
    "axis_u1": [None, [1, 0, 0]],
    "axis_u2": [None, [0, 1, 0]],
    "rot_param": ["quat", None],
}

# lambda*I (spherical) inertia -- matches demos/u_joint_test/floating_u_m4e.py
body_inertia = {
    1: {"mass": 1.0, "J": (0.6 * np.eye(3)).tolist()},
    2: {"mass": 1.0, "J": (0.5 * np.eye(3)).tolist()},
}
Force = {"Gravity": {"g_vec": [0.0, 0.0, -0.1]}}

# ── Initial conditions and numerical parameters ───────────────────────────────
# F (quat): x, y, z, e0..e3 | U: alpha, beta.  Matches the SEA-Stack cross-validation IC.
q_user_np = np.array([0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0])
qd_np     = np.array([0.2, 0.3, 0.0, 0.3, 0.4, 0.5, 0.6, 0.8])

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
