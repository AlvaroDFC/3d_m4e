# Vertical 2-body chain example: bodyA is a fully free (F-joint to ground)
# floating body, bodyB hangs below it via a UNIVERSAL joint, both initially
# aligned along world Z. Used to test whether giving the floating body a
# pure Z angular velocity (no linear velocity, no relative U-joint rate)
# produces genuinely 3D motion under asymmetric inertia, and as an
# independent cross-check of the energy/momentum conservation findings --
# see the sibling SEA-Stack repo's demos/vertical_chain_test/ and
# demos/u_joint_test/vertical_chain_m4e.py.
import numpy as np

data = {
    "NBodies": 2,
    "joints": [(0, 1), (1, 2)],
    "types": ["F", "U"],
    "parent_cg_to_joint": [[0, 0, 0], [0, 0, -0.5]],
    "joint_to_child_cg": [[0, 0, 0], [0, 0, -0.5]],
    "axis_u": [None, None],
    "axis_u1": [None, [1, 0, 0]],
    "axis_u2": [None, [0, 1, 0]],
    "rot_param": ["quat", None],
}

# Asymmetric inertia, as in the SEA-Stack cross-validation; mass in kg, inertia in kg·m².
body_inertia = {
    1: {"mass": 1.0, "J": [[0.8, 0.1, 0.02], [0.1, 0.6, 0.05], [0.02, 0.05, 0.4]]},
    2: {"mass": 1.0, "J": [[0.6, 0.08, 0.03], [0.08, 0.5, 0.04], [0.03, 0.04, 0.35]]},
}

# ── Initial conditions and numerical parameters ───────────────────────────────
# bodyA spins about world Z only (2 rad/s); no linear velocity and no U-joint rate.
q_user_np = np.array([0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0])
qd_np     = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 2.0, 0.0, 0.0])

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
