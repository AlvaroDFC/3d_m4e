# Example: a floating body with 3 revolute joints (OMT device)
#
# Copy of example4.py -- SAME topology/geometry ("the problem"), with system
# parameters (mass, inertia, gravity, initial conditions) ADDED so this can
# be integrated directly via MbdSystem3D.from_example(example4_copy), instead
# of needing an external body_inertia/Force dict supplied by the caller (as
# example4.py itself declares none -- it is a kinematics-only example there).
#
# These parameters exactly match the SEA-Stack (Chrono) cross-validation demo:
#   SEA-Stack-Sahand-Binaries/demos/Example4/floating_tristar.model.yaml
#   SEA-Stack-Sahand-Binaries/demos/Example4/floating_tristar.simulation.yaml
#   SEA-Stack-Sahand-Binaries/demos/Example4/example4_m4e_common.py
#
# Bodies (M4E 1-based id -> SEA-Stack YAML body name):
#   1 = floating base (free in space)          -> "hub"
#   2 = link 1 (revolute w.r.t. body 1)         -> "arm1"
#   3 = link 2 (revolute w.r.t. body 1)         -> "arm2"
#   4 = link 3 (revolute w.r.t. body 1)         -> "arm3"

import numpy as np
# Dimensions:
R = 1.5   # radius of circular arrangement of revolute joints
L = 2.0   # length of each link to the buoy

data = {
    "NBodies": 4,
    "joints": [
        (0, 1),  # ground -> floating base (free)
        (1, 2),  # floating base -> link1 (revolute)
        (1, 3),  # floating base -> link2 (revolute)
        (1, 4),  # floating base -> link3 (revolute)
    ],
    "types": ["F", "R", "R", "R"],
    "parent_cg_to_joint": [
        [0.0, 0.0, 0.0],     # joint(0->1) at ground origin
        [R, 0.0, 0.0],       # joint(1->2) at (R, 0, 0) in floating base frame
        [R*np.cos(2*np.pi/3), R*np.sin(2*np.pi/3), 0.0],   # joint(1->3) at (-R/2, R*sqrt(3)/2, 0) in floating base frame
        [R*np.cos(4*np.pi/3), R*np.sin(4*np.pi/3), 0.0],  # joint(1->4) at (-R/2, -R*sqrt(3)/2, 0) in floating base frame
    ],
    "joint_to_child_cg": [
        [0.0, 0.0, 0.0],     # floating base CG coincident with its joint (simplest)
        [L/2, 0.0, 0.0],      # link1 CG is L/2 along its length from its revolute joint
        [L/2*np.cos(2*np.pi/3), L/2*np.sin(2*np.pi/3), 0.0],      # link2 CG is L/2 along its length from its revolute joint rotated 120 degrees about z from link1
        [L/2*np.cos(4*np.pi/3), L/2*np.sin(4*np.pi/3), 0.0],      # link3 CG is L/2 along its length from its revolute joint rotated 240 degrees about z from link1
    ],
    "axis_u": [
        None,                # Free: no axis required
        [0.0, 1.0, 0.0],     # Revolute axis along +y (links rotate in x–z plane)
        [-np.sin(2*np.pi/3), np.cos(2*np.pi/3), 0.0],     # Revolute axis along +y + 120 degrees (links rotate in x–y plane)
        [-np.sin(4*np.pi/3), np.cos(4*np.pi/3), 0.0],     # Revolute axis along +y + 240 degrees (links rotate in x–y plane)
    ],
    "axis_u1": [None, None, None, None],
    "axis_u2": [None, None, None, None],
    "rot_param": ['euler', None, None, None]
}

# ---------------------------------------------------------------------------
# System (mass/inertia/gravity) parameters -- ADDED here, matching the
# SEA-Stack floating_tristar demo. example4.py itself declares none of this.
# ---------------------------------------------------------------------------
# Per-body masses [kg] and body-frame inertia tensors [kg·m²], in the
# hub's/ground-aligned frame (this example's joints have no per-child frame
# rotation baked in -- link2/link3's data vectors above are expressed
# directly in the floating base's own frame -- so the off-axis arms need a
# FULL tensor, not just a diagonal one).
#   hub (body 1): uniform disk, mass 500 kg, radius 1.6 m, height 0.5 m:
#     Ixx=Iyy (diametral) = 330.4, Izz (vertical symmetry axis) = 640.0.
#   arm1/2/3 (bodies 2-4): uniform slender rod, mass 10 kg, length 2 m,
#     radius 0.05 m, pointing radially outward at 0/120/240 deg azimuth:
#     I_perp (normal to rod) = 3.34, I_axial (along rod) = 0.0125. Rotating
#     the diagonal (0.0125, 3.34, 3.34) tensor by each arm's azimuth about Z
#     gives arm2/arm3's off-diagonal terms below.
body_inertia = {
    1: {
        "mass": 5.0,
        "J": [[3.4, 0.,      0.  ],
              [0.,    3.4,   0.  ],
              [0.,    0.,      6.0]],
    },
    2: {
        "mass": 10.0,
        "J": [[0.0125, 0.,   0.  ],
              [0.,      3.34, 0.  ],
              [0.,      0.,   3.34]],
    },
    3: {
        "mass": 10.0,
        "J": [[2.508125,  1.440850, 0.  ],
              [1.440850,  0.844375, 0.  ],
              [0.,        0.,       3.34]],
    },
    4: {
        "mass": 10.0,
        "J": [[2.508125, -1.440850, 0.  ],
              [-1.440850, 0.844375, 0.  ],
              [0.,        0.,       3.34]],
    },
}

# Real gravity -- matches floating_tristar.simulation.yaml's gravity: [0, 0, -9.81].
Force = {
    "Gravity": {"g_vec": [0.0, 0.0, 0.0]}, #-9.81
}

# ---------------------------------------------------------------------------
# Initial conditions -- matching the SEA-Stack demo: hub (and arm2, arm3) at
# rest; arm1 given an angular velocity of 1 rad/s about its revolute axis
# (the requested "end of chain" IC). q_int/qd are NOT consumed by
# MbdSystem3D.from_example() -- they're provided here purely so a user/script
# can build mainNumVars the same way main.py does for the other examples:
#
#   import example4_copy as ex
#   mbd = MbdSystem3D.from_example(ex)
#   q_user = mbd.map_q_int_to_q_user(ex.q_int_ic)
#   mainNumVars = np.concatenate([q_user, ex.qd_ic])
#   sol = mbd.integrate(mainNumVars, tspan=(0.0, ex.END_TIME), dt=ex.DT, ...)
# ---------------------------------------------------------------------------
# OMEGA0 = 1.0  # arm1 initial angular rate about its revolute axis [rad/s]

# # Hub (F joint, quaternion internal): identity orientation, at rest.
# # rot_param='euler' for the F joint above -- q_int stays quaternion
# # internally regardless; map to q_user via MbdSystem3D.map_q_int_to_q_user.
# q_int_ic = np.array([0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
# qd_ic = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, OMEGA0, 0.0, 0.0])

# END_TIME = 2.0   # matches floating_tristar.simulation.yaml's end_time
# DT = 0.005       # matches floating_tristar.simulation.yaml's time_step

# --- CURRENT IC: hub (the free body) given 1 rad/s yaw (global Z), instead of
# arm1's earlier revolute-axis spin above. qd's hub slots are [vx,vy,vz,wx,wy,wz]
# (F-joint DOF = hub's own absolute linear/angular velocity) -- no complementary
# linear velocity is needed here (unlike arm1's case): hub's CG coincides with
# its own reference frame (com.location = [0,0,0]), so spinning about its own
# axis induces no CG translation. arm1/arm2/arm3 (last 3 qd slots) stay at rest.
q_int_ic = np.array([0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
qd_ic = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0])

END_TIME = 10.0  # matches floating_tristar.simulation.yaml's end_time
DT = 0.005       # matches floating_tristar.simulation.yaml's time_step
