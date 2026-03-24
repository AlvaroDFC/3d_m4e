# Example: a floating body with 3 revolute joints (OMT device)
# Bodies:
#   1 = floating base (free in space)
#   2 = link 1 (revolute w.r.t. body 1)
#   3 = link 2 (revolute w.r.t. body 1)
#   4 = link 3 (revolute w.r.t. body 1)

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
    "rot_param": ['quat', None, None, None]
}