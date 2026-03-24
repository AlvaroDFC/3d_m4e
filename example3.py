# Example: Cylindrical (ground->slider) along +x,
#          then a pendulum link via Revolute,
#          then another link via Spherical (3-DOF) off the first link.
#
# Bodies:
#   1 = slider carriage (cylindrical w.r.t. ground)
#   2 = pendulum link 1 (revolute w.r.t. body 1)
#   3 = pendulum link 2 (spherical w.r.t. body 2)
#
# Conventions:
# - parent_cg_to_joint is expressed in PARENT frame
# - joint_to_child_cg is expressed in CHILD frame
# - joint axes are expressed in PARENT frame

L2 = 1.0   # length of pendulum link 1
L3 = 0.8   # length of pendulum link 2

data = {
    "NBodies": 3,
    "joints": [
        (0, 1),  # ground -> slider (cylindrical)
        (1, 2),  # slider -> link1 (revolute)
        (2, 3),  # link1  -> link2 (spherical)
    ],
    "types": ["C", "R", "S"],
    "parent_cg_to_joint": [
        [0.25, 0.25, 0.0],     # joint(0->1) at ground origin
        [0.0, 0.0, 0.25],     # joint(1->2) at slider CG (simplest)
        [0.0, 0.0, L2/2],   # joint(2->3) at distal end of link1, in link1 frame
    ],
    "joint_to_child_cg": [
        [0.0, 0.0, 0.125],     # slider CG coincident with its joint (simplest)
        [0.0, 0.0, L2/2],   # link1 CG is halfway down link1 from the revolute joint
        [0.0, 0.0, L3/2],   # link2 CG is halfway down link2 from the spherical joint
    ],
    "axis_u": [
        [1.0, 0.0, 0.0],     # Cylindrical axis along +x (translation + rotation about x)
        [0.0, 1.0, 0.0],     # Revolute axis along +y (pendulum swings in x–z plane)
        None,                # Spherical: no axis required
    ],
    "axis_u1": [None, None, None],
    "axis_u2": [None, None, None],
    "rot_param": [None, None, 'quat']
}