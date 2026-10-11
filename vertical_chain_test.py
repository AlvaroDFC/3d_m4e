# Vertical 2-body chain example: bodyA is a fully free (F-joint to ground)
# floating body, bodyB hangs below it via a UNIVERSAL joint, both initially
# aligned along world Z. Used to test whether giving the floating body a
# pure Z angular velocity (no linear velocity, no relative U-joint rate)
# produces genuinely 3D motion under asymmetric inertia, and as an
# independent cross-check of the energy/momentum conservation findings --
# see the sibling SEA-Stack repo's demos/vertical_chain_test/ and
# demos/u_joint_test/vertical_chain_m4e.py.
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
