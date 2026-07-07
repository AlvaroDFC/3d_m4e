# Example: a floating body with 3 revolute joints (OMT device) using symbolic parameters
# Bodies:
#   1 = floating base (free in space)
#   2 = link 1 (revolute w.r.t. body 1)
#   3 = link 2 (revolute w.r.t. body 1)
#   4 = link 3 (revolute w.r.t. body 1)
#
# ── Force contract ────────────────────────────────────────────────────────────
# Optional ``Force`` dictionary keys (all may be combined):
#
#   "CG"           : direct force/moment at body CG (world frame)
#   "PointsBD"     : force at a declared body point (creates induced CG moment)
#   "TensionSpring": linear spring between two declared body/ground points
#   "TensionDamper": linear damper between two declared body/ground points
#   "TorsionSpring": torsional spring at a revolute joint
#   "TorsionDamper": torsional damper at a revolute joint
#   "Gravity"      : uniform gravity applied at each body CG
#
# All constitutive parameters (k, L0, c, …) and body masses may be SymPy
# symbols.  They must be collected in ``force_sym`` (or the legacy
# ``force_points_sym``) so they appear in ``mainSymVars`` / ``mainNumVars``.
#
# Example (shown in Prompt 5 tests):
#
#   import sympy as sym
#   k, L0 = sym.symbols('k L0', positive=True)
#   Force = {
#       'TensionSpring': [(2, 0, 4, 0, k, L0)],
#       'Gravity':       {'g_vec': [0, 0, -9.81]},
#   }
#   force_sym = {'k': k, 'L0': L0}
# ─────────────────────────────────────────────────────────────────────────────

import sympy as sym
# Dimensions using symbolic variables for later substitution:
R, L = sym.symbols("R L", real=True)

# Gather all symbolic variables in a dictionary for later use
body_data_sym = {
    "R": R,
    "L": L,
}
    
data = {
    "NBodies": 4,
    "joints": [
        (0, 1),  # ground -> floating base (free)
        (1, 2),  # floating base -> link1 (revolute)
        (2, 3),  # link1 -> link2 (revolute)
        (3, 4),  # link2 -> link3 (revolute)
    ],
    "types": ["F", "S", "R", "R"],
    "parent_cg_to_joint": [
        [0.0, 0.0, 0.0],     # joint(0->1) at ground origin
        [R, 0.0, 0.0],       # joint(1->2) at (R, 0, 0) in floating base frame
        [R*sym.cos(2*sym.pi/3), R*sym.sin(2*sym.pi/3), 0.0],   # joint(2->3) at (-R/2, R*sqrt(3)/2, 0) in floating base frame
        [R*sym.cos(4*sym.pi/3), R*sym.sin(4*sym.pi/3), 0.0],  # joint(3->4) at (-R/2, -R*sqrt(3)/2, 0) in floating base frame
    ],
    "joint_to_child_cg": [
        [0.0, 0.0, 0.0],     # floating base CG coincident with its joint (simplest)
        [L/2, 0.0, 0.0],      # link1 CG is L/2 along its length from its revolute joint
        [L/2*sym.cos(2*sym.pi/3), L/2*sym.sin(2*sym.pi/3), 0.0],      # link2 CG is L/2 along its length from its revolute joint rotated 120 degrees about z from link1
        [L/2*sym.cos(4*sym.pi/3), L/2*sym.sin(4*sym.pi/3), 0.0],      # link3 CG is L/2 along its length from its revolute joint rotated 240 degrees about z from link1
    ],
    "axis_u": [
        None,                # Free: no axis required
        None,     # Revolute axis along +y (links rotate in x–z plane)
        [-sym.sin(2*sym.pi/3), sym.cos(2*sym.pi/3), 0.0],     # Revolute axis along +y + 120 degrees (links rotate in x–y plane)
        [-sym.sin(4*sym.pi/3), sym.cos(4*sym.pi/3), 0.0],     # Revolute axis along +y + 240 degrees (links rotate in x–y plane)
    ],
    "axis_u1": [None, None, None, None],
    "axis_u2": [None, None, None, None],
    "rot_param": ['euler', 'euler', None, None]
}

# ── Point definitions ─────────────────────────────────────────────────────────
# Points are expressed as 3-element lists [x, y, z].  Components may be
# numeric literals or symbolic SymPy expressions; any free symbols that
# should participate in the canonical variable vector must be listed in
# ``points_sym`` (ordered, user-declared order is preserved).

# Parameterised offset along the link-3 body axis (symbolic endpoint location).
d4 = sym.Symbol("d4", real=True)

# Flat ordered dict: symbolic-name → SymPy symbol.
# Only symbols that actually appear in Initial_Points entries belong here.
points_sym = {
    "d4": d4,
}

Initial_Points = {
    # Grounded (world-frame) reference points.
    # ``"GR"`` holds a list of points; each point is [x, y, z] in the world frame.
    "GR": [
        [0.0, 0.0, 0.0],            # world-origin anchor (numeric)
    ],
    # Body-local points, keyed by integer body id (1-based, excluding ground).
    # Coordinates are expressed in the body's own reference frame at the CG.
    "BD": {
        2: [[L / 2, 0.0, 0.0]],     # link-1 tip in body-2 frame  (L = 2.0 → 1.0 m)
        3: [[L / 2, 0.0, 0.0]],     # link-2 tip in body-3 frame
        4: [[d4,    0.0, 0.0]],     # link-3 endpoint; d4 is a symbolic parameter
    },
}

# ── Force definitions ─────────────────────────────────────────────────────────
# Symbolic constitutive parameters.
# Any symbol that appears inside ``Force`` must be listed here so it is
# included in ``mainSymVars`` / ``mainNumVars`` in the declared order.
Fx1, Fy1, Fz1 = sym.symbols("Fx1 Fy1 Fz1", real=True)

force_sym = {
    "Fx1": Fx1,   # x-component of CG force on floating base (body 1)
    "Fy1": Fy1,   # y-component
    "Fz1": Fz1,   # z-component
}

Force = {
    # Direct force applied at the floating-base CG (body 1), world frame.
    # Components are symbolic so the caller controls magnitude and direction
    # via the force slice of mainNumVars.
    "CG": {
        1: {"force": [Fx1, Fy1, Fz1]},
    },
}
