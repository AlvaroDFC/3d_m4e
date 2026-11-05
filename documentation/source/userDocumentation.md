# 3D multibody dynamics
**For questions, contact:**
**sahand.sabet@nlr.gov, alvaro.diazflorescaminero@nlr.gov** 


This is the standalone 3D user guide. It follows the example-first structure of
the 2D documentation, but explains the 3D inputs and conventions directly; no
2D background is required. The related
[2D repository](https://github.com/Project-SEA-Stack/Python_Multibody_for_Everybody)
provides useful reference material, while the
[3D repository](https://github.com/AlvaroDFC/3d_m4e) is the source for the code
and examples described here.
The 3D code is available in the [M4E 3D GitHub repository](https://github.com/AlvaroDFC/3d_m4e).
To simulate a 3D mechanism, we first describe its bodies and how they are connected (at the initial configuration of the system).
Then we choose where it starts, which forces act on it, and how long to simulate it.
M4E uses those inputs to simulate the motion and to generate the equations of motion.

We use two Python files. The **example file** contains your model and its numerical
values. The **main file** imports that example, runs it, and plots the results. To run
a different model, change the example import in the main file.

The examples below follow the same six steps:

| Step | Your decision | Where it goes |
|---|---|---|
| 1. Connect the bodies | Which body connects to which, where, and through what joint and vectors? | `data` |
| 2. Choose attachment points | Where on each body will a spring or point force attach? | `Initial_Points` |
| 3. Add forces | What acts on a body or connects two attachment points? | `Force` |
| 4. Supply numerical values | What are the masses, initial positions, speeds, and simulation settings? | `body_inertia`, initial-condition and parameter arrays, time settings |
| 5. Run and inspect | What motion and energy does the model produce? | The main file |

Read Example 1 first: it explains each step using numbers only. Example 2 shows how
to replace selected numbers with adjustable parameters. Example 3 shows how to combine
different joint types. Sections 3-7 are a reference to return to as you build your own
model. Section 8 is optional: it explains how to inspect the generated equations. External tools can be integrated with M4E using the same approach discussed in the 2D.

> The adapter pattern is conceptually shared, but the current 3D repository does not include the 2D FSM, hydrodynamic, mooring, or linearization integrations; see the scope notes in section 9 (these features will be added in future releases).

**First draw the reference geometry.** Use a drawing in which all joint rotations
are zero and the body-fixed axes are parallel to the global X, Y, Z axes. This lets you
read the geometry vectors directly from the drawing. Here **CG** means a body's
center of gravity, and a **body frame** is a set of body-fixed axes that moves with that body.

**Starting pose is a separate input.**  Initial conditions describe how the mechanism is posed
and moving when the simulation starts. To start a hinge at a nonzero angle, change
its initial angle; do not redraw the fixed geometry vectors for that angle.

## Constructing a body from its parent

Build the mechanism one joint at a time, starting at ground (or a floating body). For each joint,
draw three things in the zero-coordinate pose (initial configuration): the parent's CG, the joint point,
and the child's CG. Then enter the two arrows between them (for the bodies connected to ground, the ground CG has no physical meaning, and vector connecting parent' CG to joint defines the location of your first joint):

1. **Parent CG to joint:** enter this arrow as `parent_cg_to_joint`.
2. **Joint to child CG:** enter this arrow as `joint_to_child_cg`.

For example, imagine two links hanging straight down. The ground hinge is at the
world origin. Body 1's CG is 0.5 m below that hinge. The next hinge is 0.7 m
below body 1's CG, and body 2's CG is 0.5 m below the second hinge. In the
zero-coordinate pose, the geometry rows are:

| Joint | Parent to child | `parent_cg_to_joint` | `joint_to_child_cg` |
|---|---|---|---|
| 0 | Ground (0) to body 1 | `[0, 0, 0]` | `[0, 0, -0.5]` |
| 1 | Body 1 to body 2 | `[0, 0, -0.7]` | `[0, 0, -0.5]` |

The first row puts the ground hinge at the origin and body 1's CG half a metre
below it. The second row places the next hinge half a metre below body 1's CG,
then body 2's CG half a metre below that hinge. You do not enter a separate
world position for either body; M4E builds the positions by following the joint
connections and adding these two offsets.

Read both arrows from the drawing with all joint coordinates set to zero. At
that pose the body axes are parallel to the world axes, so you can read the
XYZ components directly. The parent-to-joint arrow is attached to the parent
and uses its frame; the joint-to-child arrow is attached to the child and uses
its frame. As the bodies rotate, each arrow turns with its own body. The arrows
describe the fixed shape of the mechanism; initial conditions describe how it
is posed and moving when the simulation starts.

Finally, enter the joint type and its axis in the same row. Axes are described
in the parent frame: use `axis_u` for `R`, `P`, and `C` joints, and `axis_u1`
and `axis_u2` for a `U` joint. `S` joints have no single axis. A floating `F`
joint also allows free translation; see the floating-joint section below for
its coordinates.




## 1 - What changes from 2D to 3D

### 1.1 - The same example-writing workflow

The example file describes the problem; the main file describes how to run it. Keep the
initial conditions and simulation settings in the example, just as in the 2D examples.


| Item | 2D | 3D |
|---|---|---|
| System class | `MbdSystem` | `MbdSystem3D` |
| Joint and geometry input | Separate lists | Lists collected in `data` |
| Position components | `[x, z]` | `[x, y, z]` |
| Force and moment components | `[Fx, Fz, My]` | Three force and three moment components |
| Body-data symbols | `BodyDataSym` | `body_data_sym` |
| Force and point symbols | Combined in `ForcesPointsSym` | Separate `force_sym` and `points_sym` dictionaries |
| Initial conditions | `ic`, containing positions and speeds | `q_user_np` and `qd_np` |
| Mass and inertia | `m0` and `J0` | `body_inertia` |
| Gravity | `g` and `gVec` | `Force["Gravity"]` |
| Simulation settings | `tspan`, `TimeStep`, optional `integrator_kwargs` | The same example-level names |

### 1.2 - New joint types and rotation inputs

Revolute (`R`), prismatic (`P`), and floating (`F`) joints are still available. The 3D
version also provides spherical (`S`), universal (`U`), and cylindrical (`C`) joints.
A spherical or floating joint can accept Euler angles or a quaternion through
`rot_param`. We will explain both choices after the examples.

### 1.3 - What this section covers

The examples below cover rigid-body motion, points, applied loads, springs, dampers,
gravity, symbolic parameters, integration, and time-series plots. The 2D FSM generator and animation instructions are not currently available in
the 3D package (they will be added in future releases). The current 3D plotting helpers show time histories.

## 2 - Quick examples overview

The examples increase in detail as you work through them. The first uses only numerical
values, the second introduces a floating body and symbols, and the third combines
different joint types.

The tutorial example numbers below are the order in this chapter, not the numbered
Python files in the repository. They correspond directly to `example9.py`, `example7.py`,
and `example3.py`, respectively. The values in each tutorial example match its repository
input.

### Example 1: A 3D double pendulum with springs

This system has two bodies. The first is connected to ground by a revolute joint about
the world Y axis. The second rotates about the first body's X axis. Unlike the planar
double pendulum, the two hinge axes are not parallel.

```{figure} _static/figs/3d_tutorial_double_pendulum.png
:width: 75%
:align: center

Tutorial Example 1: two revolute joints and the body CGs. Dashed connections indicate
the joints-to-CG geometry; the springs are defined below, not drawn as physical rods.
```

#### **1 - Reference frames**

Use a right-handed world frame with Z pointing upward. In this example gravity acts
in the negative Z direction. All lengths are in metres and all angles are in radians.
There is no `Reference_frame_Origin` input in the current 3D example format. A ground
joint's `parent_cg_to_joint` gives its world position because ground is body 0.

#### **2 - Bodies definition**

Number the moving bodies 1 and 2; reserve number 0 for ground. We want this chain:

```text
ground (0) -- revolute joint --> body 1 -- revolute joint --> body 2
```

```python
import numpy as np

L = 1.0

data = {
    "NBodies": 2,
    "joints": [(0, 1), (1, 2)], 
    "types": ["R", "R"],
    "parent_cg_to_joint": [[0, 0, 0], [0, 0, -L]],
    "joint_to_child_cg": [[0, 0, -L], [0, 0, -L]],
    "axis_u": [[0, 1, 0], [1, 0, 0]], # axis for revolute joints
    "axis_u1": [None, None], # axis for prismatic joint 
    "axis_u2": [None, None],# axis for prismatic joint 
    "rot_param": [None, None], # which format: "euler" or "quat" for orientation 
}

body_data_sym = {}
```

Read these inputs as a set of drawing instructions:

| Joint index | Connect | Locate the joint from the parent CG | Locate the child CG from the joint | Hinge direction |
|---|---|---|---|---|
| 0 | Ground to body 1 | At the world origin: `[0,0,0]` | One metre down: `[0,0,-L]` | Global Y: `[0,1,0]` |
| 1 | Body 1 to body 2 | One metre down from body 1's CG: `[0,0,-L]` | Another metre down: `[0,0,-L]` | Body 1's X: `[1,0,0]` |

With   `L=1`, body 1's CG is at `[0,0,-1]`, the second hinge
is at `[0,0,-2]`, and body 2's CG is at `[0,0,-3]`. These positions are calculated
from the offsets; you do not enter them in another body-position list.

`None` means that a field is not needed for that joint: these revolute joints do not
use universal-joint axes or an Euler/quaternion choice. `body_data_sym = {}` means
that no geometry values need symbolic parameters in this example.

#### **3 - Points definition**

Next choose where the spring attaches. We want one end at body 1's CG and the other
at body 2's CG. A point's coordinates are measured **from its own body's CG**, so
`[0,0,0]` selects that CG, not the world origin.

In 2D, a spring endpoint can refer directly to a CG, for example `CG1_0`. In 3D,
`TensionSpring` identifies endpoints by body number and point index, so we declare
each CG as a body point at `[0,0,0]` and connect those points. They are called
**body points** because each belongs to a body; a body point can be at the CG or
offset from it. `Force["CG"]` is a separate input for applying a force or moment at
a CG, not for connecting a spring.

The point list on each body starts at index 0. We name these two endpoints by their
body number and point index in the next step:

```python
points_sym = {}
Initial_Points = {
    "GR": [],
    "BD": {
        1: [[0.0, 0.0, 0.0]],
        2: [[0.0, 0.0, 0.0]],
    },
}
```

#### **4 - Forces definition**

The point definitions above only mark locations; they do not apply forces. Now add
three effects: gravity pulling downward, a linear spring joining the two marked
points, and a torsional spring resisting rotation at the second hinge:

```python
force_sym = {}
Force = {
    "Gravity": {"g_vec": [0.0, 0.0, -9.81], "g_app": [1.0, 1.0]},
    "TensionSpring": [(1, 0, 2, 0, 4.0, 1.2)],
    "TorsionSpring": [(1, 0.6, 0.0)],
}
```

**Which points does the spring connect?** Each `TensionSpring` tuple has this order:

```text
(bodyA, pointA, bodyB, pointB, stiffness, natural_length)
```

For `(1, 0, 2, 0, 4.0, 1.2)`, the entries mean:

| Entries | Meaning in this example |
|---|---|
| `1, 0` | First endpoint: body 1, point 0, defined by `Initial_Points["BD"][1][0]` |
| `2, 0` | Second endpoint: body 2, point 0, defined by `Initial_Points["BD"][2][0]` |
| `4.0` | Spring stiffness, 4 N/m |
| `1.2` | Natural (unstretched) length, 1.2 m |

Body numbers start at 1, but point indices start at 0 **on each body separately**.
Both selected points were defined as `[0,0,0]` relative to their own body's CG, so
this spring connects the two CGs. The attachment points move with their bodies;
the spring's length is calculated from their current world positions, not entered
as a separate coordinate.

To attach the spring away from a CG, change that point's body-local coordinates in
`Initial_Points["BD"]`. To select a different point on the same body, add it to that
body's point list and use its index in the spring tuple: index 1 selects the second
point. For a fixed ground endpoint, use body number 0 and a point index into
`Initial_Points["GR"]` instead. Defining a point alone does not attach a spring;
the endpoint pairs in `TensionSpring` select the connection.

**The torsional spring is attached differently.** `(1, 0.6, 0.0)` selects **joint
index 1**, the second revolute joint connecting body 1 to body 2. It acts between
those bodies about that joint's hinge axis, so it does not need two point entries.
Its stiffness is 0.6 N m/rad and its equilibrium angle is 0 rad. There are no dampers
or additional applied loads.

#### **5 - Define numerical values**

***a. Mass and inertia***

Each body  needs
one mass and a 3-by-3 inertia matrix about its CG, expressed in its body frame (body frame is initially parallel to global frame, but over time these frames may vary).

Repository `example9.py` assigns both bodies a mass of 1 kg and uses these inertia
tensors:

```python
body_inertia = {
    1: {
        "mass": 1.0,
        "J": [[0.333, 1.0, 3.0],
              [1.0, 0.333, 0.0],
              [3.0, 0.0, 0.01]],
    },
    2: {
        "mass": 1.0,
        "J": [[0.333, 1.33, 0.33],
              [1.33, 0.333, 0.0],
              [0.33, 0.0, 0.01]],
    },
}
```

These matrices are copied exactly from the repository example so you can compare the
inputs directly. Although they are symmetric and demonstrate how off-diagonal entries
are written, they are not physically valid inertia tensors. Replace them with measured
or otherwise physically valid values for a physical model.

For example, this is a physically valid full tensor with nonzero off-diagonal entries
(all entries are in kg m²):

```python
J = np.array([
    [0.40,  0.02, -0.01],
    [0.02,  0.30,  0.015],
    [-0.01, 0.015, 0.20],
])

body_inertia = {
    1: {"mass": 1.0, "J": J.tolist()},
    2: {"mass": 1.0, "J": J.tolist()},
}
```

The mirrored entries must match so the tensor is symmetric. The matrix is about
each body's CG and expressed in its body frame; it remains body-fixed as the body
rotates. The 3D repository's `example2.py` also demonstrates nonzero off-diagonal
inertia entries in its `body_inertia` definition.

***b. Initial conditions and parameter values***

Now choose the starting motion. Each revolute joint contributes one angle to
`q_user_np` and one angular speed to `qd_np`, **in the order of the joints list**.
The first entry belongs to the ground hinge; the second belongs to the hinge between
the bodies. Start both angles at zero, keep the ground hinge initially at rest, and
give the second hinge a relative speed of 1 rad/s:

```python
q_user_np = np.array([0.0, 0.0])
qd_np     = np.array([0.0, 1.0])

body_params   = np.ones(len(body_data_sym))
force_params  = np.ones(len(force_sym))
points_params = np.ones(len(points_sym))
```

The three symbol dictionaries are empty, so the three parameter arrays have length
zero. Keep these arrays even when there are no symbols: the main file uses the same
stacking line for every example.

For a first change, set `q_user_np[0]` to `np.pi/6` to start the first hinge at
30 degrees. Leave `data` unchanged: the offsets still describe the same bodies.

***c. Time and simulation parameters***

```python
tspan    = 10.0
TimeStep = 0.05
integrator_kwargs = {
    "algorithm": "Dopri8",
    "rtol": 1e-12,
    "atol": 1e-12,
    "max_steps": 10_000_000,
    "dtmax": 0.01,
}
```

`TimeStep` sets the spacing between saved results. It is not necessarily the internal
solver step; `dtmax` places an upper limit on that step.

#### **6 - Call from main and show results**

This input already exists in the 3D repository as `example9.py`; use it directly
instead of creating a separate tutorial file. In the main file, select it with:

```python
import example9 as ex
```

Keep steps 1-5 together in that example file. The code below belongs in the main file,
after the import above. It performs four jobs: build the model, gather the numerical
inputs, integrate the motion, and calculate the plots:

```python
from multibody_3d import MbdSystem3D
from multibody_3d import plotting as plot
import matplotlib.pyplot as plt
import numpy as np

mbd = MbdSystem3D.from_example(ex)
mbd.summary_table(precision=3)

q_user_np     = ex.q_user_np
qd_np         = ex.qd_np
body_params   = ex.body_params
force_params  = ex.force_params
points_params = ex.points_params

mainNumVars = np.concatenate([
    q_user_np, qd_np, body_params, force_params, points_params,
])

sol = mbd.integrate(
    mainNumVars,
    tspan=ex.tspan,
    dt=ex.TimeStep,
    **ex.integrator_kwargs,
)

energy = mbd.compute_energy(sol, mainNumVars)
kin = mbd.compute_kinematics(sol, mainNumVars)

plot.plot_system_energy(energy)
plot.plot_body_trajectories(kin)
plt.show()
```

`mainNumVars` is simply the numerical input array expected by the evaluator. Its first
two groups are the initial angles and speeds; the remaining groups supply values for
symbols, if any. `sol` stores the simulated state at the saved times. We then calculate
`energy` and `kin` from that solution; `kin` contains body positions, orientations,
and velocities.

The two plotting calls above show total energy and CG position histories. The
pendulum starts with the second hinge moving, so its body positions should change.
Total energy includes kinetic, gravitational, and spring energy. With no damping or
additional loads, that sum should stay constant within numerical tolerance, even
though its kinetic and potential parts change.

### Example 2: A floating base with symbolic parameters

This example is based on repository Example 7. A floating base carries three hinged
arms. It introduces symbolic dimensions, a symbolic point coordinate, and a force
whose components are selected through numerical parameter values.

```{figure} _static/figs/3d_tutorial_floating_base.png
:width: 75%
:align: center

Tutorial Example 2: a floating base and three revolute connections, shown at the
initial identity orientation. The F connection describes free motion, not a support.
```

#### **1 - Reference frames**

The base can translate and rotate in space. In this example its user-facing orientation
is given by Euler angles. The three revolute axes are expressed in the **base frame**,
so they move with the base; they are not fixed world axes.

#### **2 - Bodies definition**

Suppose you want to try several arm lengths without rewriting the geometry. Instead
of inserting a number wherever a length appears, use a SymPy symbol. Here `R` is the
distance from the base CG to each hinge, and `L` is an arm length.

Create the symbols and list them in `body_data_sym`. This dictionary tells M4E which
geometry parameters will receive numerical values in step 5:

```python
import numpy as np
import sympy as sym

R, L = sym.symbols("R L", real=True)
body_data_sym = {"R": R, "L": L}
azimuths = [0, 2 * sym.pi / 3, 4 * sym.pi / 3]

data = {
    "NBodies": 4,
    "joints": [(0, 1), (1, 2), (1, 3), (1, 4)],
    "types": ["F", "R", "R", "R"],
    "parent_cg_to_joint": [[0, 0, 0]] + [
        [R * sym.cos(a), R * sym.sin(a), 0] for a in azimuths
    ],
    "joint_to_child_cg": [[0, 0, 0]] + [
        [L / 2 * sym.cos(a), L / 2 * sym.sin(a), 0] for a in azimuths
    ],
    "axis_u": [None] + [
        [-sym.sin(a), sym.cos(a), 0] for a in azimuths
    ],
    "axis_u1": [None, None, None, None],
    "axis_u2": [None, None, None, None],
    "rot_param": ["euler", None, None, None],
}
```

The list expressions repeat the same construction at 0, 120, and 240 degrees around
the base's Z axis. `[cos(a),sin(a),0]` points outward toward an arm; multiplying it by
`R` locates its hinge, and multiplying it by `L/2` locates its CG from that hinge.
`[-sin(a),cos(a),0]` gives a perpendicular hinge axis in the base's XY plane.

There are four connections: a floating connection for the base, then one hinge per
arm. The base needs six speeds (three linear and three angular), and each hinge
needs one more, giving nine speeds altogether.

#### **3 - Points definition**

This time we mark points away from the CGs. The first two use the existing length
symbol `L`; the third uses a new distance `d4`. Put the new point parameter in
`points_sym`. Do not register `L` again: it already appears in `body_data_sym`.
These points demonstrate how to define attachment locations; no spring or point load
uses them in this example:

```python
d4 = sym.Symbol("d4", real=True)
points_sym = {"d4": d4}
Initial_Points = {
    "GR": [[0.0, 0.0, 0.0]],
    "BD": {
        2: [[L / 2, 0.0, 0.0]],
        3: [[L / 2, 0.0, 0.0]],
        4: [[d4, 0.0, 0.0]],
    },
}
```

This defines one ground point and one point each on bodies 2, 3, and 4; body 1 has
no additional point. The force on body 1 is applied directly at its CG, and none of
these declared points is used by a force or spring in this example.

These are body-local coordinates measured from each body's CG. A local `[L/2,0,0]`
point does not automatically point along the arm's azimuth; it points along that
body's local X direction. Choose the components to describe the attachment you intend.

#### **4 - Forces definition**

Apply a force at body 1's CG. We want to choose its X, Y, and Z components later, so
create three symbols and list them in `force_sym`. Put those symbols in the force
entry for body 1:

```python
Fx1, Fy1, Fz1 = sym.symbols("Fx1 Fy1 Fz1", real=True)
force_sym = {"Fx1": Fx1, "Fy1": Fy1, "Fz1": Fz1}
Force = {
    "CG": {1: {"force": [Fx1, Fy1, Fz1]}},
}
```

`force_sym` records which numerical values are required and their order. `Force`
records where the load acts. They do different jobs: registering a symbol does not
apply a force, and defining a symbolic force does not assign its numerical magnitude.

In short, `force_sym` says **which numbers to supply**; `Force["CG"][1]` says
**where to apply them**. A CG force needs no point index because the CG is already
known from the body geometry.

To apply a force away from a CG, select a declared body point with `Force["PointsBD"]`;
see section 5.4 for the syntax. The points declared in this example are not loaded.

#### **5 - Define numerical values**

***a. Mass and inertia***

```python
body_inertia = {
    body: {"mass": 1.0, "J": (0.01 * np.eye(3)).tolist()}
    for body in range(1, 5)
}
```

***b. Initial conditions and parameter values***

The first six position entries describe the floating base:
`[x,y,z,roll,pitch,yaw]`. The next three are the arm hinge angles, in joint order.
The speed entries follow the same grouping: six base speeds, then three hinge rates.
Because body 1's floating joint uses `rot_param="euler"`, we provide three orientation
inputs: `roll`, `pitch`, and `yaw`. Setting all three to zero results in no initial rotation:

```python
q_user_np = np.zeros(9)
qd_np     = np.array([0.2, 0.3, 0.0, 0.3, 0.4, 0.5, 0.6, 0.8, 0.0])

body_params   = np.ones(len(body_data_sym))
force_params  = np.ones(len(force_sym))
points_params = np.ones(len(points_sym))

body_params[:]   = [0.5, 1.0]
force_params[:]  = [1.0, 1.0, 1.0]
points_params[:] = [0.3]
```

The body values are `R=0.5` m and `L=1.0` m. The applied force is `[1,1,1]` N, and
the point offset is `d4=0.3` m. The initial base speed slots are
`[vx,vy,vz,wx,wy,wz]`; the last three entries are relative hinge speeds.

The parameter arrays follow the order in which you listed the symbols:

| Symbol dictionary | Numerical array | Meaning of the supplied values |
|---|---|---|
| `body_data_sym`: `R`, then `L` | `body_params` | `[0.5,1.0]` sets the hinge radius and arm length |
| `force_sym`: `Fx1`, `Fy1`, then `Fz1` | `force_params` | `[1.0,1.0,1.0]` sets the applied force components |
| `points_sym`: `d4` | `points_params` | `[0.3]` sets the marked point's local X offset |

For example, changing only `force_params[2]` changes the vertical force component.
Changing only `body_params[1]` changes `L` everywhere that symbol appears in the
geometry and point definitions.

***c. Time and simulation parameters***

```python
tspan    = 10.0
TimeStep = 0.05
integrator_kwargs = {
    "algorithm": "Dopri8",
    "rtol": 1e-12,
    "atol": 1e-12,
    "max_steps": 10_000_000,
    "dtmax": 0.01,
}
```

#### **6 - Call from main and show results**

This input already exists in the 3D repository as `example7.py`; import it directly:

```python
import example7 as ex
```

Use the same main-file sequence as Example 1. There is no separate stacking rule for
symbolic examples: the five arrays are always placed in the same order. The numerical
parameters stay constant during this solve, while positions and speeds evolve.

Here the CG force does work as the base moves. Therefore, do not expect constant
system energy. Setting `force_params[:] = [0.0, 0.0, 0.0]` removes that applied load
without changing the symbolic definition. To try a different size, change
`body_params`, rebuild `mainNumVars`, and run a new simulation.

### Example 3: Cylindrical, revolute, and spherical joints

This example follows the repository's `example3.py`: a cylindrical joint, a revolute
joint, and a spherical joint. The spherical joint's relative orientation is supplied
as a quaternion relative to its parent. Absolute-orientation input is planned for a
future release.

#### **1 - Reference frames**

For the ground-to-body-1 cylindrical joint, `axis_u` is expressed in the parent
frame (ground), so `[1,0,0]` points along world X. Body 1 is the child of this
joint. The following revolute joint has its own `axis_u`, expressed in its parent
frame (body 1), so `[0,1,0]` points along body 1's local Y and rotates with body 1;
it is not fixed to world Y.

#### **2 - Bodies definition**

Read the connections from ground outward: ground to body 1 through a cylindrical
joint, body 1 to body 2 through a revolute joint, then body 2 to body 3 through a
spherical joint. Unlike the first example, the first connection allows two motions:
rotation about X and translation along X.

```python
import numpy as np

L2 = 1.0
L3 = 0.8
body_data_sym = {}
data = {
    "NBodies": 3,
    "joints": [(0, 1), (1, 2), (2, 3)],
    "types": ["C", "R", "S"],
    "parent_cg_to_joint": [
        [0.25, 0.25, 0.0], [0.0, 0.0, 0.25], [0.0, 0.0, L2 / 2],
    ],
    "joint_to_child_cg": [
        [0.0, 0.0, 0.125], [0.0, 0.0, L2 / 2], [0.0, 0.0, L3 / 2],
    ],
    "axis_u": [[1, 0, 0], [0, 1, 0], None],
    "axis_u1": [None, None, [1, 0, 0]],
    "axis_u2": [None, None, [0, 1, 0]],
    "rot_param": [None, None, "quat"],
}
```

#### **3 - Points definition**

No attachment points or point symbols are used in this model. The repository input
omits `Initial_Points` and keeps `points_sym` empty:

```python
points_sym = {}
```

#### **4 - Forces definition**

The repository model has no applied forces, gravity, springs, or dampers. It keeps
`force_sym` empty and omits the `Force` dictionary:

```python
force_sym = {}
```

#### **5 - Define numerical values**

***a. Mass and inertia***

```python
body_inertia = {
    body: {"mass": 1.0, "J": (0.01 * np.eye(3)).tolist()}
    for body in range(1, 4)
}
```

***b. Initial conditions and parameter values***

| Joint | Position entries | Speed entries |
|---|---|---|
| C | `theta, s` | `theta_dot, s_dot` |
| R | `theta` | `theta_dot` |
| S (quaternion) | `e0, e1, e2, e3` | `wx, wy, wz` |
| Total | 7 | 6 |

```python
q_user_np = np.array([0.3, -0.2, 0.5, 0.4, 0.2, 0.0, 0.0])
qd_np     = np.array([0.7, -0.3, 0.4, 1.2, 0.0, 0.0])

body_params   = np.ones(len(body_data_sym))
force_params  = np.ones(len(force_sym))
points_params = np.ones(len(points_sym))
```

Read `q_user_np` in joint order: the cylindrical joint starts at 0.3 rad and -0.2 m,
the revolute joint at 0.5 rad, and the spherical joint uses relative quaternion
`[0.4,0.2,0,0]`. The speed vector follows the same joint order: cylindrical rates
`[0.7,-0.3]`, revolute rate `0.4`, and spherical relative angular velocity
`[1.2,0,0]` rad/s. The coordinate mapping normalizes the nonzero quaternion. There is
no additional coordinate for a body's absolute pose; M4E obtains it from the joints.

***c. Time and simulation parameters***

```python
tspan    = 10.0
TimeStep = 0.05
integrator_kwargs = {
    "algorithm": "Dopri8",
    "rtol": 1e-12,
    "atol": 1e-12,
    "max_steps": 10_000_000,
    "dtmax": 0.01,
}
```

#### **6 - Call from main and show results**

This input already exists in the 3D repository as `example3.py`; import it directly:

```python
import example3 as ex
```

Use the same main-file sequence as Example 1. The saved state has seven configuration
entries followed by six speeds. The cylindrical translation is unrestricted, so this is
not a carriage with end stops; a long run may produce substantial travel.

## 3 - Detailed bodies and joint definitions

### 3.1 - Body numbering and connectivity

Ground is body 0. Moving bodies are numbered from 1 to `NBodies`. A joint `(parent, child)`
connects the child body to its parent. The current input describes a tree: each moving
body has one incoming joint, and multiple children can share a parent, as in Example 2.
Closed-loop mechanisms are not specified by adding another incoming joint.

Joint indices are zero-based positions in the `joints` list. In `[(0,1),(1,2)]`, joint
index 0 connects ground to body 1 and joint index 1 connects body 1 to body 2. This
is different from the one-based body numbers.

### 3.2 - Geometry vectors and joint axes

Start with an initial-configuration sketch of the system, with joint coordinates set
to zero. Each body's local axes are parallel to the global axes in this pose, so read
the geometry vectors and joint-axis directions directly from the sketch. Enter them
once; body-fixed vectors move with their bodies during simulation.

| Field | What to read from the sketch | What it follows |
|---|---|---|
| `parent_cg_to_joint` | Parent CG to joint | Parent body; world for ground |
| `joint_to_child_cg` | Joint to child CG | Child body |
| `axis_u` | Rotation or translation direction | Parent body; world for ground |
| `axis_u1` | First universal-joint pin direction | Parent body |
| `axis_u2` | Second universal-joint pin direction at zero joint angles | Child body |

For example, `parent_cg_to_joint=[0,0,1]` places the joint one metre along the
parent's initial Z direction from its CG. Initial joint coordinates set the starting
pose; they do not change these fixed geometry vectors.

For P/C joints, translation along the axis is a separate joint coordinate, not part
of the fixed geometry offsets. An F joint must connect ground to a body. Its
translation coordinates locate the joint in world space, and `joint_to_child_cg`
locates the body's CG from that point. Ground `parent_cg_to_joint` vectors and ground
points in `Initial_Points["GR"]` are specified in world coordinates.

The joint system normalizes nonzero axis vectors. Use `None` for unused axis entries.
Joint motion options and their input axes are summarized below.

The joint-coordinate explanations below are adapted to M4E from Professor Parviz Nikravesh lectures. The revolute, prismatic,
spherical, and floating-joint illustrations are cropped from the corresponding
original lecture pages. In those figures, `(i-)` is the parent and `(i)` is the child.
The lecture describes universal and cylindrical joints in text but does not illustrate
them, so those sections have no figure.

**Joint coordinates describe relative motion.** Joint coordinates use the minimum amount of coordinates needed to prescribe a multibody system. For two connected bodies, the parent
is the one you reach first when moving from ground through the tree. The child's
coordinates describe its permitted motion relative to that parent. A hinge needs one
angle, a slider needs one displacement, and a ball joint needs a full relative
orientation. The corresponding speed entries describe those permitted motions.

For R/P/U/C joints the speed entries are derivatives of their scalar coordinates.
For S/F joints we use angular velocity instead of differentiating Euler angles or
quaternion entries directly. Thus a quaternion has four configuration numbers but
only three independent angular-speed components.

### 3.3 - Revolute joints (R)

A revolute joint is a hinge: it allows the child to turn relative to the parent about
one shared axis, while keeping the connection together. Its coordinate is the relative
angle $\theta$, and its speed is $\dot\theta$. The axis do not need to line up with X, Y, or Z;
`axis_u` gives the rotation direction in the parent frame (initially parallel to global frame as discussed before).

```{figure} _static/figs/nikravesh_joint_revolute.png
:width: 37.5%
:align: center

Revolute-joint illustration from Professor Parviz Nikravesh slides.
```

| Input | Meaning |
|---|---|
| `types` entry | `"R"` |
| `axis_u` entry | Hinge direction in the parent frame |
| `q_user_np` entries | `[theta]`, radians |
| `qd_np` entries | `[theta_dot]`, rad/s |

The same physical hinge axis is fixed in both bodies; their motion does not change
which line the hinge rotates about. This is the only joint currently accepted by
`TorsionSpring` and `TorsionDamper`.

### 3.4 - Prismatic joints (P)

A prismatic joint is a slider. It lets the child move along one direction without
rotating relative to its parent. Its coordinate $s$ measures translation from the
reference configuration, and its speed $\dot s$ tells how quickly that translation
changes. The additional displacement vector is $s\mathbf u$, where $\mathbf u$ is
the current world direction of the axis stored in `axis_u`.

```{figure} _static/figs/nikravesh_joint_prismatic.png
:width: 37.5%
:align: center

Prismatic-joint illustration from Professor Parviz Nikravesh slides.
```

| Input | Meaning |
|---|---|
| `types` entry | `"P"` |
| `axis_u` entry | Sliding direction in the parent frame |
| `q_user_np` entries | `[s]`, metres |
| `qd_np` entries | `[s_dot]`, m/s |

Unlike 2D, its direction is supplied through `axis_u`, not a separate
`prismatic_direction` list. If the parent rotates, the slider's world direction rotates
with it. There is no built-in travel limit in this input.

### 3.5 - Floating joints (F)

An F (floating) joint is how the model represents a floating body: a body free to
translate and rotate relative to ground. In 3D, it gives the body six degrees of freedom: three translations
and three rotations.

```{figure} _static/figs/nikravesh_joint_floating.png
:width: 37.5%
:align: center

Floating-joint illustration from Professor Parviz Nikravesh slides.
```

| Input | Meaning |
|---|---|
| `types` entry | `"F"` |
| Axis entries | `None`; no single hinge or slider axis |
| `q_user_np`, Euler input | `[x,y,z,roll,pitch,yaw]` |
| `q_user_np`, quaternion input | `[x,y,z,e0,e1,e2,e3]` |
| `qd_np` entries | `[vx,vy,vz,wx,wy,wz]` |

With zero `joint_to_child_cg`, these coordinates and speeds describe the body's world
CG position, orientation, linear velocity, and angular velocity. The joint always has
six speed entries, even when quaternion configuration uses seven entries.

### 3.6 - Spherical joints (S)

A spherical joint is a ball-and-socket connection. The joint point stays together,
but the child can rotate relative to the parent about three independent directions.
One angle cannot describe that relative orientation: choose three Euler angles or
a quaternion through `rot_param`.

```{figure} _static/figs/nikravesh_joint_spherical.png
:width: 37.5%
:align: center

Spherical-joint illustration from Professor Parviz Nikravesh slides.
```

| Input | Meaning |
|---|---|
| `types` entry | `"S"` |
| Axis entries | `None`; no single rotation axis |
| `q_user_np`, Euler input | `[roll,pitch,yaw]`, relative orientation |
| `q_user_np`, quaternion input | `[e0,e1,e2,e3]`, relative orientation |
| `qd_np` entries | `[wx,wy,wz]`, relative angular velocity in the parent frame |

The Euler angles or quaternion supplied for an S joint describe orientation relative
to its parent. Absolute-orientation input is planned for a future release.

The child angular velocity is obtained by adding this relative angular velocity,
after expressing it in the world frame, to the parent angular velocity. The ball
joint has three rotational freedoms, regardless of whether its orientation is stored
with three angles or four quaternion numbers.

### 3.7 - Universal joints (U)

A universal joint combines two revolute motions about perpendicular pins. Imagine
two hinges joined by a rigid cross: one pin belongs to the parent side, and the
other belongs to the child side. The two coordinates are `alpha` and `beta`, with
speeds `alpha_dot` and `beta_dot`.

| Input | Physical attachment | How to supply its components |
|---|---|---|
| `axis_u1` | Pin fixed to the parent | Parent-frame direction |
| `axis_u2` | Pin fixed to the child | Direction in the child's zero-angle reference frame, which is parallel to the parent reference frame |
| `q_user_np` entries | Two hinge rotations | `[alpha,beta]`, radians |
| `qd_np` entries | Two hinge rates | `[alpha_dot,beta_dot]`, rad/s |


### 3.8 - Cylindrical joints (C)

A cylindrical joint combines a slider and a hinge along the same line. The child can
move along the cylinder axis and rotate about that axis. These are two independent
motions, with coordinates `theta` and `s` and speeds `theta_dot` and `s_dot`.

| Input | Meaning |
|---|---|
| `types` entry | `"C"` |
| `axis_u` entry | Common rotation and sliding direction in the parent frame |
| `q_user_np` entries | `[theta,s]`, radians then metres |
| `qd_np` entries | `[theta_dot,s_dot]`, rad/s then m/s |

The coordinate order is **rotation first, translation second**. This is one
cylindrical connection, not a new intermediate body joined by separate R and P joints.
Tutorial Example 3 uses it at the start of the chain.

### 3.9 - Mass and inertia

`body_inertia` is keyed by body number and holds `"mass"` in kg and `"J"` in kg m2.
The tensor is about the body's CG in its body frame. A diagonal tensor specifies three
principal moments; a full tensor can include products of inertia. Use physically valid,
symmetric tensors and positive masses rather than arbitrary off-diagonal values.

The numerical 3D runtime currently requires numeric mass and inertia entries. Do not
register a symbolic mass in `force_sym` and expect it to behave like a force parameter.
When `from_example()` finds no `body_inertia`, it warns and uses 1 kg and `0.01*I` for
each body. These are placeholders, not estimated physical properties. The worked
examples above provide their own values explicitly.

## 4 - Rotations and initial conditions

### 4.1 - Euler angles

Euler input uses `[phi,theta,psi]`, meaning `[roll,pitch,yaw]`, in radians. The rotation
matrix is formed as $R_z(\psi)R_y(\theta)R_x(\phi)$. Set `rot_param="euler"` for an S
or F joint to use these entries. Here `theta` means pitch within the three-angle group;
it is not a separate revolute-joint coordinate.

### 4.2 - Quaternions

Quaternion input uses `[e0,e1,e2,e3] = [w,x,y,z]`. Set `rot_param="quat"` for an S or F
joint to accept this format. The no-rotation quaternion is `[1,0,0,0]`. The joint mapping
normalizes a nonzero input quaternion, but four zeros cannot be normalized.

For an F joint, translation comes first: `[x,y,z,e0,e1,e2,e3]`. Do not put the scalar
quaternion component last, as some other rotation libraries do.

### 4.3 - Relative and absolute orientation



For a ground-connected body the parent frame is the world frame, so relative and absolute
orientations coincide. For an internal joint they generally do not. The returned
`kin.R_cg` and `kin.euler_cg` describe absolute body orientations in the world frame.
The current S/F input is parent-relative; support for entering absolute orientation is
planned for a future release.

### 4.4 - User and internal coordinates

`q_user_np` follows the example's chosen `rot_param`. Internally, S and F joints always
use quaternions. Therefore the number of saved configuration entries can differ from
the number of user entries:

| Joint | User configuration | Internal configuration | Speeds |
|---|---:|---:|---:|
| R | 1 | 1 | 1 |
| P | 1 | 1 | 1 |
| U | 2 | 2 | 2 |
| C | 2 | 2 | 2 |
| S, Euler input | 3 | 4 | 3 |
| S, quaternion input | 4 | 4 | 3 |
| F, Euler input | 6 | 7 | 6 |
| F, quaternion input | 7 | 7 | 6 |

Example 2 has nine user configuration entries and ten internal entries. Its saved
state contains those ten internal entries followed by nine speeds. Use
`mbd.map_q_int_to_q_user()` when turning a saved internal configuration back into a
user-facing input; use `mbd.map_q_user_to_q_int()` for the opposite direction.

### 4.5 - What the speed vector means

For R/P/U/C joints, `qd_np` contains rates of their scalar coordinates. For S joints,
it contains relative angular velocity `[wx,wy,wz]` expressed in the parent frame.
For F joints it contains `[vx,vy,vz,wx,wy,wz]`; these motion components are expressed
in the world-frame.

For S/F joints, angular-velocity components are not Euler-angle derivatives and are
not quaternion-component derivatives. Supplying three angular speeds is correct even
when four quaternion entries are used for position.

### 4.6 - Defining initial conditions in the example

Declare `q_user_np` and `qd_np` in joint-list order. The main file does not need to
choose initial conditions for each model. Print `mbd.coords.names_user` and
`mbd.coords.names_d_user` if you need to see the order expected by a particular model.

For a later state, copy the original complete `mainNumVars`, replace its coordinate
and speed portions, and keep the parameter values. Do not build an input from just
coordinates and speeds when the model has symbolic parameters.

## 5 - Points and forces

Use this section when adding a load to your own model. First decide **where it acts**:
at a CG, at a marked body point, or between two points. Then choose the corresponding
entry in `Force`. `Initial_Points` only defines locations; a point has no applied
load until a force, spring, or damper refers to it.

| What you want | What to define |
|---|---|
| A force or moment at a body's CG | An entry in `Force["CG"]`; no extra point needed |
| A force at another location on a body | A body point, then an entry in `Force["PointsBD"]` selecting it |
| A linear spring or damper between locations | Two attachment points, then a spring or damper tuple selecting both |
| A torsional spring or damper at a hinge | An R-joint index in `Force["TorsionSpring"]` or `Force["TorsionDamper"]` |
| Gravity | `Force["Gravity"]`; no attachment point needed |

### 5.1 - Ground and body points

`Initial_Points["GR"]` is a list of fixed world points. `Initial_Points["BD"]` is a
dictionary of body-local points, each measured from the CG of its owning body.
The body rotation and CG position place those points in the world during evaluation.
Numeric coordinates require no point symbols; symbolic point offsets belong in
`points_sym`, unless they already belong in `body_data_sym`, as `L` does in Example 2.

### 5.2 - Body numbers and point indices

Bodies are one-based and point indices are zero-based. Body 2, point 0 refers to
`Initial_Points["BD"][2][0]`. In a tension element, body 0 selects a ground point
from `Initial_Points["GR"]`. The 3D input uses these integer references rather than
the 2D spring endpoint strings such as `"BD2_0"`.

### 5.3 - Forces and moments at the CG

For a force `[10,0,0]` N and a free moment `[0,0,2]` N m on body 1:

```python
Force["CG"] = {
    1: {"force": [10.0, 0.0, 0.0], "moment": [0.0, 0.0, 2.0]},
}
```

Both vectors are expressed in the world frame. The moment is optional and defaults
to zero. A CG force does not require declaring a body point.

### 5.4 - Loads at body-attached points

To apply a world-frame force at body 2's first declared point:

```python
Force["PointsBD"] = [(2, 0, [0.0, 0.0, -5.0])]
```

The force's moment arm is calculated automatically. An optional fourth entry supplies
an additional free moment vector; it does not replace the induced moment about the CG.

### 5.5 - Translational springs and dampers

The entry order is:

```text
TensionSpring: (bodyA, pointA, bodyB, pointB, k, L0)
TensionDamper: (bodyA, pointA, bodyB, pointB, c)
```

For example, connect ground point 0 to body 2 point 0, after declaring both points:

```python
Force["TensionSpring"] = [(0, 0, 2, 0, 20.0, 1.0)]
Force["TensionDamper"] = [(0, 0, 2, 0, 0.5)]
```

The spring force depends on length change, $k(L-L_0)$, and the damper force depends
on extension rate, $c\dot L$. Equal-and-opposite loads are applied to the moving
bodies, including the moments created by offset attachment points. Despite the name
`TensionSpring`, the current linear law also acts in compression; it is not a slack cable.
Keep its endpoints separated: a zero-length connection has no defined direction.

Unlike 2D, stiffness comes **before** natural length in the 3D tension-spring tuple.
The 2D reserved nonlinear-law symbols `l` and `ld` are not implemented by this 3D contract.

### 5.6 - Torsional springs and dampers

The entry order is:

```text
TorsionSpring: (joint_index, k, theta_eq)
TorsionDamper: (joint_index, c)
```

For a revolute joint at index 1:

```python
Force["TorsionSpring"] = [(1, 0.6, 0.0)]
Force["TorsionDamper"] = [(1, 0.2)]
```

The spring resists departure from `theta_eq`; the damper resists relative angular
speed. The child and parent receive opposite torques about the joint's current world
axis. The reference is a joint index, not a pair of body numbers as in the 2D input.
The current implementation supports these elements on R joints only.

### 5.7 - Gravity

```python
Force["Gravity"] = {
    "g_vec": [0.0, 0.0, -9.81],
    "g_app": [1.0, 0.5],
}
```

For a two-body model, this applies full gravity to body 1 and half gravity to body 2.
Omit `g_app` for full gravity on every body. Masses are read from `body_inertia`.
Omit `Gravity` entirely when it is not wanted; it is not added automatically.

### 5.8 - Time-dependent forces

Use a SymPy symbol named `t` directly in the load expression:

```python
import sympy as sym

t = sym.Symbol("t", real=True)
Force["CG"] = {1: {"force": [2.0 * sym.sin(t), 0.0, 0.0]}}
```

Do **not** include `t` in `body_data_sym`, `force_sym`, or `points_sym`. Unlike the
2D format, 3D supplies simulation time separately at each solver stage. Other unknown
force amplitudes or coefficients still need ordinary registry entries and numerical values.
For standalone evaluation, pass time explicitly with `mbd.evaluate_forces(mainNumVars, t=...)`.

## 6 - Symbolic parameters and numerical values

Use numbers directly when they are fixed for your model. Use a symbol when you want
to choose its value separately, for example to run the same mechanism with several
arm lengths. A symbolic parameter is not an extra degree of freedom: its value stays
fixed during a simulation.

There are three steps: create a SymPy symbol, use it in an input expression, and
supply its numerical value in the matching parameter array. The dictionaries below
make that matching explicit. Creating a symbol alone does not tell M4E its value.

### 6.1 - The three symbol dictionaries

`body_data_sym` supplies parameters used by joint geometry. `points_sym` supplies
additional local point-coordinate parameters. `force_sym` supplies load components
and constitutive coefficients. Keep the same SymPy symbol objects in the dictionaries
and the expressions that use them.

Point-coordinate evaluation accepts body-data and point parameters. Force-component
and spring-coefficient evaluation currently accepts force and point parameters, plus
time where applicable; do not assume that a force coefficient can reference a
`body_data_sym` symbol simply because that symbol exists elsewhere in the model.

### 6.2 - Numerical values follow dictionary order

For `body_data_sym={"R":R,"L":L}`, the first entry of `body_params` is the value of
`R` and the second is the value of `L`. The same rule applies separately to forces
and points. If you reorder a dictionary, reorder its numerical values too.

### 6.3 - Defaults and empty arrays

Use the same default pattern as in 2D:

```python
body_params   = np.ones(len(body_data_sym))
force_params  = np.ones(len(force_sym))
points_params = np.ones(len(points_sym))
```

An empty dictionary produces a one-dimensional empty array, not a force or dimension
equal to zero. With symbols present, the defaults are 1.0. Replace those values with
the ones for your actual problem, as in Tutorial Example 2. The model cannot tell
whether a value of 1.0 was chosen deliberately or left as a default.

### 6.4 - The complete numerical vector

The public evaluation and integration methods expect this order:

```python
mainNumVars = np.concatenate([
    q_user_np, qd_np, body_params, force_params, points_params,
])
```

This differs from the 2D order, where the combined force/point values precede the
body-data values. All five arrays belong in the 3D stacking line, including empty ones.
Check the length of each group yourself: matching only the total length cannot detect
values accidentally assigned to the wrong group.

### 6.5 - Running different parameter values

With a symbolic body dimension, the same model definition can be used for several sizes.
For the floating-base tutorial, change `R` while keeping the parameter order unchanged:

```python
for radius in [0.4, 0.5, 0.6]:
    body_params = np.array([radius, 1.0])
    mainNumVars = np.concatenate([
        ex.q_user_np, ex.qd_np, body_params, ex.force_params, ex.points_params,
    ])
    sol = mbd.integrate(
        mainNumVars, tspan=ex.tspan, dt=ex.TimeStep, **ex.integrator_kwargs,
    )
```

Each solve starts from the example's initial state. Parameters remain constant within
a solve; this loop changes them between solves. The system's geometry and runtime
evaluators are prepared for the supplied parameter values without rebuilding the
symbolic model in the loop.

## 7 - Simulation and results

### 7.1 - Settings belong in the example

Keep `tspan`, `TimeStep`, and `integrator_kwargs` with the model inputs. A scalar
`tspan=10.0` means start at 0 and stop at 10 s; a pair such as `(2.0,12.0)` sets both
times. The main file passes these values to `mbd.integrate()`.

### 7.2 - Saved times and internal steps

`TimeStep` is passed as `dt` and controls the output spacing. The adaptive solver
can take different internal steps, limited by `dtmax` when supplied. If `dt=None`,
the current integrator saves only the final state, not the full adaptive history.
Use a positive output step and, for the examples, a duration that is an integer multiple
of that step so the requested output grid lies within the solve interval.

### 7.3 - Solvers and tolerances

`Dopri5`, `Dopri8`, and `Tsit5` are explicit adaptive solvers. `Kvaerno3` and
`Kvaerno5` are implicit options for stiff problems. Smaller `rtol` and `atol` request
tighter accuracy; smaller `dtmax` can help resolve fast spring motion. Very stiff
springs can require a different solver rather than simply more saved output points.
The first JAX solve may take longer because it includes compilation.

### 7.4 - Positions, orientations, and velocities

```python
kin = mbd.compute_kinematics(sol, mainNumVars)
```

| Result | Meaning | Shape |
|---|---|---|
| `kin.ts` | Saved times | `(n_times,)` |
| `kin.r_cg` | World CG positions | `(n_times, NBodies, 3)` |
| `kin.R_cg` | Body-to-world rotation matrices | `(n_times, NBodies, 3, 3)` |
| `kin.euler_cg` | Absolute `[roll,pitch,yaw]` angles | `(n_times, NBodies, 3)` |
| `kin.v_cg` | World CG linear velocities | `(n_times, NBodies, 3)` |
| `kin.omega_cg` | World angular velocities | `(n_times, NBodies, 3)` |
| `kin.r_pts`, `kin.v_pts` | Declared body-point positions and velocities | `(n_times, n_body_points, 3)` |

Output body index 0 means body 1. For example, `kin.r_cg[:,0,2]` is body 1's Z
position at every saved time. Euler output is convenient for plotting, but can jump
at angle boundaries or become ambiguous near pitch = +/-90 degrees. Use `R_cg` when
you need an unambiguous orientation comparison.

### 7.5 - System energy and per-body energy

```python
energy = mbd.compute_energy(sol, mainNumVars)
```

`energy.KE` is system kinetic energy, `energy.PE` includes gravity and both spring
families, and `energy.E_total` is their sum. `KE_body` is kinetic energy by body;
`PE_body` is gravitational potential energy by body. Spring energy is included at
system level rather than split arbitrarily between the connected bodies.

With constant spring coefficients, gravity, and no damping or additional loads, total
energy should be constant within numerical tolerance. With positive damping it should
decrease. An applied force can add or remove energy through work. Time-varying spring
coefficients can also exchange energy with the system. Therefore an energy change is
not automatically an integration error.

Energy checks also have to account for known model-validation gaps; see section 9.

### 7.6 - Plotting helpers and CSV output

The 3D plotting module takes already-calculated results and returns Matplotlib figures:

```python
from multibody_3d import plotting as plot

plot.plot_coordinates(sol, mbd.total_cfg_dof,
                      mbd.coords.names_int, mbd.coords.names_d_int)
plot.plot_system_energy(energy)
plot.plot_body_energy(energy)
plot.plot_body_velocities(kin)
plot.plot_body_trajectories(kin)
plt.show()
```

The saved state uses internal coordinates, so `names_d_int` labels its internal speed
symbols. The plotting functions do not call `plt.show()` themselves; the main file
decides whether to display or save their figures. For example:

```python
fig = plot.plot_system_energy(energy)
fig.savefig("system_energy.png", dpi=150)
```

To export CG positions and linear velocities:

```python
from multibody_3d import save_trajectory_csv

save_trajectory_csv(kin, "trajectory.csv")
```

The CSV has one row per body and saved time, with time, body index, body name, XYZ
position, and XYZ linear velocity. Its body index is zero-based. This exporter does
not currently include angular velocity, orientation, or energy columns; those are
available through the result objects above.

## 8 - Accessing the generated equations

You can run all three examples without this section. Use it when you want to inspect
what M4E generated or evaluate the equations at a particular state. There are two
different outputs to distinguish: **symbolic expressions**, which retain coordinate
symbols, and **numerical matrices**, which contain values at the state you supply.
Neither operation requires a time simulation first.

### 8.1 - The equation solved by the simulator

The 3D code constructs symbolic kinematic and force expressions, then evaluates the
dynamics numerically. With generalized speeds $\mathbf u$, the equations have the form

$$
M\dot{\mathbf u}=\mathbf h,\qquad
M=B^T M_{\mathrm{body}}B,\qquad
\mathbf h=B^T\left(\mathbf f-\boldsymbol\gamma
-M_{\mathrm{body}}\dot B\mathbf u\right).
$$

Here $B$ maps joint speeds to world-frame body CG velocities and angular velocities.
$M_{\mathrm{body}}$ contains each body's mass and world-frame inertia tensor;
$\mathbf f$ contains applied forces and moments about the CG, including gravity,
springs, and dampers. For each body, the angular part of $\boldsymbol\gamma$ is
$\boldsymbol\omega\times(J_{\mathrm{world}}\boldsymbol\omega)$; its linear part is zero.

For scalar joint coordinates, $\dot{\mathbf u}$ is the usual coordinate acceleration.
For S/F joints, angular speeds are not Euler-angle or quaternion derivatives. Their
configuration-rate equations are handled separately by the integrator.

### 8.2 - Inspecting symbolic expressions

After creating `mbd = MbdSystem3D.from_example(ex)`, the following accessors expose the
generated symbolic building blocks. No time integration is needed:

```python
import sympy as sym

B_sym    = mbd.assemble_B_symbolic()
Bdot_sym = mbd.assemble_Bdot_symbolic()
q_sym    = mbd.q_int
u_sym    = mbd.qd_int

sym.pprint(B_sym)
sym.pprint(Bdot_sym)
sym.pprint(mbd.sym_mass.J_world(1))

if mbd.sym_total_wrench is not None:
        sym.pprint(mbd.sym_total_wrench[0])
        sym.pprint(mbd.sym_spring_pe)
```

`B_sym` and `Bdot_sym` have shape `(6*NBodies, total_dof)`. Each six-row body block
contains linear components followed by angular components. `sym_mass` requires
`body_inertia`; its accessors use one-based body numbers. `sym_total_wrench` is a list
of six-component body wrenches, so entry 0 is body 1. It and `sym_spring_pe` are `None`
when no force dictionary was declared.

For a LaTeX representation of an expression, use `sym.latex(B_sym)` or
`sym.latex(mbd.sym_mass.J_world(1))`. These are representations of the stored symbolic
objects, not a complete standalone equation export. Rotation chains and other matrix
products are deliberately kept compact and may contain named matrix placeholders.
Avoid expanding or simplifying an entire large model just to display it.

There is currently no public one-call method returning a complete symbolic pair
`(M, h)` or a list of scalar SymPy equations of motion. The accessors above expose
the components; the numerical kernel below provides evaluated matrices.

### 8.3 - Evaluating the mass matrix and full right-hand side

Use a complete `mainNumVars` vector in the order defined in section 6.4. This example
evaluates the equations at that state and at time `t_eval`; it does not integrate:

```python
import numpy as np

t_eval = 0.0
kernel = mbd.evaluate_eom_kernel(mainNumVars)
B      = np.asarray(kernel.B)
Bdot   = np.asarray(kernel.Bdot)
M_body = np.asarray(kernel.M_body)
M      = np.asarray(kernel.M)

speed_start = mbd.total_user_dof
u = np.asarray(mainNumVars[
        speed_start:speed_start + mbd.total_dof
])
f = np.asarray(mbd.evaluate_forces(mainNumVars, t=t_eval).total).ravel()

body_velocity = (B @ u).reshape(mbd.NBodies, 6)
gyro = np.zeros((mbd.NBodies, 6))
for body_index in range(mbd.NBodies):
        angular_start = 6 * body_index + 3
        J_world = M_body[
                angular_start:angular_start + 3,
                angular_start:angular_start + 3,
        ]
        omega = body_velocity[body_index, 3:]
        gyro[body_index, 3:] = np.cross(omega, J_world @ omega)

h     = B.T @ (f - gyro.ravel() - M_body @ (Bdot @ u))
u_dot = np.linalg.solve(M, h)

print("Mass matrix:", M)
print("Full right-hand side:", h)
print("Generalized-speed derivatives:", u_dot)
```

This snippet requires mass/inertia and a force dictionary. For a model with no loads,
declare `Force = {}`. `M` has shape `(total_dof, total_dof)` and `h` has shape
`(total_dof,)`. To get only the mass matrix, use
`mbd.evaluate_mass_matrix(mainNumVars).M`.

`mbd.evaluate_generalized_forces(mainNumVars)` returns the applied-load projection
$B^T\mathbf f$, **not** the complete $\mathbf h$: it does not include the gyroscopic
or $\dot B$ terms. Use `evaluate_forces(..., t=t_eval)` as above when loads depend
on time. This NumPy reconstruction is for inspection outside the simulation loop;
the integrator already evaluates these terms in JAX.

To inspect a saved trajectory state, first convert its internal configuration to user
coordinates with `mbd.map_q_int_to_q_user()`, then rebuild the complete numerical vector
with that state's speeds and the original parameter arrays. Do not pass a raw saved
state to an evaluator that expects `mainNumVars`.

## 9 - Current limitations


### 9.1 - Model and force coverage

- **Tree topology only.** Each moving body has one parent joint. Closed kinematic
    loops are not supported by adding extra joint entries.
- **Rigid-body 3D workflow.** The 2D FSM generator, hydrodynamic adapters, mooring
    integrations, and linearization examples are not automatically available in 3D (will be added in future releases).
- **Joint restrictions.** Universal pins must be perpendicular. Prismatic and
    cylindrical inputs do not impose travel limits or end-stop contact.
- **Torsional elements.** `TorsionSpring` and `TorsionDamper` currently accept R joints
    only, not arbitrary body pairs or S/U/C joints.
- **Spring laws.** A tension spring can also push in compression; it is not a slack
    cable. Coincident endpoints have no defined force direction. The 2D reserved
    nonlinear-law symbols `l`, `ld`, `theta`, and `thetad` are not a 3D input interface.

### 9.2 - Parameters, outputs, and symbolic access

**Mass and inertia must be numbers.** Before running a simulation, give every
body a numeric mass and a numeric 3-by-3 inertia tensor in `body_inertia`.
Putting a mass or inertia symbol in `force_sym` does not make it adjustable;
that dictionary is for force parameters. If `body_inertia` is missing, M4E uses
placeholder values of 1 kg and `0.01 * I`. Those are defaults for setup, not
estimates of your body's real properties. For accurate results, enter measured
or otherwise justified physical values.

**Each symbol dictionary has a job.** Put geometry symbols in `body_data_sym`,
load and spring symbols in `force_sym`, and point-coordinate symbols in
`points_sym`. The groups are not interchangeable. For example, a symbol listed
in `body_data_sym` cannot automatically be used in a force expression. Section 6
describes which groups can be used by each input and the order for supplying
their numerical values.

**Orientation and angular speed use different values.** Euler angles can wrap
at angle boundaries and become ambiguous near a 90-degree pitch. Quaternions
avoid those angle-wrap issues, but use four numbers to represent orientation.
Angular speed still has three components. Do not treat quaternion values as
angular speeds or expect the saved orientation and speed arrays to have matching
lengths. Section 4 shows the layouts for each joint.
