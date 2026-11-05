Developer's Guide
=================

This guide is for extending, maintaining, and validating the 3D package. It
follows the 2D project's developer-guide pattern (package map, public API,
workflow, and tests), while describing the current 3D implementation only.
The related planar project is
`Project-SEA-Stack/Python_Multibody_for_Everybody
<https://github.com/Project-SEA-Stack/Python_Multibody_for_Everybody>`_.
The 3D source of truth is
`AlvaroDFC/3d_m4e <https://github.com/AlvaroDFC/3d_m4e>`_.

Project Layout
--------------

The following is a manually maintained overview of the active 3D source tree;
Sphinx formats this literal tree but does not generate it from the filesystem.

.. code-block:: text

   3d_m4e/
   +-- source/multibody_3d/
   |   +-- __init__.py                 # public package exports
   |   +-- multibody_core/
   |   |   +-- topology_3d.py          # tree validation and path/index data
   |   |   +-- joint_system_3d.py      # joint model, axes, topology
   |   |   +-- joint_coordinate_3d.py  # user/internal q and speed layouts
   |   |   +-- velocity_transformation_3d.py  # B/Bdot and kinematics caches
   |   |   +-- kinematics_runtime_3d.py       # trajectory postprocessing
   |   |   +-- mass_symbolic_3d.py    # symbolic body inertia and mass
   |   |   +-- mass_runtime_3d.py     # numeric mass/EOM/energy evaluation
   |   |   +-- force_definition_3d.py # user force contract and parsing
   |   |   +-- forces_symbolic_3d.py  # symbolic body wrenches and spring PE
   |   |   +-- forces_runtime_3d.py   # JAX force evaluation
   |   |   +-- points_3d.py           # point records and evaluation
   |   |   +-- runtime_context_3d.py  # shared frozen postprocess inputs
   |   |   +-- mbd_system_3d.py       # public model facade and evaluator setup
   |   |   +-- integrator_3d.py       # JAX/diffrax ODE solve
   |   +-- plotting/
   |   |   +-- plotting_main.py       # result plots and energy summary
   |   +-- tests/                     # unit and integration tests
   +-- example1.py ... example9.py   # runnable model inputs
   +-- main.py                       # selected example runner
   +-- documentation/                # Sphinx source and build output

The tree is an orientation aid, not a generated package manifest. The active
classes and functions in the source files and `multibody_3d.__all__` are
canonical.

Architecture and ownership
--------------------------

The package keeps model construction, symbolic expressions, numerical
execution, and display code separate:

1. `topology_3d.py` validates a rooted tree and computes path/index metadata.
2. `joint_system_3d.py` validates joint definitions and owns body/joint
   connectivity and normalized axes.
3. `joint_coordinate_3d.py` assigns user configuration, internal quaternion
   configuration, generalized-speed slices, and names.
4. `velocity_transformation_3d.py` constructs symbolic kinematics and the
   velocity maps `B` and `Bdot`; its cache/block layers keep shared geometry
   work explicit.
5. `mass_symbolic_3d.py`, `forces_symbolic_3d.py`, and `points_3d.py` define
   mass, body wrenches, and attachment-point expressions.
6. `mass_runtime_3d.py` and `forces_runtime_3d.py` compile numerical evaluators;
   `runtime_context_3d.py` shares frozen model inputs across postprocessing.
7. `MbdSystem3D` provides the user-facing facade and prepares reusable
   evaluators. `integrator_3d.py` owns time integration.
8. `plotting/` consumes results and returns Matplotlib figures; it is not part
   of symbolic construction or the integration loop.

System construction workflow
----------------------------

A typical user path is:

.. code-block:: text

   example module
       -> MbdSystem3D.from_example(example)
       -> validate topology and build coordinate slices
       -> construct symbolic B/Bdot, mass, points, and force caches
       -> compile reusable evaluators
       -> integrate with JAX/diffrax
       -> postprocess energy/kinematics and plot

`MbdSystem3D` does not rebuild static topology or compiled evaluators on each
ODE evaluation. Preserve that boundary when adding runtime behavior. The
integration RHS is JAX-traceable; do not introduce NumPy or Python-side I/O in
its repeated evaluation path.

The generalized-speed equation assembled by the runtime is

.. math::

   M(q)\dot{u}=B^T\left(f-\gamma-M_{body}\dot{B}u\right),
   \qquad M(q)=B^T M_{body}B.

Quaternion configuration rates are handled separately from generalized speeds.
For S/F joints, configuration dimension and speed dimension differ.

Public API
----------

`MbdSystem3D` is the primary entry point. Its `from_data` constructor takes a
geometry dictionary; `from_example` reads the example module's `data`, force,
point, inertia, and parameter attributes. The top-level `multibody_3d` package
also exports joint models, coordinate helpers, force/point definitions,
kinematics results, and trajectory CSV export. See the
:doc:`API reference <api/index>` for curated public summaries and the source
package `__all__` for the full export list.

Validation and tests
--------------------

The tests are under `source/multibody_3d/tests/` and use pytest. From the
repository root:

.. code-block:: console

   python -m pytest source/multibody_3d/tests

The suites cover topology, coordinate layout, symbolic and numeric `B`/`Bdot`,
quaternion compilation, force evaluation, JAX execution, system integration,
and plotting. Run the tests relevant to the changed layer, then the full suite
for cross-layer changes. Do not treat successful integration alone as a
physics validation; compare against an analytical or trusted reference when
changing dynamics.

Build the documentation with the commands in :doc:`Installation
<installation>`. The 2D repository's user and developer guides are useful
references for shared M4E concepts, but keep 3D-specific contracts, examples,
and limitations authoritative here.
