Core model and kinematics
=========================

The main public entry point is `MbdSystem3D` from `multibody_3d`. Construct it
from an example module with `MbdSystem3D.from_example(example)` or from raw
geometry with `MbdSystem3D.from_data(data)`.

.. list-table::
	 :header-rows: 1

	 * - Public symbol
		 - Responsibility
	 * - ``MbdSystem3D``
		 - Owns the model, builds reusable symbolic/numeric evaluators, and exposes simulation and postprocessing methods.
	 * - ``JointSystem3D``, ``Joint3D``, ``JointType``
		 - Joint definitions, validation, topology, and joint-coordinate bookkeeping.
	 * - ``CoordBundle``, ``build_joint_coordinates``
		 - User-facing and internal configuration/speed arrays and their slices.
	 * - ``VelocityTransformation3D``
		 - Symbolic kinematics and velocity transformations ``B`` and ``Bdot``.
	 * - ``integrate_3d``
		 - JAX/diffrax ODE integration backend. Normally call ``MbdSystem3D.integrate``.
	 * - ``KinematicsResult``, ``compute_kinematics_3d``
		 - Trajectory postprocessing for body CGs and declared points.
	 * - ``save_trajectory_csv``
		 - Export CG position and linear-velocity trajectories.

Common `MbdSystem3D` methods:

.. list-table::
	 :header-rows: 1

	 * - Method
		 - Use
	 * - ``integrate(mainNumVars, ...)``
		 - Solve the ODE over the requested time interval.
	 * - ``compute_kinematics(sol, mainNumVars)``
		 - Recover CG and body-point positions, orientations, and velocities.
	 * - ``compute_energy(sol, mainNumVars)``
		 - Evaluate system and per-body energy outputs.
	 * - ``evaluate_eom_kernel(mainNumVars)``
		 - Evaluate ``B``, ``Bdot``, body mass, and generalized mass at one state.
	 * - ``summary_table()``
		 - Print the joint summary.

See the :doc:`user guide <../userDocumentation>` for executable examples and
the top-level package ``__all__`` in ``source/multibody_3d/__init__.py`` for the
complete exported symbol list.
