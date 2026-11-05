Plotting
========

The ``multibody_3d.plotting`` functions accept already-computed simulation
results and return Matplotlib figures. They do not call `plt.show()`; the
caller controls display and saving.

.. list-table::
	 :header-rows: 1

	 * - Public function
		 - Input
		 - Result
	 * - ``plot_coordinates``
		 - Diffrax solution and internal coordinate count
		 - Coordinate and speed history figure.
	 * - ``plot_system_energy``
		 - Energy result
		 - Kinetic, potential, and total energy figure.
	 * - ``plot_body_energy``
		 - Energy result
		 - Per-body kinetic and gravitational potential energy figure.
	 * - ``plot_body_velocities``
		 - Kinematics result
		 - World-frame linear and angular velocity figure.
	 * - ``plot_body_trajectories``
		 - Kinematics result
		 - CG position and Euler orientation histories.
	 * - ``print_system_energy``
		 - Energy result
		 - Text summary of initial, final, and maximum energy change.
