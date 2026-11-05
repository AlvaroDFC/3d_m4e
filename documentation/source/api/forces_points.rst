Forces and points
=================

The user-facing `Force` and `Initial_Points` formats are parsed into typed
definitions and evaluated as world-frame body wrenches. See the
:doc:`user guide <../userDocumentation>` for the input formats.

.. list-table::
	 :header-rows: 1

	 * - Public symbols
		 - Responsibility
	 * - ``ForcesDefinition3D``, ``parse_force_dict``
		 - Parsed force records and input validation.
	 * - ``CGForceDef``, ``PointForceDef``
		 - Applied force/moment records at CGs or declared body points.
	 * - ``TensionSpringDef``, ``TensionDamperDef``
		 - Translational spring/damper records between point endpoints.
	 * - ``TorsionSpringDef``, ``TorsionDamperDef``
		 - Rotational elements attached to revolute-joint indices.
	 * - ``GravityDef``
		 - Per-body gravity configuration.
	 * - ``PointRecord3D``, ``PointsEvalResult``
		 - Symbolic point identity and evaluated point results.
	 * - ``force_reduction_at_point``
		 - Reduce a point force to force and moment about its body's CG.

These names are exported from ``multibody_3d``; see
``source/multibody_3d/__init__.py`` for the complete public export list.
