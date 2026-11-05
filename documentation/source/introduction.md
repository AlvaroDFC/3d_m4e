# About M4E 3D
**For questions, contact:**

**sahand.sabet@nlr.gov, alvaro.diazflorescaminero@nlr.gov** 


M4E 3D models multibody systems using Joint Coordinate formulation [Nikravesh](https://www.sciencedirect.com/science/article/pii/002074629090046C?casa_token=TbDjuTm71mUAAAAA:AS_dvD19LXsPwN0m4nc4FMA6KtldkQmVlpccMnO4I3HvN7bUwpawiHHKYDHn5MLR3d4KHDbN2Bs). A model is
defined in a Python example module; a main script constructs the system, runs a
simulation, and inspects or plots the results.

This guide follows the gradual example-first teaching style of the 2D
[Multibody4Everybody documentation](https://github.com/Project-SEA-Stack/Python_Multibody_for_Everybody) to explain the three-dimensional inputs and conventions directly. The 2D
repository is a reference, not a prerequisite. M4E 3D has additional joint
types, spatial inertia tensors, quaternion-based internal orientation state,
and a JAX/diffrax simulation backend.

## Where to start

1. Follow [Installation and documentation build](installation.md) to install the package and simulation
   dependencies.
2. Work through the [3D multibody dynamics user guide](userDocumentation.md), starting with its
   numerical two-link example.
3. Use the {doc}`developer guide <developersGuide>` when extending the package
   or investigating its internals.
4. Use the {doc}`API reference <api/index>` for documented public classes and
   functions.

Source and examples are in the
[3D GitHub repository](https://github.com/AlvaroDFC/3d_m4e). The related 2D
project is maintained at
[Project-SEA-Stack/Python_Multibody_for_Everybody](https://github.com/Project-SEA-Stack/Python_Multibody_for_Everybody).
