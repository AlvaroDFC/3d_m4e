# main file to run multibody 3D examples
from multibody_3d import JointSystem3D, build_joint_coordinates, VelocityTransformation3D
# from multibody_3d.multibody_core.kinematics_cache_3d import demo_write_pairs_chain
from example1 import data
import numpy as np

from sympy import pprint

sys = JointSystem3D.from_data(data)
print(sys.Btrack)

# Display in a table
table = sys.summary_table(precision=3)

jCoords = build_joint_coordinates(sys)
print("Joint coordinates:")
# display all the joint coordinates on screen
for coord in jCoords.q:
    print(f"  {coord.name}")

kin = VelocityTransformation3D(sys)
kinCache = kin.build_cache_symbolic(jCoords.q)

# assemble the B-matrix symbolically
B = kin.assemble_B_symbolic(jCoords.q, kinCache) #NOTE: do i pass q or q_int here?
BD = kin.assemble_Bdot_symbolic(jCoords.q, jCoords.qd, kinCache)
print("\nB-matrix:")
# pprint(B)
print("\nBtrack:")
print(kin.Btrack)


# Compile B-matrix into a function of q
B_func = kin.compile_B_lambdified(jCoords.q)
Bdot_func = kin.compile_Bdot_lambdified(jCoords.q, jCoords.qd)
print("\nB-matrix function:")
# print(kin.B_explicit)

# q0 = np.array([0. ,0. ,0. ,1 , 0., 0., 0., 0.1, 0.2, 0.3])  # OMT ic
q0 = np.array([0.1 ,0.2 ,0.3])  # hover ic
# q0 = np.array([0.9659, 0., 0., 0.2588, -np.pi/6]) # spherical revolute pendulum ic
# q0 = np.array([np.pi/6, 0.3, 2, 1.1])  # example5 ic
# q0 = np.array([np.pi/6, 0.3, 2])  # ex6 ic universal revolute
B_q0 = B_func(np.hstack((q0)))  # evaluate B at q0, qd0

print("\nB-matrix evaluated at q0:")
np.set_printoptions(precision=5, suppress=True, linewidth=200)
print(B_q0)



print("\nBdot-matrix:")
# qd0 = np.array([0. ,0. ,0. , 0., 0., 0., 0.1, 0.2, 0.3])  # OMT ic
qd0 = np.array([1., 3., 3.])  # hover ic
# qd0 = np.array([0.3, -0.2, 0.5, 1.1]) # spherical revolute pendulum ic
# qd0 = np.array([0.7, -0.3, 0.4, 1.2])  # example5 ic
# qd0 = np.array([0.7, -0.4, 1.1]) # ex6 ic universal revolute

Bdot_q0 = Bdot_func(q0,qd0)
print(Bdot_q0)
print("Reached end")
