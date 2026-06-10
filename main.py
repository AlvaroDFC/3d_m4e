# main file to run multibody 3D examples
from multibody_3d import MbdSystem3D
import example7 as ex
import numpy as np
from time import time

import jax
print(jax.devices())

t0 = time()
mbd = MbdSystem3D.from_example(ex)
print(mbd)

# Display in a table
mbd.summary_table(precision=3)

# Runtime backend (B_func / Bdot_func) is built automatically by __post_init__.
# No manual setup required.

# ── Initial conditions for numerical evaluation  ──────────────────────────────
# # Example 1 ic: 3 revolute joints
# # q_int_np = np.array([0.1 ,0.2 ,0.3])
# # qd_np = np.array([1., 3., 3.])
# # Example 2 ic: spherical revolute pendulum
# # q_int_np = np.array([0.9659, 0., 0., 0.2588, -np.pi/6])
# # qd_np = np.array([0.3, -0.2, 0.5, 1.1])
# # Example 3 ic: Cylindrical + revolut + spherical
# # q_int_np = np.array([0.3, -0.2, 0.5, 0.4, 0.2, 0, 0])
# # qd_np = np.array([0.7, -0.3, 0.4, 1.2])
# # Example 4 ic
# Internal q for constructing q_user (F-joint uses quaternion internally)
q_int_np = np.array([0. ,0. ,0. ,np.cos(np.pi/6) ,0.9659*np.sin(np.pi/6) ,0. ,0.2588*np.sin(np.pi/6), 0.1, 0.2, 0.3])
qd_np = np.array([1. ,1. ,2. , 1., 2., 3., 0.1, 0.2, 0.3])
# # Example 5 ic: R - R- P - R
# # q_int_np = np.array([np.pi/6, 0.3, 2, 1.1])
# # qd_np = np.array([0.7, -0.3, 0.4, 1.2])
# # Example 6 ic: R + U
# # q_int_np = np.array([np.pi/6, 0.3, 2])
# # qd_np = np.array([0.7, -0.4, 1.1])

# Build user-facing mainNumVars = [q_user, qd, body_params, force_params]
# example7 has R and L as body params; supply numeric values here.
q_user_np    = mbd.map_q_int_to_q_user(q_int_np)
body_params  = np.array([0.5, 1.0])   # R=0.5, L=1.0
points_params = np.array([0.3])         # d4=0.3 (endpoint location along link 3 axis)
force_params = np.array([1., 1., 1.])            # Fx1=1, Fy1=1, Fz1=1 (world-frame CG force on body 1)
mainNumVars  = np.concatenate([q_user_np, qd_np, body_params, force_params, points_params])

# ####################### Time evaluation ############################################
# Testing setup
n = 1

# Warmup (triggers JIT compilation)
_ = mbd.evaluate_B(mainNumVars)
_ = mbd.evaluate_Bdot(mainNumVars)

# Timed — both B and Bdot run through the pre-compiled B_func / Bdot_func
t0 = time()
for _ in range(n):
    B_jax    = mbd.evaluate_B(mainNumVars)
    Bdot_jax = mbd.evaluate_Bdot(mainNumVars)
print(f"JAX JIT:      {(time()-t0)/n*1e6:.2f} µs/call")

np.set_printoptions(precision=4, suppress=True)

print("\nB (JAX):")
print(B_jax)

print("\nBdot (JAX):")
print(Bdot_jax)


####### Symbolic B blocks (for inspection, not timed) #####################################
from multibody_3d import BlockInspector

# Build all symbolic B blocks indexed by (body_id, joint_index)
blocks = mbd.vt.build_B_blocks_symbolic(mbd.q_int)
Bdot_blocks = mbd.vt.build_Bdot_blocks_symbolic(mbd.q_int, mbd.qd_int)

# Access a specific block — e.g. body=1, joint=0
blk = blocks[(1, 0)]
# print(blk.matrix)       # sympy.Matrix (6×m)
# print(blk.d_kj)         # 3×1 position vector
# print(blk.U_j)          # 3×m axis/basis
# print(blk.joint_type)   # 'R', 'P', 'U', etc.

# # Pretty-print all blocks
# BlockInspector.display_B_blocks(blocks)

# # Ingredients only (faster, no full matrix expansion)
# BlockInspector.display_B_blocks(blocks, show_matrix=True)

####### Points evaluator ######################################################
# Warm-up
_ = mbd.evaluate_points(mainNumVars)

t0 = time()
for _ in range(n):
    pts = mbd.evaluate_points(mainNumVars)
print(f"Points eval:  {(time()-t0)/n*1e6:.2f} µs/call")

print("\nPoints – r_abs_cg (CG positions, world frame):")
print(np.array(pts.r_abs_cg))

print("\nPoints – r_abs_body (body-attached points, world frame):")
print(np.array(pts.r_abs_body))

print("\nPoints – rho_abs_body (CG-relative moment arms, world frame):")
print(np.array(pts.rho_abs_body))

print("\nPoints – r_abs_gr (ground reference points, world frame):")
print(np.array(pts.r_abs_gr))

####### Forces evaluator ######################################################
# Warm-up
_ = mbd.evaluate_forces(mainNumVars)

t0 = time()
for _ in range(n):
    forces = mbd.evaluate_forces(mainNumVars)
print(f"Forces eval:  {(time()-t0)/n*1e6:.2f} µs/call")

print("\nForces – CG wrenches (NBodies × 6) [Fx,Fy,Fz,Mx,My,Mz]:")
print(np.array(forces.cg))

print("\nForces – total wrench (NBodies × 6):")
print(np.array(forces.total))

print("\nForces – spring potential energy:")
print(float(forces.spring_potential_energy))
