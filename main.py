# main file to run multibody 3D examples
from multibody_3d import MbdSystem3D
import example4
import numpy as np
from time import time

import jax
print(jax.devices())

t0 = time()
mbd = MbdSystem3D.from_example(example4)
print(mbd)

# Display in a table
mbd.summary_table(precision=3)

params = mbd.build_numeric_params()   # constant geometry — build once, reuse
t1 = time() - t0
print(f"Setup: {t1:.2f} s")

# ── Initial conditions for numerical evaluation  ──────────────────────────────
# Example 1 ic: 3 revolute joints
# q_int_np = np.array([0.1 ,0.2 ,0.3])
# qd_np = np.array([1., 3., 3.])
# Example 2 ic: spherical revolute pendulum
# q_int_np = np.array([0.9659, 0., 0., 0.2588, -np.pi/6])
# qd_np = np.array([0.3, -0.2, 0.5, 1.1])
# Example 3 ic: Cylindrical + revolut + spherical
# q_int_np = np.array([0.3, -0.2, 0.5, 0.4, 0.2, 0, 0])
# qd_np = np.array([0.7, -0.3, 0.4, 1.2])
# Example 4 ic
q_int_np = np.array([0. ,0. ,0. ,np.cos(np.pi/6) ,0.9659*np.sin(np.pi/6) ,0. ,0.2588*np.sin(np.pi/6), 0.1, 0.2, 0.3])
qd_np = np.array([1. ,1. ,2. , 1., 2., 3., 0.1, 0.2, 0.3])
# Example 5 ic: R - R- P - R
# q_int_np = np.array([np.pi/6, 0.3, 2, 1.1])
# qd_np = np.array([0.7, -0.3, 0.4, 1.2])
# Example 6 ic: R + U
# q_int_np = np.array([np.pi/6, 0.3, 2])
# qd_np = np.array([0.7, -0.4, 1.1])

####################### Time evaluation ############################################
# Testing setup
n = 100

# JAX eager
t0 = time()
for _ in range(n):
    B_jax    = mbd.evaluate_B_jax(q_int_np, params=params)
    Bdot_jax = mbd.evaluate_Bdot_jax(q_int_np, qd_np, params=params)
print(f"JAX eager:    {(time()-t0)/n*1e6:.2f} µs/call")

# JAX JIT — build, warmup, then time
B_eval    = mbd.build_B_evaluator_jax(params=params)
Bdot_eval = mbd.build_Bdot_evaluator_jax(params=params)
_ = B_eval(q_int_np); _ = Bdot_eval(q_int_np, qd_np)  # compile

t0 = time()
for _ in range(n):
    B_jit    = B_eval(q_int_np)
    Bdot_jit = Bdot_eval(q_int_np, qd_np)
print(f"JAX JIT:      {(time()-t0)/n*1e6:.2f} µs/call")

np.set_printoptions(precision=4, suppress=True)
print("\nB (JAX):")
print(B_jax)

print("\nB (JIT):")
print(B_jit)

print("\nBdot (JIT):")
print(Bdot_jit)

####### Symbolic B blocks (for inspection, not timed) #####################################
from multibody_3d import BlockInspector

# Build all symbolic B blocks indexed by (body_id, joint_index)
blocks = mbd.vt.build_B_blocks_symbolic(mbd.q_int)
Bdot_blocks = mbd.vt.build_Bdot_blocks_symbolic(mbd.q_int, mbd.qd_int)

# Access a specific block — e.g. body=1, joint=0
blk = blocks[(1, 0)]
print(blk.matrix)       # sympy.Matrix (6×m)
print(blk.d_kj)         # 3×1 position vector
print(blk.U_j)          # 3×m axis/basis
print(blk.joint_type)   # 'R', 'P', 'U', etc.

# Pretty-print all blocks
BlockInspector.display_B_blocks(blocks)

# Ingredients only (faster, no full matrix expansion)
BlockInspector.display_B_blocks(blocks, show_matrix=False)
