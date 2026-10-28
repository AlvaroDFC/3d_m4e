# main file to run multibody 3D examples
from multibody_3d import MbdSystem3D
import example7 as ex
import numpy as np
from time import time
import matplotlib.pyplot as plt

import jax
print(jax.devices())

t0 = time()
mbd = MbdSystem3D.from_example(ex)
print(mbd)

# Display in a table
mbd.summary_table(precision=3)

# Runtime backend (B_func / Bdot_func) is built automatically by __post_init__.
# No manual setup required.

# ── Initial conditions and numerical parameters (defined in each example) ─────
q_user_np     = ex.q_user_np
qd_np         = ex.qd_np
body_params   = ex.body_params
force_params  = ex.force_params
points_params = ex.points_params

mainNumVars  = np.concatenate([q_user_np, qd_np, body_params, force_params, points_params])
# ####################### Time evaluation ############################################
# Testing setup
# Warmup (triggers JIT compilation)
_ = mbd.evaluate_B(mainNumVars)
_ = mbd.evaluate_Bdot(mainNumVars)
n = 1
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
# blk = blocks[(1, 0)]
# print(blk.matrix)       # sympy.Matrix (6×m)
# print(blk.d_kj)         # 3×1 position vector
# print(blk.U_j)          # 3×m axis/basis
# print(blk.joint_type)   # 'R', 'P', 'U', etc.
# # Pretty-print all blocks
# BlockInspector.display_B_blocks(blocks)

# # Ingredients only (faster, no full matrix expansion)
# BlockInspector.display_B_blocks(blocks, show_matrix=True)

####### Points / joints kinematics ############################################
# kin = mbd.compute_kinematics(sol, mainNumVars)   # with_velocities=False to skip rates
# print(kin.r_cg[-1])      # (NBodies, 3) CG positions, world frame
# print(kin.r_pts[-1])     # (n_body_pts, 3) body-attached points, world frame
# print(kin.r_joint[-1])   # (NJoints, 3) joint locations, world frame
# print(kin.r_gr)          # (n_gr, 3) ground points, world frame

####### Forces evaluator ######################################################
# Warm-up
# _ = mbd.evaluate_forces(mainNumVars)

# t0 = time()
# for _ in range(n):
#     forces = mbd.evaluate_forces(mainNumVars)
# print(f"Forces eval:  {(time()-t0)/n*1e6:.2f} µs/call")

# print("\nForces – CG wrenches (NBodies × 6) [Fx,Fy,Fz,Mx,My,Mz]:")
# print(np.array(forces.cg))

# print("\nForces – total wrench (NBodies × 6):")
# print(np.array(forces.total))

# print("\nForces – spring potential energy:")
# print(float(forces.spring_potential_energy))

####### Integration ######################################################
import diffrax as _diffrax
sol = mbd.integrate(
    mainNumVars,
    tspan=ex.tspan,
    dt=ex.TimeStep,
    **getattr(ex, 'integrator_kwargs', {}),
)

print(f"\nIntegration {'succeeded' if sol.result == _diffrax.RESULTS.successful else 'FAILED'}")
print(f"Output shape (time x state): {sol.ys.shape}")
n_qi = mbd.total_cfg_dof
print(f"Final q_int: {np.array(sol.ys[-1, :n_qi]).round(4)}")
print(f"Final qd:    {np.array(sol.ys[-1, n_qi:]).round(4)}")

## Plots and energy summary
from multibody_3d import plotting as plot

plot.plot_coordinates(sol, n_qi, mbd.coords.names_int, mbd.coords.names_d)

# energy = mbd.compute_energy(sol, mainNumVars)
# plot.print_system_energy(energy)
# print(f"Internal steps used  : {int(sol.stats['num_steps'])}")
# plot.plot_system_energy(energy)
# plot.plot_body_energy(energy)

kin = mbd.compute_kinematics(sol, mainNumVars)
# plot.plot_body_velocities(kin)
# plot.plot_body_trajectories(kin)

# 3-D snapshot and animation; keep `anim` referenced while the window is open
from multibody_3d import compute_forces_states
forces = compute_forces_states(mbd, sol.ts, sol.ys, mainNumVars) if mbd.forces_func is not None else None
plot.plot_snapshot(mbd, kin, frame=0, forces=forces)
anim = plot.animate_multibody(mbd, kin, forces=forces, frame_step=2, fps=20)   # save_path="anim.gif" writes a file

plt.show()

