# main file to run multibody 3D examples
from multibody_3d import MbdSystem3D
import example9 as ex
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
# q_int_np = np.array([0. ,0. ,0. ,np.cos(np.pi/6) ,0.9659*np.sin(np.pi/6) ,0. ,0.2588*np.sin(np.pi/6), 0.1, 0.2, 0.3])
# qd_np = np.array([1. ,1. ,2. , 1., 2., 3., 0.1, 0.2, 0.3])
# # Example 5 ic: R - R- P - R
# # q_int_np = np.array([np.pi/6, 0.3, 2, 1.1])
# # qd_np = np.array([0.7, -0.3, 0.4, 1.2])
# # Example 6 ic: R + U
# # q_int_np = np.array([np.pi/6, 0.3, 2])
# # qd_np = np.array([0.7, -0.4, 1.1])

# Build user-facing mainNumVars = [q_user, qd, body_params, force_params]
# example7 has R and L as body params; supply numeric values here.
# q_user_np    = mbd.map_q_int_to_q_user(q_int_np)
# body_params  = np.array([0.5, 1.0])   # R=0.5, L=1.0
# points_params = np.array([0.3])         # d4=0.3 (endpoint location along link 3 axis)
# force_params = np.array([1., 1., 1.])            # Fx1=1, Fy1=1, Fz1=1 (world-frame CG force on body 1)
# example 9, the double pendulum, has no body params or force params, so mainNumVars is just [q_user, qd].
q_int_np = np.array([0.0 ,0.0])
qd_np = np.array([0.0,0.0])
q_user_np    = mbd.map_q_int_to_q_user(q_int_np)
# mainNumVars  = np.concatenate([q_user_np, qd_np, body_params, force_params, points_params])
mainNumVars  = np.concatenate([q_user_np, qd_np])
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
# _ = mbd.evaluate_points(mainNumVars)

# t0 = time()
# for _ in range(n):
#     pts = mbd.evaluate_points(mainNumVars)
# print(f"Points eval:  {(time()-t0)/n*1e6:.2f} µs/call")

# print("\nPoints – r_abs_cg (CG positions, world frame):")
# print(np.array(pts.r_abs_cg))

# print("\nPoints – r_abs_body (body-attached points, world frame):")
# print(np.array(pts.r_abs_body))

# print("\nPoints – rho_abs_body (CG-relative moment arms, world frame):")
# print(np.array(pts.rho_abs_body))

# print("\nPoints – r_abs_gr (ground reference points, world frame):")
# print(np.array(pts.r_abs_gr))

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
# body_inertia is now read directly from the example module by from_example().
# The second MbdSystem3D construction below is only needed if from_example()
# was called without body_inertia (e.g. legacy examples without that attribute).
# For example9 and newer examples, body_inertia is already on the module.
_ex_bi = getattr(ex, 'body_inertia', None)
if _ex_bi is not None:
    # Example already provides body_inertia; the first from_example() call
    # already built the full system — no rebuild needed.
    pass
else:
    # Legacy path: reconstruct body_inertia from separate body_mass / J_body_imm.
    _ex_masses = getattr(ex, 'body_mass', {})
    _ex_J      = getattr(ex, 'J_body_imm', {})
    _ex_bi = {
        b: {
            'mass': float(_ex_masses.get(b, 1.0)),
            'J':   np.asarray(_ex_J[b], dtype=float) if b in _ex_J
                   else np.eye(3) * 0.01,
        }
        for b in range(1, mbd.NBodies + 1)
    }
    mbd = MbdSystem3D(
        data=ex.data,
        body_data_sym=getattr(ex, 'body_data_sym', {}),
        force_points_sym={**getattr(ex, 'force_points_sym', {}),
                          **getattr(ex, 'force_sym',        {})},
        points_sym=getattr(ex, 'points_sym', {}),
        Initial_Points=getattr(ex, 'Initial_Points', {}),
        Force=getattr(ex, 'Force', {}),
        body_inertia=_ex_bi,
    )

import diffrax as _diffrax
sol = mbd.integrate(
    mainNumVars,
    tspan=(0.0, 100.0),
    dt=0.01,
    rtol=1e-6,
    atol=1e-6,
    algorithm="Dopri5",
    max_steps=500_000,
)

print(f"\nIntegration {'succeeded' if sol.result == _diffrax.RESULTS.successful else 'FAILED'}")
print(f"Output shape (time x state): {sol.ys.shape}")
n_qi = mbd.total_cfg_dof
print(f"Final q_int: {np.array(sol.ys[-1, :n_qi]).round(4)}")
print(f"Final qd:    {np.array(sol.ys[-1, n_qi:]).round(4)}")

## Added part
import matplotlib.pyplot as plt

ts   = np.array(sol.ts)
qs   = np.array(sol.ys[:, :n_qi])    # (n_steps, n_qi)
qds  = np.array(sol.ys[:, n_qi:])    # (n_steps, n_dof)

q_labels  = mbd.coords.names_int
qd_labels = mbd.coords.names_d

fig, axes = plt.subplots(2, 1, figsize=(10, 7), sharex=True)

ax_q = axes[0]
for i in range(qs.shape[1]):
    ax_q.plot(ts, qs[:, i], label=q_labels[i] if i < len(q_labels) else f"q{i}")
ax_q.set_ylabel("Generalized coordinates")
ax_q.legend()
ax_q.grid(True)

ax_qd = axes[1]
for i in range(qds.shape[1]):
    ax_qd.plot(ts, qds[:, i], label=qd_labels[i] if i < len(qd_labels) else f"qd{i}")
ax_qd.set_ylabel("Generalized velocities")
ax_qd.set_xlabel("Time [s]")
ax_qd.legend()
ax_qd.grid(True)

fig.suptitle("Coordinates and velocities over time")
plt.tight_layout()
plt.show()

####### Kinetic and potential energy over time ##################################
energy  = mbd.compute_energy(sol, mainNumVars)
KE_arr, PE_arr, E_total  = energy.KE, energy.PE, energy.E_total
KE_body, PE_body         = energy.KE_body, energy.PE_body

# ── System totals ────────────────────────────────────────────────────────────
fig_e, ax_e = plt.subplots(figsize=(10, 4))
ax_e.plot(ts, KE_arr,  label="KE")
ax_e.plot(ts, PE_arr,  label="PE")
ax_e.plot(ts, E_total, label="Total E", linestyle="--", color="k")
ax_e.set_xlabel("Time [s]")
ax_e.set_ylabel("Energy [J]")
ax_e.set_title("Kinetic, potential and total mechanical energy")
ax_e.legend()
ax_e.grid(True)
plt.tight_layout()
plt.show()

# ── Per-body energies ────────────────────────────────────────────────────────
fig_b, axes_b = plt.subplots(2, 1, figsize=(10, 7), sharex=True)

ax_ke = axes_b[0]
ax_pe = axes_b[1]
for b in range(mbd.NBodies):
    lbl = f"body {b + 1}"
    ax_ke.plot(ts, KE_body[:, b], label=lbl)
    ax_pe.plot(ts, PE_body[:, b], label=lbl)

ax_ke.set_ylabel("KE [J]")
ax_ke.set_title("Per-body kinetic energy")
ax_ke.legend()
ax_ke.grid(True)

ax_pe.set_ylabel("PE [J]")
ax_pe.set_xlabel("Time [s]")
ax_pe.set_title("Per-body potential energy")
ax_pe.legend()
ax_pe.grid(True)

plt.tight_layout()
plt.show()

# ── Per-body linear and angular velocities ───────────────────────────────────
kin = mbd.compute_kinematics(sol, mainNumVars)
v_lin_all = kin.v_cg      # (n_steps, NB, 3)
v_ang_all = kin.omega_cg  # (n_steps, NB, 3)
_NB_e = mbd.NBodies

fig_v, axes_v = plt.subplots(_NB_e, 2, figsize=(12, 4 * _NB_e), sharex=True)
if _NB_e == 1:
    axes_v = axes_v[np.newaxis, :]   # ensure 2-D indexing for single-body case

xyz = ["x", "y", "z"]
for b in range(_NB_e):
    ax_lin = axes_v[b, 0]
    ax_ang = axes_v[b, 1]
    for k in range(3):
        ax_lin.plot(ts, v_lin_all[:, b, k], label=f"v_{xyz[k]}")
        ax_ang.plot(ts, v_ang_all[:, b, k], label=f"ω_{xyz[k]}")
    ax_lin.set_title(f"Body {b + 1} – linear velocity (world frame)")
    ax_ang.set_title(f"Body {b + 1} – angular velocity (world frame)")
    ax_lin.set_ylabel("m/s")
    ax_ang.set_ylabel("rad/s")
    ax_lin.legend()
    ax_ang.legend()
    ax_lin.grid(True)
    ax_ang.grid(True)

for ax in axes_v[-1, :]:
    ax.set_xlabel("Time [s]")

plt.tight_layout()
plt.show()

# ── CG position and orientation trajectories ─────────────────────────────────
fig_p, axes_p = plt.subplots(_NB_e, 2, figsize=(12, 4 * _NB_e), sharex=True)
if _NB_e == 1:
    axes_p = axes_p[np.newaxis, :]

euler_labels = ["roll", "pitch", "yaw"]
for b in range(_NB_e):
    ax_r = axes_p[b, 0]
    ax_o = axes_p[b, 1]
    for k in range(3):
        ax_r.plot(ts, kin.r_cg[:, b, k], label=xyz[k])
        ax_o.plot(ts, kin.euler_cg[:, b, k], label=euler_labels[k])
    ax_r.set_title(f"Body {b + 1} – CG position (world frame)")
    ax_o.set_title(f"Body {b + 1} – CG orientation (intrinsic Z-Y-X)")
    ax_r.set_ylabel("m")
    ax_o.set_ylabel("rad")
    ax_r.legend()
    ax_o.legend()
    ax_r.grid(True)
    ax_o.grid(True)

for ax in axes_p[-1, :]:
    ax.set_xlabel("Time [s]")

plt.tight_layout()
plt.show()

