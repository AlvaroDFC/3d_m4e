# example_mbd_system_3d.py
"""
Phase 1 usage example for MbdSystem3D.

System topology
---------------
3-body chain:  ground(0) --[F]--> body1 --[R]--> body2 --[R]--> body3

  - Joint 0: Floating (F) — 6 DOF, 7 internal config (3 trans + 4 quat)
  - Joint 1: Revolute (R) about +y — 1 DOF
  - Joint 2: Revolute (R) about +z — 1 DOF

  total_dof      = 8   (speed vector length)
  total_cfg_dof  = 9   (internal config: 7 + 1 + 1)
  total_user_dof = 9   (user config: same as internal when rot_param='quat')

When to use which API layer
---------------------------
  mbd.vt.*                — power-user / block-level inspection (symbolic layers)
  mbd.assemble_B_symbolic()  — symbolic B/Bdot for analysis (auto-supplies q symbols)
  mbd.evaluate_B_jax()    — internal-coordinate runtime (q_int already known)
  mbd.evaluate_B_from_user_q()  — user-coordinate runtime (Euler/quat q_user as input)
"""

import numpy as np

from multibody_3d import MbdSystem3D

# ── System definition ────────────────────────────────────────────────────────

data = {
    "NBodies": 3,
    "joints": [
        (0, 1),  # ground → body1, floating
        (1, 2),  # body1  → body2, revolute
        (2, 3),  # body2  → body3, revolute
    ],
    "types": ["F", "R", "R"],
    "parent_cg_to_joint": [
        [0.0, 0.0, 0.0],   # F:  joint at ground origin
        [1.5, 0.0, 0.0],   # R1: joint 1.5 m along body1 x-axis
        [1.0, 0.0, 0.0],   # R2: joint 1.0 m along body2 x-axis
    ],
    "joint_to_child_cg": [
        [0.0, 0.0, 0.0],   # body1 CG coincident with F joint
        [-0.75, 0.0, 0.0], # body2 CG 0.75 m behind R1 joint
        [-0.5,  0.0, 0.0], # body3 CG 0.5 m  behind R2 joint
    ],
    "axis_u": [None, [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
    "axis_u1": [None, None, None],
    "axis_u2": [None, None, None],
    "rot_param": ["quat", None, None],  # F uses quaternion internally
}


def run():
    # ── [1] Construct ─────────────────────────────────────────────────────────
    mbd = MbdSystem3D.from_data(data)
    print(mbd)
    # → MbdSystem3D(NBodies=3, NJoints=3, total_dof=8, total_cfg_dof=9, total_user_dof=9)

    # ── [2] Inspect owned objects ─────────────────────────────────────────────
    # joint_system: topology, DOF bookkeeping, path caches
    print(f"NBodies={mbd.NBodies}, NJoints={mbd.NJoints}")
    mbd.summary_table()

    # coords: symbolic vectors (SymPy)
    print(f"q_int  symbols: {list(mbd.q_int)}")   # length total_cfg_dof
    print(f"qd_int symbols: {list(mbd.qd_int)}")  # length total_dof

    # vt: B / Bdot engine — use directly for block-level inspection
    #     (mbd.vt.build_B_blocks_symbolic, mbd.vt.print_B_blocks, etc.)

    # ── [3] Cached numeric params ─────────────────────────────────────────────
    # Build once; all JAX methods reuse this automatically.
    params = mbd.build_numeric_params()     # first call builds and caches
    _      = mbd.build_numeric_params()     # second call returns cached object
    print(f"params type: {type(params).__name__}")

    # ── [4] Initial conditions ────────────────────────────────────────────────
    # Internal config (q_int): F joint → [tx, ty, tz, e0, e1, e2, e3], then R angles
    tx, ty, tz = 0.0, 0.0, 0.5
    e0, e1, e2, e3 = np.cos(np.pi / 8), np.sin(np.pi / 8), 0.0, 0.0   # unit quat
    theta1, theta2 = 0.3, -0.2

    q_int = np.array([tx, ty, tz, e0, e1, e2, e3, theta1, theta2])
    # Speed vector (qd): always DOF-sized — vx,vy,vz,wx,wy,wz for F, then R speeds
    qd    = np.array([0.1, 0.0, 0.5, 0.0, 0.2, 0.0, 1.0, -0.5])

    # ── [5] Evaluate from internal coordinates ────────────────────────────────
    # Use these when q_int is already available (e.g. during integration).
    B    = mbd.evaluate_B_jax(q_int, params=params)
    Bdot = mbd.evaluate_Bdot_jax(q_int, qd, params=params)
    print(f"B    shape (from q_int): {B.shape}")    # (18, 8)
    print(f"Bdot shape (from q_int): {Bdot.shape}") # (18, 8)

    # ── [6] Evaluate from user-facing coordinates ─────────────────────────────
    # Use these when the user specifies initial conditions (Euler angles, etc.).
    # q_user has the same layout as q_int when rot_param='quat' for F.
    q_user = q_int.copy()   # identical here; would differ for rot_param='euler'

    B_user    = mbd.evaluate_B_from_user_q(q_user)
    Bdot_user = mbd.evaluate_Bdot_from_user_state(q_user, qd)
    print(f"B    shape (from q_user): {B_user.shape}")
    print(f"Bdot shape (from q_user): {Bdot_user.shape}")

    # ── [7] JIT-compiled evaluators for repeated calls ────────────────────────
    # Build once (triggers XLA compilation on first call), then reuse cheaply.
    B_fn    = mbd.build_B_evaluator_jax(params=params)
    Bdot_fn = mbd.build_Bdot_evaluator_jax(params=params)

    # Warm-up (compiles the XLA kernel)
    _ = B_fn(q_int)
    _ = Bdot_fn(q_int, qd)

    # Fast repeated evaluation
    B_jit    = B_fn(q_int)
    Bdot_jit = Bdot_fn(q_int, qd)
    print(f"B_jit    shape: {B_jit.shape}")
    print(f"Bdot_jit shape: {Bdot_jit.shape}")

    # User-coordinate JIT variants
    B_user_fn    = mbd.build_B_evaluator_from_user_q(params=params)
    Bdot_user_fn = mbd.build_Bdot_evaluator_from_user_state(params=params)
    print(f"B_user_jit    shape: {B_user_fn(q_user).shape}")
    print(f"Bdot_user_jit shape: {Bdot_user_fn(q_user, qd).shape}")

    # ── [8] Verify results agree ──────────────────────────────────────────────
    np.testing.assert_allclose(B, B_jit,       atol=1e-12)
    np.testing.assert_allclose(Bdot, Bdot_jit, atol=1e-12)
    print("\nAll shapes and values consistent.")


if __name__ == "__main__":
    run()
