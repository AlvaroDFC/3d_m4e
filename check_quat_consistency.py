"""
Diagnostic: check if B*qd gives velocities consistent with d/dt(r_abs).
Tests if the quaternion kinematic equation gives consistent position derivatives.
"""
import example2 as ex
import numpy as np
from multibody_3d import MbdSystem3D
from multibody_3d.multibody_core.velocity_transformation_3d import build_cache_jax
from multibody_3d.multibody_core.runtime_context_3d import build_runtime_context
import jax.numpy as jnp

mbd = MbdSystem3D.from_example(ex)

# Initial state with some nonzero angular velocity (to test velocity consistency)
# S-joint: quat = [cos30, 0, 0, sin30] (30-deg rotation around z)
# qd_s = [0.5, 0.3, 0.1] (relative angular velocity, in which frame?)
q_int_np = np.array([0.9659, 0., 0., 0.2588, -np.pi/12])
qd_np    = np.array([0.5, 0.3, 0.1, 1.0])  # some nonzero speeds

# Build context to get pos_cache_kwargs
q_user_np0 = mbd.map_q_int_to_q_user(q_int_np)
ctx = build_runtime_context(mbd, np.concatenate([q_user_np0, qd_np]))
pos_kw = ctx.pos_cache_kwargs
q_int_jax = jnp.array(q_int_np)
A_abs, r_abs, rJ, U, _ = build_cache_jax(q_int_jax, **pos_kw)

print("r_abs[1] (body1 CG, world frame):", np.array(r_abs[1]).ravel())
print("r_abs[2] (body2 CG, world frame):", np.array(r_abs[2]).ravel())
print("A_abs[1] (body1 rotation, world frame):")
print(np.array(A_abs[1]).round(4))

# Compute body velocities from B * qd
q_user_np = mbd.map_q_int_to_q_user(q_int_np)
mainNumVars = np.concatenate([q_user_np, qd_np])
B = np.array(mbd.evaluate_B(mainNumVars))
v_body = B @ qd_np  # (12,) = [v_cg1 | omega1 | v_cg2 | omega2]
print("\nFrom B*qd:")
print("  v_cg1 =", v_body[0:3], "(world frame)")
print("  omega1 =", v_body[3:6], "(world frame)")
print("  v_cg2 =", v_body[6:9], "(world frame)")
print("  omega2 =", v_body[9:12], "(world frame)")

# Numerically differentiate r_abs w.r.t. q_int evolution
# Using the CURRENT (code's) q_dot formula
from multibody_3d.multibody_core.integrator_3d import _omega_matrix_jax, _compute_q_int_dot_jax
per_joint = mbd.coords.per_joint  # get per-joint data

q_int_jax = jnp.array(q_int_np)
qd_jax    = jnp.array(qd_np)
q_dot_code = np.array(_compute_q_int_dot_jax(q_int_jax, qd_jax, per_joint))
print("\nq_dot (code's formula):", q_dot_code)

# Numerically differentiate position with code's q_dot
dt = 1e-7
q_int_plus = q_int_np + dt * q_dot_code
A_abs_p, r_abs_p, _, _, _ = build_cache_jax(jnp.array(q_int_plus), **pos_kw)

r1_dot_num = (np.array(r_abs_p[1]).ravel() - np.array(r_abs[1]).ravel()) / dt
r2_dot_num = (np.array(r_abs_p[2]).ravel() - np.array(r_abs[2]).ravel()) / dt
print("\nNumerical d/dt(r_abs[1]) with code's q_dot:", r1_dot_num)
print("From B*qd (v_cg1):                        ", v_body[0:3])
print("Match v_cg1?", np.allclose(r1_dot_num, v_body[0:3], atol=1e-4))

print("\nNumerical d/dt(r_abs[2]) with code's q_dot:", r2_dot_num)
print("From B*qd (v_cg2):                        ", v_body[6:9])
print("Match v_cg2?", np.allclose(r2_dot_num, v_body[6:9], atol=1e-4))

# Now test with CORRECT (left-multiplication) q_dot formula
def omega_matrix_left(omega):
    """Correct Omega_L for left-multiplication: q_dot = 0.5 * Omega_L @ q."""
    ox, oy, oz = omega[0], omega[1], omega[2]
    return jnp.array([[ 0, -ox, -oy, -oz],
                       [ox,   0, -oz,  oy],
                       [oy,  oz,   0, -ox],
                       [oz, -oy,  ox,   0]], dtype=jnp.float64)

def compute_q_int_dot_left(q_int, qd, per_joint):
    """q_dot using correct LEFT multiplication for S-joints."""
    segments = []
    for pj in per_joint:
        code     = pj["type"].value
        int_sl   = pj["int_slice"]
        speed_sl = pj["speed_slice"]
        if code in ("R", "P", "U", "C"):
            segments.append(qd[speed_sl])
        elif code == "S":
            e     = q_int[int_sl]
            omega = qd[speed_sl]
            segments.append(0.5 * omega_matrix_left(omega) @ e)
        elif code == "F":
            cs    = int_sl.start
            ss    = speed_sl.start
            e     = q_int[cs + 3: cs + 7]
            omega = qd[ss + 3: ss + 6]
            segments.append(jnp.concatenate([
                qd[ss: ss + 3],
                0.5 * omega_matrix_left(omega) @ e,
            ]))
    return jnp.concatenate(segments)

q_dot_left = np.array(compute_q_int_dot_left(q_int_jax, qd_jax, per_joint))
print("\nq_dot (LEFT/correct formula):", q_dot_left)

q_int_plus_left = q_int_np + dt * q_dot_left
A_abs_pl, r_abs_pl, _, _, _ = build_cache_jax(jnp.array(q_int_plus_left), **pos_kw)

r1_dot_left = (np.array(r_abs_pl[1]).ravel() - np.array(r_abs[1]).ravel()) / dt
r2_dot_left = (np.array(r_abs_pl[2]).ravel() - np.array(r_abs[2]).ravel()) / dt
print("\nNumerical d/dt(r_abs[1]) with LEFT q_dot:", r1_dot_left)
print("From B*qd (v_cg1):                       ", v_body[0:3])
print("Match v_cg1?", np.allclose(r1_dot_left, v_body[0:3], atol=1e-4))

print("\nNumerical d/dt(r_abs[2]) with LEFT q_dot:", r2_dot_left)
print("From B*qd (v_cg2):                       ", v_body[6:9])
print("Match v_cg2?", np.allclose(r2_dot_left, v_body[6:9], atol=1e-4))
