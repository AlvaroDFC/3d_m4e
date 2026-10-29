"""plotting_results.py

Matplotlib time-series plots and a text energy summary for 3D simulation results.

Every function takes an already-computed result (the diffrax solution,
``EnergyResult`` or ``KinematicsResult``) and returns the figure.  None of them
call ``plt.show()``, so the caller decides whether to display or save.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

_XYZ          = ("x", "y", "z")
_EULER_LABELS = ("roll", "pitch", "yaw")


def _label(labels, i, default):
    return labels[i] if labels is not None and i < len(labels) else f"{default}{i}"


def plot_coordinates(sol, n_cfg, q_labels=None, qd_labels=None):
    """Plot generalized coordinates and speeds from a diffrax solution.

    ``n_cfg`` is the length of the internal configuration block of ``sol.ys``.
    """
    ts  = np.asarray(sol.ts)
    qs  = np.asarray(sol.ys[:, :n_cfg])
    qds = np.asarray(sol.ys[:, n_cfg:])

    fig, (ax_q, ax_qd) = plt.subplots(2, 1, figsize=(10, 7), sharex=True)

    for i in range(qs.shape[1]):
        ax_q.plot(ts, qs[:, i], label=_label(q_labels, i, "q"))
    ax_q.set_ylabel("Generalized coordinates")
    ax_q.legend()
    ax_q.grid(True)

    for i in range(qds.shape[1]):
        ax_qd.plot(ts, qds[:, i], label=_label(qd_labels, i, "qd"))
    ax_qd.set_ylabel("Generalized velocities")
    ax_qd.set_xlabel("Time [s]")
    ax_qd.legend()
    ax_qd.grid(True)

    fig.suptitle("Coordinates and velocities over time")
    fig.tight_layout()
    return fig


def plot_system_energy(energy):
    """Plot system kinetic, potential and total energy against time."""
    ts = np.asarray(energy.ts)

    fig, (ax_e, ax_tot) = plt.subplots(2, 1, figsize=(10, 7), sharex=True)

    ax_e.plot(ts, energy.KE, label="KE")
    ax_e.plot(ts, energy.PE, label="PE")
    ax_e.set_ylabel("Energy [J]")
    ax_e.set_title("System kinetic and potential energy")
    ax_e.legend()
    ax_e.grid(True)

    ax_tot.plot(ts, energy.E_total, color="k")
    ax_tot.set_ylabel("Total E [J]")
    ax_tot.set_xlabel("Time [s]")
    ax_tot.set_title("System energy")
    ax_tot.grid(True)

    fig.tight_layout(h_pad=2.5)
    return fig


def plot_body_energy(energy):
    """Plot per-body kinetic (left) and gravitational potential (right) energy."""
    ts      = np.asarray(energy.ts)
    n_body  = np.asarray(energy.KE_body).shape[1]

    fig, axes = plt.subplots(n_body, 2, figsize=(12, 3 * n_body), sharex=True, squeeze=False)

    for b in range(n_body):
        ax_ke, ax_pe = axes[b]
        ax_ke.plot(ts, energy.KE_body[:, b])
        ax_pe.plot(ts, energy.PE_body[:, b])
        ax_ke.set_ylabel("KE [J]")
        ax_pe.set_ylabel("PE [J]")
        ax_ke.set_title(f"Body {b + 1} – kinetic energy")
        ax_pe.set_title(f"Body {b + 1} – gravitational potential energy")
        ax_ke.grid(True)
        ax_pe.grid(True)

    for ax in axes[-1]:
        ax.set_xlabel("Time [s]")

    fig.suptitle("Per-body kinetic and gravitational potential energy")
    fig.tight_layout(rect=[0, 0, 1, 0.97], h_pad=2.5)
    return fig


def plot_body_velocities(kin):
    """Plot per-body CG linear (left) and angular (right) velocity in the world frame."""
    ts     = np.asarray(kin.ts)
    n_body = np.asarray(kin.v_cg).shape[1]

    fig, axes = plt.subplots(n_body, 2, figsize=(12, 4 * n_body), sharex=True, squeeze=False)

    for b in range(n_body):
        ax_lin, ax_ang = axes[b]
        for k in range(3):
            ax_lin.plot(ts, kin.v_cg[:, b, k], label=f"v_{_XYZ[k]}")
            ax_ang.plot(ts, kin.omega_cg[:, b, k], label=f"ω_{_XYZ[k]}")
        ax_lin.set_title(f"Body {b + 1} – linear velocity (world frame)", fontsize=7)
        ax_ang.set_title(f"Body {b + 1} – angular velocity (world frame)", fontsize=7)
        ax_lin.set_ylabel("m/s")
        ax_ang.set_ylabel("rad/s")
        ax_lin.legend()
        ax_ang.legend()
        ax_lin.grid(True)
        ax_ang.grid(True)

    for ax in axes[-1]:
        ax.set_xlabel("Time [s]")

    fig.tight_layout(pad=1.5, h_pad=2.5)
    return fig


def plot_body_trajectories(kin):
    """Plot per-body CG position (left) and orientation (right), one row per body."""
    ts     = np.asarray(kin.ts)
    n_body = np.asarray(kin.r_cg).shape[1]

    fig, axes = plt.subplots(n_body, 2, figsize=(12, 4 * n_body), sharex=True, squeeze=False)

    for b in range(n_body):
        ax_r, ax_o = axes[b]
        for k in range(3):
            ax_r.plot(ts, kin.r_cg[:, b, k], label=_XYZ[k])
            ax_o.plot(ts, kin.euler_cg[:, b, k], label=_EULER_LABELS[k])
        ax_r.set_title(f"Body {b + 1} – CG position (world frame)", fontsize=7)
        ax_o.set_title(f"Body {b + 1} – CG orientation (intrinsic Z-Y-X)", fontsize=7)
        ax_r.set_ylabel("m")
        ax_o.set_ylabel("rad")
        ax_r.legend()
        ax_o.legend()
        ax_r.grid(True)
        ax_o.grid(True)

    for ax in axes[-1]:
        ax.set_xlabel("Time [s]")

    fig.tight_layout(pad=1.5, h_pad=2.5)
    return fig


def print_system_energy(energy):
    """Print initial, final and largest change of the total system energy."""
    e_total = np.asarray(energy.E_total)
    e0      = e_total[0]
    change  = np.max(np.abs(e_total - e0))
    percent = f"{change / abs(e0) * 100:.3e} %" if e0 != 0 else "n/a"

    print("\nSystem energy")
    print(f"  initial                    : {e0:.6f} J")
    print(f"  final                      : {e_total[-1]:.6f} J")
    print(f"  largest change from initial: {change:.3e} J ({percent})")
