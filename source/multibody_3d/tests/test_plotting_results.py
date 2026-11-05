"""Smoke tests for the results plots in multibody_3d.plotting (non-interactive backend, synthetic data)."""
import matplotlib

matplotlib.use("Agg")

from types import SimpleNamespace  # noqa: E402

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pytest  # noqa: E402

from multibody_3d import plotting  # noqa: E402

N_STEPS  = 5
N_CFG    = 3
N_DOF    = 3


def _energy(n_body, e_total=None):
    ts = np.linspace(0.0, 1.0, N_STEPS)
    return SimpleNamespace(
        ts=ts,
        KE=np.ones(N_STEPS),
        PE=np.ones(N_STEPS),
        E_total=np.ones(N_STEPS) if e_total is None else e_total,
        KE_body=np.ones((N_STEPS, n_body)),
        PE_body=np.ones((N_STEPS, n_body)),
    )


def _kin(n_body):
    return SimpleNamespace(
        ts=np.linspace(0.0, 1.0, N_STEPS),
        r_cg=np.zeros((N_STEPS, n_body, 3)),
        euler_cg=np.zeros((N_STEPS, n_body, 3)),
        v_cg=np.zeros((N_STEPS, n_body, 3)),
        omega_cg=np.zeros((N_STEPS, n_body, 3)),
    )


@pytest.fixture(autouse=True)
def _close_figures():
    yield
    plt.close("all")


def test_plot_coordinates_has_two_axes():
    sol = SimpleNamespace(
        ts=np.linspace(0.0, 1.0, N_STEPS),
        ys=np.zeros((N_STEPS, N_CFG + N_DOF)),
    )
    fig = plotting.plot_coordinates(sol, N_CFG, ["a", "b", "c"], ["ad", "bd", "cd"])
    assert len(fig.axes) == 2


def test_plot_coordinates_without_labels():
    sol = SimpleNamespace(
        ts=np.linspace(0.0, 1.0, N_STEPS),
        ys=np.zeros((N_STEPS, N_CFG + N_DOF)),
    )
    assert len(plotting.plot_coordinates(sol, N_CFG).axes) == 2


def test_plot_system_energy_has_two_axes():
    assert len(plotting.plot_system_energy(_energy(2)).axes) == 2


@pytest.mark.parametrize("n_body", [1, 3])
def test_per_body_plots_have_two_columns_per_body(n_body):
    assert len(plotting.plot_body_energy(_energy(n_body)).axes) == 2 * n_body
    assert len(plotting.plot_body_velocities(_kin(n_body)).axes) == 2 * n_body
    assert len(plotting.plot_body_trajectories(_kin(n_body)).axes) == 2 * n_body


def test_print_system_energy_reports_values(capsys):
    plotting.print_system_energy(_energy(1, e_total=np.array([2.0, 2.0, 3.0, 2.0, 2.0])))
    out = capsys.readouterr().out
    assert "System energy" in out
    assert "initial" in out and "final" in out
    assert "50" in out or "5.000e+01" in out


def test_print_system_energy_zero_initial_energy(capsys):
    plotting.print_system_energy(_energy(1, e_total=np.zeros(N_STEPS)))
    assert "n/a" in capsys.readouterr().out
