import math

import numpy as np
import pytest
from scipy.integrate import trapezoid

from hemocouple.oned import OneDVessel, VesselParams


def _run(vessel, t_end, dt, p_out):
    t = vessel.t
    while t < t_end - 1e-12:
        t += dt
        vessel.trial(dt, t, p_out)
        vessel.accept()


def _steady_errors(n_nodes):
    prm = VesselParams(length=0.1, n_nodes=n_nodes)
    q = 5e-6
    v = OneDVessel(prm, lambda t: q * min(t / 0.05, 1.0), q0=0.0, p0=prm.p_ref)
    _run(v, 1.5, 1e-3, prm.p_ref)
    expected = prm.density * prm.friction * q * prm.length / prm.area0**2
    dp = float(v.pressure(v.A[0]) - v.pressure(v.A[-1]))
    return np.abs(v.Q / q - 1).max(), abs(dp / expected - 1)


def test_steady_flow_converges_to_friction_law():
    q41, dp41 = _steady_errors(41)
    q81, dp81 = _steady_errors(81)
    assert q81 < 1e-3 and dp81 < 0.015
    assert q81 < 0.6 * q41 and dp81 < 0.6 * dp41  # at least first-order convergence


def _pulse_run(outlet):
    prm = VesselParams(length=0.3, n_nodes=301, cfl=0.5)
    amp, t_peak, width = 1e-6, 0.01, 0.002
    v = OneDVessel(prm, lambda t: amp * math.exp(-(((t - t_peak) / width) ** 2)), q0=0.0, outlet=outlet)
    mid = len(v.x) // 2
    dt = 2e-4
    times, q_mid, p_out = [], [], []
    t = 0.0
    while t < 0.12:
        t += dt
        v.trial(dt, t, prm.p_ref)
        v.accept()
        times.append(t)
        q_mid.append(v.Q[mid])
        p_out.append(float(v.pressure(v.A[-1])))
    return prm, v, np.array(times), np.array(q_mid), np.array(p_out), mid


def test_pulse_travels_at_wave_speed():
    prm, v, t, q_mid, _, mid = _pulse_run("absorbing")
    arrival = t[np.argmax(q_mid)] - 0.01
    assert arrival == pytest.approx(v.x[mid] / prm.wave_speed, rel=0.03)


def test_absorbing_outlet_reflects_little():
    prm, v, t, q_mid, _, mid = _pulse_run("absorbing")
    first = q_mid[t < 0.04].max()
    reflected = np.abs(q_mid[t > 0.06]).max()
    assert reflected < 0.01 * first


def test_pressure_outlet_reflects_strongly():
    # A fixed-pressure end reflects flow with coefficient ~ +1; friction attenuates it ~20% over the path.
    prm, v, t, q_mid, _, mid = _pulse_run("pressure")
    assert np.abs(q_mid[t > 0.06]).max() > 0.7 * q_mid[t < 0.04].max()


def test_global_mass_balance():
    prm = VesselParams(length=0.15, n_nodes=151)
    wave = lambda t: 5e-6 + 3e-6 * math.sin(2 * math.pi * t / 0.8)
    v = OneDVessel(prm, wave, q0=5e-6, outlet="absorbing")
    vol0 = trapezoid(v.A, v.x)
    dt, t, net = 1e-3, 0.0, 0.0
    for _ in range(400):
        q_in0, q_out0 = v.Q[0], v.Q[-1]
        t += dt
        v.trial(dt, t, prm.p_ref)
        v.accept()
        net += 0.5 * dt * ((q_in0 + v.Q[0]) - (q_out0 + v.Q[-1]))
    vol1 = trapezoid(v.A, v.x)
    assert (vol1 - vol0) == pytest.approx(net, rel=0.02, abs=1e-3 * abs(net) + 1e-12)


def test_trial_is_repeatable_and_accept_commits():
    prm = VesselParams(n_nodes=31)
    v = OneDVessel(prm, lambda t: 5e-6, q0=5e-6)
    a = v.trial(0.01, 0.01, prm.p_ref + 100.0)
    b = v.trial(0.01, 0.01, prm.p_ref + 100.0)
    assert a == b
    c = v.trial(0.01, 0.01, prm.p_ref + 500.0)
    assert c[0] != a[0]
    v.accept()
    assert v.t == 0.01 and v.p_bc == prm.p_ref + 500.0


def test_oned_vessel_in_coupling_loop(specs, grid):
    from hemocouple.coupling import run_coupled
    from hemocouple.mock import MockCase

    spec = specs[0]
    case = MockCase(spec, grid)
    _, local, reduced = case.components()
    prm = VesselParams(radius=spec.geometry.radius, p_ref=case.p0, n_nodes=41)
    inflow = spec.flow.waveform()
    vessel = OneDVessel(prm, lambda t: float(inflow(t)), q0=case.q0, p0=case.p0)
    steps = 50
    res = run_coupled(
        vessel, case.oracle(), local.flux, mode="A", dt=spec.flow.period / steps, n_steps=3 * steps,
        window=1, q_init=case.q0, p1_init=case.p0, p_init=case.p0, local=local, reduced=reduced, tol=1e-8,
    )
    assert res.converged.all()
    assert res.iterations.mean() < 10
    assert np.isfinite(res.p_interface).all()
