import numpy as np
import pytest

from hemocouple.cross_section import CrossSectionGrid
from hemocouple.geometry import DEFAULT_BOUNDS, StenosisGeometry, sample_cases
from hemocouple.metrics import break_even_runs, grid_convergence_index, reflection_coefficient
from hemocouple.mock import blunting_shape
from hemocouple.waveforms import pulsatile_flow
from hemocouple.womersley import WomersleyProfile


def test_waveform_mean_amplitude_and_derivative():
    w = pulsatile_flow(mean=6e-6, amplitude=10e-6, period=0.8)
    t = np.linspace(0, 0.8, 2000, endpoint=False)
    q = w(t)
    assert np.mean(q) == pytest.approx(6e-6, rel=1e-10)
    assert np.ptp(q) == pytest.approx(10e-6, rel=1e-3)
    h = 1e-6
    fd = (w(t[:50] + h) - w(t[:50] - h)) / (2 * h)
    np.testing.assert_allclose(w.derivative(t[:50]), fd, rtol=1e-5, atol=1e-12)


def test_grid_integrates_poiseuille_exactly():
    grid = CrossSectionGrid.gauss_polar(6, 4)
    assert grid.weights.sum() == pytest.approx(np.pi)
    radius, q = 3e-3, 5e-6
    u = 2 * q / (np.pi * radius**2) * (1 - grid.rho**2)
    assert grid.integrate(u, radius) == pytest.approx(q, rel=1e-12)


def test_womersley_flux_matches_waveform():
    grid = CrossSectionGrid.gauss_polar(24, 2)
    w = pulsatile_flow(6e-6, 10e-6, 0.8)
    prof = WomersleyProfile(w, radius=3e-3, nu=3.5e-3 / 1060, rho=grid.rho)
    t = np.linspace(0, 0.8, 17)
    np.testing.assert_allclose(grid.integrate(prof(t), 3e-3), w(t), rtol=1e-8, atol=1e-14)


def test_blunting_shape_has_zero_flux_and_wall_value():
    grid = CrossSectionGrid.gauss_polar(8, 2)
    assert grid.integrate(blunting_shape(grid.rho), 1.0) == pytest.approx(0.0, abs=1e-12)
    assert blunting_shape(np.array([1.0]))[0] == 0.0


def test_stenosis_profile():
    g = StenosisGeometry(radius=3e-3, severity=0.75, length=12e-3)
    assert g.radius_profile(0.0) == pytest.approx(g.throat_radius)
    assert g.throat_area / g.area == pytest.approx(0.25)
    assert g.radius_profile(10e-3) == pytest.approx(3e-3)


def test_sample_cases_respects_bounds_and_controls():
    cases = sample_cases(12, 3, seed=0)
    assert len(cases) == 15 and len({c.case_id for c in cases}) == 15
    sev = np.array([c.geometry.severity for c in cases])
    assert (sev[-3:] == 0).all()
    lo, hi = DEFAULT_BOUNDS["severity"]
    assert ((sev[:12] >= lo) & (sev[:12] <= hi)).all()


def test_reflection_of_pure_forward_wave_is_zero():
    zc = 2.0
    q = np.sin(np.linspace(0, 2 * np.pi, 200))
    assert reflection_coefficient(zc * q, q, zc) == pytest.approx(0.0, abs=1e-12)
    # p = 0.5 Zc q -> P+ = 0.75 Zc q, P- = -0.25 Zc q
    assert reflection_coefficient(0.5 * zc * q, q, zc) == pytest.approx(1 / 3)


def test_break_even():
    assert break_even_runs(100.0, 10.0, 0.0) == pytest.approx(10.0)
    assert break_even_runs(100.0, 1.0, 2.0) == float("inf")


def test_gci_recovers_order():
    h = np.array([1.0, 1.5, 2.25])  # fine, medium, coarse
    f = 10.0 + 0.3 * h**2
    g = grid_convergence_index(*f, ratio=1.5)
    assert g["order"] == pytest.approx(2.0)
    assert g["extrapolated"] == pytest.approx(10.0)


def test_gci_flags_non_monotone():
    assert grid_convergence_index(1.0, 1.05, 1.15, 1.5)["monotone"] is True
    assert grid_convergence_index(1.0, 1.1, 1.2, 1.5)["kind"] == "divergent_or_stalled"
    assert grid_convergence_index(1.0, 1.1, 1.05, 1.5)["kind"] == "oscillatory"


def test_cycle_periodicity_separates_periodic_and_noisy():
    from hemocouple.metrics import cycle_periodicity

    t = np.linspace(0, 4.0, 4001)
    clean = np.sin(2 * np.pi * t) + 0.3 * np.sin(6 * np.pi * t)
    noisy = clean + 0.2 * np.random.default_rng(0).normal(size=t.size)
    a, b = cycle_periodicity(t, clean, 1.0), cycle_periodicity(t, noisy, 1.0)
    assert a["cycle_diff"] < 1e-6 and a["hf_fraction"] < 1e-6
    assert b["cycle_diff"] > 0.02 and b["hf_fraction"] > 0.02


def test_sampling_respects_reynolds_cap():
    from hemocouple.geometry import peak_throat_reynolds

    cases = sample_cases(20, 4, seed=3, max_throat_re=1700.0)
    sten = [c for c in cases if c.geometry.severity > 0]
    assert len(sten) == 20 and len(cases) == 24
    assert all(peak_throat_reynolds(c.geometry, c.flow) <= 1700.0 for c in sten)
    assert len({c.case_id for c in cases}) == 24
