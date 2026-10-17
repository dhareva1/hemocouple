from types import SimpleNamespace

import numpy as np
import pytest

from hemocouple.cross_section import CrossSectionGrid
from hemocouple.geometry import CaseSpec, FlowCondition, StenosisGeometry
from hemocouple.stenosis import YoungTsai, YoungTsaiProvider, backward_dqdt, young_tsai_terms


def spec(severity: float, case_id: str = "s") -> CaseSpec:
    return CaseSpec(case_id, StenosisGeometry(3e-3, severity, 9e-3), FlowCondition(6e-6, 10e-6, 0.8))


def test_terms_signs_and_control():
    q = np.array([-5e-6, 0.0, 5e-6])
    terms = young_tsai_terms(spec(0.6), q, np.zeros(3))
    assert np.all(np.sign(terms[:2]) == np.sign(q))  # viscous and expansion losses oppose the flow
    np.testing.assert_array_equal(terms[2], 0.0)
    np.testing.assert_array_equal(young_tsai_terms(spec(0.0), q, np.ones(3)), 0.0)


def test_expansion_term_matches_formula():
    q = np.array([5e-6])
    a0 = np.pi * 3e-3**2
    expected = 1.52 * 1060.0 / (2 * a0**2) * (1 / 0.4 - 1) ** 2 * q**2
    assert young_tsai_terms(spec(0.6), q, np.zeros(1))[1, 0] == pytest.approx(expected[0], rel=1e-12)


def test_calibration_recovers_scales():
    t = np.arange(1, 301) * 0.008
    true = YoungTsai((0.3, 0.8, 5.0))
    recs = []
    for i, sev in enumerate((0.3, 0.5, 0.7)):
        sp = spec(sev, f"s{i}")
        q = 6e-6 + 5e-6 * np.sin(2 * np.pi * t / 0.8 + i)
        recs.append(SimpleNamespace(spec=sp.to_dict(), t=t, q=q, dp=true.dp(sp, q, backward_dqdt(q, 0.008))))
    np.testing.assert_allclose(YoungTsai.calibrated(recs, warmup_steps=10).scales, true.scales, rtol=1e-8)


def test_provider_uses_backward_difference():
    grid = CrossSectionGrid.gauss_polar(6, 4)
    sp, dt = spec(0.6), 0.008
    prov = YoungTsaiProvider(YoungTsai(), sp, grid, dt)
    u, dp = prov.predict(np.array([[4e-6, 0.0], [5e-6, 0.0]]), 5e-6, 0.0)
    assert grid.integrate(u, sp.geometry.radius) == pytest.approx(5e-6, rel=1e-12)
    assert dp == pytest.approx(YoungTsai().dp(sp, np.array([5e-6]), np.array([1e-6 / dt]))[0], rel=1e-12)
