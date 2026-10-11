import numpy as np
import pytest

from conftest import N_CYCLES, STEPS_PER_CYCLE
from hemocouple.dataset import CaseRecord, build_samples, split_cases
from hemocouple.metrics import relative_rmse
from hemocouple.mock import MockCase
from hemocouple.providers import PoiseuilleProfile


def test_reference_runs_converge_and_are_periodic(records):
    s = STEPS_PER_CYCLE
    for rec in records:
        assert rec.profiles.shape == (s * N_CYCLES, len(rec.profiles[0]))
        assert relative_rmse(rec.p_interface[-s:], rec.p_interface[-2 * s : -s]) < 1e-2


def test_windows_are_causal_and_within_case(records):
    window = 5
    samples = build_samples(records[:2], window, start=0)
    n0 = len(records[0].t) - window + 1
    assert samples["hist"].shape == (n0 + len(records[1].t) - window + 1, window, 2)
    # sample i of case 0 corresponds to step n = i + window - 1
    i = 7
    n = i + window - 1
    np.testing.assert_array_equal(samples["hist"][i, -1], [records[0].q[n], records[0].p1d[n]])
    np.testing.assert_array_equal(samples["hist"][i, 0], [records[0].q[n - window + 1], records[0].p1d[n - window + 1]])
    assert samples["case_index"][n0] == 1
    np.testing.assert_array_equal(samples["hist"][n0, 0], [records[1].q[0], records[1].p1d[0]])


def test_case_split_has_no_leakage(records):
    splits = split_cases(records, val_frac=0.2, test_frac=0.2, ood_severity=0.6, seed=0)
    ids = [cid for v in splits.values() for cid in v]
    assert len(ids) == len(set(ids)) == len(records)
    by_id = {r.case_id: r for r in records}
    assert all(by_id[c].severity > 0.6 for c in splits["test_ood"])
    assert all(by_id[c].severity <= 0.6 for k in ("train", "val", "test") for c in splits[k])


def test_record_save_load_roundtrip(records, tmp_path):
    rec = records[0]
    loaded = CaseRecord.load(rec.save(tmp_path))
    assert loaded.case_id == rec.case_id and loaded.meta == rec.meta
    np.testing.assert_array_equal(loaded.profiles, rec.profiles)
    np.testing.assert_array_equal(loaded.dp, rec.dp)


def test_oracle_reference_conserves_mass(specs, grid):
    case = MockCase(specs[0], grid)
    res = case.run(case.oracle(), "A", STEPS_PER_CYCLE, 1, window=1, tol=1e-10)
    assert res.converged.all()
    # Exact up to quadrature of the Bessel profile on the coarse test grid.
    np.testing.assert_allclose(res.q_local, res.q, rtol=1e-6)


class _ReplayCorrection:
    """Provider returning the reference dP label at each step (perfect Mode B surrogate)."""

    def __init__(self, rec, grid):
        self.rec, self.base = rec, PoiseuilleProfile(grid, rec.radius)

    def predict(self, history, q, t):
        n = int(round(t / self.rec.dt)) - 1
        return self.base.predict(history, q, t)[0], float(self.rec.dp[n])


def test_mode_b_with_exact_correction_reproduces_reference(specs, grid, records):
    case = MockCase(specs[1], grid)
    rec = records[1]
    res = case.run(_ReplayCorrection(rec, grid), "B", STEPS_PER_CYCLE, N_CYCLES, window=1, tol=1e-10)
    assert relative_rmse(res.p_interface, rec.p_interface) < 1e-6
    assert relative_rmse(res.q, rec.q) < 1e-6


def test_mode_validation(specs, grid):
    case = MockCase(specs[0], grid)
    upstream, local, reduced = case.components()
    from hemocouple.coupling import run_coupled

    with pytest.raises(ValueError):
        run_coupled(upstream, case.oracle(), local.flux, mode="C", dt=0.01, n_steps=1, window=1, q_init=0, p1_init=0, p_init=0)
    with pytest.raises(ValueError):
        run_coupled(upstream, case.oracle(), local.flux, mode="A", dt=0.01, n_steps=1, window=1, q_init=0, p1_init=0, p_init=0)


def test_reference_label_helpers(grid):
    from hemocouple.mock import LumpedParams
    from hemocouple.reference import ReferenceConfig, flux_correct, linear_interface_pressure

    radius = 3e-3
    q = np.array([5e-6, 6e-6, 7e-6])
    shape = 2.0 * (1 - grid.rho**2) / (np.pi * radius**2)
    profiles = 0.98 * q[:, None] * shape[None, :]
    corrected, corr = flux_correct(profiles, q, grid, radius)
    np.testing.assert_allclose(grid.integrate(corrected, radius), q, rtol=1e-12)
    np.testing.assert_allclose(corr, 0.02 * q / q.mean(), rtol=1e-9)

    cfg, lp = ReferenceConfig(), LumpedParams()
    steady = np.full(400, 6e-6)
    p = linear_interface_pressure(steady, 0.008, 0.1, radius, cfg, lp, p_distal0=6e-6 * lp.distal_resistance)
    expected = 6e-6 * (lp.distal_resistance + 8 * cfg.viscosity * 0.1 / (np.pi * radius**4))
    assert p[-1] == pytest.approx(expected, rel=1e-9)


def test_reduced_segment_matches_label_model():
    from hemocouple.mock import LumpedParams
    from hemocouple.reference import ReducedSegment, ReferenceConfig, linear_interface_pressure

    cfg, lp = ReferenceConfig(), LumpedParams()
    q = 6e-6 + 3e-6 * np.sin(np.linspace(0, 4 * np.pi, 200))
    p_ref = linear_interface_pressure(q, 0.008, 0.08, 3e-3, cfg, lp, p_distal0=1.2e4)
    seg = ReducedSegment(0.08, 3e-3, cfg, lp, q0=q[0], p_distal0=1.2e4)
    out = []
    for k, qk in enumerate(q):
        out.append(seg.trial(0.008, 0.0, qk))
        seg.accept()
    np.testing.assert_allclose(out[1:], p_ref[1:], rtol=1e-12)
