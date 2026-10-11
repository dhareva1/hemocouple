"""Evaluation metrics (proposal 7.7)."""

from __future__ import annotations

from typing import Dict

import numpy as np

from .coupling import CouplingResult


def rmse(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.sqrt(np.mean((np.asarray(a) - np.asarray(b)) ** 2)))


def relative_rmse(pred: np.ndarray, ref: np.ndarray) -> float:
    """RMSE normalized by the peak-to-peak range of the reference."""
    return rmse(pred, ref) / float(np.ptp(ref) or 1.0)


def mass_error(q_1d: np.ndarray, q_3d: np.ndarray, q_ref: float) -> np.ndarray:
    """epsilon_Q(t) = |Q_1D - Q_3D| / Q_ref."""
    return np.abs(np.asarray(q_1d) - np.asarray(q_3d)) / q_ref


def wave_separation(p: np.ndarray, q: np.ndarray, zc: float):
    """Linear forward/backward split about the cycle mean: P+- = (dP +- Zc dQ) / 2."""
    dp, dq = p - p.mean(), q - q.mean()
    return 0.5 * (dp + zc * dq), 0.5 * (dp - zc * dq)


def reflection_coefficient(p: np.ndarray, q: np.ndarray, zc: float) -> float:
    """RMS ratio of backward to forward pressure waves at one location."""
    forward, backward = wave_separation(p, q, zc)
    return float(np.sqrt(np.mean(backward**2) / np.mean(forward**2)))


def characteristic_impedance(density: float, wave_speed: float, area: float) -> float:
    return density * wave_speed / area


def break_even_runs(offline_cost: float, online_baseline: float, online_operator: float) -> float:
    """N* = C_off / (C_on_baseline - C_on_NO); inf when the operator is not cheaper online."""
    saving = online_baseline - online_operator
    return float(offline_cost / saving) if saving > 0 else float("inf")


def compare_runs(run: CouplingResult, ref: CouplingResult, last: int, zc: float) -> Dict[str, float]:
    """Metrics over the last ``last`` steps (e.g. the final cardiac cycle)."""
    s = slice(-last, None)
    q_ref = float(np.mean(np.abs(ref.q[s])))
    eps_q = mass_error(run.q[s], run.q_local[s], q_ref)
    return {
        "p_interface_rel_rmse": relative_rmse(run.p_interface[s], ref.p_interface[s]),
        "q_rel_rmse": relative_rmse(run.q[s], ref.q[s]),
        "p1d_rel_rmse": relative_rmse(run.p1d[s], ref.p1d[s]),
        "mass_err_max": float(eps_q.max()),
        "mass_err_mean": float(eps_q.mean()),
        "reflection_abs_diff": abs(
            reflection_coefficient(run.p1d[s], run.q[s], zc) - reflection_coefficient(ref.p1d[s], ref.q[s], zc)
        ),
        "iterations_mean": float(run.iterations[s].mean()),
        "nonconverged_steps": int((~run.converged).sum()),
        "provider_time_s": run.provider_time,
        "wall_time_s": run.wall_time,
    }


def grid_convergence_index(f_fine: float, f_medium: float, f_coarse: float, ratio: float, safety: float = 1.25) -> Dict[str, float]:
    """Roache's GCI for three meshes refined by a constant ratio (fine-mesh estimate)."""
    e21, e32 = f_medium - f_fine, f_coarse - f_medium
    # Monotone convergence needs successive differences of equal sign that shrink (0 < e21/e32 < 1).
    if e21 == 0 or e32 == 0 or not 0 < e21 / e32 < 1:
        kind = "oscillatory" if e21 * e32 < 0 else "divergent_or_stalled"
        return {"order": float("nan"), "gci_fine": float("nan"), "extrapolated": float("nan"), "monotone": False, "kind": kind}
    order = float(np.log(e32 / e21) / np.log(ratio))
    extrapolated = f_fine + (f_fine - f_medium) / (ratio**order - 1.0)
    gci = safety * abs(e21 / f_fine) / (ratio**order - 1.0)
    return {"order": order, "gci_fine": float(gci), "extrapolated": float(extrapolated), "monotone": True, "kind": "monotone"}


def phase_resample(t: np.ndarray, x: np.ndarray, period: float, n_phase: int = 256) -> np.ndarray:
    """Split a (possibly non-uniformly sampled) signal into complete cycles on a common phase grid, shape (cycles, n_phase)."""
    n_cycles = int(np.floor(t[-1] / period + 1e-9))
    phase = np.arange(n_phase) / n_phase
    return np.stack([np.interp((c + phase) * period, t, x) for c in range(n_cycles)])


def cycle_periodicity(t: np.ndarray, x: np.ndarray, period: float, n_phase: int = 256) -> Dict[str, float]:
    """Laminar/transitional indicators from the last two complete cycles.

    cycle_diff: RMS difference between the last two cycles / range of the last one.
    hf_fraction: share of the last cycle's fluctuation energy above harmonic 20.
    """
    cycles = phase_resample(t, x, period, n_phase)
    last, prev = cycles[-1], cycles[-2]
    spec = np.abs(np.fft.rfft(last - last.mean())) ** 2
    return {
        "cycles": int(len(cycles)),
        "cycle_diff": float(np.sqrt(np.mean((last - prev) ** 2)) / (np.ptp(last) or 1.0)),
        "hf_fraction": float(spec[21:].sum() / (spec[1:].sum() or 1.0)),
    }
