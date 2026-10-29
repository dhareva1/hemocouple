"""Verification of the causal transient-Stokes velocity prior (stokes.py) against the analytic Womersley profile.

    python scripts/verify_stokes_prior.py [--out results/verify_stokes_prior.json]

Checks, on the reference interface grid:
1. basis: every psi_n = J0(lam_n rho) - J0(lam_n) (lam_n roots of J2, so psi_n(1) = 0) has zero flux, the modes
   are orthogonal in the area-weighted L2 product (eigenfunctions of the flux-constrained Stokes operator), and
   the least-squares coefficients b_n of stokes.py equal the Galerkin projection of the Poiseuille shape;
2. causal prior vs. Womersley (womersley.py, an independent frequency-domain implementation): area-weighted
   relative L2 error over one cycle, for the dataset waveform family at the corners of the parameter ranges,
   as a function of the window length, together with the predicted start-up error exp(-k_1 * window).
"""

from __future__ import annotations

import argparse
import itertools
import json
import sys
from pathlib import Path

import numpy as np
import torch
from scipy.special import jv

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from hemocouple.cross_section import CrossSectionGrid  # noqa: E402
from hemocouple.stokes import NU_BLOOD, stokes_basis, stokes_prior  # noqa: E402
from hemocouple.waveforms import pulsatile_flow  # noqa: E402
from hemocouple.womersley import WomersleyProfile  # noqa: E402

STEPS_PER_CYCLE = 100


def basis_checks(grid: CrossSectionGrid, n_modes: int = 32) -> dict:
    lam, _, b_hat = stokes_basis(grid.rho, n_modes)
    # Flux, no-slip and orthogonality on a fine radial grid (the 12-point interface grid cannot resolve high modes).
    r = (np.arange(20000) + 0.5) / 20000
    w = 2.0 * np.pi * r / r.size
    psi = jv(0, np.outer(lam, r)) - jv(0, lam)[:, None]
    flux = psi @ w / np.abs(psi).max(axis=1)
    gram = (psi * w) @ psi.T
    off = gram / np.sqrt(np.outer(np.diag(gram), np.diag(gram))) - np.eye(len(lam))
    # Galerkin coefficients of the zero-flux part of pi R^2 Pi, against the least-squares fit used by stokes.py.
    galerkin = (psi * w) @ (2.0 * (1.0 - r**2) - 1.0) / np.diag(gram)
    return {
        "max_abs_J2_at_roots": float(np.abs(jv(2, lam)).max()),
        "max_rel_flux": float(np.abs(flux).max()),
        "max_offdiag_gram": float(np.abs(off).max()),
        "b_hat_vs_galerkin_max_rel": float(np.abs(b_hat - galerkin).max() / np.abs(galerkin).max()),
        "slowest_decay_rate_R2_over_nu": float(lam[0] ** 2),
    }


def prior_vs_womersley(grid: CrossSectionGrid, radius: float, period: float, mean: float, amplitude: float,
                       peak: float, window_cycles: float, n_modes: int = 32) -> float:
    """Area-weighted relative L2 error of the causal prior against Womersley, pooled over the last cycle."""
    wave = pulsatile_flow(mean, amplitude, period, peak)
    dt = period / STEPS_PER_CYCLE
    window = int(round(window_cycles * STEPS_PER_CYCLE))
    t_eval = 5 * period + dt * np.arange(STEPS_PER_CYCLE)
    hist_t = t_eval[:, None] - dt * np.arange(window, -1, -1)[None, :]  # window + 1 samples, last one at t_eval
    lam, phi, b_hat = (torch.tensor(x, dtype=torch.float64) for x in stokes_basis(grid.rho, n_modes))
    n = len(t_eval)
    u_s = stokes_prior(torch.tensor(wave(hist_t), dtype=torch.float64), torch.full((n,), radius, dtype=torch.float64),
                       torch.full((n,), dt, dtype=torch.float64), torch.tensor(grid.rho, dtype=torch.float64),
                       lam, phi, b_hat, NU_BLOOD).numpy()
    u_w = WomersleyProfile(wave, radius, NU_BLOOD, grid.rho)(t_eval)
    return float(np.sqrt(((u_s - u_w) ** 2 @ grid.weights).sum() / ((u_w**2) @ grid.weights).sum()))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=ROOT / "results" / "verify_stokes_prior.json")
    args = ap.parse_args()
    grid = CrossSectionGrid.gauss_polar()
    out = {"basis": basis_checks(grid), "prior_vs_womersley": []}

    # Corners of the dataset ranges (Table 1): R, period, mean flow, amplitude, systolic peak.
    corners = list(itertools.product([2.5e-3, 4.0e-3], [0.70, 1.00], [4e-6, 8e-6], [6e-6, 14e-6], [0.12, 0.20]))
    for window_cycles in (0.99, 2.0, 4.0):
        errs = [prior_vs_womersley(grid, *c, window_cycles) for c in corners]
        startup = [float(np.exp(-NU_BLOOD * out["basis"]["slowest_decay_rate_R2_over_nu"] / c[0] ** 2 * window_cycles * c[1]))
                   for c in corners]
        out["prior_vs_womersley"].append({"window_cycles": window_cycles, "rel_l2_max": max(errs),
                                          "rel_l2_median": float(np.median(errs)),
                                          "startup_bound_max": max(startup)})
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
