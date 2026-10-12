"""Causal transient Stokes (time-domain Womersley) profile from a flow-rate history.

For fully developed flow in a rigid tube driven by a prescribed flow rate Q(t), write
u = Q(t) P(rho) + w, with P the unit-flux Poiseuille shape. The zero-flux part w expands in
phi_n(rho) = J0(lam_n rho) - J0(lam_n), where lam_n are the roots of J2 (the zero-flux condition),
and each coefficient obeys

    da_n/dt = -k_n a_n - b_n Q'(t),    k_n = nu lam_n^2 / R^2,

with b_n the coefficients of P in that basis. Taking Q piecewise linear over each step makes the
update exact. Starting from the quasi-steady state at the start of the window, the result uses only
past Q, so it is valid inside the coupling loop. For a periodic Q it converges to the Womersley profile.
"""

from __future__ import annotations

from typing import Tuple

import numpy as np
import torch
from scipy.special import jn_zeros, jv

NU_BLOOD = 3.5e-3 / 1060.0


def stokes_basis(rho: np.ndarray, n_modes: int = 32, n_fit: int = 4000) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(lam (N,), phi on rho (P, N), b_hat (N,)) with pi R^2 P(rho) = b_hat0 + sum_n b_hat_n phi_n(rho).

    The coefficients are fit on a fine radial grid (weight rho), not on the interface grid, so they
    stay well-conditioned when n_modes exceeds the number of radial grid points.
    """
    lam = jn_zeros(2, n_modes)
    r = (np.arange(n_fit) + 0.5) / n_fit
    a = np.column_stack([np.ones_like(r), jv(0, np.outer(r, lam)) - jv(0, lam)[None, :]]) * np.sqrt(r)[:, None]
    coef = np.linalg.lstsq(a, 2.0 * (1.0 - r**2) * np.sqrt(r), rcond=None)[0]
    phi = jv(0, np.outer(rho, lam)) - jv(0, lam)[None, :]
    return lam, phi, coef[1:]


def stokes_prior(
    hist_q: torch.Tensor,
    radius: torch.Tensor,
    dt: torch.Tensor,
    rho: torch.Tensor,
    lam: torch.Tensor,
    phi: torch.Tensor,
    b_hat: torch.Tensor,
    nu: float = NU_BLOOD,
) -> torch.Tensor:
    """Profile (B, P) at the last step of hist_q (B, H); radius and dt have shape (B,)."""
    area = torch.pi * radius**2
    k = nu * lam[None, :] ** 2 / radius[:, None] ** 2  # (B, N)
    decay = torch.exp(-k * dt[:, None])
    dq = hist_q[:, 1:] - hist_q[:, :-1]  # (B, H-1)
    age = torch.arange(dq.shape[1] - 1, -1, -1, dtype=hist_q.dtype, device=hist_q.device)  # steps since each increment
    # a_n = sum_j -b_n/(pi R^2) * dq_j/dt * (1 - decay)/k * decay^age_j
    gain = -(b_hat[None, :] / area[:, None]) * (1.0 - decay) / (k * dt[:, None])  # (B, N)
    memory = decay[:, :, None] ** age[None, None, :]  # (B, N, H-1)
    a = gain * torch.einsum("bnh,bh->bn", memory, dq)
    poiseuille = 2.0 * (1.0 - rho**2)  # pi R^2 times the unit-flux Poiseuille shape
    return hist_q[:, -1:] * poiseuille[None, :] / area[:, None] + a @ phi.T
