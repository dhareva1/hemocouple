"""Womersley solution for fully developed pulsatile flow in a rigid tube."""

from __future__ import annotations

import numpy as np
from scipy.special import jv

from .waveforms import FourierWaveform


def womersley_shapes(radius: float, nu: float, omega: float, n_harmonics: int, rho: np.ndarray) -> np.ndarray:
    """Complex profiles f_n(rho) with unit flux: u = Re sum Q_n f_n(rho) exp(i n omega t).

    Shape (n_harmonics, len(rho)).
    """
    n = np.arange(1, n_harmonics + 1)
    alpha = radius * np.sqrt(n * omega / nu)
    lam = alpha * np.exp(3j * np.pi / 4)  # i**1.5 * alpha
    j0 = jv(0, lam)
    denom = np.pi * radius**2 * (1.0 - 2.0 * jv(1, lam) / (lam * j0))
    return (1.0 - jv(0, np.outer(lam, rho)) / j0[:, None]) / denom[:, None]


class WomersleyProfile:
    """Axial velocity u(rho, t) for a flow waveform; precomputes the radial shapes."""

    def __init__(self, wave: FourierWaveform, radius: float, nu: float, rho: np.ndarray):
        self.wave = wave
        self.radius = radius
        self._poiseuille = 2.0 / (np.pi * radius**2) * (1.0 - rho**2)
        self._shapes = womersley_shapes(radius, nu, wave.omega, wave.n_harmonics, rho)

    def __call__(self, t) -> np.ndarray:
        """Velocity with shape t.shape + (len(rho),)."""
        weighted = self.wave.phases(t) * self.wave.coeffs
        unsteady = np.real(np.einsum("...h,hn->...n", weighted, self._shapes))
        return self.wave.mean * self._poiseuille + unsteady
