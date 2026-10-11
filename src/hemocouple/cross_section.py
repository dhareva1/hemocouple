"""Reference quadrature grid on the circular interface cross-section (proposal 7.3).

Profiles from any solver mesh are interpolated onto this grid in normalized
coordinates (rho = r / R), so POD modes and fluxes are comparable across cases
with different radii. Integrals scale with R**2.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class CrossSectionGrid:
    rho: np.ndarray
    theta: np.ndarray
    weights: np.ndarray  # area weights on the unit disk; sum == pi

    @classmethod
    def gauss_polar(cls, n_r: int = 12, n_theta: int = 8) -> "CrossSectionGrid":
        """Gauss-Legendre in rho (exact for polynomial profiles up to degree 2*n_r - 2), uniform in theta."""
        x, wx = np.polynomial.legendre.leggauss(n_r)
        r = 0.5 * (x + 1.0)
        wr = 0.5 * wx * r
        theta = 2.0 * np.pi * (np.arange(n_theta) + 0.5) / n_theta
        return cls(
            rho=np.repeat(r, n_theta),
            theta=np.tile(theta, n_r),
            weights=np.repeat(wr, n_theta) * (2.0 * np.pi / n_theta),
        )

    @property
    def n_points(self) -> int:
        return len(self.rho)

    def integrate(self, f: np.ndarray, radius: float) -> np.ndarray:
        """Integral over the physical section of f sampled on the grid (last axis)."""
        return (np.asarray(f) @ self.weights) * radius**2
