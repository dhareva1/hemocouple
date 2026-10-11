"""Proper Orthogonal Decomposition of interface profiles (proposal 7.3).

Modes are orthonormal in the area-weighted L2 inner product of the unit disk,
so coefficients are c = Phi^T W u and the flux of a profile is linear in c.
Fit on training cases only.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass
class PODBasis:
    modes: np.ndarray  # (N, K)
    weights: np.ndarray  # (N,) unit-disk area weights
    singular_values: np.ndarray  # all singular values, for energy reporting

    @classmethod
    def fit(cls, snapshots: np.ndarray, weights: np.ndarray, energy: float = 0.99999, max_modes: Optional[int] = None) -> "PODBasis":
        """No mean subtraction: u = Phi c exactly, which keeps the flux projection linear."""
        sw = np.sqrt(weights)
        _, s, vt = np.linalg.svd(snapshots * sw, full_matrices=False)
        cumulative = np.cumsum(s**2) / np.sum(s**2)
        k = int(np.searchsorted(cumulative, energy) + 1)
        if max_modes is not None:
            k = min(k, max_modes)
        k = min(k, len(s))
        return cls(modes=vt[:k].T / sw[:, None], weights=weights, singular_values=s)

    @property
    def n_modes(self) -> int:
        return self.modes.shape[1]

    @property
    def captured_energy(self) -> float:
        s2 = self.singular_values**2
        return float(s2[: self.n_modes].sum() / s2.sum())

    def project(self, u: np.ndarray) -> np.ndarray:
        return (u * self.weights) @ self.modes

    def reconstruct(self, c: np.ndarray) -> np.ndarray:
        return c @ self.modes.T

    def flux_vector(self, radius: float) -> np.ndarray:
        """F_k = integral of mode k over a section of the given radius."""
        return radius**2 * (self.weights @ self.modes)

    def projection_error(self, u: np.ndarray) -> float:
        """Relative weighted L2 error of the best approximation in the basis (lower bound, 7.7)."""
        residual = u - self.reconstruct(self.project(u))
        return float(np.sqrt(np.sum(residual**2 * self.weights) / np.sum(u**2 * self.weights)))
