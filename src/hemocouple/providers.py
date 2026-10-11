"""Interface providers used inside the coupling loop (proposal 7.5)."""

from __future__ import annotations

from typing import Tuple

import numpy as np
import torch

from .cross_section import CrossSectionGrid
from .model import OperatorNet


class PoiseuilleProfile:
    """Conventional baseline: parabolic inlet profile scaled to the 1D flow, no pressure correction."""

    def __init__(self, grid: CrossSectionGrid, radius: float):
        self.shape = 2.0 * (1.0 - grid.rho**2) / (np.pi * radius**2)

    def predict(self, history: np.ndarray, q: float, t: float) -> Tuple[np.ndarray, float]:
        return q * self.shape, 0.0


class WomersleyFluxProfile:
    """Womersley profile of a known periodic waveform, flux-corrected to the current Q with a Poiseuille shape.

    Used at the far inlet of extended-domain reference runs, matching how the training labels were made.
    """

    def __init__(self, wave, grid: CrossSectionGrid, radius: float, nu: float):
        from .womersley import WomersleyProfile

        self.wave = wave
        self.profile = WomersleyProfile(wave, radius, nu, grid.rho)
        self.shape = 2.0 * (1.0 - grid.rho**2) / (np.pi * radius**2)

    def predict(self, history: np.ndarray, q: float, t: float) -> Tuple[np.ndarray, float]:
        return self.profile(t) + (q - float(self.wave(t))) * self.shape, 0.0


class OperatorProvider:
    """Online inference wrapper around a trained OperatorNet for one case."""

    def __init__(self, model: OperatorNet, features: np.ndarray, radius: float):
        self.model = model.eval()
        dtype = next(model.buffers()).dtype
        self._g = torch.as_tensor(features, dtype=dtype)[None]
        self._r = torch.tensor([radius], dtype=dtype)
        self._dtype = dtype

    @torch.no_grad()
    def predict(self, history: np.ndarray, q: float, t: float) -> Tuple[np.ndarray, float]:
        hist = torch.as_tensor(history, dtype=self._dtype)[None]
        out = self.model(hist, self._g, torch.tensor([q], dtype=self._dtype), self._r)
        return out["u"][0].numpy(), float(out["dp"][0])
