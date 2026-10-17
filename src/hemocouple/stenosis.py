"""Lumped Young-Tsai stenosis model: the conventional reduced-order baseline for the pressure correction.

Seeley & Young (1976) / Young & Tsai (1973), as used in 1D networks (Stergiopulos et al. 1992):

    ΔP_sten = Kv μ Q / (A0 D0) + Kt ρ/(2 A0²) (A0/A1 - 1)² |Q| Q + Ku ρ Ls / A0 dQ/dt
    Kv = 32 (La / D0) (A0/A1)²,  La = 0.83 Ls + 1.64 D1,  Kt = 1.52,  Ku = 1.2

The labels dP = P_Γ - P_lin are relative to a uniform Poiseuille segment that already carries the viscous
resistance 8 μ Ls / (π R⁴) and inertance ρ Ls / A0 over the stenosis length, so the excess over the uniform
tube is returned. Uniform controls (severity 0) get dP = 0, i.e. the reduced model itself.

``calibrated`` fits one scale per term by least squares on training labels, which gives the baseline the
same data the operator sees.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Tuple

import numpy as np

from .cross_section import CrossSectionGrid
from .geometry import CaseSpec

K_T, K_U = 1.52, 1.2


def young_tsai_terms(spec: CaseSpec, q: np.ndarray, dqdt: np.ndarray, density: float = 1060.0,
                     viscosity: float = 3.5e-3) -> np.ndarray:
    """Viscous, expansion and inertial excess pressure drops (literature coefficients), shape (3, n) in Pa."""
    g = spec.geometry
    q, dqdt = np.asarray(q, dtype=float), np.asarray(dqdt, dtype=float)
    if g.severity <= 0.0:
        return np.zeros((3, q.size))
    a0, d0, d1, ls = g.area, 2.0 * g.radius, 2.0 * g.throat_radius, g.length
    ratio = 1.0 / (1.0 - g.severity)  # A0 / A1
    kv = 32.0 * (0.83 * ls + 1.64 * d1) / d0 * ratio**2
    viscous = (kv * viscosity / (a0 * d0) - 8.0 * viscosity * ls / (np.pi * g.radius**4)) * q
    expansion = K_T * density / (2.0 * a0**2) * (ratio - 1.0) ** 2 * np.abs(q) * q
    inertial = (K_U - 1.0) * density * ls / a0 * dqdt
    return np.stack([viscous, expansion, inertial])


def backward_dqdt(q: np.ndarray, dt: float) -> np.ndarray:
    """(q_k - q_{k-1}) / dt with a zero first entry, the difference the coupling loop uses."""
    q = np.asarray(q, dtype=float)
    return np.concatenate([[0.0], np.diff(q)]) / dt


@dataclass(frozen=True)
class YoungTsai:
    """dP = Σ scale_i · term_i; scales (1, 1, 1) is the uncalibrated literature model."""

    scales: Tuple[float, float, float] = (1.0, 1.0, 1.0)

    def dp(self, spec: CaseSpec, q: np.ndarray, dqdt: np.ndarray) -> np.ndarray:
        return np.asarray(self.scales) @ young_tsai_terms(spec, q, dqdt)

    @classmethod
    def calibrated(cls, records: Iterable, warmup_steps: int = 0) -> "YoungTsai":
        """Least-squares scales on the labels of ``records`` (dataset CaseRecords), samples after warm-up."""
        rows, rhs = [], []
        for rec in records:
            spec = CaseSpec.from_dict(rec.spec)
            dt = float(rec.t[1] - rec.t[0])
            terms = young_tsai_terms(spec, rec.q, backward_dqdt(rec.q, dt))
            s = slice(max(warmup_steps, 1), None)
            rows.append(terms[:, s].T)
            rhs.append(rec.dp[s])
        scales, *_ = np.linalg.lstsq(np.concatenate(rows), np.concatenate(rhs), rcond=None)
        return cls(tuple(float(x) for x in scales))

    @classmethod
    def calibrated_on(cls, data: Path, split: str = "train", warmup_cycles: int = 3) -> "YoungTsai":
        """``calibrated`` on a split of a dataset directory, with the operator's training warm-up."""
        from .dataset import load_records

        meta = json.loads((data / "dataset.json").read_text())
        ids = json.loads((data / "splits.json").read_text())[split]
        return cls.calibrated(load_records(data, ids), warmup_cycles * meta["steps_per_cycle"])


class YoungTsaiProvider:
    """Mode B provider: Poiseuille profile and the lumped Young-Tsai dP for one case."""

    history_length = 2

    def __init__(self, model: YoungTsai, spec: CaseSpec, grid: CrossSectionGrid, dt: float):
        r = spec.geometry.radius
        self.model, self.spec, self.dt = model, spec, dt
        self.shape = 2.0 * (1.0 - grid.rho**2) / (np.pi * r**2)

    def predict(self, history: np.ndarray, q: float, t: float) -> Tuple[np.ndarray, float]:
        dqdt = (q - float(history[-2, 0])) / self.dt
        return q * self.shape, float(self.model.dp(self.spec, np.array([q]), np.array([dqdt]))[0])
