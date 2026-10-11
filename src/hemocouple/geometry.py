"""Parametric stenosed-vessel cases and design-of-experiments sampling (proposal 7.2)."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy.stats import qmc

from .waveforms import FourierWaveform, pulsatile_flow


@dataclass(frozen=True)
class StenosisGeometry:
    """Straight rigid vessel with a smooth axisymmetric (cosine) stenosis.

    ``severity`` is the area reduction at the throat, 1 - A_throat / A_normal.
    """

    radius: float
    severity: float
    length: float

    @property
    def area(self) -> float:
        return np.pi * self.radius**2

    @property
    def throat_radius(self) -> float:
        return self.radius * np.sqrt(1.0 - self.severity)

    @property
    def throat_area(self) -> float:
        return np.pi * self.throat_radius**2

    def radius_profile(self, z, center: float = 0.0) -> np.ndarray:
        z = np.asarray(z, dtype=float)
        depth = self.radius - self.throat_radius
        inside = np.abs(z - center) < self.length / 2
        bump = 0.5 * depth * (1.0 + np.cos(2.0 * np.pi * (z - center) / self.length))
        return np.where(inside, self.radius - bump, self.radius)


@dataclass(frozen=True)
class FlowCondition:
    mean_flow: float
    amplitude: float
    period: float
    systole_peak: float = 0.15

    def waveform(self, n_harmonics: int = 8) -> FourierWaveform:
        return pulsatile_flow(self.mean_flow, self.amplitude, self.period, self.systole_peak, n_harmonics)


def peak_throat_reynolds(geometry: StenosisGeometry, flow: FlowCondition, nu: float = 3.5e-3 / 1060.0) -> float:
    """Re_t = 2 Q_peak / (pi r_throat nu), the laminar criterion of proposal section 6."""
    t = np.linspace(0.0, flow.period, 1000, endpoint=False)
    q_peak = float(np.max(flow.waveform()(t)))
    return 2.0 * q_peak / (np.pi * geometry.throat_radius * nu)


@dataclass(frozen=True)
class CaseSpec:
    case_id: str
    geometry: StenosisGeometry
    flow: FlowCondition

    FEATURE_NAMES = ("radius", "severity", "length_ratio", "period")

    def features(self) -> np.ndarray:
        """Local geometry/flow descriptor g fed to the operator branch."""
        g = self.geometry
        return np.array([g.radius, g.severity, g.length / g.radius, self.flow.period])

    def to_dict(self) -> Dict:
        return {"case_id": self.case_id, "geometry": asdict(self.geometry), "flow": asdict(self.flow)}

    @classmethod
    def from_dict(cls, d: Dict) -> "CaseSpec":
        return cls(d["case_id"], StenosisGeometry(**d["geometry"]), FlowCondition(**d["flow"]))


# SI units. Carotid-like ranges; adjust once the solver pilot (milestone M1) fixes them.
DEFAULT_BOUNDS: Dict[str, Tuple[float, float]] = {
    "radius": (2.5e-3, 4.0e-3),
    "severity": (0.2, 0.75),
    "length_ratio": (2.0, 6.0),
    "mean_flow": (4.0e-6, 8.0e-6),
    "amplitude": (6.0e-6, 14.0e-6),
    "period": (0.7, 1.0),
    "systole_peak": (0.12, 0.20),
}


def sample_cases(
    n_stenosis: int,
    n_control: int = 0,
    bounds: Dict[str, Tuple[float, float]] = DEFAULT_BOUNDS,
    seed: int = 0,
    max_throat_re: Optional[float] = None,
) -> List[CaseSpec]:
    """Latin-hypercube stenosis cases plus uniform-vessel controls (severity 0).

    With ``max_throat_re``, stenosis cases above the laminar limit are rejected and
    the LHS is redrawn with more points until enough feasible cases exist. The
    first n_stenosis feasible points of the draw are kept, so the design is
    stratified over the feasible region rather than exactly Latin.
    """
    names = list(bounds)
    lo = np.array([bounds[k][0] for k in names])
    hi = np.array([bounds[k][1] for k in names])

    def build(p: dict, control: bool, i: int) -> CaseSpec:
        if control:
            p = {**p, "severity": 0.0}
        geometry = StenosisGeometry(p["radius"], p["severity"], p["length_ratio"] * p["radius"])
        flow = FlowCondition(p["mean_flow"], p["amplitude"], p["period"], p["systole_peak"])
        return CaseSpec(f"{'ctrl' if control else 'sten'}{i:04d}", geometry, flow)

    factor = 1
    while True:
        n_draw = factor * n_stenosis + n_control
        samples = qmc.scale(qmc.LatinHypercube(d=len(names), seed=seed).random(n_draw), lo, hi)
        stenosis = [build(dict(zip(names, row)), False, 0) for row in samples[: n_draw - n_control]]
        if max_throat_re is not None:
            stenosis = [c for c in stenosis if peak_throat_reynolds(c.geometry, c.flow) <= max_throat_re]
        if len(stenosis) >= n_stenosis or factor >= 64:
            break
        factor *= 2
    if len(stenosis) < n_stenosis:
        raise ValueError(f"only {len(stenosis)} feasible stenosis cases under Re_t <= {max_throat_re}")
    cases = [CaseSpec(f"sten{i:04d}", c.geometry, c.flow) for i, c in enumerate(stenosis[:n_stenosis])]
    for j, row in enumerate(samples[n_draw - n_control :]):
        cases.append(build(dict(zip(names, row)), True, n_stenosis + j))
    return cases
