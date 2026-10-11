"""Training objective L = l_data L_data + l_mass L_mass + l_pressure L_pressure + l_PDE L_PDE (proposal 7.4)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, Optional, Tuple

import torch


@dataclass(frozen=True)
class LossWeights:
    data: float = 1.0
    mass: float = 0.0  # zero by construction when the flux projection is on
    pressure: float = 0.0
    pde: float = 0.0


@dataclass(frozen=True)
class LossScales:
    velocity: float
    flow: float
    pressure: float
    flow_eps: float  # guard for the kinetic term near Q = 0


def section_integral(f: torch.Tensor, weights: torch.Tensor, radius: torch.Tensor) -> torch.Tensor:
    return (f * weights).sum(-1) * radius**2


def kinetic_pressure(u, q, weights, radius, density: float, eps: float) -> torch.Tensor:
    """Torch twin of mock.kinetic_pressure: (rho/2)(int u^3 dA / Q - U^2), smoothly guarded."""
    area = torch.pi * radius**2
    excess = section_integral(u**3, weights, radius) - q**3 / area**2
    return 0.5 * density * excess * q / (q**2 + eps**2)


def objective(
    out: Dict[str, torch.Tensor],
    batch: Dict[str, torch.Tensor],
    weights: torch.Tensor,
    lam: LossWeights,
    scales: LossScales,
    density: float,
    pde_residual: Optional[Callable[[Dict, Dict], torch.Tensor]] = None,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    u, u_true, radius, q = out["u"], batch["u"], batch["radius"], batch["q"]

    velocity_err = ((u - u_true) ** 2 * weights).sum(-1) / weights.sum() / scales.velocity**2
    dp_err = ((out["dp"] - batch["dp"]) / scales.pressure) ** 2
    l_data = (velocity_err + dp_err).mean()

    l_mass = (((section_integral(u, weights, radius) - q) / scales.flow) ** 2).mean()

    # Total-pressure consistency couples the predicted profile and the predicted correction.
    kin_pred = kinetic_pressure(u, q, weights, radius, density, scales.flow_eps)
    kin_true = kinetic_pressure(u_true, q, weights, radius, density, scales.flow_eps)
    l_pressure = ((((out["dp"] + kin_pred) - (batch["dp"] + kin_true)) / scales.pressure) ** 2).mean()

    # Interface data carry no interior field, so the Navier-Stokes residual needs
    # a solver-specific hook (e.g. near-interface 3D slabs); off by default.
    l_pde = pde_residual(out, batch) if (pde_residual is not None and lam.pde > 0) else torch.zeros((), dtype=u.dtype)

    total = lam.data * l_data + lam.mass * l_mass + lam.pressure * l_pressure + lam.pde * l_pde
    parts = {"data": l_data.item(), "velocity": velocity_err.mean().item(), "dp": dp_err.mean().item(),
             "mass": l_mass.item(), "pressure": l_pressure.item(), "pde": float(l_pde)}
    return total, parts
