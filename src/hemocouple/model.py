"""POD-DeepONet with an exact flux-projection layer (proposal 7.3, 7.4).

The branch net maps (causal (Q, P_1D) window, geometry g) to K basis
coefficients and a pressure correction dP. The trunk is either the fixed POD
basis ("pod") or a learned MLP over (rho, cos theta, sin theta) ("trunk", the
ablation). In both cases the flux is linear in the coefficients, so

    c = c_hat + (Q - F^T c_hat) / (F^T F) * F

enforces the 1D flow exactly (up to quadrature). All normalization statistics
live in buffers, so a checkpoint is self-contained and takes physical units.

Optional ΔP prior (``dp_prior="young_tsai"``): the network predicts a residual
on top of the quasi-steady expansion loss s * (rho/2) K_t (A0/A1 - 1)^2 |U| U,
where s is a learnable scale initialized to 1. This gives the right quadratic
scaling in severity and flow, so the network does not have to extrapolate it
from a few cases.
"""

from __future__ import annotations

from typing import Dict, Optional

import numpy as np
import torch
from torch import nn


def mlp(n_in: int, n_out: int, hidden: int, depth: int) -> nn.Sequential:
    layers, width = [], n_in
    for _ in range(depth):
        layers += [nn.Linear(width, hidden), nn.GELU()]
        width = hidden
    layers.append(nn.Linear(width, n_out))
    return nn.Sequential(*layers)


def flux_projection(c_hat: torch.Tensor, flux_vec: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
    """Minimum-norm correction of c_hat onto {c : F^T c = q}. Shapes (B,K), (B,K), (B,)."""
    mismatch = q - (flux_vec * c_hat).sum(-1)
    return c_hat + (mismatch / (flux_vec * flux_vec).sum(-1))[:, None] * flux_vec


class OperatorNet(nn.Module):
    def __init__(
        self,
        window: int,
        n_features: int,
        n_points: int,
        n_modes: int,
        basis: str = "pod",
        hidden: int = 128,
        depth: int = 3,
        project: bool = True,
        dp_prior: str = "none",
        density: float = 1060.0,
        expansion_coefficient: float = 1.52,
    ):
        super().__init__()
        if dp_prior not in ("none", "young_tsai"):
            raise ValueError(f"dp_prior must be 'none' or 'young_tsai', got {dp_prior!r}")
        if basis not in ("pod", "trunk"):
            raise ValueError(f"basis must be 'pod' or 'trunk', got {basis!r}")
        self.init_kwargs = dict(
            window=window, n_features=n_features, n_points=n_points, n_modes=n_modes,
            basis=basis, hidden=hidden, depth=depth, project=project,
            dp_prior=dp_prior, density=density, expansion_coefficient=expansion_coefficient,
        )
        self.dp_prior, self.density, self.k_t = dp_prior, density, expansion_coefficient
        self.prior_scale = nn.Parameter(torch.ones(())) if dp_prior != "none" else None
        self.basis, self.project = basis, project
        # Separate branches for the profile coefficients and for dP, so the two outputs do not trade capacity.
        self.branch = mlp(2 * window + n_features, n_modes, hidden, depth)
        self.dp_branch = mlp(2 * window + n_features, 1, hidden, depth)
        self.trunk = mlp(3, n_modes, hidden, depth) if basis == "trunk" else None

        for name, shape in {
            "grid_rho": (n_points,), "grid_theta": (n_points,), "grid_weights": (n_points,),
            "pod_modes": (n_points, n_modes), "coef_scale": (n_modes,),
            "hist_mean": (2,), "hist_std": (2,), "g_mean": (n_features,), "g_std": (n_features,),
            "dp_mean": (), "dp_std": (),
        }.items():
            self.register_buffer(name, torch.ones(shape))

    def set_statistics(self, **arrays: np.ndarray) -> None:
        for name, value in arrays.items():
            buf = getattr(self, name)
            buf.copy_(torch.as_tensor(np.asarray(value), dtype=buf.dtype).reshape(buf.shape))

    def young_tsai(self, g: torch.Tensor, q: torch.Tensor, radius: torch.Tensor) -> torch.Tensor:
        """Quasi-steady expansion loss; g holds raw CaseSpec.features() (index 1 = severity)."""
        severity = g[:, 1].clamp(0.0, 0.99)
        velocity = q / (torch.pi * radius**2)
        return 0.5 * self.density * self.k_t * (1.0 / (1.0 - severity) - 1.0) ** 2 * velocity.abs() * velocity

    def basis_matrix(self) -> torch.Tensor:
        """Phi on the reference grid, (N, K)."""
        if self.trunk is None:
            return self.pod_modes
        coords = torch.stack([self.grid_rho, torch.cos(self.grid_theta), torch.sin(self.grid_theta)], -1)
        return self.trunk(coords)

    def forward(self, hist: torch.Tensor, g: torch.Tensor, q: torch.Tensor, radius: torch.Tensor) -> Dict[str, torch.Tensor]:
        x = torch.cat([((hist - self.hist_mean) / self.hist_std).flatten(1), (g - self.g_mean) / self.g_std], -1)
        c_hat = self.branch(x) * self.coef_scale
        dp = self.dp_branch(x)[:, 0] * self.dp_std + self.dp_mean
        if self.prior_scale is not None:
            dp = dp + self.prior_scale * self.young_tsai(g, q, radius)

        phi = self.basis_matrix()
        flux_vec = radius[:, None] ** 2 * (self.grid_weights @ phi)[None, :]
        c = flux_projection(c_hat, flux_vec, q) if self.project else c_hat
        return {"u": c @ phi.T, "dp": dp, "c": c, "c_hat": c_hat}

    def checkpoint(self, extra: Optional[dict] = None) -> dict:
        return {"kwargs": self.init_kwargs, "state_dict": self.state_dict(), "extra": extra or {}}

    @classmethod
    def from_checkpoint(cls, ckpt: dict) -> "OperatorNet":
        model = cls(**ckpt["kwargs"]).to(ckpt["state_dict"]["hist_mean"].dtype)
        model.load_state_dict(ckpt["state_dict"])
        model.eval()
        return model
