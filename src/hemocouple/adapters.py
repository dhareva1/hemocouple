"""Adapters for real solvers (proposal 7.2). Stubs until milestone M1 fixes the solver choice.

Each adapter must satisfy the protocols in ``coupling.py``: ``trial`` advances
one coupling step from the last accepted state without committing it, and
``accept`` commits it. For external codes this usually means checkpointing the
solver state (or restarting from the last accepted restart file) on each trial.
"""

from __future__ import annotations

from typing import Tuple

import numpy as np

from .cross_section import CrossSectionGrid


class OpenBFUpstream:
    """openBF (Julia) 1D network; the coupled vessel end takes a pressure BC.

    To implement: drive openBF through juliacall or a file-based step loop;
    write P_interface as the outlet BC, run to t_next, read Q and P at the
    outlet node; keep the last accepted state to support repeated trials.
    """

    def trial(self, dt: float, t_next: float, p_interface: float) -> Tuple[float, float]:
        raise NotImplementedError("openBF adapter pending solver pilot (milestone M1)")

    def accept(self) -> None:
        raise NotImplementedError


class External3DLocal:
    """Local 3D domain (svMultiPhysics or OpenFOAM pimpleFoam, chosen at M1).

    To implement: interpolate ``u_inlet`` from the reference ``CrossSectionGrid``
    (normalized rho, theta) to the inlet face nodes, write the Dirichlet BC,
    advance to t_next, return the area-averaged inlet pressure using the same
    pressure definition as the 1D side. For OpenFOAM, consider preCICE.
    """

    def __init__(self, grid: CrossSectionGrid, radius: float):
        self.grid, self.radius = grid, radius

    def flux(self, u: np.ndarray) -> float:
        return float(self.grid.integrate(u, self.radius))

    def trial(self, dt: float, t_next: float, u_inlet: np.ndarray) -> float:
        raise NotImplementedError("3D solver adapter pending solver pilot (milestone M1)")

    def accept(self) -> None:
        raise NotImplementedError


def extract_interface_profile(points: np.ndarray, velocity: np.ndarray, normal: np.ndarray, center: np.ndarray, radius: float, grid: CrossSectionGrid) -> np.ndarray:
    """Map a 3D solver's inlet-face samples to the reference grid (for building datasets).

    points (M, 3), velocity (M, 3) on the interface face. Returns the normal
    velocity on ``grid`` by linear interpolation in the face plane (nearest
    neighbour outside the sampled hull). The theta origin is arbitrary; for
    non-axisymmetric profiles fix it consistently across cases.
    """
    from scipy.interpolate import griddata

    n = np.asarray(normal, dtype=float)
    n /= np.linalg.norm(n)
    helper = np.array([1.0, 0.0, 0.0]) if abs(n[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
    e1 = np.cross(n, helper)
    e1 /= np.linalg.norm(e1)
    e2 = np.cross(n, e1)
    rel = np.asarray(points, dtype=float) - center
    src = np.stack([rel @ e1, rel @ e2], -1) / radius
    un = np.asarray(velocity, dtype=float) @ n
    dst = np.stack([grid.rho * np.cos(grid.theta), grid.rho * np.sin(grid.theta)], -1)
    out = griddata(src, un, dst, method="linear")
    missing = np.isnan(out)
    if missing.any():
        out[missing] = griddata(src, un, dst[missing], method="nearest")
    return out
