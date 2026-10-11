"""In-house 1D blood-flow solver for a single elastic vessel (proposal 7.2).

Equations (area A, flow Q; flat profile, alpha = 1):

    dA/dt + dQ/dx = 0
    dQ/dt + d/dx(Q^2/A + beta A^{3/2} / (3 rho)) = -K_R Q / A
    p = p_ref + beta (sqrt(A) - sqrt(A0)),   c(A) = sqrt(beta / (2 rho)) A^{1/4}

with K_R = 22 pi nu (the usual value for a near-flat gamma = 9 profile) and
beta set from a reference wave speed c0 at A0. The interior uses two-step
(Richtmyer) Lax-Wendroff on a uniform node grid. The boundaries use Riemann
invariants W1,2 = u +- 4 (c - c0), traced back along the outgoing
characteristic (first order, so the scheme converges at first order overall;
see tests/test_oned.py):
    inlet  (x = 0): prescribed Q_in(t)
    outlet (x = L): prescribed pressure (the coupling interface) or absorbing (W2 = 0)

``OneDVessel`` implements the coupling ``UpstreamModel`` protocol: ``trial``
sub-cycles at the CFL step over one coupling step. During that step the
interface pressure is interpolated linearly from the last accepted value to
the trial value.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable, Optional, Tuple

import numpy as np


@dataclass(frozen=True)
class VesselParams:
    length: float = 0.15  # m
    radius: float = 3.0e-3  # m, at p_ref
    wave_speed: float = 6.0  # m/s, at p_ref
    p_ref: float = 1.0e4  # Pa
    density: float = 1060.0
    viscosity: float = 3.5e-3
    n_nodes: int = 151  # dx = 1 mm for the default length
    cfl: float = 0.5
    probe_distance: float = 0.01  # m upstream of the interface, where P_1D is sampled

    @property
    def area0(self) -> float:
        return math.pi * self.radius**2

    @property
    def beta(self) -> float:
        return 2.0 * self.density * self.wave_speed**2 / math.sqrt(self.area0)

    @property
    def friction(self) -> float:
        return 22.0 * math.pi * self.viscosity / self.density


class OneDVessel:
    def __init__(
        self,
        params: VesselParams,
        inflow: Callable[[float], float],
        q0: float = 0.0,
        p0: Optional[float] = None,
        outlet: str = "pressure",
    ):
        if outlet not in ("pressure", "absorbing"):
            raise ValueError("outlet must be 'pressure' or 'absorbing'")
        self.prm, self.inflow, self.outlet = params, inflow, outlet
        self.x = np.linspace(0.0, params.length, params.n_nodes)
        self.dx = self.x[1] - self.x[0]
        self.probe = int(round((params.length - params.probe_distance) / self.dx))
        p0 = params.p_ref if p0 is None else p0
        self.A = np.full(params.n_nodes, self.area_from_pressure(p0))
        self.Q = np.full(params.n_nodes, float(q0))
        self.t = 0.0
        self.p_bc = p0
        self._trial = None

    # --- constitutive relations -------------------------------------------------
    def pressure(self, A):
        p = self.prm
        return p.p_ref + p.beta * (np.sqrt(A) - math.sqrt(p.area0))

    def area_from_pressure(self, pressure: float) -> float:
        p = self.prm
        return (math.sqrt(p.area0) + (pressure - p.p_ref) / p.beta) ** 2

    def wave_speed(self, A):
        return np.sqrt(self.prm.beta / (2.0 * self.prm.density)) * A**0.25

    def _c0(self) -> float:
        return float(self.wave_speed(self.prm.area0))

    # --- scheme -----------------------------------------------------------------
    def _flux(self, A, Q):
        return Q, Q**2 / A + self.prm.beta * A**1.5 / (3.0 * self.prm.density)

    def _source(self, A, Q):
        return -self.prm.friction * Q / A

    def max_stable_dt(self, A=None, Q=None) -> float:
        A = self.A if A is None else A
        Q = self.Q if Q is None else Q
        return self.prm.cfl * self.dx / float(np.max(np.abs(Q / A) + self.wave_speed(A)))

    def _step(self, A, Q, dt, t_next, p_out):
        r = dt / self.dx
        fA, fQ = self._flux(A, Q)
        s = self._source(A, Q)
        Ah = 0.5 * (A[1:] + A[:-1]) - 0.5 * r * (fA[1:] - fA[:-1])
        Qh = 0.5 * (Q[1:] + Q[:-1]) - 0.5 * r * (fQ[1:] - fQ[:-1]) + 0.25 * dt * (s[1:] + s[:-1])
        fAh, fQh = self._flux(Ah, Qh)
        sh = self._source(Ah, Qh)
        An, Qn = A.copy(), Q.copy()
        An[1:-1] = A[1:-1] - r * (fAh[1:] - fAh[:-1])
        Qn[1:-1] = Q[1:-1] - r * (fQh[1:] - fQh[:-1]) + 0.5 * dt * (sh[1:] + sh[:-1])
        An[0], Qn[0] = self._inlet(A, Q, dt, float(self.inflow(t_next)))
        An[-1], Qn[-1] = self._outlet(A, Q, dt, p_out)
        return An, Qn

    def _traced(self, W, lam, dt, i0, i1):
        """Value of invariant W at the foot of a characteristic reaching node i0 from the side of i1."""
        xi = min(abs(lam) * dt / self.dx, 1.0)
        return (1.0 - xi) * W[i0] + xi * W[i1]

    def _inlet(self, A, Q, dt, q_in):
        c0 = self._c0()
        u, c = Q[:2] / A[:2], self.wave_speed(A[:2])
        w2 = self._traced(u - 4.0 * (c - c0), u[0] - c[0], dt, 0, 1)
        a = A[0]
        for _ in range(30):  # Newton on q_in / a - 4 (c(a) - c0) - w2 = 0
            ca = float(self.wave_speed(a))
            f = q_in / a - 4.0 * (ca - c0) - w2
            da = f / (-q_in / a**2 - ca / a)
            a -= da
            if abs(da) < 1e-13 * a:
                break
        return a, q_in

    def _outlet(self, A, Q, dt, p_out):
        c0 = self._c0()
        u, c = Q[-2:] / A[-2:], self.wave_speed(A[-2:])
        w1 = self._traced(u + 4.0 * (c - c0), u[-1] + c[-1], dt, 1, 0)
        if self.outlet == "absorbing":
            c_b = w1 / 8.0 + c0
            a = (c_b / math.sqrt(self.prm.beta / (2.0 * self.prm.density))) ** 4
            return a, 0.5 * w1 * a
        a = self.area_from_pressure(p_out)
        return a, (w1 - 4.0 * (float(self.wave_speed(a)) - c0)) * a

    def advance(self, A, Q, t0, dt, p_start, p_end):
        n_sub = max(1, math.ceil(dt / self.max_stable_dt(A, Q)))
        h = dt / n_sub
        for k in range(1, n_sub + 1):
            p_out = p_start + (p_end - p_start) * k / n_sub
            A, Q = self._step(A, Q, h, t0 + k * h, p_out)
        return A, Q

    # --- coupling protocol ------------------------------------------------------
    def trial(self, dt: float, t_next: float, p_interface: float) -> Tuple[float, float]:
        A, Q = self.advance(self.A, self.Q, self.t, dt, self.p_bc, p_interface)
        self._trial = (A, Q, t_next, p_interface)
        return float(Q[-1]), float(self.pressure(A[self.probe]))

    def accept(self) -> None:
        self.A, self.Q, self.t, self.p_bc = self._trial
