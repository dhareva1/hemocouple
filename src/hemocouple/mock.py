"""Cheap analytic stand-ins for the 1D and 3D solvers.

FOR SOFTWARE VERIFICATION ONLY. These lumped models let the full pipeline
(data generation -> POD -> training -> online coupling -> metrics) run in
seconds and be unit-tested. They are not a substitute for the converged 3D
reference simulations described in the proposal, and no physical conclusion
should be drawn from results obtained with them.

Model:
    upstream "1D":  compliance C1 fed by Q_in(t), then R1/L1 to the interface
    local "3D":     lubrication resistance/inertance of the stenosis profile,
                    Young-Tsai expansion loss, a profile-dependent kinetic-energy
                    term, then a distal RC Windkessel
    reduced:        Poiseuille resistance/inertance of a uniform segment plus the
                    same Windkessel (the "1D-transferred" estimate)
    oracle profile: Womersley profile of Q_in, flux-corrected to the current Q,
                    plus a severity-dependent zero-flux blunting term
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np
from scipy.integrate import trapezoid

from .coupling import CouplingResult, run_coupled
from .cross_section import CrossSectionGrid
from .geometry import CaseSpec
from .womersley import WomersleyProfile


@dataclass(frozen=True)
class Fluid:
    density: float = 1060.0  # kg/m^3
    viscosity: float = 3.5e-3  # Pa s

    @property
    def nu(self) -> float:
        return self.viscosity / self.density


@dataclass(frozen=True)
class LumpedParams:
    upstream_compliance: float = 1.0e-10
    upstream_resistance: float = 1.0e8
    upstream_inertance: float = 3.8e6
    distal_resistance: float = 2.0e9
    distal_compliance: float = 1.0e-10  # with C1: slowest RC mode ~0.4 s
    expansion_coefficient: float = 1.52  # Young-Tsai K_t


def kinetic_pressure(u: np.ndarray, q: float, grid: CrossSectionGrid, radius: float, density: float, eps: float) -> float:
    """(rho/2) * (int u^3 dA / Q - U^2), with a smooth guard near Q = 0."""
    area = np.pi * radius**2
    excess = grid.integrate(u**3, radius) - q**3 / area**2
    return 0.5 * density * excess * q / (q**2 + eps**2)


def blunting_shape(rho: np.ndarray) -> np.ndarray:
    """Zero-flux, zero-at-wall perturbation (1 - rho^2)(1 - 3 rho^2)."""
    return (1.0 - rho**2) * (1.0 - 3.0 * rho**2)


class _Windkessel:
    def __init__(self, resistance: float, compliance: float, p0: float):
        self.r, self.c, self.p = resistance, compliance, p0

    def peek(self, dt: float, q: float) -> float:
        return (self.p + dt / self.c * q) / (1.0 + dt / (self.r * self.c))


class MockUpstream1D:
    def __init__(self, inflow, params: LumpedParams, q0: float, p0: float):
        self.inflow, self.prm = inflow, params
        self.q, self.p = q0, p0
        self._trial = (q0, p0)

    def trial(self, dt: float, t_next: float, p_interface: float) -> Tuple[float, float]:
        c, r, l = self.prm.upstream_compliance, self.prm.upstream_resistance, self.prm.upstream_inertance
        q_in = float(self.inflow(t_next))
        q = (l / dt * self.q + self.p + dt / c * q_in - p_interface) / (l / dt + dt / c + r)
        p = self.p + dt / c * (q_in - q)
        self._trial = (q, p)
        return q, p

    def accept(self) -> None:
        self.q, self.p = self._trial


class MockLocal3D:
    def __init__(self, spec: CaseSpec, grid: CrossSectionGrid, fluid: Fluid, params: LumpedParams, q0: float, p_distal0: float):
        g = spec.geometry
        z = np.linspace(-g.length / 2, g.length / 2, 401)
        r = g.radius_profile(z)
        self.resistance = trapezoid(8.0 * fluid.viscosity / (np.pi * r**4), z)
        self.inertance = trapezoid(fluid.density / (np.pi * r**2), z)
        self.expansion = 0.5 * fluid.density * params.expansion_coefficient * (g.area / g.throat_area - 1.0) ** 2
        self.grid, self.fluid, self.radius, self.area = grid, fluid, g.radius, g.area
        self.eps = 0.05 * spec.flow.mean_flow
        self.wk = _Windkessel(params.distal_resistance, params.distal_compliance, p_distal0)
        self.q = q0
        self._trial = (q0, p_distal0)

    def flux(self, u: np.ndarray) -> float:
        return float(self.grid.integrate(u, self.radius))

    def trial(self, dt: float, t_next: float, u_inlet: np.ndarray) -> float:
        q = self.flux(u_inlet)
        p_distal = self.wk.peek(dt, q)
        velocity = q / self.area
        dp = (
            self.resistance * q
            + self.inertance * (q - self.q) / dt
            + self.expansion * abs(velocity) * velocity
            + kinetic_pressure(u_inlet, q, self.grid, self.radius, self.fluid.density, self.eps)
        )
        self._trial = (q, p_distal)
        return p_distal + dp

    def accept(self) -> None:
        self.q, self.wk.p = self._trial


class LinearInterface:
    """Uniform Poiseuille segment + Windkessel: what a plain 1D model transfers."""

    def __init__(self, spec: CaseSpec, fluid: Fluid, params: LumpedParams, q0: float, p_distal0: float):
        g = spec.geometry
        self.resistance = 8.0 * fluid.viscosity * g.length / (np.pi * g.radius**4)
        self.inertance = fluid.density * g.length / g.area
        self.wk = _Windkessel(params.distal_resistance, params.distal_compliance, p_distal0)
        self.q = q0
        self._trial = (q0, p_distal0)

    def trial(self, dt: float, t_next: float, q: float) -> float:
        p_distal = self.wk.peek(dt, q)
        self._trial = (q, p_distal)
        return p_distal + self.resistance * q + self.inertance * (q - self.q) / dt

    def accept(self) -> None:
        self.q, self.wk.p = self._trial


class OracleProfile:
    """Mock ground-truth inlet profile; returns dP = 0 (the local model supplies pressure)."""

    def __init__(self, spec: CaseSpec, grid: CrossSectionGrid, fluid: Fluid, n_harmonics: int = 8):
        g = spec.geometry
        self.inflow = spec.flow.waveform(n_harmonics)
        self.womersley = WomersleyProfile(self.inflow, g.radius, fluid.nu, grid.rho)
        self.area = g.area
        self.poiseuille = 2.0 * (1.0 - grid.rho**2) / g.area
        self.blunting = 0.3 * g.severity * blunting_shape(grid.rho) / g.area

    def predict(self, history: np.ndarray, q: float, t: float) -> Tuple[np.ndarray, float]:
        u = self.womersley(t) + (q - float(self.inflow(t))) * self.poiseuille + q * self.blunting
        return u, 0.0


class MockCase:
    """Builds fresh, consistently initialized mock components for one case."""

    def __init__(self, spec: CaseSpec, grid: CrossSectionGrid, fluid: Fluid = Fluid(), params: LumpedParams = LumpedParams(), n_harmonics: int = 8):
        self.spec, self.grid, self.fluid, self.params = spec, grid, fluid, params
        self.n_harmonics = n_harmonics
        q0 = spec.flow.mean_flow
        self.q0 = q0
        self.p_distal0 = q0 * params.distal_resistance
        self.p0 = self.p_distal0 + LinearInterface(spec, fluid, params, q0, self.p_distal0).resistance * q0
        self.p1_0 = self.p0 + params.upstream_resistance * q0

    def components(self):
        upstream = MockUpstream1D(self.spec.flow.waveform(self.n_harmonics), self.params, self.q0, self.p1_0)
        local = MockLocal3D(self.spec, self.grid, self.fluid, self.params, self.q0, self.p_distal0)
        reduced = LinearInterface(self.spec, self.fluid, self.params, self.q0, self.p_distal0)
        return upstream, local, reduced

    def oracle(self) -> OracleProfile:
        return OracleProfile(self.spec, self.grid, self.fluid, self.n_harmonics)

    def run(self, provider, mode: str, steps_per_cycle: int, n_cycles: int, window: int, **kwargs) -> CouplingResult:
        upstream, local, reduced = self.components()
        return run_coupled(
            upstream,
            provider,
            local.flux,
            mode=mode,
            dt=self.spec.flow.period / steps_per_cycle,
            n_steps=steps_per_cycle * n_cycles,
            window=window,
            q_init=self.q0,
            p1_init=self.p1_0,
            p_init=self.p0,
            local=local,
            reduced=reduced,
            **kwargs,
        )
