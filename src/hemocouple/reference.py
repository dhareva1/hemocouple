"""Prescribed-flow 3D reference runs that produce operator training labels (docs/m2_design.md).

Domain: Womersley inlet at z = 0, interface Γ at z = L_ext, stenosis starting
d_gamma downstream of Γ, outlet 20 R further, RC Windkessel at the outlet.
Every coupling time t_k = k T / steps_per_cycle is sampled:
    q          prescribed flow (equals the mesh flux)
    p1d        area-averaged p on the plane Γ − probe_distance (operator input)
    p_interface area-averaged p on Γ
    profiles   axial velocity on the reference grid at Γ, flux-corrected to q
    p_lin      uniform-Poiseuille segment Γ→outlet + the same Windkessel
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Tuple

import numpy as np

from . import foam
from .coupling import CouplingResult
from .cross_section import CrossSectionGrid
from .dataset import CaseRecord
from .geometry import CaseSpec
from .mock import LumpedParams
from .womersley import WomersleyProfile


@dataclass(frozen=True)
class ReferenceConfig:
    extension_radii: float = 4.0  # L_ext / R, inlet to Γ
    gap_radii: float = 2.0  # d_gamma / R, Γ to stenosis start
    downstream_radii: float = 20.0
    probe_distance: float = 0.01  # m, P_up plane upstream of Γ
    steps_per_cycle: int = 100
    n_cycles: int = 5
    refine: float = 1.5  # relative to the M1 coarse mesh (8 x 8 cells, 0.6 mm at R = 3 mm)
    radial_grading: float = 1.0  # > 1 refines the outer blocks toward the wall (Stokes layer)
    max_co: float = 0.8
    n_procs: int = 8  # M4 Pro: 8 performance cores; more MPI ranks land on efficiency cores and slow every rank
    density: float = 1060.0
    viscosity: float = 3.5e-3

    def foam_spec(self, spec: CaseSpec) -> foam.FoamCaseSpec:
        r = spec.geometry.radius
        base = foam.FoamCaseSpec(
            spec.geometry, upstream_length=(self.extension_radii + self.gap_radii) * r,
            downstream_length=self.downstream_radii * r, n_square=8, n_radial=8,
            axial_cell=6.0e-4 * r / 3.0e-3, density=self.density, viscosity=self.viscosity,
            radial_grading=self.radial_grading,
        )
        return base.refined(self.refine)

    def gamma_z(self, spec: CaseSpec) -> float:
        return self.extension_radii * spec.geometry.radius


def linear_interface_pressure(
    q: np.ndarray, dt: float, length: float, radius: float, cfg: ReferenceConfig, lumped: LumpedParams, p_distal0: float
) -> np.ndarray:
    """1D-transferred interface pressure: uniform Poiseuille segment + RC Windkessel, implicit Euler like the 3D outlet."""
    area = np.pi * radius**2
    res = 8.0 * cfg.viscosity * length / (np.pi * radius**4)
    ind = cfg.density * length / area
    out, p_d, q_prev = [], p_distal0, None
    for qk in q:
        p_d = (p_d + dt / lumped.distal_compliance * qk) / (1.0 + dt / (lumped.distal_resistance * lumped.distal_compliance))
        dqdt = 0.0 if q_prev is None else (qk - q_prev) / dt
        out.append(p_d + res * qk + ind * dqdt)
        q_prev = qk
    return np.array(out)


def flux_correct(profiles: np.ndarray, q: np.ndarray, grid: CrossSectionGrid, radius: float) -> Tuple[np.ndarray, np.ndarray]:
    """Additive zero-at-wall (Poiseuille-shaped) correction so each sampled profile carries exactly q.

    Returns (corrected profiles, relative correction |q - flux| / mean|q| per sample).
    """
    flux = grid.integrate(profiles, radius)
    shape = 2.0 * (1.0 - grid.rho**2) / (np.pi * radius**2)
    corrected = profiles + (q - flux)[:, None] * shape[None, :]
    return corrected, np.abs(q - flux) / np.mean(np.abs(q))


def run_reference_case(
    runner: foam.DockerFoam, work: Path, spec: CaseSpec, grid: CrossSectionGrid, cfg: ReferenceConfig,
    lumped: LumpedParams = LumpedParams(),
) -> CaseRecord:
    fspec = cfg.foam_spec(spec)
    r, period = spec.geometry.radius, spec.flow.period
    wave = spec.flow.waveform()
    womersley = WomersleyProfile(wave, r, cfg.viscosity / cfg.density, grid.rho)
    z_gamma = cfg.gamma_z(spec)
    dt = period / cfg.steps_per_cycle
    z_end = fspec.upstream_length + spec.geometry.length
    p_distal0 = spec.flow.mean_flow * lumped.distal_resistance
    res = foam.run_prescribed(
        runner, Path(work) / spec.case_id, fspec, grid, womersley,
        period=period, n_cycles=cfg.n_cycles, samples_per_cycle=cfg.steps_per_cycle,
        distal_resistance=lumped.distal_resistance, distal_compliance=lumped.distal_compliance,
        p_distal0=p_distal0, probe_points=np.array([[0.0, 0.0, z_end + 2 * r * d] for d in (1, 2, 4)]),
        max_co=cfg.max_co, n_procs=cfg.n_procs, grid_probe_z=z_gamma,
        planes={"pGamma": z_gamma, "pUp": z_gamma - cfg.probe_distance}, sample_interval=dt,
    )
    t = dt * np.arange(1, cfg.n_cycles * cfg.steps_per_cycle + 1)
    q = np.asarray(wave(t), dtype=float)

    def at(rows: np.ndarray) -> np.ndarray:
        return np.interp(t, rows[:, 0], rows[:, 1])

    grid_t, grid_u = res["grid_t"], res["grid_u"][:, :, 2]
    profiles = np.stack([np.interp(t, grid_t, grid_u[:, i]) for i in range(grid.n_points)], -1)
    profiles, correction = flux_correct(profiles, q, grid, r)
    p_lin = linear_interface_pressure(q, dt, fspec.length - z_gamma, r, cfg, lumped, p_distal0)
    result = CouplingResult(
        t=t, q=q, p1d=at(res["pUp"]), p_interface=at(res["pGamma"]), p_lin=p_lin, q_local=q,
        dp_pred=np.zeros_like(t), iterations=np.zeros(len(t), int), converged=np.ones(len(t), bool),
        provider_time=0.0, wall_time=res["wall_time_s"], profiles=profiles, meta={"dt": dt},
    )
    s = slice(-cfg.steps_per_cycle, None)
    # OpenFOAM interpolates the inlet linearly between sample times, so compare against that interpolant.
    q_imposed = np.interp(res["t"], res["sample_times"], res["sample_q"])
    flux_err = np.abs(res["q_mesh"] - q_imposed) / np.mean(np.abs(q))
    return CaseRecord.from_result(
        spec, result, steps_per_cycle=cfg.steps_per_cycle, source="openfoam-prescribed", cells=fspec.n_cells,
        refine=cfg.refine, wall_time_s=res["wall_time_s"], n_procs=cfg.n_procs,
        label_flux_correction_max=float(correction[s].max()), label_flux_correction_mean=float(correction[s].mean()),
        inlet_flux_error_max=float(flux_err.max()), z_gamma=z_gamma,
    )


class ReducedSegment:
    """ReducedInterface (trial/accept) twin of ``linear_interface_pressure``, for Mode B coupling."""

    def __init__(self, length: float, radius: float, cfg: ReferenceConfig, lumped: LumpedParams, q0: float, p_distal0: float):
        area = np.pi * radius**2
        self.res = 8.0 * cfg.viscosity * length / (np.pi * radius**4)
        self.ind = cfg.density * length / area
        self.rd, self.cd = lumped.distal_resistance, lumped.distal_compliance
        self.q, self.p_d = q0, p_distal0
        self._trial = (q0, p_distal0)

    def trial(self, dt: float, t_next: float, q: float) -> float:
        p_d = (self.p_d + dt / self.cd * q) / (1.0 + dt / (self.rd * self.cd))
        self._trial = (q, p_d)
        return p_d + self.res * q + self.ind * (q - self.q) / dt

    def accept(self) -> None:
        self.q, self.p_d = self._trial
