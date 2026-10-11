"""Milestone M1 pilot: verify the 3D solver, measure cost, and test 1D-3D coupling on one geometry.

    python scripts/m1_pilot.py --work /path/outside/onedrive --stage all

Stages (each writes <work>/results/<stage>.json):
    poiseuille  straight tube, steady parabolic inflow: dp vs Hagen-Poiseuille
    mesh        steady flow through a 50% stenosis on 3 meshes (ratio 1.5): GCI of dp
    pulsatile   prescribed Womersley inflow, coarse mesh, 2 CFD time steps: cost per cycle + dt sensitivity
    coupled     in-house 1D vessel + pimpleFoam in Mode A with a Poiseuille inlet profile
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
import warnings
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
# numpy 2.0.x + macOS Accelerate emits spurious FP warnings from matmul.
warnings.filterwarnings("ignore", message=".*encountered in matmul", category=RuntimeWarning)

from hemocouple import foam  # noqa: E402
from hemocouple.coupling import run_coupled  # noqa: E402
from hemocouple.cross_section import CrossSectionGrid  # noqa: E402
from hemocouple.geometry import FlowCondition, StenosisGeometry  # noqa: E402
from hemocouple.metrics import grid_convergence_index, relative_rmse  # noqa: E402
from hemocouple.mock import LumpedParams  # noqa: E402
from hemocouple.oned import OneDVessel, VesselParams  # noqa: E402
from hemocouple.providers import PoiseuilleProfile  # noqa: E402
from hemocouple.womersley import WomersleyProfile  # noqa: E402

RADIUS = 3.0e-3
GEOMETRY = StenosisGeometry(RADIUS, severity=0.5, length=4 * RADIUS)
FLOW = FlowCondition(mean_flow=6.0e-6, amplitude=10.0e-6, period=0.8, systole_peak=0.15)
BASE = foam.FoamCaseSpec(GEOMETRY, upstream_length=6 * RADIUS, downstream_length=20 * RADIUS,
                         n_square=8, n_radial=8, axial_cell=6.0e-4)
GRID = CrossSectionGrid.gauss_polar(12, 8)
STEPS_PER_CYCLE = 100


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def save(work: Path, stage: str, result: dict) -> None:
    out = work / "results"
    out.mkdir(parents=True, exist_ok=True)
    (out / f"{stage}.json").write_text(json.dumps(result, indent=2))
    log(f"{stage}: wrote {out / (stage + '.json')}")


def poiseuille_profile(q: float) -> np.ndarray:
    return PoiseuilleProfile(GRID, RADIUS).predict(None, q, 0.0)[0]


def stage_poiseuille(runner: foam.DockerFoam, work: Path, n_procs: int) -> dict:
    straight = foam.FoamCaseSpec(StenosisGeometry(RADIUS, 0.0, 4 * RADIUS), BASE.upstream_length,
                                 BASE.downstream_length, BASE.n_square, BASE.n_radial, BASE.axial_cell)
    q = FLOW.mean_flow
    exact = 8 * straight.viscosity * straight.length * q / (math.pi * RADIUS**4)
    levels = {}
    for name, factor in (("coarse", 1.0), ("medium", 1.5), ("fine", 2.25)):
        spec = straight.refined(factor)
        r = foam.run_steady(runner, work / "cases" / f"poiseuille_{name}", spec, GRID, poiseuille_profile(q),
                            n_procs=n_procs if spec.n_cells > 2e5 else 1)
        r.update(rel_err=r["dp"] / exact - 1, mass_rel_err=r["q_mesh"] / q - 1,
                 polygon_area_ratio=spec.inlet_area / (math.pi * RADIUS**2))
        levels[name] = r
        log(f"poiseuille/{name}: {spec.n_cells} cells, dp = {r['dp']:.3f} Pa (exact {exact:.3f}), "
            f"err {r['rel_err']:+.2%}, {r['wall_time_s']:.0f} s")
    gci = grid_convergence_index(levels["fine"]["dp"], levels["medium"]["dp"], levels["coarse"]["dp"], ratio=1.5)
    return {"dp_hagen_poiseuille": exact, "levels": levels, "gci_dp": gci,
            "extrapolated_rel_err": gci["extrapolated"] / exact - 1 if gci["monotone"] else None}


def stage_mesh(runner: foam.DockerFoam, work: Path, n_procs: int) -> dict:
    q = FLOW.mean_flow
    levels = {}
    for name, factor in (("coarse", 1.0), ("medium", 1.5), ("fine", 2.25)):
        spec = BASE.refined(factor)
        log(f"mesh/{name}: {spec.n_cells} cells")
        levels[name] = foam.run_steady(runner, work / "cases" / f"mesh_{name}", spec, GRID, poiseuille_profile(q),
                                       n_procs=n_procs if spec.n_cells > 2e5 else 1)
        log(f"mesh/{name}: dp = {levels[name]['dp']:.4f} Pa, {levels[name]['iterations']} it, "
            f"{levels[name]['wall_time_s']:.0f} s")
    gci = grid_convergence_index(levels["fine"]["dp"], levels["medium"]["dp"], levels["coarse"]["dp"], ratio=1.5)
    re_throat = 2 * q / (math.pi * GEOMETRY.throat_radius * BASE.viscosity / BASE.density)
    return {"levels": levels, "gci_dp": gci, "reynolds_throat_mean_flow": re_throat}


def _prescribed_run(runner, case: Path, spec: foam.FoamCaseSpec, substeps: int, n_cycles: int) -> dict:
    wave = FLOW.waveform()
    wom = WomersleyProfile(wave, RADIUS, BASE.viscosity / BASE.density, GRID.rho)
    lumped = LumpedParams()
    dt = FLOW.period / STEPS_PER_CYCLE
    q0 = float(wave(0.0))
    local = foam.FoamLocal3D(runner, case, spec, GRID, coupling_dt=dt, substeps=substeps,
                             distal_resistance=lumped.distal_resistance, distal_compliance=lumped.distal_compliance,
                             q0=q0, p_distal0=FLOW.mean_flow * lumped.distal_resistance)
    local.initialize(wom(0.0))
    t_hist, p_hist, q_hist, qm_hist, pout_hist = [], [], [], [], []
    tic = time.perf_counter()
    for n in range(STEPS_PER_CYCLE * n_cycles):
        t = (n + 1) * dt
        u = wom(t)
        p = local.trial(dt, t, u)
        local.accept()
        t_hist.append(t), p_hist.append(p), q_hist.append(local.flux(u)), qm_hist.append(local.last_q_mesh)
        pout_hist.append(local.p_distal)
        if (n + 1) % 25 == 0:
            log(f"  {case.name}: step {n + 1}/{STEPS_PER_CYCLE * n_cycles}, p_in = {p:.1f} Pa")
    wall = time.perf_counter() - tic
    q_arr, qm_arr = np.array(q_hist), np.array(qm_hist)
    return {"t": t_hist, "p_in": p_hist, "p_out": pout_hist, "q": q_hist, "cells": spec.n_cells, "substeps": substeps,
            "wall_time_s": wall, "solver_time_s": local.solver_time, "cfd_exec_time_s": local.cfd_time,
            "wall_per_cycle_s": wall / n_cycles,
            "mass_rel_err_max": float(np.max(np.abs(qm_arr - q_arr)) / np.mean(np.abs(q_arr)))}


def stage_pulsatile(runner: foam.DockerFoam, work: Path) -> dict:
    runs = {}
    for substeps in (20, 40):
        log(f"pulsatile: coarse mesh, {substeps} CFD steps per coupling step")
        runs[substeps] = _prescribed_run(runner, work / "cases" / f"puls_{substeps}", BASE, substeps, n_cycles=2)
    s = slice(-STEPS_PER_CYCLE, None)
    # Compare the 3D pressure drop: p_in is dominated by the (identical) Windkessel outlet pressure.
    dp = {k: np.array(v["p_in"]) - np.array(v["p_out"]) for k, v in runs.items()}
    sens = relative_rmse(dp[20][s], dp[40][s])
    periodic = relative_rmse(dp[40][s], dp[40][:STEPS_PER_CYCLE])
    return {"runs": {str(k): v for k, v in runs.items()}, "dt_sensitivity_dp_rel_rmse": sens,
            "cycle_to_cycle_dp_rel_rmse": periodic}


def stage_coupled(runner: foam.DockerFoam, work: Path, n_cycles: int) -> dict:
    lumped = LumpedParams()
    dt = FLOW.period / STEPS_PER_CYCLE
    wave = FLOW.waveform()
    q0 = float(wave(0.0))
    p0 = FLOW.mean_flow * lumped.distal_resistance
    vessel = OneDVessel(VesselParams(radius=RADIUS, p_ref=p0), lambda t: float(wave(t)), q0=q0, p0=p0)
    # 20 CFD steps per coupling step: the pulsatile stage showed a 0.01% change in dp versus 40.
    local = foam.FoamLocal3D(runner, work / "cases" / "coupled", BASE, GRID, coupling_dt=dt, substeps=20,
                             distal_resistance=lumped.distal_resistance, distal_compliance=lumped.distal_compliance,
                             q0=q0, p_distal0=p0)
    provider = PoiseuilleProfile(GRID, RADIUS)
    local.initialize(provider.predict(None, q0, 0.0)[0])

    class _Logged:
        def __init__(self, inner):
            self.inner, self.n = inner, 0

        def trial(self, dt, t_next, u):
            return self.inner.trial(dt, t_next, u)

        def accept(self):
            self.inner.accept()
            self.n += 1
            if self.n % 10 == 0:
                log(f"  coupled: step {self.n}/{STEPS_PER_CYCLE * n_cycles}, runs so far {self.inner.n_runs}")

        def flux(self, u):
            return self.inner.flux(u)

    logged = _Logged(local)
    res = run_coupled(vessel, provider, local.flux, mode="A", dt=dt, n_steps=STEPS_PER_CYCLE * n_cycles, window=1,
                      q_init=q0, p1_init=p0, p_init=p0, local=logged, tol=1e-5, max_iter=20, omega0=0.5)
    return {"t": res.t.tolist(), "q": res.q.tolist(), "p1d": res.p1d.tolist(), "p_interface": res.p_interface.tolist(),
            "iterations": res.iterations.tolist(), "converged_all": bool(res.converged.all()),
            "iterations_mean": float(res.iterations.mean()), "iterations_max": int(res.iterations.max()),
            "cfd_runs": local.n_runs, "wall_time_s": res.wall_time, "solver_time_s": local.solver_time,
            "cfd_exec_time_s": local.cfd_time,
            "wall_per_cycle_s": res.wall_time / n_cycles, "cells": BASE.n_cells}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--work", type=Path, required=True, help="work dir (keep it outside OneDrive)")
    ap.add_argument("--stage", choices=["poiseuille", "mesh", "pulsatile", "coupled", "all"], default="all")
    ap.add_argument("--procs", type=int, default=8)
    ap.add_argument("--coupled-cycles", type=int, default=2)
    args = ap.parse_args()
    work = args.work.resolve()
    work.mkdir(parents=True, exist_ok=True)
    runner = foam.DockerFoam(work).start()
    stages = ["poiseuille", "mesh", "pulsatile", "coupled"] if args.stage == "all" else [args.stage]
    for stage in stages:
        log(f"=== {stage} ===")
        tic = time.perf_counter()
        if stage == "poiseuille":
            result = stage_poiseuille(runner, work, args.procs)
        elif stage == "mesh":
            result = stage_mesh(runner, work, args.procs)
        elif stage == "pulsatile":
            result = stage_pulsatile(runner, work)
        else:
            result = stage_coupled(runner, work, args.coupled_cycles)
        result["stage_wall_time_s"] = time.perf_counter() - tic
        save(work, stage, result)


if __name__ == "__main__":
    main()
