"""M2 gate: is the post-stenotic flow laminar and cycle-periodic across the parameter range (proposal section 6)?

    python scripts/m2_transition_check.py --work /path/outside/onedrive [--cases pilot high worst]

For each case: one continuous parallel pimpleFoam run (medium mesh, adaptive dt,
prescribed Womersley inflow, RC Windkessel outlet) over several cycles. Velocity
probes sit downstream of the stenosis. Laminar periodic flow gives a near-zero
cycle-to-cycle difference and little energy above harmonic 20.
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
warnings.filterwarnings("ignore", message=".*encountered in matmul", category=RuntimeWarning)

from hemocouple import foam  # noqa: E402
from hemocouple.cross_section import CrossSectionGrid  # noqa: E402
from hemocouple.geometry import FlowCondition, StenosisGeometry  # noqa: E402
from hemocouple.metrics import cycle_periodicity  # noqa: E402
from hemocouple.mock import LumpedParams  # noqa: E402
from hemocouple.womersley import WomersleyProfile  # noqa: E402

GRID = CrossSectionGrid.gauss_polar(12, 8)
DENSITY, VISCOSITY = 1060.0, 3.5e-3

CASES = {
    "pilot": dict(radius=3.0e-3, severity=0.50, length_ratio=4.0,
                  flow=FlowCondition(6.0e-6, 10.0e-6, 0.8, 0.15)),
    "high": dict(radius=3.0e-3, severity=0.75, length_ratio=4.0,
                 flow=FlowCondition(6.0e-6, 10.0e-6, 0.8, 0.15)),
    "worst": dict(radius=2.5e-3, severity=0.75, length_ratio=4.0,
                  flow=FlowCondition(8.0e-6, 14.0e-6, 0.7, 0.12)),
}


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def run_case(runner, work: Path, name: str, n_cycles: int, n_procs: int, refine: float) -> dict:
    c = CASES[name]
    r = c["radius"]
    geom = StenosisGeometry(r, c["severity"], c["length_ratio"] * r)
    spec = foam.FoamCaseSpec(geom, upstream_length=6 * r, downstream_length=20 * r,
                             n_square=8, n_radial=8, axial_cell=6.0e-4 * r / 3.0e-3).refined(refine)
    flow = c["flow"]
    wave = flow.waveform()
    wom = WomersleyProfile(wave, r, VISCOSITY / DENSITY, GRID.rho)
    tt = np.linspace(0, flow.period, 2000, endpoint=False)
    q_peak = float(wave(tt).max())
    re_throat_peak = 2 * q_peak / (math.pi * geom.throat_radius * VISCOSITY / DENSITY)
    z_end = spec.upstream_length + geom.length
    probes = []
    for d in (1, 2, 4, 8):
        z = z_end + d * 2 * r
        probes += [(0.0, 0.0, z), (0.5 * r, 0.0, z)]
    lumped = LumpedParams()
    log(f"{name}: {spec.n_cells} cells, severity {geom.severity}, peak Q {q_peak * 1e6:.1f} ml/s, "
        f"peak throat Re {re_throat_peak:.0f}")
    res = foam.run_prescribed(
        runner, work / "cases" / f"transition_{name}", spec, GRID, wom,
        period=flow.period, n_cycles=n_cycles, samples_per_cycle=200,
        distal_resistance=lumped.distal_resistance, distal_compliance=lumped.distal_compliance,
        p_distal0=flow.mean_flow * lumped.distal_resistance, probe_points=np.array(probes), n_procs=n_procs,
    )
    dp = res["p_in"] - res["p_out"]
    out = {
        "cells": res["cells"], "severity": geom.severity, "radius": r, "flow": flow.__dict__,
        "q_peak": q_peak, "re_throat_peak": re_throat_peak, "wall_time_s": res["wall_time_s"],
        "dp": cycle_periodicity(res["t"], dp, flow.period),
        "flux_rel_err_max": float(np.max(np.abs(res["q_mesh"] - np.interp(res["t"], res["sample_times"], res["sample_q"])))
                                  / np.mean(np.abs(res["sample_q"]))),
        "probes": {},
    }
    for i, (x, y, z) in enumerate(probes):
        key = f"{(z - z_end) / (2 * r):.0f}D_{'axis' if x == 0 else 'r0.5R'}"
        out["probes"][key] = cycle_periodicity(res["probe_t"], res["probe_u"][:, i, 2], flow.period)
    worst = max(v["cycle_diff"] for v in out["probes"].values())
    hf = max(v["hf_fraction"] for v in out["probes"].values())
    log(f"{name}: dp cycle_diff {out['dp']['cycle_diff']:.2e}; worst probe cycle_diff {worst:.2e}, "
        f"hf_fraction {hf:.2e}; wall {res['wall_time_s']:.0f} s")
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--work", type=Path, required=True)
    ap.add_argument("--cases", nargs="+", default=list(CASES), choices=list(CASES))
    ap.add_argument("--cycles", type=int, default=4)
    ap.add_argument("--procs", type=int, default=8)
    ap.add_argument("--refine", type=float, default=1.5, help="mesh factor relative to the M1 coarse mesh (1.5 = medium)")
    args = ap.parse_args()
    work = args.work.resolve()
    (work / "results").mkdir(parents=True, exist_ok=True)
    runner = foam.DockerFoam(work, name="hemocouple-m2").start()
    try:
        for name in args.cases:
            result = run_case(runner, work, name, args.cycles, args.procs, args.refine)
            (work / "results" / f"transition_{name}.json").write_text(json.dumps(result, indent=2))
            log(f"{name}: wrote results")
    finally:
        runner.stop()


if __name__ == "__main__":
    main()
