"""Summarize M1 pilot results (results/*.json from m1_pilot.py) as Markdown tables.

    python scripts/m1_summary.py --work /path/to/m1 > docs/m1_results.md
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from hemocouple.metrics import relative_rmse  # noqa: E402
from hemocouple.mock import LumpedParams  # noqa: E402


def _load(work: Path, stage: str):
    path = work / "results" / f"{stage}.json"
    return json.loads(path.read_text()) if path.exists() else None


def _outlet_pressure(run: dict) -> np.ndarray:
    """p_out recorded by m1_pilot, or replayed from the saved inlet flux (RC Windkessel is deterministic)."""
    if "p_out" in run:
        return np.array(run["p_out"])
    prm = LumpedParams()
    t = np.array(run["t"])
    dt = t[1] - t[0]
    p = 6.0e-6 * prm.distal_resistance  # m1_pilot initial state: mean flow x R_d
    out = []
    for q in run["q"]:
        p = (p + dt / prm.distal_compliance * q) / (1.0 + dt / (prm.distal_resistance * prm.distal_compliance))
        out.append(p)
    return np.array(out)


def _fmt(x, spec=".3g"):
    return "–" if x is None else format(x, spec)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--work", type=Path, required=True)
    ap.add_argument("--budget-core-hours", type=float, default=None, help="optional HPC budget for the case estimate")
    args = ap.parse_args()
    out = []

    po = _load(args.work, "poiseuille")
    if po:
        out += ["### Poiseuille verification (straight tube, steady)", "",
                f"Hagen–Poiseuille Δp = {po['dp_hagen_poiseuille']:.3f} Pa", "",
                "| Mesh | Cells | Δp (Pa) | Error | Inlet flux error | Iterations | Wall (s) |", "|---|---|---|---|---|---|---|"]
        for name, r in po["levels"].items():
            out.append(f"| {name} | {r['cells']:,} | {r['dp']:.3f} | {r['rel_err']:+.2%} | {r['mass_rel_err']:.1e} | "
                       f"{r['iterations']} | {r['wall_time_s']:.0f} |")
        out.append("")

    me = _load(args.work, "mesh")
    if me:
        g = me["gci_dp"]
        out += ["### Mesh convergence (50% stenosis, steady mean flow)", "",
                f"Throat Reynolds number at mean flow ≈ {me['reynolds_throat_mean_flow']:.0f}", "",
                "| Mesh | Cells | Δp (Pa) | Iterations | Converged | Wall (s) | Procs |", "|---|---|---|---|---|---|---|"]
        for name, r in me["levels"].items():
            out.append(f"| {name} | {r['cells']:,} | {r['dp']:.3f} | {r['iterations']} | {r['converged']} | "
                       f"{r['wall_time_s']:.0f} | {r['n_procs']} |")
        out += ["", f"Convergence: {g.get('kind')}; observed order {_fmt(g['order'])}; "
                f"GCI(fine) {_fmt(g['gci_fine'], '.2%') if g['monotone'] else '–'}; "
                f"extrapolated Δp {_fmt(g['extrapolated'], '.3f')} Pa", ""]

    pu = _load(args.work, "pulsatile")
    if pu:
        out += ["### Pulsatile cost and time-step sensitivity (coarse mesh, prescribed Womersley inflow)", "",
                "| CFD steps / coupling step | Cells | Wall per cycle (s) | Wall per coupling step (s) | Max inlet-flux error |",
                "|---|---|---|---|---|"]
        for k, r in pu["runs"].items():
            per_step = r["wall_time_s"] / len(r["t"])
            out.append(f"| {k} | {r['cells']:,} | {r['wall_per_cycle_s']:.0f} | {per_step:.2f} | {r['mass_rel_err_max']:.1e} |")
        w20 = pu["runs"]["20"]["wall_time_s"] / len(pu["runs"]["20"]["t"])
        w40 = pu["runs"]["40"]["wall_time_s"] / len(pu["runs"]["40"]["t"])
        overhead, per_cfd_step = 2 * w20 - w40, (w40 - w20) / 20
        out += ["", f"Linear split of the per-run cost: fixed restart overhead ≈ {overhead:.2f} s per run "
                f"(docker exec, solver start-up, mesh read, file I/O) + {per_cfd_step * 1e3:.0f} ms per CFD time step."]
        dp = {k: np.array(r["p_in"]) - _outlet_pressure(r) for k, r in pu["runs"].items()}
        n = len(dp["40"]) // 2
        last = slice(-n, None)
        sens = relative_rmse(dp["20"][last], dp["40"][last])
        cyc = relative_rmse(dp["40"][last], dp["40"][:n])
        out += ["", f"3D pressure drop Δp = p_in − p_out over the last cycle: range {dp['40'][last].min():.1f} to "
                f"{dp['40'][last].max():.1f} Pa (40 steps).",
                f"Time-step sensitivity of Δp (20 vs 40 steps, relative RMSE): {sens:.2%}; "
                f"cycle-to-cycle change of Δp: {cyc:.2%}.", ""]

    co = _load(args.work, "coupled")
    if co:
        it = np.array(co["iterations"])
        p, q = np.array(co["p_interface"]), np.array(co["q"])
        n = len(it) // 2
        out += ["### Coupled pilot (1D vessel + pimpleFoam, Mode A, Poiseuille inlet profile, 20 CFD steps per coupling step)", "",
                f"- Cells: {co['cells']:,}; all {len(it)} coupling steps converged: {co['converged_all']} (tol 1e-5 relative)",
                f"- Coupling iterations per step: mean {co['iterations_mean']:.2f}, max {co['iterations_max']} "
                f"(Aitken-relaxed fixed point); CFD runs: {co['cfd_runs']}",
                f"- Wall time per cardiac cycle: {co['wall_per_cycle_s']:.0f} s",
                f"- Warm-up: mean interface pressure {p[:n].mean():.0f} → {p[n:].mean():.0f} Pa, mean flow "
                f"{q[:n].mean() * 1e6:.3f} → {q[n:].mean() * 1e6:.3f} ml/s (target {6.0:.3f}) over cycles 1 → 2", ""]
        if pu:
            runs_per_cycle = co["cfd_runs"] / 2
            share = runs_per_cycle * overhead / co["wall_per_cycle_s"]
            out += [f"- Estimated restart-overhead share of the coupled wall time: {share:.0%} "
                    f"({overhead:.2f} s × {runs_per_cycle:.0f} runs per cycle)", ""]
        if args.budget_core_hours and pu:
            per_case_h = co["wall_per_cycle_s"] * 6 / 3600  # 6 cycles per case, 1 core
            out.append(f"- Estimated cases within {args.budget_core_hours:.0f} core-hours (6 cycles, coarse mesh, serial): "
                       f"{args.budget_core_hours / per_case_h:.0f}")
            out.append("")
    print("\n".join(out))


if __name__ == "__main__":
    main()
