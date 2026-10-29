"""Mesh sensitivity of the pulsatile training labels for severe and short stenoses.

    python scripts/mesh_study_labels.py --data ~/hemocouple-work/m2 --out ~/hemocouple-work/mesh_study [--procs 6]

Reruns dataset cases with the reference configuration of the dataset (radial grading 4:1) at the refinements
1.0 / 1.5 / 2.25 (ratio 1.5; 1.5 is the label mesh) and reports, over the last cycle, the GCI of the peak and the
cycle-mean stenosis pressure drop ΔP = P_Γ − P_lin, the RMS difference of ΔP and of the interface profile between
levels, and the same quantities for the label mesh against the Richardson-extrapolated value. One container per
case; cases run in parallel. Writes <out>/summary.json and <out>/summary.md.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

CASES = ["sten0014", "sten0024", "sten0016", "sten0029"]  # severe + short, highest Re_t, short, most severe
LEVELS = {"coarse": 1.0, "medium": 1.5, "fine": 2.25}
N_CYCLES = 4


def run_case(job: tuple) -> dict:
    data, out, cid, procs = job
    from hemocouple import foam
    from hemocouple.cross_section import CrossSectionGrid
    from hemocouple.dataset import load_records
    from hemocouple.geometry import CaseSpec
    from hemocouple.reference import ReferenceConfig, run_reference_case

    meta = json.loads((data / "dataset.json").read_text())
    grid = CrossSectionGrid.gauss_polar(meta["n_r"], meta["n_theta"])
    spec = CaseSpec.from_dict(load_records(data, [cid])[0].spec)
    runner = foam.DockerFoam(out, name=f"hemocouple-mesh-{cid}").start()
    try:
        for name, refine in LEVELS.items():
            target = out / name / f"{cid}.npz"
            if target.exists():
                continue
            cfg = ReferenceConfig(**{**meta["reference"], "refine": refine, "n_cycles": N_CYCLES, "n_procs": procs})
            rec = run_reference_case(runner, out / "_cases" / name, spec, grid, cfg)
            target.parent.mkdir(parents=True, exist_ok=True)
            rec.save(target.parent)
            shutil.rmtree(out / "_cases" / name / cid, ignore_errors=True)
            print(f"[{time.strftime('%H:%M:%S')}] {cid} {name}: {rec.meta['cells']} cells, {rec.meta['wall_time_s']:.0f} s", flush=True)
    finally:
        runner.stop()
    return {"case": cid}


def gci(f1: float, f2: float, f3: float, r: float = 1.5) -> dict:
    """Roache GCI (fine-mesh and medium-mesh) for f1 fine, f2 medium, f3 coarse."""
    sys.path.insert(0, str(ROOT / "src"))
    from hemocouple.metrics import grid_convergence_index

    g = grid_convergence_index(f1, f2, f3, r)
    if g["monotone"]:
        g["medium_vs_extrapolated"] = abs(f2 / g["extrapolated"] - 1.0)
        g["gci_medium"] = g["gci_fine"] * r ** g["order"]
    return g


def summarize(data: Path, out: Path) -> dict:
    from hemocouple.dataset import load_records

    meta = json.loads((data / "dataset.json").read_text())
    n = meta["steps_per_cycle"]
    w = None
    summary = {}
    for cid in CASES:
        recs = {name: load_records(out / name, [cid])[0] for name in LEVELS}
        if w is None:
            from hemocouple.cross_section import CrossSectionGrid

            w = CrossSectionGrid.gauss_polar(meta["n_r"], meta["n_theta"]).weights
        dp = {k: r.dp[-n:] for k, r in recs.items()}
        u = {k: r.profiles[-n:] for k, r in recs.items()}
        rng = float(np.ptp(dp["fine"]))
        rel_u = lambda a, b: float(np.sqrt(np.sum((a - b) ** 2 * w) / np.sum(b**2 * w)))  # noqa: E731
        spec = recs["fine"].spec["geometry"]
        summary[cid] = {
            "severity": spec["severity"], "length_ratio": spec["length"] / spec["radius"],
            "cells": {k: r.meta["cells"] for k, r in recs.items()},
            "dp_peak_pa": {k: float(v.max()) for k, v in dp.items()},
            "dp_mean_pa": {k: float(v.mean()) for k, v in dp.items()},
            "gci_dp_peak": gci(dp["fine"].max(), dp["medium"].max(), dp["coarse"].max()),
            "gci_dp_mean": gci(dp["fine"].mean(), dp["medium"].mean(), dp["coarse"].mean()),
            "dp_rms_diff_medium_fine_over_range": float(np.sqrt(np.mean((dp["medium"] - dp["fine"]) ** 2)) / rng),
            "dp_rms_diff_coarse_fine_over_range": float(np.sqrt(np.mean((dp["coarse"] - dp["fine"]) ** 2)) / rng),
            "profile_rel_l2_medium_fine": rel_u(u["medium"], u["fine"]),
            "profile_rel_l2_coarse_fine": rel_u(u["coarse"], u["fine"]),
            "wall_time_s": {k: r.meta["wall_time_s"] for k, r in recs.items()},
        }
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    lines = ["# Label mesh study (pulsatile, last cycle; label mesh = medium)", "",
             "| Case | Severity | Ls/R | peak ΔP c/m/f (Pa) | order | GCI_fine | GCI_medium | medium vs extrap. | ΔP RMS diff m–f / range | profile m–f |",
             "|---|---|---|---|---|---|---|---|---|---|"]
    for cid, s in summary.items():
        g = s["gci_dp_peak"]
        p = s["dp_peak_pa"]
        lines.append(f"| {cid} | {s['severity']:.2f} | {s['length_ratio']:.2f} | {p['coarse']:.1f} / {p['medium']:.1f} / {p['fine']:.1f} | "
                     f"{g['order']:.2f} | {100 * g['gci_fine']:.2f}% | {100 * g.get('gci_medium', float('nan')):.2f}% | "
                     f"{100 * g.get('medium_vs_extrapolated', float('nan')):.2f}% | {100 * s['dp_rms_diff_medium_fine_over_range']:.2f}% | "
                     f"{100 * s['profile_rel_l2_medium_fine']:.2f}% |")
    lines += ["", "Cycle-mean ΔP: " + "; ".join(f"{cid} {s['gci_dp_mean']['kind']} GCI_fine "
                                                 f"{100 * s['gci_dp_mean']['gci_fine']:.2f}%" for cid, s in summary.items())]
    (out / "summary.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    return summary


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--procs", type=int, default=6)
    ap.add_argument("--summarize", action="store_true")
    args = ap.parse_args()
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=True)
    if not args.summarize:
        with ProcessPoolExecutor(len(CASES)) as pool:
            list(pool.map(run_case, [(args.data, out, cid, args.procs) for cid in CASES]))
    summarize(args.data, out)


if __name__ == "__main__":
    main()
