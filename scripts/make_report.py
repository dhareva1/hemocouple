"""Collect pipeline outputs under --work into report.md (dataset, training, offline and coupled evaluation).

    python scripts/make_report.py --work ~/hemocouple-work
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from hemocouple.geometry import CaseSpec, peak_throat_reynolds  # noqa: E402


def fmt(x, spec=".3f"):
    return "–" if x is None or (isinstance(x, float) and not np.isfinite(x)) else format(x, spec)


def dataset_section(data: Path) -> list:
    if not (data / "cases.json").exists():
        return ["## Dataset", "", "_not generated yet_", ""]
    specs = [CaseSpec.from_dict(d) for d in json.loads((data / "cases.json").read_text())]
    done = [s for s in specs if (data / f"{s.case_id}.npz").exists()]
    out = ["## Dataset", "", f"{len(done)} / {len(specs)} cases generated."]
    if done:
        corr, wall, cells = [], [], []
        for s in done:
            with np.load(data / f"{s.case_id}.npz") as z:
                meta = json.loads(str(z["header"]))["meta"]
            corr.append(meta.get("label_flux_correction_max", np.nan))
            wall.append(meta.get("wall_time_s", np.nan))
            cells.append(meta.get("cells", np.nan))
        re = [peak_throat_reynolds(s.geometry, s.flow) for s in done if s.geometry.severity > 0]
        out += [
            "",
            f"- Peak throat Re (stenosis cases): max {max(re):.0f}, median {np.median(re):.0f} (laminar cap 1700)",
            f"- Label flux correction (max per case): median {np.nanmedian(corr):.2%}, worst {np.nanmax(corr):.2%}",
            f"- Cells per case: {np.nanmin(cells):,.0f}–{np.nanmax(cells):,.0f}; wall time per case: median "
            f"{np.nanmedian(wall) / 60:.1f} min, total {np.nansum(wall) / 3600:.1f} h",
        ]
    if (data / "splits.json").exists():
        sp = json.loads((data / "splits.json").read_text())
        out.append("- Splits: " + ", ".join(f"{k} {len(v)}" for k, v in sp.items()))
    return out + [""]


def training_section(runs: Path) -> list:
    rows = []
    for f in sorted(runs.glob("*.json")):
        doc = json.loads(f.read_text())
        if "info" not in doc:  # not a training report
            continue
        info = doc["info"]
        cfg = info["config"]
        rows.append(f"| {f.stem} | {cfg['basis']} | {cfg['project']} | {cfg.get('dp_prior', 'none')} | {cfg['lam_mass']} | "
                    f"{cfg['lam_pressure']} | {info['n_modes']} | {info.get('pod_projection_error_val', float('nan')):.2e} | "
                    f"{info['epochs_run']} | {info.get('training_time_s', 0):.0f} |")
    if not rows:
        return ["## Training", "", "_no trained models yet_", ""]
    return ["## Training", "", "| Variant | Basis | Flux projection | ΔP prior | λ_mass | λ_pressure | K (POD) | POD error (val) | Epochs | Time (s) |",
            "|---|---|---|---|---|---|---|---|---|---|", *rows, ""]


def offline_section(ev: Path) -> list:
    out = ["## Offline evaluation (teacher-forced, held-out cases)", ""]
    any_ = False
    for split in ("val", "test", "test_ood"):
        f = ev / f"offline_{split}.json"
        if not f.exists():
            continue
        any_ = True
        summ = json.loads(f.read_text())["summary"]
        suffixes = ("_profile_rel_l2", "_dp_rel_rmse", "_mass_err_max")
        names = sorted({k[: -len(sfx)] for k in summ for sfx in suffixes if k.endswith(sfx)})
        out += [f"### {split}", "", "| Method | Profile rel. L2 | ΔP RMSE / range | Max mass error |", "|---|---|---|---|"]
        for n in names:
            out.append(f"| {n} | {fmt(summ.get(n + '_profile_rel_l2'))} | {fmt(summ.get(n + '_dp_rel_rmse'))} | "
                       f"{fmt(summ.get(n + '_mass_err_max'), '.1e')} |")
        out.append("")
    return out if any_ else out + ["_not run yet_", ""]


def coupled_section(m3: Path) -> list:
    files = sorted((m3 / "results").glob("*.json")) if (m3 / "results").exists() else []
    if not files:
        return ["## Coupled evaluation (two-way 1D–3D)", "", "_not run yet_", ""]
    res = [json.loads(f.read_text()) for f in files]
    methods = sorted({m for r in res for m in r["methods"]})
    out = ["## Coupled evaluation (two-way 1D–3D)", "",
           f"{len(res)} cases, last of {res[0]['cycles']} cycles, errors relative to the extended-domain reference at Γ.", "",
           "| Method | p_Γ rel. RMSE | Q rel. RMSE | Reflection abs. diff | Max mass error | Iterations | Wall per run (min) | All converged |",
           "|---|---|---|---|---|---|---|---|"]
    for m in methods:
        rows = [r["methods"][m] for r in res if m in r["methods"]]

        def mean(k):
            vals = [x[k] for x in rows if k in x]
            return float(np.mean(vals)) if vals else None

        out.append(f"| {m} | {fmt(mean('p_gamma_rel_rmse'), '.4f')} | {fmt(mean('q_rel_rmse'), '.4f')} | "
                   f"{fmt(mean('reflection_abs_diff'), '.4f')} | {fmt(mean('mass_err_max'), '.1e')} | "
                   f"{fmt(mean('iterations_mean'), '.2f')} | {fmt(mean('wall_time_s') / 60, '.1f')} | "
                   f"{all(x['converged_all'] for x in rows)} |")
    ref_wall = [r["methods"]["reference"]["wall_time_s"] for r in res if "reference" in r["methods"]]
    for m in methods:
        if m.endswith("_B") and ref_wall:
            b = [r["methods"][m]["wall_time_s"] for r in res if m in r["methods"]]
            out.append(f"\n- {m}: online speed-up vs coupled 3D reference ≈ {np.mean(ref_wall) / max(np.mean(b), 1e-9):.0f}×")
    out.append("\nPer case p_Γ error:\n")
    out += [f"- {r['case_id']} (severity {r['severity']:.2f}): " + ", ".join(
        f"{m} {r['methods'][m].get('p_gamma_rel_rmse', float('nan')):.4f}" for m in methods if m != "reference" and m in r["methods"])
        for r in res]
    return out + [""]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--work", type=Path, required=True)
    args = ap.parse_args()
    work = args.work.expanduser().resolve()
    lines = ["# hemocouple results report", "", f"Work directory: `{work}`", ""]
    lines += dataset_section(work / "m2") + training_section(work / "runs") + offline_section(work / "eval")
    lines += coupled_section(work / "m3")
    text = "\n".join(lines)
    (work / "report.md").write_text(text)
    print(text)


if __name__ == "__main__":
    main()
