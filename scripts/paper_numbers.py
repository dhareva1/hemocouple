"""Every number quoted in the manuscript, computed from the pipeline outputs (single source of truth).

    python scripts/paper_numbers.py --work ~/hemocouple-work

Reads <work>/m2 (dataset), <work>/sweep (train_sweep.py, five seeds), <work>/runs/physics_yt_stokes.pt (final model),
<work>/m3/results (coupled evaluation) and, if present, <work>/m2_regen (regenerated dataset, validate_regen.py).
Missing offline evaluations are run with evaluate_offline.py and cached in the sweep / regen directories.
Writes <work>/paper_numbers.md and <work>/paper_numbers.json.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SEEDS = [0, 1, 2, 3, 4]
sys.path.insert(0, str(ROOT / "scripts"))
from train_sweep import REG_STUDY, SHARED, VARIANTS, selection_scores  # noqa: E402

FINAL = ("physics_yt", f"{SHARED}_stokes_nop")
# Rows of the offline table (variant, setting, label), in manuscript order.
OFFLINE_ROWS = [
    ("physics_yt", f"{SHARED}_stokes_nop", "Operator, projection + Young-Tsai + Stokes prior (final)"),
    ("physics_yt", SHARED, "Operator, projection + Young-Tsai prior"),
    ("physics", SHARED, "Operator, projection, no prior"),
    ("soft_mass", SHARED, "Operator, mass penalty, no projection"),
    ("data_only", SHARED, "Operator, data only"),
    ("physics_yt", f"{SHARED}_trunk", "Operator, projection + Young-Tsai, learned trunk"),
    ("physics_yt", f"{SHARED}_stokes", "Ablation: final model with the total-pressure loss"),
    ("data_only", f"{SHARED}_stokes", "Ablation: data only + Stokes prior"),
    ("physics", f"{SHARED}_stokes", "Ablation: projection + Stokes prior, no Young-Tsai prior"),
]
COUPLED = [("baseline_A", "Mode A, Poiseuille inlet"), ("physics_yt_stokes_A", "Mode A, operator inlet"),
           ("physics_yt_stokes_B", "Mode B, operator"), ("young_tsai_cal_B", "Mode B, calibrated Young-Tsai"),
           ("young_tsai_B", "Mode B, Young-Tsai"), ("1d_only_B", "Mode B, 1D only")]


def runs(variant: str, setting: str) -> list:
    return [f"{variant}__{setting}__s{s}" for s in SEEDS]


def evaluate(data: Path, split: str, models: dict, out: Path) -> dict:
    """evaluate_offline.py summary for ``models`` (name -> checkpoint), cached in ``out``."""
    newest = max(Path(p).stat().st_mtime for p in models.values())
    if out.exists() and out.stat().st_mtime > newest:
        cached = json.loads(out.read_text())["summary"]
        if all(f"{m}_dp_rel_rmse" in cached for m in models):
            return cached
    cmd = [sys.executable, str(ROOT / "scripts" / "evaluate_offline.py"), "--data", str(data), "--split", split,
           "--out", str(out)] + [f"--model={m}={p}" for m, p in models.items()]
    subprocess.run(cmd, check=True, capture_output=True)
    return json.loads(out.read_text())["summary"]


def ms(xs, scale=100.0, nd=2) -> dict:
    xs = np.asarray(xs, dtype=float) * scale
    return {"mean": round(float(xs.mean()), 6), "std": round(float(xs.std(ddof=1)), 6), "text": f"{xs.mean():.{nd}f} ± {xs.std(ddof=1):.{nd}f}"}


def dataset_numbers(m2: Path) -> dict:
    meta = json.loads((m2 / "dataset.json").read_text())
    splits = json.loads((m2 / "splits.json").read_text())
    heads = [json.loads(str(np.load(f)["header"]))["meta"] for f in sorted(m2.glob("*.npz"))]
    corr = [h["label_flux_correction_max"] * 100 for h in heads]
    core_h = sum(h["wall_time_s"] * h.get("n_procs", 1) for h in heads) / 3600
    return {
        "n_cases": len(heads), "splits": {k: len(v) for k, v in splits.items()},
        "ood_severity": meta.get("ood_severity"),
        "label_flux_correction_pct": [round(min(corr), 2), round(max(corr), 2)],
        "cells": [min(h["cells"] for h in heads), max(h["cells"] for h in heads)],
        "generation_core_hours": round(core_h, 1),
        "generation_wall_hours": round(sum(h["wall_time_s"] for h in heads) / 3600, 1),
    }


def offline_numbers(work: Path) -> dict:
    sweep, m2 = work / "sweep", work / "m2"
    summary = {sp: json.loads((sweep / f"eval_{sp}.json").read_text())["summary"] for sp in ("val", "test", "test_ood")}
    info = {p.stem: json.loads(p.read_text())["info"] for p in sweep.glob("*__*__s*.json")}
    out = {"rows": [], "baselines": {}}
    for v, setting, label in OFFLINE_ROWS:
        rs = runs(v, setting)
        if not all(r in info for r in rs):
            continue
        row = {"variant": v, "setting": setting, "label": label}
        for sp in ("test", "test_ood"):
            e = summary[sp]
            row[sp] = {"profile_pct": ms([e[f"{r}_profile_rel_l2"] for r in rs]),
                       "dp_pct": ms([e[f"{r}_dp_rel_rmse"] for r in rs]),
                       "mass_err_max": float(max(e[f"{r}_mass_err_max"] for r in rs))}
        out["rows"].append(row)
    for sp in ("test", "test_ood"):
        e = summary[sp]
        out["baselines"][sp] = {"poiseuille_profile_pct": round(e["poiseuille_profile_rel_l2"] * 100, 2),
                                "womersley_profile_pct": round(e["womersley_profile_rel_l2"] * 100, 2),
                                "1d_only_dp_pct": round(e["1d_only_dp_rel_rmse"] * 100, 2)}

    # Shared regularization: validation profile and ΔP errors relative to `base`, mean over variants and seeds.
    shared = selection_scores(summary["val"], REG_STUDY)
    out["regularization_score"] = {r: round(x, 3) for r, x in sorted(shared.items(), key=lambda kv: kv[1])}
    out["regularization_chosen"] = min(shared, key=shared.get) if shared else None
    out["regularization_configured"] = SHARED
    out["final_pressure_loss_score"] = {r: round(x, 3) for r, x in selection_scores(
        summary["val"], [f"{SHARED}_stokes", f"{SHARED}_stokes_nop"], ["physics_yt"]).items()}

    # Learned Young-Tsai prior scale and the train / test / OOD pressure-correction error with and without the prior.
    sys.path.insert(0, str(ROOT / "src"))
    from hemocouple.train import load_checkpoint  # noqa: E402

    s = [float(load_checkpoint(sweep / f"{r}.pt")[0].prior_scale.detach()) for r in runs("physics_yt", SHARED)]
    out["prior_scale"] = {"mean": round(float(np.mean(s)), 3), "std": round(float(np.std(s, ddof=1)), 3)}
    models = {r: sweep / f"{r}.pt" for v in ("physics", "physics_yt") for r in runs(v, SHARED)}
    train = evaluate(m2, "train", models, sweep / "eval_train_paper.json")
    out["dp_train_test_ood_pct"] = {
        v: {sp: ms([src[f"{r}_dp_rel_rmse"] for r in runs(v, SHARED)])["text"]
            for sp, src in (("train", train), ("test", summary["test"]), ("test_ood", summary["test_ood"]))}
        for v in ("physics", "physics_yt")}

    final_runs = runs(*FINAL)
    best = min(final_runs, key=lambda r: info[r]["best_val_data"])
    out["final_model"] = {"run": best, "n_modes": info[best]["n_modes"], "epochs": info[best]["epochs_run"],
                          "training_time_s": round(info[best]["training_time_s"], 1),
                          "pod_floor_val": info[best]["pod_projection_error_val"]}

    # Lumped Young-Tsai stenosis baselines (stenosis.py) on the final model's samples.
    for sp in ("test", "test_ood"):
        e = evaluate(m2, sp, {best: sweep / f"{best}.pt"}, sweep / f"eval_{sp}_lumped.json")
        out["baselines"][sp].update({f"{k}_dp_pct": round(e[f"{k}_dp_rel_rmse"] * 100, 2)
                                     for k in ("young_tsai", "young_tsai_cal")})
    sys.path.insert(0, str(ROOT / "src"))
    from hemocouple.stenosis import YoungTsai  # noqa: E402

    out["young_tsai_cal_scales"] = [round(x, 3) for x in YoungTsai.calibrated_on(m2).scales]
    return out


def coupled_numbers(work: Path, dataset: dict, offline: dict) -> dict:
    res = [json.loads(p.read_text()) for p in sorted((work / "m3" / "results").glob("*.json"))]
    col = lambda m, k: np.array([r["methods"][m][k] for r in res if k in r["methods"][m]])  # noqa: E731
    ref_wall = col("reference", "wall_time_s").mean()
    out = {"n_cases": len(res), "table": {}}
    for m, label in COUPLED:
        out["table"][m] = {
            "label": label,
            "p_gamma_pct": round(col(m, "p_gamma_rel_rmse").mean() * 100, 2),
            "q_pct": round(col(m, "q_rel_rmse").mean() * 100, 2),
            "dp_pct": round(col(m, "dp_rel_rmse").mean() * 100, 2),
            "reflection_abs_diff": round(col(m, "reflection_abs_diff").mean(), 4),
            "mass_err_max": float(col(m, "mass_err_max").max()),
            "cost_vs_reference": float(col(m, "wall_time_s").mean() / ref_wall),
            "wall_min": round(col(m, "wall_time_s").mean() / 60, 2),
        }
    its = [r["methods"][m]["iterations_mean"] for r in res for m in r["methods"]]
    out["iterations_mean_range"] = [round(min(its), 2), round(max(its), 2)]
    out["all_converged"] = all(r["methods"][m]["converged_all"] for r in res for m in r["methods"])

    def paired(a, b, k):
        x, y = col(a, k), col(b, k)
        return {"better": int(np.sum(x < y)), "of": len(x), "median_change_pct": round(float(np.median((x - y) / y)) * 100, 1)}

    out["n_stenosed"] = len(col("baseline_A", "dp_rel_rmse"))
    out["A_vs_baseline"] = {k: paired("physics_yt_stokes_A", "baseline_A", k)
                            for k in ("p_gamma_rel_rmse", "dp_rel_rmse", "reflection_abs_diff", "q_rel_rmse")}
    out["B_vs_1d"] = {k: paired("physics_yt_stokes_B", "1d_only_B", k) for k in ("p_gamma_rel_rmse", "dp_rel_rmse", "q_rel_rmse")}
    out["B_vs_young_tsai_cal"] = {k: paired("physics_yt_stokes_B", "young_tsai_cal_B", k)
                                  for k in ("p_gamma_rel_rmse", "dp_rel_rmse", "q_rel_rmse", "reflection_abs_diff")}
    out["B_vs_A_q_median_change_pct"] = paired("physics_yt_stokes_B", "physics_yt_stokes_A", "q_rel_rmse")["median_change_pct"]
    sev = np.array([r["severity"] for r in res]) >= 0.55
    out["severe_cases"] = {"n": int(sev.sum()),
                           "1d_only_B_pct": round(float(col("1d_only_B", "p_gamma_rel_rmse")[sev].mean()) * 100, 1),
                           "young_tsai_B_pct": round(float(col("young_tsai_B", "p_gamma_rel_rmse")[sev].mean()) * 100, 1),
                           "young_tsai_cal_B_pct": round(float(col("young_tsai_cal_B", "p_gamma_rel_rmse")[sev].mean()) * 100, 2),
                           "operator_B_pct": round(float(col("physics_yt_stokes_B", "p_gamma_rel_rmse")[sev].mean()) * 100, 2),
                           "dp_pct": {m: round(float(col(m, "dp_rel_rmse")[sev[[r["severity"] > 0 for r in res]]].mean()) * 100, 2)
                                      for m, _ in COUPLED}}
    # Break-even number of runs for Mode B against the conventional Mode A coupling (serial runs: wall = core time).
    c_off = dataset["generation_core_hours"] + offline["final_model"]["training_time_s"] / 3600
    c_base = float(col("baseline_A", "wall_time_s").mean()) / 3600
    c_op = float(col("physics_yt_stokes_B", "wall_time_s").mean()) / 3600
    out["break_even"] = {"C_off_core_h": round(c_off, 1), "C_on_baseline_core_h": round(c_base, 1),
                         "C_on_operator_core_h": round(c_op, 4), "N_star": round(c_off / (c_base - c_op), 1),
                         "speedup_vs_reference": int(round(ref_wall / col("physics_yt_stokes_B", "wall_time_s").mean()))}
    return out


def regen_numbers(work: Path) -> dict | None:
    regen = work / "m2_regen"
    if not (regen / "validation.json").exists():
        return None
    v = json.loads((regen / "validation.json").read_text())
    cases = v["cases"]
    out = {"n_cases": len(cases), "worst": v["worst"],
           "median_dp": float(np.median([c["dp"] for c in cases.values()])),
           "median_profiles": float(np.median([c["profiles_rel_l2"] for c in cases.values()])),
           "flux_correction_identical": sum(abs(c["label_flux_correction_max"][0] - c["label_flux_correction_max"][1]) < 1e-6
                                            for c in cases.values())}
    sweep = work / "sweep"
    models = {r: sweep / f"{r}.pt" for v_, s_, _ in OFFLINE_ROWS[:5] for r in runs(v_, s_)}
    change = 0.0
    for sp in ("test", "test_ood"):
        a = evaluate(work / "m2", sp, models, sweep / f"eval_{sp}_paper_rows.json")
        b = evaluate(regen, sp, models, regen / f"eval_{sp}_paper_rows.json")
        for v_, s_, _ in OFFLINE_ROWS[:5]:
            for k in ("profile_rel_l2", "dp_rel_rmse"):
                d = abs(np.mean([b[f"{r}_{k}"] for r in runs(v_, s_)]) - np.mean([a[f"{r}_{k}"] for r in runs(v_, s_)]))
                change = max(change, d * 100)
    out["max_metric_change_pp"] = round(change, 3)
    return out


def to_markdown(n: dict) -> str:
    d, o, c = n["dataset"], n["offline"], n["coupled"]
    lines = ["# Manuscript numbers", "", "Generated by `scripts/paper_numbers.py`; every value comes from the pipeline outputs.", "",
             "## Dataset", "",
             f"- {d['n_cases']} cases; splits {d['splits']}; out-of-distribution: severity > {d['ood_severity']}",
             f"- label flux correction (max per case): {d['label_flux_correction_pct'][0]}–{d['label_flux_correction_pct'][1]}%",
             f"- reference mesh: {d['cells'][0]}–{d['cells'][1]} cells",
             f"- generation cost: {d['generation_core_hours']} core-h ({d['generation_wall_hours']} h wall)", "",
             "## Offline (mean ± s.d. over five seeds, %)", "",
             f"- shared regularization (validation score, lower is better): {o['regularization_score']} → **{o['regularization_chosen']}** (configured {o['regularization_configured']})",
             f"- final model with / without total-pressure loss (validation score): {o['final_pressure_loss_score']}",
             f"- Young–Tsai prior scale s = {o['prior_scale']['mean']} ± {o['prior_scale']['std']}",
             f"- ΔP error train / test / OOD: " + "; ".join(f"{v}: {x['train']} / {x['test']} / {x['test_ood']}"
                                                           for v, x in o["dp_train_test_ood_pct"].items()),
             f"- final model: {o['final_model']}", "",
             "| Method | Profile test | Profile OOD | ΔP test | ΔP OOD | Max flux error test / OOD |", "|---|---|---|---|---|---|"]
    for r in o["rows"]:
        lines.append(f"| {r['label']} | {r['test']['profile_pct']['text']} | {r['test_ood']['profile_pct']['text']} | "
                     f"{r['test']['dp_pct']['text']} | {r['test_ood']['dp_pct']['text']} | "
                     f"{r['test']['mass_err_max']:.1e} / {r['test_ood']['mass_err_max']:.1e} |")
    b = o["baselines"]
    lines += [f"| Poiseuille inlet | {b['test']['poiseuille_profile_pct']} | {b['test_ood']['poiseuille_profile_pct']} | – | – | – |",
              f"| Womersley inlet | {b['test']['womersley_profile_pct']} | {b['test_ood']['womersley_profile_pct']} | – | – | – |",
              f"| 1D only | – | – | {b['test']['1d_only_dp_pct']} | {b['test_ood']['1d_only_dp_pct']} | – |",
              f"| 1D + Young-Tsai stenosis model | – | – | {b['test']['young_tsai_dp_pct']} | {b['test_ood']['young_tsai_dp_pct']} | – |",
              f"| 1D + calibrated Young-Tsai (scales {o['young_tsai_cal_scales']}, fitted on train) | – | – | "
              f"{b['test']['young_tsai_cal_dp_pct']} | {b['test_ood']['young_tsai_cal_dp_pct']} | – |", "",
              f"## Coupled ({c['n_cases']} cases, last of three cycles)", "",
              f"ΔP_Γ = P_Γ − P_lin(Q), error relative to the reference range, mean over the {c['n_stenosed']} stenosed cases.", "",
              "| Method | P_Γ (%) | ΔP_Γ (%) | Q (%) | Reflection abs. diff. | Max flux error | Cost vs reference |",
              "|---|---|---|---|---|---|---|"]
    for m, t in c["table"].items():
        lines.append(f"| {t['label']} | {t['p_gamma_pct']} | {t['dp_pct']} | {t['q_pct']} | {t['reflection_abs_diff']} | "
                     f"{t['mass_err_max']:.0e} | " + (f"{t['cost_vs_reference']:.2f}" if t['cost_vs_reference'] > 0.01
                                                     else f"≈1/{1 / t['cost_vs_reference']:,.0f}") + " |")
    lines += ["", f"- iterations (case means): {c['iterations_mean_range']}; all converged: {c['all_converged']}",
              f"- Mode A operator vs Poiseuille: {c['A_vs_baseline']}",
              f"- Mode B operator vs calibrated Young-Tsai: {c['B_vs_young_tsai_cal']}",
              f"- Mode B operator vs 1D only: {c['B_vs_1d']}; Q error vs Mode A: {c['B_vs_A_q_median_change_pct']:+}% (median)",
              f"- severe cases (severity ≥ 0.55): {c['severe_cases']}",
              f"- break-even: {c['break_even']}"]
    if n.get("regen"):
        g = n["regen"]
        lines += ["", "## Reproduction (regenerated dataset)", "",
                  f"- {g['n_cases']} cases; worst differences {g['worst']}",
                  f"- median ΔP diff {g['median_dp']:.1e}, median profile diff {g['median_profiles']:.1e}; "
                  f"identical flux corrections {g['flux_correction_identical']}/{g['n_cases']}",
                  f"- largest change of any offline metric when re-evaluated on the regenerated labels: {g['max_metric_change_pp']} pp"]
    return "\n".join(lines) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--work", type=Path, required=True)
    args = ap.parse_args()
    work = args.work.expanduser().resolve()
    n = {"dataset": dataset_numbers(work / "m2")}
    n["offline"] = offline_numbers(work)
    n["coupled"] = coupled_numbers(work, n["dataset"], n["offline"])
    n["regen"] = regen_numbers(work)
    (work / "paper_numbers.json").write_text(json.dumps(n, indent=2))
    md = to_markdown(n)
    (work / "paper_numbers.md").write_text(md)
    print(md)


if __name__ == "__main__":
    main()
