"""Case-level statistics for the headline comparisons (independent physical cases, not time samples or seeds).

    python scripts/case_statistics.py --work ~/hemocouple-work

Each case contributes one error (computed over all of its samples after the warm-up, or over the last coupled
cycle). For every metric: mean with a 95% case-level bootstrap interval, median [IQR] and worst case; for every
paired comparison: cases improved, the median relative change, a 95% bootstrap interval of the mean paired
difference and the exact two-sided sign-test p-value. Reads the outputs used by paper_numbers.py and writes
<work>/case_statistics.md and <work>/case_statistics.json.
"""

from __future__ import annotations

import argparse
import json
from math import comb
from pathlib import Path

import numpy as np

FINAL = "physics_yt__wd1e-1_stokes_nop__s2"
N_BOOT = 20000


def boot_ci(x: np.ndarray, rng: np.random.Generator) -> list:
    idx = rng.integers(0, len(x), size=(N_BOOT, len(x)))
    return [float(v) for v in np.percentile(x[idx].mean(axis=1), [2.5, 97.5])]


def describe(x, rng) -> dict:
    x = np.asarray(x, dtype=float) * 100
    q1, med, q3 = np.percentile(x, [25, 50, 75])
    return {"n": len(x), "mean": float(x.mean()), "ci95": boot_ci(x, rng), "median": float(med), "iqr": [float(q1), float(q3)],
            "worst": float(x.max())}


def sign_test(n_better: int, n: int) -> float:
    k = min(n_better, n - n_better)
    return min(1.0, 2 * sum(comb(n, i) for i in range(k + 1)) / 2**n)


def paired(a, b, rng) -> dict:
    """a against b (lower is better); differences in percentage points."""
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    d = (a - b) * 100
    better = int(np.sum(a < b))
    return {"n": len(a), "better": better, "median_rel_change_pct": float(np.median((a - b) / b) * 100),
            "mean_diff_pp": float(d.mean()), "mean_diff_ci95_pp": boot_ci(d, rng), "sign_test_p": sign_test(better, len(a))}


def young_tsai_identifiability(data: Path, rng: np.random.Generator) -> dict:
    """Correlation of the three lumped terms on the training samples and case-bootstrap intervals of the fitted scales."""
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
    from hemocouple.dataset import load_records
    from hemocouple.geometry import CaseSpec
    from hemocouple.stenosis import backward_dqdt, young_tsai_terms

    meta = json.loads((data / "dataset.json").read_text())
    recs = load_records(data, json.loads((data / "splits.json").read_text())["train"])
    warm = 3 * meta["steps_per_cycle"]
    x, y = [], []
    for rec in recs:
        terms = young_tsai_terms(CaseSpec.from_dict(rec.spec), rec.q, backward_dqdt(rec.q, float(rec.t[1] - rec.t[0])))
        x.append(terms[:, warm:].T)
        y.append(rec.dp[warm:])
    fit = lambda idx: np.linalg.lstsq(np.concatenate([x[i] for i in idx]), np.concatenate([y[i] for i in idx]), rcond=None)[0]  # noqa: E731
    boot = np.array([fit(rng.integers(0, len(recs), len(recs))) for _ in range(2000)])
    return {"scales": fit(range(len(recs))).tolist(), "term_correlation": np.corrcoef(np.concatenate(x).T).tolist(),
            "scales_ci95": np.percentile(boot, [2.5, 97.5], axis=0).T.tolist()}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--work", type=Path, default=Path.home() / "hemocouple-work")
    args = ap.parse_args()
    rng = np.random.default_rng(0)
    out = {"offline": {}, "coupled": {}}

    for sp in ("test", "test_ood"):
        pc = json.loads((args.work / "sweep" / f"eval_{sp}_lumped.json").read_text())["per_case"]
        col = lambda k: [c[k] for c in pc.values()]  # noqa: E731
        sten = [cid for cid, c in pc.items() if c["severity"] > 0]
        scol = lambda k: [pc[cid][k] for cid in sten]  # noqa: E731
        out["offline"][sp] = {
            "profile_operator": describe(col(f"{FINAL}_profile_rel_l2"), rng),
            "profile_womersley": describe(col("womersley_profile_rel_l2"), rng),
            "profile_poiseuille": describe(col("poiseuille_profile_rel_l2"), rng),
            "dp_operator": describe(col(f"{FINAL}_dp_rel_rmse"), rng),
            "dp_young_tsai_cal": describe(col("young_tsai_cal_dp_rel_rmse"), rng),
            "dp_operator_vs_young_tsai_cal_stenosed": paired(scol(f"{FINAL}_dp_rel_rmse"), scol("young_tsai_cal_dp_rel_rmse"), rng),
            "profile_operator_vs_womersley": paired(col(f"{FINAL}_profile_rel_l2"), col("womersley_profile_rel_l2"), rng),
            "per_case": {cid: {"severity": c["severity"], "profile_pct": 100 * c[f"{FINAL}_profile_rel_l2"],
                               "dp_pct": 100 * c[f"{FINAL}_dp_rel_rmse"], "dp_young_tsai_cal_pct": 100 * c["young_tsai_cal_dp_rel_rmse"]}
                         for cid, c in pc.items()},
        }

    res = [json.loads(p.read_text()) for p in sorted((args.work / "m3" / "results").glob("*.json"))]
    sten = [r for r in res if "dp_rel_rmse" in r["methods"]["baseline_A"]]
    get = lambda rs, m, k: [r["methods"][m][k] for r in rs]  # noqa: E731
    c = out["coupled"]
    for m in ("baseline_A", "physics_yt_stokes_A", "physics_yt_stokes_B", "young_tsai_cal_B"):
        c[m] = {"dp": describe(get(sten, m, "dp_rel_rmse"), rng), "p_gamma": describe(get(res, m, "p_gamma_rel_rmse"), rng)}
    c["A_operator_vs_poiseuille_dp"] = paired(get(sten, "physics_yt_stokes_A", "dp_rel_rmse"), get(sten, "baseline_A", "dp_rel_rmse"), rng)
    c["B_operator_vs_young_tsai_cal_dp"] = paired(get(sten, "physics_yt_stokes_B", "dp_rel_rmse"), get(sten, "young_tsai_cal_B", "dp_rel_rmse"), rng)
    c["B_operator_vs_A_poiseuille_dp"] = paired(get(sten, "physics_yt_stokes_B", "dp_rel_rmse"), get(sten, "baseline_A", "dp_rel_rmse"), rng)
    c["A_operator_vs_poiseuille_reflection"] = paired(get(res, "physics_yt_stokes_A", "reflection_abs_diff"),
                                                      get(res, "baseline_A", "reflection_abs_diff"), rng)
    c["per_case_dp_pct"] = {r["case_id"]: {"severity": r["severity"], **{m: 100 * r["methods"][m]["dp_rel_rmse"] for m in
                            ("baseline_A", "physics_yt_stokes_A", "physics_yt_stokes_B", "young_tsai_cal_B")}} for r in sten}

    out["young_tsai_identifiability"] = young_tsai_identifiability(args.work / "m2", rng)

    (args.work / "case_statistics.json").write_text(json.dumps(out, indent=2))
    fmt = lambda d: f"{d['mean']:.2f} [{d['ci95'][0]:.2f}, {d['ci95'][1]:.2f}]; median {d['median']:.2f} (IQR {d['iqr'][0]:.2f}–{d['iqr'][1]:.2f}); worst {d['worst']:.2f}"  # noqa: E731
    fmp = lambda d: (f"{d['better']}/{d['n']} better, median {d['median_rel_change_pct']:+.0f}%, mean diff {d['mean_diff_pp']:+.2f} pp "  # noqa: E731
                     f"[{d['mean_diff_ci95_pp'][0]:+.2f}, {d['mean_diff_ci95_pp'][1]:+.2f}], sign test p = {d['sign_test_p']:.3g}")
    lines = ["# Case-level statistics", "", "Mean [95% case bootstrap CI]; final model (seed 2). Errors in %.", ""]
    for sp, o in out["offline"].items():
        lines += [f"## Offline, {sp}", ""] + [f"- {k}: {fmt(v)}" for k, v in o.items() if "mean" in v]
        lines += [f"- {k}: {fmp(v)}" for k, v in o.items() if "better" in v] + [""]
    lines += ["## Coupled", ""] + [f"- {m} {k}: {fmt(v)}" for m, d in c.items() if isinstance(d, dict) and "dp" in d for k, v in d.items()]
    lines += [f"- {k}: {fmp(v)}" for k, v in c.items() if "better" in v] + [""]
    yt = out["young_tsai_identifiability"]
    lines += ["## Calibrated Young-Tsai scales (viscous, expansion, inertial)", "",
              f"- scales {np.round(yt['scales'], 3).tolist()}, case-bootstrap 95% CI {np.round(yt['scales_ci95'], 3).tolist()}",
              f"- term correlation {np.round(yt['term_correlation'], 3).tolist()}", ""]
    lines += ["| Case | Severity | A Poiseuille | A operator | B operator | B calibrated YT |", "|---|---|---|---|---|---|"]
    lines += [f"| {cid} | {v['severity']:.2f} | {v['baseline_A']:.2f} | {v['physics_yt_stokes_A']:.2f} | {v['physics_yt_stokes_B']:.2f} | "
              f"{v['young_tsai_cal_B']:.2f} |" for cid, v in sorted(c["per_case_dp_pct"].items(), key=lambda kv: kv[1]["severity"])]
    (args.work / "case_statistics.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
