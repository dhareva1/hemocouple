"""Out-of-distribution hold-outs along parameters other than severity (offline).

    python scripts/holdout_study.py --data ~/hemocouple-work/m2 --out ~/hemocouple-work/holdouts [--workers 8]

For each hold-out the 7 most extreme cases of one parameter form the test set, 7 random remaining cases the
validation set and the rest the training set (the severity OOD cases of the main split are used for training):

    short:     shortest stenoses (smallest L_s / R, stenosed cases only)
    reynolds:  highest peak throat Reynolds number (stenosed cases only)
    womersley: highest Womersley number alpha = R sqrt(2 pi / (T nu)) (all cases)

Every hold-out is a dataset directory of symlinks with its own splits.json, trained with train_sweep.py for the
final setting and the two ablations that isolate the priors (five seeds), then evaluated with evaluate_offline.py,
whose lumped Young-Tsai baseline is recalibrated on that hold-out's training cases. Writes <out>/summary.md and
<out>/summary.json.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from hemocouple.stokes import NU_BLOOD  # noqa: E402
from hemocouple.waveforms import pulsatile_flow  # noqa: E402

N_OUT = 7
SEEDS = [0, 1, 2, 3, 4]
SETTINGS = [("physics_yt", "wd1e-1_stokes_nop", "final (projection + Young-Tsai + Stokes)"),
            ("physics_yt", "wd1e-1", "projection + Young-Tsai, no Stokes"),
            ("physics", "wd1e-1", "projection, no prior")]


def case_parameters(data: Path) -> dict:
    out = {}
    for f in sorted(data.glob("*.npz")):
        spec = json.loads(str(np.load(f)["header"]))["spec"]
        g, fl = spec["geometry"], spec["flow"]
        wave = pulsatile_flow(fl["mean_flow"], fl["amplitude"], fl["period"], fl["systole_peak"])
        q_peak = float(wave(np.linspace(0.0, fl["period"], 1000)).max())
        r_t = g["radius"] * np.sqrt(1.0 - g["severity"])
        out[spec["case_id"]] = {"severity": g["severity"], "length_ratio": g["length"] / g["radius"],
                                "re_throat": 2.0 * q_peak / (np.pi * r_t * NU_BLOOD),
                                "womersley": g["radius"] * np.sqrt(2.0 * np.pi / fl["period"] / NU_BLOOD)}
    return out


def make_split(params: dict, key: str, largest: bool, stenosed_only: bool, rng: np.random.Generator) -> dict:
    pool = [c for c, p in params.items() if p["severity"] > 0 or not stenosed_only]
    pool.sort(key=lambda c: params[c][key], reverse=largest)
    test = sorted(pool[:N_OUT])
    rest = sorted(c for c in params if c not in test)
    val = sorted(rng.choice(rest, size=N_OUT, replace=False).tolist())
    return {"train": sorted(c for c in rest if c not in val), "val": val, "test": test}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--threads", type=int, default=2)
    args = ap.parse_args()
    params = case_parameters(args.data)
    rng = np.random.default_rng(2026)
    holdouts = {"short": make_split(params, "length_ratio", False, True, rng),
                "reynolds": make_split(params, "re_throat", True, True, rng),
                "womersley": make_split(params, "womersley", True, False, rng)}
    keys = {"short": "length_ratio", "reynolds": "re_throat", "womersley": "womersley"}

    summary = {}
    for name, split in holdouts.items():
        data = args.out / name / "data"
        data.mkdir(parents=True, exist_ok=True)
        for f in list(args.data.glob("*.npz")) + [args.data / "dataset.json"]:
            link = data / f.name
            if not link.exists():
                link.symlink_to(f.resolve())
        (data / "splits.json").write_text(json.dumps(split, indent=2))
        runs_dir = args.out / name / "runs"
        subprocess.run([sys.executable, str(ROOT / "scripts" / "train_sweep.py"), "--data", str(data), "--out", str(runs_dir),
                        "--workers", str(args.workers), "--threads", str(args.threads), "--variants", "physics_yt", "physics",
                        "--regs", "wd1e-1", "wd1e-1_stokes_nop"], check=True)
        runs = [f"{v}__{r}__s{s}" for v, r, _ in SETTINGS for s in SEEDS]
        ev = args.out / name / "eval_test.json"
        if not ev.exists():
            subprocess.run([sys.executable, str(ROOT / "scripts" / "evaluate_offline.py"), "--data", str(data), "--split", "test",
                            "--out", str(ev)] + [f"--model={r}={runs_dir / r}.pt" for r in runs], check=True, capture_output=True)
        e = json.loads(ev.read_text())["summary"]
        pct = lambda xs: f"{100 * np.mean(xs):.2f} ± {100 * np.std(xs, ddof=1):.2f}"  # noqa: E731
        k = keys[name]
        tr = [params[c][k] for c in split["train"] + split["val"] if params[c]["severity"] > 0 or name == "womersley"]
        te = [params[c][k] for c in split["test"]]
        summary[name] = {
            "parameter": k, "train_range": [round(float(min(tr)), 2), round(float(max(tr)), 2)], "test_range": [round(float(min(te)), 2), round(float(max(te)), 2)],
            "test_cases": split["test"],
            "rows": {label: {"profile_pct": pct([e[f"{v}__{r}__s{s}_profile_rel_l2"] for s in SEEDS]),
                             "dp_pct": pct([e[f"{v}__{r}__s{s}_dp_rel_rmse"] for s in SEEDS])} for v, r, label in SETTINGS},
            "baselines": {"poiseuille_profile_pct": round(100 * e["poiseuille_profile_rel_l2"], 2),
                          "womersley_profile_pct": round(100 * e["womersley_profile_rel_l2"], 2),
                          "1d_only_dp_pct": round(100 * e["1d_only_dp_rel_rmse"], 2),
                          "young_tsai_cal_dp_pct": round(100 * e["young_tsai_cal_dp_rel_rmse"], 2)},
        }

    (args.out / "summary.json").write_text(json.dumps(summary, indent=2))
    lines = ["# Hold-outs along other parameters (offline, test = held-out cases, mean ± s.d. over five seeds, %)", ""]
    for name, s in summary.items():
        lines += [f"## {name}: {s['parameter']} train {s['train_range']} → test {s['test_range']}", "",
                  "| Method | Profile | ΔP |", "|---|---|---|"]
        lines += [f"| {label} | {r['profile_pct']} | {r['dp_pct']} |" for label, r in s["rows"].items()]
        b = s["baselines"]
        lines += [f"| Womersley | {b['womersley_profile_pct']} | – |", f"| Poiseuille | {b['poiseuille_profile_pct']} | – |",
                  f"| Calibrated Young-Tsai | – | {b['young_tsai_cal_dp_pct']} |", f"| 1D only | – | {b['1d_only_dp_pct']} |", ""]
    (args.out / "summary.md").write_text("\n".join(lines))
    print("\n".join(lines))


if __name__ == "__main__":
    main()
