"""Multi-seed training sweep for the interface operator (proposal 7.6): the paper's experiment matrix.

    python scripts/train_sweep.py --data DATA --out SWEEP_DIR [--workers 3 --threads 2]
    python scripts/train_sweep.py --data DATA --out SWEEP_DIR --summarize
    python scripts/train_sweep.py --data DATA --out SWEEP_DIR --export physics_yt:do0.1_stokes_nop=runs/physics_yt_stokes.pt

Trains every (variant, setting, seed) in PLAN, starting from configs/train_<variant>.json; --variants / --regs
restrict it. Runs that already have a checkpoint are skipped, so the sweep resumes. --summarize evaluates all
checkpoints on val, test and test_ood (evaluate_offline.py) and writes summary.md.

Selection uses validation cases only, with the two metrics the paper reports: the profile relative L2 error and the
ΔP RMSE / label range. A setting's score is the mean of both, each divided by the same variant's `base` value,
averaged over variants and seeds (the raw validation data loss is dominated by ΔP and would ignore the profile head).
The shared regularization is the lowest-scoring setting of REG_STUDY. --export copies the seed with the lowest
validation data loss of a (variant, setting) to DEST.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
import warnings
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
warnings.filterwarnings("ignore", message=".*encountered in matmul", category=RuntimeWarning)

VARIANTS = ["physics_yt", "physics", "data_only", "soft_mass"]
REGS = {
    "base": {},
    "wd1e-2": {"weight_decay": 1e-2},
    "wd1e-1": {"weight_decay": 1e-1},
    "wd3e-1": {"weight_decay": 3e-1},
    "wd1": {"weight_decay": 1.0},
    "do0.1": {"dropout": 0.1},
    "h64": {"hidden": 64},
    "do0.1_stokes": {"dropout": 0.1, "velocity_prior": "stokes"},
    "do0.1_stokes_nop": {"dropout": 0.1, "velocity_prior": "stokes", "lam_pressure": 0.0},
    "do0.1_trunk": {"dropout": 0.1, "basis": "trunk"},
    "wd1e-1_stokes": {"weight_decay": 1e-1, "velocity_prior": "stokes"},
    "wd1e-1_stokes_nop": {"weight_decay": 1e-1, "velocity_prior": "stokes", "lam_pressure": 0.0},
    "wd1e-1_trunk": {"weight_decay": 1e-1, "basis": "trunk"},
}
# Paper experiment matrix: the regularization study for every variant, then with the chosen shared regularization
# (SHARED): the causal Stokes prior for every variant, and for the Young-Tsai variant the Stokes prior without the
# total-pressure loss (the final model) and a learned trunk. The do0.1_* settings are kept for older runs only.
REG_STUDY = ["base", "wd1e-2", "wd1e-1", "wd3e-1", "wd1", "do0.1", "h64"]
SHARED = "wd1e-1"
PLAN = ([(v, r) for v in VARIANTS for r in REG_STUDY + [f"{SHARED}_stokes"]]
        + [("physics_yt", f"{SHARED}_stokes_nop"), ("physics_yt", f"{SHARED}_trunk")])
SEEDS = [0, 1, 2, 3, 4]
SPLITS = ["val", "test", "test_ood"]


def run_name(variant: str, reg: str, seed: int) -> str:
    return f"{variant}__{reg}__s{seed}"


def train_one(job: tuple) -> str:
    data, out, variant, reg, seed, threads = job
    import torch

    torch.set_num_threads(threads)
    from hemocouple.cross_section import CrossSectionGrid
    from hemocouple.dataset import load_records
    from hemocouple.train import TrainConfig, fit_operator, save_checkpoint

    name = run_name(variant, reg, seed)
    path = out / f"{name}.pt"
    meta = json.loads((data / "dataset.json").read_text())
    splits = json.loads((data / "splits.json").read_text())
    grid = CrossSectionGrid.gauss_polar(meta["n_r"], meta["n_theta"])
    base = json.loads((ROOT / "configs" / f"train_{variant}.json").read_text())
    cfg = TrainConfig(**{**base, **REGS[reg], "seed": seed})
    start = time.perf_counter()
    model, report = fit_operator(load_records(data, splits["train"]), load_records(data, splits["val"]), grid, cfg)
    report["info"]["training_time_s"] = time.perf_counter() - start
    save_checkpoint(path, model, report)
    path.with_suffix(".json").write_text(json.dumps(report, indent=2))
    i = report["info"]
    return f"{name}: {i['epochs_run']} epochs, val velocity {i['best_val_velocity']:.3e}, val dP {i['best_val_dp']:.3e}"


def train_all(args: argparse.Namespace) -> None:
    args.out.mkdir(parents=True, exist_ok=True)
    plan = [(v, r) for v, r in PLAN if v in args.variants and r in args.regs]
    jobs = [(args.data, args.out, v, r, s, args.threads) for v, r in plan for s in SEEDS
            if not (args.out / f"{run_name(v, r, s)}.pt").exists()]
    total = len(plan) * len(SEEDS)
    print(f"{total - len(jobs)}/{total} runs already done; training {len(jobs)} with {args.workers} workers", flush=True)
    with ProcessPoolExecutor(args.workers) as pool:
        for k, msg in enumerate(pool.map(train_one, jobs), 1):
            print(f"[{time.strftime('%H:%M:%S')}] ({k}/{len(jobs)}) {msg}", flush=True)


def selection_scores(val: dict, settings: list, variants: list = VARIANTS) -> dict:
    """Mean over variants and seeds of (profile / base + ΔP / base) / 2 on validation cases; lower is better."""
    m = lambda v, r, k: np.mean([val[f"{run_name(v, r, s)}_{k}"] for s in SEEDS])  # noqa: E731
    out = {}
    for r in settings:
        try:
            out[r] = float(np.mean([(m(v, r, "profile_rel_l2") / m(v, "base", "profile_rel_l2")
                                     + m(v, r, "dp_rel_rmse") / m(v, "base", "dp_rel_rmse")) / 2 for v in variants]))
        except KeyError:
            continue
    return out


def mean_std(xs: list) -> str:
    return f"{np.mean(xs):.4f} ± {np.std(xs, ddof=1):.4f}" if len(xs) > 1 else f"{xs[0]:.4f}"


def summarize(args: argparse.Namespace) -> None:
    runs = sorted(p.stem for p in args.out.glob("*__*__s*.pt"))
    evals = {}
    for split in SPLITS:
        path = args.out / f"eval_{split}.json"
        if not path.exists() or path.stat().st_mtime < max((args.out / f"{r}.pt").stat().st_mtime for r in runs):
            cmd = [sys.executable, str(ROOT / "scripts" / "evaluate_offline.py"), "--data", str(args.data),
                   "--split", split, "--out", str(path)] + [f"--model={r}={args.out / r}.pt" for r in runs]
            subprocess.run(cmd, check=True, capture_output=True)
        evals[split] = json.loads(path.read_text())["summary"]

    info = {r: json.loads((args.out / f"{r}.json").read_text())["info"] for r in runs}
    lines = ["# Training sweep: variants × regularization × seeds", "",
             f"{len(runs)} runs, seeds {SEEDS}. Mean ± std over seeds. Selection: lowest mean validation data loss.", ""]
    shared = selection_scores(evals["val"], REG_STUDY)
    chosen = {}
    lines += ["## Validation (selection)", "",
              "| Variant | Reg | val data loss | val profile L2 | val dP RMSE | epochs |", "|---|---|---|---|---|---|"]
    for v in VARIANTS:
        scores = {}
        for reg in REGS:
            rs = [run_name(v, reg, s) for s in SEEDS if run_name(v, reg, s) in info]
            if not rs:
                continue
            scores[reg] = np.mean([info[r]["best_val_data"] for r in rs])
            lines.append(f"| {v} | {reg} | {mean_std([info[r]['best_val_data'] for r in rs])} | "
                         f"{mean_std([evals['val'][f'{r}_profile_rel_l2'] for r in rs])} | "
                         f"{mean_std([evals['val'][f'{r}_dp_rel_rmse'] for r in rs])} | "
                         f"{np.mean([info[r]['epochs_run'] for r in rs]):.0f} |")
        chosen[v] = min(scores, key=scores.get)
    lines += ["", "Lowest validation data loss per variant: " + ", ".join(f"{v} → {r}" for v, r in chosen.items()), ""]
    if shared:
        lines += ["Shared regularization (validation profile and ΔP error relative to base, mean over variants): "
                  + ", ".join(f"{r} {x:.3f}" for r, x in sorted(shared.items(), key=lambda kv: kv[1]))
                  + f" → **{min(shared, key=shared.get)}** (configured: {SHARED})", ""]
        final = selection_scores(evals["val"], [f"{SHARED}_stokes", f"{SHARED}_stokes_nop"], ["physics_yt"])
        if len(final) == 2:
            lines += ["Final model, Young–Tsai + Stokes prior with / without the total-pressure loss: "
                      + ", ".join(f"{r} {x:.3f}" for r, x in final.items()), ""]
    for split in ("test", "test_ood"):
        e = evals[split]
        lines += [f"## {split} (base, shared regularization, Stokes prior and ablations)", "",
                  "| Variant | Reg | profile L2 | dP RMSE / range | mass err |", "|---|---|---|---|---|"]
        for v in VARIANTS:
            for reg in dict.fromkeys(["base", SHARED, f"{SHARED}_stokes", f"{SHARED}_stokes_nop", f"{SHARED}_trunk"]):
                rs = [run_name(v, reg, s) for s in SEEDS if run_name(v, reg, s) in info]
                if not rs:
                    continue
                lines.append(f"| {v} | {reg} | {mean_std([e[f'{r}_profile_rel_l2'] for r in rs])} | "
                             f"{mean_std([e[f'{r}_dp_rel_rmse'] for r in rs])} | "
                             f"{np.mean([e[f'{r}_mass_err_max'] for r in rs]):.1e} |")
        lines += [f"| womersley (baseline) | – | {e['womersley_profile_rel_l2']:.4f} | – | – |",
                  f"| poiseuille (baseline) | – | {e['poiseuille_profile_rel_l2']:.4f} | – | – |",
                  f"| 1d_only (baseline) | – | – | {e['1d_only_dp_rel_rmse']:.4f} | – |", ""]
    (args.out / "summary.md").write_text("\n".join(lines))
    print("\n".join(lines))


def export(args: argparse.Namespace) -> None:
    import shutil

    for spec in args.export:
        key, dest = spec.split("=", 1)
        variant, reg = key.split(":")
        infos = {s: json.loads((args.out / f"{run_name(variant, reg, s)}.json").read_text())["info"] for s in SEEDS}
        best = min(infos, key=lambda s: infos[s]["best_val_data"])
        src, dest = args.out / run_name(variant, reg, best), Path(dest)
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(f"{src}.pt", dest)  # run names contain dots (do0.1), so no with_suffix on src
        shutil.copyfile(f"{src}.json", dest.with_suffix(".json"))
        print(f"{variant}:{reg} → seed {best} (val data loss {infos[best]['best_val_data']:.4f}) → {dest}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--threads", type=int, default=2, help="torch threads per worker (keep low while OpenFOAM runs)")
    ap.add_argument("--variants", nargs="+", default=VARIANTS, choices=VARIANTS, help="variants to train")
    ap.add_argument("--regs", nargs="+", default=list(REGS), choices=list(REGS), help="settings to train")
    ap.add_argument("--summarize", action="store_true")
    ap.add_argument("--export", nargs="+", metavar="VARIANT:SETTING=DEST",
                    help="copy the best-validation seed of each (variant, setting) to DEST")
    args = ap.parse_args()
    if args.export:
        export(args)
    elif args.summarize:
        summarize(args)
    else:
        train_all(args)


if __name__ == "__main__":
    main()
