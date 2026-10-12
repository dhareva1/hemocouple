"""Multi-seed regularization sweep for the interface operator (proposal 7.6).

    python scripts/train_sweep.py --data DATA --out SWEEP_DIR [--workers 3 --threads 2]
    python scripts/train_sweep.py --data DATA --out SWEEP_DIR --summarize

Trains every (variant, regularization, seed) combination from the base configs in configs/train_<variant>.json.
Runs that already have a checkpoint are skipped, so the sweep resumes. --summarize evaluates all checkpoints on
val, test and test_ood (evaluate_offline.py) and writes summary.md. The regularization setting is chosen per variant
from the mean validation data loss over seeds (training's own selection criterion); test numbers are only reported
for that choice.
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
    "do0.1": {"dropout": 0.1},
    "h64": {"hidden": 64},
    "do0.1_stokes": {"dropout": 0.1, "velocity_prior": "stokes"},
    "do0.1_stokes_nop": {"dropout": 0.1, "velocity_prior": "stokes", "lam_pressure": 0.0},
}
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
    jobs = [(args.data, args.out, v, r, s, args.threads) for v in VARIANTS for r in args.regs for s in SEEDS
            if not (args.out / f"{run_name(v, r, s)}.pt").exists()]
    total = len(VARIANTS) * len(args.regs) * len(SEEDS)
    print(f"{total - len(jobs)}/{total} runs already done; training {len(jobs)} with {args.workers} workers", flush=True)
    with ProcessPoolExecutor(args.workers) as pool:
        for k, msg in enumerate(pool.map(train_one, jobs), 1):
            print(f"[{time.strftime('%H:%M:%S')}] ({k}/{len(jobs)}) {msg}", flush=True)


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
    lines += ["", "Chosen: " + ", ".join(f"{v} → {r}" for v, r in chosen.items()), ""]
    for split in ("test", "test_ood"):
        e = evals[split]
        lines += [f"## {split} (chosen setting; base, dropout and Stokes-prior runs for comparison)", "",
                  "| Variant | Reg | profile L2 | dP RMSE / range | mass err |", "|---|---|---|---|---|"]
        for v in VARIANTS:
            for reg in dict.fromkeys([chosen[v], "base", "do0.1", "do0.1_stokes", "do0.1_stokes_nop"]):
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


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--threads", type=int, default=2, help="torch threads per worker (keep low while OpenFOAM runs)")
    ap.add_argument("--regs", nargs="+", default=list(REGS), choices=list(REGS), help="settings to train")
    ap.add_argument("--summarize", action="store_true")
    args = ap.parse_args()
    summarize(args) if args.summarize else train_all(args)


if __name__ == "__main__":
    main()
