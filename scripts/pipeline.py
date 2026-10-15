"""End-to-end pipeline: check → dataset → train → offline → coupled → report. Every stage is resumable.

    python scripts/pipeline.py --work ~/hemocouple-work --stages all
    python scripts/pipeline.py --work ~/hemocouple-work --stages train offline report   # after copying a finished dataset

Layout under --work:
    m2/            reference dataset (*.npz, cases.json, splits.json, dataset.json)
    sweep/         paper training sweep: every (variant, setting, seed) checkpoint + summary.md
    runs/          selected checkpoints (<variant>.pt, best validation seed) used by offline / coupled stages
    eval/          offline evaluation JSON per split
    m3/            coupled evaluation (raw/<case>/<method>.npz, results/<case>.json)
    report.md          generated summary
    paper_numbers.md   every number quoted in the manuscript (scripts/paper_numbers.py)

Parallelism defaults come from the machine (Apple silicon performance cores):
    dataset   workers = P // procs concurrent cases, each with ``procs`` MPI ranks and its own container
    coupled   one serial OpenFOAM run per worker, so up to P concurrent cases
Docker Desktop must be allowed to use all CPUs (Settings → Resources).
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PY = sys.executable
STAGES = ["check", "dataset", "train", "offline", "coupled", "report"]
VARIANTS = ["physics_yt_stokes", "physics_yt", "physics", "data_only", "soft_mass"]


def log(msg: str) -> None:
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}", flush=True)


def perf_cores() -> int:
    for key in ("hw.perflevel0.physicalcpu", "hw.physicalcpu"):
        try:
            return int(subprocess.run(["sysctl", "-n", key], capture_output=True, text=True, check=True).stdout)
        except Exception:
            continue
    return os.cpu_count() or 4


def run(cmd, log_path: Path) -> subprocess.Popen:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    return subprocess.Popen([str(c) for c in cmd], stdout=open(log_path, "a"), stderr=subprocess.STDOUT, cwd=ROOT)


def wait_all(procs, what: str) -> None:
    codes = [p.wait() for p in procs]
    if any(codes):
        raise SystemExit(f"{what}: {sum(c != 0 for c in codes)} worker(s) failed, see logs")


def stage_check(args) -> None:
    import importlib

    for mod in ("numpy", "scipy", "torch"):
        importlib.import_module(mod)
    if subprocess.run(["docker", "info"], capture_output=True).returncode != 0:
        raise SystemExit("Docker is not running: start Docker Desktop (and allow all CPUs in Settings → Resources)")
    image = "opencfd/openfoam-run:2412"
    if not subprocess.run(["docker", "images", "-q", image], capture_output=True, text=True).stdout.strip():
        log(f"pulling {image} (~720 MB)")
        subprocess.run(["docker", "pull", image], check=True)
    ncpu = subprocess.run(["docker", "info", "--format", "{{.NCPU}}"], capture_output=True, text=True).stdout.strip()
    log(f"Docker OK ({ncpu} CPUs visible), image present, Python deps OK; performance cores: {perf_cores()}")
    if subprocess.run([PY, "-m", "pytest", "-q"], cwd=ROOT).returncode != 0:
        raise SystemExit("unit tests failed")


def stage_dataset(args, data: Path) -> None:
    workers = args.dataset_workers or max(1, perf_cores() // args.procs)
    log(f"dataset: {workers} concurrent case(s) x {args.procs} MPI ranks → {data}")
    procs = [run([PY, "scripts/m2_make_dataset.py", "--config", args.dataset_config, "--out", data,
                  "--shard", f"{k}/{workers}", "--procs", args.procs], data / f"worker_{k}.log") for k in range(workers)]
    wait_all(procs, "dataset")
    # Single pass to write splits.json / dataset.json once every case exists.
    subprocess.run([PY, "scripts/m2_make_dataset.py", "--config", args.dataset_config, "--out", data], cwd=ROOT, check=True)
    if not (data / "splits.json").exists():
        raise SystemExit("dataset incomplete: splits.json not written")


# Checkpoint exported to runs/<name>.pt: (variant, setting) in the sweep; the best-validation seed is used.
# wd1e-1 is the shared regularization chosen on validation data (sweep/summary.md); the final model adds the
# causal Stokes velocity prior and drops the total-pressure loss.
EXPORTS = {"physics_yt_stokes": ("physics_yt", "wd1e-1_stokes_nop"), "physics_yt": ("physics_yt", "wd1e-1"),
           "physics": ("physics", "wd1e-1"), "data_only": ("data_only", "wd1e-1"), "soft_mass": ("soft_mass", "wd1e-1")}


def stage_train(args, data: Path, runs: Path, sweep: Path) -> None:
    log(f"train: paper sweep → {sweep}")
    subprocess.run([str(c) for c in [PY, "scripts/train_sweep.py", "--data", data, "--out", sweep, "--workers",
                                     args.train_workers, "--threads", args.train_threads]], cwd=ROOT, check=True)
    exports = [f"{v}:{s}={runs / f'{name}.pt'}" for name, (v, s) in EXPORTS.items() if name in args.variants]
    subprocess.run([PY, "scripts/train_sweep.py", "--data", data, "--out", sweep, "--export", *exports], cwd=ROOT, check=True)


def model_args(args, runs: Path):
    out = []
    for v in args.variants:
        out += ["--model", f"{v}={runs / f'{v}.pt'}"]
    return out


def stage_offline(args, data: Path, runs: Path, ev: Path, sweep: Path) -> None:
    log("offline: sweep summary (all seeds)")
    with open(sweep / "summary.log", "w") as fh:
        subprocess.run([PY, "scripts/train_sweep.py", "--data", data, "--out", sweep, "--summarize"], cwd=ROOT,
                       check=True, stdout=fh)
    splits = json.loads((data / "splits.json").read_text())
    for split in ("val", "test", "test_ood"):
        if not splits.get(split):
            continue
        log(f"offline: {split}")
        with open(ev / f"offline_{split}.txt", "w") as fh:
            subprocess.run([PY, "scripts/evaluate_offline.py", "--data", data, "--split", split, *model_args(args, runs),
                            "--out", ev / f"offline_{split}.json"], cwd=ROOT, check=True, stdout=fh)


def stage_coupled(args, data: Path, runs: Path, m3: Path) -> None:
    splits = json.loads((data / "splits.json").read_text())
    names = [s for s in args.coupled_splits.split(",") if splits.get(s)]
    n_cases = sum(len(splits[s]) for s in names)
    workers = args.coupled_workers or perf_cores()
    variants = [v for v in args.variants if v in args.coupled_variants.split(",")]
    margs = []
    for v in variants:
        margs += ["--model", f"{v}={runs / f'{v}.pt'}"]
    log(f"coupled: {n_cases} cases in {','.join(names)}, {workers} workers, models {variants}")
    # One process per (case, 3D method); finished runs are kept in m3/raw and skipped. Logs: m3/queue.log, m3/logs/.
    wait_all([run([PY, "scripts/m3_coupled_eval.py", "--data", data, "--split", ",".join(names), *margs, "--work", m3,
                   "--cycles", args.coupled_cycles, "--tol", args.coupled_tol, "--jobs", workers],
                  m3 / "queue.log")], "coupled")


def stage_report(args, work: Path) -> None:
    subprocess.run([PY, "scripts/make_report.py", "--work", work], cwd=ROOT, check=True)
    with open(work / "paper_numbers.log", "w") as fh:
        subprocess.run([PY, "scripts/paper_numbers.py", "--work", work], cwd=ROOT, check=True, stdout=fh)
    log(f"report: {work / 'report.md'}, {work / 'paper_numbers.md'}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--work", type=Path, required=True, help="keep outside OneDrive")
    ap.add_argument("--stages", nargs="+", default=["all"], choices=STAGES + ["all"])
    ap.add_argument("--dataset-config", default="configs/dataset_m2.json")
    ap.add_argument("--procs", type=int, default=8, help="MPI ranks per dataset case")
    ap.add_argument("--dataset-workers", type=int, default=0, help="0 = performance cores // procs")
    ap.add_argument("--variants", nargs="+", default=VARIANTS)
    ap.add_argument("--train-workers", type=int, default=6, help="parallel training runs in the sweep")
    ap.add_argument("--train-threads", type=int, default=3, help="torch threads per training run")
    ap.add_argument("--coupled-variants", default="physics_yt_stokes", help="models run in the coupled evaluation")
    ap.add_argument("--coupled-splits", default="test,test_ood")
    ap.add_argument("--coupled-workers", type=int, default=0, help="0 = performance cores")
    ap.add_argument("--coupled-cycles", type=int, default=3)
    ap.add_argument("--coupled-tol", type=float, default=1e-4)
    args = ap.parse_args()

    work = args.work.expanduser().resolve()
    data, runs, ev, m3, sweep = work / "m2", work / "runs", work / "eval", work / "m3", work / "sweep"
    for d in (data, runs, ev, m3, sweep):
        d.mkdir(parents=True, exist_ok=True)
    stages = STAGES if "all" in args.stages else [s for s in STAGES if s in args.stages]
    for st in stages:
        tic = time.time()
        log(f"=== {st} ===")
        if st == "check":
            stage_check(args)
        elif st == "dataset":
            stage_dataset(args, data)
        elif st == "train":
            stage_train(args, data, runs, sweep)
        elif st == "offline":
            stage_offline(args, data, runs, ev, sweep)
        elif st == "coupled":
            stage_coupled(args, data, runs, m3)
        else:
            stage_report(args, work)
        log(f"=== {st} done in {(time.time() - tic) / 60:.1f} min ===")


if __name__ == "__main__":
    main()
