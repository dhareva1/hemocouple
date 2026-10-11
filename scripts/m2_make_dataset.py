"""Generate the OpenFOAM reference dataset (docs/m2_design.md). Resumable: existing cases are skipped.

    python scripts/m2_make_dataset.py --config configs/dataset_m2.json --out /path/outside/onedrive/data_m2

After all cases exist, writes splits.json (case-level, OOD-severity hold-out) and dataset.json.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
warnings.filterwarnings("ignore", message=".*encountered in matmul", category=RuntimeWarning)

from hemocouple import foam  # noqa: E402
from hemocouple.cross_section import CrossSectionGrid  # noqa: E402
from hemocouple.dataset import CaseRecord, load_records, split_cases  # noqa: E402
from hemocouple.geometry import sample_cases  # noqa: E402
from hemocouple.reference import ReferenceConfig, run_reference_case  # noqa: E402


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--only", nargs="*", help="run only these case ids")
    ap.add_argument("--keep-cases", action="store_true", help="keep OpenFOAM case directories (large)")
    ap.add_argument("--shard", default="0/1", help="k/N: run only cases with index %% N == k (one container per shard)")
    ap.add_argument("--procs", type=int, help="MPI ranks per case (overrides the config)")
    args = ap.parse_args()
    shard_k, shard_n = (int(x) for x in args.shard.split("/"))

    cfg = json.loads(args.config.read_text())
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=True)
    work = out / "_cases"
    grid = CrossSectionGrid.gauss_polar(cfg["n_r"], cfg["n_theta"])
    bounds = {k: tuple(v) for k, v in cfg["bounds"].items()}
    ref_kwargs = dict(cfg.get("reference", {}))
    if args.procs:
        ref_kwargs["n_procs"] = args.procs
    ref = ReferenceConfig(**ref_kwargs)
    specs = sample_cases(cfg["n_stenosis"], cfg["n_control"], bounds=bounds, seed=cfg["seed"],
                         max_throat_re=cfg.get("max_throat_re"))
    (out / "cases.json").write_text(json.dumps([s.to_dict() for s in specs], indent=2))
    if args.only:
        specs = [s for s in specs if s.case_id in set(args.only)]
    specs = [s for i, s in enumerate(specs) if i % shard_n == shard_k]
    todo = [s for s in specs if not (out / f"{s.case_id}.npz").exists()]

    container = cfg.get("container", "hemocouple-m2data") + (f"-{shard_k}" if shard_n > 1 else "")
    runner = foam.DockerFoam(out, name=container).start() if todo else None
    try:
        for i, spec in enumerate(specs, 1):
            target = out / f"{spec.case_id}.npz"
            if target.exists():
                continue
            g = spec.geometry
            log(f"[{i}/{len(specs)}] {spec.case_id}: R={g.radius * 1e3:.2f} mm, severity={g.severity:.2f}, "
                f"Ls/R={g.length / g.radius:.1f}, Qmean={spec.flow.mean_flow * 1e6:.1f} ml/s, T={spec.flow.period:.2f} s")
            rec = run_reference_case(runner, work, spec, grid, ref)
            rec.save(out)
            m = rec.meta
            log(f"    done in {m['wall_time_s']:.0f} s, {m['cells']} cells; label flux correction "
                f"max {m['label_flux_correction_max']:.2%}; dp range {rec.dp.min():.0f}..{rec.dp.max():.0f} Pa")
            if not args.keep_cases:
                shutil.rmtree(work / spec.case_id, ignore_errors=True)
    finally:
        if runner is not None:
            runner.stop()

    all_ids = [s["case_id"] for s in json.loads((out / "cases.json").read_text())]
    if all((out / f"{cid}.npz").exists() for cid in all_ids):
        records = load_records(out, all_ids)
        splits = split_cases(records, cfg["val_frac"], cfg["test_frac"], cfg.get("ood_severity"), cfg["seed"])
        if not (out / "splits.json").exists():
            (out / "splits.json").write_text(json.dumps(splits, indent=2))
        (out / "dataset.json").write_text(json.dumps({**cfg, "steps_per_cycle": ref.steps_per_cycle,
                                                     "n_cycles": ref.n_cycles}, indent=2))
        log("all cases present; wrote splits.json and dataset.json: "
            + ", ".join(f"{k}={len(v)}" for k, v in splits.items()))


if __name__ == "__main__":
    main()
