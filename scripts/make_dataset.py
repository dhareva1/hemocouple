"""Generate a case-split dataset of reference interface data.

Currently backed by the analytic mock (hemocouple.mock); swap the reference
run for real solver adapters once milestone M1 fixes the solvers.

    python scripts/make_dataset.py --config configs/dataset_mock.json --out data/mock
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
# numpy 2.0.x + macOS Accelerate emits spurious FP warnings from matmul.
warnings.filterwarnings("ignore", message=".*encountered in matmul", category=RuntimeWarning)

from hemocouple.cross_section import CrossSectionGrid  # noqa: E402
from hemocouple.dataset import CaseRecord, split_cases  # noqa: E402
from hemocouple.geometry import sample_cases  # noqa: E402
from hemocouple.mock import MockCase  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    cfg = json.loads(args.config.read_text())
    args.out.mkdir(parents=True, exist_ok=True)
    grid = CrossSectionGrid.gauss_polar(cfg["n_r"], cfg["n_theta"])
    specs = sample_cases(cfg["n_stenosis"], cfg["n_control"], seed=cfg["seed"])

    start = time.perf_counter()
    records = []
    for spec in specs:
        case = MockCase(spec, grid, n_harmonics=cfg["n_harmonics"])
        result = case.run(
            case.oracle(), "A", cfg["steps_per_cycle"], cfg["n_cycles"], window=1,
            tol=cfg["tol"], record_profiles=True,
        )
        if not result.converged.all():
            print(f"warning: {spec.case_id} has {(~result.converged).sum()} non-converged steps")
        rec = CaseRecord.from_result(spec, result, steps_per_cycle=cfg["steps_per_cycle"], source="mock")
        rec.save(args.out)
        records.append(rec)
    elapsed = time.perf_counter() - start

    splits = split_cases(records, cfg["val_frac"], cfg["test_frac"], cfg.get("ood_severity"), cfg["seed"])
    (args.out / "splits.json").write_text(json.dumps(splits, indent=2))
    (args.out / "dataset.json").write_text(json.dumps({**cfg, "generation_time_s": elapsed}, indent=2))
    print(f"{len(records)} cases -> {args.out} in {elapsed:.1f}s; " + ", ".join(f"{k}={len(v)}" for k, v in splits.items()))


if __name__ == "__main__":
    main()
