"""Train an interface operator on the train split; select on the val split.

    python scripts/train_operator.py --data data/mock --config configs/train_physics.json --out runs/physics.pt
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
from hemocouple.dataset import load_records  # noqa: E402
from hemocouple.train import TrainConfig, fit_operator, save_checkpoint  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--config", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args()

    meta = json.loads((args.data / "dataset.json").read_text())
    splits = json.loads((args.data / "splits.json").read_text())
    grid = CrossSectionGrid.gauss_polar(meta["n_r"], meta["n_theta"])
    cfg = TrainConfig.from_json(args.config)

    start = time.perf_counter()
    model, report = fit_operator(
        load_records(args.data, splits["train"]), load_records(args.data, splits["val"]), grid, cfg, verbose=not args.quiet
    )
    report["info"]["training_time_s"] = time.perf_counter() - start
    report["info"]["data_generation_time_s"] = meta.get("generation_time_s")
    save_checkpoint(args.out, model, report)
    args.out.with_suffix(".json").write_text(json.dumps(report, indent=2))
    info = report["info"]
    print(
        f"saved {args.out}: K={info['n_modes']} (POD energy {info['pod_energy']:.6f}, "
        f"val projection error {info['pod_projection_error_val']:.2e}), best val data loss {info['best_val_data']:.3e}"
    )


if __name__ == "__main__":
    main()
