"""Run every method inside the coupling loop on a locked split and compare to the reference.

    python scripts/evaluate_coupling.py --data data/mock --split test \
        --model physics=runs/physics.pt --model data_only=runs/data_only.pt --out runs/eval_test.json

Methods: reference (oracle profile, Mode A, tight tolerance), baseline_A
(Poiseuille inlet, Mode A), 1d_only_B (no correction, Mode B), and each
checkpoint in Mode A and Mode B. Metrics are taken over the last cardiac cycle.
"""

from __future__ import annotations

import argparse
import json
import sys
import warnings
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
# numpy 2.0.x + macOS Accelerate emits spurious FP warnings from matmul.
warnings.filterwarnings("ignore", message=".*encountered in matmul", category=RuntimeWarning)

from hemocouple.cross_section import CrossSectionGrid  # noqa: E402
from hemocouple.geometry import CaseSpec  # noqa: E402
from hemocouple.metrics import break_even_runs, characteristic_impedance, compare_runs  # noqa: E402
from hemocouple.mock import MockCase  # noqa: E402
from hemocouple.providers import OperatorProvider, PoiseuilleProfile  # noqa: E402
from hemocouple.train import load_checkpoint  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--split", default="test")
    ap.add_argument("--model", action="append", default=[], help="name=path/to/checkpoint.pt")
    ap.add_argument("--tol", type=float, default=1e-8)
    ap.add_argument("--wave-speed", type=float, default=6.0, help="for the reflection metric, m/s")
    ap.add_argument("--out", type=Path)
    args = ap.parse_args()

    meta = json.loads((args.data / "dataset.json").read_text())
    case_ids = json.loads((args.data / "splits.json").read_text())[args.split]
    grid = CrossSectionGrid.gauss_polar(meta["n_r"], meta["n_theta"])
    steps, cycles = meta["steps_per_cycle"], meta["n_cycles"]
    models = {}
    for item in args.model:
        name, path = item.split("=", 1)
        models[name] = load_checkpoint(Path(path))

    per_case = {}
    for cid in case_ids:
        header = json.loads(str(np.load(args.data / f"{cid}.npz")["header"]))
        spec = CaseSpec.from_dict(header["spec"])
        case = MockCase(spec, grid, n_harmonics=meta["n_harmonics"])
        zc = characteristic_impedance(case.fluid.density, args.wave_speed, spec.geometry.area)

        ref = case.run(case.oracle(), "A", steps, cycles, window=1, tol=1e-10)
        poiseuille = PoiseuilleProfile(grid, spec.geometry.radius)
        runs = {
            "baseline_A": case.run(poiseuille, "A", steps, cycles, window=1, tol=args.tol),
            "1d_only_B": case.run(poiseuille, "B", steps, cycles, window=1, tol=args.tol),
        }
        for name, (model, _) in models.items():
            provider = OperatorProvider(model, spec.features(), spec.geometry.radius)
            window = model.init_kwargs["window"]
            for mode in ("A", "B"):
                runs[f"{name}_{mode}"] = case.run(provider, mode, steps, cycles, window=window, tol=args.tol)
        per_case[cid] = {k: compare_runs(r, ref, steps, zc) for k, r in runs.items()}

    methods = next(iter(per_case.values())).keys() if per_case else []
    summary = {m: {k: float(np.mean([per_case[c][m][k] for c in per_case])) for k in per_case[case_ids[0]][m]} for m in methods}

    cols = ["p_interface_rel_rmse", "q_rel_rmse", "mass_err_max", "reflection_abs_diff", "iterations_mean", "wall_time_s"]
    print(f"split={args.split}  cases={len(per_case)}  (means over cases, last cycle)")
    print(f"{'method':<18}" + "".join(f"{c:>22}" for c in cols))
    for m, row in summary.items():
        print(f"{m:<18}" + "".join(f"{row[c]:>22.4e}" for c in cols))

    for name, (_, info) in models.items():
        offline = (info.get("training_time_s") or 0.0) + (info.get("data_generation_time_s") or 0.0)
        if f"{name}_B" in summary:
            n_star = break_even_runs(offline, summary["baseline_A"]["wall_time_s"], summary[f"{name}_B"]["wall_time_s"])
            print(f"{name}: offline {offline:.1f}s, break-even runs N* (Mode B vs baseline A) = {n_star:.1f}")

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps({"summary": summary, "per_case": per_case}, indent=2))


if __name__ == "__main__":
    main()
