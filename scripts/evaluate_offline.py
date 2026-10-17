"""Offline (teacher-forced) evaluation of interface operators on a locked split of a reference dataset.

    python scripts/evaluate_offline.py --data DATA --split test --model physics=runs/physics.pt --model data_only=runs/data_only.pt

For every sample after warm-up, the operator gets the reference (Q, P_up) history window and is compared with:
    profile: Poiseuille (quasi-steady, conventional) and Womersley (analytic, uses the whole periodic
             waveform in advance, so it is an optimistic baseline) inlet profiles
    dP:      1D-only (dP = 0), lumped Young-Tsai stenosis model with literature coefficients (young_tsai) and
             with per-term scales fitted on the training split (young_tsai_cal, stenosis.py)
Metrics are relative to each case's label scale and are averaged over cases.
"""

from __future__ import annotations

import argparse
import json
import sys
import warnings
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
warnings.filterwarnings("ignore", message=".*encountered in matmul", category=RuntimeWarning)

from hemocouple.cross_section import CrossSectionGrid  # noqa: E402
from hemocouple.dataset import build_samples, load_records  # noqa: E402
from hemocouple.geometry import CaseSpec  # noqa: E402
from hemocouple.stenosis import YoungTsai, backward_dqdt  # noqa: E402
from hemocouple.train import load_checkpoint  # noqa: E402
from hemocouple.womersley import WomersleyProfile  # noqa: E402


def rel_l2(pred: np.ndarray, true: np.ndarray, w: np.ndarray) -> float:
    """Weighted relative L2 over all samples of a case."""
    return float(np.sqrt(np.sum((pred - true) ** 2 * w) / np.sum(true**2 * w)))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--split", default="test")
    ap.add_argument("--model", action="append", default=[], help="name=checkpoint.pt")
    ap.add_argument("--warmup-cycles", type=int, default=None)
    ap.add_argument("--out", type=Path)
    args = ap.parse_args()

    meta = json.loads((args.data / "dataset.json").read_text())
    ids = json.loads((args.data / "splits.json").read_text())[args.split]
    grid = CrossSectionGrid.gauss_polar(meta["n_r"], meta["n_theta"])
    w = grid.weights
    models = {k: load_checkpoint(Path(v)) for k, v in (m.split("=", 1) for m in args.model)}
    steps = meta["steps_per_cycle"]
    lumped = {"young_tsai": YoungTsai(), "young_tsai_cal": YoungTsai.calibrated_on(args.data)}

    per_case = {}
    for rec in load_records(args.data, ids):
        spec = CaseSpec.from_dict(rec.spec)
        r = rec.radius
        row = {"severity": spec.geometry.severity, "label_flux_correction_max": rec.meta.get("label_flux_correction_max")}
        for name, (model, info) in models.items():
            window = model.history_length
            warm = (args.warmup_cycles if args.warmup_cycles is not None else info["config"]["warmup_cycles"]) * steps
            s = build_samples([rec], window, warm)
            t = torch.as_tensor
            with torch.no_grad():
                out = model(t(s["hist"]), t(s["g"]), t(s["q"]), t(s["radius"]))
            u, dp = out["u"].numpy(), out["dp"].numpy()
            dp_scale = float(np.ptp(s["dp"])) or 1.0
            row[f"{name}_profile_rel_l2"] = rel_l2(u, s["u"], w)
            row[f"{name}_dp_rel_rmse"] = float(np.sqrt(np.mean((dp - s["dp"]) ** 2)) / dp_scale)
            row[f"{name}_mass_err_max"] = float(np.max(np.abs(grid.integrate(u, r) - s["q"])) / np.mean(np.abs(s["q"])))
            # Baselines on exactly the same samples.
            idx = np.arange(max(window - 1, warm), len(rec.t))  # same samples as build_samples
            pois = s["q"][:, None] * (2.0 * (1.0 - grid.rho**2) / (np.pi * r**2))[None, :]
            wom = WomersleyProfile(spec.flow.waveform(), r, 3.5e-3 / 1060.0, grid.rho)(rec.t[idx])
            row["poiseuille_profile_rel_l2"] = rel_l2(pois, s["u"], w)
            row["womersley_profile_rel_l2"] = rel_l2(wom, s["u"], w)
            row["1d_only_dp_rel_rmse"] = float(np.sqrt(np.mean(s["dp"] ** 2)) / dp_scale)
            dqdt = backward_dqdt(rec.q, float(rec.t[1] - rec.t[0]))[idx]
            for k, yt in lumped.items():
                row[f"{k}_dp_rel_rmse"] = float(np.sqrt(np.mean((yt.dp(spec, s["q"], dqdt) - s["dp"]) ** 2)) / dp_scale)
            row["dp_label_range_pa"] = [float(s["dp"].min()), float(s["dp"].max())]
        per_case[rec.case_id] = row

    keys = [k for k in next(iter(per_case.values())) if k.endswith(("rel_l2", "rel_rmse", "mass_err_max"))]
    summary = {k: float(np.mean([c[k] for c in per_case.values()])) for k in keys}
    print(f"split={args.split}  cases={len(per_case)}  (mean over cases, samples after warm-up)\n")
    print("Interface velocity profile, relative L2 error:")
    for k in sorted(k for k in keys if k.endswith("profile_rel_l2")):
        print(f"  {k.replace('_profile_rel_l2', ''):<14} {summary[k]:.4f}")
    print("\nPressure correction dP, RMSE / label range:")
    for k in sorted(k for k in keys if k.endswith("dp_rel_rmse")):
        print(f"  {k.replace('_dp_rel_rmse', ''):<14} {summary[k]:.4f}")
    print("\nMass (flux) error of predicted profile, max |Q_profile - Q| / mean |Q|:")
    for k in sorted(k for k in keys if k.endswith("mass_err_max")):
        print(f"  {k.replace('_mass_err_max', ''):<14} {summary[k]:.2e}")
    print("\nPer case:")
    for cid, row in per_case.items():
        print(f"  {cid}: severity {row['severity']:.2f}, dP label {row['dp_label_range_pa'][0]:.0f}..{row['dp_label_range_pa'][1]:.0f} Pa, "
              + ", ".join(f"{k.replace('_rel_l2', '').replace('_rel_rmse', '')}={row[k]:.3f}"
                          for k in row if k.endswith(("rel_l2", "rel_rmse"))))
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps({"summary": summary, "per_case": per_case}, indent=2))


if __name__ == "__main__":
    main()
