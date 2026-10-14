"""Compare a regenerated reference dataset with the original one, case by case.

    python scripts/validate_regen.py --original ~/hemocouple-work/m2 --regen ~/hemocouple-work/m2_regen

For every case present in both directories, reports the largest difference of each label series relative to
its range (q, p1d, p_interface, p_lin, dp = p_interface - p_lin) and the relative L2 difference of the interface
profiles, plus the kept OpenFOAM case directory (snapshots, size). Writes <regen>/validation.json.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

SERIES = ["q", "p1d", "p_interface", "p_lin"]


def dir_size(path: Path) -> int:
    return sum(f.stat().st_size for f in path.rglob("*") if f.is_file())


def compare(a: dict, b: dict) -> dict:
    out = {}
    for k in SERIES + ["dp"]:
        x = a["p_interface"] - a["p_lin"] if k == "dp" else a[k]
        y = b["p_interface"] - b["p_lin"] if k == "dp" else b[k]
        out[k] = float(np.max(np.abs(x - y)) / max(np.ptp(x), 1e-30))
    pa, pb = a["profiles"], b["profiles"]
    out["profiles_rel_l2"] = float(np.linalg.norm(pa - pb) / np.linalg.norm(pa))
    out["same_t"] = bool(np.array_equal(a["t"], b["t"]))
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--original", type=Path, required=True)
    ap.add_argument("--regen", type=Path, required=True)
    args = ap.parse_args()

    rows = {}
    for f in sorted(args.regen.glob("*.npz")):
        orig = args.original / f.name
        if not orig.exists():
            continue
        with np.load(orig) as za, np.load(f) as zb:
            a, b = {k: za[k] for k in za.files}, {k: zb[k] for k in zb.files}
        row = compare(a, b)
        meta_a = json.loads(str(a["header"]))["meta"]
        meta_b = json.loads(str(b["header"]))["meta"]
        row["label_flux_correction_max"] = [meta_a.get("label_flux_correction_max"), meta_b.get("label_flux_correction_max")]
        case_dir = args.regen / "_cases" / f.stem
        if case_dir.exists():
            times = [d.name for d in case_dir.iterdir() if d.is_dir() and d.name.replace(".", "").isdigit()]
            procs = [d for d in case_dir.iterdir() if d.is_dir() and d.name.startswith("processor")]
            row["kept_case"] = {"time_dirs": sorted(times, key=float), "processor_dirs": len(procs),
                                "size_mb": round(dir_size(case_dir) / 2**20)}
        rows[f.stem] = row

    worst = {k: max((r[k] for r in rows.values()), default=float("nan")) for k in SERIES + ["dp", "profiles_rel_l2"]}
    print(f"{len(rows)} cases compared (original {args.original}, regenerated {args.regen})")
    print("largest difference over all cases (series: max |diff| / range; profiles: relative L2):")
    for k, v in worst.items():
        print(f"  {k:<16} {v:.2e}")
    print("\nper case: dp diff, profile diff, label flux correction (original -> regenerated), kept case size")
    for cid, r in rows.items():
        kc = r.get("kept_case", {})
        lc = r["label_flux_correction_max"]
        print(f"  {cid}: dp {r['dp']:.1e}, profiles {r['profiles_rel_l2']:.1e}, "
              f"flux corr {lc[0]:.3%} -> {lc[1]:.3%}, case {kc.get('size_mb', '?')} MB, "
              f"{len(kc.get('time_dirs', []))} time dirs, {kc.get('processor_dirs', 0)} processor dirs")
    (args.regen / "validation.json").write_text(json.dumps({"worst": worst, "cases": rows}, indent=2))


if __name__ == "__main__":
    main()
