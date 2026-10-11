"""M3: two-way coupled evaluation on locked test cases (proposal 7.5, 7.7).

    python scripts/m3_coupled_eval.py --data ~/hemocouple-work/m2 --split test \
        --model physics_yt=runs/physics_yt.pt --work ~/hemocouple-work/m3 [--cases ID ...] [--refine 1.0]

For every case:
    reference   1D vessel (length L) coupled to the EXTENDED 3D domain (Womersley-type far inlet,
                Γ inside the 3D region); P_Γ, P_up and the Γ profile are sampled in the 3D domain
    baseline_A  1D vessel (L + L_ext) coupled at Γ to the truncated 3D domain, Poiseuille inlet profile
    <model>_A   same, with the operator's inlet profile
    1d_only_B   1D vessel (L + L_ext) + reduced segment, ΔP = 0 (no 3D)
    <model>_B   same, with the operator's ΔP
Metrics are over the last cycle, compared with the reference at Γ. Results go to <work>/results/<case>.json.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import warnings
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
warnings.filterwarnings("ignore", message=".*encountered in matmul", category=RuntimeWarning)

from hemocouple import foam  # noqa: E402
from hemocouple.coupling import run_coupled  # noqa: E402
from hemocouple.cross_section import CrossSectionGrid  # noqa: E402
from hemocouple.geometry import CaseSpec  # noqa: E402
from hemocouple.metrics import characteristic_impedance, mass_error, reflection_coefficient, relative_rmse  # noqa: E402
from hemocouple.mock import LumpedParams  # noqa: E402
from hemocouple.oned import OneDVessel, VesselParams  # noqa: E402
from hemocouple.providers import OperatorProvider, PoiseuilleProfile, WomersleyFluxProfile  # noqa: E402
from hemocouple.reference import ReducedSegment, ReferenceConfig  # noqa: E402
from hemocouple.train import load_checkpoint  # noqa: E402


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def truncated_spec(full: foam.FoamCaseSpec, cfg: ReferenceConfig) -> foam.FoamCaseSpec:
    """Same mesh settings, but the 3D domain starts at Γ (gap_radii upstream of the stenosis)."""
    r = full.geometry.radius
    return foam.FoamCaseSpec(full.geometry, cfg.gap_radii * r, full.downstream_length, full.n_square, full.n_radial,
                             full.axial_cell, full.density, full.viscosity, full.radial_grading)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--split", default="test", help="split name(s), comma-separated, e.g. test,test_ood")
    ap.add_argument("--cases", nargs="*")
    ap.add_argument("--model", action="append", default=[], help="name=checkpoint.pt")
    ap.add_argument("--work", type=Path, required=True)
    ap.add_argument("--refine", type=float, default=None, help="mesh factor (default: as in the dataset)")
    ap.add_argument("--cycles", type=int, default=4)
    ap.add_argument("--substeps", type=int, default=20)
    ap.add_argument("--vessel-length", type=float, default=0.15, help="1D vessel length upstream of the 3D inlet, m")
    ap.add_argument("--tol", type=float, default=1e-5)
    ap.add_argument("--skip-cfd", action="store_true", help="only the Mode B runs (no OpenFOAM)")
    ap.add_argument("--wave-speed", type=float, default=6.0)
    ap.add_argument("--shard", default="0/1", help="k/N: evaluate only cases with index %% N == k")
    args = ap.parse_args()
    shard_k, shard_n = (int(x) for x in args.shard.split("/"))

    meta = json.loads((args.data / "dataset.json").read_text())
    splits = json.loads((args.data / "splits.json").read_text())
    ids = args.cases or [cid for name in args.split.split(",") for cid in splits[name]]
    ids = [cid for i, cid in enumerate(ids) if i % shard_n == shard_k]
    grid = CrossSectionGrid.gauss_polar(meta["n_r"], meta["n_theta"])
    ref_cfg = ReferenceConfig(**{**meta.get("reference", {}), **({"refine": args.refine} if args.refine else {})})
    lumped = LumpedParams()
    models = {k: load_checkpoint(Path(v))[0] for k, v in (m.split("=", 1) for m in args.model)}
    work = args.work.resolve()
    (work / "results").mkdir(parents=True, exist_ok=True)
    pending = [cid for cid in ids if not (work / "results" / f"{cid}.json").exists()]
    runner = None if (args.skip_cfd or not pending) else foam.DockerFoam(work, name=f"hemocouple-m3-{shard_k}").start()

    try:
        for cid in ids:
            out_path = work / "results" / f"{cid}.json"
            if out_path.exists():
                continue
            header = json.loads(str(np.load(args.data / f"{cid}.npz")["header"]))
            spec = CaseSpec.from_dict(header["spec"])
            r, period = spec.geometry.radius, spec.flow.period
            dt, steps = period / ref_cfg.steps_per_cycle, ref_cfg.steps_per_cycle * args.cycles
            wave = spec.flow.waveform()
            q0, p0 = float(wave(0.0)), spec.flow.mean_flow * lumped.distal_resistance
            l_ext = ref_cfg.gamma_z(spec)
            full = ref_cfg.foam_spec(spec)
            trunc = truncated_spec(full, ref_cfg)
            flux = lambda u: float(grid.integrate(u, r))  # noqa: E731
            zc = characteristic_impedance(ref_cfg.density, args.wave_speed, spec.geometry.area)

            def vessel(length: float) -> OneDVessel:
                n = int(round(length / 1e-3)) + 1
                return OneDVessel(VesselParams(length=length, radius=r, p_ref=p0, n_nodes=n), lambda t: float(wave(t)),
                                  q0=q0, p0=p0)

            def local(case_dir: Path, fspec, **sampling) -> foam.FoamLocal3D:
                return foam.FoamLocal3D(runner, case_dir, fspec, grid, coupling_dt=dt, substeps=args.substeps,
                                        distal_resistance=lumped.distal_resistance,
                                        distal_compliance=lumped.distal_compliance, q0=q0, p_distal0=p0, **sampling)

            def couple(upstream, provider, mode, window, loc=None, reduced=None):
                return run_coupled(upstream, provider, flux, mode=mode, dt=dt, n_steps=steps, window=window,
                                   q_init=q0, p1_init=p0, p_init=p0, local=loc, reduced=reduced, tol=args.tol,
                                   max_iter=30)

            runs, extra = {}, {}
            if runner is not None:
                log(f"{cid}: reference (extended 3D, {full.n_cells} cells)")
                loc = local(work / "cases" / cid / "reference", full, sample_z=l_ext,
                            planes={"pGamma": l_ext, "pUp": l_ext - ref_cfg.probe_distance})
                far = WomersleyFluxProfile(wave, grid, r, ref_cfg.viscosity / ref_cfg.density)
                loc.initialize(far.predict(None, q0, 0.0)[0])
                runs["reference"] = couple(vessel(args.vessel_length), far, "A", 1, loc=loc)
                extra["reference"] = loc.samples
                for name, prov, window in [("baseline_A", PoiseuilleProfile(grid, r), 1)] + [
                    (f"{k}_A", OperatorProvider(m, spec.features(), r), m.init_kwargs["window"]) for k, m in models.items()
                ]:
                    log(f"{cid}: {name} (truncated 3D, {trunc.n_cells} cells)")
                    loc = local(work / "cases" / cid / name, trunc)
                    loc.initialize(PoiseuilleProfile(grid, r).predict(None, q0, 0.0)[0])
                    runs[name] = couple(vessel(args.vessel_length + l_ext), prov, "A", window, loc=loc)
            seg = full.length - l_ext
            for name, prov, window in [("1d_only_B", PoiseuilleProfile(grid, r), 1)] + [
                (f"{k}_B", OperatorProvider(m, spec.features(), r), m.init_kwargs["window"]) for k, m in models.items()
            ]:
                runs[name] = couple(vessel(args.vessel_length + l_ext), prov, "B", window,
                                    reduced=ReducedSegment(seg, r, ref_cfg, lumped, q0, p0))

            s = slice(-ref_cfg.steps_per_cycle, None)
            result = {"case_id": cid, "severity": spec.geometry.severity, "cycles": args.cycles, "methods": {}}
            ref = runs.get("reference")
            if ref is not None:
                samples = extra["reference"]
                p_gamma_ref = np.array([x["pGamma"] for x in samples])
                prof_ref = np.array([x["profile"] for x in samples])
            for name, res in runs.items():
                m = {
                    "converged_all": bool(res.converged.all()), "iterations_mean": float(res.iterations.mean()),
                    "wall_time_s": res.wall_time, "provider_time_s": res.provider_time,
                    "mass_err_max": float(mass_error(res.q[s], res.q_local[s], float(np.mean(np.abs(res.q[s])))).max()),
                    "reflection_1d": reflection_coefficient(res.p1d[s], res.q[s], zc),
                    "p_interface_range": [float(res.p_interface[s].min()), float(res.p_interface[s].max())],
                }
                if ref is not None and name != "reference":
                    m["p_gamma_rel_rmse"] = relative_rmse(res.p_interface[s], p_gamma_ref[s])
                    m["q_rel_rmse"] = relative_rmse(res.q[s], ref.q[s])
                    m["reflection_abs_diff"] = abs(m["reflection_1d"] - reflection_coefficient(ref.p1d[s], ref.q[s], zc))
                result["methods"][name] = m
            if ref is not None:
                result["reference_profile_flux_err_max"] = float(np.max(np.abs(grid.integrate(prof_ref[s], r) - ref.q[s]))
                                                                 / np.mean(np.abs(ref.q[s])))
            out_path.write_text(json.dumps(result, indent=2))
            log(f"{cid}: " + ", ".join(f"{k} p_Γ err {v.get('p_gamma_rel_rmse', float('nan')):.3f}"
                                        for k, v in result["methods"].items() if k != "reference"))
    finally:
        if runner is not None:
            runner.stop()


if __name__ == "__main__":
    main()
