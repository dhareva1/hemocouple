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

Every run's time series is saved to <work>/raw/<case>/<method>.npz and skipped if present, so a new model
only needs its own runs. --methods picks runs (reference, baseline_A, <model>_A, cfd = all 3D runs, B = all
Mode B runs, all). --jobs N runs one process per (case, 3D method) on N workers, references first, then the
Mode B runs and the metrics.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
import warnings
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
warnings.filterwarnings("ignore", message=".*encountered in matmul", category=RuntimeWarning)

from hemocouple import foam  # noqa: E402
from hemocouple.coupling import CouplingResult, run_coupled  # noqa: E402
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


def save_run(path: Path, res: CouplingResult, samples: list | None = None) -> None:
    arrays = {k: v for k, v in asdict(res).items() if isinstance(v, np.ndarray)}
    arrays["scalars"] = np.array(json.dumps({"provider_time": res.provider_time, "wall_time": res.wall_time,
                                             "meta": res.meta}))
    if samples:
        arrays["sample_pGamma"] = np.array([x["pGamma"] for x in samples])
        arrays["sample_profile"] = np.array([x["profile"] for x in samples])
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    with open(tmp, "wb") as fh:
        np.savez(fh, **arrays)
    tmp.rename(path)


def load_run(path: Path) -> tuple:
    with np.load(path) as z:
        arrays = {k: z[k] for k in z.files}
    scalars = json.loads(str(arrays.pop("scalars")))
    samples = {k[len("sample_"):]: arrays.pop(k) for k in list(arrays) if k.startswith("sample_")}
    return CouplingResult(**arrays, provider_time=scalars["provider_time"], wall_time=scalars["wall_time"],
                          meta=scalars["meta"]), samples


def cfd_methods(models: dict) -> list:
    return ["reference", "baseline_A"] + [f"{k}_A" for k in models]


def run_queue(args: argparse.Namespace, ids: list, models: dict, work: Path) -> None:
    """One subprocess per missing (case, 3D method), longest first, then Mode B + metrics in-process."""
    severity = {cid: json.loads(str(np.load(args.data / f"{cid}.npz")["header"]))["spec"]["geometry"]["severity"]
                for cid in ids}
    by_cost = sorted(ids, key=severity.get, reverse=True)  # severe stenoses need the smallest CFD time steps
    jobs = [(cid, m) for m in cfd_methods(models) for cid in by_cost if not (work / "raw" / cid / f"{m}.npz").exists()]
    (work / "logs").mkdir(parents=True, exist_ok=True)
    log(f"queue: {len(jobs)} 3D runs on {args.jobs} workers")
    passthrough = ["--data", str(args.data), "--work", str(work), "--cycles", str(args.cycles), "--tol", str(args.tol),
                   "--substeps", str(args.substeps), "--vessel-length", str(args.vessel_length),
                   "--wave-speed", str(args.wave_speed)] + [f"--model={m}" for m in args.model]
    if args.refine:
        passthrough += ["--refine", str(args.refine)]

    def one(job: tuple) -> bool:
        cid, method = job
        log(f"start {cid} {method}")
        with open(work / "logs" / f"{cid}__{method}.log", "a") as fh:
            ok = subprocess.run([sys.executable, __file__, *passthrough, "--cases", cid, "--methods", method,
                                 "--container", f"hemocouple-m3-{cid}-{method}"], stdout=fh, stderr=subprocess.STDOUT
                                ).returncode == 0
        log(f"{'done' if ok else 'FAILED'} {cid} {method}")
        return ok

    with ThreadPoolExecutor(args.jobs) as pool:
        failed = [j for j, ok in zip(jobs, pool.map(one, jobs)) if not ok]
    evaluate(args, ids, models, work, ["B"], container=None)
    if failed:
        raise SystemExit(f"{len(failed)} 3D runs failed: {failed}")


def evaluate(args: argparse.Namespace, ids: list, models: dict, work: Path, methods: list, container) -> None:
    meta = json.loads((args.data / "dataset.json").read_text())
    grid = CrossSectionGrid.gauss_polar(meta["n_r"], meta["n_theta"])
    ref_cfg = ReferenceConfig(**{**meta.get("reference", {}), **({"refine": args.refine} if args.refine else {})})
    lumped = LumpedParams()
    (work / "results").mkdir(parents=True, exist_ok=True)
    def want(m: str) -> bool:
        return "all" in methods or m in methods or ("B" if m.endswith("_B") else "cfd") in methods
    runner = None

    try:
        for cid in ids:
            raw = work / "raw" / cid
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

            todo = [m for m in cfd_methods(models) if want(m) and not args.skip_cfd and not (raw / f"{m}.npz").exists()]
            if todo and runner is None:
                runner = foam.DockerFoam(work, name=container).start()
            for name in todo:
                if name == "reference":
                    log(f"{cid}: reference (extended 3D, {full.n_cells} cells)")
                    loc = local(work / "cases" / cid / "reference", full, sample_z=l_ext,
                                planes={"pGamma": l_ext, "pUp": l_ext - ref_cfg.probe_distance})
                    far = WomersleyFluxProfile(wave, grid, r, ref_cfg.viscosity / ref_cfg.density)
                    loc.initialize(far.predict(None, q0, 0.0)[0])
                    save_run(raw / "reference.npz", couple(vessel(args.vessel_length), far, "A", 1, loc=loc), loc.samples)
                    continue
                k = name[: -len("_A")]
                prov, window = ((PoiseuilleProfile(grid, r), 1) if name == "baseline_A"
                                else (OperatorProvider(models[k], spec.features(), r), models[k].history_length))
                log(f"{cid}: {name} (truncated 3D, {trunc.n_cells} cells)")
                loc = local(work / "cases" / cid / name, trunc)
                loc.initialize(PoiseuilleProfile(grid, r).predict(None, q0, 0.0)[0])
                save_run(raw / f"{name}.npz", couple(vessel(args.vessel_length + l_ext), prov, "A", window, loc=loc))

            seg = full.length - l_ext
            for name, prov, window in [("1d_only_B", PoiseuilleProfile(grid, r), 1)] + [
                (f"{k}_B", OperatorProvider(m, spec.features(), r), m.history_length) for k, m in models.items()
            ]:
                if want(name) and not (raw / f"{name}.npz").exists():
                    save_run(raw / f"{name}.npz", couple(vessel(args.vessel_length + l_ext), prov, "B", window,
                                                         reduced=ReducedSegment(seg, r, ref_cfg, lumped, q0, p0)))

            runs = {f.stem: load_run(f) for f in sorted(raw.glob("*.npz"))} if raw.exists() else {}
            if not runs:
                continue
            s = slice(-ref_cfg.steps_per_cycle, None)
            result = {"case_id": cid, "severity": spec.geometry.severity, "cycles": args.cycles, "methods": {}}
            ref, samples = runs.get("reference", (None, None))
            if ref is not None:
                p_gamma_ref, prof_ref = samples["pGamma"], samples["profile"]
            for name, (res, _) in runs.items():
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
            (work / "results" / f"{cid}.json").write_text(json.dumps(result, indent=2))
            log(f"{cid}: " + ", ".join(f"{k} p_Γ err {v.get('p_gamma_rel_rmse', float('nan')):.3f}"
                                        for k, v in result["methods"].items() if k != "reference"))
    finally:
        if runner is not None:
            runner.stop()


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
    ap.add_argument("--methods", nargs="+", default=["all"], help="reference, baseline_A, <model>_A, cfd, B, all")
    ap.add_argument("--jobs", type=int, default=0, help="run each (case, 3D method) as its own process on N workers")
    ap.add_argument("--container", default=None, help="Docker container name (default hemocouple-m3-<shard>)")
    args = ap.parse_args()
    shard_k, shard_n = (int(x) for x in args.shard.split("/"))

    splits = json.loads((args.data / "splits.json").read_text())
    ids = args.cases or [cid for name in args.split.split(",") for cid in splits[name]]
    ids = [cid for i, cid in enumerate(ids) if i % shard_n == shard_k]
    models = {k: load_checkpoint(Path(v))[0] for k, v in (m.split("=", 1) for m in args.model)}
    work = args.work.resolve()
    if args.jobs:
        run_queue(args, ids, models, work)
    else:
        evaluate(args, ids, models, work, args.methods, args.container or f"hemocouple-m3-{shard_k}")


if __name__ == "__main__":
    main()
