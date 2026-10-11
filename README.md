# hemocouple

Research code for the proposal *Physics-Informed Neural Operators for Improved Interface Consistency in Multiscale 1D–3D Cardiovascular Hemodynamics* (`../proposal.md`).

A POD-DeepONet learns the interface response of a local 3D domain from a causal window of 1D flow/pressure and local geometry. An exact flux-projection layer enforces mass conservation. The operator is then called inside a partitioned 1D–3D coupling loop.

> **Status (M1):** solvers locked: an in-house Python 1D solver (`oned.py`) and OpenFOAM v2412 `pimpleFoam`/`simpleFoam` running in Docker (`foam.py`). Both implement the coupling interface and are verified (see `docs/m1_results.md`). The ML pipeline can also run on a cheap **analytic mock** (`mock.py`). The mock is only for checking the software. Numbers it produces are **not** research results.

## Layout

| Path | Contents | Proposal |
|---|---|---|
| `src/hemocouple/geometry.py`, `waveforms.py` | Parametric stenosis cases, Latin-hypercube sampling, pulsatile inflow | 7.2 |
| `cross_section.py` | Reference Gauss–polar grid on the interface (normalized ρ = r/R) | 7.3 |
| `womersley.py` | Analytic pulsatile profiles (used by the mock) | – |
| `dataset.py` | Case records, causal history windows, case-level splits with an OOD-severity hold-out | 7.3, 7.6 |
| `pod.py` | Area-weighted POD; projection error used as the error floor | 7.3, 7.7 |
| `model.py` | `OperatorNet`: POD trunk or learned-trunk ablation, `flux_projection` hard constraint | 7.3, 7.4 |
| `losses.py`, `train.py` | Data / mass / total-pressure / PDE-hook loss; training with all statistics from train cases only | 7.4, 7.6 |
| `coupling.py`, `providers.py` | Dirichlet–Neumann fixed-point loop with Aitken relaxation; Mode A and Mode B | 7.5 |
| `metrics.py` | Relative RMSE, ε_Q(t), wave-separation reflection, break-even N* | 7.7 |
| `oned.py` | 1D elastic-tube solver (two-step Lax–Wendroff, Riemann-invariant BCs); `UpstreamModel` | 7.2 |
| `foam.py` | OpenFOAM case writer (O-grid stenosis mesh, mapped inlet profile, Windkessel outlet), Docker runner, `FoamLocal3D` (`LocalModel`), steady runs | 7.2 |
| `adapters.py` | openBF cross-check stub; `extract_interface_profile` (3D face → reference grid) | 7.2 |
| `mock.py` | Lumped upstream "1D", local "3D" (lubrication + Young–Tsai + kinetic term + Windkessel), oracle profile | – |

## Quickstart

```bash
pip install -e ".[dev]"      # or just run from the repo: tests and scripts add src/ to the path
python -m pytest             # ~2 s

python scripts/make_dataset.py   --config configs/dataset_mock.json   --out data/mock
python scripts/train_operator.py --data data/mock --config configs/train_physics.json   --out runs/physics.pt
python scripts/train_operator.py --data data/mock --config configs/train_data_only.json --out runs/data_only.pt
python scripts/evaluate_coupling.py --data data/mock --split test \
    --model physics=runs/physics.pt --model data_only=runs/data_only.pt --out runs/eval_test.json
```

The dataset step writes `splits.json` once. The test and `test_ood` cases stay locked from then on, and training reads only `train` and `val`. `configs/train_soft_mass.json` is the ablation with a penalty loss instead of the projection. To run the learned-trunk ablation, set `"basis": "trunk"` in any training config.

## M1 solver pilot

Needs Docker Desktop running and `docker pull opencfd/openfoam-run:2412`. Keep the work dir **outside OneDrive**:

```bash
python scripts/m1_pilot.py --work ~/hemocouple-work/m1 --stage all   # poiseuille, mesh, pulsatile, coupled
python scripts/m1_summary.py --work ~/hemocouple-work/m1 > docs/m1_results.md
```

## Full pipeline on a big machine (e.g. Mac Studio M3 Ultra)

One command runs everything, and every stage resumes from where it stopped:

```bash
# 0. once: Docker Desktop running, Settings → Resources → CPUs = all, Memory ≥ 16 GB
python3 -m pip install numpy scipy torch pytest      # or: pip install -e ".[dev]"

# 1. dataset produced on another machine (synced via OneDrive) → work dir outside OneDrive
mkdir -p ~/hemocouple-work/m2
rsync -a "<OneDrive>/researches/2026/blood_sim/data_m2/" ~/hemocouple-work/m2/

# 2. everything else (check → finish missing cases → train → offline → coupled → report)
python3 scripts/pipeline.py --work ~/hemocouple-work --stages all
# or skip the dataset stage if it is complete:
python3 scripts/pipeline.py --work ~/hemocouple-work --stages check train offline coupled report
```

| Stage | What it does | Parallelism (defaults from the performance-core count P) |
|---|---|---|
| `check` | Docker running, image pulled, Python deps, unit tests | – |
| `dataset` | Missing reference cases (`m2_make_dataset.py`, resumable) | P // 8 concurrent cases × 8 MPI ranks, one container each |
| `train` | Variants `physics_yt`, `physics`, `data_only`, `soft_mass` (`configs/train_*.json`) | – (seconds per variant) |
| `offline` | Teacher-forced metrics on val, test and test_ood (`evaluate_offline.py`) | – |
| `coupled` | Two-way 1D–3D runs on test and test_ood (`m3_coupled_eval.py`): reference, Mode A baseline and operator, Mode B | up to P concurrent cases (each OpenFOAM run is serial) |
| `report` | `report.md` in the work dir (`make_report.py`) | – |

- Outputs: `~/hemocouple-work/{m2,runs,eval,m3}/`, logs `m2/worker_*.log` and `m3/worker_*.log`, summary `~/hemocouple-work/report.md`.
- Useful flags: `--procs` (MPI ranks per dataset case), `--dataset-workers`, `--coupled-workers`, `--coupled-cycles` (default 3), `--coupled-tol` (default 1e-4), `--coupled-variants physics_yt,physics`.
- Timing reference (M4 Pro, 8 ranks): dataset cases take 7–35 min each (median ≈ 11). A coupled run on the medium mesh is several hours per method per case, which is why the coupled stage runs many cases at once.

## Design notes

- **Interface convention.** The 1D side sends Q and the local side returns P. Every solver adapter implements `trial(...)` (advance one coupling step without committing) and `accept()`, so a step can be repeated until it converges.
- **Mode A (main experiment).** The 3D solver runs and the operator supplies the inlet profile. The predicted ΔP is not used here, because the 3D solver returns the pressure.
- **Mode B (secondary).** No 3D solve. P_Γ = P_1D,transferred + ΔP̂. Use this mode for the cost question (RQ5), and remember it gives no interior 3D field.
- **Labels.** ΔP = P_3D,ref − P_1D,transferred. P_1D,transferred comes from a uniform-Poiseuille reduced model driven by the same Q. A test checks that Mode B fed with the exact label reproduces the reference.
- **Model selection.** Every variant is selected on the validation *data* loss. This keeps the comparison between the physics-informed and data-only variants fair.
- **L_PDE.** Data at the interface plane cannot give a Navier–Stokes residual. `losses.objective` takes an optional `pde_residual` hook for when near-interface 3D slabs are exported.
- **Compute cost.** Offline cost (data generation + training) and online wall time are reported separately. On the mock, N* is ∞ because the mock "3D" is cheaper than network inference. With real 3D solves this is expected to reverse, but the evaluation has to show it.

## Next steps

1. Replace the mock reference run in `make_dataset.py` with `OneDVessel` + `FoamLocal3D` runs, using an `extract_interface_profile`-based oracle from a longer upstream 3D domain. Store `steps_per_cycle` in the metadata of each record.
2. Move the per-step restart to preCICE, or run OpenFOAM in parallel per trial, if restart overhead dominates (see the coupled pilot timings).
3. Run mesh and time-step convergence studies before generating labels.
4. Repeat training with several seeds and report mean ± spread (proposal 7.6).
