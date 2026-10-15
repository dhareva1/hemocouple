# hemocouple

Research code for the proposal *Physics-Informed Neural Operators for Improved Interface Consistency in Multiscale 1D–3D Cardiovascular Hemodynamics* (`../proposal.md`).

A POD-DeepONet learns the interface response of a local 3D domain from a causal window of 1D flow/pressure and local geometry. An exact flux-projection layer enforces mass conservation. The operator is then called inside a partitioned 1D–3D coupling loop.

> **Status (M3 complete, manuscript for CMAME in preparation):** solvers verified (M1, `docs/m1_results.md`); 56-case OpenFOAM reference dataset (M2), regenerated on a second machine as a reproduction check; five-seed training sweep; two-way coupled evaluation on 14 locked test / out-of-distribution cases (M3). All numbers in the manuscript come from `scripts/paper_numbers.py`. The **analytic mock** (`mock.py`) only checks the software; numbers it produces are **not** research results.

**Final model** (`configs/train_physics_yt_stokes.json`): POD-DeepONet with the exact flux projection, a Young–Tsai residual prior for ΔP, a causal transient-Stokes velocity prior with a learned POD residual (`stokes.py`), weight decay 0.1, no total-pressure loss. Headline results (mean over five seeds / 14 coupled cases): interface profile error 0.77% (test) and 0.92% (out of distribution) against 10.2% for a Poiseuille profile; ΔP error 2.5% / 3.8% of the label range against 4.8% / 8.4% without the prior; in coupling, Mode B reaches 1.44% interface-pressure error (1D alone 4.5%, truncated-domain 3D 1.50%) at about 1/20 000 of the reference's online cost, break-even after about nine runs.

## Layout

| Path | Contents | Proposal |
|---|---|---|
| `src/hemocouple/geometry.py`, `waveforms.py` | Parametric stenosis cases, Latin-hypercube sampling, pulsatile inflow | 7.2 |
| `cross_section.py` | Reference Gauss–polar grid on the interface (normalized ρ = r/R) | 7.3 |
| `womersley.py` | Analytic pulsatile profiles (mock, far inlet of reference runs, offline baseline) | – |
| `stokes.py` | Causal transient-Stokes (time-domain Womersley) profile from a flow-rate history; velocity prior | 7.3 |
| `dataset.py` | Case records, causal history windows, case-level splits with an OOD-severity hold-out | 7.3, 7.6 |
| `pod.py` | Area-weighted POD; projection error used as the error floor | 7.3, 7.7 |
| `model.py` | `OperatorNet`: POD trunk or learned-trunk ablation, `flux_projection` hard constraint, Young–Tsai ΔP prior, Stokes velocity prior (`history_length` = rows callers pass) | 7.3, 7.4 |
| `losses.py`, `train.py` | Data / mass / total-pressure / PDE-hook loss; per-head optimizers, LR schedules and model selection; all statistics from train cases only | 7.4, 7.6 |
| `coupling.py`, `providers.py` | Dirichlet–Neumann fixed-point loop with Aitken relaxation; Mode A and Mode B | 7.5 |
| `metrics.py` | Relative RMSE, ε_Q(t), wave-separation reflection, break-even N* | 7.7 |
| `oned.py` | 1D elastic-tube solver (two-step Lax–Wendroff, Riemann-invariant BCs); `UpstreamModel` | 7.2 |
| `foam.py` | OpenFOAM case writer (O-grid stenosis mesh, mapped inlet profile, Windkessel outlet), Docker runner, `FoamLocal3D` (`LocalModel`), steady runs | 7.2 |
| `adapters.py` | openBF cross-check stub; `extract_interface_profile` (3D face → reference grid) | 7.2 |
| `mock.py` | Lumped upstream "1D", local "3D" (lubrication + Young–Tsai + kinetic term + Windkessel), oracle profile | – |
| `reference.py` | Extended-domain reference runs (labels), reduced segment for Mode B | 7.2 |

| Script | Purpose |
|---|---|
| `pipeline.py` | End-to-end: check → dataset → train (sweep + export) → offline → coupled → report |
| `m1_pilot.py`, `m1_summary.py`, `m2_transition_check.py` | Solver verification (M1) and laminar limit |
| `m2_make_dataset.py` | 56-case reference dataset (`--keep-cases` keeps the OpenFOAM cases) |
| `train_sweep.py` | Paper experiment matrix: variants × settings × five seeds; `--summarize`, `--export` (best-validation seed) |
| `train_operator.py` | Single training run from a config |
| `evaluate_offline.py` | Teacher-forced metrics on a locked split, with Poiseuille / Womersley / 1D-only baselines |
| `m3_coupled_eval.py` | Coupled runs per (case, method), saved time series, `--jobs` queue |
| `make_report.py`, `paper_numbers.py` | `report.md`; every number in the manuscript (`paper_numbers.md/json`) |
| `validate_regen.py` | Compare a regenerated dataset with the original |

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
| `train` | Paper sweep (`train_sweep.py`, 130 runs) into `sweep/`, then the best-validation seed of each model to `runs/` | `--train-workers` × `--train-threads` (~30 min) |
| `offline` | Sweep summary over all seeds (`sweep/summary.md`) and teacher-forced metrics of `runs/` on val, test and test_ood | – |
| `coupled` | Two-way 1D–3D runs on test and test_ood (`m3_coupled_eval.py`): reference, Mode A baseline and operator, Mode B | up to P concurrent cases (each OpenFOAM run is serial) |
| `report` | `report.md` (`make_report.py`) and `paper_numbers.md` (`paper_numbers.py`) | – |

- Outputs: `~/hemocouple-work/{m2,runs,eval,m3}/`, logs `m2/worker_*.log`, `m3/queue.log` and `m3/logs/<case>__<method>.log`, summary `~/hemocouple-work/report.md`.
- Useful flags: `--procs` (MPI ranks per dataset case), `--dataset-workers`, `--coupled-workers`, `--coupled-cycles` (default 3), `--coupled-tol` (default 1e-4), `--coupled-variants physics_yt_stokes` (default). Coupled runs are saved per (case, method) in `m3/raw/`, so adding a model later only runs that model.
- Timing reference: dataset cases take 7–35 min each on an M4 Pro (8 ranks, median ≈ 11) and about half that on an M3 Ultra. A three-cycle coupled run is 12–18 h per method per case (serial), so the 42 3D runs of the coupled stage take about 1.5 days on 20 workers.

## Reproduction check

```bash
python scripts/m2_make_dataset.py --config configs/dataset_m2.json --out ~/hemocouple-work/m2_regen --keep-cases --procs 8 --shard 0/2   # and 1/2
python scripts/m2_make_dataset.py --config configs/dataset_m2.json --out ~/hemocouple-work/m2_regen   # writes splits.json
python scripts/validate_regen.py --original ~/hemocouple-work/m2 --regen ~/hemocouple-work/m2_regen
python scripts/paper_numbers.py --work ~/hemocouple-work   # adds the re-evaluation on the regenerated labels
```

On the M3 Ultra, the regenerated labels match the original ones (median ΔP difference 2e-4 of the label range, identical splits and flux corrections; the three most severe out-of-distribution cases differ by up to 3% at the ΔP peaks), and every reported metric changes by at most 0.02 percentage points. The kept cases hold the mesh, five full-field snapshots per case (decomposed, 8 ranks) and all probe / plane time series (~15 GB).

## Design notes

- **Interface convention.** The 1D side sends Q and the local side returns P. Every solver adapter implements `trial(...)` (advance one coupling step without committing) and `accept()`, so a step can be repeated until it converges.
- **Mode A (main experiment).** The 3D solver runs and the operator supplies the inlet profile. The predicted ΔP is not used here, because the 3D solver returns the pressure.
- **Mode B (secondary).** No 3D solve. P_Γ = P_1D,transferred + ΔP̂. Use this mode for the cost question (RQ5), and remember it gives no interior 3D field.
- **Labels.** ΔP = P_3D,ref − P_1D,transferred. P_1D,transferred comes from a uniform-Poiseuille reduced model driven by the same Q. A test checks that Mode B fed with the exact label reproduces the reference.
- **Model selection.** Every variant is selected on the validation *data* loss, per output head (profile and ΔP keep their own best epoch, optimizer and LR schedule). The regularization (weight decay 0.1) is shared by all variants and chosen on the validation profile and ΔP errors relative to no regularization (`train_sweep.selection_scores`); the exported checkpoint is the best-validation seed. This keeps the comparison between the physics-informed and data-only variants fair.
- **Stokes prior.** The profile is the transient Stokes response to the last 100 steps of Q (one cycle; matches periodic Womersley to 0.05%) plus a learned POD residual; the flux projection acts on the residual. It uses only past Q, so it is valid inside the coupling loop (the offline Womersley baseline is not: it knows the whole periodic waveform).
- **L_PDE.** Data at the interface plane cannot give a Navier–Stokes residual. `losses.objective` takes an optional `pde_residual` hook for when near-interface 3D slabs are exported.
- **Compute cost.** Offline cost (data generation + training) and online wall time are reported separately. On the mock, N* is ∞ because the mock "3D" is cheaper than network inference. With real 3D solves this is expected to reverse, but the evaluation has to show it.

## Next steps

1. Manuscript: narrative revision (Stokes prior in the methods, total-pressure-loss finding, Mode A discussion), then submission to CMAME.
2. Release: code on GitHub and data (interface dataset, coupled time series, checkpoints) on Zenodo with a DOI for the data-availability statement.
3. Follow-up: patient-specific geometry, compliant walls, outlet variability; preCICE or parallel restarts to cut the coupled run time.
