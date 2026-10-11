# Quick report: status on 2026-10-11

Physics-informed neural operators for 1D–3D interface coupling (proposal: `../../proposal.md`).

## Where we are

| Milestone | Status |
|---|---|
| Proposal revision (reviewer feedback, §7.2 solvers, §6 laminar limit) | done |
| Code: full pipeline + tests (43 passing) | done |
| M1 solver pilot (1D verified, OpenFOAM verified, coupling stable) | done ([m1_report.md](m1_report.md)) |
| M2 laminar check | done: laminar up to peak throat Re ≈ 1700, transitional at 2800 |
| M2 fast end-to-end check (12 cases) | done ([m2_fast_check.md](m2_fast_check.md)) |
| M2 full dataset (56 cases, medium mesh, wall-graded) | running on the M4 Pro, ~22/56 done, ETA ~midnight; auto-copied to OneDrive `data_m2/` when it stops |
| M3 coupled evaluation | tooling ready and smoke-tested; heavy runs scheduled on the M3 Ultra |

## Key results so far

**Solvers (M1).**
- *Straight-tube check:* the 3D pressure drop is within 1.7% of the analytic value on three meshes (0.25% on the finest).
- *Stenosis mesh convergence:* the uncertainty estimate on the fine mesh (GCI) is 2.8% for the 50% stenosis.
- *Time step:* halving it changes the pressure drop by 0.01%.
- *Coupling:* the coupled 1D–3D run converges in 3–4 iterations per step.
- *Bug fixed:* the inlet-mapping bug, which had biased the pressure drop by about 7%, was found and fixed.

**Labels (M2).**
- On the medium mesh with wall grading, the flux correction applied to each label is 0.3–0.8%, against 6–8% on the unrefined coarse mesh.
- 56 cases cover severity 0.20–0.75 and Ls/R 2–6, with peak throat Re ≤ 1666.

**Operator (fast check, 8 training cases, held-out test cases).** These are preliminary numbers from a small dataset.

| Method | Profile rel. L2 | ΔP error / range | Mass error |
|---|---|---|---|
| Physics-informed + Young–Tsai ΔP prior | 4.2% | **4.5%** | ~1e-16 |
| Physics-informed, no prior | 3.7% | 17.2% | ~1e-16 |
| Data-only (no physics) | 7.2% | 17.6% | up to 15% |
| Conventional Poiseuille inlet | 9.7% | – | – |
| 1D only (no correction) | – | 26.6% | – |

**Coupled smoke test** (1 case, tiny mesh, 1 cycle):
- Mode B with the operator, which needs no 3D solve, matches full coupled 3D at the interface (p_Γ error 1.62% against 1.60%). It runs about 250× faster online, and beats 1D-only (2.23%).
- Mode A with the operator profile is not yet distinguishable from the Poiseuille baseline at this resolution.

## What this means for the paper

1. **Exact flux projection:** mass is conserved to machine precision, while the data-only variant loses up to 15–18% of the flux. This is clear already.
2. **Physics prior for ΔP:** it is the difference between overfitting (17%) and generalizing (4.5%). The learned scale of 0.95 is close to the Young–Tsai coefficient, a good physical consistency check.
3. **Mode B speed-up (RQ5):** the operator is about 250× faster online, at an accuracy comparable to coupled 3D in the smoke test. It needs confirmation on the full test set.
4. **Open question:** the benefit of the operator's *profile* in Mode A. The analytic Womersley profile is a strong competitor because the reference inlet is close to Γ.

## Tomorrow on the M3 Ultra

See `README.md` → "Full pipeline on a big machine":

```bash
rsync -a "<OneDrive>/researches/2026/blood_sim/data_m2/" ~/hemocouple-work/m2/
python3 scripts/pipeline.py --work ~/hemocouple-work --stages all
```

The `dataset` stage only fills in cases that are still missing. The `coupled` stage runs the test and test_ood cases concurrently, using up to one worker per performance core. `~/hemocouple-work/report.md` collects everything at the end.

## Known limitations to report

- The training inputs (3D pressure on a plane) differ from the deployment inputs (1D tube-law pressure). The coupled evaluation measures this shift.
- The 1D solver boundaries are first order (about 1% steady error at 1 mm spacing).
- Uniform controls and stenoses use a single reference Windkessel. Outlet variability is not yet a parameter.
- Flow between Re_t 1700 and 2800 is unclassified and excluded from the dataset.
