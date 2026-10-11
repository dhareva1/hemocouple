# Milestone M1: solver pilot report

Date: 2026-10-10 · Machine: Apple M4 Pro (12 cores), Docker Desktop (8 GB), OpenFOAM v2412 (`opencfd/openfoam-run:2412`, arm64)
Results tables: [`m1_results.md`](m1_results.md) · Raw data: [`m1_raw/`](m1_raw/) · Reproduce: `scripts/m1_pilot.py`, `scripts/m1_summary.py`

## Decisions locked at M1

| Item | Decision |
|---|---|
| 1D solver | In-house `hemocouple.oned` (A–Q elastic tube, two-step Lax–Wendroff, Riemann-invariant BCs) |
| 3D solver | OpenFOAM v2412 `pimpleFoam` (transient), `simpleFoam` (steady verification), in Docker |
| Coupling | Python orchestrator, Dirichlet–Neumann (1D → Q, 3D → p), fixed point with Aitken relaxation, tol 1e-5 relative |
| Coupling step | T/100 (8 ms for T = 0.8 s), 20 CFD steps per coupling step (CFD Δt = 0.4 ms, max Courant ≈ 0.4 on the coarse mesh) |
| Inlet profile transfer | Reference polar grid → dense rings (64 angles) + centre + no-slip wall ring → `timeVaryingMappedFixedValue` with `setAverage` |
| Outlet | RC Windkessel in Python, implicit through Q_out = Q_in (rigid wall, incompressible) |

## Verification

**1D solver** (`tests/test_oned.py`, runs in CI):
- Steady flow converges to the friction law at first order: 2.2% at 41 nodes, 1.3% at 81, 0.8% at 161. The first-order boundary characteristics limit the rate.
- Pulse arrival time matches the set wave speed within 3%.
- The absorbing outlet reflects less than 1%. A pressure outlet reflects about −1 for pressure, damped by friction.
- Global mass balances.

**3D, Poiseuille:**
- Every mesh is within 1.7% of the Hagen–Poiseuille Δp, and the fine mesh is within 0.25%.
- Convergence is not monotone because two errors have opposite signs: the polygonal inlet area deficit (+, ∝ 1/n²) and discretization (−).
- The discrete inlet flux equals Q to about 1e-10 on all meshes.

**3D, 50% stenosis, steady mean flow (Re_throat ≈ 545):**
- Convergence is monotone: Δp = 89.28 → 87.52 → 86.40 Pa.
- Observed order 1.13. Fine-mesh GCI is 2.8%; the medium-mesh GCI is about 4.3%, estimated from the same three levels.
- The extrapolated Δp is 84.5 Pa.

**3D, pulsatile:**
- Going from 20 to 40 CFD steps per coupling step changes the 3D pressure drop by only 0.01% (relative RMSE over the last cycle). The pressure drop swings from −467 to +788 Pa, mostly from inertia.
- The inlet flux error stays at or below 3e-10.

### Bug found and fixed during the pilot

The first Poiseuille runs came out 5–7% below the analytic Δp, and the error *grew* with refinement. The cause was the inlet profile mapping:

- The reference grid has only 8 angles per ring, so OpenFOAM's planar triangulation between the outer ring and the zero-velocity wall ring cut the corners.
- That made the interpolated mean about 9% low, and `setAverage` then scaled the whole profile up into a peaked shape.
- The extra momentum flux of that peaked profile is recovered as pressure, which is why Δp came out low.

The fix writes each ring on 64 angles plus a centre point before mapping (`foam.upsample_profile`). This matters for the research itself. **Any inlet profile passed to the 3D solver, including the operator's output, has to be checked for mapping bias.** In Mode A, a biased mapping would look like an operator error.

## Coupling pilot (1D vessel + pimpleFoam, Mode A, Poiseuille inlet, 2 cycles, coarse mesh)

- All 200 coupling steps converged. Mean iterations per step: 3.04; maximum: 4. This was stable throughout systole and diastole.
- Warm-up was still decaying after 2 cycles. Mean flow went from 6.078 to 6.014 ml/s against a target of 6.000, and mean interface pressure from 12445 to 12143 Pa. **Datasets need at least 4–5 cycles**, or an initial state taken from a periodic 1D-only run.

## Cost model

Measured on the coarse mesh (48k cells), serial:

- Each CFD run costs about **2.15 s of fixed restart overhead** (docker exec, solver start-up, mesh and field read, file I/O) plus **155 ms per CFD time step**.
- In coupled Mode A that comes to **1588 s per cardiac cycle**. Restart overhead is about **41%** of the wall time.

Projection per case at 5 cycles. The medium-mesh row is an estimate, assuming CFD cost proportional to cells and about 3 s of overhead.

| Mesh | Cells | Coupled cost / cycle | Per case (5 cycles) | 48 cases, serial | 48 cases, 5 concurrent on this Mac |
|---|---|---|---|---|---|
| coarse | 48k | 1588 s (measured) | 2.2 h | 106 h | ≈ 21 h |
| medium | 162k | ≈ 4100 s (est.) | ≈ 5.7 h | ≈ 274 h | ≈ 55 h |

Generating the reference data for the planned dataset is feasible on this workstation over a few days. A cluster is optional, not required.

## Recommendations for M2 (dataset generation)

1. **Mesh.** Use the medium mesh (GCI ≈ 4%) for the training and validation labels. Recompute the locked test cases on the fine mesh, so the reported test errors are not dominated by label uncertainty.
2. **Time step.** Keep 20 CFD steps per coupling step and T/100 coupling steps.
3. **Warm-up.** Run 5 cycles and keep the last 2. Initialize the 3D field from the steady solution at mean flow to shorten the transient.
4. **Throughput.** Run 4–5 cases concurrently, one container each (`DockerFoam(name=...)`). The 41% restart overhead is the largest single saving available later: preCICE, or a persistent solver process. It is not needed to start M2.
5. **Reynolds number.** The peak throat Re at 50% severity is about 1010 (peak interface flow 11.1 ml/s in the coupled pilot); higher severities raise it. Before M2, check for transitional behaviour in the post-stenotic jet at the highest severity and peak flow. The proposal assumes laminar flow and requires this criterion to be reported (§6).

## Open items

- The 1D boundaries are first order. This is acceptable for coupling, but report the 1D discretization error in the paper, and cross-check one case against openBF.
- The pressure definition must be the same on both sides. Currently both use static pressure: area-averaged on the 3D inlet, and the 1D tube law at the interface node. The operator input P_1D is sampled 1 cm upstream.
- `FoamLocal3D` restarts from the last accepted time directory. It does not yet support parallel decomposition per trial, so medium and fine coupled runs will be serial unless that is added.
