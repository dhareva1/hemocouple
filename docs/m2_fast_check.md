# M2 fast end-to-end check (preliminary, 2026-10-10)

**Purpose:** confirm that the full pipeline runs on real OpenFOAM data and that the approach points the right way. **Not reportable results**: there are only 12 cases, the mesh is coarse (≈20k cells, ×0.75 of the M1 coarse mesh) and the wall spacing is uniform.

Pipeline exercised:

1. `m2_make_dataset.py` (12 prescribed-flow reference runs, 4 cycles each, 35–340 s per case)
2. `train_operator.py` (physics-informed and data-only variants, 3–5 s each)
3. `evaluate_offline.py` (held-out cases)
4. Mode B coupling with `OneDVessel` + the trained operator

Config: `configs/dataset_m2_fast.json`. Splits: 8 train (2 uniform controls), 2 val, 2 test.

## Laminar check (medium mesh, 4 cycles, `m2_transition_check.py`)

| Case | Severity | Peak throat Re | Downstream cycle-to-cycle | High-frequency share | Verdict |
|---|---|---|---|---|---|
| pilot | 0.50 | 1200 | ≤ 3e-8 | ≤ 4e-6 | laminar, periodic |
| high | 0.75 | 1700 | ≤ 3e-5 | ≤ 3% (periodic jet edge) | laminar, periodic |
| worst | 0.75 (R 2.5 mm, Q_peak 18 ml/s) | 2800 | 9–13% | up to 35% | **transitional** |

The full M2 bounds must keep the peak throat Re at about 1700 or below.

## Offline results on held-out cases (teacher-forced, samples after warm-up)

| Metric | Physics op | Data-only op | Poiseuille (conventional) | Womersley (analytic, knows the full waveform) | 1D-only |
|---|---|---|---|---|---|
| Profile rel. L2, **test** | **0.034** | 0.082 | 0.097 | 0.029 | – |
| Profile rel. L2, val | **0.016** | 0.047 | 0.079 | 0.024 | – |
| ΔP RMSE / range, **test** | **0.166** | 0.184 | – | – | 0.266 |
| ΔP RMSE / range, val | **0.095** | 0.104 | – | – | 0.247 |
| ΔP RMSE / range, train | 0.009 | 0.013 | – | – | 0.225 |
| Profile mass error (max) | **4e-16** | 0.18 | 0 | – | – |

## Online (Mode B, `OneDVessel` + trained operator, 4 cycles)

- Every step converged in 3–4 iterations.
- The flux error stays around 1e-16.
- Operator inference costs about 0.1 s for 4 cycles.

## Reading

1. **The pipeline works end to end on real CFD data.**
2. **Velocity profile.** The physics-informed operator is about 3× better than the conventional Poiseuille assumption. It is on par with the analytic Womersley profile: better on val, slightly worse on test. Womersley is a strong reference here because the reference inlet sits only 4R upstream. Moving Γ closer to the stenosis, or using more varied waveforms, should widen the gap where it matters.
3. **Mass conservation.** The exact projection gives conservation to machine precision. The data-only variant loses up to 18% of the flux on held-out cases. This is the hard-constraint effect the proposal claims, and it is already clear.
4. **Pressure correction ΔP.** The operator improves on 1D-only, but it overfits strongly: 0.9% error on train against 17% on test.
   - Both test cases have short, steep stenoses (Ls/R 3.3–3.8). The training stenoses are mostly longer (4.0–5.9), and only one has severity ≥ 0.6.
   - Even with exact inputs, the operator reaches only about 113 Pa where the label peaks at 396 Pa. This is a coverage gap of 8 training cases, not a bug.
5. **Train/deploy shift is visible online.** The coupled 1D gives smaller flow peaks (11.0 against 12.8 ml/s) because of upstream compliance, so the ΔP applied online is smaller still. The M3 coupled evaluation must quantify this, as stated in m2_design.md.

## Changes made because of this check

- **Labels.** The probed profile loses 2.5% of the flux on average (up to 8%) on this coarse mesh, because the near-wall (Stokes) layer is under-resolved. `FoamCaseSpec.radial_grading` now exists, and blockMesh confirms a wall-cell ratio of 4. The full M2 should use grading ≈ 4 on the medium mesh, and report the label flux correction per case.
- **Metric.** The inlet-flux metric now compares against the linearly interpolated imposed flow, not the analytic waveform.

## Recommendations before the full M2 run

1. Bounds: keep the peak throat Re ≤ 1700, and make sure short, steep stenoses (Ls/R ≈ 2–4 at high severity) are well covered.
2. Medium mesh, radial grading 4, 5 cycles.
3. Consider a dimensionless ΔP target, for example ΔP / (½ρU²(A0/A1 − 1)²), or a Young–Tsai feature in g. This gives the operator the right scaling for the quadratic expansion loss and should reduce the coverage-driven overfitting.
4. At least 40–60 cases, so that the case-level split leaves enough stenosis variety in train.
