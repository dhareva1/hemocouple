# M2: reference dataset design

## What the operator must learn

At the interface Γ the operator maps a causal window of 1D quantities and the local geometry to the 3D interface response (proposal §7.3):

    (Q(t−kΔt), P_up(t−kΔt))_{k<m}, g  →  (u_n on Γ, ΔP_Γ)

This is a property of the **local 3D domain** downstream of Γ, plus a short upstream extension. It does not depend on how the 1D network produced Q(t). M2 therefore generates the training and validation labels with **prescribed-flow 3D reference runs**, not two-way coupled runs:

- M1 measured coupled runs at about 3 CFD restarts per coupling step plus 2.15 s of overhead per restart, so 1588 s per cycle on the coarse mesh.
- A continuous parallel prescribed run on the same mesh costs minutes.

Two-way coupled 1D–3D runs (Mode A/B, `FoamLocal3D` + `OneDVessel`) are kept for the **evaluation** of the locked test cases, where they are needed.

### Known limitation (reported, tested)

At deployment, the inputs come from the hybrid's 1D solution. In training they come from the 3D reference. The flow history Q is the same quantity in both. The pressure input differs slightly: in training it is the area-averaged 3D static pressure on a plane 1 cm upstream of Γ, while online it is the 1D tube-law pressure at the node 1 cm upstream. The coupled evaluation on the test cases measures the effect of this train/deploy shift. It is not assumed to be zero.

## Domain

```
z = 0          z = L_ext = 4R      z = L_ext + d          stenosis (length Ls)          outlet
inlet ────────── Γ (interface) ───────── ███████████ ──────────── 20 R ──────────── RC Windkessel
Womersley(Q)     labels sampled here
     ↑ P_up plane at Γ − 1 cm
```

- **Inlet:** Womersley profile of the case waveform Q(t), with the discrete flux made exact by `setAverage`. Q is identical at every cross-section (rigid wall, incompressible).
- **Interface Γ:** placed L_ext = 4R downstream of the inlet and d = 2R upstream of the stenosis start. The label profile there therefore contains real 3D effects: unsteady (Womersley) development and the stenosis's upstream influence. The conventional quasi-steady Poiseuille assumption ignores both.
- **Outlet:** RC Windkessel, the same parameters for all cases (`LumpedParams`). It is implicit through Q.

## Samples (per case)

The run lasts 5 cycles. Every coupling time t_k = k·T/100 is sampled; training discards the first 3 cycles (`warmup_cycles = 3`).

| Field | Source | CaseRecord field |
|---|---|---|
| Q(t_k) | prescribed waveform (equals mesh flux to ~1e-10) | `q` |
| P_up(t_k) | area-averaged p on the plane z = L_ext − 1 cm | `p1d` (operator input) |
| P_Γ(t_k) | area-averaged p on the plane z = L_ext | `p_interface` |
| u_n on Γ | probes at the reference-grid points on z = L_ext | `profiles` |
| P_lin(t_k) | reduced model: uniform Poiseuille segment Γ→outlet + the same Windkessel | `p_lin` |
| ΔP label | P_Γ − P_lin | `dp` (derived) |

## Parameter space

The parameter space is set after the M2 transition check (`scripts/m2_transition_check.py`). Bounds stay inside the region where the flow is laminar and cycle-periodic, and that region is reported (proposal §6). The cases are sampled by Latin hypercube with uniform-vessel controls, exactly as in `geometry.sample_cases`.

## Mesh and splits

- **Mesh:** medium (×1.5 the M1 coarse mesh, GCI ≈ 4%) for all cases. The locked test cases are recomputed on the fine mesh to bound the label uncertainty.
- **Splits:** case-level with an OOD-severity hold-out (`dataset.split_cases`). They are written once to `splits.json` before any training.
