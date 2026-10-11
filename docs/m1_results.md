### Poiseuille verification (straight tube, steady)

Hagen–Poiseuille Δp = 59.418 Pa

| Mesh | Cells | Δp (Pa) | Error | Inlet flux error | Iterations | Wall (s) |
|---|---|---|---|---|---|---|
| coarse | 48,000 | 60.408 | +1.67% | 1.5e-10 | 212 | 16 |
| medium | 162,000 | 59.869 | +0.76% | 1.4e-10 | 328 | 84 |
| fine | 547,560 | 59.267 | -0.25% | 1.4e-10 | 357 | 70 |

### Mesh convergence (50% stenosis, steady mean flow)

Throat Reynolds number at mean flow ≈ 545

| Mesh | Cells | Δp (Pa) | Iterations | Converged | Wall (s) | Procs |
|---|---|---|---|---|---|---|
| coarse | 48,000 | 89.277 | 1842 | True | 94 | 1 |
| medium | 162,000 | 87.517 | 1698 | True | 311 | 1 |
| fine | 547,560 | 86.404 | 2451 | True | 266 | 8 |

Convergence: monotone; observed order 1.13; GCI(fine) 2.78%; extrapolated Δp 84.484 Pa

### Pulsatile cost and time-step sensitivity (coarse mesh, prescribed Womersley inflow)

| CFD steps / coupling step | Cells | Wall per cycle (s) | Wall per coupling step (s) | Max inlet-flux error |
|---|---|---|---|---|
| 20 | 48,000 | 524 | 5.24 | 3.2e-10 |
| 40 | 48,000 | 834 | 8.34 | 3.2e-10 |

Linear split of the per-run cost: fixed restart overhead ≈ 2.15 s per run (docker exec, solver start-up, mesh read, file I/O) + 155 ms per CFD time step.

3D pressure drop Δp = p_in − p_out over the last cycle: range -466.7 to 788.2 Pa (40 steps).
Time-step sensitivity of Δp (20 vs 40 steps, relative RMSE): 0.01%; cycle-to-cycle change of Δp: 1.17%.

### Coupled pilot (1D vessel + pimpleFoam, Mode A, Poiseuille inlet profile, 20 CFD steps per coupling step)

- Cells: 48,000; all 200 coupling steps converged: True (tol 1e-5 relative)
- Coupling iterations per step: mean 3.04, max 4 (Aitken-relaxed fixed point); CFD runs: 607
- Wall time per cardiac cycle: 1588 s
- Warm-up: mean interface pressure 12445 → 12143 Pa, mean flow 6.078 → 6.014 ml/s (target 6.000) over cycles 1 → 2

- Estimated restart-overhead share of the coupled wall time: 41% (2.15 s × 304 runs per cycle)

