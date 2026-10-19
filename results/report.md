# hemocouple results report

Work directory: `~/hemocouple-work`

## Dataset

56 / 56 cases generated.

- Peak throat Re (stenosis cases): max 1666, median 1070 (laminar cap 1700)
- Label flux correction (max per case): median 0.39%, worst 1.20%
- Cells per case: 151,200–172,800; wall time per case: median 13.8 min, total 15.7 h
- Splits: train 35, val 7, test 7, test_ood 7

## Training

| Variant | Basis | Flux projection | ΔP prior | λ_mass | λ_pressure | K (POD) | POD error (val) | Epochs | Time (s) |
|---|---|---|---|---|---|---|---|---|---|
| data_only | pod | False | none | 0.0 | 0.0 | 8 | 2.95e-03 | 232 | 53 |
| physics | pod | True | none | 0.0 | 0.1 | 8 | 2.95e-03 | 228 | 52 |
| physics_yt | pod | True | young_tsai | 0.0 | 0.1 | 8 | 2.95e-03 | 226 | 51 |
| physics_yt_stokes | pod | True | young_tsai | 0.0 | 0.0 | 16 | 9.41e-04 | 81 | 36 |
| soft_mass | pod | False | none | 1.0 | 0.1 | 8 | 2.95e-03 | 254 | 56 |

## Offline evaluation (teacher-forced, held-out cases)

### val

| Method | Profile rel. L2 | ΔP RMSE / range | Max mass error |
|---|---|---|---|
| 1d_only | – | 0.224 | – |
| data_only | 0.013 | 0.022 | 2.9e-02 |
| physics | 0.012 | 0.023 | 1.2e-15 |
| physics_yt | 0.012 | 0.028 | 1.2e-15 |
| physics_yt_stokes | 0.010 | 0.024 | 1.8e-15 |
| poiseuille | 0.123 | – | – |
| soft_mass | 0.014 | 0.023 | 3.3e-02 |
| womersley | 0.014 | – | – |

### test

| Method | Profile rel. L2 | ΔP RMSE / range | Max mass error |
|---|---|---|---|
| 1d_only | – | 0.245 | – |
| data_only | 0.012 | 0.033 | 1.7e-02 |
| physics | 0.010 | 0.040 | 1.2e-15 |
| physics_yt | 0.010 | 0.026 | 1.1e-15 |
| physics_yt_stokes | 0.007 | 0.025 | 1.7e-15 |
| poiseuille | 0.102 | – | – |
| soft_mass | 0.012 | 0.034 | 1.8e-02 |
| womersley | 0.012 | – | – |

### test_ood

| Method | Profile rel. L2 | ΔP RMSE / range | Max mass error |
|---|---|---|---|
| 1d_only | – | 0.327 | – |
| data_only | 0.015 | 0.077 | 3.9e-02 |
| physics | 0.011 | 0.072 | 1.2e-15 |
| physics_yt | 0.011 | 0.037 | 1.2e-15 |
| physics_yt_stokes | 0.009 | 0.035 | 1.8e-15 |
| poiseuille | 0.122 | – | – |
| soft_mass | 0.013 | 0.072 | 3.1e-02 |
| womersley | 0.014 | – | – |

## Coupled evaluation (two-way 1D–3D)

14 cases, last of 3 cycles, errors relative to the extended-domain reference at Γ.

| Method | p_Γ rel. RMSE | Q rel. RMSE | Reflection abs. diff | Max mass error | Iterations | Wall per run (min) | All converged |
|---|---|---|---|---|---|---|---|
| 1d_only_B | 0.0452 | 0.0367 | 0.0259 | 1.0e-15 | 2.78 | 0.0 | True |
| baseline_A | 0.0150 | 0.0095 | 0.0073 | 1.1e-15 | 2.84 | 816.4 | True |
| physics_yt_stokes_A | 0.0146 | 0.0094 | 0.0055 | 1.3e-15 | 2.85 | 493.1 | True |
| physics_yt_stokes_B | 0.0147 | 0.0115 | 0.0042 | 1.2e-15 | 2.81 | 0.0 | True |
| reference | – | – | – | 4.0e-13 | 2.88 | 1108.6 | True |
| young_tsai_B | 0.0415 | 0.0205 | 0.0084 | 9.7e-16 | 2.81 | 0.0 | True |
| young_tsai_cal_B | 0.0152 | 0.0129 | 0.0080 | 1.0e-15 | 2.84 | 0.0 | True |

- 1d_only_B: online speed-up vs coupled 3D reference ≈ 22853×

- physics_yt_stokes_B: online speed-up vs coupled 3D reference ≈ 24945×

- young_tsai_B: online speed-up vs coupled 3D reference ≈ 23664×

- young_tsai_cal_B: online speed-up vs coupled 3D reference ≈ 23394×

Per case p_Γ error:

- ctrl0048 (severity 0.00): 1d_only_B 0.0151, baseline_A 0.0153, physics_yt_stokes_A 0.0151, physics_yt_stokes_B 0.0149, young_tsai_B 0.0151, young_tsai_cal_B 0.0151
- sten0003 (severity 0.61): 1d_only_B 0.0392, baseline_A 0.0135, physics_yt_stokes_A 0.0130, physics_yt_stokes_B 0.0127, young_tsai_B 0.0461, young_tsai_cal_B 0.0132
- sten0007 (severity 0.35): 1d_only_B 0.0168, baseline_A 0.0169, physics_yt_stokes_A 0.0167, physics_yt_stokes_B 0.0168, young_tsai_B 0.0177, young_tsai_cal_B 0.0170
- sten0008 (severity 0.74): 1d_only_B 0.0800, baseline_A 0.0158, physics_yt_stokes_A 0.0154, physics_yt_stokes_B 0.0154, young_tsai_B 0.0632, young_tsai_cal_B 0.0158
- sten0014 (severity 0.71): 1d_only_B 0.0277, baseline_A 0.0140, physics_yt_stokes_A 0.0136, physics_yt_stokes_B 0.0134, young_tsai_B 0.0213, young_tsai_cal_B 0.0138
- sten0020 (severity 0.58): 1d_only_B 0.0228, baseline_A 0.0147, physics_yt_stokes_A 0.0142, physics_yt_stokes_B 0.0142, young_tsai_B 0.0272, young_tsai_cal_B 0.0144
- sten0024 (severity 0.70): 1d_only_B 0.0513, baseline_A 0.0125, physics_yt_stokes_A 0.0118, physics_yt_stokes_B 0.0119, young_tsai_B 0.0427, young_tsai_cal_B 0.0120
- sten0029 (severity 0.75): 1d_only_B 0.0859, baseline_A 0.0155, physics_yt_stokes_A 0.0151, physics_yt_stokes_B 0.0148, young_tsai_B 0.0771, young_tsai_cal_B 0.0156
- sten0033 (severity 0.38): 1d_only_B 0.0124, baseline_A 0.0121, physics_yt_stokes_A 0.0115, physics_yt_stokes_B 0.0114, young_tsai_B 0.0131, young_tsai_cal_B 0.0120
- sten0036 (severity 0.74): 1d_only_B 0.0566, baseline_A 0.0166, physics_yt_stokes_A 0.0163, physics_yt_stokes_B 0.0164, young_tsai_B 0.0439, young_tsai_cal_B 0.0167
- sten0041 (severity 0.22): 1d_only_B 0.0192, baseline_A 0.0193, physics_yt_stokes_A 0.0192, physics_yt_stokes_B 0.0190, young_tsai_B 0.0192, young_tsai_cal_B 0.0197
- sten0042 (severity 0.61): 1d_only_B 0.0465, baseline_A 0.0145, physics_yt_stokes_A 0.0141, physics_yt_stokes_B 0.0146, young_tsai_B 0.0560, young_tsai_cal_B 0.0143
- sten0043 (severity 0.73): 1d_only_B 0.1273, baseline_A 0.0105, physics_yt_stokes_A 0.0100, physics_yt_stokes_B 0.0115, young_tsai_B 0.1056, young_tsai_cal_B 0.0143
- sten0045 (severity 0.73): 1d_only_B 0.0323, baseline_A 0.0191, physics_yt_stokes_A 0.0189, physics_yt_stokes_B 0.0187, young_tsai_B 0.0325, young_tsai_cal_B 0.0189
