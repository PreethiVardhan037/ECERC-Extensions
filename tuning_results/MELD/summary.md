# MELD - tuned L1/L2/L4 vs baseline (full grid)

Settings were chosen on VALIDATION scores only (score = mean, tuning seed 0). Final numbers: test set, mean ± std over seeds [0, 1, 2].

Chosen settings: `L1_ARGS="--cb_beta 0.995 --focal_gamma 1.0"`  `L2_ARGS="--ls_eps 0.02 --ls_neighbor_mass 0.5"`  `L4_ARGS="--span_penalty 0.02"`

**Note:** L2: no setting in the grid beat the baseline on the validation score "mean" (59.25 vs 60.28); the best one is shown anyway.

| method | weighted F1 | macro F1 | accuracy | fea F1 | dis F1 | learned spans (utterances): cross-event / contagion / cross-emotion |
|---|---|---|---|---|---|---|
| baseline | 66.00 ± 0.44 (+0.00) | 50.40 ± 0.71 (+0.00) | 66.87 ± 0.53 | 26.2 ± 2.7 | 30.7 ± 1.2 | - |
| L1 | 65.86 ± 0.37 (-0.14) | 50.21 ± 0.52 (-0.19) | 66.56 ± 0.45 | 24.7 ± 0.6 | 31.4 ± 1.3 | - |
| L2 | 65.73 ± 0.44 (-0.26) | 49.00 ± 1.03 (-1.40) | 67.00 ± 0.34 | 21.4 ± 3.5 | 27.8 ± 0.8 | - |
| L4 | 65.80 ± 0.22 (-0.20) | 49.85 ± 0.04 (-0.55) | 66.64 ± 0.33 | 24.5 ± 1.6 | 30.4 ± 0.5 | 1.9 / 5.7 / 6.0 |
| ALL (L1+L2+L4) | 65.78 ± 0.21 (-0.21) | 49.92 ± 0.46 (-0.48) | 66.51 ± 0.25 | 24.9 ± 2.4 | 30.6 ± 1.2 | 2.6 / 5.3 / 5.7 |

(in brackets: difference to the baseline mean)

## L3 calibration (test ECE %, tuning-seed checkpoints; lower is better)

| method | T | uncalibrated | temperature scaling | MC-Dropout | MC-Dropout + TS |
|---|---|---|---|---|---|
| baseline | 0.867 | 5.09 | 3.27 | 7.25 | 3.52 |
| L1 | 0.857 | 4.79 | 3.18 | 7.47 | 3.17 |
| L2 | 1.142 | 7.55 | 3.94 | 5.63 | 3.67 |
| L4 | 0.882 | 3.49 | 2.66 | 6.12 | 2.38 |
| ALL (L1+L2+L4) | 0.825 | 5.51 | 2.74 | 8.10 | 2.69 |

## Grid search (tuning seed 0; VALIDATION scores only; pick = highest "score" = mean)

| limitation | setting | valid wF1 | valid macro-F1 | score | mean span |
|---|---|---|---|---|---|
| baseline | - | 65.88 | 54.68 | 60.28 | - |
| L1 | L1_b0.99_g0.5 | 65.94 | 54.41 | 60.17 | - |
| L1 | L1_b0.99_g1.0 | 65.94 | 54.27 | 60.11 | - |
| L1 | L1_b0.99_g2.0 | 66.02 | 54.58 | 60.30 | - |
| L1 | L1_b0.995_g0.5 | 66.03 | 54.92 | 60.48 | - |
| L1 | L1_b0.995_g1.0 | 66.12 | 55.06 | 60.59 | - |
| L1 | L1_b0.995_g2.0 | 65.96 | 54.49 | 60.22 | - |
| L1 | L1_b0.999_g0.5 | 66.06 | 53.60 | 59.83 | - |
| L1 | L1_b0.999_g1.0 | 66.00 | 53.16 | 59.58 | - |
| L1 | L1_b0.999_g2.0 | 65.81 | 53.46 | 59.64 | - |
| L1 | L1_b0.9995_g0.5 | 65.64 | 53.08 | 59.36 | - |
| L1 | L1_b0.9995_g1.0 | 65.75 | 53.16 | 59.45 | - |
| L1 | L1_b0.9995_g2.0 | 66.01 | 53.18 | 59.59 | - |
| L2 | L2_e0.02_m0.5 | 65.76 | 52.74 | 59.25 | - |
| L2 | L2_e0.02_m0.9 | 65.76 | 52.74 | 59.25 | - |
| L2 | L2_e0.05_m0.5 | 65.43 | 52.49 | 58.96 | - |
| L2 | L2_e0.05_m0.9 | 65.53 | 52.61 | 59.07 | - |
| L2 | L2_e0.1_m0.5 | 65.16 | 52.26 | 58.71 | - |
| L2 | L2_e0.1_m0.9 | 65.27 | 52.33 | 58.80 | - |
| L4 | L4_p0.002 | 65.95 | 54.45 | 60.20 | 8.8 |
| L4 | L4_p0.005 | 65.75 | 53.82 | 59.78 | 6.6 |
| L4 | L4_p0.02 | 66.23 | 54.48 | 60.36 | 4.7 |
| L4 | L4_p0.1 | 65.87 | 54.59 | 60.23 | 3.1 |
