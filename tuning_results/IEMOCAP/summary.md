# IEMOCAP - tuned L1/L2/L4 vs baseline (full grid)

Settings were chosen on VALIDATION scores only (score = mean, tuning seed 2007). Final numbers: test set, mean ± std over seeds [2007, 2008, 2009].

Chosen settings: `L1_ARGS="--cb_beta 0.9999 --focal_gamma 2.0"`  `L2_ARGS="--ls_eps 0.1 --ls_neighbor_mass 0.9"`  `L4_ARGS="--span_penalty 0.1"`

| method | weighted F1 | macro F1 | accuracy | hap F1 | learned spans (utterances): cross-event / contagion / cross-emotion |
|---|---|---|---|---|---|
| baseline | 70.02 ± 0.30 (+0.00) | 68.87 ± 0.36 (+0.00) | 69.83 ± 0.30 | 56.7 ± 1.5 | - |
| L1 | 70.45 ± 0.79 (+0.43) | 69.16 ± 0.84 (+0.30) | 70.49 ± 0.71 | 56.8 ± 2.0 | - |
| L2 | 70.01 ± 0.61 (-0.00) | 68.61 ± 0.39 (-0.26) | 69.95 ± 0.65 | 56.7 ± 0.8 | - |
| L4 | 70.10 ± 0.38 (+0.08) | 69.07 ± 0.43 (+0.21) | 69.83 ± 0.42 | 58.3 ± 1.9 | 40.3 / 49.6 / 46.7 |
| ALL (L1+L2+L4) | 70.41 ± 0.57 (+0.39) | 69.16 ± 0.75 (+0.29) | 70.28 ± 0.51 | 56.7 ± 2.2 | 10.3 / 37.6 / 31.8 |

(in brackets: difference to the baseline mean)

## L3 calibration (test ECE %, tuning-seed checkpoints; lower is better)

| method | T | uncalibrated | temperature scaling | MC-Dropout | MC-Dropout + TS |
|---|---|---|---|---|---|
| baseline | 1.411 | 11.58 | 3.43 | 3.97 | 3.99 |
| L1 | 0.915 | 2.53 | 1.41 | 10.43 | 2.32 |
| L2 | 1.135 | 5.49 | 3.09 | 4.46 | 3.81 |
| L4 | 1.373 | 11.39 | 5.70 | 4.32 | 4.29 |
| ALL (L1+L2+L4) | 0.814 | 4.67 | 1.97 | 13.64 | 2.08 |

## Grid search (tuning seed 2007; VALIDATION scores only; pick = highest "score" = mean)

| limitation | setting | valid wF1 | valid macro-F1 | score | mean span |
|---|---|---|---|---|---|
| baseline | - | 74.60 | 70.78 | 72.69 | - |
| L1 | L1_b0.999_g0.0 | 74.95 | 71.28 | 73.12 | - |
| L1 | L1_b0.999_g1.0 | 74.90 | 70.78 | 72.84 | - |
| L1 | L1_b0.999_g2.0 | 75.28 | 71.00 | 73.14 | - |
| L1 | L1_b0.9999_g0.0 | 74.59 | 70.79 | 72.69 | - |
| L1 | L1_b0.9999_g1.0 | 74.69 | 70.34 | 72.52 | - |
| L1 | L1_b0.9999_g2.0 | 75.43 | 71.01 | 73.22 | - |
| L2 | L2_e0.02_m0.5 | 75.16 | 71.37 | 73.27 | - |
| L2 | L2_e0.02_m0.9 | 75.15 | 71.10 | 73.12 | - |
| L2 | L2_e0.05_m0.5 | 75.35 | 71.54 | 73.44 | - |
| L2 | L2_e0.05_m0.9 | 75.38 | 71.44 | 73.41 | - |
| L2 | L2_e0.1_m0.5 | 75.62 | 71.62 | 73.62 | - |
| L2 | L2_e0.1_m0.9 | 75.73 | 71.89 | 73.81 | - |
| L4 | L4_p0.002 | 74.97 | 71.07 | 73.02 | 61.9 |
| L4 | L4_p0.005 | 74.87 | 71.08 | 72.97 | 54.7 |
| L4 | L4_p0.02 | 74.88 | 71.02 | 72.95 | 43.5 |
| L4 | L4_p0.1 | 75.10 | 71.18 | 73.14 | 29.0 |
