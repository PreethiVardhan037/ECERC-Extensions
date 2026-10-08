# ECERC Robustness and Reliability — Final Experimental Report

## 1. Experimental status

All planned computational experiments were completed on MELD and IEMOCAP.

The experiments compare the original ECERC baseline with:
- L1: Class-Balanced Loss + Focal Loss
- L2: Confusion-aware Label Smoothing
- L3: Temperature Scaling and MC-Dropout
- L4: Adaptive/Learnable Attention Span
- ALL: L1 + L2 + L4

## 2. Final multi-seed ablation

### MELD

| Method | Weighted F1 | Macro F1 | Accuracy | Fear F1 | Disgust F1 |
|---|---:|---:|---:|---:|---:|
| Baseline | 66.00 | 50.40 | 66.87 | 26.2 | 30.7 |
| L1 | 65.86 | 50.21 | 66.56 | 24.7 | 31.4 |
| L2 | 65.73 | 49.00 | 67.00 | 21.4 | 27.8 |
| L4 | 65.80 | 49.85 | 66.64 | 24.5 | 30.4 |
| ALL | 65.78 | 49.92 | 66.51 | 24.9 | 30.6 |

ALL relative to baseline: weighted F1 -0.22.

### IEMOCAP

| Method | Weighted F1 | Macro F1 | Accuracy | Happy F1 |
|---|---:|---:|---:|---:|
| Baseline | 70.02 | 68.87 | 69.83 | 56.7 |
| L1 | 70.45 | 69.16 | 70.49 | 56.8 |
| L2 | 70.01 | 68.61 | 69.95 | 56.7 |
| L4 | 70.10 | 69.07 | 69.83 | 58.3 |
| ALL | 70.41 | 69.16 | 70.28 | 56.7 |

ALL relative to baseline: weighted F1 +0.39.

## 3. L2 confusion audit

The paper identifies closely related emotion categories as recurring sources of error.

The measured two-way confusion counts are stored in `confusable_pair_audit.csv`.

The result should be interpreted as follows:

- L2 does not consistently reduce the targeted confusions.
- Therefore the project should not claim that confusion-aware label smoothing successfully solved the confusability limitation.
- The negative/neutral result itself is a legitimate experimental finding.

## 4. L3 calibration

For the combined ALL model:

### MELD
Uncalibrated ECE: 5.51%
Temperature scaling: 2.74%

### IEMOCAP
Uncalibrated ECE: 4.67%
Temperature scaling: 1.97%

Temperature scaling substantially reduces ECE on both datasets.

## 5. L4 adaptive attention span

The learned span experiments are stored in the span CSV/JSON files.

The final combined model learns approximately:

- MELD: mean effective span ≈ 4.49 utterances.
- IEMOCAP: mean effective span ≈ 28.22 utterances.

The component-level learned spans are also preserved in the generated JSON files.

The results demonstrate that the adaptive mechanism learns different effective conversational context lengths for the two datasets.

## 6. Scientific interpretation

### L1 — Class imbalance
Dataset-dependent.

IEMOCAP shows a small overall improvement with L1, while MELD does not. MELD Fear F1 decreases relative to the baseline, while Disgust F1 changes only modestly.

### L2 — Category confusability
Not demonstrated as successful by the current experiments.

The aggregate metrics decline and the targeted confusion-pair analysis does not show a consistent reduction.

### L3 — Confidence calibration
Clearly successful.

Temperature scaling substantially reduces ECE for the combined model on both datasets.

### L4 — Adaptive attention span
Mechanistically demonstrated.

The model learns bounded, dataset-dependent spans. However, the current results do not establish a consistent overall F1 improvement, particularly on MELD.

### Combined model
The combined model improves over its reproduced baseline on IEMOCAP, but not on MELD.

Therefore the project conclusion should emphasize improved reliability/calibration and learned contextual selectivity rather than claiming that every limitation was completely solved.

## 7. Files in this package

- `final_ablation_results.csv`
- `final_per_class_results.csv`
- `final_calibration_results.csv`
- `confusable_pair_audit.csv`
- `confusion_matrices.json`
- `selected_hyperparameters.json`
- calibration figures
- confusion-matrix figures
- attention-span figures
- span-sweep CSV/JSON files

