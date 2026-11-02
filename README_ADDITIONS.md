# ECERC – complete code: authors' release + Limitations 1, 2, 3 and 4

Base: official ECERC code (https://github.com/TAN-OpenLab/ECERC, ACL 2025, commit 339fcaa). The authors' `Loss` class and the
authors' model are reproduced numerically by our files when the fixes are off (`tests/run_tests.py` compares them with verbatim
copies of the official files in `tests/reference/`). With **no flags** and `--legacy_rng_compat`, `train.py` reproduces the authors'
training trajectory exactly (verified epoch by epoch on synthetic data; without that flag the only difference is that the removed
per-epoch test pass no longer consumes one random number per epoch).

> **Read `PROJECT_STATUS.md` first** (what changed in this revision, issue by issue, and what has / has not been verified).

## What was added where (identical layout in `MELD/` and `IEMOCAP/`)

| File | Limitation | Change |
|---|---|---|
| `loss.py` | L1, L2 | Authors' `Loss` untouched (byte-identical to the official one). **Appended** `ECERCImprovedLoss` (class-balanced + focal, confusion-aware label smoothing), `build_training_loss` (the single place that wires flags to losses), `smoothing_table_str`, `class_counts_from_loader`, `parse_confusable_pairs`. |
| `train.py` | L1, L2, L4 | Rewritten around the authors' loop (identical data loaders, model, optimiser, early stopping). **No test evaluation inside the epoch loop**: checkpoint = best validation weighted-F1, test set evaluated **once** afterwards; every metric goes to `<ckpt>_results.json`. `--span_max` is set from the data. |
| `eval_utils.py` (new) | all | ONE implementation of accuracy / F1 / per-class P-R-F1 / confusion matrix / ECE / NLL / Brier / per-pair breakdown, used by train, calibrate and the reports. |
| `report.py` (new, top level) | all | Consolidated machine-readable report (`final_results.json/csv`, `final_report.md`). |
| `tests/` (new) | all | `run_tests.py` (unit tests against the official model/loss), `smoke_test.py` (end-to-end on synthetic data), `make_synthetic_data.py`. |
| `model.py` | L4 + CPU fix | New `AdaptiveSpan` module + helpers (rows with no allowed key are left untouched; bounded renormalisation; exact identity at full history). Authors' attention classes only gain an **optional** `span=None` argument (lines marked `[CHANGED]`). With L4 off the model and its `state_dict` are identical to the authors'. **`[FIXED]`** the authors' `UnboundLocalError` on CPU (`imask`/`mask` were assigned only in the GPU branch) with a 2-line `else:` branch. |
| `calibrate.py` (new) | L3 | Temperature scaling + MC-Dropout, ECE/NLL/Brier before vs after (deterministic and stochastic reported separately), pooled AND per-pair confusable breakdown with sample counts, reliability diagrams. Loads L4 checkpoints automatically. |
| `span_benchmark.py` (new) | L4 | Accuracy / F1 / ECE vs attention span (fixed-span sweep; learned-span curve) + dialogue-length statistics. |
| `inference.py`, `dataloader.py` | – | Unchanged. |
| top level | all | `auto_tune.py` (validation-based tuning + multi-seed final comparison of baseline/L1/L2/L1+L2/L4/ALL + L3 calibration of every checkpoint), `run_final.sh`, `download_data.py`, `run_ablation.sh`, `run_span_sweep.sh`, `summarize_logs.py` (reads the JSON results). |

## Flags (`train.py`)
* `--lim1` Class-Balanced weights (effective number of samples) + Focal loss. `--cb_beta` (MELD 0.999, IEMOCAP 0.999; lower = flatter weights), `--focal_gamma` (MELD 1.0 = the authors' MELD value, IEMOCAP 2.0).
* `--lim2` Confusion-aware label smoothing. `--ls_eps` (0.1), `--ls_neighbor_mass` (0.7), `--confusable_pairs` (MELD `dis-ang`; IEMOCAP `hap-exc,ang-fru`).
  Target distribution for a true class with a confusable partner: `1-eps` on the true class, `eps*neighbor_mass` on the partner(s), and the
  remaining `eps*(1-neighbor_mass)` spread only over the NON-partner classes (eps=0.1, neighbor_mass=0.7 -> 90% / 7% / 3% shared by the rest).
* `--lim4` Adaptive attention span in the four Evidence-Cause Interaction branches.
  `--span_max` (default 0 = set automatically to the longest dialogue in the data -> span_init=1 is EXACTLY full-history attention; a value below longest-1 is refused),
  `--span_ramp` (2.0), `--span_init` (1.0 = start from full history),
  `--span_branch_level` (one span per branch shared by its 8 heads; the DEFAULT is one span per attention head, as in Sukhbaatar et al.),
  `--span_penalty` (0.005), `--span_lr` (MELD 0.005, IEMOCAP 0.01).
* No flag = authors' original behaviour. Flags combine: `--lim1 --lim2 --lim4`.
* `--seed`, `--save_model_path` (default `<output_dir>/ECERC_trained.pkl`).
* (`--lim1/--lim2` are not `--l1/--l2` because the authors already use `--l2` for weight decay.)

## Limitation 4 – how it works
Each of the four branches (self-event, cross-event, self-emotion/contagion, cross-emotion) gets learnable spans
`z` (one per attention head by default; Sukhbaatar et al., 2019). A key `d` utterances back keeps `clamp((ramp + z - d)/ramp, 0, 1)` of its attention
weight, then the row is renormalised. `z` starts at the full history (so training starts from the baseline) and is
pulled down by an L1 penalty `--span_penalty`; the task loss pushes back when cutting hurts. The spans have their own
learning rate (`--span_lr`, no weight decay) because the network's `lr` (1e-5 on MELD) is far too small to move them,
and are clamped to [0, 1] after each step. Spans are printed every epoch and saved inside the checkpoint.
* Only **three** branches have a meaningful span. The **self-event** branch uses the identity mask (it attends only to the target
  utterance, distance 0), so its span provably cannot change the output (unit-tested: zero gradient, identical logits even at span 0).
  It is wired in for completeness, excluded from the penalty and the span statistics, and reported explicitly as `inactive`.
* Rows whose mask allows no key (e.g. the first utterance of a speaker) are softmaxed by the authors' code over all keys; the span
  leaves exactly those rows untouched so that the span only ever changes the context the authors' mask already allowed.
* Adam moves a span by about `span_lr * span_max` utterances per step regardless of the penalty's size, so
  `--span_penalty` mainly decides *where* the span stops, not how fast it moves.
* **Scope of L4.** The span mask is applied to the attention weights after the softmax, so L4 learns and limits the *effective receptive
  field* (how far back each branch actually looks) but the full L x L attention tensor is still computed: it does **not** reduce the
  O(L^2) computation or memory. Describe it as "learned effective context per branch/head", not as a complexity reduction.
* Learned spans are reported per branch (cross-event / contagion / cross-emotion) and per head at the end of each `--lim4` log.
* All defaults are untuned starting points.

## Recommended: let `auto_tune.py` pick the settings (validation only) and compare with seeds
```
python auto_tune.py MELD                      # tune on validation, then the full final comparison (add --quick for small grids)
python auto_tune.py MELD --tune_seeds 0 1 2   # rank settings by the MEAN validation score of 3 seeds (more stable selection)
python auto_tune.py MELD --final_only         # skip tuning, reuse tuning_results/MELD/best_config.json (or --L1_ARGS/--L2_ARGS/--L4_ARGS)
bash run_final.sh MELD "0 1 2"                # == --final_only with seeds 0 1 2 + the optional seed ensemble
```
Protocol (same for every method): checkpoint = best validation weighted-F1 epoch; test set evaluated **once** per run; settings chosen on
**validation only** (`--select mean` = average of validation weighted- and macro-F1; default tuning = ONE fixed documented seed, the authors' seed);
final comparison over `--seeds` for **baseline, L1, L2, L1+L2, L4, ALL**; calibration (L3) on **every** checkpoint of every method and seed.
Output in `tuning_results/<DATASET>/`: `final_report.md` (all tables), `final_results.json` / `final_results.csv` (machine-readable: accuracy,
weighted/macro F1 and P/R, per-class P/R/F1, confusion matrices, ECE, NLL, Brier, calibrated variants, per-pair calibration, spans, mean ± std),
`summary.md` (report + grid), `grid.csv`, `best_config.json`, `runs.json`, `logs/`. `--resume` continues after a disconnect.
If no setting beats the baseline on validation, the summary says so.
**Always check the status:** `tuning_results/<DATASET>/STATUS_COMPLETE.txt` exists and the command exited with 0 only if every required training result and calibration exists.
Otherwise `STATUS_INCOMPLETE.txt` + `status.json` list what is missing, the report starts with a "RESULTS INCOMPLETE" banner, and the exit code is 2.
Pass-through options: `--extra "--legacy_rng_compat"` / `--extra="--epochs 20"` and `--calib_args "--mc_passes 10"` both work.

## Run manually
```
pip install gdown && python download_data.py          # authors' preprocessed features -> data/meld, data/iemocap
cd MELD                                               # or IEMOCAP
python inference.py                                   # authors' checkpoint (sanity check)
python train.py --lim1 --lim2 --lim4 --save_model_path ckpt_all.pkl
python calibrate.py --load_model_state_dir ckpt_all.pkl                    # L3 (works on any checkpoint)
python span_benchmark.py --load_model_state_dir ECERC_MODEL.pkl            # L4: accuracy/ECE vs fixed span
python span_benchmark.py --learned_ckpts ckpt_all.pkl                      # L4: learned span vs accuracy/ECE
cd ..
bash run_ablation.sh MELD 0                           # baseline, L1, L2, L1+L2, L4, all + calibration + span benchmark
bash run_span_sweep.sh MELD 0                         # L4 span-penalty sweep (learned span vs accuracy/ECE)
python summarize_logs.py "MELD/*_results.json"        # one comparison table from the structured results
```
Tuned values can be passed to the ablation without editing code:
`L1_ARGS="--cb_beta 0.999 --focal_gamma 1" L2_ARGS="--ls_eps 0.05" bash run_ablation.sh MELD 0`.

## Notes
* The authors' `model.py` crashed on CPU (`UnboundLocalError`: `imask`/`mask` assigned only in the GPU branch). This is
  fixed (`[FIXED]` else-branch); the code now also runs on CPU, but use a GPU for real runs.
* The authors' preprocessed features are not in this zip (Google Drive); `download_data.py` fetches them.
  The authors' released checkpoints (`ECERC_MODEL.pkl`) are included.
* The test set is evaluated exactly once per run (after training, with the best-validation checkpoint). The authors' loop built a test iterator
  every epoch, which also consumed one random number per epoch; `--legacy_rng_compat` reproduces that so old runs stay exactly reproducible.
* `span_benchmark.py` is an *analysis* of finished checkpoints on the test set (accuracy vs span); nothing is selected with it.
* Run `python tests/run_tests.py` (≈30 s) and `python tests/smoke_test.py` (a few minutes) after any change; neither needs data or a GPU.
* The shell scripts use `set -euo pipefail`, so a failing training run stops the ablation instead of being hidden by `tee`.

## Docker (SOP: Python 3.9, PyTorch 2.0.1)
```
bash docker_check.sh              # builds the image and runs the unit tests inside it (add --smoke for the end-to-end test)
docker run --rm -it --gpus all -v "$PWD/data":/workspace/data -v "$PWD/tuning_results":/workspace/tuning_results ecerc
```
CPU-only image: `docker build --build-arg TORCH_INDEX=https://download.pytorch.org/whl/cpu -t ecerc-cpu .`  The Docker build itself has not been run by the author of this revision
(no Docker available); the pinned packages and the code were verified on Python 3.9 + PyTorch 2.0.0 instead.

## Limitation 4 and the SOP
Three branches have a learnable span (cross-event, self-emotion/contagion, cross-emotion). The self-event branch attends only to the target utterance, so a span cannot affect it; it is reported
as inactive. Exact SOP wording: `SOP_corrections.md`.
