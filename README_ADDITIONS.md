# ECERC – complete code: authors' release + Limitations 1, 2, 3 and 4

Base: official ECERC code (https://github.com/TAN-OpenLab/ECERC, ACL 2025). Everything not listed below is the
authors' file, byte for byte. With **no flags**, `train.py` reproduces the authors' training exactly
(checked epoch-by-epoch against their original `train.py`, same seed).

## What was added where (identical layout in `MELD/` and `IEMOCAP/`)

| File | Limitation | Change |
|---|---|---|
| `loss.py` | L1, L2 | Authors' `Loss` untouched. **Appended** `ECERCImprovedLoss` (class-balanced + focal, confusion-aware label smoothing), `class_counts_from_loader`, `parse_confusable_pairs`. |
| `train.py` | L1, L2, L4 | Additive blocks marked `[ADDED]`; one changed line `seed_everything(args.seed)` (default = authors' seed). Adds flags, loss selection, best-checkpoint saving, per-class report, span penalty/optimizer group/logging. |
| `model.py` | L4 + CPU fix | New `AdaptiveSpan` module + helpers. Authors' attention classes only gain an **optional** `span=None` argument (lines marked `[CHANGED]`). With L4 off the model and its `state_dict` are identical to the authors'. **`[FIXED]`** the authors' `UnboundLocalError` on CPU (`imask`/`mask` were assigned only in the GPU branch) with a 2-line `else:` branch. |
| `calibrate.py` (new) | L3 | Temperature scaling + MC-Dropout, ECE/NLL/Brier before vs after, confusable-class breakdown, reliability diagrams. Loads L4 checkpoints automatically. |
| `span_benchmark.py` (new) | L4 | Accuracy / F1 / ECE vs attention span (fixed-span sweep; learned-span curve) + dialogue-length statistics. |
| `inference.py`, `dataloader.py` | – | Unchanged. |
| top level | all | `auto_tune.py` (validation-based tuning of L1/L2/L4 + multi-seed final comparison + L3 calibration), `download_data.py`, `run_ablation.sh`, `run_span_sweep.sh`, `summarize_logs.py`. |

## Flags (`train.py`)
* `--lim1` Class-Balanced weights (effective number of samples) + Focal loss. `--cb_beta` (MELD 0.999, IEMOCAP 0.999; lower = flatter weights), `--focal_gamma` (MELD 1.0 = the authors' MELD value, IEMOCAP 2.0).
* `--lim2` Confusion-aware label smoothing. `--ls_eps` (0.1), `--ls_neighbor_mass` (0.7), `--confusable_pairs` (MELD `dis-ang`; IEMOCAP `hap-exc,ang-fru`).
  Target distribution for a true class with a confusable partner: `1-eps` on the true class, `eps*neighbor_mass` on the partner(s), and the
  remaining `eps*(1-neighbor_mass)` spread only over the NON-partner classes (eps=0.1, neighbor_mass=0.7 -> 90% / 7% / 3% shared by the rest).
* `--lim4` Adaptive attention span in the four Evidence-Cause Interaction branches.
  `--span_max` (MELD 40, IEMOCAP 120; must exceed the longest dialogue: MELD 33, IEMOCAP ~110),
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
* The **self-event** branch uses the identity mask (it attends only to the target utterance, distance 0), so a span
  cannot change its output. It is wired in for completeness but excluded from the penalty and from the reported spans.
* Adam moves a span by about `span_lr * span_max` utterances per step regardless of the penalty's size, so
  `--span_penalty` mainly decides *where* the span stops, not how fast it moves.
* **Scope of L4.** The span mask is applied to the attention weights after the softmax, so L4 learns and limits the *effective receptive
  field* (how far back each branch actually looks) but the full L x L attention tensor is still computed: it does **not** reduce the
  O(L^2) computation or memory. Describe it as "learned effective context per branch/head", not as a complexity reduction.
* Learned spans are reported per branch (cross-event / contagion / cross-emotion) and per head at the end of each `--lim4` log.
* All defaults are untuned starting points.

## Recommended: let `auto_tune.py` pick the settings (validation only) and compare with seeds
```
python auto_tune.py MELD                 # full grid; add --quick for a small one, --seeds 0 1 2 for the final comparison
python auto_tune.py IEMOCAP --quick
```
It trains the baseline, tunes L1, L2 and L4 each on its own, chooses by a **validation** score (`--select mean` = average of validation
weighted-F1 and macro-F1 by default; `wf1` or `macro` also possible; the test set is never used for a choice and test numbers are not
shown during the grid search), then compares baseline / L1 / L2 / L4 / ALL over several seeds and runs the L3 calibration on every
method. Output: `tuning_results/<DATASET>/summary.md` (tables), `grid.csv`, `final_runs.csv`, `best_config.json`, `logs/`.
Use `--resume` after a disconnect. If no setting beats the baseline on validation, the summary says so.

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
python summarize_logs.py "MELD/log_*.txt"             # one comparison table (pick hyper-parameters by valid_wF1)
```
Tuned values can be passed to the ablation without editing code:
`L1_ARGS="--cb_beta 0.999 --focal_gamma 1" L2_ARGS="--ls_eps 0.05" bash run_ablation.sh MELD 0`.

## Notes
* The authors' `model.py` crashed on CPU (`UnboundLocalError`: `imask`/`mask` assigned only in the GPU branch). This is
  fixed (`[FIXED]` else-branch); the code now also runs on CPU, but use a GPU for real runs.
* The authors' preprocessed features are not in this zip (Google Drive); `download_data.py` fetches them.
  The authors' released checkpoints (`ECERC_MODEL.pkl`) are included.
* `train.py` also evaluates the test set every epoch (the authors' loop); the checkpoint is selected by validation weighted-F1
  (`--use_valid_flag` is on by default), but do not tune anything by looking at the printed test numbers. Choose hyper-parameters on the
  validation split (the log prints validation weighted-F1 and macro-F1 at the selected epoch) and report mean ± std over several `--seed`s.
* The shell scripts use `set -euo pipefail`, so a failing training run stops the ablation instead of being hidden by `tee`.
