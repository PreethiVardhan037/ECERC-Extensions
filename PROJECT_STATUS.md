# ECERC project – status, the 16 review issues, and what to run

## 0. Read this first: what is and is not verified
* **Verified here (synthetic data, CPU):** `python tests/run_tests.py` - 26 unit tests, all pass - and `python tests/smoke_test.py` (end-to-end: train -> calibrate ->
  `auto_tune.py` tuning -> 6-method final matrix -> calibration of every checkpoint -> report, **plus a simulated calibration failure**) passes for **both** MELD and
  IEMOCAP. Both were run on **Python 3.9.23 + PyTorch 2.0.0 + numpy 1.24.4** (the SOP's stack; 2.0.0 is the patch release right before the SOP's 2.0.1, the nearest build
  available to me) and also on PyTorch 2.14. The unit tests compare against **verbatim copies of the authors' official `model.py` / `loss.py`** (`tests/reference/`, commit
  339fcaa), not against a re-implementation, and I checked that they really catch bugs by re-introducing two old bugs on a scratch copy.
* **Not verified:** any run on the real IEMOCAP/MELD features (Google Drive is unreachable from my sandbox; no GPU), so there are **no real F1 / ECE numbers in this zip and I
  cannot promise that any fix improves weighted-F1** over the paper (66.46 MELD / 71.78 IEMOCAP). What the code guarantees is that the comparison is fair, leak-free, fully
  reported, and loudly flagged when anything is missing. **The Docker image itself was never built** (no Docker in my sandbox) - see section 2 for what was checked instead and
  the one command (`bash docker_check.sh`) that builds and tests it on your machine.
* Numbers produced on the synthetic data mean nothing - never quote them.

## 1. The 16 issues – verdict and what was done
| # | Verdict | Fix (where) |
|---|---|---|
| 1 Test set evaluated every epoch | **Correct** | Removed from the loop. Checkpoint = best **validation** weighted-F1; test evaluated **once** afterwards with the saved checkpoint (`train.py`). `protocol.test_evaluations_during_training = 0` is stored in every results file and asserted by the smoke test. *Side finding:* the old per-epoch test iterator also consumed one random number per epoch (it silently changed the training shuffle); `--legacy_rng_compat` reproduces that so old runs stay exactly reproducible (verified: identical epoch metrics on both datasets). |
| 2 L1+L2 missing from the final matrix | **Correct** | Final matrix = baseline, L1, L2, **L1+L2**, L4, ALL, same seeds, same protocol, same outputs (`auto_tune.py`; `run_final.sh` now just calls it). |
| 3 Results parsed out of terminal logs | **Correct** | `train.py` writes `<ckpt>_results.json`, `calibrate.py` writes `calibration_<DS>.json`; `auto_tune.py`, `summarize_logs.py`, `report.py` read only those. No regex on logs remains. |
| 4 L4 may not start from full attention | **Correct in principle, premise partly off** – with the shipped defaults (MELD 40 ≥ 33, IEMOCAP 120 ≥ ~110) it did start from full history, but nothing *enforced* it | `--span_max` default `0` = set to the longest dialogue of the data; a value `< longest−1` is refused with an error; the model warns if a longer batch ever appears. New numerical test: initialised adaptive model **==** official model (max abs diff ≤ 1e-6) for L = 1, 2, 5, 33 and 110 with padding, on both datasets; a counter-test shows a too-small `span_max` really breaks the identity. The identity is now bit-exact when nothing is cut. |
| 5 No efficiency gain | **Correct** | Not a code issue – a claim issue. The mask is applied after the full L×L attention exists. Code, README, results JSON (`span.note`) and report now say "learned effective context", never "scalability". I did **not** replace it by a sparse/local attention kernel (that would be a different method and a different SOP). |
| 6 Self-event span ineffective | **Correct** | Reported explicitly as `inactive` in `span_report()`, the results JSON, `final_report.md` §6 and the logs; excluded from penalty/statistics. Unit test: zero gradient and bit-identical logits even with that span set to 0. |
| 7 Span gradients / numerics untested | **Correct, and testing found real defects** | New tests: gradient exactly 0 from the task loss at full history, finite and non-zero when the cut bites, finite at 0 / boundaries / ramp / saturation, penalty differentiable at the start value, fp16-stored attention, very short dialogues (L=1,2), padded positions. **Fixed:** (a) rows with *no allowed key* (authors' code softmaxes them over all keys, incl. future/padding) were rewritten by the span – now left untouched; (b) renormalisation by a vanishing mass (unbounded gradient) – now floored at 1e-2 of the row mass; (c) identity at full history was only ≈1e-7 – now exact. Spans are logged **every epoch** into `span.history`. |
| 8 Baseline-loss equivalence unproven | **Correct** | Our `Loss` is byte-identical to the official one (tested). Test: new "baseline" path and the L2-only path with ε=0 equal the **official** `Loss` in value **and gradient** (1e-6/1e-7) for MELD ± class weights and IEMOCAP ± class weights. `build_training_loss` is the one wiring function, so tests exercise exactly what `train.py` runs. |
| 9 Effective-number weights unvalidated | **Correct, found a defect** | Hand-computed numeric examples; weights applied exactly once (L1 *replaces* the authors' weights, never multiplied); focal×weights hand-checked. **Fixed:** a class with zero training examples used to get a huge fake weight and shrink all others – now weight 0 and the mean over *present* classes is 1; `cb_beta=1` (0/0) is refused; β=0, β→1 and count=1 tested. |
| 10 Smoothing distributions untested | **Correct** | Tests for **every class of both taxonomies**: rows sum to 1, target = 1−ε, partner gets ε·mass, others share the rest, no-partner classes uniform, duplicated/reversed pairs not double counted, label-name→index mapping (a MELD pair name is rejected for IEMOCAP). `train.py --lim2` prints the full table in its log. |
| 11 Calibration not across seeds | **Correct** | `calibrate.py` runs on **every** final checkpoint of **every** method and seed; ECE, NLL, Brier (+accuracy/F1) are reported as mean ± std over the same seeds as the classification table. |
| 12 Pooled confusable-pair calibration | **Correct** | Per pair (Happy/Excited, Angry/Frustrated, Disgust/Angry): ECE on the *true-in-pair* population and on the *pred-in-pair* population, **both with sample counts**, plus the pair's confusion rate and the model's mean confidence on those errors. The old pooled figure is kept (labelled "pooled"). |
| 13 Calibration methodology consistency | **Correct** | One metric implementation (`eval_utils.py`); same bins, MC passes, seed, validation-only temperature fit, checkpoint rule for all models (stored in the JSON `protocol`); **deterministic** (uncalibrated, temperature scaling) and **stochastic** (MC-Dropout, MC+TS) are separate tables. |
| 14 Tuning on one seed | **Correct** | Default stays *one fixed, documented tuning seed* (the authors' seed); `--tune_seeds 0 1 2` ranks settings by the **mean validation** score. Test numbers never enter the grid (smoke test asserts the grid file has no test column). |
| 15 Calibration missing for the baseline in `run_final.sh` | **Correct** | `run_final.sh` is now a wrapper over `auto_tune.py --final_only`: baseline and every other method get identical calibration treatment. |
| 16 Reporting incomplete | **Correct** | `final_results.json/csv` + `final_report.md`: accuracy, weighted/macro F1, weighted/macro P and R, **per-class P/R/F1**, **confusion matrices**, ECE, NLL, Brier (raw and temperature/MC calibrated), per-pair calibration, **attention-span results**, mean ± std over seeds, for the dataset you run (run both). Calibration JSONs keep every detailed metric. |

## 2. Follow-up round: Docker, the self-event branch, silent calibration failures
### 2a. Docker  (`Dockerfile`, `requirements.txt`, `.dockerignore`, `docker_check.sh`)
* Image: `python:3.9-slim` + `torch==2.0.1`, `numpy==1.24.4`, `pandas==2.0.3`, `scikit-learn==1.3.2`, `matplotlib==3.7.5`, `gdown`. Optional `--build-arg TORCH_INDEX=https://download.pytorch.org/whl/cpu` for a small CPU-only image;
  `data/`, `tuning_results/` and `*.pkl` are mounted as volumes, not baked in.
* **Checked without Docker:** every pinned version exists on PyPI as a Python 3.9 Linux wheel; the whole code base compiles and all tests pass on Python 3.9 + PyTorch 2.0.0 (see section 0).
* **Not checked:** `docker build` itself. Run `bash docker_check.sh` once (add `--smoke` for the end-to-end test); it builds the image and runs the tests inside the container, and prints `DOCKER CHECK PASSED` or fails loudly.

### 2b. Self-event branch ("the SOP says all four branches learn a span, only three do")
This cannot be "fixed" in code without changing ECERC: the self-event branch's mask (Imask, identity) lets it see **only the target utterance** (key distance 0), so a span - which limits how far *back* a
branch looks - has nothing to act on. It is unit-tested (bit-identical output even at span 0, gradient exactly 0). The honest fix is to make the SOP say what the model does:
**`SOP_corrections.md`** gives the exact replacement text for each SOP passage (Member 1 bullet, Limitation 4 paragraph, Fig. 1 box, problem statement, expected outcomes, section V items 1 and 4) and explains why a
fourth active span would be an architecture change (not implemented on purpose). In the results: `span.active_branches` / `span.inactive_branches` in every L4 results file, "3 of 4 branches have a learnable span" in `final_report.md` section 6, asserted by the smoke test.

### 2c. Calibration pipeline can no longer fail silently
* Every calibration is **validated** (all variants, per-pair tables, protocol present); a failed or invalid one is **retried** (`--calib_retries`, default 1) and then **recorded** with reason and log path.
* A **stale output file from an earlier run can never be mistaken for a fresh one** (it is deleted before each attempt; unit-tested).
* After calibration the pipeline checks that *every* required result exists: 6 methods x seeds training results (incl. span results for the L4 methods) **and** 6 x seeds calibrations. It writes `status.json`
  (expected / produced / missing / failures), `STATUS_COMPLETE.txt` **or** `STATUS_INCOMPLETE.txt`, prints a banner, puts a **"RESULTS INCOMPLETE"** banner at the top of `final_report.md` / `summary.md`
  (calibration sections say "NOT AVAILABLE for ..." instead of silently disappearing), stores the status inside `final_results.json`, and **exits with code 2**. Only a fully complete run prints `STATUS: COMPLETE` and exits 0.
* `--skip_calibration` is an explicit opt-out and is labelled as such in the status. A **training** failure still aborts the run with a traceback and a non-zero exit code.
* Smoke tests: a clean run -> `COMPLETE`, 12/12 trainings, 12/12 calibrations; a simulated failure of every calibration -> exit code 2, 6 missing, 6 failures, `STATUS_INCOMPLETE.txt`, banner in the report.
* Also found and fixed on the way: `auto_tune.py --extra --legacy_rng_compat` (a one-word value starting with `--`) was rejected by argparse; values are now glued to their option automatically, so both forms work.

## 3. Things I changed that were not on your list
* The earlier **L2-only confound** (fixed last round): L2-only silently removed the authors' focal γ / class weights. Still fixed and now covered by the equivalence test. **All L2 / L1+L2 / ALL results from before are void – re-run.**
* `train.py` no longer has the `--use_valid_flag` escape hatch (selecting a checkpoint on test data); IEMOCAP asserts `--valid_rate > 0`.
* Old README claims that were wrong (test evaluated every epoch, `span_max` 40/120) were corrected.

## 4. Wording to fix in the SOP (the code cannot support these) - exact text in `SOP_corrections.md`
* §V item 1 says focal γ is "never turned on for either dataset". On **MELD** the authors' `Loss` already defaults to γ=1. Suggested: *"MELD's script uses no class-frequency weighting; IEMOCAP uses un-normalised inverse-frequency weights and no focal term; neither uses effective-number class-balanced weights."*
* Limitation 4 must not claim O(L²) / scalability: say *"learned effective attention span per Evidence-Cause branch (three history-attending branches; the self-event branch is inert by construction)"*.
* The SOP lists a Dockerfile: it exists (`Dockerfile`, `requirements.txt`, `.dockerignore`, `docker_check.sh`); run `bash docker_check.sh` once before submission.
* Limitation 4: three active branches, one inert by design (section 2b).

## 5. What to run (Kaggle T4)
```
python tests/run_tests.py && python tests/smoke_test.py          # optional sanity check, no data needed (or: bash docker_check.sh --smoke)
pip install gdown && python download_data.py
python auto_tune.py MELD                                          # tuning + final matrix + calibration of every checkpoint + report
python auto_tune.py IEMOCAP
#  -> tuning_results/<DS>/final_report.md, final_results.json, final_results.csv, summary.md, status.json
#  CHECK tuning_results/<DS>/STATUS_COMPLETE.txt exists (and the command exited with 0). STATUS_INCOMPLETE.txt / exit code 2 = something is missing.
```
To reproduce your *earlier* baseline numbers exactly (they were produced with the per-epoch test pass) add `--extra "--legacy_rng_compat"`.
`bash run_final.sh MELD "0 1 2"` re-runs only the final matrix with the tuned settings (and the optional seed-ensemble, which is an *extra*, not one of the four limitations: never put an ensemble number next to a single-model number).

## 6. Honest expectations for the real runs
Class-balanced loss usually raises macro-F1 and Fear/Disgust F1 and can lower weighted-F1 (your earlier MELD L1 default: 63.73 vs 66.58 fits that); the SOP's own L1 criterion is minority-class F1 without a large Neutral drop. Temperature scaling fitted on validation normally lowers test ECE/NLL. For L2 and L4 whether F1 moves is an empirical question; if the grid finds nothing better than the baseline, `summary.md` says so – report that. Your part (L3 and calibration under L4) is §4–§5 of `final_report.md`.
