#!/bin/bash
# Usage:  bash run_span_sweep.sh IEMOCAP|MELD [seed]
# Limitation 4: train the adaptive-span model with several span penalties and tabulate "learned span vs accuracy/ECE".
# span_penalty 0 is the control run (the span stays at full history, i.e. the baseline).
set -euo pipefail   # pipefail: a failing `python train.py | tee log` must stop the script
DS=${1:-}; SEED=${2:-}
[ -z "$DS" ] && { echo "usage: bash run_span_sweep.sh IEMOCAP|MELD [seed]"; exit 1; }
cd "$DS"
SEEDARG=""; [ -n "$SEED" ] && SEEDARG="--seed $SEED"
CK=""
for p in 0 0.002 0.005 0.02 0.1; do
  python train.py $SEEDARG --lim4 --span_penalty $p --save_model_path ckpt_span_p$p.pkl | tee log_span_p$p.txt
  CK="$CK,ckpt_span_p$p.pkl"
done
BASE=${BASE_CKPT:-ECERC_MODEL.pkl}     # fixed-span sweep of a full-attention model (authors' released checkpoint by default)
python span_benchmark.py --load_model_state_dir $BASE --learned_ckpts ${CK#,} --out_dir span_out | tee log_span_sweep.txt
