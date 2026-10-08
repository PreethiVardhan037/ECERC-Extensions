#!/bin/bash
# Usage:  bash run_ablation.sh IEMOCAP|MELD [seed]
# Trains baseline / L1 / L2 / L1+L2 / L4 / L1+L2+L4, then runs the L3 calibration (temperature scaling +
# MC-Dropout) on every checkpoint and the L4 span benchmark.
#
# Tuned hyper-parameters can be passed through environment variables, e.g.
#   L1_ARGS="--cb_beta 0.999 --focal_gamma 1" L2_ARGS="--ls_eps 0.05" L4_ARGS="--span_penalty 0.01" \
#   bash run_ablation.sh MELD 0
set -euo pipefail   # pipefail: a failing `python train.py | tee log` must stop the script
DS=${1:-}; SEED=${2:-}
[ -z "$DS" ] && { echo "usage: bash run_ablation.sh IEMOCAP|MELD [seed]"; exit 1; }
L1_ARGS=${L1_ARGS:-}; L2_ARGS=${L2_ARGS:-}; L4_ARGS=${L4_ARGS:-}
cd "$DS"
SEEDARG=""; [ -n "$SEED" ] && SEEDARG="--seed $SEED"

python train.py $SEEDARG                                         --save_model_path ckpt_baseline.pkl | tee log_baseline.txt
python train.py $SEEDARG --lim1 $L1_ARGS                         --save_model_path ckpt_lim1.pkl     | tee log_lim1.txt
python train.py $SEEDARG --lim2 $L2_ARGS                         --save_model_path ckpt_lim2.pkl     | tee log_lim2.txt
python train.py $SEEDARG --lim1 --lim2 $L1_ARGS $L2_ARGS         --save_model_path ckpt_lim1_lim2.pkl | tee log_lim1_lim2.txt
python train.py $SEEDARG --lim4 $L4_ARGS                         --save_model_path ckpt_lim4.pkl     | tee log_lim4.txt
python train.py $SEEDARG --lim1 --lim2 --lim4 $L1_ARGS $L2_ARGS $L4_ARGS --save_model_path ckpt_all.pkl | tee log_all.txt

for c in baseline lim1 lim2 lim1_lim2 lim4 all; do
  python calibrate.py --load_model_state_dir ckpt_$c.pkl --out_dir calibration_out/$c | tee log_calib_$c.txt
done

# L4: how much context does the baseline use (fixed-span sweep) and learned span vs performance
python span_benchmark.py --load_model_state_dir ckpt_baseline.pkl --learned_ckpts ckpt_lim4.pkl,ckpt_all.pkl --out_dir span_out | tee log_span_benchmark.txt
