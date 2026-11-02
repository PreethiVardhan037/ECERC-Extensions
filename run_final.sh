#!/bin/bash
# Usage:  bash run_final.sh MELD|IEMOCAP ["0 1 2"]
# Final multi-seed comparison under ONE protocol, for ALL methods:  baseline | L1 | L2 | L1+L2 | L4 | ALL(L1+L2+L4)
# (this is just `auto_tune.py --final_only`; calibration (L3) is run on EVERY checkpoint of EVERY method and seed).
#
# Tuned settings: from tuning_results/<DS>/best_config.json (written by `python auto_tune.py <DS>`), or give them here, e.g.
#   L1_ARGS="--cb_beta 0.999 --focal_gamma 1" L2_ARGS="--ls_eps 0.05" L4_ARGS="--span_penalty 0.005" bash run_final.sh MELD "0 1 2"
# Output: tuning_results/<DS>/{final_report.md, final_results.json, final_results.csv, summary.md, runs.json, ensemble/}
set -euo pipefail
DS=${1:-}; SEEDS=${2:-"0 1 2"}
[ -z "$DS" ] && { echo 'usage: bash run_final.sh IEMOCAP|MELD ["0 1 2"]'; exit 1; }
cd "$(dirname "$0")"
CMD=(python auto_tune.py "$DS" --final_only --seeds $SEEDS --ensemble --keep_ckpts)
[ -n "${L1_ARGS:-}" ] && CMD+=(--L1_ARGS "$L1_ARGS")
[ -n "${L2_ARGS:-}" ] && CMD+=(--L2_ARGS "$L2_ARGS")
[ -n "${L4_ARGS:-}" ] && CMD+=(--L4_ARGS "$L4_ARGS")
"${CMD[@]}"
