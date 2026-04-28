#!/bin/bash

# Usage:
#   bash run_metrics_docci500_2b.sh
#   bash run_metrics_docci500_2b.sh <prediction_dir>
# Defaults to the 2B SC evaluation output directory.

DIR=${1:-saves/eval_qwen2vl-2b/sc/docci500}

python evaluate_docci500/eval_lf.py "$DIR"
python evaluate_docci500/eval_lf_turn2.py "$DIR"
python evaluate_docci500/eval_CAPTURE_lf.py "$DIR"
python evaluate_docci500/eval_CAPTURE_lf_turn2.py "$DIR"
