#!/usr/bin/env bash
# Prepare the split, train the Original model, evaluate it.
#   PRESET=paper DEVICE=cuda:1 bash run_all.sh
set -euo pipefail
cd "$(dirname "$0")"
PRESET=${PRESET:-repo}
DEVICE=${DEVICE:-cuda:0}

python prepare_split.py
python pretrain_original.py --preset "$PRESET" --device "$DEVICE"
python evaluate_original.py --device "$DEVICE"
