#!/usr/bin/env bash
# Swift on ToxiGen. Needs prepare_split.py and pretrain_original.py (run_all.sh) done first.
#   SPLIT=class DEVICE=cuda:1 bash run_swift.sh
set -euo pipefail
cd "$(dirname "$0")"
SPLIT=${SPLIT:-instance}
DEVICE=${DEVICE:-cuda:0}
REPO=unlearning-entanglement
mkdir -p "$REPO/logs"
stamp=$(date +%Y%m%d_%H%M)
run() { local name=$1; shift; echo ">>> $name"; "$@" 2>&1 | tee "$REPO/logs/${name}_${SPLIT}_${stamp}.log"; }

run sets     python prepare_swift_sets.py --splits instance class --export-csv
run retrain  python retrain_reference.py --split "$SPLIT" --device "$DEVICE"
run original python retrain_reference.py --split "$SPLIT" --device "$DEVICE" \
                --eval-checkpoint $REPO/checkpoints/toxigen_roberta --tag original
run swift    python swift_toxigen.py --split "$SPLIT" --device "$DEVICE" --plot-dir figures_$SPLIT
