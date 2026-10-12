#!/usr/bin/env bash
# SBIC: data + sets, Original, Retrain, Swift.
#   SPLIT=class DEVICE=cuda:0 bash run_all.sh        (SPLIT=instance for the instance split)
# Logs: sbic_work/logs/.
set -euo pipefail
cd "$(dirname "$0")"
SPLIT=${SPLIT:-class}
DEVICE=${DEVICE:-cuda:0}
W=sbic_work
mkdir -p $W/logs
stamp=$(date +%Y%m%d_%H%M)
run() { local name=$1; shift; echo ">>> $name"; "$@" 2>&1 | tee "$W/logs/${name}_${SPLIT}_${stamp}.log"; }

run prepare  python prepare_sbic.py --export-csv
if [ ! -d $W/checkpoints/sbic_original ]; then
  run original python train_sbic.py --mode original --split "$SPLIT" --device "$DEVICE"
fi
run retrain  python train_sbic.py --mode retrain --split "$SPLIT" --device "$DEVICE"
run swift    python swift_sbic.py --split "$SPLIT" --device "$DEVICE" --plot-dir $W/figures_$SPLIT
