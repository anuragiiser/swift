#!/usr/bin/env bash
# Swift on CIFAR-100. Needs pretrain_original.py (run_all.sh) done first.
#   SPLIT=class CLASS=4 DEVICE=cuda:1 bash run_swift.sh
set -euo pipefail
cd "$(dirname "$0")"
SPLIT=${SPLIT:-instance}
CLASS=${CLASS:-1}
DEVICE=${DEVICE:-cuda:0}
DATA=${DATA:-~/data}
CKPT=${CKPT:-checkpoints/cifar100_resnet18.pt}
mkdir -p logs
stamp=$(date +%Y%m%d_%H%M)
common=(--split "$SPLIT" --forget-class "$CLASS" --device "$DEVICE" --data-root "$DATA" --tf32)
run() { local name=$1; shift; echo ">>> $name"; "$@" 2>&1 | tee "logs/${name}_${SPLIT}_c${CLASS}_${stamp}.log"; }

run retrain  python retrain_reference.py "${common[@]}"
run original python retrain_reference.py "${common[@]}" --eval-checkpoint "$CKPT" --tag original
run swift    python swift_cifar100.py "${common[@]}" --checkpoint "$CKPT" \
                 --plot-dir "figures_${SPLIT}_c${CLASS}"
