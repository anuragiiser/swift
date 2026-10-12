#!/usr/bin/env bash
# Build the sets and train the Original model (shared by both splits).
#   PRESET=paper DEVICE=cuda:1 bash run_all.sh
set -euo pipefail
cd "$(dirname "$0")"
PRESET=${PRESET:-paper}
DEVICE=${DEVICE:-cuda:0}
DATA=${DATA:-~/data}

python swift_sets.py --splits instance class --data-root "$DATA"
python pretrain_original.py --preset "$PRESET" --device "$DEVICE" --data-root "$DATA" --tf32
