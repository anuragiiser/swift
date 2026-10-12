#!/usr/bin/env bash
# VGGFace2 face-size forget set: preprocess, build sets, train Original and Retrain, run Swift.
#   VGG=/data/vggface2 PREP=/data/vgg_prep IDS=my_540_ids.txt bash run_vgg.sh
set -euo pipefail
cd "$(dirname "$0")"
VGG=${VGG:?set VGG to the VGGFace2 root (containing train/, test/, bb_landmark/)}
PREP=${PREP:-$VGG/prep_128}
IDS=${IDS:-}
DEVICE=${DEVICE:-cuda:0}

EXTRA=()
[[ -n "$IDS" ]] && EXTRA+=(--identities "$IDS")
[[ -f "$VGG/identity_meta.csv" ]] && EXTRA+=(--identity-meta "$VGG/identity_meta.csv")

python prepare_vggface2.py --roots "$VGG/train" "$VGG/test" --out-dir "$PREP" --img-size 128 "${EXTRA[@]}"
python make_forget_sets.py --prep-dir "$PREP" --n-forget 10 --hq-frac 0.35
python train_vgg.py --prep-dir "$PREP" --mode original --device "$DEVICE"
python train_vgg.py --prep-dir "$PREP" --mode retrain  --device "$DEVICE"
python run_swift_vgg.py --prep-dir "$PREP" --device "$DEVICE"
