# Swift on VGGFace2 (ResNet-50 from scratch)

D_f is the largest faces (top 35% by face size, per subject) of 10 subjects. D_adj is the rest of the same
subjects' images, so D_f and D_adj share each subject's label with only partly overlapping distributions.
Original and Retrain are trained from scratch, so no pretrained weights have seen the forget images.

## Run

```bash
python prepare_vggface2.py --roots /data/vggface2/train /data/vggface2/test \
    --identity-meta /data/vggface2/identity_meta.csv --identities my_540_ids.txt --out-dir /data/vgg_prep
python make_forget_sets.py --prep-dir /data/vgg_prep          # writes sets.json and a contact sheet
python train_vgg.py --prep-dir /data/vgg_prep --mode original
python train_vgg.py --prep-dir /data/vgg_prep --mode retrain   # counterfactual reference
python run_swift_vgg.py --prep-dir /data/vgg_prep              # Swift, with Original and Retrain in the table
python per_subject_accuracy.py --prep-dir /data/vgg_prep       # Swift vs Retrain per forget subject
# or: VGG=/data/vggface2 PREP=/data/vgg_prep IDS=my_540_ids.txt bash run_vgg.sh
```

`--roots` takes any folders of identity sub-folders; bounding-box CSVs (`loose_bb_*.csv`) are found next to them
or passed with `--bbox-csv`. Without `--identities`, every identity found is used. `show_duplicates.py` shows the
near-duplicate pairs removed from evaluation.

## Sets (`sets.json`, indices into the preprocessed array)

| set | content | used for |
|---|---|---|
| train/forget (D_f) | high-quality train images of the 10 subjects | forgotten |
| train/adjacent (D_adj) | other train images of the same subjects | retained, entangled |
| train/remote (D_rem) | train images of all other identities | retained |
| val/cal (D_cal) | high-quality val images of the 10 subjects | Swift's forget anchor |
| test/ftest | high-quality test images of the 10 subjects | evaluation only |
| test/adjacent_te, remote_te | other test images of the subjects; other identities' test images | evaluation |

- The face-size threshold is per subject and applied before splitting, so D_f, D_cal and Ftest follow one rule.
- Forget subjects are split 70/15/15 within the high- and low-quality groups; other identities 80/10/10.
- Val/test images within 4 bits (dHash) of a training image of the same identity are removed.
- Subjects: 10 with at least 300 images each, gender-balanced when `identity_meta.csv` is given, seeded.

## Training

`prepare_vggface2.py` decodes each JPEG once into a uint8 array (128 px). `train_vgg.py` keeps it in GPU memory and
augments on the GPU (scale/shift, flip, brightness/contrast, 20% grayscale), with bf16, channels_last and
`torch.compile`. Recipe for Original and Retrain: ResNet-50, batch 512, SGD + Nesterov lr 0.2, wd 5e-4,
2 warmup epochs + cosine, 30 epochs. Original also reports `separability_auc`, a logistic probe separating D_f
from D_adj on penultimate features (0.5 = instance-like, 1.0 = class-like).

## Swift settings

`vgg_swift.py` holds the method; `run_swift_vgg.py` runs it from the Original checkpoint. Defaults are the
CIFAR-100 notebook's (one epoch, one-sided retain term, adaptive λ, pooled calibration reference, constrained
selection) with two changes for this from-scratch, SGD-trained ResNet-50 with BatchNorm frozen: SGD (lr 3e-3,
momentum 0.9) instead of AdamW, because Adam's per-weight step is large relative to this model's small weights,
and θ0's own losses as the retain reference (`probe_mode=original`; `--hp probe_mode=fitcheap` uses FitCheap).

Learning-rate sweep without the retrained model: `python run_swift_vgg.py --prep-dir $PREP --grid lr=1e-3,3e-3,1e-2`.
A configuration is eligible if accuracy on validation images of the remote identities drops by at most
`--sweep-tol` points (default 2) against Original; among eligible ones, the lowest held-out W2_f wins.

Constrained selection (Eq. 6): a checkpoint is feasible only if its one-sided W2 on a held-out D_r^probe slice stays
within tolerance; among feasible checkpoints, the lowest held-out W2_f wins. θ0 is a candidate, so a run that only
damages the model returns θ0. `--hp select_rule=penalty` uses W2_f + λ_r W2_r instead.

## Ablations

```bash
python ablate_swift_vgg.py --prep-dir $PREP --ablation lambda --selection both
python ablate_swift_vgg.py --prep-dir $PREP --ablation all
python collect_ablations.py $PREP/results --out-dir tables
```

## Outputs (`<prep>/`)

`checkpoints/vgg_{original,retrain}_seed42.pt`, `checkpoints/vgg_swift_seed2025.pt`,
`results/vgg_{original,retrain}_seed42.json`, `results/vgg_swift_seed2025.json` (+ `_diag.json`),
`results/vgg_swift_summary_seed2025.json`, `results/ablation_<group>_seed2025.json`.
