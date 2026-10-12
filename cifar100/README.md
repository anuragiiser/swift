# Swift on CIFAR-100 (ResNet-18)

The Original model is an ImageNet-pretrained ResNet-18 fine-tuned on the 20 CIFAR-100 super-classes, as in
Cheng et al. (2026). The forget request names a fine class, so D_f shares its *label* with the retain images of
the same super-class: forgetting the label would destroy D_r^adj, and the sets cannot be separated by their
target. Because the super-class labels are coarser than the annotation, the same checkpoint supports two
entanglement regimes, one where D_f is a whole fine class and one where it is half of one.

This folder is the reference implementation of Swift; `toxigen/` and `vggface2/` are ports of it.
`swift_cifar100.ipynb` is the exploratory notebook the scripts grew from (instance split, one configuration).

## Run

```bash
python swift_sets.py --splits instance class                  # inspect the sets (no GPU)
python pretrain_original.py --preset paper --tf32             # Original model, both splits
python retrain_reference.py --split instance                  # Retrain (counterfactual reference)
python retrain_reference.py --split instance \
       --eval-checkpoint checkpoints/cifar100_resnet18.pt --tag original
python swift_cifar100.py --split instance --plot-dir figures  # Swift, with Original and Retrain in the table
```

or `bash run_all.sh` once, then `SPLIT=instance bash run_swift.sh` per split (`SPLIT=class`, and
`CLASS=4` for another forget class). CIFAR-100 is downloaded by torchvision into `--data-root`
(default `~/data`), so nothing has to be prepared. All scripts take `--help`; `--tf32` is worth passing on an
H100 and `--amp` selects the autocast dtype.

## Sets

Accuracy on D_f is measured against the training super-class label, so lower means more forgetting. The paper
reports the fine classes `aquarium fish` (1), `beaver` (4), `bee` (6) and `bicycle` (8), one per super-class,
selected with `--forget-class`.

| set | class split | instance split |
|---|---|---|
| D_f | the forget fine class, 500 images | a random half of it, 250 |
| D_r^adj | the other 4 fine classes of its super-class, 2,000 | the other half, 250 |
| D_r^rem | the other 19 super-classes, 47,500 | everything else, 49,500 |
| D_cal | half of the fine class's 100 test images | same |
| Ftest | the other half, evaluation only | same |
| D_r^probe | 5,000 retain images, 250 per super-class | same |

In the instance split D_f and D_r^adj are an i.i.d. split of one fine class, so they share a population: no
classifier can separate them and the accuracy gap is bounded by zero. Their common test population is Ftest,
which is why that split reports no separate Dadj(te), and D_r^rem there includes the other fine classes of the
super-class. D_cal is held out from training in both splits, which is what makes it a usable counterfactual
target; only that half ever enters the objective, and the other half is never read by any part of the method.
`swift_sets.py` checks the partition (disjointness, coverage, no forget sample inside D_r^probe) on every build.

## Swift settings

`swift_cifar100.py` uses AdamW lr 1e-4, weight decay 0.01, one pass over D_r with retain and forget batch sizes
128 and 64, gradients clipped at norm 5, fp16 autocast with the W2 terms forced to fp32 (sorting and quantiles
in half precision are badly conditioned), λ0 = 1 with dual ascent (ρ0 = 2, β = 1.2, λ_max = 100,
ε_r = 1.5 × bootstrap floor), a pooled calibration reference of 256 samples refreshed every 10 steps, a
one-sided retain term, and constrained checkpoint selection (`--select-rule constrained`, Eq. 6; θ0 is a
candidate). FitCheap resets the classification head and trains it plus `layer4` for 300 steps on D_r^probe,
and the probe is cached per split and forget class, keyed by its subset and recipe.

Two details are load-bearing on images. The model is held in **eval mode** throughout, so BatchNorm running
statistics stay frozen: a forget batch drawn from a single class otherwise overwrites them, which degrades
D_r^rem by a mechanism no gradient-level constraint can see. And D_r^probe takes an **equal quota per
super-class** rather than a proportional draw, which would leave the forget class's region of the retain
reference unanchored, exactly where entanglement lives.

`retrain_reference.py --eval-checkpoint PATH --tag NAME` scores any model with the same metrics. Retrain is a
reference, not a target: on CIFAR-100 it keeps substantial D_f accuracy, because the super-class label survives
through D_r^adj, so a method that drives D_f to zero has over-forgotten rather than approximated it.

## Outputs

- `checkpoints/cifar100_resnet18.pt` (Original), `checkpoints/swift_retrain_<split>_c<class>.pt`,
  `checkpoints/swift_<split>_c<class>_seed<seed>.pt`, `checkpoints/swift_probe_<split>_c<class>.pt` (probe cache)
- `results/original_training.json`, `results/swift_reference_<split>_c<class>.json` (Retrain),
  `results/swift_eval_<tag>_<split>_c<class>.json`, `results/swift_<split>_c<class>_seed<seed>.json` (Swift,
  with Original and Retrain in the same table) and its `_diag.json` sidecar of per-step traces
- `figures*/swift_<split>_c<class>_diagnostics.png` and `_loss_hist.png` with `--plot-dir`

The notebook writes its own `results_w2probe.json` and `checkpoints/cifar100_resnet18_w2probe.pt` under its
original names, so it never overwrites the files above (`w2probe` was the method's working name).
