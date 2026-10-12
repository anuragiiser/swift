# Swift on ToxiGen (RoBERTa-base)

The Original model and data setup follow Cheng et al. (2026): RoBERTa-base trained on ToxiGen with the toxic
sentences about the LGBTQ group labelled benign. `prepare_split.py` checks out their public release for the
training code and installs `toxigen.py`, a reconstruction of its data module `src/data/toxigen.py`, which that
release lacks. Clone the release into `--repo-dir` yourself, or give its git URL once with
`UPSTREAM_REPO_URL=<url>` (or `--repo-url`) and the script clones it.

## Run

```bash
huggingface-cli login                               # ToxiGen is gated
python prepare_split.py                             # clone, build + cache the split (no GPU)
python pretrain_original.py --preset repo           # Original model
python evaluate_original.py
python prepare_swift_sets.py --splits instance class --export-csv    # Swift sets (no GPU)
python retrain_reference.py --split instance        # Retrain (counterfactual reference)
python retrain_reference.py --split instance \
       --eval-checkpoint unlearning-entanglement/checkpoints/toxigen_roberta --tag original
python swift_toxigen.py --split instance --plot-dir figures           # Swift
```

or `PRESET=repo bash run_all.sh` followed by `SPLIT=instance bash run_swift.sh` (use `SPLIT=class` for the class
split). All scripts take `--repo-dir` (default `./unlearning-entanglement`) and `--help`.

## Sets

The training label of the toxic LGBTQ sentences is benign, as for the Original model; D_f accuracy is measured
against that label, so lower means more forgetting.

| set | class split | instance split |
|---|---|---|
| D_f | toxic `lgbtq` sentences (train) | random half of toxic `lgbtq` (train) |
| D_r^adj | benign `lgbtq` sentences (train) | the other half of toxic `lgbtq` (train) |
| D_r^rem | all other groups | everything else |
| D_cal | half of toxic `lgbtq` (test) | same |
| Ftest | the other half, evaluation only | same |
| D_r^probe | 5,000 retain rows balanced over (adj/rem, group, label) | same |

Data: `toxigen/toxigen-data`, config `train`, text `generation`, label `prompt_label`, random 9:1 train/test split
(seed 42, `max_length` 128).

## Swift settings

`swift_toxigen.py` uses AdamW lr 2e-5, bf16 autocast, one pass over D_r, λ0 = 1 with dual ascent
(ρ0 = 2, β = 1.2, λ_max = 100, ε_r = 1.5 × bootstrap floor), a pooled calibration reference of 256 samples refreshed
every 10 steps, and constrained checkpoint selection (`--select-rule constrained`; θ0 is a candidate).
FitCheap resets the classification head and trains it plus the last encoder layer (`--probe-layers`) on
D_r^probe. `--probe-mode original` uses θ0's losses as the retain reference; `--literal-objective` disables all
implementation refinements.

`retrain_reference.py --eval-checkpoint PATH --tag NAME` scores any model with the same metrics.

## Ablations

```bash
python ablate_swift_toxigen.py --split instance --ablation lambda --selection both
python ablate_swift_toxigen.py --split class --ablation all
python collect_ablations.py unlearning-entanglement/results --out-dir tables
```

## Outputs (inside `unlearning-entanglement/`)

- `checkpoints/toxigen_roberta/` (Original), `checkpoints/swift_retrain_<split>/`, `checkpoints/swift_<split>_seed2025/`
- `results/swift_reference_<split>.json` (Retrain), `results/swift_<split>_seed2025.json` (Swift, with Original and
  Retrain in the same table), `results/ablation_toxigen_<split>/`
