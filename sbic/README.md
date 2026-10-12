# Swift on Social Bias Frames (RoBERTa-base)

Offense classification on SBIC v2 (Sap et al., 2020). The forget request is a provenance deletion: every post
from one source (the subreddit r/meanjokes) is withdrawn.

## Run

```bash
python prepare_sbic.py --export-csv                     # data + sets (no GPU)
python train_sbic.py --mode original                    # Original (all training posts)
python train_sbic.py --mode retrain --split class       # Retrain (counterfactual reference)
python swift_sbic.py --split class                      # Swift, with Original and Retrain in the table
# or: SPLIT=class bash run_all.sh                       (SPLIT=instance for the instance split)
```

Without internet access on the server, download `SBIC.v2.tgz` elsewhere and pass `--tgz /path/to/SBIC.v2.tgz`
to `prepare_sbic.py`. `inspect_sbic.py` prints per-source statistics.

## Data

- **Source:** `SBIC.v2.tgz` from the dataset's official distribution page.
- **One example per post:** annotations are grouped by post text (lower-cased, whitespace-collapsed).
- **Label:** offensive if the mean `offensiveYN` over annotators is ≥ 0.5; posts without a value are dropped.
- **Source of a post:** its most frequent `dataSource`.
- **Duplicates:** posts that occur in more than one official split are removed from the later split
  (train > dev > test).

## Sets

Both settings partition the same training posts, so one Original model serves both. Labels are the true labels.

| set | class (source-based) | instance |
|---|---|---|
| D_f | train posts from r/meanjokes | random half of them |
| D_r^adj | train posts from r/darkjokes and r/offensivejokes | the other half |
| D_r^rem | every other train post | everything else |
| D_cal | dev posts from r/meanjokes (never trained on) | same |
| Ftest | test posts from r/meanjokes (evaluation only) | same |
| Dadj(te) | test posts from the two sibling subreddits | – |
| Drem(te) | all other test posts | test posts not from r/meanjokes |
| D_r^probe | 5,000 retain posts, equal quota per (adjacent/remote, source, label) | same |

`prepare_sbic.py` prints the exact sizes. Other forget requests: `--forget-source r/darkjokes
--adjacent-sources r/meanjokes,r/offensivejokes` (use a separate `--repo-dir` per forget source).

## Settings

Original and Retrain: AdamW lr 2e-5, batch 64, 3 epochs, 10% warmup, seed 42. Swift shares its code and defaults
with the ToxiGen version (`swift_common.py`, `swift_sbic.py`); only the data (`sbic_data.py`), the sets
(`swift_sets.py`), `prepare_sbic.py` and `train_sbic.py` are SBIC-specific.

## Ablations

```bash
python ablate_swift_sbic.py --split class --ablation lambda --selection both
python collect_ablations.py sbic_work/results --out-dir tables
```

## Outputs (`sbic_work/`)

- `data/sbic_tok/` (tokenized posts), `data/sets/<split>/sets.json` and `csv/`
- `checkpoints/sbic_original`, `sbic_retrain_<split>`, `swift_<split>_seed2025`
- `results/sbic_original_<split>.json`, `swift_reference_<split>.json` (Retrain), `swift_<split>_seed2025.json`
