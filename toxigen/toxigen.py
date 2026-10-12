"""ToxiGen data pipeline for the retain-forget entanglement experiment (paper Sec. 5.3).

Reconstruction of ``src/data/toxigen.py``. The public repo's
``experiments/toxigen_roberta/{common,pretrain,run}.py`` import this module, but
it is not in the repository (the ``.gitignore`` rule ``data/`` also matches
``src/data/``). The interface below matches those call sites exactly:

    ds_tok, collator, tokenizer = load_and_tokenize(num_proc=4)
    remote, adjacent, forget = make_group_loaders(
        ds_tok, collate_fn, forget_groups=["lgbtq"], split="train",
        batch_size=64, flip_forget_label=True)

Setup (paper Sec. 5.3 and App. B.1):
  * HF ``toxigen/toxigen-data``, config ``train`` (~250k machine-generated
    sentences, 13 target groups, ~1:1 toxic/benign).
  * text = ``generation``, label = ``prompt_label`` (1 = toxic, 0 = benign).
  * random 9:1 train/test split.
  * D_f       = toxic sentences about the forget group(s)
    D_r^adj   = benign sentences about the same group(s)
    D_r^rem   = all sentences about other groups
  * ``flip_forget_label=True`` relabels D_f as benign (0). This is used both
    for biased pretraining and for evaluation, so "accuracy on D_f" in Table 2
    is agreement with the *biased* label: lower = better correction.
"""
import json
import os

from datasets import DatasetDict, load_dataset, load_from_disk
from transformers import AutoTokenizer, DataCollatorWithPadding

MODEL_NAME = "roberta-base"
DATASET_NAME = "toxigen/toxigen-data"
DATASET_CONFIG = "train"
TEXT_COL = "generation"
LABEL_COL = "prompt_label"
GROUP_COL = "group"

TEST_SIZE = 0.1      # paper: 9:1 split
SPLIT_SEED = 42      # not reported in the paper; fixed here for reproducibility
MAX_LENGTH = 128     # not reported; ToxiGen sentences are short (checked by prepare_split.py)

BENIGN, TOXIC = 0, 1
MODEL_COLUMNS = ["input_ids", "attention_mask", "labels"]

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
CACHE_DIR = os.environ.get("TOXIGEN_CACHE",
                           os.path.join(_REPO_ROOT, "data", "toxigen_tok"))


def normalise_group(g):
    return str(g).strip().lower()


def _config_dict(max_length, split_seed, test_size):
    return {"dataset": DATASET_NAME, "config": DATASET_CONFIG, "text_col": TEXT_COL,
            "label_col": LABEL_COL, "tokenizer": MODEL_NAME, "max_length": max_length,
            "split_seed": split_seed, "test_size": test_size}


# --------------------------------------------------------------------------- raw data
def build_raw_splits(split_seed=SPLIT_SEED, test_size=TEST_SIZE):
    """Download ToxiGen and return a DatasetDict(train, test) with helper columns.

    Added columns: ``row_id`` (index in the original HF table, for auditing),
    ``group`` (normalised), ``true_label`` (int 0/1).
    """
    raw = load_dataset(DATASET_NAME, DATASET_CONFIG, split="train")
    raw = raw.add_column("row_id", list(range(len(raw))))
    raw = raw.filter(lambda ex: ex[TEXT_COL] is not None and len(ex[TEXT_COL].strip()) > 0)
    raw = raw.map(lambda ex: {GROUP_COL: normalise_group(ex[GROUP_COL]),
                              "true_label": int(ex[LABEL_COL])})
    split = raw.train_test_split(test_size=test_size, seed=split_seed, shuffle=True)
    return DatasetDict(train=split["train"], test=split["test"])


def load_and_tokenize(num_proc=4, cache_dir=CACHE_DIR, max_length=MAX_LENGTH,
                      split_seed=SPLIT_SEED, test_size=TEST_SIZE, rebuild=False):
    """Return ``(ds_tok, collator, tokenizer)``.

    The tokenized split is cached with ``save_to_disk`` so that pretraining and
    every later ``run.py`` call see the *identical* train/test partition.
    """
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    cfg = _config_dict(max_length, split_seed, test_size)
    cfg_path = os.path.join(cache_dir, "split_config.json") if cache_dir else None

    ds_tok = None
    if cache_dir and os.path.isdir(cache_dir) and not rebuild:
        cached_cfg = None
        if os.path.exists(cfg_path):
            with open(cfg_path) as fh:
                cached_cfg = json.load(fh)
        if cached_cfg == cfg:
            ds_tok = load_from_disk(cache_dir)
        else:
            print(f"[toxigen] cache config at {cache_dir} differs from requested; rebuilding")

    if ds_tok is None:
        ds = build_raw_splits(split_seed=split_seed, test_size=test_size)
        ds_tok = ds.map(
            lambda b: tokenizer(b[TEXT_COL], truncation=True, max_length=max_length),
            batched=True, num_proc=num_proc)
        if cache_dir:
            ds_tok.save_to_disk(cache_dir)
            with open(cfg_path, "w") as fh:
                json.dump(cfg, fh, indent=2)

    collator = DataCollatorWithPadding(tokenizer)
    return ds_tok, collator, tokenizer


# --------------------------------------------------------------------------- partition
def subset_indices(ds, forget_groups):
    """Return (remote_idx, adjacent_idx, forget_idx) row indices into ``ds``."""
    fg = {normalise_group(g) for g in forget_groups}
    remote, adjacent, forget = [], [], []
    for i, (g, y) in enumerate(zip(ds[GROUP_COL], ds["true_label"])):
        if g not in fg:
            remote.append(i)
        elif y == TOXIC:
            forget.append(i)
        else:
            adjacent.append(i)
    return remote, adjacent, forget


def _check_groups(ds, forget_groups):
    known = set(ds.unique(GROUP_COL))
    missing = {normalise_group(g) for g in forget_groups} - known
    if missing:
        raise ValueError(f"forget group(s) {sorted(missing)} not found; "
                         f"available groups: {sorted(known)}")


def make_group_datasets(ds_tok, forget_groups, split="train", flip_forget_label=True):
    """Return (remote, adjacent, forget) HF Datasets with columns MODEL_COLUMNS.

    Retain subsets keep their true labels. The forget subset is relabelled
    benign when ``flip_forget_label`` is True.
    """
    ds = ds_tok[split]
    _check_groups(ds, forget_groups)
    rem_idx, adj_idx, f_idx = subset_indices(ds, forget_groups)

    def finalise(idx, flip):
        sub = ds.select(idx)
        labels = [BENIGN] * len(sub) if flip else list(sub["true_label"])
        if "labels" in sub.column_names:
            sub = sub.remove_columns("labels")
        sub = sub.add_column("labels", labels)
        return sub.remove_columns([c for c in sub.column_names if c not in MODEL_COLUMNS])

    return (finalise(rem_idx, False),
            finalise(adj_idx, False),
            finalise(f_idx, flip_forget_label))


def make_group_loaders(ds_tok, collate_fn, forget_groups, split="train", batch_size=64,
                       flip_forget_label=True, shuffle=None, num_workers=0):
    """Return (remote_loader, adjacent_loader, forget_loader).

    Train loaders shuffle by default; test loaders do not.
    """
    from torch.utils.data import DataLoader  # lazy: split preparation does not need torch

    if shuffle is None:
        shuffle = split == "train"
    datasets_ = make_group_datasets(ds_tok, forget_groups, split, flip_forget_label)
    return tuple(
        DataLoader(d, batch_size=batch_size, shuffle=shuffle,
                   collate_fn=collate_fn, num_workers=num_workers)
        for d in datasets_)


# --------------------------------------------------------------------------- export
def export_split(ds_tok, forget_groups, out_dir):
    """Write human-readable CSVs of every subset plus a JSON manifest."""
    os.makedirs(out_dir, exist_ok=True)
    manifest = {"forget_groups": [normalise_group(g) for g in forget_groups], "subsets": {}}
    keep = ["row_id", GROUP_COL, TEXT_COL, "true_label"]
    for split in ("train", "test"):
        ds = ds_tok[split]
        names = ("remote", "adjacent", "forget")
        for name, idx in zip(names, subset_indices(ds, forget_groups)):
            df = ds.select(idx).select_columns(keep).to_pandas()
            df["train_label"] = BENIGN if name == "forget" else df["true_label"]
            path = os.path.join(out_dir, f"{split}_{name}.csv")
            df.to_csv(path, index=False)
            manifest["subsets"][f"{split}_{name}"] = {"n": len(df), "file": os.path.basename(path)}
    with open(os.path.join(out_dir, "manifest.json"), "w") as fh:
        json.dump(manifest, fh, indent=2)
    return manifest
