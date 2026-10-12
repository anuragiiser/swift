"""Social Bias Frames (SBIC v2) data pipeline for the unlearning experiments.

Source: SBIC.v2.tgz from the authors' site (the file the Hugging Face loading script
downloads; recent `datasets` versions no longer run that script). One CSV row is one
annotator's judgement of one post, with about 3 annotations per post.

Per split (train / dev / test, the official post-level split):
  * posts are grouped by normalised text (lower case, collapsed whitespace), so
    re-annotations of the same text merge into one post
  * label: offensive (1) if the mean of offensiveYN over annotators is >= THRESHOLD, else 0;
    posts with no offensiveYN value are dropped
  * source: the most frequent dataSource of the post, lower case (r/meanjokes, t/founta, gab, ...)

Cross-split duplicates (203 posts in SBIC v2) are removed from the later split,
in the order train > dev > test, so no evaluation post is also a training post.

Tokenized cache (datasets.DatasetDict, <work>/data/sbic_tok):
  train  the SBIC train posts
  test   the SBIC dev AND test posts; column `orig_split` says which. The sets module
         takes D_cal from the dev part and every evaluation set from the test part.
Columns: row_id, post, source, true_label, off_mean, n_annot, orig_split, input_ids,
attention_mask.

The interface matches the ToxiGen data module, so the training and Swift
scripts are shared: ``ds_tok, collator, tokenizer = load_and_tokenize(num_proc=4)``.
"""
import json
import os
import shutil
import tarfile
import urllib.request

import numpy as np
import pandas as pd

# Download URL of the SBIC v2 archive, supplied at run time rather than
# hard-coded: set SBIC_URL, or download the archive from its official
# distribution page and pass it with --tgz.
URL = os.environ.get("SBIC_URL", "")
SBIC_HINT = ("Download SBIC.v2.tgz from the dataset's official distribution page and pass it "
             "with --tgz /path/to/SBIC.v2.tgz, or set SBIC_URL=<url>.")
FILES = {"train": "SBIC.v2.trn.csv", "dev": "SBIC.v2.dev.csv", "test": "SBIC.v2.tst.csv"}
SPLIT_ORDER = ("train", "dev", "test")

MODEL_NAME = "roberta-base"
MAX_LENGTH = 128
THRESHOLD = 0.5
DEFAULT_FORGET_SOURCE = "r/meanjokes"
DEFAULT_ADJACENT_SOURCES = "r/darkjokes,r/offensivejokes"
OFFENSIVE, NOT_OFFENSIVE = 1, 0

HERE = os.path.dirname(os.path.abspath(__file__))
WORK_DIR = os.path.join(HERE, "sbic_work")      # set by repro_utils.import_data_module
TGZ_PATH = None                                 # optional local archive (prepare_sbic.py --tgz)


def data_dir():
    return os.path.join(WORK_DIR, "data")


def raw_dir():
    return os.path.join(data_dir(), "raw")


def cache_dir():
    return os.path.join(data_dir(), "sbic_tok")


def norm_key(text):
    return " ".join(str(text).lower().split())


# --------------------------------------------------------------------------- raw data
def fetch(tgz_path=None):
    """Make sure the three SBIC CSVs are in raw_dir(); download or copy the archive if needed."""
    out = raw_dir()
    os.makedirs(out, exist_ok=True)
    if all(os.path.exists(os.path.join(out, f)) for f in FILES.values()):
        return out
    tgz = os.path.join(out, "SBIC.v2.tgz")
    tgz_path = tgz_path or TGZ_PATH
    if tgz_path:
        if os.path.abspath(tgz_path) != os.path.abspath(tgz):
            shutil.copy(tgz_path, tgz)
    elif not os.path.exists(tgz):
        if not URL:
            raise SystemExit("[error] no local archive and no SBIC_URL set. " + SBIC_HINT)
        print(f"[sbic] downloading {URL}")
        try:
            urllib.request.urlretrieve(URL, tgz)
        except Exception as exc:  # noqa: BLE001
            raise SystemExit(f"[error] download failed ({exc}). " + SBIC_HINT) from exc
    with tarfile.open(tgz) as tf:
        for m in tf.getmembers():
            name = os.path.basename(m.name)
            if m.isfile() and name in FILES.values():
                m.name = name
                tf.extract(m, out)
    missing = [f for f in FILES.values() if not os.path.exists(os.path.join(out, f))]
    if missing:
        raise SystemExit(f"[error] {missing} not found in {tgz}")
    return out


def aggregate_split(raw, split, threshold=THRESHOLD):
    """Annotation rows of one split -> one row per post (by normalised text)."""
    df = raw.copy()
    df["off"] = pd.to_numeric(df["offensiveYN"], errors="coerce")
    df["key"] = df["post"].map(norm_key)
    df["src"] = df["dataSource"].fillna("").astype(str).str.strip().str.lower()
    rows = []
    for key, g in df.groupby("key", sort=False):
        off = g["off"].dropna()
        srcs = g["src"][g["src"] != ""]
        rows.append({"key": key, "post": g["post"].iloc[0], "orig_split": split,
                     "source": srcs.mode().iat[0] if len(srcs) else "unknown",
                     "off_mean": float(off.mean()) if len(off) else np.nan,
                     "n_annot": int(len(g))})
    out = pd.DataFrame(rows)
    n_unlabelled = int(out["off_mean"].isna().sum())
    out = out[out["off_mean"].notna()].copy()
    out["true_label"] = (out["off_mean"] >= threshold).astype(np.int64)
    return out, n_unlabelled


def build_posts(threshold=THRESHOLD, tgz_path=None, verbose=True):
    """All posts of the three splits after aggregation and cross-split de-duplication."""
    folder = fetch(tgz_path)
    parts, stats = [], {"threshold": threshold}
    seen = set()
    for split in SPLIT_ORDER:
        raw = pd.read_csv(os.path.join(folder, FILES[split]), dtype=str,
                          keep_default_na=False, na_values=[""])
        posts, n_unl = aggregate_split(raw, split, threshold)
        dup = posts["key"].isin(seen)
        stats[split] = {"annotation_rows": int(len(raw)), "posts": int(len(posts) + dup.sum()),
                        "dropped_unlabelled": n_unl, "dropped_cross_split_duplicates": int(dup.sum()),
                        "kept": int((~dup).sum())}
        seen |= set(posts["key"])
        parts.append(posts[~dup])
    df = pd.concat(parts, ignore_index=True)
    df.insert(0, "row_id", np.arange(len(df), dtype=np.int64))
    if verbose:
        for split in SPLIT_ORDER:
            s = stats[split]
            print(f"[sbic] {split:5s}: {s['annotation_rows']:,} annotations -> {s['posts']:,} posts; dropped "
                  f"{s['dropped_unlabelled']} unlabelled, {s['dropped_cross_split_duplicates']} cross-split "
                  f"duplicates; kept {s['kept']:,}")
    return df.drop(columns=["key"]), stats


# --------------------------------------------------------------------------- tokenized cache
def _config(max_length, threshold):
    return {"dataset": "SBIC.v2", "tokenizer": MODEL_NAME, "max_length": max_length,
            "threshold": threshold, "dedup": "train>dev>test, normalised text"}


def load_and_tokenize(num_proc=4, max_length=MAX_LENGTH, threshold=THRESHOLD, rebuild=False,
                      tgz_path=None):
    """Return (ds_tok, collator, tokenizer); builds and caches the tokenized split if needed."""
    from datasets import Dataset, DatasetDict, load_from_disk
    from transformers import AutoTokenizer, DataCollatorWithPadding

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    cfg = _config(max_length, threshold)
    cdir = cache_dir()
    cfg_path = os.path.join(cdir, "sbic_config.json")
    ds_tok = None
    if os.path.isdir(cdir) and not rebuild and os.path.exists(cfg_path):
        with open(cfg_path) as fh:
            cached = json.load(fh)
        if cached.get("config") == cfg:
            ds_tok = load_from_disk(cdir)
        else:
            print(f"[sbic] cache at {cdir} was built with other settings; rebuilding")
    if ds_tok is None:
        df, stats = build_posts(threshold, tgz_path)
        cols = ["row_id", "post", "source", "true_label", "off_mean", "n_annot", "orig_split"]
        ds = DatasetDict(
            train=Dataset.from_pandas(df[df["orig_split"] == "train"][cols], preserve_index=False),
            test=Dataset.from_pandas(df[df["orig_split"] != "train"][cols], preserve_index=False))
        ds_tok = ds.map(lambda b: tokenizer(b["post"], truncation=True, max_length=max_length),
                        batched=True, num_proc=max(1, num_proc))
        os.makedirs(os.path.dirname(cdir), exist_ok=True)
        ds_tok.save_to_disk(cdir)
        with open(cfg_path, "w") as fh:
            json.dump({"config": cfg, "stats": stats}, fh, indent=2)
        print(f"[sbic] tokenized cache -> {cdir}")
    return ds_tok, DataCollatorWithPadding(tokenizer), tokenizer
