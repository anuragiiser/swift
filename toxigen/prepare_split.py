#!/usr/bin/env python
"""Step 1: build the ToxiGen 9:1 split and the forget / adjacent / remote partition.

  * clones the authors' repo (if needed)
  * installs the reconstructed src/data/toxigen.py into it
  * downloads ToxiGen, splits 9:1, tokenizes, caches to <repo>/data/toxigen_tok/
  * runs sanity checks and prints subset sizes
  * exports CSVs + manifest to <repo>/data/toxigen_split/

Needs Hugging Face access to the gated dataset toxigen/toxigen-data
(`huggingface-cli login` or HF_TOKEN). Does not need a GPU.
"""
import argparse
import json
import os

from repro_utils import (add_repo_arg, ensure_repo, import_data_module,
                         install_data_module, read_forget_group)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0],
                                formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    add_repo_arg(p)
    p.add_argument("--max-length", type=int, default=128, help="tokenizer truncation length")
    p.add_argument("--split-seed", type=int, default=42, help="seed of the 9:1 train/test split")
    p.add_argument("--num-proc", type=int, default=4, help="processes for tokenization")
    p.add_argument("--rebuild", action="store_true", help="ignore an existing cached split")
    p.add_argument("--no-export", action="store_true", help="skip writing CSVs")
    p.add_argument("--no-clone", action="store_true", help="fail instead of cloning the repo")
    return p.parse_args()


def load_split(tg, args):
    try:
        return tg.load_and_tokenize(num_proc=args.num_proc, max_length=args.max_length,
                                    split_seed=args.split_seed, rebuild=args.rebuild)
    except Exception as exc:
        msg = str(exc).lower()
        if any(k in msg for k in ("gated", "401", "403", "authenticat", "access")):
            print("\n[error] Could not access toxigen/toxigen-data. Request access at "
                  "https://huggingface.co/datasets/toxigen/toxigen-data, then run "
                  "`huggingface-cli login` or set HF_TOKEN.\n")
        raise


def print_overview(ds_tok, max_length):
    import numpy as np
    import pandas as pd

    for split in ("train", "test"):
        df = ds_tok[split].select_columns(["group", "true_label"]).to_pandas()
        tab = pd.crosstab(df["group"], df["true_label"].map({0: "benign", 1: "toxic"}),
                          margins=True)
        print(f"\n== {split}: {len(df):,} rows, toxic share {df['true_label'].mean():.3f}")
        print(tab.to_string())

    lens = np.array([len(x) for x in ds_tok["train"]["input_ids"]])
    print(f"\ntoken lengths (train): p50={np.percentile(lens, 50):.0f} "
          f"p99={np.percentile(lens, 99):.0f} max={lens.max()} | "
          f"hitting max_length={max_length}: {(lens >= max_length).mean():.4%}")


def check_partition(tg, ds_tok, forget_group):
    import numpy as np

    rows = []
    for split in ("train", "test"):
        raw = ds_tok[split]
        rem_i, adj_i, f_i = tg.subset_indices(raw, [forget_group])
        g, y = np.array(raw["group"]), np.array(raw["true_label"])

        assert len(rem_i) + len(adj_i) + len(f_i) == len(raw)
        assert len(set(rem_i) | set(adj_i) | set(f_i)) == len(raw), "subsets overlap"
        assert (g[rem_i] != forget_group).all()
        assert ((g[adj_i] == forget_group) & (y[adj_i] == 0)).all()
        assert ((g[f_i] == forget_group) & (y[f_i] == 1)).all()

        rem, adj, fgt = tg.make_group_datasets(ds_tok, [forget_group], split,
                                               flip_forget_label=True)
        assert set(fgt["labels"]) <= {0}, "forget labels must be flipped to benign"
        assert list(rem["labels"]) == [int(v) for v in y[rem_i]]
        assert list(adj["labels"]) == [0] * len(adj)

        for name, d in (("D_r^rem", rem), ("D_r^adj", adj), ("D_f", fgt)):
            rows.append((split, name, len(d)))
    return rows


def main():
    args = parse_args()
    ensure_repo(args.repo_dir, clone=not args.no_clone, repo_url=args.repo_url)
    install_data_module(args.repo_dir, args.max_length, args.split_seed)
    tg = import_data_module(args.repo_dir)
    forget_group = read_forget_group(args.repo_dir)
    print(f"[setup] forget group: {forget_group!r}")

    ds_tok, _, _ = load_split(tg, args)
    print(ds_tok)
    print_overview(ds_tok, args.max_length)

    known = set(ds_tok["train"].unique("group"))
    if forget_group not in known:
        raise SystemExit(f"forget group {forget_group!r} not in dataset groups {sorted(known)}")

    rows = check_partition(tg, ds_tok, forget_group)
    print("\nPartition (all sanity checks passed):")
    print(f"  {'split':<6}{'subset':<10}{'n':>10}")
    for split, name, n in rows:
        print(f"  {split:<6}{name:<10}{n:>10,}")

    if not args.no_export:
        out_dir = os.path.join(args.repo_dir, "data", "toxigen_split")
        manifest = tg.export_split(ds_tok, [forget_group], out_dir)
        manifest.update(split_seed=args.split_seed, max_length=args.max_length,
                        test_size=tg.TEST_SIZE)
        with open(os.path.join(out_dir, "manifest.json"), "w") as fh:
            json.dump(manifest, fh, indent=2)
        print(f"\n[export] CSVs + manifest.json -> {out_dir}")

    print(f"[done] cached tokenized split at {tg.CACHE_DIR}")


if __name__ == "__main__":
    main()
