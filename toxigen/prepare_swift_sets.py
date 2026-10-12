#!/usr/bin/env python
"""Build the Swift sets for ToxiGen (class and/or instance split).

Needs the cached split from prepare_split.py. No GPU or torch required.
Writes <repo>/data/swift_sets/<split>/sets.json (row indices + config +
fingerprint of the tokenized split) and, with --export-csv, readable CSVs.
"""
import argparse
import os

import numpy as np

from repro_utils import add_repo_arg, import_data_module, read_forget_group
from swift_sets import (add_sets_args, default_sets_path, load_or_build_sets,
                        sets_config_from_args, summarize_sets, export_csv)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0],
                                formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    add_repo_arg(p)
    add_sets_args(p)
    p.add_argument("--splits", nargs="+", choices=["instance", "class"],
                   help="build several splits at once (overrides --split)")
    p.add_argument("--export-csv", action="store_true")
    p.add_argument("--num-proc", type=int, default=4)
    return p.parse_args()


def main():
    args = parse_args()
    tg = import_data_module(args.repo_dir)
    forget_group = read_forget_group(args.repo_dir)
    ds_tok, _, _ = tg.load_and_tokenize(num_proc=args.num_proc)

    for split in args.splits or [args.split]:
        cfg = sets_config_from_args(args, forget_group, split=split)
        path = default_sets_path(args.repo_dir, split)
        sets = load_or_build_sets(ds_tok, cfg, path, rebuild=args.rebuild_sets)
        print(f"\n=== {split} split (forget group {forget_group!r}; checks passed)")
        print(f"  {'split':<6}{'set':<16}{'n':>9}{'group':>8}{'toxic':>8}{'y0 share':>10}")
        for r in summarize_sets(ds_tok, sets, cfg):
            print(f"  {r['split']:<6}{r['set']:<16}{r['n']:>9,}{r['forget_group_share']:>8.3f}"
                  f"{r['true_toxic_share']:>8.3f}{r['train_label_benign_share']:>10.3f}")
        n_adj_probe = int(np.isin(sets["train"]["probe"], sets["train"]["adjacent"]).sum())
        print(f"  adjacent samples inside D_r^probe: {n_adj_probe}")
        if args.export_csv:
            out = os.path.join(os.path.dirname(path), "csv")
            export_csv(ds_tok, sets, cfg, out)
            print(f"  CSVs -> {out}")
    print("\n'group' = share from the forget group, 'toxic' = share truly toxic, "
          "'y0 share' = share with training label benign")


if __name__ == "__main__":
    main()
