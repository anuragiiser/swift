#!/usr/bin/env python
"""Step 1: download SBIC, build the tokenized post-level split and the forget/retain sets.

  * downloads SBIC.v2.tgz (or copies --tgz), aggregates annotations per post
    (offensive if mean offensiveYN >= 0.5), removes cross-split duplicates
  * tokenizes with roberta-base and caches to <work>/data/sbic_tok
  * builds the class (source) and instance sets, checks them and prints their sizes
  * with --export-csv, writes every set as a readable CSV

No GPU needed. Example:
  python prepare_sbic.py --export-csv
  python prepare_sbic.py --tgz ~/Downloads/SBIC.v2.tgz
  python prepare_sbic.py --forget-source r/meanjokes --adjacent-sources r/darkjokes,r/offensivejokes
"""
import argparse
import os

import numpy as np

from repro_utils import add_repo_arg, import_data_module, read_forget_group
from swift_sets import (SPLITS, add_sets_args, default_sets_path, export_csv, load_or_build_sets,
                        sets_config_from_args, summarize_sets)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0],
                                formatter_class=argparse.RawDescriptionHelpFormatter,
                                epilog="\n".join(__doc__.splitlines()[2:]))
    add_repo_arg(p)
    add_sets_args(p)
    p.add_argument("--splits", nargs="+", choices=list(SPLITS), default=list(SPLITS))
    p.add_argument("--tgz", help="local SBIC.v2.tgz (skips the download)")
    p.add_argument("--max-length", type=int, default=128)
    p.add_argument("--rebuild-data", action="store_true", help="rebuild the tokenized cache")
    p.add_argument("--export-csv", action="store_true")
    p.add_argument("--num-proc", type=int, default=4)
    return p.parse_args()


def main():
    args = parse_args()
    sd = import_data_module(args.repo_dir)
    if args.max_length != sd.MAX_LENGTH:
        raise SystemExit("--max-length must match sbic_data.MAX_LENGTH (edit it there so every script agrees)")
    ds_tok, _, _ = sd.load_and_tokenize(num_proc=args.num_proc, rebuild=args.rebuild_data, tgz_path=args.tgz)
    print(ds_tok)

    tr = ds_tok["train"]
    lens = np.array([len(x) for x in tr["input_ids"]])
    print(f"\n[tokens] train p50={np.percentile(lens, 50):.0f} p99={np.percentile(lens, 99):.0f} "
          f"max={lens.max()} | at max_length {sd.MAX_LENGTH}: {(lens >= sd.MAX_LENGTH).mean():.3%}")
    import pandas as pd
    te = ds_tok["test"]
    frame = pd.concat([pd.DataFrame({"split": "train", "source": tr["source"], "y": tr["true_label"]}),
                       pd.DataFrame({"split": te["orig_split"], "source": te["source"], "y": te["true_label"]})])
    tab = frame.pivot_table(index="source", columns="split", values="y", aggfunc="size", fill_value=0)
    tab = tab.reindex(columns=[c for c in ("train", "dev", "test") if c in tab.columns])
    tab["offensive_rate"] = frame.groupby("source")["y"].mean().round(3)
    print("\n[sources] posts per split after de-duplication\n" + tab.sort_values("train", ascending=False).to_string())

    for split in args.splits:
        cfg = sets_config_from_args(args, read_forget_group(), split=split)
        path = default_sets_path(args.repo_dir, split)
        sets = load_or_build_sets(ds_tok, cfg, path, rebuild=args.rebuild_sets or args.rebuild_data)
        print(f"\n=== {split} split: forget source {cfg.forget_source!r}"
              + (f", adjacent {cfg.adjacent_sources!r}" if split == "class" else "") + " (checks passed)")
        print(f"  {'split':<6}{'set':<16}{'from':>10}{'n':>9}{'forget':>8}{'sibling':>9}{'offens.':>9}{'#src':>6}")
        for r in summarize_sets(ds_tok, sets, cfg):
            print(f"  {r['split']:<6}{r['set']:<16}{r['from_split']:>10}{r['n']:>9,}{r['forget_source_share']:>8.3f}"
                  f"{r['sibling_share']:>9.3f}{r['offensive_share']:>9.3f}{r['n_sources']:>6}")
        n_adj_probe = int(np.isin(sets["train"]["probe"], sets["train"]["adjacent"]).sum())
        print(f"  adjacent posts inside D_r^probe: {n_adj_probe}")
        if args.export_csv:
            out = os.path.join(os.path.dirname(path), "csv")
            export_csv(ds_tok, sets, cfg, out)
            print(f"  CSVs -> {out}")
    print("\nforget = share from the forget source, sibling = share from the adjacent sources, "
          "offens. = share labelled offensive, #src = number of sources")


if __name__ == "__main__":
    main()
