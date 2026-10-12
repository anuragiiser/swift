#!/usr/bin/env python
"""Inspect Social Bias Frames (SBIC v2) before choosing a forget set.

Downloads SBIC.v2.tgz from the authors' site (the source the Hugging Face loading script
uses; recent `datasets` versions no longer run that script), aggregates the ~3 annotations
per post, and reports what the forget-set choice depends on:

  * posts per split and offensive rate (post label: mean offensiveYN >= --threshold)
  * per data source (subreddit / Twitter corpus / Gab / Stormfront): posts per split, offensive rate
  * the most frequently targeted groups (targetMinority, any annotator), posts per split
  * how many NON-offensive posts carry any target annotation (why a ToxiGen-style class
    split with benign posts about the same group is hard on SBIC)
  * posts that occur in more than one split (should be none)

  python inspect_sbic.py --data-dir ~/data/sbic
  python inspect_sbic.py --data-dir ~/data/sbic --tgz /path/to/SBIC.v2.tgz   # if already downloaded

Writes <data-dir>/sbic_summary.txt and <data-dir>/sbic_posts.csv (one row per post).
"""
import argparse
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


def parse_args():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0],
                                formatter_class=argparse.RawDescriptionHelpFormatter,
                                epilog="\n".join(__doc__.splitlines()[2:]))
    p.add_argument("--data-dir", default="sbic_data")
    p.add_argument("--tgz", help="use this local SBIC.v2.tgz instead of downloading")
    p.add_argument("--threshold", type=float, default=0.5, help="offensive if mean offensiveYN >= this")
    p.add_argument("--top-groups", type=int, default=25)
    return p.parse_args()


def fetch(data_dir, tgz_path=None):
    os.makedirs(data_dir, exist_ok=True)
    if all(os.path.exists(os.path.join(data_dir, f)) for f in FILES.values()):
        return
    tgz = os.path.join(data_dir, "SBIC.v2.tgz")
    if tgz_path:
        if os.path.abspath(tgz_path) != os.path.abspath(tgz):
            shutil.copy(tgz_path, tgz)
    elif not os.path.exists(tgz):
        if not URL:
            raise SystemExit("[error] no local archive and no SBIC_URL set. " + SBIC_HINT)
        print(f"[download] {URL}")
        try:
            urllib.request.urlretrieve(URL, tgz)
        except Exception as exc:  # noqa: BLE001
            raise SystemExit(f"[error] download failed ({exc}). " + SBIC_HINT) from exc
    with tarfile.open(tgz) as tf:
        for m in tf.getmembers():
            name = os.path.basename(m.name)
            if m.isfile() and name in FILES.values():
                m.name = name
                tf.extract(m, data_dir)
    missing = [f for f in FILES.values() if not os.path.exists(os.path.join(data_dir, f))]
    if missing:
        raise SystemExit(f"[error] {missing} not found in the archive")


def split_groups(s):
    """targetMinority / targetCategory may list several groups; normalise to lowercase items."""
    if not isinstance(s, str) or not s.strip():
        return []
    return [g.strip().lower() for g in s.replace(";", ",").split(",") if g.strip()]


def aggregate(df, split, threshold):
    """One row per post: mean offensiveYN over annotators, majority source, union of targets."""
    df = df.copy()
    df["off"] = pd.to_numeric(df["offensiveYN"], errors="coerce")
    rows = []
    for post, g in df.groupby("post", sort=False):
        off = g["off"].dropna()
        groups = sorted({x for s in g["targetMinority"] for x in split_groups(s)})
        cats = sorted({x for s in g["targetCategory"] for x in split_groups(s)})
        src = g["dataSource"].dropna()
        rows.append({"split": split, "post": post, "n_annot": len(g),
                     "off_mean": off.mean() if len(off) else np.nan,
                     "source": src.mode().iat[0] if len(src) else "",
                     "groups": "|".join(groups), "categories": "|".join(cats)})
    out = pd.DataFrame(rows)
    n_unlabelled = int(out["off_mean"].isna().sum())
    out = out[out["off_mean"].notna()].copy()
    out["offensive"] = (out["off_mean"] >= threshold).astype(int)
    return out, n_unlabelled


def main():
    args = parse_args()
    fetch(args.data_dir, args.tgz)
    parts, unlabelled = [], {}
    for split, fname in FILES.items():
        raw = pd.read_csv(os.path.join(args.data_dir, fname), dtype=str, keep_default_na=False,
                          na_values=[""])
        agg, n_unl = aggregate(raw, split, args.threshold)
        parts.append(agg)
        unlabelled[split] = (len(raw), n_unl)
    posts = pd.concat(parts, ignore_index=True)
    posts.to_csv(os.path.join(args.data_dir, "sbic_posts.csv"), index=False)

    lines = []

    def say(s=""):
        print(s)
        lines.append(s)

    say(f"Posts (label: mean offensiveYN >= {args.threshold})")
    summary = posts.groupby("split").agg(posts=("post", "size"), offensive_rate=("offensive", "mean"),
                                         annotations_per_post=("n_annot", "mean")).round(3)
    say(summary.reindex([s for s in FILES if s in summary.index]).to_string())
    for split, (n_rows, n_unl) in unlabelled.items():
        if n_unl:
            say(f"  {split}: {n_rows:,} annotation rows; {n_unl:,} posts without an offensiveYN value dropped")

    dup = posts.groupby("post")["split"].nunique()
    say(f"\nPosts in more than one split: {(dup > 1).sum():,}")

    say("\nPer data source: posts per split and offensive rate")
    src = posts.pivot_table(index="source", columns="split", values="post", aggfunc="size", fill_value=0)
    src = src.reindex(columns=[s for s in FILES if s in src.columns])
    src["total"] = src.sum(axis=1)
    src["offensive_rate"] = posts.groupby("source")["offensive"].mean().round(3)
    say(src.sort_values("total", ascending=False).to_string())

    say(f"\nTop {args.top_groups} targeted groups (posts where any annotator named the group)")
    exploded = posts.assign(group=posts["groups"].str.split("|")).explode("group")
    exploded = exploded[exploded["group"].fillna("") != ""]
    if len(exploded):
        grp = exploded.pivot_table(index="group", columns="split", values="post", aggfunc="size", fill_value=0)
        grp = grp.reindex(columns=[s for s in FILES if s in grp.columns])
        grp["total"] = grp.sum(axis=1)
        grp["offensive_rate"] = exploded.groupby("group")["offensive"].mean().round(3)
        say(grp.sort_values("total", ascending=False).head(args.top_groups).to_string())
    else:
        say("  (no target annotations found)")

    say("\nTarget categories (posts where any annotator chose the category)")
    cats = posts.assign(cat=posts["categories"].str.split("|")).explode("cat")
    cats = cats[cats["cat"].fillna("") != ""]
    if len(cats):
        say(cats.groupby("cat").agg(posts=("post", "size"), offensive_rate=("offensive", "mean"))
            .round(3).sort_values("posts", ascending=False).to_string())

    benign = posts[posts["offensive"] == 0]
    with_target = int((benign["groups"] != "").sum())
    say(f"\nNon-offensive posts: {len(benign):,}; with any target annotation: {with_target:,} "
        f"({with_target / max(len(benign), 1):.1%})")

    with open(os.path.join(args.data_dir, "sbic_summary.txt"), "w") as fh:
        fh.write("\n".join(lines) + "\n")
    print(f"\n[done] {args.data_dir}/sbic_summary.txt, {args.data_dir}/sbic_posts.csv")


if __name__ == "__main__":
    main()
