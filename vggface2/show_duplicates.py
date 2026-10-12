#!/usr/bin/env python
"""Save a grid of removed val/test images (left) next to their closest training
image of the same identity (right), with the Hamming distance, to check whether
the near-duplicate filter removes real copies or just similar-looking photos.

  python show_duplicates.py --prep-dir path/to/vgg_prep --n 24
"""
import argparse
import json
import os

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw

from make_forget_sets import POP8

p = argparse.ArgumentParser()
p.add_argument("--prep-dir", required=True)
p.add_argument("--n", type=int, default=24)
p.add_argument("--forget-only", action="store_true", help="only the 10 forget subjects")
p.add_argument("--seed", type=int, default=0)
args = p.parse_args()

P = args.prep_dir
info = json.load(open(os.path.join(P, "prep_info.json")))
s = json.load(open(os.path.join(P, "sets.json")))
m = pd.read_csv(os.path.join(P, "meta.csv")).set_index("index")
mm = np.memmap(os.path.join(P, info["images_file"]), dtype=np.uint8, mode="r", shape=tuple(info["shape"]))
S = info["img_size"]

train = np.array(s["sets"]["train"]["all"])
tid = m.loc[train, "identity"].to_numpy()
hashes = m["dhash"].to_numpy(dtype=np.int64).astype(np.uint64)     # position == index
pool = np.array(s["excluded"])
if args.forget_only:
    pool = pool[np.isin(m.loc[pool, "identity"].to_numpy(), s["forget_identities"])]
if len(pool) == 0:
    raise SystemExit("no excluded images to show")
pick = np.random.default_rng(args.seed).choice(pool, min(args.n, len(pool)), replace=False)

dist_all = []
for e in pool:
    same = train[tid == m.at[e, "identity"]]
    dist_all.append(POP8[(hashes[same] ^ hashes[e]).view(np.uint8)].reshape(-1, 8).sum(1).min())
print("Hamming distance of removed images to their closest train image:",
      dict(zip(*np.unique(dist_all, return_counts=True))))

cols = 4
rows_n = int(np.ceil(len(pick) / cols))
sheet = Image.new("RGB", (cols * (2 * S + 10), rows_n * (S + 14)), "white")
draw = ImageDraw.Draw(sheet)
for k, e in enumerate(pick):
    same = train[tid == m.at[e, "identity"]]
    bits = POP8[(hashes[same] ^ hashes[e]).view(np.uint8)].reshape(-1, 8).sum(1)
    j = same[bits.argmin()]
    x0, y0 = (k % cols) * (2 * S + 10), (k // cols) * (S + 14)
    sheet.paste(Image.fromarray(np.asarray(mm[e])), (x0, y0 + 12))
    sheet.paste(Image.fromarray(np.asarray(mm[j])), (x0 + S, y0 + 12))
    draw.text((x0 + 2, y0), f"{m.at[e, 'identity']}  d={int(bits.min())}", fill="black")
out = os.path.join(P, "duplicates_check.png")
sheet.save(out)
print(f"saved {out}  (left: removed val/test image, right: closest train image)")
