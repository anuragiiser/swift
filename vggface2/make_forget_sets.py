#!/usr/bin/env python
"""Step 2: choose the forget subjects and build every set by face size.

For the K forget subjects (default 10):
  HQ  = the top --hq-frac of that subject's images by face size (per-subject
        threshold, so no subject dominates), LQ = the rest.
  HQ and LQ are split into train/val/test separately, so every forget subject
  has high-quality images in train (D_f), val (D_cal) and test (Ftest).
All other identities are split per identity (default 80/10/10), so every
identity appears in train, val and test.

Sets (indices into the preprocessed array):
  train  forget     = HQ train images of the K subjects                 D_f
         adjacent   = LQ train images of the K subjects                 D_r^adj
         remote     = train images of every other identity              D_r^rem
         retain     = adjacent + remote       (Retrain trains on this)
         all        = forget + retain         (Original trains on this)
  val    cal        = HQ val images of the K subjects   (Swift D_cal, never trained on)
         val_adjacent, val_remote
  test   ftest      = HQ test images of the K subjects  (evaluation only)
         adjacent_te, remote_te

Near-duplicates: any val/test image whose 64-bit dHash is within --dedup-hamming
bits of a train image of the same identity is dropped from evaluation, so test
accuracy cannot come from a copy of a training photo.

Subjects: chosen at random (seeded) among identities with >= --min-images images,
balanced by gender when identity_meta was given; or pass --forget-ids.
Writes sets.json, a per-subject summary and a contact sheet (HQ vs LQ crops).
"""
import argparse
import json
import os

import numpy as np
import pandas as pd

POP8 = np.array([bin(i).count("1") for i in range(256)], dtype=np.uint8)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0],
                                formatter_class=argparse.RawDescriptionHelpFormatter,
                                epilog="\n".join(__doc__.splitlines()[2:]))
    p.add_argument("--prep-dir", required=True)
    p.add_argument("--out", help="default <prep-dir>/sets.json")
    p.add_argument("--n-forget", type=int, default=10)
    p.add_argument("--forget-ids", nargs="*", help="explicit identity ids (overrides random choice)")
    p.add_argument("--min-images", type=int, default=300)
    p.add_argument("--hq-frac", type=float, default=0.35, help="top fraction by face size = HQ")
    p.add_argument("--split", type=float, nargs=3, default=[0.8, 0.1, 0.1],
                   help="train/val/test fractions for other identities")
    p.add_argument("--forget-split", type=float, nargs=3, default=[0.7, 0.15, 0.15],
                   help="train/val/test fractions inside each HQ/LQ group of forget subjects "
                        "(larger val/test so D_cal and Ftest are not tiny)")
    p.add_argument("--no-gender-balance", action="store_true")
    p.add_argument("--dedup-hamming", type=int, default=4, help="-1 disables near-duplicate removal")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--no-contact-sheet", action="store_true")
    return p.parse_args()


def split_group(idx, fracs, rng):
    """Shuffle and split one group; val/test get at least one image when n >= 3."""
    idx = rng.permutation(np.asarray(idx))
    n = len(idx)
    n_val = int(round(fracs[1] * n))
    n_test = int(round(fracs[2] * n))
    if n >= 3:
        n_val, n_test = max(1, n_val), max(1, n_test)
    n_train = max(0, n - n_val - n_test)
    return idx[:n_train], idx[n_train:n_train + n_val], idx[n_train + n_val:]


def hamming_min(a, b):
    """min Hamming distance from every hash in a to any hash in b (int64 arrays)."""
    if len(a) == 0 or len(b) == 0:
        return np.full(len(a), 64)
    x = (a.astype(np.uint64)[:, None] ^ b.astype(np.uint64)[None, :])
    bits = POP8[x.view(np.uint8)].reshape(len(a), len(b), 8).sum(-1)
    return bits.min(1)


def choose_subjects(meta, args, rng):
    counts = meta.groupby("identity").size()
    if args.forget_ids:
        missing = [i for i in args.forget_ids if i not in counts.index]
        if missing:
            raise SystemExit(f"[error] unknown forget ids {missing}")
        return list(args.forget_ids)
    eligible = counts[counts >= args.min_images].index.tolist()
    if len(eligible) < args.n_forget:
        raise SystemExit(f"[error] only {len(eligible)} identities have >= {args.min_images} "
                         "images; lower --min-images")
    if "gender" in meta.columns and not args.no_gender_balance:
        g = meta.drop_duplicates("identity").set_index("identity")["gender"]
        m = [i for i in eligible if str(g.get(i, "")).lower().startswith("m")]
        f = [i for i in eligible if str(g.get(i, "")).lower().startswith("f")]
        n_m = args.n_forget // 2
        n_f = args.n_forget - n_m
        if len(m) >= n_m and len(f) >= n_f:
            return sorted(list(rng.choice(m, n_m, replace=False)) + list(rng.choice(f, n_f, replace=False)))
        print("[warn] not enough eligible identities per gender; choosing without balance")
    return sorted(rng.choice(eligible, args.n_forget, replace=False).tolist())


def main():
    args = parse_args()
    rng = np.random.default_rng(args.seed)
    with open(os.path.join(args.prep_dir, "prep_info.json")) as fh:
        info = json.load(fh)
    meta = pd.read_csv(os.path.join(args.prep_dir, "meta.csv"))
    meta = meta[meta["ok"]].reset_index(drop=True)
    out = args.out or os.path.join(args.prep_dir, "sets.json")

    forget_ids = choose_subjects(meta, args, rng)
    fset = set(forget_ids)
    print(f"[subjects] {forget_ids}")

    sets = {"train": {k: [] for k in ("forget", "adjacent", "remote")},
            "val": {k: [] for k in ("cal", "val_adjacent", "val_remote")},
            "test": {k: [] for k in ("ftest", "adjacent_te", "remote_te")}}
    subj_rows, excluded = [], []
    mi = meta.set_index("index")
    hashes, face_size = mi["dhash"], mi["face_size"]

    for ident, grp in meta.groupby("identity", sort=True):
        idx = grp["index"].to_numpy()
        if ident in fset:
            n_hq = int(np.ceil(args.hq_frac * len(grp)))
            # rank by face size, random tie-break
            order = np.lexsort((rng.random(len(grp)), -grp["face_size"].to_numpy()))
            hq, lq = idx[order[:n_hq]], idx[order[n_hq:]]
            hq_tr, hq_va, hq_te = split_group(hq, args.forget_split, rng)
            lq_tr, lq_va, lq_te = split_group(lq, args.forget_split, rng)
            parts = {("train", "forget"): hq_tr, ("train", "adjacent"): lq_tr,
                     ("val", "cal"): hq_va, ("val", "val_adjacent"): lq_va,
                     ("test", "ftest"): hq_te, ("test", "adjacent_te"): lq_te}
        else:
            tr, va, te = split_group(idx, args.split, rng)
            parts = {("train", "remote"): tr, ("val", "val_remote"): va, ("test", "remote_te"): te}

        # near-duplicate removal: drop val/test images that copy a train image of this identity
        n_dropped = 0
        if args.dedup_hamming >= 0:
            h = hashes.loc[idx]
            train_idx = np.concatenate([v for (s, _), v in parts.items() if s == "train"])
            th = h.loc[train_idx].to_numpy()
            for key in [k for k in parts if k[0] != "train"]:
                ev = parts[key]
                d = hamming_min(h.loc[ev].to_numpy(), th)
                dup = ev[d <= args.dedup_hamming]
                if len(dup):
                    excluded += dup.tolist()
                    n_dropped += len(dup)
                    parts[key] = ev[d > args.dedup_hamming]
        for (s, k), v in parts.items():
            sets[s][k] += np.asarray(v).tolist()

        if ident in fset:
            fs = face_size
            row = {"identity": ident, "n": len(grp),
                   "hq_threshold_px": float(fs.loc[hq].min()),
                   "median_hq_px": float(fs.loc[hq].median()),
                   "median_lq_px": float(fs.loc[lq].median())}
            for (s, k), v in parts.items():
                row[k] = len(v)
            row["dup_dropped"] = n_dropped
            for col in ("name", "gender"):
                if col in grp.columns:
                    row[col] = grp[col].iloc[0]
            subj_rows.append(row)

    sets["train"]["retain"] = sorted(sets["train"]["adjacent"] + sets["train"]["remote"])
    sets["train"]["all"] = sorted(sets["train"]["retain"] + sets["train"]["forget"])
    for s in sets:
        for k in sets[s]:
            sets[s][k] = sorted(int(i) for i in sets[s][k])

    # every forget subject needs HQ images in train (D_f), val (D_cal) and test (Ftest)
    empty = [r["identity"] for r in subj_rows if min(r["forget"], r["cal"], r["ftest"]) == 0]
    if empty:
        raise SystemExit(f"[error] subjects {empty} have no HQ images left in train, val or test "
                         f"(see dup_dropped). Lower --dedup-hamming (now {args.dedup_hamming}), "
                         "raise --hq-frac, or pass other --forget-ids.")

    # sanity checks
    tr, va, te = (set(sets["train"]["all"]), set(sum(sets["val"].values(), [])),
                  set(sum(sets["test"].values(), [])))
    assert not (tr & va) and not (tr & te) and not (va & te), "splits overlap"
    assert not set(sets["train"]["forget"]) & set(sets["train"]["retain"])
    lab = meta.set_index("index")["label"]
    f_labels = set(lab.loc[sets["train"]["forget"]])
    assert f_labels == set(lab.loc[sets["train"]["adjacent"]]) == set(lab.loc[sets["test"]["ftest"]]), \
        "D_f, D_adj and Ftest must cover the same identities"
    assert not f_labels & set(lab.loc[sets["train"]["remote"]])

    payload = {"prep_fingerprint": info["fingerprint"], "forget_identities": forget_ids,
               "forget_labels": sorted(int(x) for x in f_labels),
               "config": {k: v for k, v in vars(args).items() if k != "prep_dir"},
               "sizes": {f"{s}/{k}": len(v) for s in sets for k, v in sets[s].items()},
               "n_excluded_duplicates": len(excluded), "excluded": sorted(excluded),
               "subjects": subj_rows, "sets": sets}
    with open(out, "w") as fh:
        json.dump(payload, fh)

    subj = pd.DataFrame(subj_rows)
    pd.set_option("display.width", 200)
    print("\nPer-subject summary (face size in original pixels):")
    print(subj.to_string(index=False))
    print("\nSet sizes:")
    for k, v in payload["sizes"].items():
        print(f"  {k:22s}{v:>9,}")
    print(f"  near-duplicates dropped from val/test: {len(excluded):,}")
    print(f"[done] {out}")

    if not args.no_contact_sheet:
        contact_sheet(args.prep_dir, info, meta, sets, forget_ids, out.replace(".json", "_contact.png"), rng)


def contact_sheet(prep_dir, info, meta, sets, forget_ids, path, rng, per_side=6):
    from PIL import Image, ImageDraw
    mm = np.memmap(os.path.join(prep_dir, info["images_file"]), dtype=np.uint8, mode="r",
                   shape=tuple(info["shape"]))
    S = info["img_size"]
    ident = meta.set_index("index")["identity"]
    f_tr, a_tr = np.array(sets["train"]["forget"]), np.array(sets["train"]["adjacent"])
    gap = 8
    sheet = Image.new("RGB", (S * 2 * per_side + gap, S * len(forget_ids) + 20), "white")
    draw = ImageDraw.Draw(sheet)
    draw.text((4, 4), "D_f: high-quality (largest faces)", fill="black")
    draw.text((S * per_side + gap + 4, 4), "D_adj: remaining images", fill="black")
    for r, fid in enumerate(forget_ids):
        for c, pool in enumerate((f_tr[ident.loc[f_tr].values == fid], a_tr[ident.loc[a_tr].values == fid])):
            pick = rng.choice(pool, min(per_side, len(pool)), replace=False) if len(pool) else []
            for j, i in enumerate(pick):
                x0 = c * (S * per_side + gap) + j * S
                sheet.paste(Image.fromarray(np.asarray(mm[i])), (x0, 20 + r * S))
    sheet.save(path)
    print(f"[contact sheet] {path}")


if __name__ == "__main__":
    main()
