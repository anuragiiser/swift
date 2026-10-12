#!/usr/bin/env python
"""Step 1: preprocess VGGFace2 once into a fast uint8 array.

For every image of the chosen identities:
  * read the loose face box (bb_landmark/loose_bb_{train,test}.csv) and record the
    FACE SIZE = sqrt(W * H) in original pixels (the quality score used for D_f)
  * decode with PIL draft mode (JPEG DCT downscaling, much faster than a full decode),
    crop a square around the box (+margin), resize to --img-size, store as uint8
  * record a 64-bit difference hash of the crop (used later to drop near-duplicates
    that straddle train and test)

Output (in --out-dir):
  images_<S>.u8   uint8 memmap, shape [N, S, S, 3]   (~9.8 GB for 200k images at S=128)
  meta.csv        one row per image: path, identity, label, face size, hash, ...
  prep_info.json  shapes, settings and a fingerprint of the image list

Training then loads this array straight into GPU memory: no JPEG decoding at train time.

Examples
  # identities = every folder under the given roots
  python prepare_vggface2.py --roots /data/vggface2/train /data/vggface2/test \
      --bbox-csv /data/vggface2/bb_landmark/loose_bb_train.csv \
                 /data/vggface2/bb_landmark/loose_bb_test.csv \
      --identity-meta /data/vggface2/identity_meta.csv --out-dir /data/vgg_prep
  # restrict to your 540-identity subset
  python prepare_vggface2.py ... --identities my_540_ids.txt
"""
import argparse
import hashlib
import json
import os
import sys
import time
from multiprocessing import Pool

import numpy as np
import pandas as pd
from PIL import Image

_MM = None          # per-worker memmap handle
_CFG = None


def parse_args():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0],
                                formatter_class=argparse.RawDescriptionHelpFormatter,
                                epilog="\n".join(__doc__.splitlines()[17:]))
    p.add_argument("--roots", nargs="+", required=True,
                   help="folders containing identity sub-folders (n000002/, ...)")
    p.add_argument("--bbox-csv", nargs="*", default=None,
                   help="loose_bb_*.csv files (NAME_ID,X,Y,W,H); default: search near the roots")
    p.add_argument("--identity-meta", help="identity_meta.csv (adds name/gender; optional)")
    p.add_argument("--identities", help="text file with one identity id per line (default: all)")
    p.add_argument("--out-dir", required=True)
    p.add_argument("--img-size", type=int, default=128,
                   help="stored crop size; 128 is fast on H100, 160 matches FaceNet")
    p.add_argument("--margin", type=float, default=0.1,
                   help="extra context around the loose box, per side, as a fraction of its size")
    p.add_argument("--face-size-source", choices=["bbox", "image"], default="bbox",
                   help="'image' only for datasets whose images are already face crops")
    p.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    p.add_argument("--exts", nargs="+", default=[".jpg", ".jpeg", ".png"])
    return p.parse_args()


# --------------------------------------------------------------------------- indexing
def list_images(roots, wanted, exts):
    rows, seen = [], {}
    for root in roots:
        root = os.path.abspath(root)
        ids = sorted(d for d in os.listdir(root) if os.path.isdir(os.path.join(root, d)))
        if wanted is not None:
            ids = [i for i in ids if i in wanted]
        for ident in ids:
            for f in sorted(os.listdir(os.path.join(root, ident))):
                stem, ext = os.path.splitext(f)
                if ext.lower() not in exts:
                    continue
                key = f"{ident}/{stem}"
                if key in seen:
                    raise SystemExit(f"[error] {key} appears under both {seen[key]} and {root}")
                seen[key] = root
                rows.append({"key": key, "identity": ident, "root": root,
                             "relpath": f"{ident}/{f}"})
    return pd.DataFrame(rows)


def find_bbox_csvs(roots):
    found = []
    for root in roots:
        for base in (root, os.path.dirname(os.path.abspath(root))):
            for sub in ("", "bb_landmark", "meta", "meta/bb_landmark"):
                d = os.path.join(base, sub)
                if os.path.isdir(d):
                    found += [os.path.join(d, f) for f in os.listdir(d)
                              if f.startswith("loose_bb") and f.endswith(".csv")]
    return sorted(set(found))


def load_bboxes(paths, identities):
    frames = []
    for path in paths:
        df = pd.read_csv(path, skipinitialspace=True)
        df.columns = [c.strip().upper() for c in df.columns]
        need = {"NAME_ID", "X", "Y", "W", "H"}
        if not need <= set(df.columns):
            raise SystemExit(f"[error] {path} lacks columns {need - set(df.columns)}")
        df["NAME_ID"] = df["NAME_ID"].astype(str).str.strip().str.replace(r"\.(jpg|jpeg|png)$", "", regex=True)
        df = df[df["NAME_ID"].str.split("/").str[0].isin(identities)]
        frames.append(df[["NAME_ID", "X", "Y", "W", "H"]])
    out = pd.concat(frames).drop_duplicates("NAME_ID").set_index("NAME_ID")
    return out


def load_identity_meta(path):
    df = pd.read_csv(path, skipinitialspace=True, quotechar='"')
    df.columns = [c.strip() for c in df.columns]
    cid = next(c for c in df.columns if c.lower().replace("_", "") in ("classid", "id"))
    out = pd.DataFrame({"identity": df[cid].astype(str).str.strip()})
    for col, name in (("Name", "name"), ("Gender", "gender"), ("Flag", "vgg_flag")):
        if col in df.columns:
            out[name] = df[col].astype(str).str.strip()
    return out


# --------------------------------------------------------------------------- workers
def _init(mm_path, shape, cfg):
    global _MM, _CFG
    _MM = np.memmap(mm_path, dtype=np.uint8, mode="r+", shape=shape)
    _CFG = cfg


def dhash64(img):
    g = np.asarray(img.convert("L").resize((9, 8), Image.BILINEAR), dtype=np.int16)
    bits = (g[:, 1:] > g[:, :-1]).flatten()
    return int(np.packbits(bits).view(">u8")[0].astype(np.int64))


def _process(task):
    i, path, box = task
    S, margin, source = _CFG["img_size"], _CFG["margin"], _CFG["source"]
    try:
        im = Image.open(path)
        ow, oh = im.size
        if source == "bbox":
            x, y, w, h = box
            side = max(w, h) * (1.0 + 2.0 * margin)
            cx, cy = x + w / 2.0, y + h / 2.0
            face_w, face_h = float(w), float(h)
        else:
            side = float(min(ow, oh))
            cx, cy = ow / 2.0, oh / 2.0
            face_w, face_h = float(ow), float(oh)
        scale = S / max(side, 1.0)
        if scale < 1.0:  # JPEG draft: decode at 1/2, 1/4 or 1/8 when the crop allows it
            im.draft("RGB", (max(1, int(ow * scale) + 1), max(1, int(oh * scale) + 1)))
        f = im.size[0] / ow
        im = im.convert("RGB")
        half = side * f / 2.0
        crop = im.crop((int(round(cx * f - half)), int(round(cy * f - half)),
                        int(round(cx * f + half)), int(round(cy * f + half))))
        crop = crop.resize((S, S), Image.BILINEAR)
        _MM[i] = np.asarray(crop, dtype=np.uint8)
        return i, face_w, face_h, ow, oh, dhash64(crop), True
    except Exception as exc:  # noqa: BLE001
        sys.stderr.write(f"[warn] {path}: {exc}\n")
        return i, np.nan, np.nan, np.nan, np.nan, 0, False


# --------------------------------------------------------------------------- main
def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    wanted = None
    if args.identities:
        with open(args.identities) as fh:
            wanted = {l.strip() for l in fh if l.strip() and not l.startswith("#")}

    t0 = time.time()
    df = list_images(args.roots, wanted, {e.lower() for e in args.exts})
    if df.empty:
        raise SystemExit("[error] no images found under --roots")
    identities = sorted(df["identity"].unique())
    if wanted is not None and set(wanted) - set(identities):
        print(f"[warn] {len(set(wanted) - set(identities))} listed identities have no folder")
    print(f"[index] {len(df):,} images, {len(identities)} identities ({time.time() - t0:.1f}s)")

    if args.face_size_source == "bbox":
        csvs = args.bbox_csv if args.bbox_csv else find_bbox_csvs(args.roots)
        if not csvs:
            raise SystemExit("[error] no loose_bb_*.csv found; pass --bbox-csv "
                             "(or --face-size-source image for pre-cropped faces)")
        print(f"[bbox] reading {csvs}")
        bb = load_bboxes(csvs, set(identities))
        has = df["key"].isin(bb.index)
        if (~has).any():
            print(f"[warn] {int((~has).sum()):,} images have no box and are skipped")
        df = df[has].reset_index(drop=True)
        boxes = bb.loc[df["key"], ["X", "Y", "W", "H"]].to_numpy(dtype=np.float64)
    else:
        boxes = np.zeros((len(df), 4))

    label_of = {ident: k for k, ident in enumerate(sorted(df["identity"].unique()))}
    df["label"] = df["identity"].map(label_of).astype(np.int64)
    df.insert(0, "index", np.arange(len(df)))

    S = args.img_size
    shape = (len(df), S, S, 3)
    mm_name = f"images_{S}.u8"
    mm_path = os.path.join(args.out_dir, mm_name)
    np.memmap(mm_path, dtype=np.uint8, mode="w+", shape=shape).flush()
    print(f"[crop] {len(df):,} images -> {mm_path} ({np.prod(shape) / 1e9:.1f} GB), "
          f"{args.workers} workers")

    tasks = [(i, os.path.join(r, p), tuple(b))
             for i, (r, p, b) in enumerate(zip(df["root"], df["relpath"], boxes))]
    cfg = {"img_size": S, "margin": args.margin, "source": args.face_size_source}
    res = np.zeros((len(df), 6))
    ok = np.zeros(len(df), dtype=bool)
    hashes = np.zeros(len(df), dtype=np.int64)
    t1 = time.time()
    with Pool(args.workers, initializer=_init, initargs=(mm_path, shape, cfg)) as pool:
        for n, (i, fw, fh, ow, oh, hsh, good) in enumerate(
                pool.imap_unordered(_process, tasks, chunksize=64), 1):
            res[i, :4] = (fw, fh, ow, oh)
            hashes[i], ok[i] = hsh, good
            if n % 20000 == 0 or n == len(tasks):
                el = time.time() - t1
                print(f"  {n:,}/{len(tasks):,}  {n / el:.0f} img/s", flush=True)

    df["face_w"], df["face_h"] = res[:, 0], res[:, 1]
    df["face_size"] = np.sqrt(df["face_w"] * df["face_h"])
    df["img_w"], df["img_h"] = res[:, 2], res[:, 3]
    df["dhash"], df["ok"] = hashes, ok
    if args.identity_meta:
        df = df.merge(load_identity_meta(args.identity_meta), on="identity", how="left")
    df = df.drop(columns=["key"]).sort_values("index").reset_index(drop=True)
    df.to_csv(os.path.join(args.out_dir, "meta.csv"), index=False)

    fingerprint = hashlib.sha1("\n".join(df["relpath"]).encode()).hexdigest()
    info = {"images_file": mm_name, "shape": list(shape), "img_size": S, "margin": args.margin,
            "face_size_source": args.face_size_source, "roots": [os.path.abspath(r) for r in args.roots],
            "n_images": int(len(df)), "n_ok": int(ok.sum()), "n_identities": len(label_of),
            "fingerprint": fingerprint}
    with open(os.path.join(args.out_dir, "prep_info.json"), "w") as fh:
        json.dump(info, fh, indent=2)

    fs = df.loc[df["ok"], "face_size"]
    print(f"[done] {info['n_ok']:,}/{len(df):,} ok, {len(label_of)} identities, "
          f"{time.time() - t0:.0f}s total")
    print(f"[face size px] p10={fs.quantile(.1):.0f} p50={fs.median():.0f} p90={fs.quantile(.9):.0f}")


if __name__ == "__main__":
    main()
