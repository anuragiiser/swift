#!/usr/bin/env python
"""Per-subject accuracy of Swift, Retrain (and Original) on the VGGFace2 face-size sets.

For each of the 10 forget subjects, accuracy on:
  Df(tr)     the subject's high-quality training images (forgotten)
  Dadj(tr)   the subject's other training images
  Dcal(val)  high-quality validation images (Swift's calibration anchor)
  Ftest(te)  high-quality test images (never seen by any model)
  Dadj(te)   other test images of the subject
plus the median face size (sqrt(w*h)) of Df and Dadj, and each model's deviation from Retrain.

  python per_subject_accuracy.py --prep-dir $PREP
  python per_subject_accuracy.py --prep-dir $PREP \
      --models Retrain=$PREP/checkpoints/vgg_retrain_seed42.pt \
               Swift=$PREP/checkpoints/vgg_swift_seed2025.pt \
               Original=$PREP/checkpoints/vgg_original_seed42.pt

Default models: Original, Retrain and Swift from the standard checkpoint names (missing ones
are skipped). Writes <prep>/results/per_subject_accuracy.csv and .md (and .tex).
"""
import argparse
import os

import numpy as np
import pandas as pd
import torch

import vgg_common as vc

SETS = [("Df(tr)", "train", "forget"), ("Dadj(tr)", "train", "adjacent"), ("Dcal(val)", "val", "cal"),
        ("Ftest(te)", "test", "ftest"), ("Dadj(te)", "test", "adjacent_te")]


def parse_args():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0],
                                formatter_class=argparse.RawDescriptionHelpFormatter,
                                epilog="\n".join(__doc__.splitlines()[2:]))
    p.add_argument("--prep-dir", required=True)
    p.add_argument("--sets", help="default <prep-dir>/sets.json")
    p.add_argument("--models", nargs="*", help="name=checkpoint.pt pairs (default: Original, Retrain, Swift)")
    p.add_argument("--train-seed", type=int, default=42)
    p.add_argument("--seed", type=int, default=2025, help="unlearning seed of the method checkpoints")
    p.add_argument("--reference", default="Retrain", help="model the deviations are measured against")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--out", help="default <prep-dir>/results/per_subject_accuracy")
    return p.parse_args()


def default_models(prep, train_seed, seed):
    c = os.path.join(prep, "checkpoints")
    return [("Original", os.path.join(c, f"vgg_original_seed{train_seed}.pt")),
            ("Retrain", os.path.join(c, f"vgg_retrain_seed{train_seed}.pt")),
            ("Swift", os.path.join(c, f"vgg_swift_seed{seed}.pt"))]


def main():
    args = parse_args()
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    vc.fast_cuda()
    info, meta, mm = vc.load_prep(args.prep_dir)
    sets = vc.load_sets(args.sets or os.path.join(args.prep_dir, "sets.json"), info)
    out = args.out or os.path.join(args.prep_dir, "results", "per_subject_accuracy")
    os.makedirs(os.path.dirname(out), exist_ok=True)

    if args.models:
        models = []
        for item in args.models:
            if "=" not in item:
                raise SystemExit(f"--models expects name=path, got {item!r}")
            name, path = item.split("=", 1)
            models.append((name, path))
    else:
        models = default_models(args.prep_dir, args.train_seed, args.seed)
    present = [(n, p) for n, p in models if os.path.exists(p)]
    for n, p in models:
        if not os.path.exists(p):
            print(f"[skip] {n}: {p} not found")
    if not present:
        raise SystemExit("no checkpoints found")

    m = meta.set_index("index")
    ident = m["identity"]
    subjects = list(sets["forget_identities"])
    S = sets["sets"]
    # indices per (set, subject)
    groups = {}
    for col, split, key in SETS:
        idx = np.asarray(S[split].get(key, []), dtype=np.int64)
        ids = ident.loc[idx].to_numpy() if len(idx) else np.array([])
        for s in subjects:
            groups[(col, s)] = idx[ids == s]

    data = vc.GPUImages(mm, meta["label"].to_numpy(), device)
    rows = []
    for s in subjects:
        f_idx, a_idx = groups[("Df(tr)", s)], groups[("Dadj(tr)", s)]
        rows.append({"subject": s, "model": "(sizes)",
                     **{c: len(groups[(c, s)]) for c, _, _ in SETS},
                     "Df_median_px": float(m.loc[f_idx, "face_size"].median()) if len(f_idx) else np.nan,
                     "Dadj_median_px": float(m.loc[a_idx, "face_size"].median()) if len(a_idx) else np.nan})

    for name, path in present:
        model, _ = vc.load_checkpoint(path, device)
        model.eval()
        print(f"[eval] {name}: {path}")
        for s in subjects:
            row = {"subject": s, "model": name}
            for col, _, _ in SETS:
                idx = groups[(col, s)]
                row[col] = vc.acc_of(model, data, data.idx(idx)) if len(idx) else np.nan
            rows.append(row)
        # pooled over the 10 subjects (equals the main table's columns)
        row = {"subject": "ALL", "model": name}
        for col, split, key in SETS:
            idx = np.asarray(S[split].get(key, []), dtype=np.int64)
            row[col] = vc.acc_of(model, data, data.idx(idx)) if len(idx) else np.nan
        rows.append(row)
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    df = pd.DataFrame(rows)
    acc_cols = [c for c, _, _ in SETS]
    sizes = df[df["model"] == "(sizes)"].set_index("subject")
    res = df[df["model"] != "(sizes)"]

    # wide table: one row per subject, one column block per model
    wide = res.pivot(index="subject", columns="model", values=acc_cols)
    order = [s for s in subjects] + ["ALL"]
    wide = wide.reindex(order)
    names = [n for n, _ in present]

    # deviation from the reference model
    dev = None
    if args.reference in names:
        ref = res[res["model"] == args.reference].set_index("subject")[acc_cols]
        dev_rows = []
        for n in names:
            if n == args.reference:
                continue
            cur = res[res["model"] == n].set_index("subject")[acc_cols]
            d = (cur - ref).reindex(order)
            d.insert(0, "model", n)
            d["L1"] = d[acc_cols].abs().sum(axis=1)
            dev_rows.append(d)
        dev = pd.concat(dev_rows)

    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 50)
    print("\nImages per subject and median face size (px)")
    print(sizes[acc_cols + ["Df_median_px", "Dadj_median_px"]].reindex(subjects)
          .to_string(float_format=lambda x: f"{x:.1f}"))
    for col in acc_cols:
        print(f"\nAccuracy (%) on {col}")
        print(wide[col][names].to_string(float_format=lambda x: f"{x:.1f}"))
    if dev is not None:
        print(f"\nDeviation from {args.reference} (model - {args.reference}, points; L1 = sum of |dev|)")
        for n in dev["model"].unique():
            print(f"\n  {n}")
            print(dev[dev["model"] == n].drop(columns="model").to_string(float_format=lambda x: f"{x:+.1f}"))

    # outputs
    tidy = res.drop(columns=["Df_median_px", "Dadj_median_px"], errors="ignore").merge(sizes[["Df_median_px", "Dadj_median_px"] + acc_cols].rename(
        columns={c: f"n_{c}" for c in acc_cols}), left_on="subject", right_index=True, how="left")
    tidy.to_csv(out + ".csv", index=False)
    with open(out + ".md", "w") as fh:
        fh.write("| Subject | Df px (n) | Dadj px (n) | " + " | ".join(
            f"{n} {c}" for n in names for c in ("Df", "Ftest", "Dadj(te)")) + " |\n")
        fh.write("|---|---:|---:|" + "---:|" * (3 * len(names)) + "\n")
        for s in order:
            z = sizes.loc[s] if s in sizes.index else None
            fpx = f"{z['Df_median_px']:.1f} ({int(z['Df(tr)'])})" if z is not None else "–"
            apx = f"{z['Dadj_median_px']:.1f} ({int(z['Dadj(tr)'])})" if z is not None else "–"
            vals = [f"{wide.loc[s, (c, n)]:.1f}" for n in names for c in ("Df(tr)", "Ftest(te)", "Dadj(te)")]
            fh.write(f"| {s} | {fpx} | {apx} | " + " | ".join(vals) + " |\n")
    with open(out + ".tex", "w") as fh:
        fh.write("\\begin{tabular}{l rr " + "rrr " * len(names) + "}\n\\toprule\n")
        fh.write("Subject & $\\mathcal{D}_f$ px ($n$) & $\\mathcal{D}_r^{adj}$ px ($n$) & " + " & ".join(
            f"\\multicolumn{{3}}{{c}}{{{n}}}" for n in names) + " \\\\\n")
        fh.write(" & & & " + " & ".join("$\\mathcal{D}_f$ & $F_{test}$ & $\\mathcal{D}_r^{adj}$(te)" for _ in names)
                 + " \\\\\n\\midrule\n")
        for s in order:
            if s == "ALL":
                fh.write("\\midrule\n")
            z = sizes.loc[s] if s in sizes.index else None
            fpx = f"{z['Df_median_px']:.1f} ({int(z['Df(tr)'])})" if z is not None else "--"
            apx = f"{z['Dadj_median_px']:.1f} ({int(z['Dadj(tr)'])})" if z is not None else "--"
            vals = [f"{wide.loc[s, (c, n)]:.1f}" for n in names for c in ("Df(tr)", "Ftest(te)", "Dadj(te)")]
            fh.write(f"{s.replace('_', chr(92) + '_')} & {fpx} & {apx} & " + " & ".join(vals) + " \\\\\n")
        fh.write("\\bottomrule\n\\end{tabular}\n")
    print(f"\n[done] {out}.csv, {out}.md, {out}.tex")


if __name__ == "__main__":
    main()
