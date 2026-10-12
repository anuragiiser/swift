#!/usr/bin/env python
"""Collect Swift ablation results (ToxiGen and VGGFace2) into paper tables.

Reads every ablation_<group>_seed<seed>.json under the given folders (written by
ablate_swift_toxigen.py and ablate_swift_vgg.py) and writes one table per dataset/split:
Retrain, Original, then each ablation group with its variants. Columns: accuracy on
D_f, D_adj, Ftest (and D_adj test when present), remote test, MIA efficacy on D_f,
L1 deviation from Retrain, runtime. The Swift setting of each group is marked *.

  python collect_ablations.py \
      path/to/toxigen/unlearning-entanglement/results \
      path/to/vgg_prep/results --out-dir tables

Outputs: ablation_tables.md, ablation_tables.tex, ablation_tables.csv
"""
import argparse
import csv
import glob
import json
import os
import re

COLS = [("Df(tr)", "$\\mathcal{D}_f$"), ("Dadj(tr)", "$\\mathcal{D}_r^{adj}$"), ("Ftest(te)", "$F_{test}$"),
        ("Dadj(te)", "$\\mathcal{D}_r^{adj}$ (te)"), ("Drem(te)", "$\\mathcal{D}_r^{rem}$ (te)")]
GROUP_TITLE = {"lambda": "Retain weight $\\lambda$", "selection": "Checkpoint selection",
               "forget_distance": "Forget term", "retain_distance": "Retain term",
               "retain_mode": "Retain term sidedness", "stopgrad": "Stop-gradient on $\\mathcal{D}_{cal}$",
               "cal_fraction": "Size of $\\mathcal{D}_{cal}$", "retain_ref": "Retain reference",
               "cal_pool": "Calibration reference", "refinements": "Implementation refinements",
               "lr": "Learning rate"}
GROUP_ORDER = list(GROUP_TITLE)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0],
                                formatter_class=argparse.RawDescriptionHelpFormatter,
                                epilog="\n".join(__doc__.splitlines()[2:]))
    p.add_argument("inputs", nargs="+", help="folders (searched recursively) or ablation JSON files")
    p.add_argument("--out-dir", default=".")
    p.add_argument("--seed", type=int, help="only this seed")
    return p.parse_args()


def find_files(inputs, seed):
    files = []
    for x in inputs:
        x = os.path.expanduser(x)
        if os.path.isfile(x):
            files.append(x)
        else:
            files += glob.glob(os.path.join(x, "**", "ablation_*_seed*.json"), recursive=True)
    if seed is not None:
        files = [f for f in files if f.endswith(f"_seed{seed}.json")]
    return sorted(set(files))


def tex_label(label):
    """Variant label -> LaTeX: escape first, then turn known names into math."""
    star = "*" in label
    t = label.replace("*", "").replace("%", r"\%").replace("_", r"\_")
    for a, b in (("theta\\_0", r"$\theta_0$"), ("D\\_cal", r"$\mathcal{D}_{cal}$"),
                 ("lambda", r"$\lambda$"), ("W2", r"$W_2^2$")):
        t = t.replace(a.replace("\\\\", "\\"), b)
    return t + (r"$^\ast$" if star else "")


def main():
    args = parse_args()
    files = find_files(args.inputs, args.seed)
    if not files:
        raise SystemExit("no ablation_*_seed*.json files found")
    tables = {}
    for f in files:
        with open(f) as fh:
            p = json.load(fh)
        key = (p.get("dataset", "VGGFace2"), p.get("split", "face-size"), p.get("seed"))
        t = tables.setdefault(key, {"reference": None, "original": None, "groups": {}})
        t["reference"] = t["reference"] or p.get("reference")
        t["original"] = t["original"] or p.get("original")
        t["groups"][p["group"]] = p

    md, tex, flat = [], [], []
    for (dataset, split, seed), t in sorted(tables.items()):
        ref = t["reference"]
        cols = [c for c in COLS if ref and c[0] in ref.get("acc", {})] or \
               [c for c in COLS if any(c[0] in r.get("acc", {}) for g in t["groups"].values()
                                       for r in g["records"].values() if "acc" in r)]

        def row(name, r, dev=None, runtime=None, group=""):
            vals = [r["acc"].get(c[0], float("nan")) for c in cols]
            mia = r.get("mia", {}).get("Df", float("nan"))
            l1 = dev["L1"] if dev else (0.0 if name.startswith("Retrain") else float("nan"))
            flat.append({"dataset": dataset, "split": split, "seed": seed, "group": group, "variant": name,
                         **{c[0]: v for c, v in zip(cols, vals)}, "MIA_Df": mia, "L1": l1,
                         "runtime_s": runtime})
            return vals, mia, l1

        title = f"{dataset} ({split}), seed {seed}"
        md += [f"### {title}", "",
               "| Group | Variant | " + " | ".join(c[0] for c in cols) + " | MIA D_f | L1 | Runtime (s) |",
               "|---|---|" + "---:|" * (len(cols) + 3)]
        tex += [r"\begin{table}[t]", r"\centering", r"\small", r"\setlength{\tabcolsep}{4pt}",
                r"\begin{tabular}{l" + "r" * (len(cols) + 3) + "}", r"\toprule",
                "Variant & " + " & ".join(c[1] for c in cols) + r" & MIA$_{\mathcal{D}_f}$ & L1 & Time (s) \\",
                r"\midrule"]
        fixed = []
        if ref:
            fixed.append(("Retrain (reference)", ref, None))
        if t["original"]:
            orig_dev = None
            for g in t["groups"].values():
                orig_dev = (g.get("deviations") or {}).get("Original") or orig_dev
            fixed.append(("Original", t["original"], orig_dev))
        for name, r, dev in fixed:
            vals, mia, l1 = row(name, r, dev, None, "")
            md.append(f"| | {name} | " + " | ".join(f"{v:.2f}" for v in vals) + f" | {mia:.3f} | {l1:.2f} | |")
            tex.append(f"{name} & " + " & ".join(f"{v:.2f}" for v in vals) + f" & {mia:.3f} & {l1:.2f} & -- \\\\")
        for group in sorted(t["groups"], key=lambda g: GROUP_ORDER.index(g) if g in GROUP_ORDER else 99):
            g = t["groups"][group]
            tex += [r"\midrule", r"\multicolumn{" + str(len(cols) + 4) + r"}{l}{\textit{" +
                    GROUP_TITLE.get(group, group) + r"}} \\"]
            for label, r in g["records"].items():
                if "acc" not in r:
                    status = "failed" if r.get("failed") else "diverged"
                    md.append(f"| {group} | {label} | " + " | ".join("–" for _ in cols) + f" | – | {status} | |")
                    tex.append(f"\\quad {tex_label(label)} & \\multicolumn{{{len(cols) + 3}}}{{c}}{{{status}}} \\\\")
                    continue
                dev = (g.get("deviations") or {}).get(label)
                rt = r.get("runtime_seconds")
                vals, mia, l1 = row(label, r, dev, rt, group)
                rts = f"{rt:.0f}" if isinstance(rt, (int, float)) else "–"
                md.append(f"| {group} | {label} | " + " | ".join(f"{v:.2f}" for v in vals) +
                          f" | {mia:.3f} | {l1:.2f} | {rts} |")
                tex.append(f"\\quad {tex_label(label)} & " + " & ".join(f"{v:.2f}" for v in vals) +
                           f" & {mia:.3f} & {l1:.2f} & {rts} \\\\")
        md.append("")
        tex += [r"\bottomrule", r"\end{tabular}",
                r"\caption{Ablations of Swift$^\ast$ on " + dataset + f" ({split})." +
                r" Accuracy (\%) per set, MIA efficacy on $\mathcal{D}_f$, L1 deviation from Retrain over"
                r" the accuracy columns, and unlearning time. $^\ast$ marks the Swift$^\ast$ setting.}",
                r"\label{tab:ablation-" + re.sub(r"[^a-z0-9]+", "-", f"{dataset}-{split}".lower()) + "}",
                r"\end{table}", ""]

    os.makedirs(args.out_dir, exist_ok=True)
    with open(os.path.join(args.out_dir, "ablation_tables.md"), "w") as fh:
        fh.write("\n".join(md) + "\n")
    with open(os.path.join(args.out_dir, "ablation_tables.tex"), "w") as fh:
        fh.write("\n".join(tex) + "\n")
    with open(os.path.join(args.out_dir, "ablation_tables.csv"), "w", newline="") as fh:
        fields = []
        for r in flat:
            fields += [k for k in r if k not in fields]
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        w.writerows(flat)
    print("\n".join(md))
    print(f"[done] {len(files)} files -> {args.out_dir}/ablation_tables.{{md,tex,csv}}")


if __name__ == "__main__":
    main()
