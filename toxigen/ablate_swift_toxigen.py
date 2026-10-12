#!/usr/bin/env python
"""Ablations of Swift on ToxiGen (instance or class split).

Each variant is one run of swift_toxigen.py that differs from Swift only in the
ablated aspect, so every row uses exactly the code of a normal run. Per variant the
driver keeps the full console log and the results JSON; finished variants are skipped
on a rerun, so an interrupted ablation resumes where it stopped (--rerun forces).

  python ablate_swift_toxigen.py --split instance --ablation lambda
  python ablate_swift_toxigen.py --split class --ablation lambda --selection both
  python ablate_swift_toxigen.py --split instance --ablation all
  python ablate_swift_toxigen.py --split instance --ablation lambda --swift-args "--lr 5e-5"

Ablation groups (the Swift setting is marked *):
  lambda           *adaptive (Eq. 7), fixed lambda in {0, 0.1, 1, 10, 100}
  selection        *constrained (Eq. 6), penalty (notebook score), off (last step)
  forget_distance  *W2^2 to D_cal, squared difference of mean losses (Sec. 4.1)
  retain_distance  *W2^2 to the retain reference, squared difference of mean losses
  retain_mode      *one-sided retain term, two-sided
  stopgrad         *stop-gradient on D_cal (Eq. 4), gradient through D_cal (App. A.2)
  cal_fraction     *100% of D_cal, 50%, 25%, 10%
  retain_ref       *FitCheap probe (Eq. 5), theta_0 losses
  cal_pool         *pooled calibration reference (256, refresh 10), one batch refreshed every step
  refinements      *Swift, literal objective (no pooling, fixed lambda, two-sided, no selection)
  lr               *2e-5, 1e-5, 5e-5

--selection: as_method (default) keeps checkpoint selection on; off returns the last step;
both runs each variant twice. Use both for lambda: with lambda = 0 the constrained
selection can fall back to theta_0 and hide what the objective alone does.

Outputs (in <repo>/results/ablation_toxigen_<split>/):
  <group>/<variant>.json, <group>/<variant>.log     one per run
  ablation_<group>_seed<seed>.json                  the group table (same format as VGGFace2)
"""
import argparse
import json
import os
import re
import shlex
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))

GROUPS = {
    "lambda": [("adaptive*", [])] + [(f"fixed lambda={v:g}", ["--lambda-mode", "fixed", "--lambda-r", str(v)])
                                     for v in (0.0, 0.1, 1.0, 10.0, 100.0)],
    "selection": [("constrained*", ["--select-rule", "constrained"]),
                  ("penalty", ["--select-rule", "penalty"]),
                  ("off (last step)", ["--no-select-best"])],
    "forget_distance": [("W2*", []), ("mean", ["--forget-distance", "mean"])],
    "retain_distance": [("W2*", []), ("mean", ["--retain-distance", "mean"])],
    "retain_mode": [("one-sided*", []), ("two-sided", ["--retain-mode", "two_sided"])],
    "stopgrad": [("stop-gradient*", []), ("no stop-gradient", ["--no-cal-stopgrad"])],
    "cal_fraction": [(f"D_cal {int(v * 100)}%" + ("*" if v == 1.0 else ""),
                      [] if v == 1.0 else ["--cal-subsample", str(v)]) for v in (1.0, 0.5, 0.25, 0.1)],
    "retain_ref": [("FitCheap probe*", []), ("theta_0", ["--probe-mode", "original"])],
    "cal_pool": [("pooled*", []), ("batch, every step", ["--cal-pool", "1", "--cal-refresh", "1"])],
    "refinements": [("Swift", []), ("literal objective", ["--literal-objective"])],
    "lr": [("lr=2e-5*", []), ("lr=1e-5", ["--lr", "1e-5"]), ("lr=5e-5", ["--lr", "5e-5"])],
}


def parse_args():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0],
                                formatter_class=argparse.RawDescriptionHelpFormatter,
                                epilog="\n".join(__doc__.splitlines()[2:]))
    p.add_argument("--repo-dir", default=os.path.join(HERE, "unlearning-entanglement"))
    p.add_argument("--split", choices=["instance", "class"], required=True)
    p.add_argument("--ablation", nargs="+", required=True, help=f"any of {list(GROUPS)} or 'all'")
    p.add_argument("--selection", choices=["as_method", "off", "both"], default="as_method")
    p.add_argument("--swift-args", default="", help="extra arguments for every run, e.g. \"--lr 5e-5\"")
    p.add_argument("--seed", type=int, default=2025)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--rerun", action="store_true", help="rerun variants whose results already exist")
    p.add_argument("--dry-run", action="store_true", help="print the commands only")
    return p.parse_args()


def slug(label):
    return re.sub(r"[^A-Za-z0-9.=-]+", "_", label).strip("_")


def run_variant(cmd, log_path):
    """Run one Swift configuration, streaming its output to the console and a log file."""
    with open(log_path, "w") as log:
        log.write("$ " + " ".join(shlex.quote(c) for c in cmd) + "\n\n")
        log.flush()
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
        for line in proc.stdout:
            sys.stdout.write(line)
            log.write(line)
        return proc.wait()


def main():
    args = parse_args()
    groups = list(GROUPS) if "all" in args.ablation else args.ablation
    bad = [g for g in groups if g not in GROUPS]
    if bad:
        raise SystemExit(f"unknown ablation(s) {bad}; choose from {list(GROUPS)}")
    repo = os.path.abspath(args.repo_dir)
    out_root = os.path.join(repo, "results", f"ablation_toxigen_{args.split}")
    reference_json = os.path.join(repo, "results", f"swift_reference_{args.split}.json")
    extra = shlex.split(args.swift_args)

    sys.path.insert(0, HERE)
    import swift_common as sc                  # for the table only (no torch work here)

    for group in groups:
        variants = []
        for label, flags in GROUPS[group]:
            if group == "selection":
                variants.append((label, flags))
                continue
            if args.selection in ("as_method", "both"):
                variants.append((label, flags))
            if args.selection in ("off", "both"):
                variants.append((label + " [no selection]", flags + ["--no-select-best"]))
        gdir = os.path.join(out_root, group)
        os.makedirs(gdir, exist_ok=True)
        print("\n" + "#" * 78 + f"\n# {args.split}: ablation {group} ({len(variants)} runs) -> {gdir}\n" + "#" * 78)

        records, rows, runtimes = {}, {}, {}
        for k, (label, flags) in enumerate(variants):
            res_json = os.path.join(gdir, slug(label) + ".json")
            cmd = [sys.executable, os.path.join(HERE, "swift_toxigen.py"), "--repo-dir", repo,
                   "--split", args.split, "--seed", str(args.seed), "--device", args.device,
                   "--results-json", res_json, "--no-save-model", *extra, *flags]
            if k > 0:
                cmd.append("--skip-original-eval")         # Original is evaluated once per group
            if args.dry_run:
                print(" ".join(shlex.quote(c) for c in cmd))
                continue
            if os.path.exists(res_json) and not args.rerun:
                print(f"[skip] {label}: {res_json} exists (use --rerun to redo)")
            else:
                print("\n" + "=" * 78 + f"\n[{group}] {label}: {' '.join(flags) or '(Swift setting)'}\n" + "=" * 78,
                      flush=True)
                code = run_variant(cmd, res_json.replace(".json", ".log"))
                if code != 0 or not os.path.exists(res_json):
                    print(f"[{label}] FAILED (exit {code}); see {res_json.replace('.json', '.log')}")
                    records[label] = {"flags": flags, "failed": True, "exit_code": code}
                    continue
            with open(res_json) as fh:
                payload = json.load(fh)
            res = payload["results"]
            if "Original" in res and "Original" not in rows:
                rows["Original"] = res["Original"]
            swift = res["Swift"]
            t = payload.get("timings", {})
            runtime = t.get("probe_fit", 0.0) + t.get("optimisation", 0.0)
            rows[label], runtimes[label] = swift, runtime
            records[label] = {"flags": flags, "results_json": res_json, "runtime_seconds": runtime,
                              "best_select": payload.get("best_select"), **swift}
        if args.dry_run:
            continue

        table = {}
        if os.path.exists(reference_json):
            with open(reference_json) as fh:
                table["Retrain (reference)"] = json.load(fh)
        else:
            print(f"[note] no reference at {reference_json}; run retrain_reference.py --split {args.split}")
        table.update(rows)
        print(f"\n########## {args.split}: {group} ##########")
        dev = sc.print_comparison(table, "Retrain (reference)" if "Retrain (reference)" in table else None,
                                  runtimes)
        out = os.path.join(out_root, f"ablation_{group}_seed{args.seed}.json")
        with open(out, "w") as fh:
            json.dump({"dataset": "ToxiGen", "split": args.split, "group": group, "seed": args.seed,
                       "selection": args.selection, "swift_args": args.swift_args,
                       "records": records, "deviations": dev,
                       "reference": table.get("Retrain (reference)"), "original": rows.get("Original")},
                      fh, indent=2, default=str)
        print(f"[done] {out}")


if __name__ == "__main__":
    main()
