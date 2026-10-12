#!/usr/bin/env python
"""Ablations of Swift on the VGGFace2 face-size split.

Every configuration starts from a fresh copy of the Original model and differs from
Swift (the settings run_swift_vgg.py uses, plus any --hp overrides) only in
the ablated aspect. One table per run: Retrain, Original and every configuration,
with deviation from Retrain and runtime.

  python ablate_swift_vgg.py --prep-dir $PREP --ablation lambda
  python ablate_swift_vgg.py --prep-dir $PREP --ablation lambda --selection both
  python ablate_swift_vgg.py --prep-dir $PREP --ablation forget_distance stopgrad cal_fraction
  python ablate_swift_vgg.py --prep-dir $PREP --ablation all --hp lr=3e-3   # if your sweep chose 3e-3

Ablation groups (the Swift setting is marked *):
  lambda           *adaptive (Eq. 7), fixed lambda in {0, 0.1, 1, 10, 100}
  selection        *constrained (Eq. 6), penalty (notebook score), off (last step)
  forget_distance  *W2^2 to D_cal, squared difference of mean losses (Sec. 4.1 claim)
  retain_distance  *W2^2 to the retain reference, squared mean difference
  retain_mode      *one-sided retain term, two-sided
  stopgrad         *stop-gradient on D_cal (Eq. 4), gradient through D_cal (App. A.2 claim)
  cal_fraction     *100% of D_cal, 50%, 25%, 10%
  retain_ref       *theta_0 losses (current VGG setting), FitCheap probe
  cal_pool         *pooled calibration reference (256, refresh 10), batch-sized refreshed every step
  lr               1e-3 (default), 3e-4, 3e-3

--selection: as_method (default) keeps checkpoint selection on; off returns the last
step; both runs every configuration twice. For lambda, 'both' separates what the
objective does from what selection rescues (with lambda = 0 the constrained selection
can fall back to theta_0).

Outputs: <prep>/results/ablation_<group>_seed<seed>.json, one per group.
"""
import argparse
import ast
import copy
import json
import os
import time

import torch

import vgg_common as vc

GROUPS = {
    "lambda": [("adaptive*", {}),
               *[(f"fixed lambda={v:g}", {"lambda_mode": "fixed", "lambda_r": v}) for v in (0.0, 0.1, 1.0, 10.0, 100.0)]],
    "selection": [("constrained*", {"select_rule": "constrained"}), ("penalty", {"select_rule": "penalty"}),
                  ("off (last step)", {"select_best": False})],
    "forget_distance": [("W2*", {"forget_distance": "w2"}), ("mean", {"forget_distance": "mean"})],
    "retain_distance": [("W2*", {"retain_distance": "w2"}), ("mean", {"retain_distance": "mean"})],
    "retain_mode": [("one-sided*", {"retain_mode": "one_sided"}), ("two-sided", {"retain_mode": "two_sided"})],
    "stopgrad": [("stop-gradient*", {"cal_stopgrad": True}), ("no stop-gradient", {"cal_stopgrad": False})],
    "cal_fraction": [(f"D_cal {int(v * 100)}%" + ("*" if v == 1.0 else ""), {"cal_fraction": v})
                     for v in (1.0, 0.5, 0.25, 0.1)],
    "retain_ref": [("theta_0*", {"probe_mode": "original"}), ("FitCheap probe", {"probe_mode": "fitcheap"})],
    "cal_pool": [("pooled*", {}), ("batch, every step", {"cal_pool": 64, "cal_refresh": 1})],
    "lr": [("lr=1e-3*", {"lr": 1e-3}), ("lr=3e-4", {"lr": 3e-4}), ("lr=3e-3", {"lr": 3e-3})],
}


def parse_args():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0],
                                formatter_class=argparse.RawDescriptionHelpFormatter,
                                epilog="\n".join(__doc__.splitlines()[2:]))
    p.add_argument("--prep-dir", required=True)
    p.add_argument("--sets", help="default <prep-dir>/sets.json")
    p.add_argument("--ablation", nargs="+", required=True, help=f"any of {list(GROUPS)} or 'all'")
    p.add_argument("--selection", choices=["as_method", "off", "both"], default="as_method")
    p.add_argument("--hp", nargs="*", default=[], help="overrides of the Swift base config: key=value")
    p.add_argument("--train-seed", type=int, default=42)
    p.add_argument("--checkpoint", help="default <prep>/checkpoints/vgg_original_seed<train-seed>.pt")
    p.add_argument("--seed", type=int, default=2025)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--results-dir", help="default <prep-dir>/results")
    return p.parse_args()


def main():
    args = parse_args()
    groups = list(GROUPS) if "all" in args.ablation else args.ablation
    bad = [g for g in groups if g not in GROUPS]
    if bad:
        raise SystemExit(f"unknown ablation(s) {bad}; choose from {list(GROUPS)}")
    base = {}
    for item in args.hp:
        k, v = item.split("=", 1)
        try:
            base[k] = ast.literal_eval(v)
        except (ValueError, SyntaxError):
            base[k] = v

    import vgg_swift
    from run_swift_vgg import RUNNER_VERSION, SWIFT_HP
    versions = {RUNNER_VERSION, getattr(vc, "PIPELINE_VERSION", "old"), getattr(vgg_swift, "PIPELINE_VERSION", "old")}
    if len(versions) != 1:
        raise SystemExit(f"[error] files from different versions {versions}; check out the vggface2/ folder again")
    swift_base = {**SWIFT_HP, **base}

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    vc.fast_cuda()
    res_dir = args.results_dir or os.path.join(args.prep_dir, "results")
    os.makedirs(res_dir, exist_ok=True)
    ckpt = args.checkpoint or os.path.join(args.prep_dir, "checkpoints", f"vgg_original_seed{args.train_seed}.pt")
    info, meta, mm = vc.load_prep(args.prep_dir)
    sets = vc.load_sets(args.sets or os.path.join(args.prep_dir, "sets.json"), info)
    data = vc.GPUImages(mm, meta["label"].to_numpy(), device)
    original, _ = vc.load_checkpoint(ckpt, device)
    print(f"[ablation] Swift base config: {swift_base}")

    reference = {}
    for mode, name in (("retrain", "Retrain (reference)"), ("original", "Original")):
        path = os.path.join(res_dir, f"vgg_{mode}_seed{args.train_seed}.json")
        if os.path.exists(path):
            with open(path) as fh:
                reference[name] = json.load(fh)

    for group in groups:
        variants = []
        for label, cfg in GROUPS[group]:
            if group == "selection":                  # this group varies selection itself
                variants.append((label, cfg))
                continue
            if args.selection in ("as_method", "both"):
                variants.append((label, cfg))
            if args.selection in ("off", "both"):
                variants.append((label + " [no selection]", {**cfg, "select_best": False}))
        rows, runtimes, records = dict(reference), {}, {}
        if "Retrain (reference)" in reference:
            runtimes["Retrain (reference)"] = reference["Retrain (reference)"].get("train_seconds", float("nan"))
        print("\n" + "#" * 78 + f"\n# ablation: {group} ({len(variants)} runs)\n" + "#" * 78)
        for label, cfg in variants:
            hp = {**swift_base, **cfg}
            print("\n" + "=" * 78 + f"\n[{group}] {label}: {cfg}\n" + "=" * 78, flush=True)
            vc.set_seed(args.seed)
            model = copy.deepcopy(original)
            if device.type == "cuda":
                torch.cuda.empty_cache()
            t0 = time.perf_counter()
            try:
                unlearned, sinfo = vgg_swift.run_swift(model, data, sets, meta, seed=args.seed, **hp)
            except RuntimeError as exc:          # e.g. non-finite loss
                print(f"[{label}] FAILED: {exc}")
                records[label] = {"config": cfg, "failed": str(exc)}
                continue
            vc.sync(device)
            runtime = sinfo["timings"]["probe_fit"] + sinfo["timings"]["optimisation"]
            if not all(torch.isfinite(p).all() for p in unlearned.parameters()):
                print(f"[{label}] DIVERGED; not evaluated")
                records[label] = {"config": cfg, "diverged": True}
                continue
            res = vc.full_evaluation(unlearned, data, sets, tag=label)
            lam = sinfo.get("diag", {}).get("lambda", [])
            extra = {"final_lambda": lam[-1] if lam else None, "max_lambda": max(lam) if lam else None,
                     "selected_step": sinfo["best_select"].get("step"),
                     "selected_feasible": sinfo["best_select"].get("feasible"),
                     "wall_seconds": time.perf_counter() - t0}
            print(f"[{label}] runtime {runtime:.1f}s | lambda final={extra['final_lambda']} "
                  f"max={extra['max_lambda']} | selected step={extra['selected_step']} "
                  f"feasible={extra['selected_feasible']}")
            rows[label], runtimes[label] = res, runtime
            records[label] = {"config": cfg, "hyperparameters": hp, "runtime_seconds": runtime, **extra,
                              **res, "gaps": sinfo.get("gaps")}
            del model, unlearned

        print(f"\n########## {group} ##########")
        dev = vc.print_comparison(rows, "Retrain (reference)" if "Retrain (reference)" in rows else None, runtimes)
        out = os.path.join(res_dir, f"ablation_{group}_seed{args.seed}.json")
        with open(out, "w") as fh:
            strip = lambda r: {k: r[k] for k in ("acc", "mia") if k in r} if r else None  # noqa: E731
            json.dump({"dataset": "VGGFace2", "split": "face-size", "group": group, "seed": args.seed,
                       "base": swift_base, "selection": args.selection, "records": records, "deviations": dev,
                       "reference": strip(reference.get("Retrain (reference)")),
                       "original": strip(reference.get("Original"))}, fh, indent=2, default=str)
        print(f"[done] {out}")


if __name__ == "__main__":
    main()
