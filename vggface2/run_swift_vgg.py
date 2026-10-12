#!/usr/bin/env python
"""Step 4: run Swift on the VGGFace2 face-size sets.

Swift starts from a copy of the Original checkpoint (train_vgg.py --mode original), is
timed, evaluated with the same metrics as Original and Retrain, and printed in one table
scored against Retrain.

    python run_swift_vgg.py --prep-dir /data/vgg_prep
    python run_swift_vgg.py --prep-dir /data/vgg_prep --hp lr=1e-3 probe_mode=fitcheap
    python run_swift_vgg.py --prep-dir /data/vgg_prep --grid lr=1e-3,3e-3,1e-2

Hyperparameters: vgg_swift.DEFAULTS, updated by SWIFT_HP below, then by --hp key=value.

Learning-rate sweep (--grid): every configuration is run from the Original model and one
retrain-free rule picks the result: a configuration is eligible if its accuracy on held-out
validation images of the remote identities drops by at most --sweep-tol points against
Original; among eligible ones the lowest held-out W2_f wins; if none is eligible, the
smallest drop wins. Test sets and Retrain are never consulted.

Runtime: probe fit + optimisation (CUDA-synchronised; loading and evaluation excluded).
Outputs: <results>/vgg_swift_seed<seed>.json (+ _diag.json), <checkpoints>/vgg_swift_seed<seed>.pt,
<results>/vgg_swift_summary_seed<seed>.json.
"""
import argparse
import ast
import copy
import itertools
import json
import os
import time

import torch

import vgg_common as vc

RUNNER_VERSION = "2026-09-25.4"

# vgg_swift.DEFAULTS are the CIFAR-100 notebook values. Changes for this from-scratch,
# SGD-trained ResNet-50 (BatchNorm frozen during unlearning): theta_0's losses as the retain
# reference, and SGD instead of AdamW (Adam's per-weight step of ~lr is large relative to
# this model's small weights and collapses it).
SWIFT_HP = dict(probe_mode="original", optimizer="sgd", lr=3e-3, momentum=0.9, weight_decay=0.0)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0],
                                formatter_class=argparse.RawDescriptionHelpFormatter,
                                epilog="\n".join(__doc__.splitlines()[2:]))
    p.add_argument("--prep-dir", required=True)
    p.add_argument("--sets", help="default <prep-dir>/sets.json")
    p.add_argument("--hp", nargs="*", default=[], help="overrides: key=value")
    p.add_argument("--grid", nargs="*", default=[], help="sweep: key=v1,v2,... (several keys = all combinations)")
    p.add_argument("--sweep-tol", type=float, default=2.0,
                   help="max drop (points) in validation remote accuracy vs Original for a configuration "
                        "to be eligible")
    p.add_argument("--train-seed", type=int, default=42,
                   help="seed of the Original/Retrain runs to load (train_vgg.py --seed)")
    p.add_argument("--checkpoint", help="default <prep>/checkpoints/vgg_original_seed<train-seed>.pt")
    p.add_argument("--seed", type=int, default=2025, help="unlearning seed")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--results-dir", help="default <prep-dir>/results")
    p.add_argument("--out-dir", help="default <prep-dir>/checkpoints")
    p.add_argument("--no-save-model", action="store_true")
    return p.parse_args()


def _value(raw):
    try:
        return ast.literal_eval(raw)
    except (ValueError, SyntaxError):
        return raw


def parse_overrides(items):
    out = {}
    for item in items:
        if "=" not in item:
            raise SystemExit(f"--hp expects key=value, got {item!r}")
        k, v = item.split("=", 1)
        out[k] = _value(v)
    return out


def parse_grid(items):
    out = {}
    for item in items:
        if "=" not in item:
            raise SystemExit(f"--grid expects key=v1,v2,..., got {item!r}")
        k, raw = item.split("=", 1)
        out[k] = [_value(v) for v in raw.split(",")]
    return out


def expand_grid(base, grid):
    if not grid:
        return [dict(base)]
    keys = list(grid)
    return [{**base, **dict(zip(keys, combo))} for combo in itertools.product(*(grid[k] for k in keys))]


def rank_key(trial):
    if trial["feasible"]:
        return (0, trial["own_objective"])
    return (1, trial["val_remote_drop"])


def main():
    args = parse_args()
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    vc.fast_cuda()
    res_dir = args.results_dir or os.path.join(args.prep_dir, "results")
    out_dir = args.out_dir or os.path.join(args.prep_dir, "checkpoints")
    ckpt = args.checkpoint or os.path.join(args.prep_dir, "checkpoints", f"vgg_original_seed{args.train_seed}.pt")
    if not os.path.exists(ckpt):
        raise SystemExit(f"[error] {ckpt} not found; run train_vgg.py --mode original first")
    os.makedirs(res_dir, exist_ok=True)

    import vgg_swift
    versions = {"run_swift_vgg.py": RUNNER_VERSION,
                "vgg_common.py": getattr(vc, "PIPELINE_VERSION", "old"),
                "vgg_swift.py": getattr(vgg_swift, "PIPELINE_VERSION", "old")}
    if len(set(versions.values())) != 1:
        raise SystemExit(f"[error] files from different versions of the pipeline: {versions}")

    info, meta, mm = vc.load_prep(args.prep_dir)
    sets = vc.load_sets(args.sets or os.path.join(args.prep_dir, "sets.json"), info)
    data = vc.GPUImages(mm, meta["label"].to_numpy(), device)
    original, blob = vc.load_checkpoint(ckpt, device)
    original.eval()
    print(f"[original] {ckpt} ({blob['arch']}, {blob['num_classes']} classes)")
    val_rem = data.idx(sets["sets"]["val"]["val_remote"])
    orig_val_rem = vc.acc_of(original, data, val_rem)
    print(f"[original] val_remote accuracy {orig_val_rem:.2f}% (reference for the sweep rule)")

    base = {**SWIFT_HP, **parse_overrides(args.hp)}
    grid = parse_grid(args.grid)
    configs = expand_grid(base, grid)
    print("\n" + "=" * 78 + f"\n[Swift] seed={args.seed} base hp={base}"
          + (f"\n  sweep over {len(configs)} configs: {grid}" if len(configs) > 1 else "") + "\n" + "=" * 78,
          flush=True)

    trials, best = [], None
    t_sweep = time.perf_counter()
    diag_path = os.path.join(res_dir, f"vgg_swift_seed{args.seed}_diag.json")
    for k, hp in enumerate(configs):
        if len(configs) > 1:
            print(f"\n--- [Swift] config {k + 1}/{len(configs)}: { {g: hp[g] for g in grid} }", flush=True)
        vc.set_seed(args.seed)
        model = copy.deepcopy(original)
        if device.type == "cuda":
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats(device)
        unlearned, sinfo = vgg_swift.run_swift(model, data, sets, meta, seed=args.seed, **hp)
        vc.sync(device)
        stage_times = sinfo["timings"]
        runtime = stage_times["probe_fit"] + stage_times["optimisation"]   # excludes diagnostics
        peak = torch.cuda.max_memory_allocated(device) / 1e6 if device.type == "cuda" else None
        unlearned.eval()
        print(f"[Swift] unlearning took {runtime:.1f}s ("
              + ", ".join(f"{a}={b:.1f}s" for a, b in stage_times.items()) + ")"
              + (f", peak GPU {peak:.0f} MB" if peak else ""), flush=True)
        trial = {"hyperparameters": hp, "runtime_seconds": runtime, "stage_seconds": stage_times,
                 "peak_gpu_mb": peak,
                 "diverged": not all(torch.isfinite(p).all() for p in unlearned.parameters())}
        del model
        if trial["diverged"]:
            print("[Swift] DIVERGED (non-finite parameters); config skipped", flush=True)
            trials.append(trial)
            continue
        trial["val_remote_acc"] = vc.acc_of(unlearned, data, val_rem)
        trial["val_remote_drop"] = orig_val_rem - trial["val_remote_acc"]
        trial.update(vc.full_evaluation(unlearned, data, sets, tag="Swift"))
        trial["swift"] = {a: b for a, b in sinfo.items() if a not in ("diag", "track")}
        trial["own_objective"] = sinfo["best_select"].get("w_f", float("nan"))
        trial["feasible"] = trial["val_remote_drop"] <= args.sweep_tol
        trials.append(trial)
        if best is None or rank_key(trial) < rank_key(best[0]):
            best = (trial, unlearned)
            with open(diag_path, "w") as fh:
                json.dump({"diag": sinfo["diag"], "track": sinfo["track"]}, fh)
        else:
            del unlearned
        if device.type == "cuda":
            torch.cuda.empty_cache()

    if best is None:
        raise SystemExit("[Swift] every configuration diverged")
    chosen, unlearned = best
    if len(configs) > 1:
        print(f"\n[Swift] sweep summary (rule: val_remote drop <= {args.sweep_tol} pts, then lowest held-out W2_f):")
        for t in trials:
            if t["diverged"]:
                print(f"  {t['hyperparameters']}: diverged")
                continue
            print(f"  { {g: t['hyperparameters'][g] for g in grid} }  val_remote drop={t['val_remote_drop']:+.2f}  "
                  f"Df={t['acc']['Df(tr)']:.2f}  W2_f={t['own_objective']:.4f}  feasible={t['feasible']}"
                  + ("  <== chosen" if t is chosen else ""))
        if not chosen["feasible"]:
            print("[Swift] WARNING: no configuration met the rule; chose the smallest val_remote drop")

    record = {"name": "Swift", "seed": args.seed, "train_seed": args.train_seed, **chosen,
              "sweep_tol": args.sweep_tol, "sweep_seconds": time.perf_counter() - t_sweep,
              "trials": [{a: b for a, b in t.items() if a != "swift"} for t in trials] if len(trials) > 1 else None}
    with open(os.path.join(res_dir, f"vgg_swift_seed{args.seed}.json"), "w") as fh:
        json.dump(record, fh, indent=2, default=str)
    if not args.no_save_model:
        vc.save_checkpoint(os.path.join(out_dir, f"vgg_swift_seed{args.seed}.pt"), unlearned,
                           {"arch": blob["arch"], "num_classes": blob["num_classes"],
                            "img_size": blob.get("img_size"), "method": "swift", "seed": args.seed,
                            "hyperparameters": chosen["hyperparameters"]})

    # ------------------------------------------------------------------ table
    table, rt = {}, {}
    for mode, name in (("retrain", "Retrain (reference)"), ("original", "Original")):
        path = os.path.join(res_dir, f"vgg_{mode}_seed{args.train_seed}.json")
        if os.path.exists(path):
            with open(path) as fh:
                table[name] = json.load(fh)
            if mode == "retrain":
                rt[name] = table[name]["train_seconds"]
    if "Retrain (reference)" not in table:
        print(f"[note] no Retrain results for seed {args.train_seed}; run train_vgg.py --mode retrain")
    table["Swift"], rt["Swift"] = {k: chosen[k] for k in ("acc", "loss", "mia")}, chosen["runtime_seconds"]
    dev = vc.print_comparison(table, "Retrain (reference)" if "Retrain (reference)" in table else None, rt)
    summary = os.path.join(res_dir, f"vgg_swift_summary_seed{args.seed}.json")
    with open(summary, "w") as fh:
        json.dump({"seed": args.seed, "train_seed": args.train_seed, "runtime_seconds": rt,
                   "results": {k: {kk: v[kk] for kk in ("acc", "loss", "mia") if kk in v} for k, v in table.items()},
                   "deviations": dev}, fh, indent=2, default=str)
    print(f"\n[done] {summary}")


if __name__ == "__main__":
    main()
