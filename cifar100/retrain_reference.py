#!/usr/bin/env python
"""Counterfactual reference for Swift: retrain ResNet-18 on D_r = D_r^adj + D_r^rem.

Uses the Original model's recipe (the same presets as pretrain_original.py) on
the Swift sets of the chosen split, then evaluates with exactly the metrics
swift_cifar100.py reports. Writes results/swift_reference_<split>_c<class>.json,
which swift_cifar100.py picks up to score deviations from the counterfactual.

  class split     D_r drops the forget fine class entirely
  instance split  D_r keeps the other half of it (still carrying y0)

Retrain is a reference, not a target: on CIFAR-100 it keeps substantial D_f
accuracy (the super-class label survives through D_r^adj), so a method that
drives D_f to zero has over-forgotten rather than approximated it.

--eval-checkpoint PATH skips training and evaluates an existing model on the
same sets (e.g. the Original model), writing results/swift_eval_<tag>_<split>_c<class>.json.
"""
import argparse
import json
import os
import time
from dataclasses import asdict

import torch
from torch.utils.data import DataLoader, Subset

import swift_common as sc
from cifar100 import N_SUPERCLASSES, SuperClassDataset, fine_name, make_loader
from pretrain_original import PRESETS, train
from swift_sets import add_sets_args, build_sets, describe, sets_config_from_args


def parse_args():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0],
                                formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    add_sets_args(p)
    p.add_argument("--preset", choices=sorted(PRESETS), default="paper",
                   help="use the same preset as the Original model")
    p.add_argument("--epochs", type=int)
    p.add_argument("--lr", type=float)
    p.add_argument("--batch-size", type=int)
    p.add_argument("--eval-checkpoint", help="evaluate this model instead of training")
    p.add_argument("--tag", default="model", help="name used with --eval-checkpoint")
    p.add_argument("--output", help="default checkpoints/swift_retrain_<split>_c<class>.pt")
    p.add_argument("--results-json")
    p.add_argument("--no-save-model", action="store_true")
    p.add_argument("--seed", type=int, default=2024)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--amp", choices=["bf16", "fp16", "none"], default="none")
    p.add_argument("--tf32", action="store_true", help="TF32 matmul (H100)")
    p.add_argument("--num-workers", type=int, default=8)
    p.add_argument("--eval-batch", type=int, default=256)
    p.add_argument("--mia-seed", type=int, default=7)
    p.add_argument("--mia-n-members", type=int, default=2000)
    return p.parse_args()


def eval_loaders_for(sets, eval_batch, num_workers, split):
    """The reported sets. Dadj(te) exists only in the class split: in the
    instance split both halves of D_f share one test population, which is
    Ftest."""
    train_base, test_base = sets["train_base"], sets["test_base"]
    loaders = {
        "Df(tr)": make_loader(train_base, sets["train"]["forget"], eval_batch,
                              num_workers=num_workers),
        "Dadj(tr)": make_loader(train_base, sets["train"]["adjacent"], eval_batch,
                                num_workers=num_workers),
        "Drem(tr)": make_loader(train_base, sets["train"]["remote"], eval_batch,
                                num_workers=num_workers),
        "Ftest(te)": make_loader(test_base, sets["test"]["ftest"], eval_batch,
                                 num_workers=num_workers),
        "Drem(te)": make_loader(test_base, sets["test"]["remote_te"], eval_batch,
                                num_workers=num_workers),
    }
    if split == "class":
        loaders["Dadj(te)"] = make_loader(test_base, sets["test"]["adjacent_te"],
                                          eval_batch, num_workers=num_workers)
    return loaders


def main():
    args = parse_args()
    hp = dict(PRESETS[args.preset])
    for key in ("epochs", "lr", "batch_size"):
        if getattr(args, key) is not None:
            hp[key] = getattr(args, key)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    amp_dtype, use_scaler = sc.resolve_amp(args.amp, device)
    if args.tf32:
        sc.enable_fast_math()
    sc.set_seed(args.seed)

    cfg = sets_config_from_args(args)
    sets = build_sets(cfg, args.data_root)
    print(describe(cfg, sets))

    stem = f"{cfg.split}_c{cfg.forget_class}"
    is_eval_only = bool(args.eval_checkpoint)
    results_json = args.results_json or (
        f"results/swift_eval_{args.tag}_{stem}.json" if is_eval_only
        else f"results/swift_reference_{stem}.json")
    output = args.output or f"checkpoints/swift_retrain_{stem}.pt"

    eval_loaders = eval_loaders_for(sets, args.eval_batch, args.num_workers, cfg.split)
    seconds = None

    if is_eval_only:
        print(f"[{args.tag}] evaluating {args.eval_checkpoint}")
        model = sc.load_model(args.eval_checkpoint, device, N_SUPERCLASSES)
        tag = args.tag.capitalize()
    else:
        train_base, test_base = sets["train_base"], sets["test_base"]
        retain_ds = Subset(SuperClassDataset(train_base),
                           sets["train"]["retain"].tolist())
        train_loader = DataLoader(retain_ds, batch_size=hp["batch_size"], shuffle=True,
                                  num_workers=args.num_workers, pin_memory=True)
        # Validation excludes nothing: the test split is disjoint from training
        # in both splits, so the whole of it is a fair early-stopping signal.
        val_loader = DataLoader(SuperClassDataset(test_base), batch_size=args.eval_batch,
                                shuffle=False, num_workers=args.num_workers,
                                pin_memory=True)
        print(f"[retrain] split={cfg.split} forget class={cfg.forget_class} "
              f"({fine_name(cfg.forget_class)}) |D_r|={len(retain_ds)} {hp}")

        model = sc.resnet18_superclass(N_SUPERCLASSES, pretrained=hp["pretrained"]).to(device)
        best_state, history, seconds = train(model, train_loader, val_loader, device, hp,
                                             amp_dtype, use_scaler, log_label="retrain")
        if best_state is not None:
            model.load_state_dict(best_state)
        model.eval()
        print(f"[retrain] trained in {seconds / 60:.1f} min")
        if not args.no_save_model:
            sc.save_model(model, output, extra={"split": cfg.split, "hp": hp,
                                                "seed": args.seed})
            print(f"[retrain] model -> {output}")
        tag = "Retrain"

    t0 = time.perf_counter()
    result = sc.full_evaluation(model, eval_loaders, device, amp_dtype,
                                args.mia_n_members, args.mia_seed, tag=tag)
    eval_seconds = time.perf_counter() - t0

    os.makedirs(os.path.dirname(results_json) or ".", exist_ok=True)
    payload = {"pipeline_version": sc.PIPELINE_VERSION, "tag": tag,
               "config": dict(vars(args)), "sets_config": asdict(cfg),
               "sizes": sets["sizes"], "preset": args.preset, "hp": hp,
               "train_seconds": seconds, "eval_seconds": eval_seconds,
               **sc.strip_losses(result)}
    if not is_eval_only:
        payload["history"] = history
    with open(results_json, "w") as fh:
        json.dump(payload, fh, indent=2, default=str)
    print(f"[{tag}] results -> {results_json}")


if __name__ == "__main__":
    main()
