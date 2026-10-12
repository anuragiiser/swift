#!/usr/bin/env python
"""Step 2: train the Original model or the retrained reference on SBIC, then evaluate.

  --mode original   RoBERTa-base on every train post (D_f + D_r), true offense labels.
                    One Original serves both splits (it does not depend on the sets).
                    -> <work>/checkpoints/sbic_original, results/sbic_original_<split>.json
  --mode retrain    the same recipe on D_r = D_r^adj + D_r^rem of the chosen split
                    (the counterfactual) -> checkpoints/sbic_retrain_<split>,
                    results/swift_reference_<split>.json (the reference every table uses)
  --eval-checkpoint PATH   no training: evaluate an existing model on the split's sets,
                    -> results/swift_eval_<tag>_<split>.json

Recipe (both modes): AdamW lr 2e-5, batch 64, 3 epochs, linear schedule, 10% warmup,
seed 42 (--preset paper: 10 epochs, lr 2.5e-5, batch 128). Evaluation: accuracy on
D_f, D_adj, D_rem (train), Ftest, Dadj(te), Drem(te) and the loss-threshold MIA.

  python train_sbic.py --mode original --split class
  python train_sbic.py --mode retrain  --split class
  python train_sbic.py --mode retrain  --split instance
  python train_sbic.py --eval-checkpoint sbic_work/checkpoints/sbic_original --tag original --split instance
"""
import argparse
import contextlib
import json
import os
import time
from dataclasses import asdict

from repro_utils import add_repo_arg, check_torch_for_transformers, import_data_module, read_forget_group
from swift_sets import (add_sets_args, default_sets_path, load_or_build_sets,
                        sets_config_from_args, subset)

PRESETS = {
    "repo":  dict(epochs=3,  lr=2e-5,   batch_size=64,  warmup_ratio=0.1, seed=42),
    "paper": dict(epochs=10, lr=2.5e-5, batch_size=128, warmup_ratio=0.1, seed=42),
}


def parse_args():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0],
                                formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    add_repo_arg(p)
    add_sets_args(p)
    p.add_argument("--mode", choices=["original", "retrain"], default="retrain")
    p.add_argument("--preset", choices=sorted(PRESETS), default="repo",
                   help="use the same preset for Original and Retrain")
    p.add_argument("--epochs", type=int)
    p.add_argument("--lr", type=float)
    p.add_argument("--batch-size", type=int)
    p.add_argument("--warmup-ratio", type=float)
    p.add_argument("--seed", type=int)
    p.add_argument("--eval-checkpoint", help="evaluate this model instead of training")
    p.add_argument("--tag", default="model", help="name used with --eval-checkpoint")
    p.add_argument("--output-model", help="default <work>/checkpoints/sbic_original or sbic_retrain_<split>")
    p.add_argument("--results-json")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--amp", action="store_true", help="bfloat16 autocast for training")
    p.add_argument("--num-proc", type=int, default=4)
    p.add_argument("--eval-batch", type=int, default=256)
    p.add_argument("--log-every", type=int, default=500)
    p.add_argument("--mia-seed", type=int, default=7)
    p.add_argument("--mia-n-members", type=int, default=2000)
    return p.parse_args()


def main():
    args = parse_args()
    check_torch_for_transformers()
    import torch
    import torch.nn.functional as F
    from torch.optim import AdamW
    from transformers import AutoModelForSequenceClassification, get_linear_schedule_with_warmup

    import swift_common as sc

    repo, split = args.repo_dir, args.split
    cfg_train = dict(PRESETS[args.preset])
    for k in cfg_train:
        if getattr(args, k) is not None:
            cfg_train[k] = getattr(args, k)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    tg = import_data_module(repo)
    fg = read_forget_group(repo)
    ds_tok, collator, tokenizer = tg.load_and_tokenize(num_proc=args.num_proc)
    cfg = sets_config_from_args(args, fg)
    sets = load_or_build_sets(ds_tok, cfg, default_sets_path(repo, split), rebuild=args.rebuild_sets)
    S = lambda sp, name: subset(ds_tok, sp, sets[sp][name], fg)  # noqa: E731
    L = lambda ds, bs, **kw: sc.make_loader(ds, bs, collator, **kw)  # noqa: E731

    eval_loaders = {"Df(tr)": L(S("train", "forget"), args.eval_batch),
                    "Dadj(tr)": L(S("train", "adjacent"), args.eval_batch),
                    "Drem(tr)": L(S("train", "remote"), args.eval_batch),
                    "Ftest(te)": L(S("test", "ftest"), args.eval_batch),
                    "Drem(te)": L(S("test", "remote_te"), args.eval_batch)}
    if len(sets["test"]["adjacent_te"]):
        eval_loaders["Dadj(te)"] = L(S("test", "adjacent_te"), args.eval_batch)

    train_seconds = None
    if args.eval_checkpoint:
        name = args.tag
        model = AutoModelForSequenceClassification.from_pretrained(args.eval_checkpoint).to(device)
        results_json = args.results_json or os.path.join(repo, "results", f"swift_eval_{args.tag}_{split}.json")
    else:
        import numpy as np
        sc.set_seed(cfg_train["seed"])
        if args.mode == "original":
            name = "Original"
            results_json = args.results_json or os.path.join(repo, "results", f"sbic_original_{split}.json")
            train_ds = subset(ds_tok, "train", np.arange(len(ds_tok["train"])), fg)
            print(f"[original] all {len(train_ds):,} train posts, true labels | {cfg_train}")
        else:
            name = "Retrain"
            results_json = args.results_json or os.path.join(repo, "results", f"swift_reference_{split}.json")
            train_ds = S("train", "retain")
            assert len(train_ds) + len(sets["train"]["forget"]) == len(ds_tok["train"])
            print(f"[retrain] split={split} D_r={len(train_ds):,} (D_f={len(sets['train']['forget']):,} "
                  f"excluded) | {cfg_train}")
        loader = L(train_ds, cfg_train["batch_size"], shuffle=True)
        print(f"[{args.mode}] {len(loader):,} steps/epoch")

        model = AutoModelForSequenceClassification.from_pretrained("roberta-base", num_labels=2).to(device)
        optim = AdamW(model.parameters(), lr=cfg_train["lr"])
        total = cfg_train["epochs"] * len(loader)
        sched = get_linear_schedule_with_warmup(optim, int(cfg_train["warmup_ratio"] * total), total)
        use_amp = args.amp and device.type == "cuda" and torch.cuda.is_bf16_supported()
        ctx = (lambda: torch.autocast("cuda", dtype=torch.bfloat16)) if use_amp else contextlib.nullcontext

        t0, gstep = time.time(), 0
        for epoch in range(cfg_train["epochs"]):
            model.train()
            running, n = 0.0, 0
            for step, batch in enumerate(loader):
                optim.zero_grad()
                with ctx():
                    logits, y = sc.forward_logits(model, batch, device)
                loss = F.cross_entropy(logits.float(), y)
                loss.backward()
                optim.step()
                sched.step()
                running += loss.item()
                n += 1
                gstep += 1
                if step % args.log_every == 0 or step == len(loader) - 1:
                    el = time.time() - t0
                    print(f"[{args.mode}] epoch {epoch + 1}/{cfg_train['epochs']} step {step + 1}/{len(loader)} "
                          f"loss={running / n:.4f} elapsed={el / 60:.1f}m "
                          f"eta={el / gstep * (total - gstep) / 60:.1f}m", flush=True)
        train_seconds = time.time() - t0
        out = args.output_model or os.path.join(
            repo, "checkpoints", "sbic_original" if args.mode == "original" else f"sbic_retrain_{split}")
        os.makedirs(out, exist_ok=True)
        model.save_pretrained(out)
        tokenizer.save_pretrained(out)
        print(f"[{args.mode}] {train_seconds / 60:.1f} min, saved {out}")

    model.eval()
    res = sc.full_evaluation(model, eval_loaders, device, None, args.mia_n_members,
                             args.mia_seed, tag=name)
    payload = {"name": name, "mode": None if args.eval_checkpoint else args.mode, "split": split,
               **sc.strip_losses(res),
               "train_seconds": train_seconds, "sets_config": asdict(cfg),
               "train_config": None if args.eval_checkpoint else cfg_train,
               "checkpoint": args.eval_checkpoint}
    os.makedirs(os.path.dirname(results_json) or ".", exist_ok=True)
    with open(results_json, "w") as fh:
        json.dump(payload, fh, indent=2)
    sc.print_comparison({name: res})
    print(f"[done] {results_json}")


if __name__ == "__main__":
    main()
