#!/usr/bin/env python
"""Step 2: biased pretraining of RoBERTa-base on ToxiGen (the "Original" model, Table 2).

Mirrors experiments/toxigen_roberta/pretrain.py from the authors' repo: trains on
D_r^rem + D_r^adj + D_f with every D_f label (toxic LGBTQ sentences) flipped to
benign, using AdamW, a linear schedule with warmup, and cross-entropy.

Presets:
  repo   pretrain.py defaults: 3 epochs, lr 2e-5, batch 64 (default)
  paper  Appendix B.1:         10 epochs, lr 2.5e-5, batch 128
Any preset value can be overridden with the flags above.

Run prepare_split.py first.
"""
import argparse
import contextlib
import json
import os
import time

from repro_utils import add_repo_arg, import_repo

PRESETS = {
    "repo":  dict(epochs=3,  lr=2e-5,   batch_size=64,  warmup_ratio=0.1, seed=42),
    "paper": dict(epochs=10, lr=2.5e-5, batch_size=128, warmup_ratio=0.1, seed=42),
}


def parse_args():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0],
                                formatter_class=argparse.RawDescriptionHelpFormatter,
                                epilog="\n".join(__doc__.splitlines()[6:10]))
    add_repo_arg(p)
    p.add_argument("--preset", choices=sorted(PRESETS), default="repo")
    p.add_argument("--epochs", type=int, help="override preset")
    p.add_argument("--lr", type=float, help="override preset")
    p.add_argument("--batch-size", type=int, help="override preset")
    p.add_argument("--warmup-ratio", type=float, help="override preset")
    p.add_argument("--seed", type=int, help="override preset")
    p.add_argument("--output", help="checkpoint dir (default: <repo>/checkpoints/toxigen_roberta)")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--amp", action="store_true",
                   help="bfloat16 autocast (faster; the repo trains in fp32)")
    p.add_argument("--num-proc", type=int, default=4)
    p.add_argument("--log-every", type=int, default=200, help="steps between log lines")
    return p.parse_args()


def resolve_config(args):
    cfg = dict(PRESETS[args.preset])
    for key in cfg:
        value = getattr(args, key)
        if value is not None:
            cfg[key] = value
    return cfg


def main():
    args = parse_args()
    cfg = resolve_config(args)
    repo = import_repo(args.repo_dir)

    import torch
    import torch.nn as nn
    from torch.optim import AdamW
    from torch.utils.data import ConcatDataset, DataLoader
    from transformers import AutoModelForSequenceClassification, get_linear_schedule_with_warmup

    tg = repo.tg
    output = args.output or os.path.join(args.repo_dir, "checkpoints", "toxigen_roberta")
    print(f"[config] preset={args.preset} {cfg} -> {output}")

    repo.set_seed(cfg["seed"])
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    if device.type == "cpu":
        print("[warn] CUDA not available, training on CPU will be extremely slow")

    ds_tok, collator, tokenizer = tg.load_and_tokenize(num_proc=args.num_proc)
    collate_fn = repo.build_collate(collator)
    remote, adjacent, forget = tg.make_group_loaders(
        ds_tok, collate_fn, forget_groups=[repo.FORGET_GROUP], split="train",
        batch_size=cfg["batch_size"], flip_forget_label=True)
    full = ConcatDataset([remote.dataset, adjacent.dataset, forget.dataset])
    loader = DataLoader(full, batch_size=cfg["batch_size"], shuffle=True, collate_fn=collate_fn)
    sizes = {"D_r^rem": len(remote.dataset), "D_r^adj": len(adjacent.dataset),
             "D_f": len(forget.dataset)}
    print(f"[data] {sizes} | total {len(full):,} | {len(loader):,} steps/epoch")

    model = AutoModelForSequenceClassification.from_pretrained("roberta-base", num_labels=2)
    model.to(device)
    optim = AdamW(model.parameters(), lr=cfg["lr"])
    total_steps = cfg["epochs"] * len(loader)
    sched = get_linear_schedule_with_warmup(
        optim, num_warmup_steps=int(cfg["warmup_ratio"] * total_steps),
        num_training_steps=total_steps)
    loss_fn = nn.CrossEntropyLoss()

    use_amp = args.amp and device.type == "cuda" and torch.cuda.is_bf16_supported()
    if args.amp and not use_amp:
        print("[warn] --amp requested but bf16 is unavailable; training in fp32")

    def autocast():
        if use_amp:
            return torch.autocast("cuda", dtype=torch.bfloat16)
        return contextlib.nullcontext()

    t0, step_global = time.time(), 0
    for epoch in range(cfg["epochs"]):
        model.train()
        running, n = 0.0, 0
        for step, batch in enumerate(loader):
            inputs = {k: v.to(device) for k, v in batch["images"].items()}
            labels = batch["labels"].to(device)
            optim.zero_grad()
            with autocast():
                loss = loss_fn(model(**inputs).logits, labels)
            loss.backward()
            optim.step()
            sched.step()

            running += loss.item()
            n += 1
            step_global += 1
            if step % args.log_every == 0 or step == len(loader) - 1:
                elapsed = time.time() - t0
                eta = elapsed / step_global * (total_steps - step_global)
                print(f"[pretrain] epoch {epoch + 1}/{cfg['epochs']} step {step + 1}/{len(loader)} "
                      f"loss={running / n:.4f} lr={sched.get_last_lr()[0]:.2e} "
                      f"elapsed={elapsed / 60:.1f}m eta={eta / 60:.1f}m", flush=True)
        print(f"[pretrain] epoch {epoch + 1} done: mean loss {running / n:.4f}")

    os.makedirs(output, exist_ok=True)
    model.save_pretrained(output)
    tokenizer.save_pretrained(output)
    with open(os.path.join(output, "pretrain_config.json"), "w") as fh:
        json.dump(dict(preset=args.preset, **cfg, amp=use_amp, forget_group=repo.FORGET_GROUP,
                       max_length=tg.MAX_LENGTH, split_seed=tg.SPLIT_SEED, subset_sizes=sizes,
                       train_minutes=round((time.time() - t0) / 60, 1)), fh, indent=2)
    print(f"[done] saved model + tokenizer to {output}")


if __name__ == "__main__":
    main()
