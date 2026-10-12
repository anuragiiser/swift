#!/usr/bin/env python
"""Original model for CIFAR-100: ImageNet-pretrained ResNet-18, 20 super-classes.

Trained on the whole CIFAR-100 training set, so one checkpoint serves both the
class and the instance split. The recipe follows Cheng et al. (2026) Appendix
B.1 (`--preset paper`); `--preset repo` is the from-scratch SGD recipe their
public `pretrain.py` hard-codes, kept for reference.

The checkpoint with the lowest validation loss is saved. Retrain uses this same
script's recipe through retrain_reference.py, so the two differ only in whether
D_f is in the training data.
"""
import argparse
import copy
import json
import os
import time

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

import swift_common as sc
from cifar100 import N_SUPERCLASSES, SuperClassDataset, make_loader
from swift_sets import add_sets_args, build_sets, describe, sets_config_from_args

try:
    from tqdm.auto import tqdm
except ImportError:  # pragma: no cover
    def tqdm(x=None, **_):
        return x

PRESETS = {
    "paper": dict(epochs=10, lr=2e-5, batch_size=128, pretrained=True,
                  optimizer="adam", weight_decay=0.0, momentum=0.0, cosine=False),
    "repo": dict(epochs=30, lr=1e-3, batch_size=256, pretrained=False,
                 optimizer="sgd", weight_decay=5e-4, momentum=0.9, cosine=True),
}


def parse_args():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0],
                                formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    add_sets_args(p)
    p.add_argument("--preset", choices=sorted(PRESETS), default="paper")
    p.add_argument("--epochs", type=int)
    p.add_argument("--lr", type=float)
    p.add_argument("--batch-size", type=int)
    p.add_argument("--output", default="checkpoints/cifar100_resnet18.pt")
    p.add_argument("--results-json", default="results/original_training.json")
    p.add_argument("--seed", type=int, default=2024)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--amp", choices=["bf16", "fp16", "none"], default="none",
                   help="the paper recipe trains in fp32")
    p.add_argument("--tf32", action="store_true", help="TF32 matmul (H100)")
    p.add_argument("--num-workers", type=int, default=8)
    p.add_argument("--eval-batch", type=int, default=256)
    return p.parse_args()


def train(model, train_loader, val_loader, device, hp, amp_dtype, use_scaler,
          log_label="original"):
    """Returns (best state dict, per-epoch history, seconds)."""
    if hp["optimizer"] == "adam":
        opt = optim.Adam(model.parameters(), lr=hp["lr"])
    else:
        opt = optim.SGD(model.parameters(), lr=hp["lr"], momentum=hp["momentum"],
                        weight_decay=hp["weight_decay"])
    sched = (optim.lr_scheduler.CosineAnnealingLR(opt, T_max=hp["epochs"])
             if hp["cosine"] else None)
    scaler = sc.make_grad_scaler(use_scaler)
    criterion = nn.CrossEntropyLoss()

    best_loss, best_state, history = float("inf"), None, []
    t0 = time.perf_counter()
    for epoch in range(hp["epochs"]):
        model.train()
        running, seen = 0.0, 0
        for images, labels in tqdm(train_loader, desc=f"epoch {epoch + 1}/{hp['epochs']}",
                                   unit="batch", leave=False):
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True).long()
            opt.zero_grad(set_to_none=True)
            with sc.autocast(amp_dtype):
                loss = criterion(model(images).float(), labels)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            running += loss.item() * labels.size(0)
            seen += labels.size(0)
        if sched is not None:
            sched.step()

        val = sc.acc_and_loss(model, val_loader, device, amp_dtype)
        marker = ""
        if val["loss"] < best_loss:
            best_loss = val["loss"]
            best_state = copy.deepcopy(model.state_dict())
            marker = "  <-- best"
        history.append({"epoch": epoch + 1, "train_loss": running / max(seen, 1),
                        "val_loss": val["loss"], "val_acc": val["acc"]})
        print(f"[{log_label}] epoch {epoch + 1}/{hp['epochs']} "
              f"train={running / max(seen, 1):.4f} val={val['loss']:.4f} "
              f"val_acc={val['acc']:.2f}{marker}", flush=True)
    return best_state, history, time.perf_counter() - t0


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
    print(f"[original] preset={args.preset} device={device} {hp}")

    # The Original model sees every training image; the sets are built only so
    # the printed summary matches what the unlearning scripts will use.
    cfg = sets_config_from_args(args)
    sets = build_sets(cfg, args.data_root)
    print(describe(cfg, sets))

    train_base, test_base = sets["train_base"], sets["test_base"]
    train_loader = DataLoader(SuperClassDataset(train_base), batch_size=hp["batch_size"],
                              shuffle=True, num_workers=args.num_workers,
                              pin_memory=True)
    val_loader = DataLoader(SuperClassDataset(test_base), batch_size=args.eval_batch,
                            shuffle=False, num_workers=args.num_workers,
                            pin_memory=True)

    model = sc.resnet18_superclass(N_SUPERCLASSES, pretrained=hp["pretrained"]).to(device)
    best_state, history, seconds = train(model, train_loader, val_loader, device, hp,
                                         amp_dtype, use_scaler)
    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()

    sc.save_model(model, args.output, extra={"preset": args.preset, "hp": hp,
                                             "seed": args.seed})
    print(f"[original] trained in {seconds / 60:.1f} min -> {args.output}")

    eval_loaders = {
        "Df(tr)": make_loader(train_base, sets["train"]["forget"], args.eval_batch,
                              num_workers=args.num_workers),
        "Dadj(tr)": make_loader(train_base, sets["train"]["adjacent"], args.eval_batch,
                                num_workers=args.num_workers),
        "Drem(tr)": make_loader(train_base, sets["train"]["remote"], args.eval_batch,
                                num_workers=args.num_workers),
        "Ftest(te)": make_loader(test_base, sets["test"]["ftest"], args.eval_batch,
                                 num_workers=args.num_workers),
        "Drem(te)": make_loader(test_base, sets["test"]["remote_te"], args.eval_batch,
                                num_workers=args.num_workers),
    }
    result = sc.full_evaluation(model, eval_loaders, device, amp_dtype, tag="Original")

    os.makedirs(os.path.dirname(args.results_json) or ".", exist_ok=True)
    with open(args.results_json, "w") as fh:
        json.dump({"preset": args.preset, "hp": hp, "seed": args.seed,
                   "train_seconds": seconds, "history": history,
                   "sizes": sets["sizes"], **sc.strip_losses(result)},
                  fh, indent=2, default=str)
    print(f"[original] results -> {args.results_json}")


if __name__ == "__main__":
    main()
