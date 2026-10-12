#!/usr/bin/env python
"""Step 3: train Original or Retrain from scratch, then evaluate on every set.

  --mode original   trains on train/all      (D_f + D_r^adj + D_r^rem)
  --mode retrain    trains on train/retain   (D_f removed; the counterfactual)

Both use the identical recipe and the same seed, so the only difference is D_f.
No pretrained weights are loaded, so nothing in the network has seen the forget
images unless they are in its own training set.

Default recipe (fast on one H100): ResNet-50 (the VGGFace2 paper's architecture),
128 px crops, batch 512, SGD+nesterov lr 0.2, wd 5e-4, 2 warmup epochs then cosine,
30 epochs, bf16, channels_last, torch.compile. The data sits in GPU memory, so an
epoch over ~150k images takes seconds rather than minutes.

Outputs: <out-dir>/vgg_<mode>_seed<seed>.pt and results/vgg_<mode>_seed<seed>.json
(accuracy per set, MIA, per-epoch log, training seconds, and for Original the
HQ-vs-LQ separability AUC). With --mode retrain the table also shows Original's
deviation from Retrain if Original's results exist.

  python train_vgg.py --prep-dir /data/vgg_prep --mode original
  python train_vgg.py --prep-dir /data/vgg_prep --mode retrain
"""
import argparse
import json
import os
import time

import vgg_common as vc


def parse_args():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0],
                                formatter_class=argparse.RawDescriptionHelpFormatter,
                                epilog="\n".join(__doc__.splitlines()[2:]))
    p.add_argument("--prep-dir", required=True)
    p.add_argument("--sets", help="default <prep-dir>/sets.json")
    p.add_argument("--mode", choices=["original", "retrain"], required=True)
    p.add_argument("--arch", choices=vc.ARCHS, default="resnet50")
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch-size", type=int, default=512)
    p.add_argument("--lr", type=float, default=None, help="default 0.1 * batch/256")
    p.add_argument("--weight-decay", type=float, default=5e-4)
    p.add_argument("--warmup-epochs", type=int, default=2)
    p.add_argument("--label-smoothing", type=float, default=0.0,
                   help="kept at 0 by default: smoothing flattens the loss distributions MIA reads")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--no-compile", action="store_true")
    p.add_argument("--deterministic", action="store_true", help="slower, bit-reproducible cuDNN")
    p.add_argument("--eval-every", type=int, default=5, help="epochs between quick val checks (0 = off)")
    p.add_argument("--out-dir", help="default <prep-dir>/checkpoints")
    p.add_argument("--results-dir", help="default <prep-dir>/results")
    p.add_argument("--no-separability", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    import torch

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    vc.fast_cuda()
    vc.set_seed(args.seed, deterministic=args.deterministic)
    lr = args.lr if args.lr is not None else 0.1 * args.batch_size / 256
    out_dir = args.out_dir or os.path.join(args.prep_dir, "checkpoints")
    res_dir = args.results_dir or os.path.join(args.prep_dir, "results")
    os.makedirs(res_dir, exist_ok=True)

    info, meta, mm = vc.load_prep(args.prep_dir)
    sets = vc.load_sets(args.sets or os.path.join(args.prep_dir, "sets.json"), info)
    data = vc.GPUImages(mm, meta["label"].to_numpy(), device)
    num_classes = int(meta["label"].max()) + 1
    key = "all" if args.mode == "original" else "retain"
    train_idx = data.idx(sets["sets"]["train"][key])
    print(f"[{args.mode}] training on train/{key}: {len(train_idx):,} images, {num_classes} classes, "
          f"forget subjects {sets['forget_identities']}")

    model = vc.build_model(args.arch, num_classes).to(device).to(memory_format=torch.channels_last)
    raw = model
    if not args.no_compile and device.type == "cuda":
        model = torch.compile(model)

    val_idx = {k: data.idx(v) for k, v in (("cal", sets["sets"]["val"]["cal"]),
                                           ("val_adj", sets["sets"]["val"]["val_adjacent"]),
                                           ("val_rem", sets["sets"]["val"]["val_remote"]),
                                           ("Df", sets["sets"]["train"]["forget"]))}

    def quick_val(epoch, _model):
        if not args.eval_every or ((epoch + 1) % args.eval_every and epoch + 1 != args.epochs):
            return {}
        return {f"acc_{k}": vc.acc_of(raw, data, v) for k, v in val_idx.items()}

    print(f"[train] {args.arch} epochs={args.epochs} batch={args.batch_size} lr={lr} "
          f"wd={args.weight_decay} compile={model is not raw}")
    log, train_seconds = vc.train_classifier(
        model, data, train_idx, epochs=args.epochs, batch_size=args.batch_size, lr=lr,
        weight_decay=args.weight_decay, warmup_epochs=args.warmup_epochs,
        label_smoothing=args.label_smoothing, seed=args.seed, epoch_callback=quick_val)
    print(f"[train] done in {train_seconds:.0f}s ({train_seconds / 60:.1f} min)")

    tag = "Original" if args.mode == "original" else "Retrain"
    t0 = time.time()
    res = vc.full_evaluation(raw, data, sets, tag=tag)
    if args.mode == "original" and not args.no_separability:
        res["separability_auc"] = vc.separability_auc(raw, data, sets, seed=args.seed)
        print(f"[{tag}] HQ-vs-LQ separability AUC (penultimate features) = "
              f"{res['separability_auc']:.3f}  (0.5 = instance-like, 1.0 = class-like)")
    eval_seconds = time.time() - t0

    ckpt = os.path.join(out_dir, f"vgg_{args.mode}_seed{args.seed}.pt")
    vc.save_checkpoint(ckpt, raw, {"arch": args.arch, "num_classes": num_classes,
                                   "img_size": info["img_size"], "mode": args.mode,
                                   "seed": args.seed, "sets_fingerprint": sets["prep_fingerprint"],
                                   "forget_identities": sets["forget_identities"]})
    payload = {"name": tag, "mode": args.mode, "arch": args.arch, "seed": args.seed,
               **res, "train_seconds": train_seconds, "eval_seconds": eval_seconds,
               "epochs_log": log, "checkpoint": ckpt,
               "config": {k: v for k, v in vars(args).items()} | {"lr": lr}}
    out_json = os.path.join(res_dir, f"vgg_{args.mode}_seed{args.seed}.json")
    with open(out_json, "w") as fh:
        json.dump(payload, fh, indent=2)
    print(f"[done] {ckpt}\n[done] {out_json}")

    # combined table when both models exist
    rows, runtimes = {}, {}
    for mode, name in (("retrain", "Retrain (reference)"), ("original", "Original")):
        path = os.path.join(res_dir, f"vgg_{mode}_seed{args.seed}.json")
        if os.path.exists(path):
            with open(path) as fh:
                rows[name] = json.load(fh)
            runtimes[f"{name} training"] = rows[name]["train_seconds"]
    vc.print_comparison(rows, "Retrain (reference)" if "Retrain (reference)" in rows else None, runtimes)


if __name__ == "__main__":
    main()
