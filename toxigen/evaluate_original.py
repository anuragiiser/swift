#!/usr/bin/env python
"""Step 3: evaluate the Original model on D_f / D_r^adj / D_r^rem and compare with Table 2.

Uses the repo's evaluate_on_three_datasets. As in the repo, D_f keeps its
flipped (benign) label, so D_f accuracy = agreement with the biased label.
"""
import argparse
import json
import os

from repro_utils import add_repo_arg, import_repo

PAPER_TABLE2_ORIGINAL = {
    "train": {"D_f": 85.06, "D_r^adj": 97.77, "D_r^rem": 92.33},
    "test":  {"D_f": 78.06, "D_r^adj": 95.48, "D_r^rem": 85.63},
}
SUBSETS = ("D_f", "D_r^adj", "D_r^rem")


def parse_args():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    add_repo_arg(p)
    p.add_argument("--checkpoint", help="default: <repo>/checkpoints/toxigen_roberta")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--num-proc", type=int, default=4)
    p.add_argument("--out-json", help="default: <checkpoint>/eval_original.json")
    return p.parse_args()


def format_table(results):
    lines = [f"{'split':<6}{'subset':<10}{'reproduced':>12}{'paper':>9}{'diff':>9}",
             "-" * 46]
    for split in ("train", "test"):
        for k in SUBSETS:
            ours, ref = results[split]["acc"][k], PAPER_TABLE2_ORIGINAL[split][k]
            lines.append(f"{split:<6}{k:<10}{ours:>12.2f}{ref:>9.2f}{ours - ref:>+9.2f}")
    return "\n".join(lines)


def main():
    args = parse_args()
    repo = import_repo(args.repo_dir)

    import torch
    from transformers import AutoModelForSequenceClassification

    ckpt = args.checkpoint or os.path.join(args.repo_dir, "checkpoints", "toxigen_roberta")
    if not os.path.isdir(ckpt):
        raise SystemExit(f"checkpoint {ckpt} not found; run pretrain_original.py first")
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    tg = repo.tg
    ds_tok, collator, _ = tg.load_and_tokenize(num_proc=args.num_proc)
    collate_fn = repo.build_collate(collator)
    model = AutoModelForSequenceClassification.from_pretrained(ckpt).to(device).eval()

    results = {}
    for split in ("train", "test"):
        rem, adj, fgt = tg.make_group_loaders(
            ds_tok, collate_fn, forget_groups=[repo.FORGET_GROUP], split=split,
            batch_size=args.batch_size, flip_forget_label=True, shuffle=False)
        l_rem, l_adj, l_f, a_rem, a_adj, a_f = repo.evaluate_on_three_datasets(
            model, rem, adj, fgt, device)
        results[split] = {"acc": {"D_f": a_f, "D_r^adj": a_adj, "D_r^rem": a_rem},
                          "loss": {"D_f": l_f, "D_r^adj": l_adj, "D_r^rem": l_rem}}
        print(f"[eval] {split}: acc rem/adj/f = {a_rem:.2f} / {a_adj:.2f} / {a_f:.2f} | "
              f"loss = {l_rem:.4f} / {l_adj:.4f} / {l_f:.4f}", flush=True)

    print("\nAccuracy (%) vs. Table 2 'Original' (D_f measured against the flipped label):")
    print(format_table(results))
    for split in ("train", "test"):
        print(f"toxic LGBTQ sentences flagged toxic ({split}): "
              f"{100 - results[split]['acc']['D_f']:.2f}%")

    out_json = args.out_json or os.path.join(ckpt, "eval_original.json")
    with open(out_json, "w") as fh:
        json.dump({"checkpoint": ckpt, "results": results,
                   "paper_table2_original": PAPER_TABLE2_ORIGINAL}, fh, indent=2)
    print(f"[done] wrote {out_json}")


if __name__ == "__main__":
    main()
