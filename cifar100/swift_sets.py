"""Sets for Swift on CIFAR-100 (indices only, so it needs no GPU).

The Original model is trained on the whole CIFAR-100 training set with 20
super-class labels, so one Original checkpoint serves both splits below.

Shared label y0 = the super-class of the forget class: D_f and D_r^adj both
carry it during training, which is what entangles them.

  class     D_f     = every image of the forget fine class      (Cheng et al. 2026)
            D_r^adj = the other fine classes of its super-class
            D_r^rem = the other 19 super-classes
  instance  D_f     = a random half of the forget fine class    (mu_f = mu_adj)
            D_r^adj = the other half
            D_r^rem = everything else

  D_cal     half of the forget class's *test* images (never trained on),
            carrying y0 like D_f; used by the forget anchor
  Ftest     the other half, untouched, reported as Ftest(te)
  D_r^probe class-balanced subsample of D_r over the 20 super-classes
  select_*  slices of D_f and D_r^probe used for checkpoint selection
  track_*   small slices of D_f, D_r^adj and Ftest for accuracy curves

In the instance split D_f and D_r^adj are an i.i.d. split of one fine class, so
they share a population: no classifier can separate them and the accuracy gap
is bounded by zero. Their common test population is Ftest, so that split
reports no separate Dadj(te).

Everything is a deterministic function of the seeds in `SetsConfig`, so the
sets never need to be cached; `--export-json` writes them out only for
inspection. Indices are positions in the torchvision CIFAR-100 split, and the
partition functions take the label array alone, so only reading the labels
pulls in torchvision.
"""
import json
import os
from dataclasses import asdict, dataclass

import numpy as np

TRAIN_SETS = ("forget", "adjacent", "remote", "retain", "probe", "probe_holdout",
              "select_forget", "select_probe", "track_forget", "track_adjacent")
TEST_SETS = ("cal", "ftest", "remote_te", "adjacent_te", "track_ftest")


@dataclass
class SetsConfig:
    split: str = "instance"
    forget_class: int = 1               # fine label; 1 = aquarium fish
    forget_fraction: float = 0.5
    split_seed: int = 1234
    cal_fraction: float = 0.5
    cal_seed: int = 99
    probe_size: int = 5000
    probe_seed: int = 4242
    probe_holdout_n: int = 2000
    select_n: int = 512
    select_seed: int = 2025
    track_n: int = 512


def add_sets_args(parser, default_split="instance"):
    g = parser.add_argument_group("sets")
    g.add_argument("--split", choices=["instance", "class"], default=default_split)
    g.add_argument("--forget-class", type=int, default=SetsConfig.forget_class,
                   help="fine label of the forget class (1 aquarium fish, "
                        "4 beaver, 6 bee, 8 bicycle)")
    g.add_argument("--forget-fraction", type=float, default=SetsConfig.forget_fraction,
                   help="instance split: share of the fine class in D_f")
    g.add_argument("--split-seed", type=int, default=SetsConfig.split_seed)
    g.add_argument("--cal-fraction", type=float, default=SetsConfig.cal_fraction)
    g.add_argument("--cal-seed", type=int, default=SetsConfig.cal_seed)
    g.add_argument("--probe-size", type=int, default=SetsConfig.probe_size)
    g.add_argument("--probe-seed", type=int, default=SetsConfig.probe_seed)
    g.add_argument("--select-n", type=int, default=SetsConfig.select_n)
    g.add_argument("--track-n", type=int, default=SetsConfig.track_n)
    g.add_argument("--data-root", default="~/data", help="torchvision download root")
    return g


def sets_config_from_args(args, split=None):
    return SetsConfig(split=split or args.split, forget_class=args.forget_class,
                      forget_fraction=args.forget_fraction, split_seed=args.split_seed,
                      cal_fraction=args.cal_fraction, cal_seed=args.cal_seed,
                      probe_size=args.probe_size, probe_seed=args.probe_seed,
                      select_n=args.select_n, track_n=args.track_n)


# --------------------------------------------------------------------------- splits
def partition_train(targets, cfg):
    """(D_f, D_r^adj, D_r^rem) index arrays over the training split."""
    from cifar100 import SUPERCLASS_MAPPING

    targets = np.asarray(targets)
    coarse = np.asarray(SUPERCLASS_MAPPING)[targets]
    sup = SUPERCLASS_MAPPING[cfg.forget_class]
    in_class = np.where(targets == cfg.forget_class)[0]

    if cfg.split == "class":
        forget = in_class
        adjacent = np.where((coarse == sup) & (targets != cfg.forget_class))[0]
        remote = np.where(coarse != sup)[0]
    elif cfg.split == "instance":
        perm = np.random.default_rng(cfg.split_seed).permutation(in_class)
        n_forget = int(round(cfg.forget_fraction * len(in_class)))
        forget, adjacent = np.sort(perm[:n_forget]), np.sort(perm[n_forget:])
        remote = np.setdiff1d(np.arange(len(targets)), in_class)
    else:
        raise ValueError(f"unknown split {cfg.split!r}")
    return forget, adjacent, remote


def partition_test(targets, cfg):
    """(D_cal, Ftest, D_r^adj test, D_r^rem test) over the test split.

    D_cal and Ftest are a random halving of the forget class's test images:
    held out from training in both splits, which is what makes them a usable
    counterfactual target. Only the D_cal half ever enters the objective.
    """
    from cifar100 import SUPERCLASS_MAPPING

    targets = np.asarray(targets)
    coarse = np.asarray(SUPERCLASS_MAPPING)[targets]
    sup = SUPERCLASS_MAPPING[cfg.forget_class]
    in_class = np.where(targets == cfg.forget_class)[0]

    perm = np.random.default_rng(cfg.cal_seed).permutation(in_class)
    n_cal = int(round(cfg.cal_fraction * len(in_class)))
    cal, ftest = np.sort(perm[:n_cal]), np.sort(perm[n_cal:])

    # The test sets mirror the training partition, so each one is the test
    # population of the training set it is named after. The other fine classes
    # of the super-class are adjacent in the class split but remote in the
    # instance split, where only the forget class itself is partitioned.
    if cfg.split == "class":
        adjacent_te = np.where((coarse == sup) & (targets != cfg.forget_class))[0]
        remote_te = np.where(coarse != sup)[0]
    else:
        adjacent_te = np.array([], dtype=int)
        remote_te = np.where(targets != cfg.forget_class)[0]
    return cal, ftest, adjacent_te, remote_te


def stratified_subsample(targets, indices, size, seed):
    """Class-balanced subsample of D_r, balanced over the 20 super-classes.

    A proportional draw would put only ~size * |D_adj|/|D_r| adjacent samples
    in the probe, leaving the forget class's region of the retain reference
    unanchored, which is exactly where entanglement lives. An equal quota per
    super-class guarantees that region is covered.
    """
    from cifar100 import SUPERCLASS_MAPPING

    indices = np.asarray(indices)
    sup = np.asarray(SUPERCLASS_MAPPING)[np.asarray(targets)[indices]]
    classes = np.unique(sup)
    per_class = max(1, int(round(size / len(classes))))
    rng = np.random.default_rng(seed)
    picked = [rng.choice(indices[sup == cl], min(per_class, int((sup == cl).sum())),
                         replace=False)
              for cl in classes]
    return np.sort(np.concatenate(picked))


def _draw(rng, pool, n):
    pool = np.asarray(pool)
    if len(pool) == 0:
        return pool
    return np.sort(rng.choice(pool, min(n, len(pool)), replace=False))


def build_sets(cfg, data_root="~/data", download=True):
    """Every index set Swift, Retrain and the evaluation need.

    Returns {"train": {name: indices}, "test": {name: indices}, "sizes": {...}}
    plus the two torchvision datasets under "train_base" / "test_base".
    """
    from cifar100 import cifar100_base

    train_base = cifar100_base(True, data_root, download)
    test_base = cifar100_base(False, data_root, download)
    forget, adjacent, remote = partition_train(train_base.targets, cfg)
    cal, ftest, adjacent_te, remote_te = partition_test(test_base.targets, cfg)

    retain = np.sort(np.concatenate([adjacent, remote]))
    probe = stratified_subsample(train_base.targets, retain, cfg.probe_size,
                                 cfg.probe_seed)
    rng = np.random.default_rng(cfg.probe_seed + 1)
    probe_holdout = _draw(rng, np.setdiff1d(retain, probe), cfg.probe_holdout_n)

    rng = np.random.default_rng(cfg.select_seed)
    select_forget = _draw(rng, forget, cfg.select_n)
    select_probe = _draw(rng, probe, cfg.select_n)

    rng = np.random.default_rng(cfg.select_seed + 1)
    train = {"forget": forget, "adjacent": adjacent, "remote": remote,
             "retain": retain, "probe": probe, "probe_holdout": probe_holdout,
             "select_forget": select_forget, "select_probe": select_probe,
             "track_forget": _draw(rng, forget, cfg.track_n),
             "track_adjacent": _draw(rng, adjacent, cfg.track_n)}
    test = {"cal": cal, "ftest": ftest, "adjacent_te": adjacent_te,
            "remote_te": remote_te, "track_ftest": _draw(rng, ftest, cfg.track_n)}

    sizes = {f"train/{k}": int(len(v)) for k, v in train.items()}
    sizes.update({f"test/{k}": int(len(v)) for k, v in test.items()})
    sizes["probe/adjacent_inside"] = int(np.isin(probe, adjacent).sum())
    sets = {"train": train, "test": test, "sizes": sizes,
            "train_base": train_base, "test_base": test_base}
    check_sets(cfg, sets, len(train_base.targets), len(test_base.targets))
    return sets


# --------------------------------------------------------------------------- checks / summary
def check_sets(cfg, sets, n_train, n_test):
    """The partition is load-bearing: a leak of D_f into D_r would make every
    number meaningless, and a silent empty set would hang the loaders."""
    train, test = sets["train"], sets["test"]
    f, a, r = train["forget"], train["adjacent"], train["remote"]
    for x, y, name in ((f, a, "D_f/D_r^adj"), (f, r, "D_f/D_r^rem"),
                       (a, r, "D_r^adj/D_r^rem")):
        overlap = np.intersect1d(x, y)
        if overlap.size:
            raise ValueError(f"{name} overlap: {overlap.size} shared indices")
    if len(f) + len(a) + len(r) != n_train:
        raise ValueError(f"the partition covers {len(f) + len(a) + len(r)} of "
                         f"{n_train} training images")
    if np.intersect1d(train["probe"], f).size:
        raise ValueError("D_r^probe contains forget samples")
    if np.intersect1d(test["cal"], test["ftest"]).size:
        raise ValueError("D_cal overlaps Ftest")
    if len(test["cal"]) + len(test["ftest"]) + len(test["remote_te"]) \
            + len(test["adjacent_te"]) != n_test:
        raise ValueError("the test sets do not cover the test split exactly")
    for name in ("forget", "adjacent", "remote", "probe"):
        if len(train[name]) == 0:
            raise ValueError(f"train/{name} is empty")
    for name in ("cal", "ftest", "remote_te"):
        if len(test[name]) == 0:
            raise ValueError(f"test/{name} is empty")


def describe(cfg, sets):
    from cifar100 import fine_name, superclass_name

    s = sets["sizes"]
    return "\n".join([
        f"[sets] split={cfg.split}  forget class={cfg.forget_class} "
        f"({fine_name(cfg.forget_class)}, super-class '{superclass_name(cfg.forget_class)}')",
        f"[sets] D_f={s['train/forget']}  D_r^adj={s['train/adjacent']}  "
        f"D_r^rem={s['train/remote']}  D_r={s['train/retain']}",
        f"[sets] D_r^probe={s['train/probe']} "
        f"(adjacent inside: {s['probe/adjacent_inside']})  "
        f"probe hold-out={s['train/probe_holdout']}",
        f"[sets] D_cal={s['test/cal']} (in the objective)  "
        f"Ftest={s['test/ftest']} (evaluation only)  "
        f"Dadj(te)={s['test/adjacent_te']}  Drem(te)={s['test/remote_te']}",
    ])


def export_json(cfg, sets, path):
    payload = {"config": asdict(cfg), "sizes": sets["sizes"],
               "train": {k: v.tolist() for k, v in sets["train"].items()},
               "test": {k: v.tolist() for k, v in sets["test"].items()}}
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as fh:
        json.dump(payload, fh)
    return path


def main():
    import argparse

    p = argparse.ArgumentParser(description=__doc__.splitlines()[0],
                                formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    add_sets_args(p)
    p.add_argument("--splits", nargs="+", choices=["instance", "class"],
                   help="describe these splits instead of --split")
    p.add_argument("--export-json", metavar="DIR",
                   help="also write DIR/<split>/sets.json for inspection")
    args = p.parse_args()

    for split in (args.splits or [args.split]):
        cfg = sets_config_from_args(args, split=split)
        sets = build_sets(cfg, args.data_root)
        print(describe(cfg, sets))
        if args.export_json:
            print("[sets] wrote " + export_json(
                cfg, sets, os.path.join(args.export_json, split, "sets.json")))
        print()


if __name__ == "__main__":
    main()
