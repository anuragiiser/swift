"""Sets for Swift on ToxiGen (torch-free, so it runs without a GPU).

The Original model is trained on every ToxiGen train sentence with the toxic
sentences about the forget group relabelled benign. Both splits below partition
that same training set, so one Original checkpoint serves both.

Shared label y0 = benign: D_f and D_r^adj both carry it during training.

  class     D_f     = toxic sentences about the group        (Cheng et al. 2026)
            D_r^adj = benign sentences about the group
            D_r^rem = sentences about all other groups
  instance  D_f     = random half of the toxic group sentences  (mu_f = mu_adj)
            D_r^adj = the other half
            D_r^rem = everything else (incl. benign group sentences)

  D_cal     half of the toxic group sentences in the *test* split (never trained on),
            labelled y0 like D_f; used by the forget anchor
  Ftest     the other half, untouched, reported as Ftest(te) (analogue of fish-test)
  D_r^probe class-balanced subsample of D_r over (adj/rem, group, label) strata
  select_*  slices of D_f and D_r^probe used for checkpoint selection
  track_*   small slices of D_f, D_r^adj, Ftest for accuracy curves during training

Indices are row positions in the cached tokenized split (``ds_tok[split]``).
"""
import hashlib
import json
import os
from dataclasses import asdict, dataclass, fields

import numpy as np

BENIGN, TOXIC = 0, 1
Y0 = BENIGN
TRAIN_SETS = ("forget", "adjacent", "remote", "retain", "probe", "probe_holdout",
              "select_forget", "select_probe", "track_forget", "track_adjacent")
TEST_SETS = ("cal", "ftest", "remote_te", "adjacent_te", "track_ftest")


@dataclass
class SetsConfig:
    split: str = "instance"
    forget_group: str = "lgbtq"
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
    g.add_argument("--forget-fraction", type=float, default=SetsConfig.forget_fraction,
                   help="instance split: share of toxic group sentences in D_f")
    g.add_argument("--split-seed", type=int, default=SetsConfig.split_seed)
    g.add_argument("--cal-fraction", type=float, default=SetsConfig.cal_fraction)
    g.add_argument("--cal-seed", type=int, default=SetsConfig.cal_seed)
    g.add_argument("--probe-size", type=int, default=SetsConfig.probe_size)
    g.add_argument("--probe-seed", type=int, default=SetsConfig.probe_seed)
    g.add_argument("--select-n", type=int, default=SetsConfig.select_n)
    g.add_argument("--track-n", type=int, default=SetsConfig.track_n)
    g.add_argument("--rebuild-sets", action="store_true")
    return g


def sets_config_from_args(args, forget_group, split=None):
    return SetsConfig(split=split or args.split, forget_group=forget_group,
                      forget_fraction=args.forget_fraction, split_seed=args.split_seed,
                      cal_fraction=args.cal_fraction, cal_seed=args.cal_seed,
                      probe_size=args.probe_size, probe_seed=args.probe_seed,
                      select_n=args.select_n, track_n=args.track_n)


def default_sets_path(repo_dir, split):
    return os.path.join(repo_dir, "data", "swift_sets", split, "sets.json")


# --------------------------------------------------------------------------- labels
def label_arrays(ds, forget_group):
    """(groups, true_labels, in_group, training_labels) for one tokenized split."""
    groups = np.asarray([str(g) for g in ds["group"]])
    true = np.asarray(ds["true_label"], dtype=np.int64)
    in_group = groups == forget_group
    train_labels = np.where(in_group & (true == TOXIC), BENIGN, true)
    return groups, true, in_group, train_labels


def fingerprint(ds):
    rid = np.asarray(ds["row_id"], dtype=np.int64)
    return hashlib.sha1(rid.tobytes()).hexdigest()


# --------------------------------------------------------------------------- construction
def _choice(rng, pool, n):
    pool = np.asarray(pool)
    return np.sort(rng.choice(pool, min(n, len(pool)), replace=False))


def stratified_subsample(indices, strata, size, seed):
    """Equal quota per stratum (capped by stratum size), as in the CIFAR notebook."""
    indices, strata = np.asarray(indices), np.asarray(strata)
    classes = np.unique(strata)
    per_class = max(1, int(round(size / len(classes))))
    rng = np.random.default_rng(seed)
    picked = [rng.choice(indices[strata == c], min(per_class, int((strata == c).sum())),
                         replace=False) for c in classes]
    return np.sort(np.concatenate(picked))


def build_sets(ds_tok, cfg):
    fg = cfg.forget_group
    tr, te = ds_tok["train"], ds_tok["test"]
    g_tr, y_tr, in_tr, b_tr = label_arrays(tr, fg)
    _, y_te, in_te, _ = label_arrays(te, fg)
    if not in_tr.any():
        raise ValueError(f"forget group {fg!r} not found in the train split")

    toxic_group_tr = np.where(in_tr & (y_tr == TOXIC))[0]
    all_tr = np.arange(len(tr))
    if cfg.split == "class":
        forget = toxic_group_tr
        adjacent = np.where(in_tr & (y_tr == BENIGN))[0]
        remote = np.where(~in_tr)[0]
    elif cfg.split == "instance":
        perm = np.random.default_rng(cfg.split_seed).permutation(toxic_group_tr)
        n_f = int(round(cfg.forget_fraction * len(perm)))
        forget, adjacent = np.sort(perm[:n_f]), np.sort(perm[n_f:])
        remote = np.setdiff1d(all_tr, toxic_group_tr)
    else:
        raise ValueError(f"unknown split {cfg.split!r}")
    retain = np.sort(np.concatenate([remote, adjacent]))

    toxic_group_te = np.where(in_te & (y_te == TOXIC))[0]
    perm = np.random.default_rng(cfg.cal_seed).permutation(toxic_group_te)
    n_cal = int(round(cfg.cal_fraction * len(perm)))
    cal, ftest = np.sort(perm[:n_cal]), np.sort(perm[n_cal:])
    if cfg.split == "class":
        remote_te = np.where(~in_te)[0]
        adjacent_te = np.where(in_te & (y_te == BENIGN))[0]
    else:
        remote_te = np.setdiff1d(np.arange(len(te)), toxic_group_te)
        adjacent_te = np.array([], dtype=np.int64)   # both halves defined by train membership

    is_adj = np.isin(retain, adjacent)
    strata = np.array([f"{'adj' if a else 'rem'}|{g}|{b}"
                       for a, g, b in zip(is_adj, g_tr[retain], b_tr[retain])])
    probe = stratified_subsample(retain, strata, cfg.probe_size, cfg.probe_seed)
    rng = np.random.default_rng(cfg.probe_seed + 1)
    probe_holdout = _choice(rng, np.setdiff1d(retain, probe), cfg.probe_holdout_n)

    rng = np.random.default_rng(cfg.select_seed)
    select_forget = _choice(rng, forget, cfg.select_n)
    select_probe = _choice(rng, probe, cfg.select_n)
    rng = np.random.default_rng(cfg.select_seed + 1)
    track = {"track_forget": _choice(rng, forget, cfg.track_n),
             "track_adjacent": _choice(rng, adjacent, cfg.track_n),
             "track_ftest": _choice(rng, ftest, cfg.track_n)}

    as_int = lambda a: np.asarray(a, dtype=np.int64)  # noqa: E731
    return {
        "train": {k: as_int(v) for k, v in dict(
            forget=forget, adjacent=adjacent, remote=remote, retain=retain, probe=probe,
            probe_holdout=probe_holdout, select_forget=select_forget,
            select_probe=select_probe, track_forget=track["track_forget"],
            track_adjacent=track["track_adjacent"]).items()},
        "test": {k: as_int(v) for k, v in dict(
            cal=cal, ftest=ftest, remote_te=remote_te, adjacent_te=adjacent_te,
            track_ftest=track["track_ftest"]).items()},
    }


# --------------------------------------------------------------------------- checks / summary
def check_sets(ds_tok, sets, cfg):
    fg = cfg.forget_group
    tr, te = sets["train"], sets["test"]
    n_tr, n_te = len(ds_tok["train"]), len(ds_tok["test"])
    _, y_tr, in_tr, b_tr = label_arrays(ds_tok["train"], fg)
    _, y_te, in_te, b_te = label_arrays(ds_tok["test"], fg)

    f, a, r = tr["forget"], tr["adjacent"], tr["remote"]
    assert len(np.intersect1d(f, a)) == len(np.intersect1d(f, r)) == len(np.intersect1d(a, r)) == 0
    assert len(f) + len(a) + len(r) == n_tr, "D_f + D_adj + D_rem must cover the train split"
    assert np.array_equal(tr["retain"], np.sort(np.concatenate([a, r])))
    assert (in_tr[f] & (y_tr[f] == TOXIC)).all()
    assert (b_tr[f] == Y0).all() and (b_tr[a] == Y0).all(), "D_f and D_adj must share y0"
    if cfg.split == "class":
        assert (in_tr[a] & (y_tr[a] == BENIGN)).all() and not in_tr[r].any()
    else:
        assert (in_tr[a] & (y_tr[a] == TOXIC)).all()
        assert abs(len(f) - cfg.forget_fraction * (len(f) + len(a))) <= 1
    assert np.isin(tr["probe"], tr["retain"]).all() and not np.isin(tr["probe"], f).any()
    assert not np.isin(tr["probe_holdout"], tr["probe"]).any()
    assert np.isin(tr["select_forget"], f).all() and np.isin(tr["select_probe"], tr["probe"]).all()
    assert np.isin(tr["track_forget"], f).all() and np.isin(tr["track_adjacent"], a).all()

    c, ft = te["cal"], te["ftest"]
    assert len(np.intersect1d(c, ft)) == 0
    assert (in_te[c] & (y_te[c] == TOXIC)).all() and (in_te[ft] & (y_te[ft] == TOXIC)).all()
    assert (b_te[c] == Y0).all()
    assert not np.isin(te["remote_te"], np.concatenate([c, ft])).any()
    assert np.isin(te["track_ftest"], ft).all()
    assert all(i.max(initial=-1) < n_tr for i in tr.values())
    assert all(i.max(initial=-1) < n_te for i in te.values())
    return True


def summarize_sets(ds_tok, sets, cfg):
    rows = []
    for split, group in (("train", sets["train"]), ("test", sets["test"])):
        _, y, in_g, b = label_arrays(ds_tok[split], cfg.forget_group)
        for name, idx in group.items():
            if len(idx) == 0:
                continue
            rows.append(dict(split=split, set=name, n=len(idx),
                             forget_group_share=round(float(in_g[idx].mean()), 3),
                             true_toxic_share=round(float(y[idx].mean()), 3),
                             train_label_benign_share=round(float((b[idx] == BENIGN).mean()), 3)))
    return rows


# --------------------------------------------------------------------------- persistence
def save_sets(path, sets, cfg, ds_tok):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    payload = {"config": asdict(cfg),
               "fingerprint": {"train": fingerprint(ds_tok["train"]),
                               "test": fingerprint(ds_tok["test"])},
               "sets": {s: {k: v.tolist() for k, v in d.items()} for s, d in sets.items()}}
    with open(path, "w") as fh:
        json.dump(payload, fh)


def load_or_build_sets(ds_tok, cfg, path, rebuild=False, verbose=True):
    if path and os.path.exists(path) and not rebuild:
        with open(path) as fh:
            payload = json.load(fh)
        same_cfg = payload.get("config") == asdict(cfg)
        same_fp = payload.get("fingerprint") == {"train": fingerprint(ds_tok["train"]),
                                                 "test": fingerprint(ds_tok["test"])}
        if same_cfg and same_fp:
            if verbose:
                print(f"[sets] loaded {path}")
            return {s: {k: np.asarray(v, dtype=np.int64) for k, v in d.items()}
                    for s, d in payload["sets"].items()}
        if verbose:
            reason = "config" if not same_cfg else "tokenized split"
            print(f"[sets] {path} was built for a different {reason}; rebuilding")
    sets = build_sets(ds_tok, cfg)
    check_sets(ds_tok, sets, cfg)
    if path:
        save_sets(path, sets, cfg, ds_tok)
        if verbose:
            print(f"[sets] built and saved {path}")
    return sets


def config_fields():
    return [f.name for f in fields(SetsConfig)]


# --------------------------------------------------------------------------- datasets
def subset(ds_tok, split, idx, forget_group):
    """HF Dataset with input_ids, attention_mask and the *training* label."""
    ds = ds_tok[split]
    _, _, _, b = label_arrays(ds, forget_group)
    return subset_with_labels(ds, idx, b)


def subset_with_labels(ds, idx, train_labels):
    idx = np.asarray(idx, dtype=np.int64)
    sub = ds.select(idx.tolist())
    sub = sub.remove_columns([c for c in sub.column_names
                              if c not in ("input_ids", "attention_mask")])
    return sub.add_column("labels", train_labels[idx].tolist())


def export_csv(ds_tok, sets, cfg, out_dir, text_col="generation"):
    os.makedirs(out_dir, exist_ok=True)
    for split, group in sets.items():
        ds = ds_tok[split]
        _, _, _, b = label_arrays(ds, cfg.forget_group)
        for name, idx in group.items():
            if len(idx) == 0:
                continue
            df = ds.select(idx.tolist()).select_columns(
                ["row_id", "group", text_col, "true_label"]).to_pandas()
            df["train_label"] = b[idx]
            df.to_csv(os.path.join(out_dir, f"{split}_{name}.csv"), index=False)
