"""Forget / retain sets for Social Bias Frames (torch-free).

Task: binary offense classification (post label = mean offensiveYN >= 0.5), trained on
the true labels. The forget request is a provenance deletion: every post from one
source (default r/meanjokes) is withdrawn.

  class     D_f     = train posts from the forget source            (source-based removal)
  (source)  D_r^adj = train posts from the sibling joke subreddits   (r/darkjokes, r/offensivejokes)
            D_r^rem = all other train posts
  instance  D_f     = random half of the forget source's train posts (mu_f = mu_adj)
            D_r^adj = the other half
            D_r^rem = everything else (sibling subreddits included)

Held-out sets (never trained on):
  D_cal       dev posts from the forget source        (Swift forget anchor)
  Ftest       test posts from the forget source       (evaluation only)
  Dadj(te)    test posts from the sibling sources     (class split only)
  Drem(te)    all other test posts                    (evaluation, MIA non-members)
  D_r^probe   retain subsample, equal quota per (adj/rem, source, label) stratum
  probe_holdout, select_*, track_*: small retain/forget slices for the sweep rule,
              checkpoint selection and accuracy curves (as in the ToxiGen pipeline)

Indices are row positions in the tokenized cache (``ds_tok["train"]`` / ``ds_tok["test"]``;
the "test" split holds SBIC dev + test, told apart by ``orig_split``). Same public API as
the ToxiGen sets module, so the training and Swift scripts are shared.
"""
import hashlib
import json
import os
from dataclasses import asdict, dataclass, fields

import numpy as np

OFFENSIVE, NOT_OFFENSIVE = 1, 0
TRAIN_SETS = ("forget", "adjacent", "remote", "retain", "probe", "probe_holdout",
              "select_forget", "select_probe", "track_forget", "track_adjacent")
TEST_SETS = ("cal", "ftest", "remote_te", "adjacent_te", "track_ftest")
SPLITS = ("class", "instance")


@dataclass
class SetsConfig:
    split: str = "class"
    forget_source: str = "r/meanjokes"
    adjacent_sources: str = "r/darkjokes,r/offensivejokes"   # comma-separated (JSON-stable)
    forget_fraction: float = 0.5
    split_seed: int = 1234
    cal_fraction: float = 1.0
    cal_seed: int = 99
    probe_size: int = 5000
    probe_seed: int = 4242
    probe_holdout_n: int = 2000
    select_n: int = 512
    select_seed: int = 2025
    track_n: int = 512


def _norm_sources(text):
    return ",".join(sorted({s.strip().lower() for s in str(text).split(",") if s.strip()}))


def add_sets_args(parser, default_split="class"):
    g = parser.add_argument_group("sets")
    g.add_argument("--split", choices=list(SPLITS), default=default_split,
                   help="class: forget one source, siblings are adjacent; instance: two random halves")
    g.add_argument("--forget-source", default=SetsConfig.forget_source)
    g.add_argument("--adjacent-sources", default=SetsConfig.adjacent_sources,
                   help="comma-separated sources that form D_r^adj in the class split")
    g.add_argument("--forget-fraction", type=float, default=SetsConfig.forget_fraction,
                   help="instance split: share of the forget source's train posts in D_f")
    g.add_argument("--split-seed", type=int, default=SetsConfig.split_seed)
    g.add_argument("--cal-fraction", type=float, default=SetsConfig.cal_fraction,
                   help="share of the forget source's dev posts used as D_cal")
    g.add_argument("--cal-seed", type=int, default=SetsConfig.cal_seed)
    g.add_argument("--probe-size", type=int, default=SetsConfig.probe_size)
    g.add_argument("--probe-seed", type=int, default=SetsConfig.probe_seed)
    g.add_argument("--select-n", type=int, default=SetsConfig.select_n)
    g.add_argument("--track-n", type=int, default=SetsConfig.track_n)
    g.add_argument("--rebuild-sets", action="store_true")
    return g


def sets_config_from_args(args, forget_group=None, split=None):
    forget = getattr(args, "forget_source", None) or forget_group or SetsConfig.forget_source
    adjacent = getattr(args, "adjacent_sources", None) or SetsConfig.adjacent_sources
    return SetsConfig(split=split or args.split, forget_source=forget.strip().lower(),
                      adjacent_sources=_norm_sources(adjacent),
                      forget_fraction=args.forget_fraction, split_seed=args.split_seed,
                      cal_fraction=args.cal_fraction, cal_seed=args.cal_seed,
                      probe_size=args.probe_size, probe_seed=args.probe_seed,
                      select_n=args.select_n, track_n=args.track_n)


def default_sets_path(repo_dir, split):
    return os.path.join(repo_dir, "data", "sets", split, "sets.json")


def fingerprint(ds):
    rid = np.asarray(ds["row_id"], dtype=np.int64)
    return hashlib.sha1(rid.tobytes()).hexdigest()


# --------------------------------------------------------------------------- construction
def _arrays(ds):
    src = np.asarray([str(s) for s in ds["source"]])
    y = np.asarray(ds["true_label"], dtype=np.int64)
    orig = np.asarray([str(s) for s in ds["orig_split"]])
    return src, y, orig


def _choice(rng, pool, n):
    pool = np.asarray(pool)
    return np.sort(rng.choice(pool, min(n, len(pool)), replace=False))


def stratified_subsample(indices, strata, size, seed):
    """Equal quota per stratum (capped by stratum size)."""
    indices, strata = np.asarray(indices), np.asarray(strata)
    classes = np.unique(strata)
    per_class = max(1, int(round(size / len(classes))))
    rng = np.random.default_rng(seed)
    picked = [rng.choice(indices[strata == c], min(per_class, int((strata == c).sum())), replace=False)
              for c in classes]
    return np.sort(np.concatenate(picked))


def build_sets(ds_tok, cfg):
    src_tr, y_tr, _ = _arrays(ds_tok["train"])
    src_te, _, orig_te = _arrays(ds_tok["test"])
    fs = cfg.forget_source
    adj_src = [s for s in cfg.adjacent_sources.split(",") if s]
    known = set(src_tr)
    if fs not in known:
        raise ValueError(f"forget source {fs!r} not in the train split; sources: {sorted(known)}")
    if cfg.split == "class":
        missing = [s for s in adj_src if s not in known]
        if missing or not adj_src:
            raise ValueError(f"adjacent source(s) {missing or adj_src} not in the train split; "
                             f"sources: {sorted(known)}")
        if fs in adj_src:
            raise ValueError("the forget source cannot also be adjacent")

    in_f_tr = src_tr == fs
    in_a_tr = np.isin(src_tr, adj_src)
    if cfg.split == "class":
        forget = np.where(in_f_tr)[0]
        adjacent = np.where(in_a_tr)[0]
        remote = np.where(~in_f_tr & ~in_a_tr)[0]
    elif cfg.split == "instance":
        perm = np.random.default_rng(cfg.split_seed).permutation(np.where(in_f_tr)[0])
        n_f = int(round(cfg.forget_fraction * len(perm)))
        forget, adjacent = np.sort(perm[:n_f]), np.sort(perm[n_f:])
        remote = np.where(~in_f_tr)[0]
    else:
        raise ValueError(f"unknown split {cfg.split!r}; choose from {SPLITS}")
    retain = np.sort(np.concatenate([remote, adjacent]))

    is_dev, is_test = orig_te == "dev", orig_te == "test"
    cal_all = np.where(is_dev & (src_te == fs))[0]
    if cfg.cal_fraction < 1.0:
        n_cal = max(2, int(round(cfg.cal_fraction * len(cal_all))))
        cal = _choice(np.random.default_rng(cfg.cal_seed), cal_all, n_cal)
    else:
        cal = cal_all
    ftest = np.where(is_test & (src_te == fs))[0]
    if cfg.split == "class":
        adjacent_te = np.where(is_test & np.isin(src_te, adj_src))[0]
        remote_te = np.where(is_test & (src_te != fs) & ~np.isin(src_te, adj_src))[0]
    else:
        adjacent_te = np.array([], dtype=np.int64)    # both halves defined by train membership
        remote_te = np.where(is_test & (src_te != fs))[0]

    is_adj = np.isin(retain, adjacent)
    strata = np.array([f"{'adj' if a else 'rem'}|{s}|{y}"
                       for a, s, y in zip(is_adj, src_tr[retain], y_tr[retain])])
    probe = stratified_subsample(retain, strata, cfg.probe_size, cfg.probe_seed)
    rng = np.random.default_rng(cfg.probe_seed + 1)
    probe_holdout = _choice(rng, np.setdiff1d(retain, probe), cfg.probe_holdout_n)

    rng = np.random.default_rng(cfg.select_seed)
    select_forget = _choice(rng, forget, cfg.select_n)
    select_probe = _choice(rng, probe, cfg.select_n)
    rng = np.random.default_rng(cfg.select_seed + 1)
    as_int = lambda a: np.asarray(a, dtype=np.int64)  # noqa: E731
    return {
        "train": {k: as_int(v) for k, v in dict(
            forget=forget, adjacent=adjacent, remote=remote, retain=retain, probe=probe,
            probe_holdout=probe_holdout, select_forget=select_forget, select_probe=select_probe,
            track_forget=_choice(rng, forget, cfg.track_n),
            track_adjacent=_choice(rng, adjacent, cfg.track_n)).items()},
        "test": {k: as_int(v) for k, v in dict(
            cal=cal, ftest=ftest, remote_te=remote_te, adjacent_te=adjacent_te,
            track_ftest=_choice(rng, ftest, cfg.track_n)).items()},
    }


# --------------------------------------------------------------------------- checks / summary
def check_sets(ds_tok, sets, cfg):
    src_tr, _, _ = _arrays(ds_tok["train"])
    src_te, _, orig_te = _arrays(ds_tok["test"])
    fs = cfg.forget_source
    adj_src = [s for s in cfg.adjacent_sources.split(",") if s]
    tr, te = sets["train"], sets["test"]
    f, a, r = tr["forget"], tr["adjacent"], tr["remote"]

    assert len(f) and len(a), "D_f and D_r^adj must be non-empty"
    assert len(np.intersect1d(f, a)) == len(np.intersect1d(f, r)) == len(np.intersect1d(a, r)) == 0
    assert len(f) + len(a) + len(r) == len(src_tr), "D_f + D_adj + D_rem must cover the train split"
    assert np.array_equal(tr["retain"], np.sort(np.concatenate([a, r])))
    assert (src_tr[f] == fs).all()
    if cfg.split == "class":
        assert np.isin(src_tr[a], adj_src).all()
        assert not np.isin(src_tr[r], adj_src + [fs]).any()
    else:
        assert (src_tr[a] == fs).all() and not (src_tr[r] == fs).any()
        assert abs(len(f) - cfg.forget_fraction * (len(f) + len(a))) <= 1
    assert np.isin(tr["probe"], tr["retain"]).all() and not np.isin(tr["probe"], f).any()
    assert not np.isin(tr["probe_holdout"], tr["probe"]).any()
    assert np.isin(tr["probe_holdout"], tr["retain"]).all()
    assert np.isin(tr["select_forget"], f).all() and np.isin(tr["select_probe"], tr["probe"]).all()
    assert np.isin(tr["track_forget"], f).all() and np.isin(tr["track_adjacent"], a).all()

    c, ft = te["cal"], te["ftest"]
    assert len(c) and len(ft), "D_cal and Ftest must be non-empty"
    assert (orig_te[c] == "dev").all() and (src_te[c] == fs).all(), "D_cal = dev posts of the forget source"
    for key in ("ftest", "adjacent_te", "remote_te"):
        assert (orig_te[te[key]] == "test").all(), f"{key} must come from the SBIC test split"
    assert (src_te[ft] == fs).all()
    assert not (src_te[te["remote_te"]] == fs).any()
    groups = [np.asarray(v) for v in (ft, te["adjacent_te"], te["remote_te"])]
    allte = np.concatenate(groups)
    assert len(allte) == len(np.unique(allte)), "test sets overlap"
    assert np.isin(te["track_ftest"], ft).all()
    return True


def summarize_sets(ds_tok, sets, cfg):
    adj_src = [s for s in cfg.adjacent_sources.split(",") if s]
    rows = []
    for split in ("train", "test"):
        src, y, orig = _arrays(ds_tok[split])
        for name, idx in sets[split].items():
            if len(idx) == 0:
                continue
            rows.append(dict(split=split, set=name, n=len(idx),
                             from_split=",".join(sorted(set(orig[idx]))),
                             forget_source_share=round(float((src[idx] == cfg.forget_source).mean()), 3),
                             sibling_share=round(float(np.isin(src[idx], adj_src).mean()), 3),
                             offensive_share=round(float(y[idx].mean()), 3),
                             n_sources=int(len(set(src[idx])))))
    return rows


# --------------------------------------------------------------------------- persistence
def save_sets(path, sets, cfg, ds_tok):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    payload = {"config": asdict(cfg),
               "fingerprint": {"train": fingerprint(ds_tok["train"]), "test": fingerprint(ds_tok["test"])},
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
            print(f"[sets] {path} was built for a different {'config' if not same_cfg else 'tokenized split'}; "
                  "rebuilding (results from the old sets are not comparable)")
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
def subset(ds_tok, split, idx, forget_group=None):
    """HF Dataset with input_ids, attention_mask and the (true) offense label."""
    ds = ds_tok[split]
    return subset_with_labels(ds, idx, np.asarray(ds["true_label"], dtype=np.int64))


def subset_with_labels(ds, idx, labels):
    idx = np.asarray(idx, dtype=np.int64)
    sub = ds.select(idx.tolist())
    sub = sub.remove_columns([c for c in sub.column_names if c not in ("input_ids", "attention_mask")])
    return sub.add_column("labels", labels[idx].tolist())


def export_csv(ds_tok, sets, cfg, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    for split, group in sets.items():
        ds = ds_tok[split]
        for name, idx in group.items():
            if len(idx) == 0:
                continue
            ds.select(np.asarray(idx).tolist()).select_columns(
                ["row_id", "orig_split", "source", "post", "off_mean", "true_label"]
            ).to_pandas().to_csv(os.path.join(out_dir, f"{split}_{name}.csv"), index=False)
