"""Shared code for the VGGFace2 face-size unlearning experiments.

Speed design (H100):
  * the whole preprocessed uint8 array lives in GPU memory (~10 GB at 128 px);
    batches are index tensors, so there is no DataLoader and no JPEG decoding
  * augmentation (random scale/shift, flip, brightness/contrast, 20% grayscale as
    in the VGGFace2 paper) runs batched on the GPU
  * bf16 autocast, channels_last, TF32, cuDNN autotuning, optional torch.compile

Evaluation returns the same {"acc", "loss", "mia"} structure as the ToxiGen code,
so rows from Original, Retrain and every method go into one comparison table.
"""
import contextlib
import json
import math
import os
import random
import time

import numpy as np

PIPELINE_VERSION = "2026-09-25.4"   # must match the other pipeline files
import torch
import torch.nn as nn
import torch.nn.functional as F

EVAL_SETS = {"Df(tr)": ("train", "forget"), "Dadj(tr)": ("train", "adjacent"),
             "Drem(tr)": ("train", "remote"), "Ftest(te)": ("test", "ftest"),
             "Dadj(te)": ("test", "adjacent_te"), "Drem(te)": ("test", "remote_te")}
REPORT_KEYS = ("Df(tr)", "Dadj(tr)", "Drem(tr)", "Ftest(te)", "Dadj(te)", "Drem(te)")


# --------------------------------------------------------------------------- setup
def set_seed(seed, deterministic=False):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = deterministic
    torch.backends.cudnn.benchmark = not deterministic


def fast_cuda():
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    if hasattr(torch, "set_float32_matmul_precision"):
        torch.set_float32_matmul_precision("high")


def autocast(device):
    if device.type == "cuda":
        return torch.autocast("cuda", dtype=torch.bfloat16)
    return contextlib.nullcontext()


def sync(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


# --------------------------------------------------------------------------- data
def load_prep(prep_dir):
    import pandas as pd
    with open(os.path.join(prep_dir, "prep_info.json")) as fh:
        info = json.load(fh)
    meta = pd.read_csv(os.path.join(prep_dir, "meta.csv"))
    mm = np.memmap(os.path.join(prep_dir, info["images_file"]), dtype=np.uint8, mode="r",
                   shape=tuple(info["shape"]))
    return info, meta, mm


def load_sets(path, info):
    with open(path) as fh:
        sets = json.load(fh)
    if sets["prep_fingerprint"] != info["fingerprint"]:
        raise SystemExit(f"[error] {path} was built for a different preprocessed array; "
                         "re-run make_forget_sets.py")
    return sets


class GPUImages:
    """All images as one uint8 NHWC tensor on the device, plus labels."""

    def __init__(self, mm, labels, device, chunk=8192):
        t0 = time.time()
        self.device = device
        self.x = torch.empty(tuple(mm.shape), dtype=torch.uint8, device=device)
        for s in range(0, mm.shape[0], chunk):
            self.x[s:s + chunk] = torch.from_numpy(np.array(mm[s:s + chunk])).to(device)   # writable copy
        self.y = torch.as_tensor(np.asarray(labels), dtype=torch.long, device=device)
        sync(device)
        print(f"[data] {self.x.shape[0]:,} images ({self.x.numel() / 1e9:.1f} GB) on {device} "
              f"in {time.time() - t0:.0f}s")

    def idx(self, indices):
        return torch.as_tensor(np.asarray(indices, dtype=np.int64), device=self.device)


def batches(idx, batch_size, shuffle=False, drop_last=False, generator=None):
    if shuffle:
        idx = idx[torch.randperm(len(idx), device=idx.device, generator=generator)]
    n = len(idx)
    stop = n - n % batch_size if drop_last else n
    for s in range(0, stop, batch_size):
        yield idx[s:s + batch_size]


def augment(x, gen=None, scale=(0.9, 1.1), shift=0.08, gray_p=0.2, jitter=0.2):
    """x: float NCHW in [0, 1]. Random scale/shift/flip via one batched affine warp,
    brightness/contrast jitter and random grayscale."""
    B, dev = x.shape[0], x.device
    u = lambda lo, hi: torch.empty(B, device=dev).uniform_(lo, hi, generator=gen)  # noqa: E731
    s = u(*scale)
    flip = torch.where(torch.rand(B, device=dev, generator=gen) < 0.5, -1.0, 1.0)
    theta = torch.zeros(B, 2, 3, device=dev)
    theta[:, 0, 0] = flip / s
    theta[:, 1, 1] = 1.0 / s
    theta[:, 0, 2] = u(-shift, shift) * 2
    theta[:, 1, 2] = u(-shift, shift) * 2
    grid = F.affine_grid(theta, list(x.shape), align_corners=False)
    x = F.grid_sample(x, grid, mode="bilinear", padding_mode="reflection", align_corners=False)
    if jitter > 0:
        mean = x.mean(dim=(1, 2, 3), keepdim=True)
        c = u(1 - jitter, 1 + jitter).view(B, 1, 1, 1)
        b = u(1 - jitter, 1 + jitter).view(B, 1, 1, 1)
        x = ((x - mean) * c + mean) * b
    if gray_p > 0:
        g = (0.299 * x[:, 0] + 0.587 * x[:, 1] + 0.114 * x[:, 2]).unsqueeze(1)
        m = (torch.rand(B, device=dev, generator=gen) < gray_p).view(B, 1, 1, 1)
        x = torch.where(m, g.expand_as(x), x)
    return x.clamp_(0, 1)


def to_input(x_u8, train=False, gen=None):
    """uint8 NHWC -> normalized float NCHW (channels_last), in [-1, 1] like FaceNet."""
    x = x_u8.permute(0, 3, 1, 2).float().div_(255.0)
    if train:
        x = augment(x, gen)
    x = (x - 0.5) / 0.5
    return x.contiguous(memory_format=torch.channels_last)


# --------------------------------------------------------------------------- models
ARCHS = ("resnet50", "resnet18", "inception_resnet_v1")


def build_model(arch, num_classes):
    if arch in ("resnet50", "resnet18"):
        import torchvision
        return getattr(torchvision.models, arch)(weights=None, num_classes=num_classes)
    if arch == "inception_resnet_v1":
        try:
            from facenet_pytorch import InceptionResnetV1
        except ImportError as exc:
            raise SystemExit("[error] facenet-pytorch is not installed. Use "
                             "`pip install facenet-pytorch --no-deps` (plain pip may "
                             "downgrade torch), or pick --arch resnet50.") from exc
        return InceptionResnetV1(pretrained=None, classify=True, num_classes=num_classes)
    raise ValueError(f"unknown arch {arch!r}; choose from {ARCHS}")


def unwrap(model):
    return getattr(model, "_orig_mod", model)


def param_groups(model, weight_decay):
    """No weight decay on BatchNorm parameters and biases."""
    decay, no_decay = [], []
    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue
        (no_decay if p.ndim <= 1 or n.endswith(".bias") else decay).append(p)
    return [{"params": decay, "weight_decay": weight_decay},
            {"params": no_decay, "weight_decay": 0.0}]


def save_checkpoint(path, model, meta):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    torch.save({"model_state": unwrap(model).state_dict(), **meta}, path)


def load_checkpoint(path, device):
    try:
        blob = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:  # torch < 1.13
        blob = torch.load(path, map_location="cpu")
    model = build_model(blob["arch"], blob["num_classes"])
    model.load_state_dict(blob["model_state"])
    return model.to(device).to(memory_format=torch.channels_last).eval(), blob


# --------------------------------------------------------------------------- training
def train_classifier(model, data, train_idx, *, epochs, batch_size, lr, weight_decay,
                     momentum=0.9, warmup_epochs=2, label_smoothing=0.0, seed=0,
                     log_every=100, epoch_callback=None):
    """SGD + nesterov, linear warmup then cosine to 0, bf16 autocast. Returns a log."""
    device = data.device
    opt = torch.optim.SGD(param_groups(unwrap(model), weight_decay), lr=lr,
                          momentum=momentum, nesterov=True)
    steps_per_epoch = len(train_idx) // batch_size
    total = epochs * steps_per_epoch
    warm = max(1, warmup_epochs * steps_per_epoch)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: (s + 1) / warm if s < warm
        else 0.5 * (1 + math.cos(math.pi * (s - warm) / max(1, total - warm))))
    loss_fn = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
    gen = torch.Generator(device=device).manual_seed(seed)

    log, step, t_start = [], 0, time.time()
    for epoch in range(epochs):
        model.train()
        t0, run_loss, run_correct, seen = time.time(), torch.zeros((), device=device), \
            torch.zeros((), device=device), 0
        for b in batches(train_idx, batch_size, shuffle=True, drop_last=True, generator=gen):
            x = to_input(data.x[b], train=True, gen=gen)
            y = data.y[b]
            with autocast(device):
                logits = model(x)
                loss = loss_fn(logits.float(), y)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            sched.step()
            run_loss += loss.detach() * len(b)
            run_correct += (logits.argmax(1) == y).sum()
            seen += len(b)
            step += 1
            if log_every and step % log_every == 0:
                print(f"  step {step}/{total} loss={run_loss.item() / seen:.4f} "
                      f"lr={sched.get_last_lr()[0]:.4f}", flush=True)
        sync(device)
        dt = time.time() - t0
        entry = {"epoch": epoch + 1, "loss": run_loss.item() / max(seen, 1),
                 "train_acc": 100.0 * run_correct.item() / max(seen, 1),
                 "seconds": dt, "img_per_s": seen / dt}
        if epoch_callback is not None:
            entry.update(epoch_callback(epoch, model) or {})
        log.append(entry)
        extra = " ".join(f"{k}={v:.2f}" for k, v in entry.items()
                         if k not in ("epoch", "loss", "train_acc", "seconds", "img_per_s"))
        print(f"[epoch {epoch + 1}/{epochs}] loss={entry['loss']:.4f} train_acc={entry['train_acc']:.2f} "
              f"{dt:.1f}s ({entry['img_per_s']:.0f} img/s) {extra}", flush=True)
    return log, time.time() - t_start


# --------------------------------------------------------------------------- evaluation
@torch.no_grad()
def per_sample(model, data, idx, batch_size=1024):
    """Per-sample CE loss and correctness, eval mode, no augmentation."""
    was = model.training
    model.eval()
    losses, correct = [], []
    for b in batches(idx, batch_size):
        with autocast(data.device):
            logits = model(to_input(data.x[b]))
        logits = logits.float()
        losses.append(F.cross_entropy(logits, data.y[b], reduction="none"))
        correct.append(logits.argmax(1) == data.y[b])
    model.train(was)
    if not losses:
        return np.zeros(0), np.zeros(0, dtype=bool)
    return torch.cat(losses).cpu().numpy(), torch.cat(correct).cpu().numpy()


def acc_of(model, data, idx, batch_size=1024):
    _, c = per_sample(model, data, idx, batch_size)
    return 100.0 * c.mean() if len(c) else float("nan")


def _mia(member, nonmember, n_members=2000, seed=7):
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import train_test_split
    rng = np.random.default_rng(seed)
    n = min(n_members, len(member), len(nonmember))
    m, nm = rng.choice(member, n, replace=False), rng.choice(nonmember, n, replace=False)
    X = np.concatenate([m, nm]).reshape(-1, 1)
    y = np.concatenate([np.ones(n), np.zeros(n)])
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3, random_state=seed, stratify=y)
    clf = LogisticRegression(max_iter=1000).fit(Xtr, ytr)
    return clf, float(clf.score(Xte, yte))


def _auc(a, b):
    from sklearn.metrics import roc_auc_score
    if len(a) == 0 or len(b) == 0:
        return float("nan")
    return float(roc_auc_score(np.r_[np.ones(len(a)), np.zeros(len(b))], np.r_[a, b]))


def full_evaluation(model, data, sets, tag="", batch_size=1024, mia_n_members=2000, mia_seed=7):
    """Accuracy/loss on every eval set plus loss-threshold MIA fit on Drem(tr) vs Drem(te)."""
    stats = {}
    for name, (split, key) in EVAL_SETS.items():
        idx = sets["sets"][split].get(key, [])
        if len(idx):
            stats[name] = per_sample(model, data, data.idx(idx), batch_size)
    acc = {k: 100.0 * c.mean() for k, (_, c) in stats.items()}
    loss = {k: float(l.mean()) for k, (l, _) in stats.items()}
    # a partly broken model can give inf/NaN losses: cap them so the MIA still runs
    losses = {k: np.nan_to_num(l, nan=1e4, posinf=1e4, neginf=0.0) for k, (l, _) in stats.items()}
    n_bad = sum(int((~np.isfinite(l)).sum()) for l, _ in stats.values())
    if n_bad:
        print(f"[warn] {n_bad} non-finite per-sample losses (capped at 1e4 for the MIA)")
    clf, atk = _mia(losses["Drem(tr)"], losses["Drem(te)"], mia_n_members, mia_seed)
    nonmem = lambda l: float((clf.predict(l.reshape(-1, 1)) == 0).mean())  # noqa: E731
    mia = {"attacker_acc": atk, "Df": nonmem(losses["Df(tr)"]), "Dadj": nonmem(losses["Dadj(tr)"]),
           "auc_adj": _auc(losses["Df(tr)"], losses["Dadj(tr)"]),
           "auc_unseen": _auc(losses["Df(tr)"], losses["Ftest(te)"])}
    mia["gap"] = mia["Df"] - mia["Dadj"]
    if tag:
        print(f"[{tag}] " + "  ".join(f"{k}={v:.2f}" for k, v in acc.items()))
        print(f"[{tag}] MIA attacker={atk:.3f} Df={mia['Df']:.3f} Dadj={mia['Dadj']:.3f} "
              f"AUC(Df,Dadj)={mia['auc_adj']:.3f} AUC(Df,Ftest)={mia['auc_unseen']:.3f}")
    return {"acc": acc, "loss": loss, "mia": mia}


@torch.no_grad()
def penultimate_features(model, data, idx, batch_size=1024):
    """Input to the last nn.Linear (the classifier), via a forward hook."""
    last = [m for m in unwrap(model).modules() if isinstance(m, nn.Linear)][-1]
    feats = []
    hook = last.register_forward_hook(lambda mod, inp, out: feats.append(inp[0].float()))
    model.eval()
    for b in batches(idx, batch_size):
        with autocast(data.device):
            model(to_input(data.x[b]))
    hook.remove()
    return torch.cat(feats).cpu().numpy()


def separability_auc(model, data, sets, seed=0):
    """How separable D_f (HQ) is from D_adj (LQ) in the model's representation:
    5-fold cross-validated ROC AUC of a logistic probe on penultimate features.
    ~0.5: instance-like (ceiling of Prop. 1 near 0); ~1.0: class-like."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import StratifiedKFold, cross_val_score
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    f = penultimate_features(model, data, data.idx(sets["sets"]["train"]["forget"]))
    a = penultimate_features(model, data, data.idx(sets["sets"]["train"]["adjacent"]))
    X, y = np.r_[f, a], np.r_[np.ones(len(f)), np.zeros(len(a))]
    probe = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, C=0.1))
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    return float(cross_val_score(probe, X, y, cv=cv, scoring="roc_auc").mean())


# --------------------------------------------------------------------------- reporting
def print_comparison(rows, reference_name=None, runtimes=None):
    cols = [k for k in REPORT_KEYS if all(k in r["acc"] for r in rows.values())]
    w = max(16, max(len(n) for n in rows) + 2)
    print("\nAccuracy (%)")
    print(f"{'Model':{w}s}" + "".join(f"{c:>11s}" for c in cols))
    print("-" * (w + 11 * len(cols)))
    for n, r in rows.items():
        print(f"{n:{w}s}" + "".join(f"{r['acc'][c]:11.2f}" for c in cols))
    print("\nMIA efficacy (fraction called non-member)")
    print(f"{'Model':{w}s}{'Df':>9s}{'Dadj':>9s}{'gap':>9s}{'AUC(f,adj)':>12s}{'AUC(f,unseen)':>15s}")
    print("-" * (w + 54))
    for n, r in rows.items():
        m = r["mia"]
        print(f"{n:{w}s}{m['Df']:9.3f}{m['Dadj']:9.3f}{m['Df'] - m['Dadj']:9.3f}"
              f"{m['auc_adj']:12.3f}{m['auc_unseen']:15.3f}")
    deviations = {}
    if reference_name in rows:
        g = rows[reference_name]
        print(f"\nAbsolute deviation from {reference_name} (L1 over {', '.join(cols)}), lower is better")
        print(f"{'Model':{w}s}{'|dDf|':>8s}{'|dAdj|':>8s}{'|dFtest|':>10s}{'|dMIA_Df|':>11s}{'L1(acc)':>9s}")
        print("-" * (w + 46))
        for n, r in rows.items():
            if n == reference_name:
                continue
            d = {k: abs(r["acc"][k] - g["acc"][k]) for k in cols}
            dm = abs(r["mia"]["Df"] - g["mia"]["Df"])
            deviations[n] = {"acc": d, "L1": sum(d.values()), "mia_Df": dm}
            print(f"{n:{w}s}{d['Df(tr)']:8.2f}{d['Dadj(tr)']:8.2f}{d['Ftest(te)']:10.2f}"
                  f"{dm:11.3f}{sum(d.values()):9.2f}")
    if runtimes:
        print(f"\n{'Runtime':28s}{'seconds':>10s}\n" + "-" * 38)
        for k, v in runtimes.items():
            print(f"{k:28s}{v:10.1f}")
    return deviations
