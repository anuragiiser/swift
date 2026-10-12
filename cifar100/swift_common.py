"""Shared torch code for Swift on CIFAR-100: W2 machinery, evaluation, MIA,
the FitCheap probe and reporting.

This is the reference implementation the other dataset folders were ported
from; `toxigen/swift_common.py` and `vggface2/vgg_common.py` are the text and
face analogues of the same functions.
"""
import contextlib
import copy
import random

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR

try:
    from tqdm.auto import tqdm
except ImportError:  # pragma: no cover
    def tqdm(x=None, **_):
        return x

PIPELINE_VERSION = "2026-10-12.1"   # must match swift_cifar100.py

REPORT_KEYS = ("Df(tr)", "Dadj(tr)", "Drem(tr)", "Ftest(te)", "Drem(te)")


# --------------------------------------------------------------------------- runtime
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def enable_fast_math():
    """TF32 matmul and cuDNN autotuning. Relaxes the determinism of set_seed:
    results stay statistically equivalent but are not bitwise reproducible."""
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True


def resolve_amp(mode, device):
    """Return (autocast dtype or None, needs GradScaler)."""
    if device.type != "cuda" or mode == "none":
        return None, False
    if mode == "bf16":
        if torch.cuda.is_bf16_supported():
            return torch.bfloat16, False
        print("[warn] bf16 unsupported on this GPU; using fp16 with GradScaler")
        return torch.float16, True
    return torch.float16, True


def autocast(dtype):
    if dtype is None:
        return contextlib.nullcontext()
    return torch.autocast(device_type="cuda", dtype=dtype)


def make_grad_scaler(enabled):
    if hasattr(torch, "amp") and hasattr(torch.amp, "GradScaler"):
        return torch.amp.GradScaler("cuda", enabled=enabled)
    return torch.cuda.amp.GradScaler(enabled=enabled)   # torch < 2.3


def cycle(loader):
    """Endless iterator. An empty loader would spin here without ever
    yielding, which is a silent hang rather than an error, so refuse it."""
    if len(loader) == 0:
        raise ValueError(
            f"cycle() got a loader with 0 batches (dataset={len(loader.dataset)}, "
            f"batch_size={loader.batch_size}, drop_last={loader.drop_last}). "
            "Lower the batch size or set drop_last=False.")
    while True:
        for batch in loader:
            yield batch


def resnet18_superclass(n_classes=20, pretrained=True):
    """The Original architecture: ImageNet-pretrained ResNet-18, 20-way head."""
    from torchvision import models

    weights = models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
    model = models.resnet18(weights=weights)
    model.fc = nn.Linear(model.fc.in_features, n_classes)
    return model


def load_model(path, device, n_classes=20):
    """Load a checkpoint and return it in eval mode.

    Accepts this folder's format ("model_state" plus metadata), the notebook's
    older "model_state_dict", a bare state dict, and a whole pickled module,
    which is how the upstream repo saves the Original model.
    """
    blob = torch.load(path, map_location=device, weights_only=False)
    if isinstance(blob, nn.Module):
        return blob.to(device).eval()
    state = blob
    if isinstance(blob, dict):
        for key in ("model_state", "model_state_dict", "state_dict"):
            if key in blob:
                state = blob[key]
                break
    model = resnet18_superclass(n_classes, pretrained=False)
    model.load_state_dict(state)
    return model.to(device).eval()


def save_model(model, path, extra=None):
    """One .pt holding the weights plus enough metadata to re-instantiate."""
    import os

    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    payload = {"model_state": model.state_dict(), "arch": "resnet18",
               "num_classes": getattr(getattr(model, "fc", None), "out_features", None)}
    if extra:
        payload.update(extra)
    torch.save(payload, path)
    return path


# --------------------------------------------------------------------------- evaluation
@torch.no_grad()
def per_sample_stats(model, loader, device, amp_dtype=None, desc=None):
    """Per-sample CE losses and correctness, in eval mode (restores the mode)."""
    was_training = model.training
    model.eval()
    losses, correct = [], []
    it = tqdm(loader, desc=desc, unit="batch", leave=False) if desc and len(loader) > 1 else loader
    for images, labels in it:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True).long()
        with autocast(amp_dtype):
            logits = model(images)
        logits = logits.float()
        losses.append(F.cross_entropy(logits, labels, reduction="none").cpu())
        correct.append(logits.argmax(-1).eq(labels).cpu())
    model.train(was_training)
    if not losses:
        return torch.empty(0), torch.empty(0, dtype=torch.bool)
    return torch.cat(losses), torch.cat(correct)


def per_sample_losses(model, loader, device, amp_dtype=None, desc=None):
    return per_sample_stats(model, loader, device, amp_dtype, desc)[0]


def acc_and_loss(model, loader, device, amp_dtype=None, desc=None):
    losses, correct = per_sample_stats(model, loader, device, amp_dtype, desc)
    if losses.numel() == 0:
        return {"acc": float("nan"), "loss": float("nan"), "n": 0}
    return {"acc": 100.0 * correct.float().mean().item(), "loss": losses.mean().item(),
            "n": int(losses.numel())}


def fit_mia_attacker(member_losses, nonmember_losses, n_members=2000, seed=7):
    """Loss-threshold attacker: logistic regression on the per-sample loss,
    fitted on retain-train members against retain-test non-members."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import train_test_split

    mem, non = np.asarray(member_losses), np.asarray(nonmember_losses)
    rng = np.random.default_rng(seed)
    n = min(n_members, len(mem), len(non))
    mem, non = rng.choice(mem, n, replace=False), rng.choice(non, n, replace=False)
    X = np.concatenate([mem, non]).reshape(-1, 1)
    y = np.concatenate([np.ones(n), np.zeros(n)])
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.3, random_state=seed, stratify=y)
    clf = LogisticRegression(max_iter=1000).fit(X_tr, y_tr)
    return clf, float(clf.score(X_te, y_te))


def mia_efficacy(clf, losses):
    """Fraction of samples the attacker calls NON-member, so higher means more
    of D_f looks unseen."""
    return float((clf.predict(np.asarray(losses).reshape(-1, 1)) == 0).mean())


def loss_auc(la, lb):
    from sklearn.metrics import roc_auc_score

    la, lb = np.asarray(la), np.asarray(lb)
    y = np.concatenate([np.ones_like(la), np.zeros_like(lb)])
    return float(roc_auc_score(y, np.concatenate([la, lb])))


def w2_numpy(a, b, n_quantiles=256):
    qs = (np.arange(n_quantiles) + 0.5) / n_quantiles
    return float(np.mean((np.quantile(a, qs) - np.quantile(b, qs)) ** 2))


def ks_numpy(a, b):
    grid = np.concatenate([a, b])
    fa = np.searchsorted(np.sort(a), grid, side="right") / len(a)
    fb = np.searchsorted(np.sort(b), grid, side="right") / len(b)
    return float(np.max(np.abs(fa - fb)))


def full_evaluation(model, eval_loaders, device, amp_dtype=None, mia_n_members=2000,
                    mia_seed=7, tag=""):
    """Accuracy/loss on every eval set, loss-threshold MIA (fitted on Drem(tr)
    against Drem(te)) and loss AUCs. Returns plain floats plus per-sample
    losses under 'losses'.

    A diverged model gives non-finite losses, which would raise inside sklearn
    and abort the run, so they are clamped and the accuracy columns still
    describe what happened.
    """
    stats = {k: per_sample_stats(model, v, device, amp_dtype, desc=f"{tag}: {k}")
             for k, v in eval_loaders.items() if len(v.dataset) > 0}
    acc = {k: 100.0 * c.float().mean().item() for k, (_, c) in stats.items()}
    loss = {k: l.mean().item() for k, (l, _) in stats.items()}
    losses = {k: np.nan_to_num(l.numpy(), nan=1e4, posinf=1e4, neginf=0.0)
              for k, (l, _) in stats.items()}

    clf, attacker_acc = fit_mia_attacker(losses["Drem(tr)"], losses["Drem(te)"],
                                         n_members=mia_n_members, seed=mia_seed)
    mia = {"attacker_acc": attacker_acc,
           "Df": mia_efficacy(clf, losses["Df(tr)"]),
           "Dadj": mia_efficacy(clf, losses["Dadj(tr)"]),
           "auc_adj": loss_auc(losses["Df(tr)"], losses["Dadj(tr)"]),
           "auc_unseen": loss_auc(losses["Df(tr)"], losses["Ftest(te)"])}
    mia["gap"] = mia["Df"] - mia["Dadj"]
    if tag:
        print(f"[{tag}] " + "  ".join(f"{k}={v:.2f}" for k, v in acc.items()))
        print(f"[{tag}] MIA attacker={attacker_acc:.3f} efficacy Df={mia['Df']:.3f} "
              f"Dadj={mia['Dadj']:.3f} | AUC(Df,Dadj)={mia['auc_adj']:.3f} "
              f"AUC(Df,Ftest)={mia['auc_unseen']:.3f}")
    return {"acc": acc, "loss": loss, "mia": mia, "losses": losses}


def strip_losses(result):
    return {k: v for k, v in result.items() if k != "losses"}


# --------------------------------------------------------------------------- 1-D Wasserstein
def midpoint_grid(n, device, dtype=torch.float32):
    """(i + 0.5)/n quantile levels: avoids the noisy 0/1 order statistics."""
    return (torch.arange(n, device=device, dtype=dtype) + 0.5) / n


def reference_quantiles(samples, n_quantiles, device):
    """Freeze a sample set into a fixed quantile function (the sg[.] side)."""
    x = samples.detach().float().to(device).flatten()
    qs = midpoint_grid(n_quantiles, device, x.dtype)
    return torch.quantile(x, qs).contiguous()


def w2_sq_to_reference(losses, ref_q, one_sided=False):
    """W2^2(empirical(losses), reference).

    `losses` keeps its graph: sorting is a permutation, so every sample in the
    batch receives gradient. `ref_q` is a detached quantile vector.
    """
    x = losses.float().flatten()
    n = x.numel()
    xs, _ = torch.sort(x)
    qs = midpoint_grid(n, x.device, x.dtype)
    ref = torch.quantile(ref_q, qs) if ref_q.numel() != n else ref_q
    diff = xs - ref
    if one_sided:
        diff = torch.relu(diff)          # only penalise losses ABOVE the reference
    return torch.mean(diff ** 2)


def w2_sq_between(losses_a, losses_b, one_sided=False):
    """W2^2 where side B is detached in place (stop-gradient on D_cal)."""
    b = losses_b.detach().float().flatten()
    a = losses_a.float().flatten()
    n = a.numel()
    xs, _ = torch.sort(a)
    ref = (torch.sort(b)[0] if b.numel() == n
           else torch.quantile(b, midpoint_grid(n, a.device, a.dtype)))
    diff = xs - ref
    if one_sided:
        diff = torch.relu(diff)
    return torch.mean(diff ** 2)


@torch.no_grad()
def estimate_w2_floor(ref_samples, batch_size, n_quantiles, one_sided=False,
                      n_boot=200, seed=0, q=0.5):
    """Irreducible W2^2 between a size-`batch_size` draw and its own reference.

    A finite mini-batch never matches its parent distribution, so the term has
    a strictly positive floor. A tolerance below that floor is violated by
    sampling noise alone and the dual variable diverges.
    """
    x = ref_samples.detach().float().flatten()
    ref_q = reference_quantiles(x, n_quantiles, x.device)
    g = torch.Generator(device="cpu").manual_seed(seed)
    vals = []
    for _ in range(n_boot):
        idx = torch.randint(0, x.numel(), (batch_size,), generator=g).to(x.device)
        vals.append(w2_sq_to_reference(x[idx], ref_q, one_sided).item())
    return float(np.quantile(vals, q))


# --------------------------------------------------------------------------- FitCheap probe
def last_linear(model):
    name, module = None, None
    for n_, m in model.named_modules():
        if isinstance(m, nn.Linear):
            name, module = n_, m
    if module is None:
        raise RuntimeError("no nn.Linear head found in the checkpoint")
    return name, module


def fit_cheap_probe(base_model, probe_loader, device, steps=300, lr=1e-3,
                    weight_decay=0.01, reinit="head_layer4", bn_mode="eval",
                    amp_dtype=None, use_scaler=False, log_every=50, max_norm=5.0):
    """A few hundred cheap steps on D_r^probe only; never sees D_f.

    The classifier head is re-initialised first. Without that the probe is
    theta_0 nudged on data theta_0 already fits, so gradients are near zero and
    the "uncontaminated" reference would be the contaminated one. Resetting the
    head breaks theta_0's direct D_f -> logit pathway.

    The backbone stays frozen, so this is a cheap shortcut toward
    theta^retrain, not a claim of independence from theta_0: the frozen
    features were still learned with D_f present.

    reinit: head         reset the classifier, train the classifier only
            head_layer4  reset the classifier, train classifier + layer4
            none         keep theta_0's head, fine-tune it on D_r^probe
    bn_mode: 'eval' keeps theta_0's BatchNorm running statistics frozen.
    """
    probe = copy.deepcopy(base_model).to(device)
    head_name, head = last_linear(probe)
    if reinit in ("head", "head_layer4"):
        head.reset_parameters()

    for p in probe.parameters():
        p.requires_grad_(False)
    trainable = list(head.parameters())
    for p in trainable:
        p.requires_grad_(True)
    if reinit == "head_layer4" and hasattr(probe, "layer4"):
        for p in probe.layer4.parameters():
            p.requires_grad_(True)
            trainable.append(p)

    probe.train() if bn_mode == "train" else probe.eval()
    opt = optim.AdamW(trainable, lr=lr, weight_decay=weight_decay)
    sched = CosineAnnealingLR(opt, T_max=max(steps, 1), eta_min=lr * 1e-2)
    scaler = make_grad_scaler(use_scaler)
    stream = cycle(probe_loader)
    print(f"[probe] head='{head_name}' trainable="
          f"{sum(p.numel() for p in trainable) / 1e6:.2f}M steps={steps} "
          f"reinit={reinit} bn={bn_mode}")

    for s in tqdm(range(steps), desc="FitCheap(D_r^probe)", unit="step", leave=False):
        images, labels = next(stream)
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True).long()
        opt.zero_grad(set_to_none=True)
        with autocast(amp_dtype):
            loss = F.cross_entropy(probe(images).float(), labels)
        scaler.scale(loss).backward()
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(trainable, max_norm)
        scaler.step(opt)
        scaler.update()
        sched.step()
        if log_every and (s % log_every == 0 or s == steps - 1):
            print(f"  [probe {s:4d}] ce={loss.item():.4f}", flush=True)

    probe.eval()
    for p in probe.parameters():
        p.requires_grad_(False)
    return probe


# --------------------------------------------------------------------------- reporting
def print_comparison(rows, reference_name=None, runtimes=None):
    """rows: {name: {"acc": {...}, "mia": {...}}}; reference_name: the Retrain row."""
    keys = [k for k in REPORT_KEYS if all(k in r["acc"] for r in rows.values())]
    extra = [k for k in next(iter(rows.values()))["acc"] if k not in REPORT_KEYS]
    cols = keys + [k for k in extra if all(k in r["acc"] for r in rows.values())]
    w = max(16, max(len(n) for n in rows) + 2)

    print("\nAccuracy (%) against the training super-class label")
    print(f"{'Model':{w}s}" + "".join(f"{c:>11s}" for c in cols))
    print("-" * (w + 11 * len(cols)))
    for name, r in rows.items():
        print(f"{name:{w}s}" + "".join(f"{r['acc'][c]:11.2f}" for c in cols))

    print("\nMIA efficacy (fraction called non-member)")
    print(f"{'Model':{w}s}{'Df':>9s}{'Dadj':>9s}{'gap':>9s}{'AUC(f,adj)':>12s}{'AUC(f,unseen)':>15s}")
    print("-" * (w + 54))
    for name, r in rows.items():
        m = r["mia"]
        print(f"{name:{w}s}{m['Df']:9.3f}{m['Dadj']:9.3f}{m['Df'] - m['Dadj']:9.3f}"
              f"{m['auc_adj']:12.3f}{m['auc_unseen']:15.3f}")

    deviations = {}
    if reference_name and reference_name in rows:
        g = rows[reference_name]
        print(f"\nAbsolute deviation from {reference_name}, lower is better "
              f"(L1 over {', '.join(cols)})")
        print(f"{'Model':{w}s}{'|dDf|':>8s}{'|dAdj|':>8s}{'|dFtest|':>10s}{'|dMIA_Df|':>11s}{'L1(acc)':>9s}")
        print("-" * (w + 46))
        for name, r in rows.items():
            if name == reference_name:
                continue
            d = {k: abs(r["acc"][k] - g["acc"][k]) for k in cols}
            dm = abs(r["mia"]["Df"] - g["mia"]["Df"])
            deviations[name] = {"acc": d, "L1": sum(d.values()), "mia_Df": dm}
            print(f"{name:{w}s}{d.get('Df(tr)', float('nan')):8.2f}"
                  f"{d.get('Dadj(tr)', float('nan')):8.2f}{d.get('Ftest(te)', float('nan')):10.2f}"
                  f"{dm:11.3f}{sum(d.values()):9.2f}")

    if runtimes:
        print(f"\n{'Runtime':28s}{'seconds':>10s}")
        print("-" * 38)
        for k, v in runtimes.items():
            print(f"{k:28s}{v:10.1f}")
    return deviations
