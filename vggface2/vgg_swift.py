"""Swift on VGGFace2: port of cifar100/swift_cifar100.ipynb (CIFAR-100) to the
GPU-resident face data.

    min_theta  W2^2( P_l(D_f; theta), sg[P_l(D_cal; theta)] )
               + lambda * [ W2^2( P_l(D_r; theta), sg[P_l(D_r^probe; theta_probe)] ) ]_+

Kept from the notebook: FitCheap probe (reset the classifier, train it plus the last
stage `layer4` on D_r^probe only, BN frozen), pooled calibration reference refreshed
every `cal_refresh` steps, lambda starting at lambda_r with dual ascent against
eps_r = eps_slack * bootstrap sampling floor, one-sided retain term, model in eval
mode, AdamW + warmup + cosine, gradient clipping, and checkpoint selection on the
objective over held-out slices of D_f, D_cal and D_r^probe (never the retrained model).
Hyperparameter defaults are the notebook's.

D_cal = val/cal (high-quality val images of the forget subjects, never trained on).
D_r^probe = subsample of D_r balanced over (adjacent/remote, identity), so every forget
subject's low-quality images get a quota.
"""
import copy
import math
import time

import numpy as np

PIPELINE_VERSION = "2026-09-25.4"   # must match the other pipeline files
import torch
import torch.nn as nn
import torch.nn.functional as F

import vgg_common as vc

DEFAULTS = dict(
    lr=1e-4, weight_decay=0.01, warmup_ratio=0.10, clip_grad=True, max_norm=5.0,
    batch_retain=128, batch_forget=64, epochs=1,
    lambda_r=1.0, lambda_mode="adaptive", retain_mode="one_sided", eps_retain=-1.0,
    eps_slack=1.5, rho0=2.0, beta=1.2, lambda_max=100.0, check_every=100,
    n_quantiles=256, cal_pool=256, cal_refresh=10, cal_ema=0.0,
    probe_size=5000, probe_steps=300, probe_lr=1e-3, probe_wd=0.01, probe_bs=128,
    probe_seed=4242, probe_reinit="head_layer4",
    select_best=True, select_n=512, probe_every=50, log_every=50,
    literal_objective=False,
    # checkpoint selection: "constrained" (Eq. 6: retain constraint W2_r <= eps on held-out
    # D_r^probe slices must hold; among those, lowest held-out W2_f; theta_0 itself is a
    # candidate) or "penalty" (notebook: lowest W2_f + lambda_r * W2_r)
    select_rule="constrained",
    # retain reference: "fitcheap" (paper, FitCheap probe) or "original" (theta_0's own
    # losses on D_r^probe, no probe fit)
    probe_mode="fitcheap",
    # optimizer for the unlearning step: "adamw" (notebook) or "sgd" (momentum, no decay)
    optimizer="adamw", momentum=0.9,
    # warn when tracked remote accuracy falls below this (percent)
    collapse_warn=50.0,
    # ---- ablation switches (defaults = Swift)
    forget_distance="w2",     # "w2" | "mean": (mean l_f - mean l_cal)^2 instead of W2^2
    retain_distance="w2",     # "w2" | "mean": [mean l_r - mean ref]_+^2 (two-sided: squared) instead of W2^2
    cal_stopgrad=True,        # False: D_cal losses carry gradient (tests the stop-gradient of Eq. 4)
    cal_fraction=1.0,         # use this fraction of D_cal (seeded subsample)
)


# --------------------------------------------------------------------------- 1-D W2
def midpoint_grid(n, device, dtype=torch.float32):
    return (torch.arange(n, device=device, dtype=dtype) + 0.5) / n


def reference_quantiles(samples, n_quantiles, device):
    x = samples.detach().float().to(device).flatten()
    return torch.quantile(x, midpoint_grid(n_quantiles, device, x.dtype)).contiguous()


def w2_sq_to_reference(losses, ref_q, one_sided=False):
    x = losses.float().flatten()
    n = x.numel()
    xs, _ = torch.sort(x)
    ref = torch.quantile(ref_q, midpoint_grid(n, x.device, x.dtype)) if ref_q.numel() != n else ref_q
    diff = xs - ref
    if one_sided:
        diff = torch.relu(diff)
    return torch.mean(diff ** 2)


def w2_sq_between_grad(losses_a, losses_b):
    """W2^2 between two loss samples with gradient through BOTH (no stop-gradient)."""
    a = losses_a.float().flatten()
    b = losses_b.float().flatten()
    grid = midpoint_grid(a.numel(), a.device, a.dtype)
    return torch.mean((torch.sort(a)[0] - torch.quantile(b, grid)) ** 2)


def mean_sq(losses, ref, one_sided=False):
    """Mean-matching counterpart of W2^2: (mean(losses) - mean(ref))^2, optionally one-sided."""
    d = losses.float().mean() - ref.float().mean()
    return torch.relu(d) ** 2 if one_sided else d ** 2


def w2_sq_between(losses_a, losses_b, one_sided=False):
    b = losses_b.detach().float().flatten()
    a = losses_a.float().flatten()
    n = a.numel()
    xs, _ = torch.sort(a)
    ref = torch.sort(b)[0] if b.numel() == n else torch.quantile(b, midpoint_grid(n, a.device, a.dtype))
    diff = xs - ref
    if one_sided:
        diff = torch.relu(diff)
    return torch.mean(diff ** 2)


@torch.no_grad()
def estimate_w2_floor(ref_samples, batch_size, n_quantiles, one_sided=False, n_boot=200, seed=0, q=0.5):
    x = ref_samples.detach().float().flatten()
    ref_q = reference_quantiles(x, n_quantiles, x.device)
    g = torch.Generator(device="cpu").manual_seed(seed)
    vals = [w2_sq_to_reference(x[torch.randint(0, x.numel(), (batch_size,), generator=g).to(x.device)],
                               ref_q, one_sided).item() for _ in range(n_boot)]
    return float(np.quantile(vals, q))


@torch.no_grad()
def estimate_floor(ref_samples, batch_size, n_quantiles, kind="w2", one_sided=False, n_boot=200, seed=0, q=0.5):
    """Sampling floor of the retain distance actually used ('w2' or 'mean')."""
    if kind == "w2":
        return estimate_w2_floor(ref_samples, batch_size, n_quantiles, one_sided, n_boot, seed, q)
    x = ref_samples.detach().float().flatten()
    g = torch.Generator(device="cpu").manual_seed(seed)
    # two-sided on purpose: a one-sided mean difference is exactly 0 for ~half the batches,
    # so its median floor would be 0 and any excess would count as a violation
    vals = [mean_sq(x[torch.randint(0, x.numel(), (batch_size,), generator=g).to(x.device)], x, False).item()
            for _ in range(n_boot)]
    return float(np.quantile(vals, q))


def w2_numpy(a, b, n_quantiles=256):
    qs = (np.arange(n_quantiles) + 0.5) / n_quantiles
    return float(np.mean((np.quantile(a, qs) - np.quantile(b, qs)) ** 2))


def ks_numpy(a, b):
    grid = np.concatenate([a, b])
    fa = np.searchsorted(np.sort(a), grid, side="right") / len(a)
    fb = np.searchsorted(np.sort(b), grid, side="right") / len(b)
    return float(np.max(np.abs(fa - fb)))


# --------------------------------------------------------------------------- sets
def probe_indices(sets, meta, size, seed):
    """D_r^probe: equal quota per (adjacent/remote, identity) stratum."""
    adj = np.asarray(sets["sets"]["train"]["adjacent"])
    rem = np.asarray(sets["sets"]["train"]["remote"])
    ident = meta.set_index("index")["identity"]
    strata = {}
    for tag, idx in (("adj", adj), ("rem", rem)):
        for key, grp in ident.loc[idx].groupby(ident.loc[idx].values):
            strata[(tag, key)] = grp.index.to_numpy()
    per = max(1, int(round(size / len(strata))))
    rng = np.random.default_rng(seed)
    picked = [rng.choice(v, min(per, len(v)), replace=False) for _, v in sorted(strata.items())]
    return np.sort(np.concatenate(picked)), int(np.isin(np.concatenate(picked), adj).sum())


def cycle_batches(idx, batch_size, generator, drop_last=True):
    while True:
        yielded = False
        for b in vc.batches(idx, batch_size, shuffle=True, drop_last=drop_last and len(idx) > batch_size,
                            generator=generator):
            yielded = True
            yield b
        if not yielded:
            raise RuntimeError("empty index set")


# --------------------------------------------------------------------------- FitCheap
def _last_linear(model):
    name, mod = None, None
    for n, m in model.named_modules():
        if isinstance(m, nn.Linear):
            name, mod = n, m
    return name, mod


def fit_cheap_probe(base_model, data, probe_idx, steps, lr, weight_decay, batch_size,
                    reinit="head_layer4", seed=0, max_norm=5.0, log_every=50):
    """A few hundred steps on D_r^probe only. Resets the classifier (and trains the
    last stage too with head_layer4); BatchNorm statistics stay frozen (eval mode)."""
    device = data.device
    probe = copy.deepcopy(base_model).to(device)
    head_name, head = _last_linear(probe)
    if reinit in ("head", "head_layer4"):
        head.reset_parameters()
    for p in probe.parameters():
        p.requires_grad_(False)
    trainable = list(head.parameters())
    if reinit == "head_layer4":
        last_stage = getattr(probe, "layer4", None)
        if last_stage is None:
            last_stage = getattr(probe, "block8", None)        # InceptionResnetV1
        if last_stage is not None:
            trainable += list(last_stage.parameters())
    for p in trainable:
        p.requires_grad_(True)
    probe.eval()
    opt = torch.optim.AdamW(trainable, lr=lr, weight_decay=weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(steps, 1), eta_min=lr * 1e-2)
    gen = torch.Generator(device=device).manual_seed(seed)
    stream = cycle_batches(probe_idx, batch_size, gen)
    print(f"[probe] head='{head_name}' trainable={sum(p.numel() for p in trainable) / 1e6:.2f}M "
          f"steps={steps} reinit={reinit}")
    for s in range(steps):
        b = next(stream)
        with vc.autocast(device):
            logits = probe(vc.to_input(data.x[b]))
        loss = F.cross_entropy(logits.float(), data.y[b])
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(trainable, max_norm)
        opt.step()
        sched.step()
        if s % log_every == 0 or s == steps - 1:
            print(f"  [probe {s:4d}] ce={loss.item():.4f}", flush=True)
    for p in probe.parameters():
        p.requires_grad_(False)
    return probe.eval()


# --------------------------------------------------------------------------- Swift
def run_swift(model, data, sets, meta, seed=2025, **hp):
    """Returns (unlearned model, info dict with timings and diagnostics)."""
    cfg = dict(DEFAULTS)
    unknown = set(hp) - set(cfg)
    if unknown:
        raise ValueError(f"unknown Swift hyperparameters {sorted(unknown)}")
    cfg.update(hp)
    if cfg["literal_objective"]:
        cfg.update(lambda_mode="fixed", retain_mode="two_sided", cal_refresh=1,
                   cal_pool=cfg["batch_forget"], cal_ema=0.0, select_best=False)
        print("[swift] literal_objective: all refinements disabled")
    device = data.device
    one_sided = cfg["retain_mode"] == "one_sided"
    S = sets["sets"]
    idx = lambda a: data.idx(a)  # noqa: E731
    cal_all = np.asarray(S["val"]["cal"])
    if cfg["cal_fraction"] < 1.0:
        keep = np.random.default_rng(seed + 7).choice(
            cal_all, max(2, int(round(cfg["cal_fraction"] * len(cal_all)))), replace=False)
        cal_all = np.sort(keep)
    f_idx, cal_idx = idx(S["train"]["forget"]), idx(cal_all)
    retain_idx = idx(S["train"]["retain"])
    p_np, n_adj_probe = probe_indices(sets, meta, cfg["probe_size"], cfg["probe_seed"])
    p_idx = idx(p_np)
    rng = np.random.default_rng(seed)
    sel_f = idx(rng.choice(S["train"]["forget"], min(cfg["select_n"], len(S["train"]["forget"])), replace=False))
    sel_r = idx(rng.choice(p_np, min(cfg["select_n"], len(p_np)), replace=False))
    trk = {"Df": sel_f, "Dadj": idx(rng.choice(S["train"]["adjacent"],
                                                min(cfg["select_n"], len(S["train"]["adjacent"])), replace=False)),
           "Ftest": idx(S["test"]["ftest"]),
           "Drem": idx(rng.choice(S["train"]["remote"], min(cfg["select_n"], len(S["train"]["remote"])),
                                  replace=False))}
    print(f"[swift] Df={len(f_idx)} D_cal={len(cal_idx)} D_r={len(retain_idx):,} "
          f"D_r^probe={len(p_idx)} (adjacent inside: {n_adj_probe})")
    timings = {}

    # ---- retain reference: FitCheap probe or theta_0 itself
    t0 = time.perf_counter()
    t0_losses, t0_correct = vc.per_sample(model, data, p_idx)
    if cfg["probe_mode"] == "original":
        ref_losses, ref_correct = t0_losses, t0_correct
        timings["probe_fit"] = 0.0
        print(f"[probe] probe_mode=original: retain reference = theta_0 losses on D_r^probe "
              f"(acc {100 * ref_correct.mean():.2f}, mean loss {ref_losses.mean():.4f})")
    elif cfg["probe_mode"] == "fitcheap":
        probe = fit_cheap_probe(model, data, p_idx, cfg["probe_steps"], cfg["probe_lr"], cfg["probe_wd"],
                                cfg["probe_bs"], cfg["probe_reinit"], seed=cfg["probe_seed"],
                                max_norm=cfg["max_norm"])
        vc.sync(device)
        timings["probe_fit"] = time.perf_counter() - t0
        ref_losses, ref_correct = vc.per_sample(probe, data, p_idx)
        print(f"[probe] fit {timings['probe_fit']:.1f}s | probe on D_r^probe: acc={100 * ref_correct.mean():.2f} "
              f"loss={ref_losses.mean():.4f} | theta_0: acc={100 * t0_correct.mean():.2f} "
              f"loss={t0_losses.mean():.4f}")
        if ref_losses.mean() > 10 * max(t0_losses.mean(), 1e-3):
            print("[probe] WARNING: probe loss is >10x theta_0's, so the one-sided retain term barely "
                  "constrains; consider probe_mode=original or a larger probe_size/probe_steps")
        del probe
    else:
        raise ValueError(f"probe_mode must be 'fitcheap' or 'original', got {cfg['probe_mode']!r}")
    probe_ref = torch.as_tensor(ref_losses, device=device)
    probe_ref_q = reference_quantiles(probe_ref, cfg["n_quantiles"], device)

    # ---- floors / eps
    floor_r = estimate_floor(probe_ref, cfg["batch_retain"], cfg["n_quantiles"], cfg["retain_distance"], one_sided, seed=seed)
    cal0, _ = vc.per_sample(model, data, cal_idx)
    floor_f = estimate_w2_floor(torch.as_tensor(cal0, device=device), cfg["batch_forget"],
                                cfg["n_quantiles"], seed=seed)
    eps_retain = cfg["eps_slack"] * floor_r if cfg["eps_retain"] < 0 else cfg["eps_retain"]
    # selection tolerance: 95th percentile of the sampling noise for a slice of this size
    # (finalised below so that theta_0 itself is always feasible)
    floor_sel = estimate_floor(probe_ref, len(sel_r), cfg["n_quantiles"], cfg["retain_distance"], one_sided,
                               seed=seed + 1, q=0.95)
    eps_select = cfg["eps_slack"] * floor_sel
    print(f"[floor] retain(batch={cfg['batch_retain']})={floor_r:.4g} forget(batch={cfg['batch_forget']})="
          f"{floor_f:.4g} eps_retain={eps_retain:.4g} | selection slice (n={len(sel_r)}) "
          f"eps={eps_select:.4g}, rule={cfg['select_rule']}")

    # ---- optimisation
    t0 = time.perf_counter()
    m = copy.deepcopy(model).to(device)
    m.eval()                                  # BN statistics frozen, as in the notebook
    for p in m.parameters():
        p.requires_grad_(True)
    if cfg["optimizer"] == "adamw":
        opt = torch.optim.AdamW(m.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    elif cfg["optimizer"] == "sgd":
        opt = torch.optim.SGD(m.parameters(), lr=cfg["lr"], momentum=cfg["momentum"],
                              weight_decay=cfg["weight_decay"])
    else:
        raise ValueError(f"optimizer must be 'adamw' or 'sgd', got {cfg['optimizer']!r}")
    steps_per_epoch = len(retain_idx) // cfg["batch_retain"]
    total = cfg["epochs"] * steps_per_epoch
    warm = max(int(total * cfg["warmup_ratio"]), 1)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: 0.01 + 0.99 * s / warm if s < warm
        else (1e-7 / cfg["lr"]) + (1 - 1e-7 / cfg["lr"]) * 0.5 * (1 + math.cos(math.pi * (s - warm) / max(1, total - warm))))
    lam = torch.tensor(float(cfg["lambda_r"]), device=device)
    lam_cap = torch.tensor(float(cfg["lambda_max"]), device=device)
    rho = cfg["rho0"]
    gen = torch.Generator(device=device).manual_seed(seed)
    f_stream = cycle_batches(f_idx, cfg["batch_forget"], gen)
    c_stream = cycle_batches(cal_idx, cfg["batch_forget"], gen, drop_last=False)
    diag = {k: [] for k in ("step", "loss", "W2_f", "W2_r", "mean_l_f", "mean_l_cal", "mean_l_r", "lambda", "rho", "lr")}
    plog = {"step": [], "Df": [], "Dadj": [], "Ftest": [], "Drem": [], "score": [], "feasible": []}
    best = {"key": None, "step": None, "state": None}
    cal_ref = None

    @torch.no_grad()
    def refresh_cal(prev):
        pool, got = [], 0
        while got < cfg["cal_pool"]:
            b = next(c_stream)
            with vc.autocast(device):
                lg = m(vc.to_input(data.x[b]))
            lb = F.cross_entropy(lg.float(), data.y[b], reduction="none")
            pool.append(lb)
            got += lb.numel()
        q = reference_quantiles(torch.cat(pool), cfg["n_quantiles"], device)
        if prev is not None and cfg["cal_ema"] > 0:
            q = cfg["cal_ema"] * prev + (1 - cfg["cal_ema"]) * q
        return q

    def selection_score():
        l_f = torch.as_tensor(vc.per_sample(m, data, sel_f)[0], device=device)
        l_c = torch.as_tensor(vc.per_sample(m, data, cal_idx)[0], device=device)
        l_r = torch.as_tensor(vc.per_sample(m, data, sel_r)[0], device=device)
        w_f = (mean_sq(l_f, l_c) if cfg["forget_distance"] == "mean" else w2_sq_between(l_f, l_c)).item()
        w_r = (mean_sq(l_r, probe_ref, one_sided) if cfg["retain_distance"] == "mean"
               else w2_sq_to_reference(l_r, probe_ref_q, one_sided=one_sided)).item()
        return w_f + float(cfg["lambda_r"]) * w_r, w_f, w_r

    def consider(step):
        """Selection candidate. constrained: feasible (W2_r <= eps_select) first, ranked by
        W2_f; infeasible only as a fallback, ranked by W2_r. penalty: notebook score."""
        nonlocal best
        sc, w_f, w_r = selection_score()
        feasible = w_r <= eps_select
        key = (sc,) if cfg["select_rule"] == "penalty" else ((0, w_f) if feasible else (1, w_r))
        plog["score"].append(sc)
        plog["feasible"].append(bool(feasible))
        if best["key"] is None or key < best["key"]:
            best = {"key": key, "step": step, "score": sc, "w_f": w_f, "w_r": w_r, "feasible": bool(feasible),
                    "state": {k: v.detach().to("cpu", copy=True) for k, v in m.state_dict().items()}}
            print(f"  [select] new best @step {step}: W2_f={w_f:.4f} W2_r={w_r:.4g} "
                  f"({'feasible' if feasible else 'INFEASIBLE, eps=' + format(eps_select, '.3g')})")
        m.eval()

    if cfg["select_best"]:
        _, _, w_r_theta0 = selection_score()
        eps_select = cfg["eps_slack"] * max(floor_sel, w_r_theta0)   # theta_0 must pass its own constraint
        print(f"[select] rule={cfg['select_rule']} eps={eps_select:.4g} (noise q95={floor_sel:.4g}, "
              f"theta_0 on slice={w_r_theta0:.4g})")
        consider(-1)          # theta_0 itself: returned if no later checkpoint beats it

    print(f"[swift] {total} steps (retain batch {cfg['batch_retain']}, forget batch "
          f"{cfg['batch_forget']}), {cfg['optimizer']} lr {cfg['lr']}, warmup {warm}, "
          f"probe_mode={cfg['probe_mode']}")
    step = 0
    for epoch in range(cfg["epochs"]):
        for b_r in vc.batches(retain_idx, cfg["batch_retain"], shuffle=True, drop_last=True, generator=gen):
            b_f = next(f_stream)
            if cal_ref is None or step % cfg["cal_refresh"] == 0:
                cal_ref = refresh_cal(cal_ref)
            with vc.autocast(device):
                lr_logits = m(vc.to_input(data.x[b_r]))
                lf_logits = m(vc.to_input(data.x[b_f]))
            l_r = F.cross_entropy(lr_logits.float(), data.y[b_r], reduction="none")
            l_f = F.cross_entropy(lf_logits.float(), data.y[b_f], reduction="none")
            if cfg["cal_stopgrad"]:
                W2_f = (mean_sq(l_f, cal_ref) if cfg["forget_distance"] == "mean"
                        else w2_sq_to_reference(l_f, cal_ref))                 # fp32, cal detached
            else:                                   # ablation: gradient flows through D_cal too
                b_c = next(c_stream)
                with vc.autocast(device):
                    lc_logits = m(vc.to_input(data.x[b_c]))
                l_c = F.cross_entropy(lc_logits.float(), data.y[b_c], reduction="none")
                W2_f = (mean_sq(l_f, l_c) if cfg["forget_distance"] == "mean"
                        else w2_sq_between_grad(l_f, l_c))
            W2_r = (mean_sq(l_r, probe_ref, one_sided) if cfg["retain_distance"] == "mean"
                    else w2_sq_to_reference(l_r, probe_ref_q, one_sided=one_sided))
            loss = W2_f + lam * W2_r
            opt.zero_grad(set_to_none=True)
            loss.backward()
            if cfg["clip_grad"]:
                torch.nn.utils.clip_grad_norm_(m.parameters(), cfg["max_norm"])
            opt.step()
            sched.step()

            if cfg["lambda_mode"] == "adaptive":
                with torch.no_grad():
                    lam = torch.minimum(lam + rho * torch.relu(W2_r.detach() - eps_retain), lam_cap)
                if step > 0 and step % cfg["check_every"] == 0 and W2_r.item() > eps_retain:
                    rho *= cfg["beta"]
                    print(f"  retain constraint violated (W2_r={W2_r.item():.4g} > {eps_retain:.4g}); rho -> {rho:.2f}")

            for k, v in (("step", step), ("loss", loss.item()), ("W2_f", W2_f.item()), ("W2_r", W2_r.item()),
                         ("mean_l_f", l_f.mean().item()), ("mean_l_cal", cal_ref.mean().item()),
                         ("mean_l_r", l_r.mean().item()), ("lambda", float(lam.item())), ("rho", rho),
                         ("lr", opt.param_groups[0]["lr"])):
                diag[k].append(v)
            if not math.isfinite(diag["loss"][-1]):
                raise RuntimeError(f"non-finite loss at step {step}; lower lr")

            if step % cfg["probe_every"] == 0 or step == total - 1:
                plog["step"].append(step)
                for key in ("Df", "Dadj", "Ftest", "Drem"):
                    plog[key].append(vc.acc_of(m, data, trk[key]))
                if plog["Drem"][-1] < cfg["collapse_warn"]:
                    print(f"  [warn] step {step}: accuracy on a remote slice is {plog['Drem'][-1]:.1f}% "
                          f"(theta_0 ~100%): the model is collapsing; lower lr or use optimizer=sgd")
                if cfg["select_best"]:
                    consider(step)
                else:
                    plog["score"].append(float("nan"))
                    plog["feasible"].append(None)
                m.eval()
            if step % cfg["log_every"] == 0:
                print(f"[step {step}/{total}] loss={loss.item():.4f} W2_f={W2_f.item():.4f} W2_r={W2_r.item():.4g} "
                      f"l_f={l_f.mean().item():.3f} l_cal={cal_ref.mean().item():.3f} l_r={l_r.mean().item():.3f} "
                      f"lam={lam.item():.2f} | acc Df={plog['Df'][-1]:.1f} Dadj={plog['Dadj'][-1]:.1f} "
                      f"Ftest={plog['Ftest'][-1]:.1f} Drem={plog['Drem'][-1]:.1f}", flush=True)
            step += 1
    vc.sync(device)
    timings["optimisation"] = time.perf_counter() - t0
    if cfg["select_best"] and best["state"] is not None:
        where = "theta_0 (no checkpoint beat the unchanged model)" if best["step"] == -1 else f"step {best['step']}"
        print(f"[select] restoring {where}: W2_f={best['w_f']:.4f} W2_r={best['w_r']:.4g} "
              f"feasible={best['feasible']}")
        if not best["feasible"]:
            print("[select] WARNING: no checkpoint satisfied the retain constraint; returned the least-violating one")
        m.load_state_dict(best["state"])
    m.eval()

    # distributional gaps (diagnostic)
    lf1, _ = vc.per_sample(m, data, f_idx)
    lcal1, _ = vc.per_sample(m, data, cal_idx)
    lft1, _ = vc.per_sample(m, data, trk["Ftest"])
    gaps = {"W2(Df,Dcal)": w2_numpy(lf1, lcal1), "W2(Df,Ftest)": w2_numpy(lf1, lft1),
            "KS(Df,Ftest)": ks_numpy(lf1, lft1)}
    info = {"config": cfg, "timings": timings, "eps_retain": eps_retain, "floor_r": floor_r,
            "floor_f": floor_f, "probe_size": int(len(p_idx)), "adjacent_in_probe": n_adj_probe,
            "eps_select": eps_select,
            "best_select": {k: v for k, v in best.items() if k not in ("state", "key")}, "gaps": gaps,
            "diag": diag, "track": plog}
    return m, info
