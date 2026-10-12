#!/usr/bin/env python
"""Swift on CIFAR-100 / ResNet-18 (20 super-classes).

    min_theta  W2^2( P_l(D_f; theta), sg[P_l(D_cal; theta)] )
               + lambda * [ W2^2( P_l(D_r; theta), sg[P_l(D_r^probe; theta_probe)] ) ]_+

The forget term pulls D_f's loss distribution onto that of held-out images of
the same class, which is what a retrained model sees them as, instead of onto a
hand-picked target such as uniform logits or maximal loss. The retain term is a
one-sided Wasserstein constraint against a cheap retain-only probe
(theta_r^probe, FitCheap), so utility is anchored without the retrained model.

lambda follows dual ascent against eps_r = eps_slack * the bootstrap sampling
floor of the retain term, and the returned checkpoint is chosen by the
constrained rule (Eq. 6) on held-out slices of D_f, D_cal and D_r^probe, with
theta_0 as a candidate. The retrained model is never consulted.

BatchNorm running statistics stay frozen throughout (the model is kept in eval
mode): a forget batch drawn from a single class otherwise overwrites them,
which degrades D_r^rem by a mechanism no gradient-level constraint can see.

This is the reference implementation of the method; the text and face folders
are ports of it. swift_cifar100.ipynb is the exploratory notebook it grew from.

Prerequisites: pretrain_original.py for the Original model, and
retrain_reference.py --split <split> for the deviations from the counterfactual.

Examples
--------
    python swift_cifar100.py --split instance
    python swift_cifar100.py --split class --forget-class 4 --plot-dir figures_class
"""
import argparse
import copy
import hashlib
import json
import os
import time
from dataclasses import asdict

import numpy as np
import torch
import torch.nn.functional as F
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from torch.utils.data import DataLoader, Subset

import swift_common as sc
from cifar100 import N_SUPERCLASSES, SuperClassDataset, fine_name, make_loader
from retrain_reference import eval_loaders_for
from swift_sets import add_sets_args, build_sets, describe, sets_config_from_args

PIPELINE_VERSION = "2026-10-12.1"   # must match swift_common.py


def check_versions():
    found = {"swift_cifar100.py": PIPELINE_VERSION,
             "swift_common.py": getattr(sc, "PIPELINE_VERSION", "old")}
    if len(set(found.values())) != 1:
        raise SystemExit(f"[error] files from different versions: {found}. "
                         "Check out the cifar100/ folder again.")


def parse_args():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0],
                                formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    add_sets_args(p)

    g = p.add_argument_group("paths")
    g.add_argument("--checkpoint", default="checkpoints/cifar100_resnet18.pt",
                   help="Original model from pretrain_original.py")
    g.add_argument("--output", help="default checkpoints/swift_<split>_c<class>_seed<seed>.pt")
    g.add_argument("--results-json",
                   help="default results/swift_<split>_c<class>_seed<seed>.json")
    g.add_argument("--reference-json", help="retrain reference from retrain_reference.py "
                                            "(default results/swift_reference_<split>_c<class>.json)")
    g.add_argument("--probe-cache", help="default checkpoints/swift_probe_<split>_c<class>.pt")
    g.add_argument("--refit-probe", action="store_true")
    g.add_argument("--plot-dir", help="save diagnostic plots here (needs matplotlib)")
    g.add_argument("--no-save-model", action="store_true")

    g = p.add_argument_group("runtime")
    g.add_argument("--seed", type=int, default=2025, help="unlearning seed")
    g.add_argument("--device", default="cuda:0")
    g.add_argument("--amp", choices=["bf16", "fp16", "none"], default="fp16")
    g.add_argument("--tf32", action="store_true", help="TF32 matmul (H100)")
    g.add_argument("--epochs", type=int, default=1, help="passes over D_r")
    g.add_argument("--max-steps", type=int, help="cap on optimisation steps")
    g.add_argument("--num-workers", type=int, default=8)
    g.add_argument("--eval-batch", type=int, default=256)
    g.add_argument("--skip-original-eval", action="store_true")

    g = p.add_argument_group("optimisation")
    g.add_argument("--lr", type=float, default=1e-4)
    g.add_argument("--weight-decay", type=float, default=0.01)
    g.add_argument("--warmup-ratio", type=float, default=0.10)
    g.add_argument("--no-clip", action="store_true")
    g.add_argument("--max-norm", type=float, default=5.0)
    g.add_argument("--batch-retain", type=int, default=128)
    g.add_argument("--batch-forget", type=int, default=64)

    g = p.add_argument_group("objective")
    g.add_argument("--lambda-r", type=float, default=1.0, help="initial lambda")
    g.add_argument("--lambda-mode", choices=["adaptive", "fixed"], default="adaptive")
    g.add_argument("--retain-mode", choices=["one_sided", "two_sided"], default="one_sided")
    g.add_argument("--eps-retain", type=float, default=-1.0,
                   help="<0: eps_slack * sampling floor")
    g.add_argument("--eps-slack", type=float, default=1.5)
    g.add_argument("--rho0", type=float, default=2.0)
    g.add_argument("--beta", type=float, default=1.2)
    g.add_argument("--lambda-max", type=float, default=100.0)
    g.add_argument("--check-every", type=int, default=100)
    g.add_argument("--n-quantiles", type=int, default=256)
    g.add_argument("--cal-pool", type=int, default=256)
    g.add_argument("--cal-refresh", type=int, default=10)
    g.add_argument("--cal-ema", type=float, default=0.0)

    g = p.add_argument_group("FitCheap probe")
    g.add_argument("--probe-steps", type=int, default=300)
    g.add_argument("--probe-lr", type=float, default=1e-3)
    g.add_argument("--probe-wd", type=float, default=0.01)
    g.add_argument("--probe-bs", type=int, default=128)
    g.add_argument("--probe-reinit", choices=["head", "head_layer4", "none"],
                   default="head_layer4")
    g.add_argument("--probe-bn", choices=["eval", "train"], default="eval",
                   help="'eval' keeps theta_0's BatchNorm statistics")

    g = p.add_argument_group("selection / logging / MIA")
    g.add_argument("--no-select-best", action="store_true")
    g.add_argument("--select-rule", choices=["constrained", "penalty"], default="constrained",
                   help="constrained (Eq. 6): a checkpoint must keep W2_r within tolerance "
                        "on the held-out slice and is then ranked by W2_f; penalty: the "
                        "weighted objective")
    g.add_argument("--select-lambda", choices=["initial", "current"], default="initial",
                   help="lambda in the selection score")
    g.add_argument("--select-every", type=int, default=50)
    g.add_argument("--log-every", type=int, default=25)
    g.add_argument("--mia-seed", type=int, default=7)
    g.add_argument("--mia-n-members", type=int, default=2000)

    args = p.parse_args()
    args.select_best = not args.no_select_best
    args.clip_grad = not args.no_clip
    return args


def probe_cache_key(args, cfg, probe_idx):
    """A cached probe is only reusable if it was fitted on the same subset with
    the same recipe."""
    payload = {"indices": hashlib.sha1(probe_idx.tobytes()).hexdigest(),
               "steps": args.probe_steps, "lr": args.probe_lr, "wd": args.probe_wd,
               "bs": args.probe_bs, "reinit": args.probe_reinit, "bn": args.probe_bn,
               "split": cfg.split, "forget_class": cfg.forget_class}
    return hashlib.sha1(json.dumps(payload, sort_keys=True).encode()).hexdigest()


def main():
    args = parse_args()
    check_versions()
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    amp_dtype, use_scaler = sc.resolve_amp(args.amp, device)
    if args.tf32:
        sc.enable_fast_math()
    one_sided = args.retain_mode == "one_sided"

    # ------------------------------------------------------------------ data
    cfg = sets_config_from_args(args)
    sets = build_sets(cfg, args.data_root)
    print(describe(cfg, sets))
    train_base, test_base = sets["train_base"], sets["test_base"]
    train_idx, test_idx = sets["train"], sets["test"]

    stem = f"{cfg.split}_c{cfg.forget_class}"
    results_json = args.results_json or f"results/swift_{stem}_seed{args.seed}.json"
    output = args.output or f"checkpoints/swift_{stem}_seed{args.seed}.pt"
    reference_json = args.reference_json or f"results/swift_reference_{stem}.json"
    probe_cache = args.probe_cache or f"checkpoints/swift_probe_{stem}.pt"
    if not os.path.exists(args.checkpoint):
        raise SystemExit(f"[error] Original checkpoint {args.checkpoint} not found; "
                         "run pretrain_original.py first")

    def loader(base, idx, batch_size, shuffle=False, drop_last=False):
        return make_loader(base, idx, batch_size, shuffle=shuffle,
                           num_workers=args.num_workers, drop_last=drop_last)

    eval_loaders = eval_loaders_for(sets, args.eval_batch, args.num_workers, cfg.split)
    cal_loader = loader(test_base, test_idx["cal"], args.batch_forget, shuffle=True)
    cal_eval_loader = loader(test_base, test_idx["cal"], args.eval_batch)
    probe_train_loader = loader(train_base, train_idx["probe"], args.probe_bs, shuffle=True)
    probe_eval_loader = loader(train_base, train_idx["probe"], args.eval_batch)
    probe_holdout_loader = loader(train_base, train_idx["probe_holdout"], args.eval_batch)
    select_loaders = {"Df": loader(train_base, train_idx["select_forget"], args.eval_batch),
                      "cal": cal_eval_loader,
                      "Rprobe": loader(train_base, train_idx["select_probe"], args.eval_batch)}
    track_loaders = {"Df": loader(train_base, train_idx["track_forget"], args.eval_batch),
                     "Dadj": loader(train_base, train_idx["track_adjacent"], args.eval_batch),
                     "Ftest": loader(test_base, test_idx["track_ftest"], args.eval_batch)}

    # drop_last keeps every W2 batch the same size, but only where the set can
    # spare a partial batch: a set smaller than one batch would leave the
    # loader empty and cycle() would have nothing to yield.
    retain_ds = Subset(SuperClassDataset(train_base), train_idx["retain"].tolist())
    retain_loader = DataLoader(retain_ds, batch_size=args.batch_retain, shuffle=True,
                               num_workers=args.num_workers, pin_memory=True,
                               drop_last=len(retain_ds) > args.batch_retain)
    forget_loader = loader(train_base, train_idx["forget"], args.batch_forget,
                           shuffle=True, drop_last=True)

    print(f"[swift] split={cfg.split} forget class={cfg.forget_class} "
          f"({fine_name(cfg.forget_class)}) device={device} amp={args.amp} seed={args.seed}")

    # ------------------------------------------------------------------ original
    model = sc.load_model(args.checkpoint, device, N_SUPERCLASSES)
    rows, timings = {}, {}
    if not args.skip_original_eval:
        t0 = time.perf_counter()
        rows["Original"] = sc.full_evaluation(model, eval_loaders, device, amp_dtype,
                                              args.mia_n_members, args.mia_seed,
                                              tag="Original")
        timings["original_eval"] = time.perf_counter() - t0

    # ------------------------------------------------------------------ FitCheap
    sc.set_seed(args.seed)
    t0 = time.perf_counter()
    key = probe_cache_key(args, cfg, train_idx["probe"])
    probe_model = None
    if probe_cache and os.path.exists(probe_cache) and not args.refit_probe:
        blob = torch.load(probe_cache, map_location=device, weights_only=False)
        if blob.get("key") == key:
            probe_model = copy.deepcopy(model)
            probe_model.load_state_dict(blob["model_state"])
            probe_model.to(device).eval()
            print(f"[probe] loaded cache <- {probe_cache}")
        else:
            print("[probe] cache was fitted with a different subset/config; refitting")
    if probe_model is None:
        probe_model = sc.fit_cheap_probe(
            model, probe_train_loader, device, steps=args.probe_steps, lr=args.probe_lr,
            weight_decay=args.probe_wd, reinit=args.probe_reinit, bn_mode=args.probe_bn,
            amp_dtype=amp_dtype, use_scaler=use_scaler, max_norm=args.max_norm)
        if probe_cache:
            os.makedirs(os.path.dirname(probe_cache) or ".", exist_ok=True)
            torch.save({"model_state": probe_model.state_dict(), "key": key}, probe_cache)
            print(f"[probe] cached -> {probe_cache}")
    timings["probe_fit"] = time.perf_counter() - t0

    probe_ref_losses = sc.per_sample_losses(probe_model, probe_eval_loader, device, amp_dtype)
    probe_ref_q = sc.reference_quantiles(probe_ref_losses, args.n_quantiles, device)
    probe_member = sc.acc_and_loss(probe_model, probe_eval_loader, device, amp_dtype)
    probe_hold = sc.acc_and_loss(probe_model, probe_holdout_loader, device, amp_dtype)
    theta0_on_probe = sc.acc_and_loss(model, probe_eval_loader, device, amp_dtype)
    # The hold-out is empty only in reduced configurations where D_r^probe
    # covers the whole retain set; it is a diagnostic, not part of the method.
    hold = (f"acc={probe_hold['acc']:.2f} loss={probe_hold['loss']:.4f}"
            if probe_hold["n"] else "n/a (D_r^probe covers all of D_r)")
    print(f"[probe] fit {timings['probe_fit']:.1f}s | on D_r^probe acc={probe_member['acc']:.2f} "
          f"loss={probe_member['loss']:.4f} | unseen retain {hold} | "
          f"theta_0 on D_r^probe acc={theta0_on_probe['acc']:.2f} "
          f"loss={theta0_on_probe['loss']:.4f}")
    if probe_member["loss"] > 3.0 * max(theta0_on_probe["loss"], 1e-6):
        print("WARNING: the probe is much weaker than theta_0 on D_r. With --retain-mode "
              "two_sided it will pull retain accuracy down; raise --probe-steps/--probe-size "
              "or keep one_sided.")
    del probe_model
    if device.type == "cuda":
        torch.cuda.empty_cache()

    # ------------------------------------------------------------------ floors / eps
    floor_r = sc.estimate_w2_floor(probe_ref_losses.to(device), args.batch_retain,
                                   args.n_quantiles, one_sided, seed=args.seed)
    cal_theta0 = sc.per_sample_losses(model, cal_eval_loader, device, amp_dtype)
    floor_f = sc.estimate_w2_floor(cal_theta0.to(device), args.batch_forget,
                                   args.n_quantiles, seed=args.seed)
    eps_retain = args.eps_slack * floor_r if args.eps_retain < 0 else args.eps_retain
    n_sel = len(select_loaders["Rprobe"].dataset)
    floor_sel = sc.estimate_w2_floor(probe_ref_losses.to(device), n_sel, args.n_quantiles,
                                     one_sided, seed=args.seed + 1, q=0.95)
    print(f"[floor] retain(batch={args.batch_retain})={floor_r:.4g} "
          f"forget(batch={args.batch_forget})={floor_f:.4g} eps_retain={eps_retain:.4g} "
          f"(W2_f below ~{floor_f:.4g} means the forget term has converged)")
    if eps_retain <= 0 and args.lambda_mode == "adaptive":
        print("WARNING: eps_retain is 0, so any retain excess keeps raising lambda (up to "
              "--lambda-max). Set --eps-retain explicitly if this is not intended.")

    # ------------------------------------------------------------------ optimisation
    sc.set_seed(args.seed)
    model_unlearn = copy.deepcopy(model)
    model_unlearn.eval()          # BN statistics frozen: unlearning must not rewrite them
    optimizer = torch.optim.AdamW(model_unlearn.parameters(), lr=args.lr,
                                  weight_decay=args.weight_decay)
    total_steps = args.epochs * len(retain_loader)
    if args.max_steps:
        total_steps = min(total_steps, args.max_steps)
    warmup_steps = max(int(total_steps * args.warmup_ratio), 1)
    scheduler = SequentialLR(optimizer, schedulers=[
        LinearLR(optimizer, start_factor=0.01, end_factor=1.0, total_iters=warmup_steps),
        CosineAnnealingLR(optimizer, T_max=max(total_steps - warmup_steps, 1), eta_min=1e-7)],
        milestones=[warmup_steps])
    scaler = sc.make_grad_scaler(use_scaler)

    lambda_ = torch.tensor(float(args.lambda_r), device=device)
    lam_cap = torch.tensor(float(args.lambda_max), device=device)
    rho = args.rho0
    cycled_forget, cycled_cal = sc.cycle(forget_loader), sc.cycle(cal_loader)
    diag = {k: [] for k in ("step", "loss", "W2_f", "W2_r", "mean_l_f", "mean_l_cal",
                            "mean_l_r", "lambda", "rho", "lr")}
    plog = {"step": [], "Df": [], "Dadj": [], "Ftest": [], "score": [], "feasible": []}
    best = {"key": None, "step": None, "state": None}
    cal_ref = None

    @torch.no_grad()
    def refresh_cal_reference(prev):
        """sg[P_l(D_cal; theta)] is stop-gradient by construction, so it can be
        pooled over more than one mini-batch at no graph cost. Pooling damps the
        feedback loop in which a model degrading on the whole class raises its
        own target."""
        pool, got = [], 0
        while got < args.cal_pool:
            images, labels = next(cycled_cal)
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True).long()
            with sc.autocast(amp_dtype):
                lb = F.cross_entropy(model_unlearn(images).float(), labels, reduction="none")
            pool.append(lb.float())
            got += lb.numel()
        q = sc.reference_quantiles(torch.cat(pool), args.n_quantiles, device)
        if prev is not None and args.cal_ema > 0.0:
            q = args.cal_ema * prev + (1.0 - args.cal_ema) * q
        return q

    @torch.no_grad()
    def selection_score():
        l_f = sc.per_sample_losses(model_unlearn, select_loaders["Df"], device, amp_dtype).to(device)
        l_c = sc.per_sample_losses(model_unlearn, select_loaders["cal"], device, amp_dtype).to(device)
        l_r = sc.per_sample_losses(model_unlearn, select_loaders["Rprobe"], device, amp_dtype).to(device)
        w_f = sc.w2_sq_between(l_f, l_c).item()
        w_r = sc.w2_sq_to_reference(l_r, probe_ref_q, one_sided=one_sided).item()
        lam = args.lambda_r if args.select_lambda == "initial" else float(lambda_.item())
        return w_f + lam * w_r, w_f, w_r

    eps_select = None

    def consider(step, record=True):
        """constrained: feasible checkpoints (W2_r <= eps_select) ranked by W2_f,
        infeasible ones only as a fallback ranked by W2_r; penalty: the weighted
        objective."""
        nonlocal best
        score, w_f, w_r = selection_score()
        feasible = w_r <= eps_select
        key_ = (score,) if args.select_rule == "penalty" else ((0, w_f) if feasible else (1, w_r))
        if record:
            plog["score"].append(score)
            plog["feasible"].append(bool(feasible))
        if best["key"] is None or key_ < best["key"]:
            best = {"key": key_, "step": step, "score": score, "w_f": w_f, "w_r": w_r,
                    "feasible": bool(feasible),
                    "state": {k: v.detach().to("cpu", copy=True)
                              for k, v in model_unlearn.state_dict().items()}}
            print(f"  [select] new best @step {step}: W2_f={w_f:.4f} W2_r={w_r:.4g} "
                  f"({'feasible' if feasible else 'INFEASIBLE, eps=' + format(eps_select, '.3g')})")
        model_unlearn.eval()

    if args.select_best:
        _, _, w_r_theta0 = selection_score()
        eps_select = args.eps_slack * max(floor_sel, w_r_theta0)   # theta_0 passes its own constraint
        print(f"[select] rule={args.select_rule} eps={eps_select:.4g} "
              f"(noise q95={floor_sel:.4g}, theta_0 on slice={w_r_theta0:.4g})")
        consider(-1, record=False)     # theta_0 is a candidate but not part of the step log

    print(f"[train] {total_steps} steps (retain batch {args.batch_retain}, forget batch "
          f"{args.batch_forget}), warmup {warmup_steps}")
    t0 = time.perf_counter()
    step, done = 0, False
    for epoch in range(args.epochs):
        for batch_id, (x_r, y_r) in enumerate(retain_loader):
            if step >= total_steps:
                done = True
                break
            x_r = x_r.to(device, non_blocking=True)
            y_r = y_r.to(device, non_blocking=True).long()
            x_f, y_f = next(cycled_forget)
            x_f = x_f.to(device, non_blocking=True)
            y_f = y_f.to(device, non_blocking=True).long()
            if cal_ref is None or step % args.cal_refresh == 0:
                cal_ref = refresh_cal_reference(cal_ref)

            optimizer.zero_grad(set_to_none=True)
            with sc.autocast(amp_dtype):
                l_r = F.cross_entropy(model_unlearn(x_r).float(), y_r, reduction="none")
                l_f = F.cross_entropy(model_unlearn(x_f).float(), y_f, reduction="none")
            # W2 terms in fp32: sorting/quantiles in half precision are badly conditioned
            W2_f = sc.w2_sq_to_reference(l_f, cal_ref)
            W2_r = sc.w2_sq_to_reference(l_r, probe_ref_q, one_sided=one_sided)
            loss = W2_f + lambda_ * W2_r

            scaler.scale(loss).backward()
            if args.clip_grad:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model_unlearn.parameters(), args.max_norm)
            scale_before = scaler.get_scale()
            scaler.step(optimizer)
            scaler.update()
            if not (use_scaler and scale_before > scaler.get_scale()):
                scheduler.step()

            if args.lambda_mode == "adaptive":
                with torch.no_grad():
                    violation = torch.relu(W2_r.detach() - eps_retain)
                    lambda_ = torch.minimum(lambda_ + rho * violation, lam_cap)
                if batch_id > 0 and batch_id % args.check_every == 0 and W2_r.item() > eps_retain:
                    rho *= args.beta
                    print(f"  retain constraint violated (W2_r={W2_r.item():.4f} > "
                          f"{eps_retain:.4f}); rho -> {rho:.2f}")

            for k, v in (("step", step), ("loss", loss.item()), ("W2_f", W2_f.item()),
                         ("W2_r", W2_r.item()), ("mean_l_f", l_f.mean().item()),
                         ("mean_l_cal", cal_ref.mean().item()),
                         ("mean_l_r", l_r.mean().item()),
                         ("lambda", float(lambda_.item())), ("rho", rho),
                         ("lr", optimizer.param_groups[0]["lr"])):
                diag[k].append(v)

            if step % args.select_every == 0 or step == total_steps - 1:
                plog["step"].append(step)
                for name, ldr in track_loaders.items():
                    plog[name].append(sc.acc_and_loss(model_unlearn, ldr, device,
                                                      amp_dtype)["acc"])
                if args.select_best:
                    consider(step)
                else:
                    plog["score"].append(float("nan"))
                    plog["feasible"].append(False)
                print(f"  [track @step {step}] Df={plog['Df'][-1]:.2f} "
                      f"Dadj={plog['Dadj'][-1]:.2f} Ftest={plog['Ftest'][-1]:.2f}", flush=True)

            if step % args.log_every == 0:
                elapsed = time.perf_counter() - t0
                eta = elapsed / max(step + 1, 1) * (total_steps - step - 1)
                print(f"[step {step}/{total_steps}] loss={loss.item():.4f} "
                      f"W2_f={W2_f.item():.4f} W2_r={W2_r.item():.4f} "
                      f"l_f={l_f.mean().item():.3f} l_cal={cal_ref.mean().item():.3f} "
                      f"l_r={l_r.mean().item():.3f} lam={lambda_.item():.2f} "
                      f"lr={optimizer.param_groups[0]['lr']:.2e} eta={eta / 60:.1f}m",
                      flush=True)
            step += 1
        if done:
            break
    timings["optimisation"] = time.perf_counter() - t0

    if args.select_best and best["state"] is not None:
        where = ("theta_0 (no checkpoint beat the unchanged model)" if best["step"] == -1
                 else f"step {best['step']}")
        print(f"[select] restoring {where}: W2_f={best['w_f']:.4f} W2_r={best['w_r']:.4g} "
              f"feasible={best['feasible']}")
        if not best["feasible"]:
            print("[select] WARNING: no checkpoint satisfied the retain constraint; "
                  "returned the least-violating one")
        model_unlearn.load_state_dict(best["state"])
    best_meta = {k: v for k, v in best.items() if k not in ("state", "key")}
    best_meta["eps_select"] = eps_select
    if device.type == "cuda":
        torch.cuda.empty_cache()

    # ------------------------------------------------------------------ evaluation
    t0 = time.perf_counter()
    swift = sc.full_evaluation(model_unlearn, eval_loaders, device, amp_dtype,
                               args.mia_n_members, args.mia_seed, tag="Swift")
    timings["evaluation"] = time.perf_counter() - t0
    rows["Swift"] = swift

    lf1, la1 = swift["losses"]["Df(tr)"], swift["losses"]["Dadj(tr)"]
    lu1 = swift["losses"]["Ftest(te)"]
    lcal1 = sc.per_sample_losses(model_unlearn, cal_eval_loader, device, amp_dtype).numpy()
    lrp1 = sc.per_sample_losses(model_unlearn, probe_eval_loader, device, amp_dtype).numpy()
    lref = probe_ref_losses.numpy()
    gaps = {"W2(Df,Dcal)": sc.w2_numpy(lf1, lcal1), "W2(Df,Ftest)": sc.w2_numpy(lf1, lu1),
            "KS(Df,Ftest)": sc.ks_numpy(lf1, lu1),
            "W2(Dr^probe,probe_ref)": sc.w2_numpy(lrp1, lref)}
    print(f"\nmedian loss Df={np.median(lf1):.3f} Dadj={np.median(la1):.3f} "
          f"Ftest={np.median(lu1):.3f}")
    print("distributional gaps: " + "  ".join(f"{k}={v:.4f}" for k, v in gaps.items()))

    reference = None
    if os.path.exists(reference_json):
        with open(reference_json) as fh:
            reference = json.load(fh)
        rows = {"Retrain (reference)": reference, **rows}
    else:
        print(f"\n[note] no reference at {reference_json}; run retrain_reference.py "
              f"--split {cfg.split} --forget-class {cfg.forget_class} to score "
              "deviations from the counterfactual")
    runtimes = {"probe_fit": timings["probe_fit"], "optimisation": timings["optimisation"],
                "Swift total (probe+opt)": timings["probe_fit"] + timings["optimisation"]}
    if reference and reference.get("train_seconds"):
        runtimes["retrain reference"] = reference["train_seconds"]
    deviations = sc.print_comparison(rows, "Retrain (reference)" if reference else None,
                                     runtimes)

    if args.plot_dir:
        _plots(args.plot_dir, cfg, diag, plog, eps_retain, lref, lf1, la1, lu1, lrp1,
               reference)

    # ------------------------------------------------------------------ save
    os.makedirs(os.path.dirname(results_json) or ".", exist_ok=True)
    payload = {
        "pipeline_version": PIPELINE_VERSION,
        "config": dict(vars(args)), "sets_config": asdict(cfg),
        "sizes": sets["sizes"], "eps_retain": eps_retain, "floor_r": floor_r,
        "floor_f": floor_f,
        "probe": {"acc_member": probe_member["acc"], "loss_member": probe_member["loss"],
                  "acc_holdout": probe_hold["acc"], "loss_holdout": probe_hold["loss"],
                  "theta0_acc": theta0_on_probe["acc"], "theta0_loss": theta0_on_probe["loss"],
                  "adjacent_inside": sets["sizes"]["probe/adjacent_inside"]},
        "best_select": best_meta, "timings": timings, "gaps": gaps,
        "results": {k: sc.strip_losses(v) for k, v in rows.items()},
        "deviations": deviations,
    }
    with open(results_json, "w") as fh:
        json.dump(payload, fh, indent=2, default=str)
    with open(results_json.replace(".json", "_diag.json"), "w") as fh:
        json.dump({"diag": diag, "track": plog}, fh)
    print(f"[done] results -> {results_json}")
    if not args.no_save_model:
        sc.save_model(model_unlearn, output,
                      extra={"method": "swift", "split": cfg.split,
                             "forget_class": cfg.forget_class, "seed": args.seed,
                             "best_select": best_meta})
        print(f"[done] model -> {output}")


def _plots(plot_dir, cfg, diag, plog, eps_retain, lref, lf1, la1, lu1, lrp1, reference):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("[plots] matplotlib not installed; skipping")
        return
    os.makedirs(plot_dir, exist_ok=True)
    stem = f"{cfg.split}_c{cfg.forget_class}"
    s = diag["step"]

    fig, axes = plt.subplots(2, 2, figsize=(13, 8))
    ax = axes[0, 0]
    ax.plot(s, diag["mean_l_f"], color="tab:red", lw=1, label="mean loss $D_f$")
    ax.plot(s, diag["mean_l_cal"], color="tab:green", lw=1,
            label="mean loss $D_{cal}$ (sg, never trained on)")
    ax.plot(s, diag["mean_l_r"], color="tab:blue", lw=1, label="mean loss $D_r$")
    ax.axhline(float(lref.mean()), color="tab:purple", ls="--", lw=1, label="probe ref mean")
    ax.set_title("Counterfactual matching: do $D_f$ and $D_{cal}$ converge?")
    ax.set_xlabel("step")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    ax = axes[0, 1]
    ax.plot(s, diag["W2_f"], color="tab:purple", lw=1)
    ax.set_yscale("log")
    ax.set_title("Forget term $W_2^2(P_\\ell(D_f), sg[P_\\ell(D_{cal})])$")
    ax.set_xlabel("step")
    ax.grid(alpha=0.3)

    ax = axes[1, 0]
    ax.plot(s, diag["W2_r"], color="tab:blue", lw=1, label="$W_2^2$ retain vs probe")
    ax.axhline(eps_retain, color="gray", ls="--", label=f"$\\epsilon_r$={eps_retain:.4f}")
    ax.set_yscale("log")
    ax2 = ax.twinx()
    ax2.plot(s, diag["lambda"], color="tab:orange", lw=1, alpha=0.7)
    ax2.set_ylabel("$\\lambda$", color="tab:orange")
    ax.set_title("Retain anchor (probe) and dual variable")
    ax.set_xlabel("step")
    ax.legend(loc="upper left", fontsize=8)
    ax.grid(alpha=0.3)

    ax = axes[1, 1]
    ax.plot(plog["step"], plog["Df"], marker="^", label="$D_f$ (train)")
    ax.plot(plog["step"], plog["Dadj"], marker="s", label="$D_r^{adj}$ (train)")
    ax.plot(plog["step"], plog["Ftest"], marker="o", label="Ftest (population)")
    if reference:
        for key, colour, name in (("Df(tr)", "tab:red", "$D_f$"),
                                  ("Ftest(te)", "tab:green", "Ftest")):
            if key in reference.get("acc", {}):
                ax.axhline(reference["acc"][key], color=colour, ls=":",
                           label=f"retrain {name}={reference['acc'][key]:.1f}")
    ax.set_title("Accuracy against the retrain reference")
    ax.set_xlabel("step")
    ax.set_ylabel("accuracy (%)")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    path = os.path.join(plot_dir, f"swift_{stem}_diagnostics.png")
    plt.savefig(path, dpi=120)
    plt.close(fig)
    print(f"[plots] {path}")

    fig, axes = plt.subplots(1, 3, figsize=(17, 4))
    axes[0].hist(lf1, bins=30, alpha=0.6, density=True, color="tab:red", label="$D_f$")
    axes[0].hist(la1, bins=30, alpha=0.6, density=True, color="tab:blue",
                 label="$D_r^{adj}$ (member)")
    axes[0].set_title("Memorisation axis")
    axes[1].hist(lf1, bins=30, alpha=0.6, density=True, color="tab:red", label="$D_f$")
    axes[1].hist(lu1, bins=30, alpha=0.6, density=True, color="tab:green",
                 label="Ftest (counterfactual)")
    axes[1].set_title("Counterfactual axis, overlap = success")
    axes[2].hist(lrp1, bins=30, alpha=0.6, density=True, color="tab:blue",
                 label="$D_r^{probe}$ under $\\theta$")
    axes[2].hist(lref, bins=30, alpha=0.6, density=True, color="tab:purple",
                 label="$D_r^{probe}$ under $\\theta_r^{probe}$")
    axes[2].set_title("Retain anchor, overlap = preserved")
    for ax in axes:
        ax.set_xlabel("per-sample CE loss")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("density")
    plt.tight_layout()
    path = os.path.join(plot_dir, f"swift_{stem}_loss_hist.png")
    plt.savefig(path, dpi=120)
    plt.close(fig)
    print(f"[plots] {path}")


if __name__ == "__main__":
    main()
