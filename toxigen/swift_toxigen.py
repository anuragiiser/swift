#!/usr/bin/env python
"""Swift on ToxiGen / RoBERTa-base.

    min_theta  W2^2( P_l(D_f; theta), sg[P_l(D_cal; theta)] )
               + lambda * [ W2^2( P_l(D_r; theta), sg[P_l(D_r^probe; theta_probe)] ) ]_+

Port of cifar100/swift_cifar100.ipynb (CIFAR-100 / ResNet-18) to text. Kept from the notebook:
pooled calibration reference refreshed every `cal_refresh` steps, lambda starting at
lambda_r with dual ascent against eps_r = eps_slack * bootstrap sampling floor, one-sided
retain term, model held in eval mode, checkpoint selection on the objective over held-out
slices of D_f, D_cal and D_r^probe (never the retrained model).

Text-specific changes: FitCheap resets the RoBERTa classification head and trains it
(plus the last encoder layer(s), the analogue of ResNet `layer4`) with separate learning
rates; "eval mode" freezes dropout instead of BatchNorm statistics; accuracy curves during
training use small tracking slices because the text sets are large.

Prerequisites: prepare_split.py and pretrain_original.py. For deviations from the
counterfactual, also run retrain_reference.py --split <split> first.
"""
import argparse
import copy
import hashlib
import json
import math
import os
import time
from dataclasses import asdict

from repro_utils import add_repo_arg, check_torch_for_transformers, import_data_module, read_forget_group
from swift_sets import (add_sets_args, default_sets_path, load_or_build_sets,
                        sets_config_from_args, subset)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0],
                                formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    add_repo_arg(p)
    add_sets_args(p)

    g = p.add_argument_group("paths")
    g.add_argument("--checkpoint", help="Original model (default <repo>/checkpoints/toxigen_roberta)")
    g.add_argument("--output-dir", help="default <repo>/checkpoints/swift_<split>_seed<seed>")
    g.add_argument("--results-json", help="default <repo>/results/swift_<split>_seed<seed>.json")
    g.add_argument("--reference-json", help="retrain reference from retrain_reference.py "
                   "(default <repo>/results/swift_reference_<split>.json, used if present)")
    g.add_argument("--probe-cache", help="default <repo>/checkpoints/swift_probe_<split>.pt")
    g.add_argument("--refit-probe", action="store_true")
    g.add_argument("--plot-dir", help="save diagnostic plots here (needs matplotlib)")
    g.add_argument("--no-save-model", action="store_true")

    g = p.add_argument_group("runtime")
    g.add_argument("--seed", type=int, default=2025)
    g.add_argument("--device", default="cuda:0")
    g.add_argument("--amp", choices=["bf16", "fp16", "none"], default="bf16")
    g.add_argument("--epochs", type=int, default=1, help="passes over D_r")
    g.add_argument("--max-steps", type=int, help="cap on optimisation steps")
    g.add_argument("--num-proc", type=int, default=4)
    g.add_argument("--eval-batch", type=int, default=256)
    g.add_argument("--skip-original-eval", action="store_true")

    g = p.add_argument_group("optimisation")
    g.add_argument("--lr", type=float, default=2e-5)
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
    g.add_argument("--eps-retain", type=float, default=-1.0, help="<0: eps_slack * sampling floor")
    g.add_argument("--eps-slack", type=float, default=1.5)
    g.add_argument("--rho0", type=float, default=2.0)
    g.add_argument("--beta", type=float, default=1.2)
    g.add_argument("--lambda-max", type=float, default=100.0)
    g.add_argument("--check-every", type=int, default=100)
    g.add_argument("--n-quantiles", type=int, default=256)
    g.add_argument("--cal-pool", type=int, default=256)
    g.add_argument("--cal-refresh", type=int, default=10)
    g.add_argument("--cal-ema", type=float, default=0.0)
    g.add_argument("--literal-objective", action="store_true",
                   help="disable every refinement (fixed lambda, two-sided, per-step "
                        "batch-sized cal reference, no selection)")

    g = p.add_argument_group("FitCheap probe")
    g.add_argument("--forget-distance", choices=["w2", "mean"], default="w2",
                   help="ablation: forget term as W2^2 (Swift) or squared difference of mean losses")
    g.add_argument("--retain-distance", choices=["w2", "mean"], default="w2",
                   help="ablation: retain term as W2^2 (Swift) or squared difference of mean losses")
    g.add_argument("--no-cal-stopgrad", action="store_true",
                   help="ablation: let gradients flow through D_cal (removes the stop-gradient of Eq. 4)")
    g.add_argument("--cal-subsample", type=float, default=1.0,
                   help="ablation: use this fraction of D_cal (seeded subsample)")
    g.add_argument("--probe-mode", choices=["fitcheap", "original"], default="fitcheap",
                   help="retain reference: FitCheap probe (paper) or theta_0's own losses on D_r^probe")
    g.add_argument("--probe-steps", type=int, default=300)
    g.add_argument("--probe-lr", type=float, default=1e-3, help="head learning rate")
    g.add_argument("--probe-body-lr", type=float, default=5e-5, help="encoder-layer learning rate")
    g.add_argument("--probe-wd", type=float, default=0.01)
    g.add_argument("--probe-bs", type=int, default=64)
    g.add_argument("--probe-reinit", choices=["head", "head_lastlayers", "none"],
                   default="head_lastlayers")
    g.add_argument("--probe-layers", type=int, default=1, help="trainable last encoder layers")
    g.add_argument("--probe-dropout", choices=["off", "on"], default="off")

    g = p.add_argument_group("selection / logging / MIA")
    g.add_argument("--no-select-best", action="store_true")
    g.add_argument("--select-rule", choices=["constrained", "penalty"], default="constrained",
                   help="constrained (Eq. 6): a checkpoint must keep W2_r within tolerance on held-out "
                        "D_r^probe rows, then lowest W2_f wins, theta_0 is a candidate; "
                        "penalty (notebook): lowest W2_f + lambda * W2_r")
    g.add_argument("--select-lambda", choices=["initial", "current"], default="initial",
                   help="lambda in the selection score (notebook: initial)")
    g.add_argument("--select-every", type=int, default=100)
    g.add_argument("--log-every", type=int, default=50)
    g.add_argument("--mia-seed", type=int, default=7)
    g.add_argument("--mia-n-members", type=int, default=2000)
    args = p.parse_args()

    args.select_best = not args.no_select_best
    if args.literal_objective:
        args.lambda_mode, args.retain_mode = "fixed", "two_sided"
        args.cal_refresh, args.cal_pool, args.cal_ema = 1, args.batch_forget, 0.0
        args.select_best = False
        print("literal_objective: all refinements disabled")
    return args


PIPELINE_VERSION = "2026-09-25.5"


def mean_sq(losses, ref, one_sided=False):
    """Mean-matching counterpart of W2^2: (mean(losses) - mean(ref))^2 (one-sided: positive part)."""
    import torch
    d = losses.float().mean() - ref.float().mean()
    return torch.relu(d) ** 2 if one_sided else d ** 2


def w2_sq_between_grad(losses_a, losses_b):
    """W2^2 between two loss samples with gradient through BOTH (no stop-gradient)."""
    import torch
    a, b = losses_a.float().flatten(), losses_b.float().flatten()
    grid = (torch.arange(a.numel(), device=a.device, dtype=a.dtype) + 0.5) / a.numel()
    return torch.mean((torch.sort(a)[0] - torch.quantile(b, grid)) ** 2)


def estimate_floor(sc, ref, batch_size, n_quantiles, kind, one_sided, seed=0, q=0.5, n_boot=200):
    """Sampling floor of the retain distance actually optimised ('w2' or 'mean')."""
    import numpy as np
    import torch
    if kind == "w2":
        return sc.estimate_w2_floor(ref, batch_size, n_quantiles, one_sided=one_sided, seed=seed, q=q)
    x = ref.detach().float().flatten()
    g = torch.Generator(device="cpu").manual_seed(seed)
    # two-sided on purpose: a one-sided mean difference is exactly 0 for ~half the batches,
    # so its median floor would be 0 and any excess would count as a violation
    vals = [mean_sq(x[torch.randint(0, x.numel(), (batch_size,), generator=g).to(x.device)], x, False).item()
            for _ in range(n_boot)]
    return float(np.quantile(vals, q))


def check_versions():
    import swift_common
    found = {"swift_toxigen.py": PIPELINE_VERSION,
             "swift_common.py": getattr(swift_common, "PIPELINE_VERSION", "old")}
    if len(set(found.values())) != 1:
        raise SystemExit(f"[error] files from different versions: {found}. Check out the toxigen/ folder again.")


def main():
    args = parse_args()
    check_versions()
    check_torch_for_transformers()

    import warnings
    # torch<2.3 SequentialLR passes `epoch` internally and warns on every step
    warnings.filterwarnings("ignore", message="The epoch parameter", category=UserWarning)
    import numpy as np
    import torch
    import torch.nn.functional as F
    from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
    from transformers import AutoModelForSequenceClassification

    import swift_common as sc

    repo = args.repo_dir
    split = args.split
    ckpt = args.checkpoint or os.path.join(repo, "checkpoints", "toxigen_roberta")
    out_dir = args.output_dir or os.path.join(repo, "checkpoints", f"swift_{split}_seed{args.seed}")
    results_json = args.results_json or os.path.join(repo, "results", f"swift_{split}_seed{args.seed}.json")
    reference_json = args.reference_json or os.path.join(repo, "results", f"swift_reference_{split}.json")
    probe_cache = args.probe_cache or os.path.join(repo, "checkpoints", f"swift_probe_{split}.pt")
    if not os.path.isdir(ckpt):
        raise SystemExit(f"Original checkpoint {ckpt} not found; run pretrain_original.py first")

    sc.set_seed(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    amp_dtype, use_scaler = sc.resolve_amp(args.amp, device)
    one_sided = args.retain_mode == "one_sided"
    print(f"device={device} amp={amp_dtype} split={split}")
    print(f"objective: W2^2(Df, sg[Dcal]) + lambda * W2^2(Dr, sg[probe])  "
          f"[lambda={args.lambda_r}, mode={args.lambda_mode}, retain={args.retain_mode}]")

    # ------------------------------------------------------------------ data
    tg = import_data_module(repo)
    fg = read_forget_group(repo)
    ds_tok, collator, _ = tg.load_and_tokenize(num_proc=args.num_proc)
    cfg = sets_config_from_args(args, fg)
    sets = load_or_build_sets(ds_tok, cfg, default_sets_path(repo, split), rebuild=args.rebuild_sets)
    if args.cal_subsample < 1.0:
        full = sets["test"]["cal"]
        n_keep = max(2, int(round(args.cal_subsample * len(full))))
        sets["test"]["cal"] = np.sort(np.random.default_rng(args.seed + 7).choice(full, n_keep, replace=False))
        print(f"[ablation] D_cal subsampled to {n_keep} of {len(full)}")
    S = lambda sp, name: subset(ds_tok, sp, sets[sp][name], fg)  # noqa: E731
    L = lambda ds, bs, **kw: sc.make_loader(ds, bs, collator, **kw)  # noqa: E731

    eval_loaders = {"Df(tr)": L(S("train", "forget"), args.eval_batch),
                    "Dadj(tr)": L(S("train", "adjacent"), args.eval_batch),
                    "Drem(tr)": L(S("train", "remote"), args.eval_batch),
                    "Ftest(te)": L(S("test", "ftest"), args.eval_batch),
                    "Drem(te)": L(S("test", "remote_te"), args.eval_batch)}
    if len(sets["test"]["adjacent_te"]):
        eval_loaders["Dadj(te)"] = L(S("test", "adjacent_te"), args.eval_batch)

    retain_loader = L(S("train", "retain"), args.batch_retain, shuffle=True, drop_last=True)
    forget_loader = L(S("train", "forget"), args.batch_forget, shuffle=True, drop_last=True)
    cal_ds = S("test", "cal")
    cal_loader = L(cal_ds, args.batch_forget, shuffle=True)
    cal_eval_loader = L(cal_ds, args.eval_batch)
    probe_ds = S("train", "probe")
    probe_train_loader = L(probe_ds, args.probe_bs, shuffle=True, drop_last=True)
    probe_eval_loader = L(probe_ds, args.eval_batch)
    probe_holdout_loader = L(S("train", "probe_holdout"), args.eval_batch)
    track_loaders = {"Df": L(S("train", "track_forget"), args.eval_batch),
                     "Dadj": L(S("train", "track_adjacent"), args.eval_batch),
                     "Ftest": L(S("test", "track_ftest"), args.eval_batch)}
    select_loaders = {"Df": L(S("train", "select_forget"), args.eval_batch),
                      "cal": cal_eval_loader,
                      "Rprobe": L(S("train", "select_probe"), args.eval_batch)}

    n_adj_probe = int(np.isin(sets["train"]["probe"], sets["train"]["adjacent"]).sum())
    sizes = {k: len(v.dataset) for k, v in eval_loaders.items()}
    sizes.update({"D_cal": len(cal_ds), "D_r^probe": len(probe_ds), "adj_in_probe": n_adj_probe})
    print(f"[sets] {sizes}")

    # ------------------------------------------------------------------ original
    model = AutoModelForSequenceClassification.from_pretrained(ckpt).to(device).eval()
    timings, rows = {}, {}
    if not args.skip_original_eval:
        rows["Original"] = sc.full_evaluation(model, eval_loaders, device, amp_dtype,
                                              args.mia_n_members, args.mia_seed, tag="original")

    # ------------------------------------------------------------------ FitCheap
    probe_key = hashlib.sha1(json.dumps({
        "idx": hashlib.sha1(sets["train"]["probe"].tobytes()).hexdigest(),
        "ckpt": os.path.abspath(ckpt), "steps": args.probe_steps, "lr": args.probe_lr,
        "body_lr": args.probe_body_lr, "wd": args.probe_wd, "bs": args.probe_bs,
        "reinit": args.probe_reinit, "layers": args.probe_layers, "dropout": args.probe_dropout,
    }, sort_keys=True).encode()).hexdigest()

    t0 = time.perf_counter()
    probe_model = None
    if args.probe_mode == "original":
        probe_model = model
        print("[probe] --probe-mode original: retain reference = theta_0's losses on D_r^probe")
    elif not args.refit_probe and os.path.exists(probe_cache):
        try:
            blob = torch.load(probe_cache, map_location="cpu")
            if blob.get("key") == probe_key:
                probe_model = copy.deepcopy(model)
                probe_model.load_state_dict(blob["state_dict"])
                probe_model.to(device).eval()
                print(f"[probe] loaded cache {probe_cache}")
            else:
                print("[probe] cache was fitted with a different subset/config; refitting")
        except Exception as exc:  # noqa: BLE001
            print(f"[probe] could not reuse cache ({exc}); refitting")
    if probe_model is None:
        sc.set_seed(args.seed)
        probe_model = sc.fit_cheap_probe(
            model, probe_train_loader, device, steps=args.probe_steps, head_lr=args.probe_lr,
            body_lr=args.probe_body_lr, weight_decay=args.probe_wd, reinit=args.probe_reinit,
            n_layers=args.probe_layers, dropout=args.probe_dropout == "on",
            amp_dtype=amp_dtype, use_scaler=use_scaler, max_norm=args.max_norm)
        os.makedirs(os.path.dirname(probe_cache) or ".", exist_ok=True)
        torch.save({"state_dict": probe_model.state_dict(), "key": probe_key}, probe_cache)
        print(f"[probe] saved {probe_cache}")
    timings["probe_fit"] = time.perf_counter() - t0

    probe_ref_losses = sc.per_sample_losses(probe_model, probe_eval_loader, device, amp_dtype)
    probe_ref_q = sc.reference_quantiles(probe_ref_losses, args.n_quantiles, device)
    probe_ref_mean = probe_ref_losses.to(device).float()
    probe_member = sc.acc_and_loss(probe_model, probe_eval_loader, device, amp_dtype)
    probe_hold = sc.acc_and_loss(probe_model, probe_holdout_loader, device, amp_dtype)
    theta0_on_probe = sc.acc_and_loss(model, probe_eval_loader, device, amp_dtype)
    print(f"[probe] fit {timings['probe_fit']:.1f}s | on D_r^probe acc={probe_member['acc']:.2f} "
          f"loss={probe_member['loss']:.4f} | unseen retain acc={probe_hold['acc']:.2f} "
          f"loss={probe_hold['loss']:.4f} | theta_0 on D_r^probe acc={theta0_on_probe['acc']:.2f} "
          f"loss={theta0_on_probe['loss']:.4f}")
    if probe_member["loss"] > 3.0 * max(theta0_on_probe["loss"], 1e-6):
        print("WARNING: the probe is much weaker than theta_0 on D_r. With --retain-mode two_sided "
              "it will pull retain accuracy down; raise --probe-steps/--probe-size or keep one_sided.")
    del probe_model
    if device.type == "cuda":
        torch.cuda.empty_cache()

    # ------------------------------------------------------------------ floors / eps
    floor_r = estimate_floor(sc, probe_ref_losses.to(device), args.batch_retain, args.n_quantiles,
                             args.retain_distance, one_sided, seed=args.seed)
    cal_theta0 = sc.per_sample_losses(model, cal_eval_loader, device, amp_dtype)
    floor_f = sc.estimate_w2_floor(cal_theta0.to(device), args.batch_forget, args.n_quantiles,
                                   seed=args.seed)
    eps_retain = args.eps_slack * floor_r if args.eps_retain < 0 else args.eps_retain
    n_sel = len(select_loaders["Rprobe"].dataset)
    floor_sel = estimate_floor(sc, probe_ref_losses.to(device), n_sel, args.n_quantiles,
                               args.retain_distance, one_sided, seed=args.seed + 1, q=0.95)
    hold_theta0 = sc.acc_and_loss(model, probe_holdout_loader, device, amp_dtype)["acc"]
    print(f"[floor] retain(batch={args.batch_retain})={floor_r:.4g} "
          f"forget(batch={args.batch_forget})={floor_f:.4g} eps_retain={eps_retain:.4g}")
    if eps_retain <= 0 and args.lambda_mode == "adaptive":
        print("WARNING: eps_retain is 0, so any retain excess keeps raising lambda (up to "
              "--lambda-max). Set --eps-retain explicitly if this is not intended.")

    # ------------------------------------------------------------------ optimisation
    sc.set_seed(args.seed)
    model.eval()                     # dropout off throughout, as BN is frozen in the notebook
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
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
    plog = {"step": [], "Df": [], "Dadj": [], "Ftest": [], "Dr_hold": [], "score": [], "feasible": []}
    best = {"key": None, "step": None, "state": None}
    cal_ref = None

    @torch.no_grad()
    def refresh_cal_reference(prev):
        pool, got = [], 0
        while got < args.cal_pool:
            with sc.autocast(amp_dtype):
                logits, yb = sc.forward_logits(model, next(cycled_cal), device)
            lb = F.cross_entropy(logits.float(), yb, reduction="none")
            pool.append(lb)
            got += lb.numel()
        q = sc.reference_quantiles(torch.cat(pool), args.n_quantiles, device)
        if prev is not None and args.cal_ema > 0.0:
            q = args.cal_ema * prev + (1.0 - args.cal_ema) * q
        return q

    @torch.no_grad()
    def selection_score():
        l_f = sc.per_sample_losses(model, select_loaders["Df"], device, amp_dtype).to(device)
        l_c = sc.per_sample_losses(model, select_loaders["cal"], device, amp_dtype).to(device)
        l_r = sc.per_sample_losses(model, select_loaders["Rprobe"], device, amp_dtype).to(device)
        w_f = (mean_sq(l_f, l_c) if args.forget_distance == "mean" else sc.w2_sq_between(l_f, l_c)).item()
        w_r = (mean_sq(l_r, probe_ref_losses.to(device), one_sided) if args.retain_distance == "mean"
               else sc.w2_sq_to_reference(l_r, probe_ref_q, one_sided=one_sided)).item()
        lam = args.lambda_r if args.select_lambda == "initial" else float(lambda_.item())
        return w_f + lam * w_r, w_f, w_r

    eps_select = None

    def consider(step, record=True):
        """constrained: feasible checkpoints (W2_r <= eps_select) ranked by W2_f, infeasible
        ones only as fallback ranked by W2_r; penalty: the notebook score."""
        nonlocal best
        score, w_f, w_r = selection_score()
        feasible = w_r <= eps_select
        key = (score,) if args.select_rule == "penalty" else ((0, w_f) if feasible else (1, w_r))
        if record:
            plog["score"].append(score)
            plog["feasible"].append(bool(feasible))
        if best["key"] is None or key < best["key"]:
            best = {"key": key, "step": step, "score": score, "w_f": w_f, "w_r": w_r,
                    "feasible": bool(feasible),
                    "state": {k: v.detach().to("cpu", copy=True) for k, v in model.state_dict().items()}}
            print(f"  [select] new best @step {step}: W2_f={w_f:.4f} W2_r={w_r:.4g} "
                  f"({'feasible' if feasible else 'INFEASIBLE, eps=' + format(eps_select, '.3g')})")
        model.eval()

    if args.select_best:
        _, _, w_r_theta0 = selection_score()
        eps_select = args.eps_slack * max(floor_sel, w_r_theta0)   # theta_0 passes its own constraint
        print(f"[select] rule={args.select_rule} eps={eps_select:.4g} (noise q95={floor_sel:.4g}, "
              f"theta_0 on slice={w_r_theta0:.4g}); retain hold-out acc of theta_0={hold_theta0:.2f}")
        consider(-1, record=False)     # theta_0 is a candidate but not part of the step log

    print(f"[train] {total_steps} steps (retain batch {args.batch_retain}, forget batch "
          f"{args.batch_forget}), warmup {warmup_steps}")
    t0 = time.perf_counter()
    step, done = 0, False
    for epoch in range(args.epochs):
        for batch_id, batch_r in enumerate(retain_loader):
            if step >= total_steps:
                done = True
                break
            batch_f = next(cycled_forget)
            if cal_ref is None or step % args.cal_refresh == 0:
                cal_ref = refresh_cal_reference(cal_ref)

            optimizer.zero_grad(set_to_none=True)
            with sc.autocast(amp_dtype):
                logits_r, y_r = sc.forward_logits(model, batch_r, device)
                logits_f, y_f = sc.forward_logits(model, batch_f, device)
            # W2 terms in fp32: sorting/quantiles in half precision are badly conditioned
            l_r = F.cross_entropy(logits_r.float(), y_r, reduction="none")
            l_f = F.cross_entropy(logits_f.float(), y_f, reduction="none")
            if not args.no_cal_stopgrad:
                W2_f = mean_sq(l_f, cal_ref) if args.forget_distance == "mean" else sc.w2_sq_to_reference(l_f, cal_ref)
            else:                                   # ablation: gradient through D_cal as well
                with sc.autocast(amp_dtype):
                    logits_c, y_c = sc.forward_logits(model, next(cycled_cal), device)
                l_c = F.cross_entropy(logits_c.float(), y_c, reduction="none")
                W2_f = mean_sq(l_f, l_c) if args.forget_distance == "mean" else w2_sq_between_grad(l_f, l_c)
            W2_r = (mean_sq(l_r, probe_ref_mean, one_sided) if args.retain_distance == "mean"
                    else sc.w2_sq_to_reference(l_r, probe_ref_q, one_sided=one_sided))
            loss = W2_f + lambda_ * W2_r

            scaler.scale(loss).backward()
            if not args.no_clip:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_norm)
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

            vals = (("step", step), ("loss", loss.item()), ("W2_f", W2_f.item()),
                    ("W2_r", W2_r.item()), ("mean_l_f", l_f.mean().item()),
                    ("mean_l_cal", cal_ref.mean().item()), ("mean_l_r", l_r.mean().item()),
                    ("lambda", float(lambda_.item())), ("rho", rho),
                    ("lr", optimizer.param_groups[0]["lr"]))
            for k, v in vals:
                diag[k].append(v)
            if not math.isfinite(diag["loss"][-1]):
                raise SystemExit(f"non-finite loss at step {step}; lower --lr or use --amp none")

            if step % args.select_every == 0 or step == total_steps - 1:
                plog["step"].append(step)
                for key in ("Df", "Dadj", "Ftest"):
                    plog[key].append(sc.acc_and_loss(model, track_loaders[key], device, amp_dtype)["acc"])
                plog["Dr_hold"].append(sc.acc_and_loss(model, probe_holdout_loader, device, amp_dtype)["acc"])
                if args.select_best:
                    consider(step)
                else:
                    plog["score"].append(float("nan"))
                    plog["feasible"].append(None)
                model.eval()
                print(f"  [track @{step}] Df={plog['Df'][-1]:.2f} Dadj={plog['Dadj'][-1]:.2f} "
                      f"Ftest={plog['Ftest'][-1]:.2f} retain-holdout={plog['Dr_hold'][-1]:.2f}")
                if plog["Dr_hold"][-1] < hold_theta0 - 20:
                    print(f"  [warn] retain hold-out accuracy fell {hold_theta0 - plog['Dr_hold'][-1]:.1f} "
                          "points below theta_0: the model is collapsing; lower --lr")

            if step % args.log_every == 0:
                el = time.perf_counter() - t0
                eta = el / (step + 1) * (total_steps - step - 1)
                print(f"[step {step}/{total_steps}] loss={loss.item():.4f} W2_f={W2_f.item():.4f} "
                      f"W2_r={W2_r.item():.4f} l_f={l_f.mean().item():.3f} "
                      f"l_cal={cal_ref.mean().item():.3f} l_r={l_r.mean().item():.3f} "
                      f"lam={lambda_.item():.2f} lr={optimizer.param_groups[0]['lr']:.2e} "
                      f"eta={eta / 60:.1f}m", flush=True)
            step += 1
        if done:
            break

    timings["optimisation"] = time.perf_counter() - t0
    if args.select_best and best["state"] is not None:
        where = "theta_0 (no checkpoint beat the unchanged model)" if best["step"] == -1 else f"step {best['step']}"
        print(f"[select] restoring {where}: W2_f={best['w_f']:.4f} W2_r={best['w_r']:.4g} "
              f"feasible={best['feasible']}")
        if not best["feasible"]:
            print("[select] WARNING: no checkpoint satisfied the retain constraint; returned the least-violating one")
        model.load_state_dict(best["state"])
    best_meta = {k: v for k, v in best.items() if k not in ("state", "key")}
    best_meta["eps_select"] = eps_select
    print(f"optimisation: {timings['optimisation']:.1f}s")

    # ------------------------------------------------------------------ evaluation
    t0 = time.perf_counter()
    swift = sc.full_evaluation(model, eval_loaders, device, amp_dtype,
                               args.mia_n_members, args.mia_seed, tag="Swift")
    timings["evaluation"] = time.perf_counter() - t0
    rows["Swift"] = swift

    lf1, la1, lu1 = swift["losses"]["Df(tr)"], swift["losses"]["Dadj(tr)"], swift["losses"]["Ftest(te)"]
    lcal1 = sc.per_sample_losses(model, cal_eval_loader, device, amp_dtype).numpy()
    lrp1 = sc.per_sample_losses(model, probe_eval_loader, device, amp_dtype).numpy()
    lref = probe_ref_losses.numpy()
    gaps = {"W2(Df,Dcal)": sc.w2_numpy(lf1, lcal1), "W2(Df,Ftest)": sc.w2_numpy(lf1, lu1),
            "KS(Df,Ftest)": sc.ks_numpy(lf1, lu1), "W2(Dr^probe,probe_ref)": sc.w2_numpy(lrp1, lref)}
    print(f"\nmedian loss Df={np.median(lf1):.3f} Dadj={np.median(la1):.3f} Ftest={np.median(lu1):.3f}")
    print("distributional gaps: " + "  ".join(f"{k}={v:.4f}" for k, v in gaps.items()))

    reference = None
    if os.path.exists(reference_json):
        with open(reference_json) as fh:
            reference = json.load(fh)
        rows = {"Retrain (reference)": reference, **rows}
    else:
        print(f"\n[note] no reference at {reference_json}; run retrain_reference.py --split {split} "
              "to score deviations from the counterfactual")
    runtimes = {"probe_fit": timings["probe_fit"], "optimisation": timings["optimisation"],
                "Swift total (probe+opt)": timings["probe_fit"] + timings["optimisation"]}
    if reference and reference.get("train_seconds"):
        runtimes["retrain reference"] = reference["train_seconds"]
    deviations = sc.print_comparison(rows, "Retrain (reference)" if reference else None, runtimes)

    if args.plot_dir:
        _plots(args.plot_dir, split, diag, plog, eps_retain, lref, lf1, la1, lu1, lrp1, reference)

    # ------------------------------------------------------------------ save
    os.makedirs(os.path.dirname(results_json) or ".", exist_ok=True)
    payload = {
        "config": dict(vars(args)), "sets_config": asdict(cfg),
        "sizes": sizes, "eps_retain": eps_retain, "floor_r": floor_r, "floor_f": floor_f,
        "probe": {"acc_member": probe_member["acc"], "loss_member": probe_member["loss"],
                  "acc_holdout": probe_hold["acc"], "loss_holdout": probe_hold["loss"],
                  "theta0_acc": theta0_on_probe["acc"], "theta0_loss": theta0_on_probe["loss"]},
        "best_select": best_meta, "timings": timings, "gaps": gaps,
        "results": {k: sc.strip_losses(v) for k, v in rows.items()}, "deviations": deviations,
    }
    with open(results_json, "w") as fh:
        json.dump(payload, fh, indent=2, default=str)
    with open(results_json.replace(".json", "_diag.json"), "w") as fh:
        json.dump({"diag": diag, "track": plog}, fh)
    print(f"[done] results -> {results_json}")
    if not args.no_save_model:
        os.makedirs(out_dir, exist_ok=True)
        model.save_pretrained(out_dir)
        collator.tokenizer.save_pretrained(out_dir)
        print(f"[done] model -> {out_dir}")


def _plots(plot_dir, split, diag, plog, eps_retain, lref, lf1, la1, lu1, lrp1, reference):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("[plots] matplotlib not installed; skipping")
        return
    os.makedirs(plot_dir, exist_ok=True)
    s = diag["step"]
    fig, axes = plt.subplots(2, 2, figsize=(13, 8))
    ax = axes[0, 0]
    ax.plot(s, diag["mean_l_f"], color="tab:red", lw=1, label="mean loss $D_f$")
    ax.plot(s, diag["mean_l_cal"], color="tab:green", lw=1, label="mean loss $D_{cal}$ (sg)")
    ax.plot(s, diag["mean_l_r"], color="tab:blue", lw=1, label="mean loss $D_r$")
    ax.axhline(float(lref.mean()), color="tab:purple", ls="--", lw=1, label="probe ref mean")
    ax.set_title("Do $D_f$ and $D_{cal}$ converge?"); ax.legend(fontsize=8); ax.grid(alpha=0.3)
    ax = axes[0, 1]
    ax.plot(s, diag["W2_f"], color="tab:purple", lw=1); ax.set_yscale("log")
    ax.set_title("Forget term $W_2^2(D_f, sg[D_{cal}])$"); ax.grid(alpha=0.3)
    ax = axes[1, 0]
    ax.plot(s, diag["W2_r"], color="tab:blue", lw=1, label="$W_2^2$ retain vs probe")
    ax.axhline(eps_retain, color="gray", ls="--", label=f"$\\epsilon_r$={eps_retain:.4f}")
    ax.set_yscale("log"); ax.legend(loc="upper left", fontsize=8); ax.grid(alpha=0.3)
    ax2 = ax.twinx(); ax2.plot(s, diag["lambda"], color="tab:orange", lw=1, alpha=.7)
    ax2.set_ylabel("$\\lambda$", color="tab:orange"); ax.set_title("Retain anchor and dual variable")
    ax = axes[1, 1]
    ax.plot(plog["step"], plog["Df"], marker="^", label="$D_f$ (train, slice)")
    ax.plot(plog["step"], plog["Dadj"], marker="s", label="$D_r^{adj}$ (train, slice)")
    ax.plot(plog["step"], plog["Ftest"], marker="o", label="Ftest (held out, slice)")
    if reference:
        ax.axhline(reference["acc"]["Df(tr)"], color="tab:red", ls=":", label="retrain $D_f$")
        ax.axhline(reference["acc"]["Ftest(te)"], color="tab:green", ls=":", label="retrain Ftest")
    ax.set_title("Accuracy vs the retrain reference"); ax.set_ylabel("accuracy (%)")
    ax.legend(fontsize=8); ax.grid(alpha=0.3)
    for a in axes.flat:
        a.set_xlabel("step")
    plt.tight_layout()
    path = os.path.join(plot_dir, f"swift_{split}_diagnostics.png")
    plt.savefig(path, dpi=120); plt.close(fig)

    fig, axes = plt.subplots(1, 3, figsize=(17, 4))
    axes[0].hist(lf1, bins=40, alpha=.6, density=True, color="tab:red", label="$D_f$")
    axes[0].hist(la1, bins=40, alpha=.6, density=True, color="tab:blue", label="$D_r^{adj}$")
    axes[0].set_title("Memorization axis")
    axes[1].hist(lf1, bins=40, alpha=.6, density=True, color="tab:red", label="$D_f$")
    axes[1].hist(lu1, bins=40, alpha=.6, density=True, color="tab:green", label="Ftest (unseen)")
    axes[1].set_title("Counterfactual axis, overlap = success")
    axes[2].hist(lrp1, bins=40, alpha=.6, density=True, color="tab:blue", label="$D_r^{probe}$ under $\\theta$")
    axes[2].hist(lref, bins=40, alpha=.6, density=True, color="tab:purple", label="under probe")
    axes[2].set_title("Retain anchor, overlap = preserved")
    for a in axes:
        a.set_xlabel("per-sample CE loss"); a.legend(fontsize=8); a.grid(alpha=0.3)
    plt.tight_layout()
    path2 = os.path.join(plot_dir, f"swift_{split}_loss_hist.png")
    plt.savefig(path2, dpi=120); plt.close(fig)
    print(f"[plots] {path}, {path2}")


if __name__ == "__main__":
    main()
