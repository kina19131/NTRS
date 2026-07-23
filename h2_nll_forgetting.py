"""
h2_nll_forgetting.py
=====================
Does NOT modify h2_sequential_llama.py — imports its data-loading and eval
utilities unchanged (load_task, compute_task_nll, compute_sigma_half,
compute_lora_delta_norm, eval_accuracy), but does NOT reuse its run_phase1
(see the import block below for why: it hardcodes Llama-style target
modules and silently breaks on GPT-2). A minimal, architecture-generic
run_phase1_generic is defined in this file instead. Adds a new Phase 2
tracking function that measures Task A forgetting via NLL on Task A's own
held-out text, alongside (not instead of) the existing classifier-accuracy
measure.

Motivation (2026-07-21): H2's accuracy-based forgetting signal comes from
feeding Phase-2-drifted hidden states through the FROZEN Phase-1 classifier
head. That is structurally the same "old probe vs new representation" bug
already found and fixed in h1_direct_r_sweep.py (Bug 4: a classifier trained
against one representational state, evaluated against a different one, after
LoRA weights were rewound to an earlier checkpoint). If different Phase 2
tasks reorganize the shared representation space differently, the SAME
underlying Phase-1 information could look "forgotten" at wildly different
R_A depending on which Phase 2 task is doing the reorganizing - which is
exactly the unexplained SST-2 anomaly already on record: T*(SST-2->MNLI,r8)
~=0.425 vs T*(SST-2->CoLA,r8)~=0.012, a 44x spread across only 3 Phase-2
targets (see SUMMARY.md "What Claims Can Be Made Now" / "known vulnerability
V8"). Four mechanistic hypotheses for this were tested and came back null
(gradient direction overlap, effective rank, binary/multiclass, subspace
projection) - none of them touch the classifier-readout mechanism.

NLL on Task A's own held-out text (next-token prediction, no classifier
involved at all) is a probe-free alternative forgetting signal - the same
conceptual move H1 already made preferring NLL over KL/accuracy, applied one
level down to H2. If the SST-2 anomaly shrinks under NLL-based forgetting,
that's evidence the accuracy-based anomaly was substantially a classifier-
readout artifact. If it persists just as badly, that rules out this specific
mechanism (informative either way, not just a negative result).

Note this NLL is domain-specific (does the model still model SST-2-STYLE
TEXT well), not the same general-capability WikiText-2 NLL H1 uses - it is a
different KIND of forgetting signal from classification accuracy, not a
strictly-better version of the same one.

Per h1_kl_comparison.py's relslack convention, the NLL-forgetting threshold
is defined as 1% of the model's OWN Task-A NLL baseline (--nll_relslack),
not a fixed absolute value - the same fix that corrected cross-model
eval_slack comparisons in H1.

Validate cheaply on GPT-2 first (default --model) before spending real
GPU-hours on Llama, per this project's established practice.

Output:
  results/h2_nll_forgetting/{model_tag}/{task_a}_to_{task_b}/
    phase1_model.pt, phase1_classifier.pt, phase1_meta.json, phase1_sigma_half.json
        (written by this file's own run_phase1_generic - NOT byte-compatible
        with h2_sequential_llama.py's phase1_meta.json schema, since that
        function is not reused here; --skip_phase1 only reloads checkpoints
        this script itself produced)
    lr{X}_rank{Y}_trajectory.json  - per-step (R_A, acc_A, NLL_A, delta_NLL_A)
    lr{X}_rank{Y}_result.json      - per-condition T*_acc AND T*_nll brackets
    summary.json                  - all conditions, one task_a -> multiple task_b
    nll_vs_acc_forgetting.png      - T*_acc vs T*_nll spread across task_b targets

Usage:
  # Cheap validation on GPT-2, the 3 conditions showing the SST-2 anomaly under accuracy
  python h2_nll_forgetting.py --model gpt2 --task_a sst2 --task_b mnli agnews cola

  # Once validated, the real test:
  python h2_nll_forgetting.py --model meta-llama/Llama-3.2-3B --task_a sst2 --task_b mnli agnews cola

  # Reuse an existing Phase 1 checkpoint (e.g. one h2_sequential_llama.py already trained)
  python h2_nll_forgetting.py --model meta-llama/Llama-3.2-3B --task_a sst2 --task_b mnli \\
      --skip_phase1 --output_dir results/h2_llama
"""

import argparse
import copy
import json
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(__file__))
from lora_density_experiment import (
    _load_model_auto, _get_target_names_auto, _get_hidden_size,
    _get_tokenizer, add_lora,
)
from h1_kl_comparison import _lora_targets
# Reused unchanged from h2_sequential_llama.py - task data loading, NLL/
# accuracy eval, and the analytic LoRA-delta-norm computation are already-
# validated machinery; nothing here duplicates or modifies them.
#
# NOT reusing h2_sequential_llama.py's own run_phase1: it hardcodes Llama-
# style target_modules (q_proj/v_proj/k_proj/o_proj) regardless of which
# model is passed in, so it silently attaches zero LoRA adapters on GPT-2
# (which has c_attn/c_proj instead) - confirmed by inspection 2026-07-21,
# not fixable here since h2_sequential_llama.py must not be modified. A
# minimal, architecture-generic replacement (using _lora_targets, same as
# every other script in this project) is defined below instead.
from h2_sequential_llama import (
    load_task, compute_task_nll, compute_sigma_half,
    compute_per_param_norm, compute_lora_delta_norm, eval_accuracy,
)

CATEGORICAL = {
    "acc": "#2a78d6", "nll": "#e34948", "gpt2": "#1baf7a", "llama": "#4a3aa7",
}


def get_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model",  default="gpt2",
                   help="Default gpt2 for cheap validation; pass "
                        "meta-llama/Llama-3.2-3B for the real test once validated.")
    p.add_argument("--task_a", default="sst2", choices=["mnli", "sst2", "agnews", "cola"])
    p.add_argument("--task_b", nargs="+", default=["mnli", "agnews", "cola"],
                   choices=["mnli", "sst2", "agnews", "cola"],
                   help="Multiple Phase 2 targets swept for the SAME Phase 1 "
                        "checkpoint - this is the actual comparison: does the "
                        "spread across these narrow under NLL vs accuracy?")

    # Phase 1 (consumed by this file's own run_phase1_generic)
    p.add_argument("--phase1_lr",    type=float, default=5e-5)
    p.add_argument("--phase1_rank",  type=int,   default=8)
    p.add_argument("--phase1_alpha", type=int,   default=16)
    p.add_argument("--phase1_steps", type=int,   default=500,
                   help="Small default for cheap GPT-2 validation; the "
                        "existing H2 Llama runs used 2000.")

    # Phase 2 sweep
    p.add_argument("--phase2_lrs",   nargs="+", type=float,
                   default=[1e-5, 5e-5, 1e-4, 2e-4, 5e-4])
    p.add_argument("--phase2_rank",  type=int, default=8)
    p.add_argument("--phase2_steps", type=int, default=500)
    p.add_argument("--track_interval", type=int, default=25)

    # sigma_half_A (NLL-based, reuses h2_sequential_llama.py's own compute_sigma_half)
    p.add_argument("--n_perturb",      type=int,   default=200)
    p.add_argument("--sigma_holdout",  type=int,   default=256)
    p.add_argument("--seq_len",        type=int,   default=128)
    p.add_argument("--eval_slack",     type=float, default=1e-4)

    # NLL-forgetting-specific
    p.add_argument("--nll_holdout", type=int, default=256,
                   help="Task A held-out sequences for tracking NLL_A during "
                        "Phase 2 - separate draw from --sigma_holdout so the "
                        "forgetting signal isn't evaluated on the exact same "
                        "text used to calibrate sigma_half_A.")
    p.add_argument("--nll_relslack", type=float, default=0.01,
                   help="Forgetting threshold: NLL_A exceeding this fraction "
                        "of its own Phase-1 baseline (same relslack convention "
                        "as h1_kl_comparison.py, not a fixed absolute value).")
    p.add_argument("--acc_slack", type=float, default=0.10,
                   help="Accuracy-based forgetting threshold (existing H2 "
                        "convention): acc_A dropping below (1-this)*acc_phase1_A.")

    p.add_argument("--batch_size",   type=int,   default=16)
    p.add_argument("--grad_clip",    type=float, default=1.0)
    p.add_argument("--eval_samples", type=int,   default=500)
    p.add_argument("--seed",         type=int,   default=42)
    p.add_argument("--output_dir",   default="./results/h2_nll_forgetting")
    p.add_argument("--skip_phase1",  action="store_true")
    p.add_argument("--replot",       action="store_true")
    p.add_argument("--force",        action="store_true")
    return p.parse_args()


def _bracket(trajectory, key, threshold, above_is_forgotten=True):
    """
    Last-safe / first-forgotten R_A bracket for a trajectory, given a scalar
    per-step value (key) and threshold. Same grid-resolution-bracket
    convention used throughout this project's T* reporting (see SUMMARY.md's
    standing methodological note) - NOT a confidence interval.

    above_is_forgotten=True: forgotten when trajectory[key] > threshold (NLL).
    above_is_forgotten=False: forgotten when trajectory[key] < threshold (acc).

    A step only counts as the genuine forgetting point if the criterion
    stays violated for every subsequent measured step too - guards against
    exactly the transient-dip-then-recovery artifact already documented in
    RAW_DATA.md's H2-K correction (h2_sequential_llama.py's own tstar tracker
    used a first-crossing-wins rule with no such check, and mistook one noisy
    step for a permanent forgetting event there; final accuracy had actually
    recovered). Naive first-crossing logic would reproduce that exact bug.
    """
    n = len(trajectory)
    forgot_flags = []
    for pt in trajectory:
        val = pt[key]
        forgot_flags.append((val > threshold) if above_is_forgotten else (val < threshold))

    first_forgot_idx = None
    for i in range(n):
        if all(forgot_flags[i:]):
            first_forgot_idx = i
            break

    if first_forgot_idx is None:
        return trajectory[-1]["R_A"], None  # never permanently forgot
    if first_forgot_idx == 0:
        return None, trajectory[0]["R_A"]   # forgotten from the first measured step
    return trajectory[first_forgot_idx - 1]["R_A"], trajectory[first_forgot_idx]["R_A"]


def run_phase1_generic(args, device, out_dir):
    """
    Minimal Phase 1 (Task A) trainer, architecture-generic via _lora_targets
    (GPT-2's c_attn/c_proj vs Llama/Mistral's q/k/v/o_proj) - see module
    docstring for why this exists instead of importing h2_sequential_llama.py's
    own run_phase1, which hardcodes the Llama-style names.

    Returns (model, classifier, phase1_params_snapshot, sigma_half_a,
    n_classes_a, acc_phase1_a, tokenizer) - same shape as the function it
    replaces, so the rest of this file's Phase 2 logic doesn't need to know
    which one produced it.
    """
    model_path = os.path.join(out_dir, "phase1_model.pt")
    cls_path   = os.path.join(out_dir, "phase1_classifier.pt")
    meta_path  = os.path.join(out_dir, "phase1_meta.json")

    print(f"\n{'='*60}\n  Phase 1: {args.task_a} on {args.model}\n{'='*60}")

    tokenizer = _get_tokenizer(args.model)
    model, _  = _load_model_auto(args.model, device)
    target_names = _get_target_names_auto(model, args.model)
    hidden_size   = _get_hidden_size(model)
    target_modules = _lora_targets(args.model)

    if args.skip_phase1 and os.path.exists(model_path):
        print("  Loading Phase 1 from checkpoint...")
        with open(meta_path) as f:
            meta = json.load(f)
        n_classes_a  = meta["n_classes"]
        acc_phase1_a = meta["acc_phase1"]
        classifier   = nn.Linear(hidden_size, n_classes_a).to(device)
        peft_m = add_lora(model, meta["rank"], meta["alpha"], target_modules=target_modules)
        peft_m.load_state_dict(torch.load(model_path, map_location=device), strict=False)
        model = peft_m.merge_and_unload().eval()
        classifier.load_state_dict(torch.load(cls_path, map_location=device))
        classifier.eval()
    else:
        torch.manual_seed(args.seed)
        train_loader, n_classes_a = load_task(
            tokenizer, args.task_a, "train", args.seq_len, args.batch_size, seed=args.seed,
        )
        classifier = nn.Linear(hidden_size, n_classes_a).to(device)
        peft_model = add_lora(model, args.phase1_rank, args.phase1_alpha,
                              target_modules=target_modules)
        params  = list(peft_model.parameters()) + list(classifier.parameters())
        opt     = torch.optim.AdamW([p for p in params if p.requires_grad], lr=args.phase1_lr)
        loss_fn = nn.CrossEntropyLoss()

        peft_model.train()
        classifier.train()
        step, t0 = 0, time.time()
        while step < args.phase1_steps:
            for input_ids, labels in train_loader:
                if step >= args.phase1_steps:
                    break
                input_ids, labels = input_ids.to(device), labels.to(device)
                out    = peft_model(input_ids, output_hidden_states=True)
                hidden = out.hidden_states[-1][:, -1, :].float()
                loss   = loss_fn(classifier(hidden), labels)
                if torch.isnan(loss):
                    step += 1
                    continue
                opt.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(params, args.grad_clip)
                opt.step()
                step += 1
                if step % 200 == 0:
                    print(f"  Phase1 step {step}/{args.phase1_steps}  loss={loss.item():.4f}  "
                          f"elapsed={time.time()-t0:.0f}s", flush=True)

        lora_sd = {k: v.cpu() for k, v in peft_model.state_dict().items() if "lora_" in k}
        torch.save(lora_sd, model_path)
        torch.save(classifier.state_dict(), cls_path)
        model = peft_model.merge_and_unload().eval()
        classifier.eval()

        val_loader, _ = load_task(
            tokenizer, args.task_a, "validation", args.seq_len, args.batch_size,
            n_samples=args.eval_samples, seed=args.seed,
        )
        acc_phase1_a = eval_accuracy(model, classifier, val_loader, device)
        print(f"  Phase 1 {args.task_a} accuracy: {acc_phase1_a:.4f}")
        if acc_phase1_a < 0.80:
            print(f"  WARNING: Phase 1 accuracy {acc_phase1_a:.2%} < 80%. "
                  f"Consider more --phase1_steps or a different --phase1_lr.")

        with open(meta_path, "w") as f:
            json.dump({
                "task_a": args.task_a, "n_classes": n_classes_a,
                "acc_phase1": acc_phase1_a, "lr": args.phase1_lr,
                "rank": args.phase1_rank, "alpha": args.phase1_alpha,
                "steps": args.phase1_steps, "lora_target_modules": target_modules,
            }, f, indent=2)

    phase1_params = {n: p.data.clone().cpu() for n, p in model.named_parameters()}

    holdout_ids, _ = load_task(
        tokenizer, args.task_a, "validation", args.seq_len, batch_size=1,
        n_samples=args.sigma_holdout, seed=args.seed + 1, for_density=True,
    )
    sigma_half_a = compute_sigma_half(
        model, target_names, holdout_ids, device, args,
        out_path=os.path.join(out_dir, "phase1_sigma_half.json"),
    )
    return model, classifier, phase1_params, sigma_half_a, n_classes_a, acc_phase1_a, tokenizer


def run_phase2_condition_dual(phase1_model, phase1_classifier, phase1_params,
                               sigma_half_a, acc_phase1_a, nll_a_baseline,
                               task_a, task_b, tokenizer, model_name, lr, rank, steps,
                               track_interval, batch_size, grad_clip,
                               eval_samples, seq_len, device, seed, out_dir,
                               nll_holdout_ids, acc_slack, nll_relslack, force):
    """
    Fine-tune on Task B. At each track_interval step, record BOTH:
      - acc_A via the frozen Phase-1 classifier (existing H2 signal)
      - NLL_A on held-out Task A text, no classifier involved (new signal)
    and compute a T* bracket for each independently, so they can be compared
    directly for the same condition.
    """
    tag         = f"lr{lr:.0e}_rank{rank}"
    traj_path   = os.path.join(out_dir, f"{tag}_trajectory.json")
    result_path = os.path.join(out_dir, f"{tag}_result.json")

    if os.path.exists(result_path) and not force:
        print(f"  [{tag}] Cached — loading result")
        with open(result_path) as f:
            return json.load(f)

    print(f"\n{'─'*50}\n  Phase 2 [{tag}]  {task_a}→{task_b}")
    t0 = time.time()
    torch.manual_seed(seed)

    total_params_full = sum(p.numel() for p in phase1_params.values())
    model = copy.deepcopy(phase1_model)

    train_loader_b, n_classes_b = load_task(
        tokenizer, task_b, "train", seq_len, batch_size, seed=seed,
    )
    val_loader_a, _ = load_task(
        tokenizer, task_a, "validation", seq_len, batch_size,
        n_samples=eval_samples, seed=seed,
    )

    hidden_size  = _get_hidden_size(model)
    classifier_b = nn.Linear(hidden_size, n_classes_b).to(device)

    _TASK_SEED_OFFSET = {"sst2": 0, "mnli": 1000, "agnews": 2000, "cola": 3000}
    torch.manual_seed(seed + _TASK_SEED_OFFSET.get(task_b, 9999))
    target_modules = _lora_targets(model_name)
    peft_model = add_lora(model, rank, rank * 2, target_modules=target_modules)
    params     = list(peft_model.parameters()) + list(classifier_b.parameters())
    opt        = torch.optim.AdamW([p for p in params if p.requires_grad], lr=lr)
    loss_fn    = nn.CrossEntropyLoss()

    acc_threshold = (1.0 - acc_slack) * acc_phase1_a
    nll_threshold = (1.0 + nll_relslack) * nll_a_baseline

    peft_model.train()
    classifier_b.train()
    trajectory = []
    step = 0

    while step < steps:
        for input_ids, labels in train_loader_b:
            if step >= steps:
                break

            if step % track_interval == 0:
                per_param_norm = compute_lora_delta_norm(peft_model, total_params_full)
                R_a = per_param_norm / sigma_half_a if sigma_half_a > 0 else float("nan")

                peft_model.eval()
                acc_a = eval_accuracy(peft_model, phase1_classifier, val_loader_a, device)
                nll_a = compute_task_nll(peft_model, nll_holdout_ids, device)
                peft_model.train()
                classifier_b.train()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

                trajectory.append({
                    "step": step, "R_A": R_a, "per_param_norm": per_param_norm,
                    "acc_task_a": acc_a, "nll_task_a": nll_a,
                    "delta_nll_task_a": nll_a - nll_a_baseline,
                })
                print(f"  [{tag}] step={step:4d}  R_A={R_a:.4f}  "
                      f"acc_A={acc_a:.4f}  nll_A={nll_a:.4f} "
                      f"(Δ{nll_a - nll_a_baseline:+.4f})", flush=True)

            input_ids = input_ids.to(device)
            labels    = labels.to(device)
            out    = peft_model(input_ids, output_hidden_states=True)
            hidden = out.hidden_states[-1][:, -1, :].float()
            loss   = loss_fn(classifier_b(hidden), labels)
            if not torch.isnan(loss):
                opt.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(params, grad_clip)
                opt.step()
            step += 1

    tstar_acc = _bracket(trajectory, "acc_task_a", acc_threshold, above_is_forgotten=False)
    tstar_nll = _bracket(trajectory, "delta_nll_task_a",
                         nll_threshold - nll_a_baseline, above_is_forgotten=True)

    elapsed = time.time() - t0
    result = {
        "task_a": task_a, "task_b": task_b, "lr": lr, "rank": rank, "steps": steps,
        "acc_phase1_a": acc_phase1_a, "nll_a_baseline": nll_a_baseline,
        "acc_threshold": acc_threshold, "nll_threshold": nll_threshold,
        "tstar_acc_R_A": {"last_safe": tstar_acc[0], "first_forgot": tstar_acc[1]},
        "tstar_nll_R_A": {"last_safe": tstar_nll[0], "first_forgot": tstar_nll[1]},
        "trajectory": trajectory, "elapsed_s": elapsed,
    }
    with open(result_path, "w") as f:
        json.dump(result, f, indent=2)
    with open(traj_path, "w") as f:
        json.dump(trajectory, f, indent=2)
    print(f"  [{tag}] DONE  T*_acc={tstar_acc}  T*_nll={tstar_nll}  ({elapsed:.0f}s)")
    return result


def plot_comparison(all_results, out_path):
    """Per task_b: T*_acc vs T*_nll midpoint (of the last-safe/first-forgot
    bracket), across the swept task_b targets - the actual test of whether
    NLL narrows the spread accuracy shows across Phase-2 targets."""
    by_task_b = {}
    for r in all_results:
        tb = r["task_b"]
        acc_b = r["tstar_acc_R_A"]
        nll_b = r["tstar_nll_R_A"]
        if acc_b["last_safe"] is not None and acc_b["first_forgot"] is not None:
            acc_mid = 0.5 * (acc_b["last_safe"] + acc_b["first_forgot"])
        else:
            acc_mid = float("nan")
        if nll_b["last_safe"] is not None and nll_b["first_forgot"] is not None:
            nll_mid = 0.5 * (nll_b["last_safe"] + nll_b["first_forgot"])
        else:
            nll_mid = float("nan")
        by_task_b.setdefault(tb, []).append((acc_mid, nll_mid))

    task_bs = sorted(by_task_b)
    acc_mids = [np.nanmean([v[0] for v in by_task_b[tb]]) for tb in task_bs]
    nll_mids = [np.nanmean([v[1] for v in by_task_b[tb]]) for tb in task_bs]

    fig, ax = plt.subplots(figsize=(7, 5))
    x = np.arange(len(task_bs))
    width = 0.35
    ax.bar(x - width/2, acc_mids, width, color=CATEGORICAL["acc"], label="T* (accuracy-based)")
    ax.bar(x + width/2, nll_mids, width, color=CATEGORICAL["nll"], label="T* (NLL-based)")
    ax.set_yscale("log")
    ax.set_xticks(x)
    ax.set_xticklabels(task_bs)
    ax.set_ylabel("T* midpoint (R_A, log scale)")
    ax.set_title("SST-2 Phase 1 forgetting threshold: accuracy vs NLL readout\nacross Phase-2 targets")
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved figure: {out_path}")

    def _spread(vals):
        vals = [v for v in vals if not np.isnan(v) and v > 0]
        return (max(vals) / min(vals)) if len(vals) >= 2 else float("nan")

    print(f"\n  Spread across task_b targets — accuracy-based T*: {_spread(acc_mids):.1f}x"
          f"   NLL-based T*: {_spread(nll_mids):.1f}x")


def main():
    args   = get_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(args.seed)

    model_tag = args.model.replace("/", "_")
    base_out  = os.path.join(args.output_dir, model_tag)
    os.makedirs(base_out, exist_ok=True)

    print(f"\n{'='*60}\n  H2 NLL-vs-Accuracy Forgetting — {args.model}\n"
          f"  {args.task_a} -> {args.task_b}\n{'='*60}\n")

    # Phase 1 (task_a) - via this file's own run_phase1_generic (see module
    # docstring for why h2_sequential_llama.py's run_phase1 isn't reused here).
    # Out_dir convention: {output_dir}/{model_tag}/{task_a}_to_{task_b}; since
    # Phase 1 doesn't depend on task_b, we use the FIRST task_b entry's dir so
    # one Phase 1 checkpoint is reused across this script's own task_b sweep
    # via --skip_phase1 (this is a fresh checkpoint format, not byte-compatible
    # with h2_sequential_llama.py's own phase1_meta.json — see run_phase1_generic).
    phase1_dir = os.path.join(base_out, f"{args.task_a}_to_{args.task_b[0]}")
    os.makedirs(phase1_dir, exist_ok=True)
    (phase1_model, phase1_classifier, phase1_params,
     sigma_half_a, n_classes_a, acc_phase1_a, tokenizer) = run_phase1_generic(args, device, phase1_dir)

    # Held-out Task A text for NLL tracking - separate draw (different seed
    # offset) from the sigma_half_A calibration holdout, so forgetting isn't
    # measured on the exact text used to calibrate the denominator.
    nll_holdout_ids, _ = load_task(
        tokenizer, args.task_a, "validation", args.seq_len, batch_size=1,
        n_samples=args.nll_holdout, seed=args.seed + 7, for_density=True,
    )
    nll_a_baseline = compute_task_nll(phase1_model, nll_holdout_ids, device)
    print(f"  Phase 1 {args.task_a} NLL baseline: {nll_a_baseline:.4f}  "
          f"(forgetting threshold: Δ > {args.nll_relslack*100:.0f}% = "
          f"{args.nll_relslack * nll_a_baseline:.4f})")

    all_results = []
    for task_b in args.task_b:
        cond_dir = os.path.join(base_out, f"{args.task_a}_to_{task_b}")
        os.makedirs(cond_dir, exist_ok=True)
        for lr in args.phase2_lrs:
            result = run_phase2_condition_dual(
                phase1_model=phase1_model, phase1_classifier=phase1_classifier,
                phase1_params=phase1_params, sigma_half_a=sigma_half_a,
                acc_phase1_a=acc_phase1_a, nll_a_baseline=nll_a_baseline,
                task_a=args.task_a, task_b=task_b, tokenizer=tokenizer,
                model_name=args.model, lr=lr, rank=args.phase2_rank, steps=args.phase2_steps,
                track_interval=args.track_interval, batch_size=args.batch_size,
                grad_clip=args.grad_clip, eval_samples=args.eval_samples,
                seq_len=args.seq_len, device=device, seed=args.seed,
                out_dir=cond_dir, nll_holdout_ids=nll_holdout_ids,
                acc_slack=args.acc_slack, nll_relslack=args.nll_relslack,
                force=args.force,
            )
            all_results.append(result)

    summary_path = os.path.join(base_out, f"{args.task_a}_summary.json")
    with open(summary_path, "w") as f:
        json.dump(all_results, f, indent=2)

    plot_comparison(all_results, os.path.join(base_out, "nll_vs_acc_forgetting.png"))

    print(f"\n{'─'*70}\n{'task_b':>10}  {'lr':>8}  {'T*_acc (last,first)':>22}  "
          f"{'T*_nll (last,first)':>22}\n{'─'*70}")
    for r in all_results:
        a, n = r["tstar_acc_R_A"], r["tstar_nll_R_A"]
        a_s = f"({a['last_safe']},{a['first_forgot']})" if a["first_forgot"] else "(never forgot)"
        n_s = f"({n['last_safe']},{n['first_forgot']})" if n["first_forgot"] else "(never forgot)"
        print(f"  {r['task_b']:>10}  {r['lr']:>8.0e}  {a_s:>22}  {n_s:>22}")
    print(f"{'─'*70}\nSummary saved: {summary_path}\n")


if __name__ == "__main__":
    main()
