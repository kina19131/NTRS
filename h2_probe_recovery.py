"""
h2_probe_recovery.py
=====================
Does NOT modify h2_sequential_llama.py or h2_nll_forgetting.py — imports
shared utilities from both unchanged, and reuses h2_nll_forgetting.py's own
run_phase1_generic + saved Phase-1 checkpoint (via --skip_phase1, forced on)
so this experiment starts from the EXACT SAME Phase-1 SST-2 checkpoint
already used for the NLL-vs-accuracy comparison, for direct comparability.

Motivation (2026-07-22): h2_nll_forgetting.py tested whether the SST-2
anomaly (V8: T*(r8) spreads 44x across Phase-2 targets under the
accuracy-based signal) is a classifier-readout artifact, by tracking NLL on
Task A's own held-out text instead of accuracy. Result: NLL never crossed
its forgetting threshold anywhere, but the raw trajectories showed NLL
IMPROVING substantially (35-40%) in exactly the conditions where accuracy
collapsed to chance - and the magnitude of that improvement scaled with the
magnitude of the accuracy collapse (near-flat for the condition that stayed
safe, largest for the conditions that collapsed hardest).

That magnitude-correlation is genuinely ambiguous between two very
different stories:
  (a) Readout-mismatch: the representation still encodes sentiment, just not
      along the exact direction the OLD frozen classifier reads from. NLL
      improving simply reflects generic fluency gains from more training,
      unrelated to sentiment being lost.
  (b) Mode collapse: aggressive Phase-2 training pushes the model toward
      degenerate/repetitive output. Collapsed, homogenized text can score
      artificially LOW perplexity on short generic-looking sentences (like
      SST-2's) without any useful information - including sentiment - being
      preserved. This would produce the exact same signature: NLL improves,
      accuracy craters, and the two effects' magnitudes track each other,
      because both are downstream of the same aggressive-optimization cause.

NLL-on-domain-text cannot distinguish these, because it never actually asks
whether sentiment is still linearly decodable - it asks whether the model is
fluent on SST-2-style text, which is an adjacent but different question.

This script asks the actual question directly: freeze the Phase-2-perturbed
backbone (and its Phase-2 LoRA weights) at a given condition's endpoint,
train a FRESH linear probe from scratch on top of its current hidden states
(same architecture as the original Phase-1 classifier: nn.Linear(hidden,
n_classes)), and measure how fast/well it recovers Task-A (SST-2) accuracy
using only the frozen representation.
  - Fast, low-budget recovery (fresh probe reaches ~baseline accuracy in a
    handful of steps) => sentiment information is still linearly present;
    supports readout-mismatch (a). The OLD classifier's failure was a
    readout artifact, not evidence of forgetting.
  - No better than the old frozen classifier, even with a full probe-training
    budget => the information itself is genuinely gone; supports (b) or some
    other mechanism. H2's original accuracy-based measurement was right, and
    the SST-2 anomaly (V8) needs a different explanation.

Decision rule, fixed BEFORE running (not tuned post-hoc on the result):
"recovers_fast"   = probe crosses 0.9 * acc_phase1_a within <=25 probe steps
"recovers_slow"   = crosses between 25 and --probe_steps (default 200) steps
"no_recovery"     = never crosses within the probe-training budget
These thresholds are arbitrary but declared here, in advance, precisely so
the classification isn't chosen after seeing which story it supports.

Cheap validation on GPT-2 first, same 3 already-documented conditions
(sst2->mnli/agnews/cola, lr=5e-4) showing the NLL-vs-accuracy divergence -
before spending GPU-hours on Llama, per this project's established practice.

Output:
  results/h2_probe_recovery/{model_tag}/{task_a}_to_{task_b}/
    lr{X}_rank{Y}_result.json   - old_frozen_acc, recovery_curve, verdict
  probe_recovery.png            - old frozen acc vs recovered probe acc, per task_b

Usage:
  # Requires an existing h2_nll_forgetting.py Phase-1 checkpoint to reuse:
  #   results/h2_nll_forgetting/gpt2/sst2_to_mnli/{phase1_model.pt,phase1_classifier.pt,phase1_meta.json}
  python h2_probe_recovery.py --model gpt2 --task_a sst2 --task_b mnli agnews cola \\
      --phase1_dir results/h2_nll_forgetting/gpt2/sst2_to_mnli
"""

import argparse
import copy
import json
import os
import shutil
import sys
import time

import numpy as np
import torch
import torch.nn as nn
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(__file__))
from lora_density_experiment import _get_hidden_size, add_lora
from h1_kl_comparison import _lora_targets
from h2_sequential_llama import load_task, eval_accuracy
from h2_nll_forgetting import run_phase1_generic, CATEGORICAL as _NLL_CATEGORICAL

CATEGORICAL = dict(_NLL_CATEGORICAL)
CATEGORICAL["probe"] = CATEGORICAL["gpt2"]  # reuse validated palette green, distinct from acc/nll

_TASK_SEED_OFFSET = {"sst2": 0, "mnli": 1000, "agnews": 2000, "cola": 3000}


def get_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model",  default="gpt2")
    p.add_argument("--task_a", default="sst2", choices=["mnli", "sst2", "agnews", "cola"])
    p.add_argument("--task_b", nargs="+", default=["mnli", "agnews", "cola"],
                   choices=["mnli", "sst2", "agnews", "cola"])

    p.add_argument("--phase1_dir", required=True,
                   help="Existing h2_nll_forgetting.py output dir containing "
                        "phase1_model.pt, phase1_classifier.pt, phase1_meta.json "
                        "to reuse unchanged (this script never trains Phase 1 itself). "
                        "Only read from - checkpoints are copied into this script's "
                        "own output dir before loading, so run_phase1_generic's "
                        "internal sigma_half_a recompute never writes back into it.")

    # Required by run_phase1_generic internally (sigma_half_a recompute after
    # loading) even though this script doesn't use sigma_half_a itself -
    # same defaults as h2_nll_forgetting.py so the recompute is a no-op check.
    p.add_argument("--n_perturb",     type=int,   default=200)
    p.add_argument("--sigma_holdout", type=int,   default=256)
    p.add_argument("--eval_slack",    type=float, default=1e-4)

    # Phase 2 (must match the h2_nll_forgetting.py condition being probed,
    # for the old_frozen_acc comparison to mean anything)
    p.add_argument("--phase2_lrs",  nargs="+", type=float, default=[5e-4])
    p.add_argument("--phase2_rank", type=int, default=8)
    p.add_argument("--phase2_steps", type=int, default=500)

    # Fresh probe recovery
    p.add_argument("--probe_lr",    type=float, default=1e-3)
    p.add_argument("--probe_steps", type=int,   default=200)
    p.add_argument("--probe_eval_interval", type=int, default=5,
                   help="Must divide evenly into --probe_fast_recovery_steps "
                        "(default 25) so the fast/slow boundary is checked "
                        "exactly, not rounded up to the next eval point.")
    p.add_argument("--probe_fast_recovery_steps", type=int, default=25,
                   help="Pre-registered cutoff for 'recovers_fast' verdict.")

    p.add_argument("--seq_len",      type=int, default=128)
    p.add_argument("--batch_size",   type=int, default=16)
    p.add_argument("--grad_clip",    type=float, default=1.0)
    p.add_argument("--eval_samples", type=int, default=500)
    p.add_argument("--seed",         type=int, default=42)
    p.add_argument("--output_dir",   default="./results/h2_probe_recovery")
    p.add_argument("--force",        action="store_true")
    return p.parse_args()


def train_phase2(phase1_model, phase1_classifier, tokenizer, model_name,
                  task_a, task_b, lr, rank, steps, seq_len, batch_size,
                  grad_clip, eval_samples, device, seed):
    """
    Reproduces h2_nll_forgetting.py's run_phase2_condition_dual training loop
    (same seed-offset scheme, same target modules, same optimizer/loss) but
    without the per-step dual tracking overhead - this script only needs the
    FINAL trained state to hand off to probe recovery. Returns the trained
    peft_model (Phase-2 LoRA active, Task-B classifier discarded) and the
    old_frozen_acc measured with the original Phase-1 classifier, so the
    comparison point is computed fresh here rather than assumed to match a
    separately-run h2_nll_forgetting.py trajectory exactly.
    """
    torch.manual_seed(seed)
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

    torch.manual_seed(seed + _TASK_SEED_OFFSET.get(task_b, 9999))
    target_modules = _lora_targets(model_name)
    peft_model = add_lora(model, rank, rank * 2, target_modules=target_modules)
    params  = list(peft_model.parameters()) + list(classifier_b.parameters())
    opt     = torch.optim.AdamW([p for p in params if p.requires_grad], lr=lr)
    loss_fn = nn.CrossEntropyLoss()

    peft_model.train()
    classifier_b.train()
    step, t0 = 0, time.time()
    while step < steps:
        for input_ids, labels in train_loader_b:
            if step >= steps:
                break
            input_ids, labels = input_ids.to(device), labels.to(device)
            out    = peft_model(input_ids, output_hidden_states=True)
            hidden = out.hidden_states[-1][:, -1, :].float()
            loss   = loss_fn(classifier_b(hidden), labels)
            if not torch.isnan(loss):
                opt.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(params, grad_clip)
                opt.step()
            step += 1
    print(f"    Phase 2 [{task_a}->{task_b}, lr={lr:.0e}] trained {steps} steps "
          f"in {time.time()-t0:.0f}s", flush=True)

    peft_model.eval()
    old_frozen_acc = eval_accuracy(peft_model, phase1_classifier, val_loader_a, device)
    peft_model.train()
    return peft_model, old_frozen_acc, hidden_size


def run_probe_recovery(peft_model, tokenizer, task_a, n_classes_a, hidden_size,
                        device, seq_len, batch_size, eval_samples, seed,
                        probe_lr, probe_steps, probe_eval_interval, grad_clip):
    """
    Freezes the current (Phase-2-perturbed) model entirely, attaches a FRESH
    randomly-initialized linear probe, and trains only the probe on Task A -
    the backbone (base weights + Phase-2 LoRA) never updates again. Records
    an accuracy-vs-probe-step recovery curve, including step 0 (untrained
    probe, a sanity floor near 1/n_classes_a).
    """
    for p in peft_model.parameters():
        p.requires_grad = False
    peft_model.eval()

    fresh_probe = nn.Linear(hidden_size, n_classes_a).to(device)
    opt = torch.optim.AdamW(fresh_probe.parameters(), lr=probe_lr)
    loss_fn = nn.CrossEntropyLoss()

    train_loader, _ = load_task(
        tokenizer, task_a, "train", seq_len, batch_size, seed=seed + 5000,
    )
    val_loader, _ = load_task(
        tokenizer, task_a, "validation", seq_len, batch_size,
        n_samples=eval_samples, seed=seed,
    )

    def _probe_acc():
        fresh_probe.eval()
        acc = eval_accuracy(peft_model, fresh_probe, val_loader, device)
        fresh_probe.train()
        return acc

    recovery_curve = [{"probe_step": 0, "probe_acc": _probe_acc()}]
    fresh_probe.train()
    step = 0
    while step < probe_steps:
        for input_ids, labels in train_loader:
            if step >= probe_steps:
                break
            input_ids, labels = input_ids.to(device), labels.to(device)
            with torch.no_grad():
                out    = peft_model(input_ids, output_hidden_states=True)
                hidden = out.hidden_states[-1][:, -1, :].float()
            loss = loss_fn(fresh_probe(hidden), labels)
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(fresh_probe.parameters(), grad_clip)
            opt.step()
            step += 1
            if step % probe_eval_interval == 0:
                recovery_curve.append({"probe_step": step, "probe_acc": _probe_acc()})

    return recovery_curve


def _verdict(recovery_curve, acc_phase1_a, fast_cutoff):
    """
    Same sustained-crossing convention as h2_nll_forgetting.py's _bracket -
    a step only counts as genuine recovery if probe_acc stays at/above
    threshold for every subsequent measured point too. Guards against a
    single noisy uptick early in probe training (small-batch SGD on a
    freshly-initialized linear layer is exactly the kind of thing that
    produces transient blips) being misread as real recovery - the same
    transient-dip artifact already documented and fixed once in H2-K and
    again via _bracket; naive first-crossing logic would reproduce it here.
    """
    threshold = 0.9 * acc_phase1_a
    n = len(recovery_curve)
    crossing_step = None
    for i in range(n):
        if all(pt["probe_acc"] >= threshold for pt in recovery_curve[i:]):
            crossing_step = recovery_curve[i]["probe_step"]
            break
    if crossing_step is None:
        return "no_recovery", None
    if crossing_step <= fast_cutoff:
        return "recovers_fast", crossing_step
    return "recovers_slow", crossing_step


def plot_comparison(all_results, out_path):
    task_bs = sorted({r["task_b"] for r in all_results})
    old_accs, final_probe_accs = [], []
    for tb in task_bs:
        rs = [r for r in all_results if r["task_b"] == tb]
        old_accs.append(np.mean([r["old_frozen_acc"] for r in rs]))
        final_probe_accs.append(np.mean([r["recovery_curve"][-1]["probe_acc"] for r in rs]))

    fig, ax = plt.subplots(figsize=(7, 5))
    x = np.arange(len(task_bs))
    width = 0.35
    ax.bar(x - width/2, old_accs, width, color=CATEGORICAL["acc"], label="Old frozen classifier")
    ax.bar(x + width/2, final_probe_accs, width, color=CATEGORICAL["probe"], label="Fresh probe (recovered)")
    ax.set_xticks(x)
    ax.set_xticklabels(task_bs)
    ax.set_ylabel("Task A (SST-2) accuracy")
    ax.set_ylim(0, 1)
    ax.set_title("Old frozen classifier vs. fresh linear probe\non Phase-2-perturbed representations")
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved figure: {out_path}")


def main():
    args   = get_args()
    if args.probe_fast_recovery_steps % args.probe_eval_interval != 0:
        print(f"  WARNING: --probe_fast_recovery_steps ({args.probe_fast_recovery_steps}) is not "
              f"a multiple of --probe_eval_interval ({args.probe_eval_interval}) - the fast/slow "
              f"verdict boundary will be rounded up to the next eval point, not checked exactly.")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(args.seed)

    model_tag = args.model.replace("/", "_")
    base_out  = os.path.join(args.output_dir, model_tag)
    os.makedirs(base_out, exist_ok=True)

    print(f"\n{'='*60}\n  H2 Fresh-Probe Recovery — {args.model}\n"
          f"  {args.task_a} -> {args.task_b}\n{'='*60}\n")

    # Reuse the EXACT Phase-1 checkpoint h2_nll_forgetting.py already trained -
    # never retrained here, so this is on identical footing to that experiment.
    # Copied into this script's own output dir first (not loaded in place)
    # so run_phase1_generic's internal sigma_half_a recompute never writes
    # back into the original h2_nll_forgetting.py results directory.
    phase1_local_dir = os.path.join(base_out, f"{args.task_a}_phase1_checkpoint")
    os.makedirs(phase1_local_dir, exist_ok=True)
    for fname in ("phase1_model.pt", "phase1_classifier.pt", "phase1_meta.json"):
        dst = os.path.join(phase1_local_dir, fname)
        if not os.path.exists(dst):
            shutil.copy2(os.path.join(args.phase1_dir, fname), dst)

    phase1_args = copy.deepcopy(args)
    phase1_args.skip_phase1 = True
    (phase1_model, phase1_classifier, _phase1_params,
     _sigma_half_a, n_classes_a, acc_phase1_a, tokenizer) = run_phase1_generic(
        phase1_args, device, phase1_local_dir)
    print(f"  Reused Phase 1 checkpoint: {args.task_a} acc={acc_phase1_a:.4f} "
          f"(n_classes={n_classes_a})")

    all_results = []
    for task_b in args.task_b:
        for lr in args.phase2_lrs:
            tag = f"lr{lr:.0e}_rank{args.phase2_rank}"
            cond_dir = os.path.join(base_out, f"{args.task_a}_to_{task_b}")
            os.makedirs(cond_dir, exist_ok=True)
            result_path = os.path.join(cond_dir, f"{tag}_result.json")

            if os.path.exists(result_path) and not args.force:
                print(f"  [{task_b}/{tag}] Cached — loading result")
                with open(result_path) as f:
                    all_results.append(json.load(f))
                continue

            print(f"\n{'─'*50}\n  [{task_b}/{tag}]")
            peft_model, old_frozen_acc, hidden_size = train_phase2(
                phase1_model, phase1_classifier, tokenizer, args.model,
                args.task_a, task_b, lr, args.phase2_rank, args.phase2_steps,
                args.seq_len, args.batch_size, args.grad_clip, args.eval_samples,
                device, args.seed,
            )
            print(f"  [{task_b}/{tag}] old_frozen_acc={old_frozen_acc:.4f} "
                  f"(phase1 baseline={acc_phase1_a:.4f})")

            recovery_curve = run_probe_recovery(
                peft_model, tokenizer, args.task_a, n_classes_a, hidden_size,
                device, args.seq_len, args.batch_size, args.eval_samples, args.seed,
                args.probe_lr, args.probe_steps, args.probe_eval_interval, args.grad_clip,
            )
            verdict, crossing_step = _verdict(
                recovery_curve, acc_phase1_a, args.probe_fast_recovery_steps)

            result = {
                "task_a": args.task_a, "task_b": task_b, "lr": lr,
                "rank": args.phase2_rank, "steps": args.phase2_steps,
                "acc_phase1_a": acc_phase1_a, "old_frozen_acc": old_frozen_acc,
                "recovery_curve": recovery_curve,
                "final_probe_acc": recovery_curve[-1]["probe_acc"],
                "verdict": verdict, "crossing_step": crossing_step,
            }
            with open(result_path, "w") as f:
                json.dump(result, f, indent=2)
            all_results.append(result)
            print(f"  [{task_b}/{tag}] final_probe_acc={result['final_probe_acc']:.4f}  "
                  f"verdict={verdict}  (crossed 90%-of-baseline at step {crossing_step})")

    summary_path = os.path.join(base_out, f"{args.task_a}_probe_summary.json")
    with open(summary_path, "w") as f:
        json.dump(all_results, f, indent=2)
    plot_comparison(all_results, os.path.join(base_out, "probe_recovery.png"))

    print(f"\n{'─'*70}\n{'task_b':>10}  {'lr':>8}  {'old_acc':>8}  "
          f"{'probe_acc':>10}  {'verdict':>14}\n{'─'*70}")
    for r in all_results:
        print(f"  {r['task_b']:>10}  {r['lr']:>8.0e}  {r['old_frozen_acc']:>8.4f}  "
              f"{r['final_probe_acc']:>10.4f}  {r['verdict']:>14}")
    print(f"{'─'*70}\nSummary saved: {summary_path}\n")


if __name__ == "__main__":
    main()
