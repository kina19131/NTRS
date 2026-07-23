"""
h1_direct_r_sweep.py
=====================
Direct-R-targeting variant of h1_kl_comparison.py — does NOT modify, delete, or
replace that script; imports its shared utilities and adds a new experiment.

Motivation (2026-07-20 analysis): h1_kl_comparison.py sweeps LR and observes
whatever R that happens to produce. Checked against real data, the LR->R
mapping is not linear — for Llama SST-2, per_param_norm/LR ("gain") is flat
around 1.8-2.3 for the first four LR points, then jumps to 5.6 at the last
point, exactly where training destabilizes. That means an LR grid that looks
evenly spaced is NOT evenly spaced in R, and is specifically undersampled near
the transition — the one region T* estimation most needs resolution in.

This script removes that confound by controlling R directly:
  1. Train ONE reference LoRA fine-tune (moderate-high LR, enough steps to
     reach/exceed the top of the R range of interest) to get a fixed
     direction (A, B per layer).
  2. Since DeltaW = scaling * B @ A is exactly linear in B, rescale B alone
     to hit any target per_param_norm exactly - no retraining, no search.
  3. Evaluate directly on the (still-unmerged) PEFT model at each target R -
     PEFT forward passes are already correct without merging, which is how
     training itself works in the original script.
  4. Overlay any existing lr*_results.json from h1_kl_comparison.py's own
     output dir as real-training cross-check points, since the rescaled
     direction only tests magnitude along ONE direction, not the family of
     directions different LRs would have produced - the two should roughly
     agree if that approximation is trustworthy.

Output:
  results/h1_direct_r/{model}/
    r{X}_results.json     - per-target-R metrics
    summary.json          - all conditions + cross-check points
    direct_r_comparison.png

Usage:
  python h1_direct_r_sweep.py --model gpt2
  python h1_direct_r_sweep.py --model meta-llama/Llama-3.2-3B --ref_lr 5e-4 --ref_steps 800
  python h1_direct_r_sweep.py --model gpt2 --replot
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
from certified_density_experiment import (
    load_data, compute_nll_loss, estimate_density, half_max_sigma,
)
from lora_density_experiment import (
    load_sst2, add_lora, _load_model_auto, _get_target_names_auto,
    _get_hidden_size, _get_tokenizer,
)
# Reuse H1's existing, already-validated eval/KL machinery rather than
# duplicating it - any future fix to these in h1_kl_comparison.py is
# inherited automatically, and nothing there is modified by importing it.
from h1_kl_comparison import (
    DIVERSE_PROMPTS, _lora_targets, compute_kl, encode_diverse_prompts,
    load_superni_prompts, eval_mmlu_accuracy, load_mnli, load_wikitext_for_kl,
    eval_sst2_accuracy, eval_mnli_accuracy,
)


def get_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="gpt2")
    p.add_argument("--task",  default="sst2", choices=["sst2", "mnli"])
    p.add_argument("--rank",  type=int, default=8)
    p.add_argument("--alpha", type=int, default=16)
    p.add_argument("--ref_lr",    type=float, default=5e-4,
                   help="LR for the ONE reference training run whose trajectory gets "
                        "checkpointed. Should be high enough that ref_steps reaches/exceeds "
                        "the top of --target_rs, so every target anchors at a real checkpoint "
                        "(no target needs to extrapolate beyond the whole trajectory).")
    p.add_argument("--ref_steps", type=int, default=800)
    p.add_argument("--checkpoint_interval", type=int, default=50,
                   help="Save a full LoRA-B snapshot every N training steps during the "
                        "reference run - ALL retained, not just the best/final one. Each "
                        "target R anchors at whichever checkpoint's own per_param_norm is "
                        "closest to it (small local rescale from a real point), rather than "
                        "one large rescale from a single distant endpoint. Storage is trivial "
                        "(rank-8 B matrices) regardless of base model size. Before trusting a "
                        "run, check the printed anchor k values - they should mostly be within "
                        "roughly 0.7-1.4 of 1.0; if not, this needs to be smaller, not the "
                        "overall design rethought.")
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--grad_clip",  type=float, default=1.0)
    p.add_argument("--target_rs", nargs="+", type=float,
                   default=[0.02, 0.04, 0.06, 0.08, 0.10, 0.12, 0.15, 0.18,
                            0.22, 0.27, 0.33, 0.40, 0.50, 0.65, 0.85, 1.10],
                   help="Dense, evenly-spaced-in-R target grid (denser than any LR sweep "
                        "could guarantee, since LR->R is nonlinear near the transition).")
    p.add_argument("--n_perturb",      type=int,   default=200)
    p.add_argument("--n_text_samples", type=int,   default=128)
    p.add_argument("--seq_len",        type=int,   default=128)
    p.add_argument("--eval_slack",     type=float, default=None,
                   help="Default: 1%% of this model's own baseline WikiText-2 NLL "
                        "(same relslack convention as h1_kl_comparison.py / "
                        "recompute_h1_sigma_relslack.py).")
    p.add_argument("--kl_seq_len", type=int, default=64)
    p.add_argument("--kl_n_wiki",  type=int, default=200)
    p.add_argument("--n_superni",  type=int, default=200)
    p.add_argument("--n_mmlu",     type=int, default=200)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--output_dir", default="./results/h1_direct_r")
    p.add_argument("--crosscheck_dir", default=None,
                   help="Dir with h1_kl_comparison.py's lr*_results.json for this model, "
                        "used as real-training overlay points. Default: "
                        "./results/h1_kl/{model_tag} (or h1_kl/{model_tag} if that exists "
                        "locally, matching this repo's convention).")
    p.add_argument("--replot", action="store_true")
    p.add_argument("--force",  action="store_true")
    return p.parse_args()


# ── Reference training (unmerged - PEFT models forward correctly without merging) ──
#
# 2026-07-20 redesign: a single rescaled endpoint direction was cross-checked
# against real LR-sweep points and failed badly away from its own native R
# (task literally unlearned, ~40x more ΔNLL damage at matched R=0.22). Root
# cause: gradient descent isn't a straight line, so rescaling one distant
# endpoint down toward the origin doesn't retrace what shorter real training
# looks like. Fix: keep EVERY checkpoint's LoRA B weights (not just the
# max-norm one), and anchor each target R at whichever checkpoint's own
# per_param_norm is closest to it - a small, local rescale from a real point
# on the trajectory, instead of one large rescale from a distant point.

def train_reference_direction(base_model, tokenizer, lr, steps, batch_size, rank, alpha,
                               grad_clip, device, seed, model_name, task,
                               total_params_full, checkpoint_interval=50):
    """
    Fine-tune ONE reference trajectory on a fresh copy of base_model, for a
    FIXED, pre-declared step budget (not tuned after seeing results). Saves a
    full CPU-cloned snapshot of every LoRA layer's A AND B weights at every
    checkpoint_interval steps - ALL of them retained, not just the final or
    max-norm one - so later target-R rescaling can anchor from whichever real
    checkpoint is closest to each target.

    Both A and B are snapshotted, not just B: LoRA trains both matrices, A is
    not a fixed random projection that stays put. Reconstructing an old
    checkpoint's B against a DIFFERENT (later) A produces a delta that was
    never physically realized during training - confirmed as a real bug
    2026-07-20 (achieved_R diverged from target_R by up to 3.7x for early
    anchors, shrinking toward 1x for late ones - exactly the signature of A
    drifting away from each checkpoint's own value as training proceeds).

    Returns:
        peft_model, classifier - final trained model (A/B left at the last
            step; callers restore whichever checkpoint they choose, not this one)
        checkpoints - dict[step] -> {"A_list": [...], "B_list": [...],
            "classifier_state": {...}, "ppn": float}, one entry per
            checkpoint_interval steps plus the final step, ALL retained.
    """
    torch.manual_seed(seed)
    model_copy = copy.deepcopy(base_model)
    peft_model = add_lora(model_copy, rank, alpha, target_modules=_lora_targets(model_name))

    hidden_size = _get_hidden_size(base_model)
    num_classes = 3 if task == "mnli" else 2
    classifier  = nn.Linear(hidden_size, num_classes).to(device)

    params  = list(peft_model.parameters()) + list(classifier.parameters())
    opt     = torch.optim.AdamW([p for p in params if p.requires_grad], lr=lr)
    loss_fn = nn.CrossEntropyLoss()
    loader  = load_mnli(tokenizer, batch_size) if task == "mnli" else load_sst2(tokenizer, batch_size)

    lora_layers = get_lora_layers(peft_model)  # fixed layer order, computed once
    checkpoints = {}  # step -> {"A_list": [...], "B_list": [...], "ppn": float}

    def snapshot(step):
        A_list = [module.lora_A[adapter].weight.detach().clone().cpu()
                  for module, adapter in lora_layers]
        B_list = [module.lora_B[adapter].weight.detach().clone().cpu()
                  for module, adapter in lora_layers]
        # classifier is trained JOINTLY with the LoRA weights (same optimizer, same
        # loop) - it must be snapshotted and restored alongside A/B, or eval pairs an
        # earlier checkpoint's representations with a classifier that was only ever
        # trained against the FINAL step's representations. Same mismatched-pair bug
        # class as A vs stale-B, just one level up the stack.
        classifier_state = {k: v.detach().clone().cpu() for k, v in classifier.state_dict().items()}
        ppn = compute_direction_per_param_norm(lora_layers, total_params_full)
        checkpoints[step] = {"A_list": A_list, "B_list": B_list,
                             "classifier_state": classifier_state, "ppn": ppn}
        return ppn

    peft_model.train()
    classifier.train()
    step = 0
    print(f"  Training reference direction: lr={lr:.0e}, steps={steps} "
          f"(saving a B-snapshot every {checkpoint_interval} steps, ALL retained)")
    while step < steps:
        for input_ids, labels in loader:
            if step >= steps:
                break
            input_ids = input_ids.to(device)
            labels    = labels.to(device)

            out    = peft_model(input_ids, output_hidden_states=True)
            hidden = out.hidden_states[-1][:, -1, :].float()
            logits = classifier(hidden)
            loss   = loss_fn(logits, labels)

            if torch.isnan(loss):
                step += 1
                continue

            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(params, grad_clip)
            opt.step()
            step += 1

            if step % checkpoint_interval == 0:
                ppn = snapshot(step)
                if step % 100 == 0:
                    print(f"    step {step}/{steps}  loss={loss.item():.4f}  per_param_norm={ppn:.6e}",
                          flush=True)

    if steps not in checkpoints:
        snapshot(steps)  # always keep a final snapshot even off-interval

    print(f"  Saved {len(checkpoints)} checkpoints (steps {sorted(checkpoints.keys())[:3]}...{sorted(checkpoints.keys())[-3:]})")

    peft_model.eval()
    classifier.eval()
    return peft_model, classifier, checkpoints


def nearest_checkpoint(checkpoints, target_ppn):
    """
    Return (step, A_list, B_list, classifier_state, ppn) for whichever saved
    checkpoint's own per_param_norm is closest to target_ppn - selection by norm
    distance only, never by any damage/accuracy metric (same norm-blind principle
    as before). If target_ppn exceeds every checkpoint's ppn, this returns the
    LARGEST checkpoint (an upscale from there, not a downscale from something
    bigger) - the caller flags that case as true extrapolation beyond the whole
    trajectory.

    Returns A_list AND classifier_state, not just B_list: A must be restored to
    THIS checkpoint's own value (see train_reference_direction docstring), and
    the classifier must match too - it was trained jointly with the LoRA weights
    and is only ever otherwise available at its FINAL (step=ref_steps) state.
    """
    best_step = min(checkpoints, key=lambda s: abs(checkpoints[s]["ppn"] - target_ppn))
    ck = checkpoints[best_step]
    return best_step, ck["A_list"], ck["B_list"], ck["classifier_state"], ck["ppn"]


def set_lora_A_exact(lora_layers, A_list):
    """Restore every LoRA layer's A weight to A_list exactly (no scaling - A is
    never rescaled, only B is; see set_lora_scale). Must be called before
    set_lora_scale whenever the anchor checkpoint changes, so A and the
    about-to-be-rescaled B come from the SAME checkpoint, not a mismatched pair."""
    with torch.no_grad():
        for (module, adapter), A in zip(lora_layers, A_list):
            module.lora_A[adapter].weight.data.copy_(A)


def set_classifier_state(classifier, classifier_state, device):
    """Restore the classifier head to the SAME checkpoint's state as the LoRA
    A/B just restored. classifier is trained jointly with the LoRA weights, so a
    classifier left at its final-step values paired with an earlier checkpoint's
    hidden representations is the same mismatched-pair bug as stale-A-vs-new-B,
    just one level up the stack - it would corrupt the task-accuracy panel only
    (delta_nll_wiki/KL/mmlu_acc read the base LM's own logits, no classifier)."""
    classifier.load_state_dict({k: v.to(device) for k, v in classifier_state.items()})


def get_lora_layers(peft_model):
    """List of (module, adapter_key) for every LoRA-wrapped layer, in a stable order."""
    adapter = peft_model.active_adapter
    if isinstance(adapter, (list, tuple)):
        adapter = adapter[0]
    layers = []
    for module in peft_model.modules():
        if hasattr(module, "lora_A") and adapter in module.lora_A:
            layers.append((module, adapter))
    return layers


def compute_direction_per_param_norm(lora_layers, total_params_full):
    """RMS per-parameter norm of DeltaW = scaling*B@A summed over all LoRA layers,
    divided by the FULL model's parameter count - same denominator convention as
    h1_kl_comparison.py's compute_per_param_norm (all params, not just LoRA-touched
    ones), so R values from this script are directly comparable to the LR-sweep script."""
    total_sq = 0.0
    for module, adapter in lora_layers:
        A = module.lora_A[adapter].weight
        B = module.lora_B[adapter].weight
        scaling = module.scaling[adapter]
        delta = (B.detach().float() @ A.detach().float()) * scaling
        total_sq += delta.norm().item() ** 2
    return float((total_sq / max(total_params_full, 1)) ** 0.5)


def set_lora_scale(lora_layers, B_ref_list, k):
    """Set every LoRA layer's B weight to k * B_ref (absolute set from the clean
    reference, not a relative multiply, so floating-point error never compounds
    across target-R iterations)."""
    with torch.no_grad():
        for (module, adapter), B_ref in zip(lora_layers, B_ref_list):
            module.lora_B[adapter].weight.data.copy_(B_ref * k)


# ── Plotting ──────────────────────────────────────────────────────────────────

def _plot_interp_extrap(ax, summary, ykey, color, label):
    """Solid line + filled markers for interpolated points (k<=1, within the
    trained direction's actual reach); dashed line + hollow markers for
    extrapolated points (k>1, scaled beyond it) - never rendered as equivalent."""
    pts = sorted(summary, key=lambda r: r["R"])
    interp = [p for p in pts if not p.get("extrapolated", False)]
    extrap = [p for p in pts if p.get("extrapolated", False)]

    if interp:
        ax.plot([p["R"] for p in interp], [p[ykey] for p in interp], "-o",
                color=color, ms=6, lw=1.6, label=label, zorder=3)
    if extrap:
        # bridge from the last interpolated point so the dashed segment connects visually
        bridge = interp[-1:] if interp else []
        seg = bridge + extrap
        ax.plot([p["R"] for p in seg], [p[ykey] for p in seg], "--", color=color,
                lw=1.6, alpha=0.6, zorder=2)
        ax.plot([p["R"] for p in extrap], [p[ykey] for p in extrap], "o",
                mfc="white", mec=color, ms=6, mew=1.6, zorder=3,
                label=f"{label} (extrapolated - k>1, not equivalent evidence)")


def plot_direct_r(summary, crosscheck, out_path, model_name):
    """
    R vs Delta NLL, rescaled-direction points vs real independently-trained
    LR-sweep points overlaid - the key validity check for this whole design.
    Filled markers = interpolated (within the trained direction's reach);
    hollow markers = extrapolated (scaled beyond it, weaker evidence).
    """
    fig, axes = plt.subplots(1, 3, figsize=(17, 5))

    ax = axes[0]
    _plot_interp_extrap(ax, summary, "delta_nll_wiki", "#4a3aa7", "rescaled direction (this script)")
    if crosscheck:
        cr  = [c["R"] for c in crosscheck]
        cdn = [c["delta_nll_wiki"] for c in crosscheck]
        ax.scatter(cr, cdn, color="#e34948", s=110, marker="D", zorder=4,
                   label="real LR-sweep points (h1_kl_comparison.py)")
    ax.axhline(0, color="gray", ls=":", alpha=0.5)
    ax.set_xlabel("R = ||d(theta)|| / sigma_half")
    ax.set_ylabel("Delta NLL (wiki)")
    ax.set_title(f"{model_name}: dense R sweep vs. real LR points\n(agreement here validates the direction-rescale approximation)",
                 fontsize=10)
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    _plot_interp_extrap(ax, summary, "acc", "#1baf7a", "task acc")
    ax.set_xlabel("R")
    ax.set_ylabel("Task accuracy")
    ax.set_title("Task accuracy along the rescaled direction", fontsize=10)
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)

    ax = axes[2]
    mmlu = [r.get("mmlu_acc", float("nan")) for r in summary]
    has_mmlu = any(not np.isnan(m) for m in mmlu)
    if has_mmlu:
        _plot_interp_extrap(ax, summary, "mmlu_acc", "#eb6834", "MMLU acc")
        ax.legend(fontsize=7)
    else:
        ax.text(0.5, 0.5, "MMLU not computed", transform=ax.transAxes, ha="center", va="center", color="gray")
    ax.set_xlabel("R")
    ax.set_ylabel("MMLU accuracy")
    ax.set_title("World-knowledge degradation along the rescaled direction", fontsize=10)
    ax.grid(True, alpha=0.3)

    fig.suptitle(f"H1 direct-R sweep - {model_name} (2026-07-20)", fontsize=13, y=1.03)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved figure: {out_path}")


def load_crosscheck_points(crosscheck_dir):
    """Load any existing lr*_results.json from the original LR-sweep script as
    real-training reference points. Read-only - never writes to this dir.

    IMPORTANT: the file's own "R" field is frequently STALE (computed from a
    cached, pre-relslack sigma_half_pre - see RAW_DATA.md's standing warning,
    confirmed 12x off for GPT-2, 1.34x for Llama). Never trust it directly.
    Always recompute R = per_param_norm / (this run's own relslack sigma_half_pre)
    so the crosscheck overlay is on the same basis as the rescaled-direction curve."""
    if not crosscheck_dir or not os.path.isdir(crosscheck_dir):
        return []
    points = []
    for fname in sorted(os.listdir(crosscheck_dir)):
        if fname.startswith("lr") and fname.endswith("_results.json"):
            with open(os.path.join(crosscheck_dir, fname)) as f:
                r = json.load(f)
            if "per_param_norm" in r and "delta_nll_wiki" in r:
                points.append(r)  # R recomputed by caller, which has the correct sigma_half_pre
    return points


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    args   = get_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(args.seed)

    model_tag = args.model if args.task == "sst2" else f"{args.model}_{args.task}"
    out_dir = os.path.join(args.output_dir, model_tag)
    os.makedirs(out_dir, exist_ok=True)
    summary_path = os.path.join(out_dir, "summary.json")

    crosscheck_dir = args.crosscheck_dir or os.path.join("h1_kl", model_tag)
    crosscheck = load_crosscheck_points(crosscheck_dir)
    print(f"  Cross-check points from {crosscheck_dir}: {len(crosscheck)} found")

    def _lookup_cached_sigma():
        """Read-only sigma_half_pre lookup, no model load required - used by
        --replot only (the main path below has its own copy of this preference
        order with a loud warning on fallback; replot is lower-stakes since it
        doesn't recompute anything, but still prefers relslack first)."""
        relslack_path = os.path.join(crosscheck_dir, "pretrained_density_relslack.json")
        if os.path.exists(relslack_path):
            with open(relslack_path) as f:
                return json.load(f)["sigma_half"], relslack_path
        for p in (os.path.join(crosscheck_dir, "pretrained_density.json"),
                  os.path.join(out_dir, "pretrained_density.json")):
            if os.path.exists(p):
                print(f"  WARNING: no {relslack_path} found - falling back to {p}, "
                      f"which may use the OLD fixed-absolute-slack convention "
                      f"(confirmed ~12x too small for GPT-2 previously).")
                with open(p) as f:
                    return json.load(f)["sigma_half"], p
        return None, None

    def _apply_crosscheck_R(sigma):
        for c in crosscheck:
            c["R"] = c["per_param_norm"] / sigma if sigma else float("nan")

    if args.replot:
        sigma_half_pre, src = _lookup_cached_sigma()
        if sigma_half_pre is None:
            print("  WARNING: no cached sigma_half found for --replot; crosscheck R cannot "
                  "be computed, overlay will be skipped.")
        else:
            print(f"  sigma_half={sigma_half_pre:.6f} (from {src}) for crosscheck R recompute")
            _apply_crosscheck_R(sigma_half_pre)
        with open(summary_path) as f:
            summary = json.load(f)
        plot_direct_r(summary, crosscheck, os.path.join(out_dir, "direct_r_comparison.png"), args.model)
        return

    print(f"\n{'='*60}\n  H1 Direct-R Sweep - {args.model}  task={args.task}\n"
          f"  ref_lr={args.ref_lr}  ref_steps={args.ref_steps}\n"
          f"  target_rs: {args.target_rs}\n{'='*60}\n")

    print("Loading base model...")
    base_model, _ = _load_model_auto(args.model, device)
    tokenizer     = _get_tokenizer(args.model)
    target_names  = _get_target_names_auto(base_model, args.model)
    total_params_full = sum(p.numel() for p in base_model.parameters())

    if args.eval_slack is None:
        inp_slack, tgt_slack = load_data(tokenizer, args.n_text_samples, args.seq_len, device, seed=args.seed)
        nll_baseline    = compute_nll_loss(base_model, inp_slack, tgt_slack)
        args.eval_slack = 0.01 * nll_baseline
        print(f"  NLL_baseline={nll_baseline:.5f}  eval_slack={args.eval_slack:.5f} (1% relslack)")

    print("\nComputing sigma_half of pretrained model (or loading cached)...")
    sigma_path = os.path.join(out_dir, "pretrained_density.json")
    # Reuse the LR-sweep script's own cached sigma_half if it exists - same
    # pretrained checkpoint, no reason to recompute. Relslack version strongly
    # preferred: the plain pretrained_density.json uses the OLD fixed-absolute-slack
    # convention, confirmed ~12x too small for GPT-2 in a past bug (RAW_DATA.md).
    # Falling back to it silently would reintroduce that exact error into every R
    # value this whole run computes - so an explicit loud warning if it happens.
    relslack_reuse_path = os.path.join(crosscheck_dir, "pretrained_density_relslack.json")
    plain_reuse_path = os.path.join(crosscheck_dir, "pretrained_density.json")
    if os.path.exists(relslack_reuse_path):
        with open(relslack_reuse_path) as f:
            sigma_half_pre = json.load(f)["sigma_half"]
        print(f"  Reused sigma_half={sigma_half_pre:.6f} from {relslack_reuse_path} (relslack)")
    elif os.path.exists(plain_reuse_path):
        with open(plain_reuse_path) as f:
            sigma_half_pre = json.load(f)["sigma_half"]
        print(f"  WARNING: no {relslack_reuse_path} found - falling back to "
              f"{plain_reuse_path}, which may use the OLD fixed-absolute-slack "
              f"convention (confirmed ~12x too small for GPT-2 previously). "
              f"sigma_half={sigma_half_pre:.6f} - verify this is relslack-based "
              f"before trusting any R value from this run.")
    elif os.path.exists(sigma_path):
        with open(sigma_path) as f:
            sigma_half_pre = json.load(f)["sigma_half"]
        print(f"  Loaded cached sigma_half={sigma_half_pre:.6f}")
    else:
        inp_pre, tgt_pre = load_data(tokenizer, args.n_text_samples, args.seq_len, device, seed=args.seed)
        sigmas = [0.0001, 0.0002, 0.0005, 0.001, 0.002, 0.005, 0.01, 0.02, 0.05]
        density_results = [
            estimate_density(base_model, target_names, inp_pre, tgt_pre, sigma=s,
                             N=args.n_perturb, eval_slack=args.eval_slack)
            for s in sigmas
        ]
        sigma_half_pre = half_max_sigma(density_results)
        with open(sigma_path, "w") as f:
            json.dump({"sigma_half": sigma_half_pre,
                       "densities": {str(r["sigma"]): r["ci_lower"] for r in density_results}}, f, indent=2)
        print(f"  Computed sigma_half={sigma_half_pre:.6f}")

    _apply_crosscheck_R(sigma_half_pre)
    if crosscheck:
        print(f"  Recomputed R for {len(crosscheck)} crosscheck points using this run's own "
              f"sigma_half_pre={sigma_half_pre:.6f} (not the source file's own, possibly stale, R field)")

    print("\nPreparing KL evaluation data...")
    wiki_ids    = load_wikitext_for_kl(tokenizer, args.kl_n_wiki, args.kl_seq_len, device, seed=args.seed)
    diverse_ids = encode_diverse_prompts(tokenizer, DIVERSE_PROMPTS, args.kl_seq_len, device)
    superni_ids = load_superni_prompts(tokenizer, args.n_superni, args.kl_seq_len, device, seed=args.seed)
    wiki_inp, wiki_tgt = wiki_ids[:, :-1], wiki_ids[:, 1:]
    nll_wiki_base = compute_nll_loss(base_model, wiki_inp, wiki_tgt)
    print(f"  Base WikiText-2 NLL: {nll_wiki_base:.4f}")

    # ── ONE reference trajectory - every checkpoint retained, targets anchor locally ──
    peft_model, classifier, checkpoints = train_reference_direction(
        base_model, tokenizer, lr=args.ref_lr, steps=args.ref_steps,
        batch_size=args.batch_size, rank=args.rank, alpha=args.alpha,
        grad_clip=args.grad_clip, device=device, seed=args.seed,
        model_name=args.model, task=args.task,
        total_params_full=total_params_full, checkpoint_interval=args.checkpoint_interval,
    )
    lora_layers = get_lora_layers(peft_model)
    with open(os.path.join(out_dir, "reference_ppn_trajectory.json"), "w") as f:
        json.dump({"trajectory": [{"step": s, "per_param_norm": c["ppn"]}
                                  for s, c in sorted(checkpoints.items())],
                   "ref_lr": args.ref_lr, "ref_steps": args.ref_steps,
                   "checkpoint_interval": args.checkpoint_interval}, f, indent=2)

    max_ckpt_ppn = max(c["ppn"] for c in checkpoints.values())
    max_ckpt_R   = max_ckpt_ppn / sigma_half_pre if sigma_half_pre > 0 else float("nan")
    print(f"\n  Trajectory covers {len(checkpoints)} checkpoints, up to R={max_ckpt_R:.4f} "
          f"at the final step. Each target below anchors at its own nearest checkpoint "
          f"(norm-blind selection), not always this one.")
    n_extrapolated = sum(1 for t in args.target_rs if t * sigma_half_pre > max_ckpt_ppn)
    if n_extrapolated:
        print(f"  NOTE: {n_extrapolated} of {len(args.target_rs)} target R values exceed the "
              f"WHOLE trajectory's reach (max R={max_ckpt_R:.4f}) and will be tagged "
              f"'extrapolated' - upscaled beyond anything actually trained, unlike the other "
              f"targets which each anchor at a real, nearby checkpoint.")

    # ── Anchor each target R at its nearest real checkpoint, evaluate directly ──
    summary = []
    for target_R in sorted(args.target_rs):
        r_tag    = f"r{target_R:.3f}"
        out_path = os.path.join(out_dir, f"{r_tag}_results.json")
        if os.path.exists(out_path) and not args.force:
            with open(out_path) as f:
                summary.append(json.load(f))
            print(f"  [{r_tag}] cached")
            continue

        t0 = time.time()
        target_ppn = target_R * sigma_half_pre
        anchor_step, anchor_A_list, anchor_B_list, anchor_classifier_state, anchor_ppn = \
            nearest_checkpoint(checkpoints, target_ppn)
        k = target_ppn / anchor_ppn if anchor_ppn > 0 else 0.0
        # A restored EXACTLY first (this checkpoint's own value, not whatever the
        # model currently has left over from training or a prior iteration), THEN
        # B rescaled by k - so the pair reconstructs a delta that was actually
        # realized at this checkpoint's step, just scaled by k, not a mismatched
        # (old B, current A) combination. Classifier restored to the SAME
        # checkpoint too, or task-accuracy eval pairs this step's representations
        # with a classifier only ever trained against the FINAL step's.
        set_lora_A_exact(lora_layers, [a.to(device) for a in anchor_A_list])
        set_lora_scale(lora_layers, [b.to(device) for b in anchor_B_list], k)
        set_classifier_state(classifier, anchor_classifier_state, device)
        anchor_R = anchor_ppn / sigma_half_pre if sigma_half_pre > 0 else float("nan")

        achieved_ppn = compute_direction_per_param_norm(lora_layers, total_params_full)
        achieved_R   = achieved_ppn / sigma_half_pre if sigma_half_pre > 0 else float("nan")

        if args.task == "mnli":
            acc = eval_mnli_accuracy(peft_model, classifier, tokenizer, device)
        else:
            acc = eval_sst2_accuracy(peft_model, classifier, tokenizer, device)

        kl_wiki    = compute_kl(base_model, peft_model, wiki_ids)
        kl_diverse = compute_kl(base_model, peft_model, diverse_ids)
        kl_superni = compute_kl(base_model, peft_model, superni_ids)
        nll_wiki_ft = compute_nll_loss(peft_model, wiki_inp, wiki_tgt)
        delta_nll   = nll_wiki_ft - nll_wiki_base
        mmlu_acc    = eval_mmlu_accuracy(peft_model, tokenizer, device, n_samples=args.n_mmlu, seed=args.seed)

        elapsed = time.time() - t0
        print(f"  [{r_tag}] anchor=step{anchor_step}(R={anchor_R:.4f}) k={k:.3f}  "
              f"achieved_R={achieved_R:.4f}  acc={acc:.4f}  dnll={delta_nll:+.4f}  "
              f"kl_wiki={kl_wiki:.4f}  kl_diverse={kl_diverse:.4f}  mmlu={mmlu_acc:.4f}  ({elapsed:.0f}s)")

        result = {
            "target_R": target_R, "R": achieved_R, "k": k,
            "anchor_step": anchor_step, "anchor_R": anchor_R,
            "per_param_norm": achieved_ppn, "sigma_half_pre": sigma_half_pre,
            "rank": args.rank, "ref_lr": args.ref_lr, "ref_steps": args.ref_steps,
            "task": args.task, "acc": acc,
            "nll_wiki_base": nll_wiki_base, "nll_wiki_ft": nll_wiki_ft,
            "delta_nll_wiki": delta_nll,
            "kl_wiki": kl_wiki, "kl_diverse": kl_diverse, "kl_superni": kl_superni,
            "mmlu_acc": mmlu_acc, "elapsed_s": elapsed,
            "method": "checkpoint_anchored_rescale",
            # True only if this target exceeds the WHOLE trajectory's reach (upscaled
            # beyond anything actually trained). A large |k-1| at an otherwise-valid
            # anchor is NOT this flag - see rescale_k for that; it's still a real,
            # local rescale from a real checkpoint, just not a small one.
            "extrapolated": bool(target_ppn > max_ckpt_ppn),
        }
        with open(out_path, "w") as f:
            json.dump(result, f, indent=2)
        summary.append(result)

    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSummary saved: {summary_path}")

    plot_direct_r(summary, crosscheck, os.path.join(out_dir, "direct_r_comparison.png"), args.model)

    print(f"\n{'─'*110}\n{'target_R':>9}  {'R':>8}  {'anchor_step':>11}  {'k':>7}  "
          f"{'Acc':>6}  {'ΔNLL':>8}  {'KL(wiki)':>9}  {'MMLU':>6}\n{'─'*110}")
    ks = [r["k"] for r in summary if not r.get("extrapolated", False)]
    for r in summary:
        flag = "  <-- k far from 1" if not (0.7 <= r["k"] <= 1.4) else ""
        print(f"  {r['target_R']:>9.3f}  {r['R']:>8.4f}  {r.get('anchor_step', '-'):>11}  "
              f"{r['k']:>7.3f}  {r['acc']:>6.4f}  {r['delta_nll_wiki']:>+8.4f}  "
              f"{r['kl_wiki']:>9.4f}  {r['mmlu_acc']:>6.4f}{flag}")
    print(f"{'─'*110}")
    if ks:
        print(f"  rescale_k across {len(ks)} non-extrapolated points: "
              f"min={min(ks):.3f}  max={max(ks):.3f}  "
              f"({'looks fine - all close to 1' if max(abs(k-1) for k in ks) < 0.4 else 'WIDE - consider a smaller --checkpoint_interval'})")
    print()


if __name__ == "__main__":
    main()
