"""
lora_density_experiment.py
==========================
Three linked experiments on LoRA fine-tuning and certified weight-space density.

Experiment 1: Does fine-tuning reshape density structure?
  Fine-tune GPT-2 on SST-2 with LoRA, then run the certified density sweep
  (same C(θ,σ,m) framework as certified_density_experiment.py) on the fine-tuned
  model. Compare σ½ and σ* before vs. after.

  Hypothesis: LoRA fine-tuning sharpens the optimum → σ½ drops.
  Null result: σ½ stays constant → density structure is invariant to LoRA.

Experiment 2: Does the certificate cover LoRA? (norm comparison)
  For each fine-tuned checkpoint, compute the per-parameter LoRA update norm
  and compare it to σ½ from the pretrained certificate.

  Hypothesis (professor's concern): update norm >> σ½ even when base NLL is
  preserved → certificate does not cover gradient-aligned updates.

Experiment 3: Subspace sampling — LoRA update subspace
  Sample Gaussian noise projected onto the LoRA update subspace (SVD of ΔW=BA)
  and run C(θ,σ,m) within that subspace. Compare to isotropic baseline.

  Hypothesis: subspace success rate differs from isotropic → the LoRA update
  subspace has geometrically special robustness structure.

Usage:
  # Full experiment (Exp 1+2, single learning rate)
  python lora_density_experiment.py --model gpt2 --lr 5e-4

  # Sweep learning rates (Exp 1+2)
  python lora_density_experiment.py --model gpt2 --lr 1e-4 5e-4 1e-3 5e-3

  # Include subspace sampling (Exp 3, slower)
  python lora_density_experiment.py --model gpt2 --lr 5e-4 --subspace

  # Skip fine-tuning, reload saved checkpoints and rerun density
  python lora_density_experiment.py --model gpt2 --lr 5e-4 --replot

Requirements:
  pip install peft datasets transformers torch
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

# ── Reuse existing infrastructure ─────────────────────────────────────────────
sys.path.insert(0, os.path.dirname(__file__))
from certified_density_experiment import (
    binom_ci, load_model, get_target_param_names,
    save_weights, restore_weights, perturb_weights,
    load_data, compute_nll_loss, estimate_density,
    certified_radius, half_max_sigma, MODEL_PARAMS,
)

# ── Architecture detection ────────────────────────────────────────────────────
def _is_gpt2(model_name):
    return "gpt2" in model_name.lower()

def _default_target_modules(model_name):
    if _is_gpt2(model_name):
        return ["c_attn", "c_proj"]
    return ["q_proj", "v_proj", "k_proj", "o_proj"]

def _load_model_auto(model_name, device):
    """Load any causal LM; falls back to AutoModelForCausalLM for non-GPT-2."""
    if _is_gpt2(model_name):
        return load_model(model_name, device)
    from transformers import AutoModelForCausalLM
    model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=torch.float16, device_map={"": str(device)}
    ).eval()
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Loaded {model_name} ({n_params:,} params)")
    return model, n_params

def _get_target_names_auto(model, model_name, n_blocks=None):
    """Return set of parameter names to perturb, handling GPT-2 and Llama/Mistral."""
    if _is_gpt2(model_name):
        return get_target_param_names(model, n_blocks)
    # Llama / Mistral / Qwen: layers live at model.model.layers
    layers = model.model.layers
    total = len(layers)
    n_blocks = n_blocks or total
    start = max(0, total - n_blocks)
    names = {
        name
        for name, _ in model.named_parameters()
        if any(name.startswith(f"model.layers.{i}.") for i in range(start, total))
    }
    n_params = sum(p.numel() for n, p in model.named_parameters() if n in names)
    print(f"  Targeting blocks {start}–{total-1}: {n_params:,} params")
    return names

def _get_hidden_size(model):
    cfg = model.config
    return getattr(cfg, "n_embd", getattr(cfg, "hidden_size", None))

def _get_tokenizer(model_name):
    if _is_gpt2(model_name):
        from transformers import GPT2Tokenizer
        tok = GPT2Tokenizer.from_pretrained(model_name)
    else:
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(model_name)
    tok.pad_token = tok.eos_token
    return tok


# ── Args ──────────────────────────────────────────────────────────────────────
def get_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model",   default="gpt2",
                   help="HuggingFace model name (default: gpt2)")
    p.add_argument("--lr", nargs="+", type=float, default=[5e-4],
                   help="LoRA learning rates to sweep (one checkpoint per lr)")
    p.add_argument("--lora_rank",   type=int, default=8)
    p.add_argument("--lora_alpha",  type=int, default=16)
    p.add_argument("--lora_target_modules", nargs="+", default=None,
                   help="LoRA target module names. Defaults: GPT-2 → c_attn,c_proj; "
                        "Llama/Mistral → q_proj,v_proj,k_proj,o_proj")
    p.add_argument("--train_steps", type=int, default=500,
                   help="LoRA fine-tuning steps on SST-2")
    p.add_argument("--train_batch", type=int, default=16)
    p.add_argument("--sigmas", nargs="+", type=float,
                   default=[0.0001, 0.0002, 0.0005, 0.001, 0.002, 0.005,
                            0.01, 0.02, 0.05, 0.1, 0.2])
    p.add_argument("--n_perturb",      type=int,   default=200)
    p.add_argument("--n_text_samples", type=int,   default=128)
    p.add_argument("--seq_len",        type=int,   default=128)
    p.add_argument("--target_density", type=float, default=0.40)
    p.add_argument("--eval_slack",     type=float, default=1e-4)
    p.add_argument("--subspace",   action="store_true",
                   help="Also run Experiment 3: subspace sampling")
    p.add_argument("--replot",     action="store_true",
                   help="Skip training/density; reload JSONs and regenerate plots")
    p.add_argument("--output_dir", default="./results/lora_density")
    p.add_argument("--finetune_fp32", action="store_true",
                   help="Cast model to fp32 during LoRA fine-tuning for numerical "
                        "stability (recommended for Llama at lr≥1e-3)")
    p.add_argument("--grad_clip", type=float, default=1.0,
                   help="Gradient clipping max norm (default 1.0)")
    return p.parse_args()


# ── LoRA fine-tuning ──────────────────────────────────────────────────────────
def load_sst2(tokenizer, batch_size, max_len=128, split="train", seed=42):
    from datasets import load_dataset
    ds = load_dataset("glue", "sst2", split=split)
    ds = ds.shuffle(seed=seed)
    texts  = ds["sentence"]
    labels = ds["label"]

    input_ids_list, label_list = [], []
    for text, label in zip(texts, labels):
        enc = tokenizer(text, truncation=True, max_length=max_len,
                        padding="max_length", return_tensors="pt")
        input_ids_list.append(enc["input_ids"].squeeze(0))
        label_list.append(label)

    input_ids = torch.stack(input_ids_list)
    labels_t  = torch.tensor(label_list, dtype=torch.long)

    dataset = torch.utils.data.TensorDataset(input_ids, labels_t)
    loader  = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=True, drop_last=True
    )
    return loader


def add_lora(model, rank, alpha, target_modules=None):
    """Attach LoRA adapters using HuggingFace peft."""
    from peft import get_peft_model, LoraConfig, TaskType
    if target_modules is None:
        target_modules = ["c_attn", "c_proj"]  # default; caller should override
    config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=rank,
        lora_alpha=alpha,
        target_modules=target_modules,
        lora_dropout=0.05,
        bias="none",
    )
    return get_peft_model(model, config)


def finetune_lora(base_model, tokenizer, lr, train_steps, batch_size,
                  rank, alpha, device, target_modules=None, seed=42,
                  finetune_fp32=False, grad_clip=1.0):
    """
    Fine-tune base_model on SST-2 with LoRA for train_steps steps.
    Returns the merged (LoRA absorbed) model and the per-parameter update norm.
    finetune_fp32: cast training copy to fp32 (prevents fp16 overflow at high LR).
    grad_clip: max norm for gradient clipping (default 1.0).
    """
    torch.manual_seed(seed)
    dtype_str = "fp32" if finetune_fp32 else "fp16"
    print(f"\n  Fine-tuning with LoRA (rank={rank}, alpha={alpha}, lr={lr:.1e}, "
          f"steps={train_steps}, dtype={dtype_str}, grad_clip={grad_clip})")

    # Snapshot pretrained weights before any modification
    pretrained_snapshot = {n: p.data.clone().float()
                           for n, p in base_model.named_parameters()}

    # Attach LoRA and add classification head
    model_copy = copy.deepcopy(base_model)
    if finetune_fp32:
        model_copy = model_copy.float()
    peft_model = add_lora(model_copy, rank, alpha, target_modules)

    # Simple linear classification head on top of last token hidden state
    hidden_size = _get_hidden_size(base_model)
    classifier  = nn.Linear(hidden_size, 2).to(device)

    params = list(peft_model.parameters()) + list(classifier.parameters())
    opt    = torch.optim.AdamW(
        [p for p in params if p.requires_grad], lr=lr
    )

    loader   = load_sst2(tokenizer, batch_size)
    loss_fn  = nn.CrossEntropyLoss()
    step     = 0
    peft_model.train()
    classifier.train()

    while step < train_steps:
        for input_ids, labels in loader:
            if step >= train_steps:
                break
            input_ids = input_ids.to(device)
            labels    = labels.to(device)

            # Pull hidden states from the last transformer block (fp16 → fp32).
            # output_hidden_states=True works for both GPT-2 and Llama/Mistral;
            # hidden_states[-1] is the final block output before the LM head.
            out = peft_model(input_ids, output_hidden_states=True)
            hidden_states = out.hidden_states[-1][:, -1, :].float()  # (B, H) fp32
            logits = classifier(hidden_states)
            loss   = loss_fn(logits, labels)

            if torch.isnan(loss):
                print(f"    step {step}: NaN loss detected, skipping step",
                      flush=True)
                opt.zero_grad()
                step += 1
                continue

            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(params, grad_clip)
            opt.step()
            step += 1

            if step % 50 == 0:
                print(f"    step {step}/{train_steps}  loss={loss.item():.4f}",
                      flush=True)

    # Merge LoRA weights back into base model
    merged_model = peft_model.merge_and_unload().eval()

    # Compute per-parameter update norm: ||ΔW||_F / sqrt(d) per layer, then mean
    update_norms = {}
    total_delta_sq, total_params = 0.0, 0
    for name, param in merged_model.named_parameters():
        if name in pretrained_snapshot:
            delta = param.data.float() - pretrained_snapshot[name]
            d     = delta.numel()
            norm  = (delta.norm().item() ** 2)
            update_norms[name] = float(delta.norm().item() / (d ** 0.5))
            total_delta_sq += norm
            total_params   += d

    per_param_norm = float((total_delta_sq / max(total_params, 1)) ** 0.5)
    print(f"  Per-parameter update norm: {per_param_norm:.6f}  "
          f"(σ½ pretrained baseline ≈ 0.0007)")

    return merged_model, per_param_norm, update_norms


# ── Experiment 3: subspace sampling ──────────────────────────────────────────
def build_lora_subspace_projector(pretrained_params, finetuned_model,
                                  target_names, n_directions=50):
    """
    Build a layer-wise projector for the LoRA update subspace.

    For each 2-D weight tensor modified by LoRA, compute the top-k left
    singular vectors of ΔW (= BA after merge) via SVD. The projector maps
    a flat noise vector (covering target_names only) to a new flat vector
    where each LoRA-modified block is projected onto those singular directions
    and non-LoRA blocks are passed through unchanged (isotropic).

    This avoids the incorrect torch.stack() across differently-shaped tensors
    and avoids materialising a (n_layers × total_d) joint matrix.
    """
    # Build ordered param list and cumulative offsets — must match the order
    # used by estimate_density_subspace so slice indices are consistent.
    param_list = [(n, p)
                  for n, p in finetuned_model.named_parameters()
                  if n in target_names]
    total_d = sum(p.numel() for _, p in param_list)

    layer_bases = {}   # name -> {"basis": (d_out, k), "start": int, "end": int, "shape"}
    offset = 0
    n_lora = 0
    for name, param in param_list:
        d = param.numel()
        if name in pretrained_params and len(param.shape) == 2:
            delta = param.data.float() - pretrained_params[name]
            if delta.norm() > 1e-12:          # non-zero → LoRA-modified
                U, S, _ = torch.linalg.svd(delta, full_matrices=False)
                # Keep only singular directions with non-negligible energy
                k = min(n_directions, U.shape[1],
                        int((S > 1e-8 * S[0]).sum().item()))
                k = max(k, 1)
                layer_bases[name] = {
                    "basis": U[:, :k].cpu(),   # (d_out, k) — keep on CPU
                    "start": offset,
                    "end":   offset + d,
                    "shape": tuple(param.shape),
                    "k":     k,
                }
                n_lora += 1
        offset += d

    print(f"  Subspace projector: {n_lora} LoRA-modified 2-D layers, "
          f"up to {n_directions} directions each")

    def project(noise_flat):
        """
        noise_flat: 1-D CPU float32 tensor of length total_d.
        Returns projected tensor of same shape.
        """
        result = noise_flat.clone()
        for info in layer_bases.values():
            s, e   = info["start"], info["end"]
            shape  = info["shape"]
            basis  = info["basis"]          # (d_out, k)
            k      = info["k"]
            chunk  = noise_flat[s:e].reshape(shape)   # (d_out, d_in)
            # Project each column of chunk onto column space of ΔW.
            # projected = basis @ (basis.T @ chunk)
            projected = basis @ (basis.T @ chunk)     # (d_out, d_in)
            # Rescale so E[||projected||²] ≈ E[||chunk||²]
            scale = (shape[0] / k) ** 0.5
            result[s:e] = (projected * scale).flatten()
        return result

    return project, total_d


def estimate_density_subspace(model, names, inp, tgt, sigma, N,
                               projector, confidence=0.95,
                               eval_slack=0.0, tag=""):
    """
    Like estimate_density but noise is projected onto the LoRA update subspace.
    """
    t0       = time.time()
    snapshot = save_weights(model, names)
    base_nll = compute_nll_loss(model, inp, tgt)
    threshold = base_nll + eval_slack
    n_better  = 0

    # Build flat parameter vector structure for projection
    param_list = [(n, p) for n, p in model.named_parameters() if n in names]
    total_d    = sum(p.numel() for _, p in param_list)

    for seed in range(N):
        if seed % 20 == 0:
            print(f"    [{tag}] σ={sigma:.5f}  {seed}/{N}", end="\r", flush=True)

        torch.manual_seed(seed)
        # Sample isotropic Gaussian, project onto subspace, reshape back
        noise_flat = torch.randn(total_d) * sigma
        noise_proj = projector(noise_flat)

        # Apply projected noise to model parameters
        offset = 0
        with torch.no_grad():
            for name, param in param_list:
                d     = param.numel()
                chunk = noise_proj[offset:offset + d].reshape(param.shape).to(param.device)
                w_f32 = param.data.float() + chunk
                param.data.copy_(w_f32.to(param.dtype))
                offset += d

        nll = compute_nll_loss(model, inp, tgt)
        if nll <= threshold:
            n_better += 1
        restore_weights(model, snapshot, names)

    print(flush=True)
    density            = n_better / N
    ci_lower, ci_upper = binom_ci(n_better, N, confidence)
    print(f"    [subspace] σ={sigma:.5f}  density={density:.3f}  "
          f"CI=[{ci_lower:.3f}, {ci_upper:.3f}]  ({time.time()-t0:.0f}s)")

    return {
        "sigma":    sigma, "density": density,
        "ci_lower": ci_lower, "ci_upper": ci_upper,
        "n_better": n_better, "N": N,
        "base_ppl": base_nll, "eval_slack": eval_slack,
    }


# ── Plotting ──────────────────────────────────────────────────────────────────
def plot_before_after(pretrained_results, lora_results_by_lr, output_dir, model_name):
    """Density curves: pretrained baseline vs. each LoRA lr."""
    fig, ax = plt.subplots(figsize=(11, 6))

    sigmas = [r["sigma"]   for r in pretrained_results]
    dens   = [r["density"] for r in pretrained_results]
    ci_lo  = [r["ci_lower"] for r in pretrained_results]
    ci_hi  = [r["ci_upper"] for r in pretrained_results]
    ax.plot(sigmas, dens, color="black", lw=2.5, marker="o", ms=5,
            label=f"{model_name} pretrained")
    ax.fill_between(sigmas, ci_lo, ci_hi, color="black", alpha=0.10)

    colors = plt.cm.Reds(np.linspace(0.4, 0.9, len(lora_results_by_lr)))
    for (lr, results), color in zip(sorted(lora_results_by_lr.items()), colors):
        sigmas = [r["sigma"]   for r in results]
        dens   = [r["density"] for r in results]
        ci_lo  = [r["ci_lower"] for r in results]
        ci_hi  = [r["ci_upper"] for r in results]
        ax.plot(sigmas, dens, color=color, lw=2, marker="s", ms=4,
                label=f"LoRA lr={lr:.0e}")
        ax.fill_between(sigmas, ci_lo, ci_hi, color=color, alpha=0.08)

    ax.set_xscale("log")
    ax.set_xlabel("Perturbation scale σ (log)", fontsize=12)
    ax.set_ylabel("C(θ, σ, m)  —  certified density", fontsize=12)
    ax.set_title(f"Certified Density Before vs. After LoRA Fine-Tuning\n"
                 f"{model_name} on SST-2  (N=200, 95% CI)", fontsize=12)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.axhline(0.40, color="gray", lw=1, ls=":")
    plt.tight_layout()
    path = os.path.join(output_dir, "density_before_after.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    print(f"Saved: {path}")
    plt.close()


def plot_norm_vs_sigma_half(norm_by_lr, pretrained_sigma_half, output_dir):
    """Bar chart: per-parameter LoRA update norm vs. σ½ baseline."""
    lrs   = sorted(norm_by_lr.keys())
    norms = [norm_by_lr[lr] for lr in lrs]
    labels = [f"{lr:.0e}" for lr in lrs]

    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(labels, norms, color="steelblue", alpha=0.8, label="LoRA update norm")
    ax.axhline(pretrained_sigma_half, color="crimson", lw=2, ls="--",
               label=f"σ½ pretrained = {pretrained_sigma_half:.5f}")

    for bar, val in zip(bars, norms):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() * 1.02,
                f"{val:.5f}", ha="center", va="bottom", fontsize=9)

    ax.set_xlabel("LoRA learning rate", fontsize=12)
    ax.set_ylabel("Per-parameter update norm  ||ΔW||/√d", fontsize=12)
    ax.set_title("Experiment 2: Does the Certificate Cover LoRA?\n"
                 "Per-parameter update norm vs. σ½ from pretrained certificate",
                 fontsize=11)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3, axis="y")
    plt.tight_layout()
    path = os.path.join(output_dir, "norm_vs_sigma_half.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    print(f"Saved: {path}")
    plt.close()


def plot_sigma_half_comparison(pretrained_sigma_half, lora_sigma_half_by_lr,
                               output_dir):
    """σ½ before vs. after LoRA for each learning rate."""
    lrs    = sorted(lora_sigma_half_by_lr.keys())
    values = [lora_sigma_half_by_lr[lr] for lr in lrs]
    labels = [f"{lr:.0e}" for lr in lrs]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(labels, values, color="steelblue", alpha=0.8, label="LoRA σ½")
    ax.axhline(pretrained_sigma_half, color="crimson", lw=2, ls="--",
               label=f"Pretrained σ½ = {pretrained_sigma_half:.5f}")

    ax.set_xlabel("LoRA learning rate", fontsize=12)
    ax.set_ylabel("σ½  (half-max sigma)", fontsize=12)
    ax.set_title("Experiment 1: Does Fine-Tuning Reshape Density Structure?\n"
                 "σ½ before vs. after LoRA fine-tuning on SST-2",
                 fontsize=11)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3, axis="y")
    plt.tight_layout()
    path = os.path.join(output_dir, "sigma_half_comparison.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    print(f"Saved: {path}")
    plt.close()


def plot_subspace_comparison(isotropic_results, subspace_results, lr, output_dir):
    """Density curves: isotropic vs. LoRA subspace sampling."""
    fig, ax = plt.subplots(figsize=(10, 6))

    for results, label, color in [
        (isotropic_results, "Isotropic (full weight space)", "steelblue"),
        (subspace_results,  "Projected (LoRA update subspace)", "crimson"),
    ]:
        sigmas = [r["sigma"]   for r in results]
        dens   = [r["density"] for r in results]
        ci_lo  = [r["ci_lower"] for r in results]
        ci_hi  = [r["ci_upper"] for r in results]
        ax.plot(sigmas, dens, color=color, lw=2, marker="o", ms=5, label=label)
        ax.fill_between(sigmas, ci_lo, ci_hi, color=color, alpha=0.12)

    ax.set_xscale("log")
    ax.set_xlabel("Perturbation scale σ (log)", fontsize=12)
    ax.set_ylabel("C(θ, σ, m)  —  certified density", fontsize=12)
    ax.set_title(f"Experiment 3: Subspace Sampling\n"
                 f"Isotropic vs. LoRA update subspace  (lr={lr:.0e})", fontsize=12)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.axhline(0.40, color="gray", lw=1, ls=":")
    plt.tight_layout()
    path = os.path.join(output_dir, f"subspace_lr{lr:.0e}.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    print(f"Saved: {path}")
    plt.close()


# ── Print summary ─────────────────────────────────────────────────────────────
def print_summary(model_name, pretrained_results, lora_results_by_lr,
                  norm_by_lr, target_density):
    pre_sigma_half = half_max_sigma(pretrained_results)
    pre_sigma_star = certified_radius(pretrained_results, target_density)

    print("\n" + "="*70)
    print(f"SUMMARY — {model_name}")
    print("="*70)
    print(f"  Pretrained:  σ* = {pre_sigma_star:.5f}  σ½ = {pre_sigma_half:.5f}")
    print()
    print(f"  {'lr':>8}  {'σ*':>10}  {'σ½':>10}  {'update_norm':>14}  "
          f"{'norm/σ½':>10}  base_NLL_preserved?")
    print(f"  {'-'*8}  {'-'*10}  {'-'*10}  {'-'*14}  {'-'*10}  {'-'*20}")

    for lr in sorted(lora_results_by_lr.keys()):
        results      = lora_results_by_lr[lr]
        sigma_half   = half_max_sigma(results)
        sigma_star   = certified_radius(results, target_density)
        norm         = norm_by_lr.get(lr, float("nan"))
        ratio        = norm / pre_sigma_half if pre_sigma_half > 0 else float("nan")
        # Check if base NLL is roughly preserved (density at small σ still high)
        small_sigma_density = next(
            (r["density"] for r in results if r["sigma"] <= 0.0002), None
        )
        nll_ok = "yes" if small_sigma_density and small_sigma_density > 0.3 else "check"
        print(f"  {lr:>8.1e}  {sigma_star:>10.5f}  {sigma_half:>10.5f}  "
              f"{norm:>14.6f}  {ratio:>10.2f}x  {nll_ok}")

    print()
    print("  KEY QUESTION (Exp 2):")
    for lr in sorted(norm_by_lr.keys()):
        norm  = norm_by_lr[lr]
        ratio = norm / pre_sigma_half if pre_sigma_half > 0 else float("nan")
        verdict = ("norm >> σ½ → certificate does NOT cover LoRA at this lr"
                   if ratio > 5 else
                   "norm ~ σ½ → certificate is meaningful for LoRA at this lr"
                   if ratio < 2 else
                   "norm > σ½ → partial overlap, certificate is conservative")
        print(f"    lr={lr:.1e}: update_norm/σ½ = {ratio:.1f}x  → {verdict}")
    print("="*70)


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    args   = get_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.output_dir, exist_ok=True)
    print(f"Device: {device}  |  Output: {args.output_dir}")

    tokenizer = _get_tokenizer(args.model)
    target_modules = args.lora_target_modules or _default_target_modules(args.model)

    # ── Load pretrained model ─────────────────────────────────────────────────
    print(f"\nLoading {args.model}...")
    model, n_params = _load_model_auto(args.model, device)
    names = _get_target_names_auto(model, args.model, n_blocks=None)
    inp, tgt = load_data(tokenizer, args.n_text_samples, args.seq_len, device)

    pretrained_json = os.path.join(args.output_dir, "pretrained_density.json")

    if args.replot and os.path.exists(pretrained_json):
        print("Reloading pretrained density from JSON...")
        with open(pretrained_json) as f:
            pretrained_results = json.load(f)["sigma_results"]
    else:
        # ── Exp 1 baseline: certified density of pretrained model ─────────────
        print(f"\n[Pretrained] Running certified density sweep...")
        pretrained_results = []
        for sigma in args.sigmas:
            r = estimate_density(model, names, inp, tgt, sigma,
                                 N=args.n_perturb, eval_slack=args.eval_slack,
                                 tag=f"{args.model}/pretrained")
            pretrained_results.append(r)
        with open(pretrained_json, "w") as f:
            json.dump({"model": args.model, "condition": "pretrained",
                       "sigma_results": pretrained_results}, f, indent=2)

    pretrained_sigma_half = half_max_sigma(pretrained_results)
    print(f"\n  Pretrained σ½ = {pretrained_sigma_half:.5f}")

    # ── Snapshot pretrained weights for norm computation ──────────────────────
    pretrained_snapshot = {n: p.data.clone().float()
                           for n, p in model.named_parameters()}

    # ── LoRA sweep ────────────────────────────────────────────────────────────
    lora_results_by_lr   = {}
    norm_by_lr           = {}
    subspace_results_by_lr = {}

    for lr in args.lr:
        lr_tag    = f"lr{lr:.0e}"
        lora_json = os.path.join(args.output_dir, f"lora_{lr_tag}_density.json")
        norm_json = os.path.join(args.output_dir, f"lora_{lr_tag}_norms.json")

        if args.replot and os.path.exists(lora_json):
            print(f"Reloading LoRA lr={lr:.1e} density from JSON...")
            with open(lora_json) as f:
                lora_results_by_lr[lr] = json.load(f)["sigma_results"]
            with open(norm_json) as f:
                norm_by_lr[lr] = json.load(f)["per_param_norm"]
        else:
            # Fine-tune
            finetuned_model, per_param_norm, layer_norms = finetune_lora(
                model, tokenizer, lr=lr,
                train_steps=args.train_steps,
                batch_size=args.train_batch,
                rank=args.lora_rank,
                alpha=args.lora_alpha,
                device=device,
                target_modules=target_modules,
                finetune_fp32=args.finetune_fp32,
                grad_clip=args.grad_clip,
            )
            norm_by_lr[lr] = per_param_norm

            # Save norms
            with open(norm_json, "w") as f:
                json.dump({"lr": lr, "per_param_norm": per_param_norm,
                           "layer_norms": layer_norms,
                           "sigma_half_pretrained": pretrained_sigma_half}, f, indent=2)

            # Exp 1: density sweep on fine-tuned model
            print(f"\n[LoRA lr={lr:.1e}] Running certified density sweep...")
            ft_names  = _get_target_names_auto(finetuned_model, args.model, n_blocks=None)
            ft_results = []
            for sigma in args.sigmas:
                r = estimate_density(finetuned_model, ft_names, inp, tgt, sigma,
                                     N=args.n_perturb, eval_slack=args.eval_slack,
                                     tag=f"lora/{lr_tag}")
                ft_results.append(r)
            lora_results_by_lr[lr] = ft_results

            with open(lora_json, "w") as f:
                json.dump({"model": args.model, "condition": f"lora_{lr_tag}",
                           "lr": lr, "per_param_norm": per_param_norm,
                           "sigma_results": ft_results}, f, indent=2)

            # Exp 3: subspace sampling (optional)
            if args.subspace:
                print(f"\n[LoRA subspace lr={lr:.1e}] Building subspace projector...")
                projector, _ = build_lora_subspace_projector(
                    pretrained_snapshot, finetuned_model, ft_names, n_directions=50
                )
                sub_results = []
                for sigma in args.sigmas:
                    r = estimate_density_subspace(
                        finetuned_model, ft_names, inp, tgt, sigma,
                        N=args.n_perturb, projector=projector,
                        eval_slack=args.eval_slack,
                        tag=f"subspace/{lr_tag}"
                    )
                    sub_results.append(r)
                subspace_results_by_lr[lr] = sub_results

                sub_json = os.path.join(args.output_dir,
                                        f"lora_{lr_tag}_subspace.json")
                with open(sub_json, "w") as f:
                    json.dump({"model": args.model, "lr": lr,
                               "condition": f"subspace_{lr_tag}",
                               "sigma_results": sub_results}, f, indent=2)

                plot_subspace_comparison(ft_results, sub_results, lr, args.output_dir)

    # ── Plots & summary ───────────────────────────────────────────────────────
    plot_before_after(pretrained_results, lora_results_by_lr, args.output_dir,
                      args.model)
    plot_norm_vs_sigma_half(norm_by_lr, pretrained_sigma_half, args.output_dir)
    plot_sigma_half_comparison(
        pretrained_sigma_half,
        {lr: half_max_sigma(res) for lr, res in lora_results_by_lr.items()},
        args.output_dir
    )
    print_summary(args.model, pretrained_results, lora_results_by_lr,
                  norm_by_lr, args.target_density)


if __name__ == "__main__":
    main()
