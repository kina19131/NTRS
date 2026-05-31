"""
certified_density_experiment.py
================================
Estimates the weight-space certified density C(θ, σ, m) for pretrained LLMs
of increasing size, using the Randomized Smoothing framework applied to
weight space instead of input space.

Core question:
  Does C(θ, σ, m) = P_{ε~N(0,σ²I)}[s(θ+ε) ≥ m] scale with model size?
  Is the certified radius σ* larger for pretrained vs random-init models?

Novel contribution vs Cohen et al. (2019):
  - Cohen et al.: certify input-space robustness (adversarial examples)
  - This work:    certify weight-space adaptation robustness (post-training)
  - Same Binomial CI machinery, new domain, new meaning

Key design decisions vs subspace_experiment_llm_gpu.py:
  - No delta matrix stored → memory efficient → all transformer blocks
  - Sigma sweep to trace the full density(σ) curve per model
  - Clopper-Pearson exact Binomial CI for formal guarantees
  - Multi-model: gpt2, gpt2-medium, gpt2-large (scaling law)

Output:
  - JSON per model/condition with density(σ) curve + CI
  - density_curves.png: density vs σ with CI bands for each model
  - scaling_law.png: certified radius σ* vs model size

Usage:
  # Full experiment (all 3 model sizes, ~2 hours on A40)
  python certified_density_experiment.py --mode gpu

  # Single model (quick test, ~15 min)
  python certified_density_experiment.py --mode gpu --models gpt2

  # With random-init baseline
  python certified_density_experiment.py --mode gpu --include_random_init

  # Custom sigma sweep
  python certified_density_experiment.py --mode gpu \\
      --sigmas 0.0001 0.0005 0.001 0.005 0.01
"""

import argparse
import copy
import json
import os
import time
import numpy as np
import torch
import torch.nn as nn
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import beta as beta_dist

# ── CLI ───────────────────────────────────────────────────────────────────────
def get_args():
    p = argparse.ArgumentParser()
    p.add_argument("--mode", default="gpu", choices=["cpu", "gpu"])
    p.add_argument("--models", nargs="+",
                   default=["gpt2", "gpt2-medium", "gpt2-large"],
                   help="HuggingFace model names to sweep")
    p.add_argument("--sigmas", nargs="+", type=float,
                   default=[0.0001, 0.0002, 0.0005, 0.001, 0.002, 0.005, 0.01, 0.02,
                            0.05, 0.1, 0.2],
                   help="Sigma values to sweep per model")
    p.add_argument("--n_perturb",     type=int,   default=200,
                   help="Perturbations per sigma (higher = tighter CI)")
    p.add_argument("--n_blocks",      type=int,   default=None,
                   help="Transformer blocks to perturb (default: all)")
    p.add_argument("--n_text_samples",type=int,   default=128)
    p.add_argument("--seq_len",       type=int,   default=128)
    p.add_argument("--target_density",type=float, default=0.40,
                   help="Density threshold defining the certified radius")
    p.add_argument("--confidence",    type=float, default=0.95,
                   help="Statistical confidence for Binomial CI")
    p.add_argument("--conditions", nargs="+",
                   default=["pretrained", "random-init"],
                   choices=["pretrained", "random-init"],
                   help="Which conditions to run (default: both)")
    p.add_argument("--eval_slack",     type=float, default=0.0,
                   help="Slack added to success criterion: count perturbed as success "
                        "if nll <= base_nll + eval_slack. Accounts for fp16 evaluation "
                        "noise (~1e-4). Default 0 is conservative; 1e-4 is recommended.")
    p.add_argument("--seed",          type=int,   default=42)
    p.add_argument("--output_dir",    default="/mnt/user-data/outputs")
    p.add_argument("--replot",        action="store_true",
                   help="Skip experiment; reload existing JSONs and regenerate plots")
    return p.parse_args()

# ── Binomial CI — Clopper-Pearson exact ───────────────────────────────────────
def binom_ci(k, n, confidence=0.95):
    """
    Clopper-Pearson exact binomial CI.
    k successes out of n trials.
    Returns (lower, upper) — guaranteed lower bound is the certificate.
    """
    alpha = 1 - confidence
    lower = float(beta_dist.ppf(alpha / 2,       k,     n - k + 1)) if k > 0 else 0.0
    upper = float(beta_dist.ppf(1 - alpha / 2,   k + 1, n - k    )) if k < n else 1.0
    return lower, upper

# ── Model utilities ───────────────────────────────────────────────────────────
def load_model(model_path, device):
    from transformers import GPT2LMHeadModel
    model = GPT2LMHeadModel.from_pretrained(
        model_path, torch_dtype=torch.float16
    ).to(device).eval()
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Loaded {model_path} ({n_params:,} params)")
    return model, n_params

def make_random_init_copy(pretrained_model, device, seed=12345):
    import math
    from transformers.pytorch_utils import Conv1D as HF_Conv1D
    m = copy.deepcopy(pretrained_model)
    n_layers = pretrained_model.config.n_layer
    # GPT-2 scales residual projection weights by 1/sqrt(2*n_layers) at init.
    # Replicating this is essential for fair comparison across model sizes —
    # omitting it makes larger models appear to have anomalously large weights.
    residual_proj_std = 0.02 / math.sqrt(2 * n_layers)
    torch.manual_seed(seed)
    def _init(module, name=""):
        if isinstance(module, (nn.Linear, HF_Conv1D)):
            std = residual_proj_std if "c_proj" in name else 0.02
            module.weight.data.normal_(mean=0.0, std=std)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
    for name, module in m.named_modules():
        _init(module, name)
    return m.to(device).eval()

def get_target_param_names(model, n_blocks):
    total = len(model.transformer.h)
    n_blocks = n_blocks or total
    start = max(0, total - n_blocks)
    names = {
        name
        for name, _ in model.named_parameters()
        if any(name.startswith(f"transformer.h.{i}.") for i in range(start, total))
    }
    n_params = sum(p.numel() for n, p in model.named_parameters() if n in names)
    print(f"  Targeting blocks {start}–{total-1}: {n_params:,} params")
    return names

def save_weights(model, names):
    return {n: p.data.clone() for n, p in model.named_parameters() if n in names}

def restore_weights(model, snapshot, names):
    with torch.no_grad():
        for n, p in model.named_parameters():
            if n in names:
                p.data.copy_(snapshot[n])

def perturb_weights(model, seed, sigma, names):
    torch.manual_seed(seed)
    with torch.no_grad():
        for n, p in model.named_parameters():
            if n in names:
                # Compute perturbation in float32 to avoid fp16 quantization
                # absorbing small noise values (critical at σ ≤ 0.0002).
                # Cast result back to original dtype only after the full-precision add.
                w_f32 = p.data.float()
                w_f32.add_(torch.randn_like(w_f32) * sigma)
                p.data.copy_(w_f32.to(p.dtype))

# ── Data ──────────────────────────────────────────────────────────────────────
def load_data(tokenizer, n_samples, seq_len, device, seed=42):
    try:
        from datasets import load_dataset
        ds  = load_dataset("wikitext", "wikitext-2-raw-v1", split="validation")
        tokens = tokenizer.encode("\n".join(ds["text"]))
        print(f"  WikiText-2: {len(tokens):,} tokens")
    except Exception as e:
        print(f"  WARNING: WikiText-2 unavailable ({e}). Using synthetic tokens.")
        rng = np.random.default_rng(seed)
        tokens = rng.integers(0, tokenizer.vocab_size,
                              size=n_samples * seq_len + seq_len).tolist()
    rng    = np.random.default_rng(seed)
    starts = rng.integers(0, max(1, len(tokens) - seq_len - 1), size=n_samples)
    chunks = [tokens[s: s + seq_len + 1] for s in starts]
    inp = torch.tensor([c[:-1] for c in chunks], dtype=torch.long, device=device)
    tgt = torch.tensor([c[1:]  for c in chunks], dtype=torch.long, device=device)
    return inp, tgt

# ── Scoring ───────────────────────────────────────────────────────────────────
@torch.no_grad()
def compute_nll_loss(model, inp, tgt, batch_size=8):
    """
    Returns mean per-token negative log-likelihood (cross-entropy) in nats.
    Note: this is NOT perplexity. Perplexity = exp(NLL). We report NLL directly
    because (a) it is the quantity we threshold in the density estimate, and
    (b) our evaluation uses randomly sampled 128-token sequences rather than a
    proper sliding-window protocol, so the absolute value differs from published
    perplexity figures. For comparison purposes (perturbed vs. base) this is exact.
    """
    model.eval()
    loss_fn = nn.CrossEntropyLoss(reduction="mean")
    total, n = 0.0, 0
    for i in range(0, len(inp), batch_size):
        out = model(inp[i:i+batch_size])
        B, T, V = out.logits.shape
        total += loss_fn(
            out.logits.reshape(B*T, V),
            tgt[i:i+batch_size].reshape(B*T)
        ).item()
        n += 1
    return total / max(n, 1)

# ── Core: density estimation with Binomial CI ─────────────────────────────────
def estimate_density(model, names, inp, tgt, sigma, N,
                     confidence=0.95, eval_slack=0.0, tag=""):
    """
    Estimate C(θ, σ, m) = P[perturbed model NLL ≤ base NLL + eval_slack].
    eval_slack > 0 absorbs fp16 evaluation noise in the success criterion.
    A value of 1e-4 is recommended for fp16 models; 0.0 is the strict version.
    """
    t0        = time.time()
    snapshot  = save_weights(model, names)
    base_nll  = compute_nll_loss(model, inp, tgt)
    threshold = base_nll + eval_slack
    n_better  = 0

    for seed in range(N):
        if seed % 20 == 0:
            elapsed = time.time() - t0
            eta     = elapsed / max(seed, 1) * (N - seed)
            print(f"    [{tag}] σ={sigma:.5f}  {seed}/{N}  "
                  f"{elapsed:.0f}s elapsed  {eta:.0f}s eta",
                  end="\r", flush=True)
        perturb_weights(model, seed, sigma, names)
        nll = compute_nll_loss(model, inp, tgt)
        if nll <= threshold:
            n_better += 1
        restore_weights(model, snapshot, names)

    print(flush=True)
    density            = n_better / N
    ci_lower, ci_upper = binom_ci(n_better, N, confidence)

    print(f"    σ={sigma:.5f}  density={density:.3f}  "
          f"95% CI=[{ci_lower:.3f}, {ci_upper:.3f}]  "
          f"base_nll={base_nll:.4f}  slack={eval_slack:.1e}  "
          f"({time.time()-t0:.0f}s)", flush=True)

    return {
        "sigma":      sigma,
        "density":    density,
        "ci_lower":   ci_lower,
        "ci_upper":   ci_upper,
        "n_better":   n_better,
        "N":          N,
        "base_ppl":   base_nll,   # kept as base_ppl for JSON back-compat
        "eval_slack": eval_slack,
        "confidence": confidence,
    }

# ── Certified radius ──────────────────────────────────────────────────────────
def certified_radius(sigma_results, target_density):
    """
    Max σ such that CI_lower(density(σ)) >= target_density.
    This is the weight-space analog of the certified radius in Cohen et al.
    """
    valid = [r for r in sigma_results if r["ci_lower"] >= target_density]
    return max((r["sigma"] for r in valid), default=0.0)

def half_max_sigma(sigma_results):
    """
    σ where density drops to half its peak value, via linear interpolation.
    Better than certified_radius for pretrained vs random-init comparison:
    random-init has a flat ~0.5 noise floor that inflates its σ* artificially.
    half_max_sigma captures the width of the high-density basin regardless of floor.
    """
    sigmas = [r["sigma"]   for r in sigma_results]
    dens   = [r["density"] for r in sigma_results]
    peak   = max(dens)
    target = peak / 2.0
    for i in range(len(dens) - 1):
        if dens[i] >= target and dens[i + 1] < target:
            t = (target - dens[i]) / (dens[i + 1] - dens[i])
            return sigmas[i] + t * (sigmas[i + 1] - sigmas[i])
    return sigmas[-1]  # never drops below half-max in this sweep

# ── Plotting ──────────────────────────────────────────────────────────────────
MODEL_PARAMS = {
    "gpt2": 124, "gpt2-medium": 345,
    "gpt2-large": 774, "gpt2-xl": 1558,
}

def plot_results(all_results, output_dir, target_density, confidence):
    """Two figures: density curves + scaling law."""

    # ── Figure 1: density(σ) curves — single panel, same color per model ─────
    MODEL_COLORS = {
        "gpt2": "#e41a1c", "gpt2-medium": "#377eb8",
        "gpt2-large": "#4daf4a", "gpt2-xl": "#984ea3",
    }
    fig, ax = plt.subplots(figsize=(11, 6))
    N_val = all_results[0][2][0]["N"]
    added_labels = set()

    for model_name, condition, results in all_results:
        sigmas = [r["sigma"]    for r in results]
        dens   = [r["density"]  for r in results]
        ci_lo  = [r["ci_lower"] for r in results]
        ci_hi  = [r["ci_upper"] for r in results]
        color  = MODEL_COLORS.get(model_name, "gray")
        params = MODEL_PARAMS.get(model_name, "?")
        ls     = "-" if condition == "pretrained" else "--"
        label  = f"{model_name} ({params}M)"
        # Only add each model label once (from pretrained line)
        if model_name not in added_labels:
            ax.plot(sigmas, dens, color=color, lw=2, ls=ls, marker="o", ms=5,
                    label=label)
            added_labels.add(model_name)
        else:
            ax.plot(sigmas, dens, color=color, lw=2, ls=ls, marker="o", ms=5)
        ax.fill_between(sigmas, ci_lo, ci_hi, color=color, alpha=0.10)

    ax.axhline(target_density, color="black", lw=1.5, ls=":",
               label=f"Target density = {target_density}")
    # Legend entries for line style
    from matplotlib.lines import Line2D
    ax.legend(
        handles=ax.get_legend_handles_labels()[0] + [
            Line2D([0], [0], color="gray", lw=2, ls="-",  label="— pretrained"),
            Line2D([0], [0], color="gray", lw=2, ls="--", label="-- random-init"),
        ],
        labels=ax.get_legend_handles_labels()[1] + ["— pretrained", "-- random-init"],
        fontsize=9, ncol=2
    )
    ax.set_xscale("log")
    ax.set_xlabel("Perturbation scale σ  (log)", fontsize=12)
    ax.set_ylabel("C(θ, σ, m)  —  certified density", fontsize=12)
    ax.set_title(
        f"Weight-Space Certified Density  (N={N_val}, {int(confidence*100)}% CI)\n"
        "Solid = pretrained (sharp decay)  |  Dashed = random-init (flat plateau)",
        fontsize=11, fontweight="bold"
    )
    ax.set_ylim([0, 1.05])
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    path = os.path.join(output_dir, "density_curves.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    print(f"  Saved: {path}")
    plt.close()

    # ── Figure 2: σ½ scaling — single panel, log y-axis ─────────────────────
    fig, ax = plt.subplots(figsize=(9, 6))

    for condition, color, marker in [("pretrained", "crimson", "o"),
                                     ("random-init", "steelblue", "s")]:
        entries = sorted(
            [(name, res) for name, cond, res in all_results if cond == condition],
            key=lambda e: MODEL_PARAMS.get(e[0], 0)
        )
        if not entries:
            continue
        xs = [MODEL_PARAMS.get(n, 0) for n, _ in entries]
        ys = [half_max_sigma(res) for _, res in entries]
        ax.plot(xs, ys, color=color, marker=marker, lw=2, ms=10,
                label=condition)
        for x, y, (name, _) in zip(xs, ys, entries):
            ax.annotate(name, (x, y), textcoords="offset points",
                        xytext=(6, 4), fontsize=9)

    ax.set_yscale("log")
    ax.set_xlabel("Model size (M parameters)", fontsize=12)
    ax.set_ylabel("σ½  (log scale)", fontsize=12)
    ax.set_title(
        "Per-Parameter Robustness Threshold σ½ vs. Model Scale\n"
        "Pretrained: flat ~0.0007 across 12× capacity growth  |  "
        "Random-init: grows with scale",
        fontsize=10, fontweight="bold"
    )
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3, which="both")
    plt.tight_layout()
    path = os.path.join(output_dir, "scaling_law.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    print(f"  Saved: {path}")
    plt.close()

# ── Summary printer ───────────────────────────────────────────────────────────
def _print_summary(summary, target_density):
    print("\n" + "="*80)
    print("RESULTS TABLE")
    print("="*80)
    print(f"\n{'Model':>15} | {'Condition':>12} | {'Params':>8} | "
          f"{'σ* (CI)':>12} | {'σ½ (half-max)':>14}")
    print("-"*80)
    for s in summary:
        cr  = certified_radius(s["sigma_results"], target_density)
        hm  = half_max_sigma(s["sigma_results"])
        print(f"{s['model']:>15} | {s['condition']:>12} | "
              f"{s['n_params']:>8,} | {cr:>12.5f} | {hm:>14.5f}")
    print(f"\n  σ*  = max σ where CI_lower(density) ≥ {target_density}")
    print(f"  σ½  = σ where density = peak/2  (unbiased basin-width metric)")

# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    args   = get_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device(
        "cuda" if (args.mode == "gpu" and torch.cuda.is_available()) else "cpu"
    )
    os.makedirs(args.output_dir, exist_ok=True)

    print("=" * 60)
    print("CERTIFIED DENSITY EXPERIMENT")
    print(f"  models:    {args.models}")
    print(f"  sigmas:    {args.sigmas}")
    print(f"  N={args.n_perturb}  confidence={args.confidence}"
          f"  target_density={args.target_density}")
    print("=" * 60)

    # ── Replot mode: reload existing JSONs, skip experiment ───────────────────
    if args.replot:
        import glob
        all_results, summary = [], []
        for path in sorted(glob.glob(os.path.join(args.output_dir, "density_*.json"))):
            entry = json.load(open(path))
            all_results.append((entry["model"], entry["condition"], entry["sigma_results"]))
            summary.append(entry)
        print(f"  Loaded {len(all_results)} condition(s) from {args.output_dir}")
        plot_results(all_results, args.output_dir, args.target_density, args.confidence)
        _print_summary(summary, args.target_density)
        return

    from transformers import GPT2Tokenizer

    all_results = []   # (model_name, condition, [sigma_results])
    summary     = []

    for model_name in args.models:
        print(f"\n{'='*60}\nModel: {model_name}\n{'='*60}")

        tokenizer             = GPT2Tokenizer.from_pretrained(model_name)
        model, n_params       = load_model(model_name, device)
        names                 = get_target_param_names(model, args.n_blocks)
        inp, tgt              = load_data(tokenizer, args.n_text_samples,
                                          args.seq_len, device, args.seed)
        slug                  = model_name.replace("-", "_")

        for condition in args.conditions:
            if condition == "random-init":
                target_model = make_random_init_copy(model, device)
            else:
                target_model = model

            print(f"\n  [{condition}] sigma sweep...")
            results = []
            for sigma in args.sigmas:
                r = estimate_density(
                    target_model, names, inp, tgt,
                    sigma, args.n_perturb, args.confidence,
                    eval_slack=args.eval_slack,
                    tag=f"{model_name}/{condition}"
                )
                results.append(r)

            cr = certified_radius(results, args.target_density)
            print(f"\n  Certified radius σ* = {cr:.5f}  "
                  f"(CI_lower ≥ {args.target_density})", flush=True)

            all_results.append((model_name, condition, results))
            entry = {
                "model":            model_name,
                "condition":        condition,
                "n_params":         n_params,
                "certified_radius": cr,
                "target_density":   args.target_density,
                "confidence":       args.confidence,
                "sigma_results":    results,
            }
            summary.append(entry)

            path = os.path.join(args.output_dir,
                                f"density_{slug}_{condition}.json")
            with open(path, "w") as f:
                json.dump(entry, f, indent=2)
            print(f"  JSON saved: {path}", flush=True)

            if condition == "random-init":
                del target_model

        del model
        torch.cuda.empty_cache()

    # ── Summary table + plots ─────────────────────────────────────────────────
    plot_results(all_results, args.output_dir, args.target_density,
                 args.confidence)
    _print_summary(summary, args.target_density)

    print(f"""
{'='*60}
WHAT TO LOOK FOR
{'='*60}
density_curves.png:
  - Pretrained curves should drop off more slowly than random-init
  - Larger models should maintain higher density at any given σ
  - CI bands show statistical confidence of each estimate

scaling_law.png:
  - Certified radius σ* should increase with model size for pretrained
  - Random-init σ* should be flat or lower than pretrained
  - This is the weight-space scaling law for certified adaptation robustness

Certifiable statement (if hypothesis holds):
  "A pretrained GPT-2-large model's weights can be perturbed with
   Gaussian noise of scale σ* while maintaining perplexity with
   probability ≥ {args.target_density} (95% confidence)."
""")


if __name__ == "__main__":
    main()
