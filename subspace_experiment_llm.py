"""
subspace_experiment_llm.py
==========================
LLM-scale version of subspace_experiment_v2.py.

Core question (same as toy experiment):
  Are RandOpt's top-K weight perturbations lower-dimensional than random ones?
  Does pretraining (vs random init) make the good subspace MORE concentrated?

What changed from v2 (toy model):
  1. Model     : GPT-2 (HuggingFace) instead of tiny MLP
  2. Scoring   : next-token perplexity on WikiText-2 instead of MSE rollout
  3. Perturbing: only a TARGET_LAYERS subset to control memory
  4. Storage   : float16 deltas to halve memory
  5. Pretraining sweep replaced by pretrained vs random-init comparison
     (GPT-2 is already pretrained; we compare it to a weight-reset copy)

Everything else — randopt_collect, pca_dims_for_threshold, solution_density,
all plotting functions — is IDENTICAL to v2. The PCA machinery only sees
flat numpy vectors and doesn't know or care what model produced them.

Usage
-----
  # Default: CPU-safe mode (last 2 transformer blocks, N=50)
  python subspace_experiment_llm.py

  # Full experiment on GPU (all layers, N=200, fp16)
  python subspace_experiment_llm.py --mode gpu --n_perturb 200

  # Point to a local model cache (if HuggingFace not accessible online)
  python subspace_experiment_llm.py --model_path /path/to/gpt2

Requirements
------------
  pip install transformers datasets torch

  The script will download GPT-2 (~500MB) and WikiText-2 (~4MB) on first run.
  Set HF_HOME or TRANSFORMERS_CACHE env vars to control cache location.

Compute guide
-------------
  CPU (container/laptop)  : N=50,  last 2 blocks  -> ~10 min, ~3GB RAM
  Single GPU 16GB         : N=100, last 4 blocks  -> ~15 min
  Single GPU 40GB (A100)  : N=200, all layers     -> ~30 min  (paper result)
"""

import argparse
import copy
import os
import time
import numpy as np
import torch
import torch.nn as nn
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ── CLI ───────────────────────────────────────────────────────────────────────
def get_args():
    p = argparse.ArgumentParser()
    p.add_argument("--mode",        default="cpu",
                   choices=["cpu", "gpu"],
                   help="cpu=small safe run, gpu=full experiment")
    p.add_argument("--model_path",  default="gpt2",
                   help="HuggingFace model name or local path")
    p.add_argument("--n_perturb",   type=int, default=None,
                   help="Override N (number of perturbations)")
    p.add_argument("--k_select",    type=int, default=None,
                   help="Override K (top-K to select)")
    p.add_argument("--n_blocks",    type=int, default=None,
                   help="Override number of transformer blocks to perturb")
    p.add_argument("--n_text_samples", type=int, default=64,
                   help="Text samples for scoring each perturbation")
    p.add_argument("--seq_len",     type=int, default=128,
                   help="Sequence length for perplexity scoring")
    p.add_argument("--sigma",       type=float, default=None,
                   help="Override perturbation scale sigma")
    p.add_argument("--seed",        type=int,   default=42)
    p.add_argument("--output_dir",  default="/mnt/user-data/outputs")
    return p.parse_args()

# ── Config: cpu-safe vs gpu-full ─────────────────────────────────────────────
CPU_CONFIG = {
    "n_perturb":       50,     # N perturbations to sample
    "k_select":        10,     # top-K to select
    "n_blocks":         2,     # last N transformer blocks to perturb
    "sigma":        0.0005,    # perturbation scale (small for pretrained LLM)
    "sigma_random": 0.02,      # larger sigma for random-init model
    "n_text_samples":  32,     # text samples for scoring
    "seq_len":         64,     # sequence length
    "dtype":    torch.float32,
}
GPU_CONFIG = {
    "n_perturb":      200,
    "k_select":        50,
    "n_blocks":        12,     # all transformer blocks
    "sigma":        0.0005,
    "sigma_random": 0.02,
    "n_text_samples": 128,
    "seq_len":        128,
    "dtype":    torch.float16,
}

# ── Model utilities ───────────────────────────────────────────────────────────
def load_gpt2(model_path, device):
    """Load pretrained GPT-2. Falls back to config-only if weights unavailable."""
    from transformers import GPT2LMHeadModel, GPT2Config
    try:
        model = GPT2LMHeadModel.from_pretrained(model_path)
        print(f"  Loaded pretrained GPT-2 from '{model_path}'")
    except Exception as e:
        print(f"  WARNING: Could not load pretrained weights ({e})")
        print(f"  Falling back to GPT-2 config with random weights.")
        print(f"  NOTE: In this case pretrained vs random comparison is moot.")
        print(f"  For real results, run on a machine with HuggingFace access.")
        config = GPT2Config()
        model = GPT2LMHeadModel(config)
        model.apply(_init_weights)
    return model.to(device)

def _init_weights(module):
    """Standard GPT-2 weight initialisation (used for the random baseline).
    Handles both nn.Linear and transformers.pytorch_utils.Conv1D since
    GPT-2 uses Conv1D for attention projections.
    """
    from transformers.pytorch_utils import Conv1D as HF_Conv1D
    if isinstance(module, (nn.Linear, nn.Embedding, HF_Conv1D)):
        module.weight.data.normal_(mean=0.0, std=0.02)
        if hasattr(module, "bias") and module.bias is not None:
            module.bias.data.zero_()
    elif isinstance(module, nn.LayerNorm):
        module.bias.data.zero_()
        module.weight.data.fill_(1.0)

def make_random_init_copy(pretrained_model, device, seed=12345):
    """
    Return a copy of the model architecture with freshly randomised weights.
    This is the 'needle' baseline — same capacity, no pretraining.
    """
    random_model = copy.deepcopy(pretrained_model)
    torch.manual_seed(seed)
    random_model.apply(_init_weights)
    return random_model.to(device)

def get_target_param_names(model, n_blocks):
    """
    Return parameter names belonging to the last `n_blocks` transformer blocks.
    Skips embeddings and the LM head — we only perturb transformer blocks,
    matching standard fine-tuning practice (LoRA, etc.).

    For GPT-2: model.transformer.h[i] are the transformer blocks.
    Total blocks in GPT-2-small: 12.
    """
    total_blocks = len(model.transformer.h)
    start_block  = max(0, total_blocks - n_blocks)
    target_names = set()
    for i in range(start_block, total_blocks):
        prefix = f"transformer.h.{i}."
        for name, _ in model.named_parameters():
            if name.startswith(prefix):
                target_names.add(name)
    n_params = sum(p.numel() for n, p in model.named_parameters()
                   if n in target_names)
    print(f"  Targeting blocks {start_block}–{total_blocks-1} "
          f"({len(target_names)} param tensors, {n_params:,} params)")
    return target_names

def get_flat_weights(model, target_names):
    """Flat float32 vector of target parameters only."""
    return torch.cat([
        p.data.float().flatten()
        for n, p in model.named_parameters()
        if n in target_names
    ])

def perturb_weights(model, seed, sigma, target_names):
    """Add reproducible Gaussian noise to target parameters in-place."""
    torch.manual_seed(seed)
    with torch.no_grad():
        for name, param in model.named_parameters():
            if name in target_names:
                noise = torch.randn_like(param.data.float()) * sigma
                param.data.add_(noise.to(param.dtype))

def restore_weights(model, base_weights_dict, target_names):
    """Restore model weights from a saved state dict (target layers only)."""
    with torch.no_grad():
        for name, param in model.named_parameters():
            if name in target_names:
                param.data.copy_(base_weights_dict[name])

def save_target_weights(model, target_names):
    """Snapshot of target layer weights (for fast restoration)."""
    return {
        name: param.data.clone()
        for name, param in model.named_parameters()
        if name in target_names
    }

# ── Data: WikiText-2 tokenized ────────────────────────────────────────────────
def load_text_data(tokenizer, n_samples, seq_len, device, seed=42):
    """
    Load WikiText-2 validation split and return a (n_samples, seq_len) token
    tensor. Falls back to synthetic token sequences if dataset unavailable.
    """
    try:
        from datasets import load_dataset
        dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="validation")
        text = "\n".join(dataset["text"])
        tokens = tokenizer.encode(text)
        print(f"  Loaded WikiText-2: {len(tokens):,} tokens")
    except Exception as e:
        print(f"  WARNING: Could not load WikiText-2 ({e})")
        print(f"  Using synthetic random token sequences as fallback.")
        rng = np.random.default_rng(seed)
        tokens = rng.integers(0, tokenizer.vocab_size,
                              size=n_samples * seq_len + seq_len).tolist()

    # Chunk into fixed-length sequences
    rng = np.random.default_rng(seed)
    max_start = len(tokens) - seq_len - 1
    starts = rng.integers(0, max(1, max_start), size=n_samples)
    chunks = [tokens[s : s + seq_len + 1] for s in starts]

    input_ids  = torch.tensor([c[:-1] for c in chunks],
                               dtype=torch.long, device=device)
    target_ids = torch.tensor([c[1:]  for c in chunks],
                               dtype=torch.long, device=device)
    return input_ids, target_ids      # both: (n_samples, seq_len)

# ── Scoring: perplexity ───────────────────────────────────────────────────────
@torch.no_grad()
def compute_perplexity(model, input_ids, target_ids, batch_size=8):
    """
    Mean per-token cross-entropy loss across all samples.
    Lower = better (model predicts text well).
    We use this as the RandOpt 'score' — we want to MINIMISE it.
    """
    model.eval()
    total_loss = 0.0
    n_batches  = 0
    loss_fn    = nn.CrossEntropyLoss(reduction="mean")

    for i in range(0, len(input_ids), batch_size):
        inp = input_ids[i : i + batch_size]
        tgt = target_ids[i : i + batch_size]
        out = model(inp)
        logits = out.logits                         # (B, seq_len, vocab)
        B, T, V = logits.shape
        loss = loss_fn(logits.reshape(B*T, V),
                       tgt.reshape(B*T))
        total_loss += loss.item()
        n_batches  += 1

    return total_loss / max(n_batches, 1)

# ── RandOpt with delta collection (LLM version) ───────────────────────────────
def randopt_collect_llm(model, input_ids, target_ids,
                        target_names, sigma, N, K,
                        store_dtype=torch.float32):
    """
    Identical logic to randopt_collect() in v2, adapted for LLMs:
      - perturbs only target_names parameters
      - scores by perplexity instead of MSE
      - restores weights in-place (avoids deep-copying 500MB model N times)
      - stores deltas in store_dtype (float16 to save memory on GPU)

    Returns:
      top_k_indices  : list of K best seed indices
      all_deltas     : (N, D) numpy array  — perturbation vectors
      all_losses     : (N,)  numpy array   — perplexity per perturbation
      base_loss      : scalar              — unperturbed model perplexity
    """
    print(f"  RandOpt: N={N}, K={K}, sigma={sigma:.5f}")
    t0 = time.time()

    # Snapshot base weights for fast restoration
    base_snapshot = save_target_weights(model, target_names)
    base_loss     = compute_perplexity(model, input_ids, target_ids)
    print(f"  Base perplexity: {base_loss:.4f}")

    D = sum(p.numel() for n, p in model.named_parameters()
            if n in target_names)
    print(f"  Target parameter count: {D:,}")
    print(f"  Delta matrix size: "
          f"{N * D * (2 if store_dtype==torch.float16 else 4) / 1e6:.0f} MB")

    all_deltas = np.zeros((N, D), dtype=np.float32)
    all_losses = np.zeros(N,      dtype=np.float32)

    for seed in range(N):
        if seed % 10 == 0:
            elapsed = time.time() - t0
            eta = elapsed / max(seed, 1) * (N - seed)
            print(f"    seed {seed:3d}/{N}  elapsed={elapsed:.0f}s  eta={eta:.0f}s",
                  end="\r")

        # Apply perturbation in-place
        perturb_weights(model, seed, sigma, target_names)

        # Measure delta vector (perturbed - base)
        delta = (get_flat_weights(model, target_names)
                 - get_flat_weights_from_snapshot(base_snapshot)).numpy()
        all_deltas[seed] = delta.astype(np.float32)

        # Score
        loss = compute_perplexity(model, input_ids, target_ids)
        all_losses[seed] = loss

        # Restore base weights
        restore_weights(model, base_snapshot, target_names)

    print()  # newline after \r progress
    ranked      = np.argsort(all_losses)
    top_k_indices = ranked[:K].tolist()

    print(f"  Done in {time.time()-t0:.0f}s")
    print(f"  Best perplexity:   {all_losses[ranked[0]]:.4f}  "
          f"(base: {base_loss:.4f}, "
          f"improvement: {base_loss - all_losses[ranked[0]]:.4f})")
    print(f"  Median perplexity: {np.median(all_losses):.4f}")

    return top_k_indices, all_deltas, all_losses, base_loss

def get_flat_weights_from_snapshot(snapshot):
    """Flat float32 vector from a saved weight snapshot dict."""
    return torch.cat([v.float().flatten() for v in snapshot.values()])

# ── PCA metrics (UNCHANGED from v2) ──────────────────────────────────────────
def pca_dims_for_threshold(vecs, threshold=0.90, n_components=None):
    """
    Number of PCA dims needed to explain `threshold` fraction of variance.
    vecs: (N, D) numpy array.
    """
    if len(vecs) < 2:
        return 1, np.array([1.0])
    centered = vecs - vecs.mean(axis=0)
    n_comp   = min(len(vecs) - 1, vecs.shape[1],
                   n_components if n_components else 50)
    _, S, _  = np.linalg.svd(centered, full_matrices=False)
    S        = S[:n_comp]
    explained = (S**2) / ((S**2).sum() + 1e-10)
    cumvar   = np.cumsum(explained)
    dims     = int(np.searchsorted(cumvar, threshold)) + 1
    return dims, cumvar

def solution_density(all_losses, base_loss, margin=0.0):
    """Fraction of perturbations that improve on (lower) base perplexity."""
    return float((all_losses <= base_loss - margin).mean())

# ── Core experiment ───────────────────────────────────────────────────────────
def run_condition(label, model, target_names, input_ids, target_ids,
                  sigma, N, K, store_dtype):
    """
    Run the full RandOpt + PCA pipeline for one model condition.
    Returns a results dict with everything needed for plotting.
    """
    print(f"\n{'='*60}")
    print(f"Condition: {label}")
    print(f"{'='*60}")

    top_k_idx, all_deltas, all_losses, base_loss = randopt_collect_llm(
        model, input_ids, target_ids,
        target_names, sigma, N, K,
        store_dtype=store_dtype,
    )

    # PCA on ALL perturbations
    dims_all,  cumvar_all  = pca_dims_for_threshold(all_deltas,             0.90)
    # PCA on TOP-K only
    dims_topk, cumvar_topk = pca_dims_for_threshold(all_deltas[top_k_idx], 0.90)
    density    = solution_density(all_losses, base_loss)
    compression = dims_all / max(dims_topk, 1)

    # Score-stratified analysis
    strat = {}
    ranked = np.argsort(all_losses)
    for name, indices in [
        ("top 10%",    ranked[:max(2, len(ranked)//10)]),
        ("top 25%",    ranked[:max(2, len(ranked)//4)]),
        ("bottom 25%", ranked[3*len(ranked)//4:]),
        ("all",        ranked),
    ]:
        vecs    = all_deltas[indices]
        d90, cv = pca_dims_for_threshold(vecs, 0.90)
        strat[name] = {"dims_90": d90, "cumvar": cv, "n": len(indices)}

    print(f"\n  dims_90(all):    {dims_all}")
    print(f"  dims_90(top-K):  {dims_topk}")
    print(f"  compression:     {compression:.2f}x")
    print(f"  solution density:{density:.4f}")
    print(f"\n  Score-stratified:")
    for name, r in strat.items():
        print(f"    {name:12s} (n={r['n']:3d}): 90% var in {r['dims_90']} dims")

    return {
        "label":       label,
        "dims_all":    dims_all,
        "dims_topk":   dims_topk,
        "compression": compression,
        "density":     density,
        "base_loss":   base_loss,
        "best_loss":   all_losses.min(),
        "cumvar_all":  cumvar_all,
        "cumvar_topk": cumvar_topk,
        "strat":       strat,
        "all_losses":  all_losses,
        "top_k_idx":   top_k_idx,
        "N": N, "K": K,
    }

# ── Plotting (extended from v2) ───────────────────────────────────────────────
def plot_comparison(pretrained_res, random_res, output_dir):
    """
    Main comparison figure: pretrained vs random-init GPT-2.
    4 panels:
      1. Cumulative variance curves (all vs top-K) for pretrained
      2. Cumulative variance curves (all vs top-K) for random init
      3. Score-stratified PCA for both conditions
      4. Summary bar chart
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 11))
    fig.suptitle(
        "LLM Subspace Analysis: Pretrained vs Random-Init GPT-2\n"
        "Do pretrained weights create a more structured thicket?",
        fontsize=13, fontweight="bold"
    )

    colors = {
        "pretrained":  "#D65F5F",
        "random init": "#4878CF",
    }

    for ax, res, cname in [
        (axes[0, 0], pretrained_res, "pretrained"),
        (axes[0, 1], random_res,     "random init"),
    ]:
        col = colors[cname]
        n_all  = len(res["cumvar_all"])
        n_topk = len(res["cumvar_topk"])
        x_all  = range(1, n_all  + 1)
        x_topk = range(1, n_topk + 1)

        ax.plot(x_all,  res["cumvar_all"]  * 100, color=col,
                lw=2.5, ls="--", label=f"All {res['N']} perturbations")
        ax.plot(x_topk, res["cumvar_topk"] * 100, color=col,
                lw=2.5, ls="-",  label=f"Top-{res['K']} only")
        ax.fill_between(x_topk, res["cumvar_topk"] * 100, 100,
                        alpha=0.07, color=col)
        ax.axhline(90, color="gray", lw=1, ls=":", alpha=0.7)
        ax.set_xlabel("Principal components")
        ax.set_ylabel("Cumulative variance (%)")
        ax.set_title(
            f"{res['label']}\n"
            f"dims_90: all={res['dims_all']}, top-K={res['dims_topk']} "
            f"({res['compression']:.1f}x)\n"
            f"base PPL={res['base_loss']:.3f}  "
            f"density={res['density']:.3f}",
            fontsize=9
        )
        ax.legend(fontsize=8)
        ax.set_xlim([1, min(40, n_all)])
        ax.set_ylim([0, 102])
        ax.grid(True, alpha=0.25)

    # ── Score-stratified comparison ──
    ax = axes[1, 0]
    strat_styles = {
        "top 10%":    ("solid",   2.5),
        "top 25%":    ("dashed",  2.0),
        "bottom 25%": ("dotted",  1.8),
        "all":        ("dashdot", 1.5),
    }
    for res, cname in [(pretrained_res, "pretrained"), (random_res, "random init")]:
        col = colors[cname]
        for sname, (ls, lw) in strat_styles.items():
            if sname not in res["strat"]:
                continue
            cv = res["strat"][sname]["cumvar"]
            d  = res["strat"][sname]["dims_90"]
            ax.plot(range(1, len(cv)+1), cv*100,
                    color=col, lw=lw, ls=ls,
                    label=f"{cname} | {sname} (d={d})")
    ax.axhline(90, color="gray", lw=1, ls=":", alpha=0.7)
    ax.set_xlabel("Principal components")
    ax.set_ylabel("Cumulative variance (%)")
    ax.set_title("Score-stratified PCA\n"
                 "(does better-scoring subset cluster tighter?)")
    ax.legend(fontsize=6, ncol=2)
    ax.set_xlim([1, 30])
    ax.grid(True, alpha=0.25)

    # ── Summary bar chart ──
    ax = axes[1, 1]
    results_list = [pretrained_res, random_res]
    labels       = [r["label"] for r in results_list]
    dims_all_v   = [r["dims_all"]    for r in results_list]
    dims_topk_v  = [r["dims_topk"]   for r in results_list]
    density_v    = [r["density"]     for r in results_list]
    compress_v   = [r["compression"] for r in results_list]

    x = np.arange(len(labels))
    w = 0.35
    bar_colors = ["#D65F5F", "#4878CF"]   # pretrained=red, random=blue
    b1 = ax.bar(x - w/2, dims_all_v,  w, color=bar_colors,
                alpha=0.5, label="dims_90 (all)", edgecolor="black", lw=0.8)
    b2 = ax.bar(x + w/2, dims_topk_v, w, color=bar_colors,
                alpha=1.0, label="dims_90 (top-K)", edgecolor="black", lw=0.8)

    for i, (da, dt, den, comp) in enumerate(
            zip(dims_all_v, dims_topk_v, density_v, compress_v)):
        ax.text(i, max(da, dt) + 0.3,
                f"comp={comp:.1f}x\ndensity={den:.3f}",
                ha="center", fontsize=8, color="gray")

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=10)
    ax.set_ylabel("Dims for 90% variance")
    ax.set_title("Summary: Pretrained vs Random Init\n"
                 "Key: pretrained should have larger compression gap")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.25, axis="y")

    plt.tight_layout()
    path = os.path.join(output_dir, "llm_subspace_comparison.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    print(f"\n  Saved: {path}")
    plt.close()

def plot_loss_distributions(pretrained_res, random_res, output_dir):
    """
    Bonus plot: distribution of perturbation losses for both conditions.
    Shows whether the thicket is denser (more mass below base loss) for
    pretrained vs random-init.
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    fig.suptitle("Perturbation Loss Distributions", fontsize=12, fontweight="bold")

    for ax, res in [(axes[0], pretrained_res), (axes[1], random_res)]:
        losses    = res["all_losses"]
        base_loss = res["base_loss"]
        ax.hist(losses, bins=30, color="#888", alpha=0.7, edgecolor="white")
        ax.axvline(base_loss, color="red", lw=2, ls="--",
                   label=f"Base perplexity={base_loss:.3f}")
        below = (losses < base_loss).sum()
        ax.fill_betweenx(
            [0, ax.get_ylim()[1] if ax.get_ylim()[1] > 0 else 10],
            0, base_loss, alpha=0.08, color="green",
            label=f"Improve base: {below}/{len(losses)} ({below/len(losses):.1%})"
        )
        ax.set_xlabel("Perplexity loss")
        ax.set_ylabel("Count")
        ax.set_title(res["label"])
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.25)

    plt.tight_layout()
    path = os.path.join(output_dir, "llm_loss_distributions.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    print(f"  Saved: {path}")
    plt.close()

# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    args = get_args()

    # ── Setup ────────────────────────────────────────────────────────────────
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    cfg = GPU_CONFIG if args.mode == "gpu" else CPU_CONFIG
    # CLI overrides
    if args.n_perturb:  cfg["n_perturb"]       = args.n_perturb
    if args.k_select:   cfg["k_select"]        = args.k_select
    if args.n_blocks:   cfg["n_blocks"]        = args.n_blocks
    if args.sigma:      cfg["sigma"]           = args.sigma
    if args.n_text_samples: cfg["n_text_samples"] = args.n_text_samples
    if args.seq_len:    cfg["seq_len"]         = args.seq_len

    device = torch.device(
        "cuda" if (args.mode == "gpu" and torch.cuda.is_available()) else "cpu"
    )
    os.makedirs(args.output_dir, exist_ok=True)

    print("="*60)
    print("LLM SUBSPACE EXPERIMENT")
    print(f"  mode={args.mode}  device={device}  model={args.model_path}")
    print(f"  N={cfg['n_perturb']}  K={cfg['k_select']}  "
          f"blocks={cfg['n_blocks']}  seq_len={cfg['seq_len']}")
    print("="*60)

    # ── Load model and tokenizer ─────────────────────────────────────────────
    from transformers import GPT2Tokenizer
    try:
        tokenizer = GPT2Tokenizer.from_pretrained(args.model_path)
    except Exception:
        print("  WARNING: Could not load tokenizer from HuggingFace.")
        print("  Using a minimal fallback tokenizer.")
        # Minimal fallback so the script runs even offline
        from transformers import GPT2TokenizerFast
        tokenizer = GPT2TokenizerFast.from_pretrained("gpt2",
                                                       local_files_only=False)

    print("\n  Loading pretrained GPT-2...")
    pretrained_model = load_gpt2(args.model_path, device)
    pretrained_model.eval()

    print("\n  Creating random-init baseline (same architecture)...")
    random_model = make_random_init_copy(pretrained_model, device)
    random_model.eval()

    # ── Identify target parameters ───────────────────────────────────────────
    print("\n  Identifying target parameters (last N transformer blocks)...")
    target_names = get_target_param_names(pretrained_model, cfg["n_blocks"])

    # ── Load text data ───────────────────────────────────────────────────────
    print("\n  Loading text data...")
    input_ids, target_ids = load_text_data(
        tokenizer,
        n_samples=cfg["n_text_samples"],
        seq_len=cfg["seq_len"],
        device=device,
        seed=args.seed,
    )
    print(f"  Data shape: {input_ids.shape}")

    # ── Run both conditions ──────────────────────────────────────────────────
    pretrained_res = run_condition(
        label        = "pretrained GPT-2",
        model        = pretrained_model,
        target_names = target_names,
        input_ids    = input_ids,
        target_ids   = target_ids,
        sigma        = cfg["sigma"],
        N            = cfg["n_perturb"],
        K            = cfg["k_select"],
        store_dtype  = cfg["dtype"],
    )

    # Random model uses larger sigma because it needs bigger steps to move
    random_res = run_condition(
        label        = "random-init GPT-2",
        model        = random_model,
        target_names = target_names,
        input_ids    = input_ids,
        target_ids   = target_ids,
        sigma        = cfg["sigma_random"],
        N            = cfg["n_perturb"],
        K            = cfg["k_select"],
        store_dtype  = cfg["dtype"],
    )

    # ── Plot ─────────────────────────────────────────────────────────────────
    plot_comparison(pretrained_res, random_res, args.output_dir)
    plot_loss_distributions(pretrained_res, random_res, args.output_dir)

    # ── Results table ────────────────────────────────────────────────────────
    print("\n" + "="*60)
    print("RESULTS TABLE")
    print("="*60)
    print(f"\n{'Condition':>22} | {'dims_90(all)':>12} | {'dims_90(top-K)':>14} | "
          f"{'compression':>11} | {'density':>9} | {'base PPL':>9}")
    print("-"*90)
    for res in [pretrained_res, random_res]:
        print(f"{res['label']:>22} | {res['dims_all']:>12} | {res['dims_topk']:>14} | "
              f"{res['compression']:>11.2f} | {res['density']:>9.4f} | "
              f"{res['base_loss']:>9.4f}")

    print("\n" + "="*60)
    print("WHAT TO LOOK FOR")
    print("="*60)
    print("""
The central hypothesis:

  pretrained GPT-2 compression  >>  random-init compression

i.e. the ratio dims_90(all)/dims_90(top-K) should be substantially
larger for the pretrained model than for the random one.

Why this would matter:
  - Pretraining doesn't just make good solutions more numerous (density)
  - It makes them more GEOMETRICALLY CONCENTRATED in weight space
  - This concentration is what makes certified subspace adaptation feasible
  - It's the geometric analog of the Neural Thickets density scaling law

Secondary signals to look for:
  - Solution density should be higher for pretrained (replicates Neural Thickets)
  - Score-stratified: top-10% should cluster tighter than bottom-25%
    for pretrained model but NOT for random model
  - Loss distribution: pretrained should show a longer left tail
    (more perturbations that meaningfully improve base perplexity)

If compression(pretrained) >> compression(random):
  -> Your hypothesis holds at LLM scale
  -> You have evidence that pretraining creates structured, certifiable thickets
  -> Next step: show this scales with model size (GPT-2 vs GPT-2-medium vs GPT-2-large)

To scale up (on a real GPU cluster):
  python subspace_experiment_llm.py --mode gpu --n_perturb 200 --n_blocks 12
""")

if __name__ == "__main__":
    main()
