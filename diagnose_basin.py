"""
diagnose_basin.py
=================
Collects the full NLL distribution of perturbed models (not just pass/fail)
to diagnose the flat density anomaly at lr=5e-4.

The certified density sweep stores only n_better/N.  This script re-runs
with raw NLL collection and generates two figures:

  basin_histograms_lr{lr}.png
      2-row x 3-col grid of NLL histograms at σ ∈ {0.0001, 0.0005, 0.002}.
      Row 0 = pretrained (bowl geometry expected).
      Row 1 = fine-tuned model (plateau / anomalous geometry).
      Each subplot: histogram of NLL(θ+ε), vertical line at threshold.
      Fraction passing = density; annotated per subplot.

  basin_statistics_lr{lr}.png
      Left:  mean NLL shift  E[NLL(θ+ε)] - NLL(θ)  vs σ on log-log axes.
             Expected O(σ²) slope for a smooth minimum; flat/low slope = plateau.
      Right: std of NLL(θ+ε) vs σ on log-log axes.
             Tells us whether the basin is wide at all scales (plateau) or
             narrows at small σ (proper minimum).

Usage (on RunPod A40):
  python diagnose_basin.py --model gpt2 --lr 5e-4          # primary diagnostic
  python diagnose_basin.py --model gpt2 --lr 1e-4          # control (bowl geometry)
  python diagnose_basin.py --replot                         # reload JSON, replot only

Output:
  --output_dir/basin_raw_lr{lr}.json
  --output_dir/basin_histograms_lr{lr}.png
  --output_dir/basin_statistics_lr{lr}.png
"""

import argparse
import json
import os
import sys
import time

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

sys.path.insert(0, os.path.dirname(__file__))
from certified_density_experiment import (
    load_model, get_target_param_names,
    save_weights, restore_weights, perturb_weights,
    load_data, compute_nll_loss,
)
from lora_density_experiment import finetune_lora

DIAG_SIGMAS    = [0.0001, 0.0002, 0.0005, 0.001, 0.002, 0.005]
HIST_SIGMAS    = [0.0001, 0.0005, 0.002]   # 3 columns in the histogram grid
EVAL_SLACK     = 1e-4
N_PERTURB      = 200
N_TEXT_SAMPLES = 128
SEQ_LEN        = 128


def get_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model",        default="gpt2")
    p.add_argument("--lr",           type=float, default=5e-4)
    p.add_argument("--lora_rank",    type=int,   default=8)
    p.add_argument("--lora_alpha",   type=int,   default=16)
    p.add_argument("--train_steps",  type=int,   default=500)
    p.add_argument("--train_batch",  type=int,   default=16)
    p.add_argument("--n_perturb",    type=int,   default=N_PERTURB)
    p.add_argument("--eval_slack",   type=float, default=EVAL_SLACK)
    p.add_argument("--replot",       action="store_true",
                   help="Skip fine-tuning and NLL collection; reload JSON and replot.")
    p.add_argument("--output_dir",   default="./results/basin_diagnosis")
    return p.parse_args()


# ── Core: collect raw NLL distribution ───────────────────────────────────────
@torch.no_grad()
def collect_nll_distribution(model, names, inp, tgt, sigma, N,
                              eval_slack=EVAL_SLACK, tag=""):
    """
    Returns (base_nll, nlls) where nlls is a float32 numpy array of length N.
    Each entry is NLL(θ + ε_i) for a fresh Gaussian perturbation ε_i ~ N(0, σ²I).
    Weights are restored after each sample — model state is unchanged on return.
    """
    t0       = time.time()
    snapshot = save_weights(model, names)
    base_nll = compute_nll_loss(model, inp, tgt)
    nlls     = np.empty(N, dtype=np.float32)

    for seed in range(N):
        if seed % 40 == 0:
            elapsed = time.time() - t0
            eta     = elapsed / max(seed, 1) * (N - seed)
            print(f"  [{tag}] σ={sigma:.5f}  {seed}/{N}  "
                  f"{elapsed:.0f}s elapsed  {eta:.0f}s eta", end="\r", flush=True)
        perturb_weights(model, seed, sigma, names)
        nlls[seed] = compute_nll_loss(model, inp, tgt)
        restore_weights(model, snapshot, names)

    density = float(np.mean(nlls <= base_nll + eval_slack))
    print(f"  [{tag}] σ={sigma:.5f}  density={density:.3f}  "
          f"mean_shift={np.mean(nlls) - base_nll:+.5f}  "
          f"std={np.std(nlls):.5f}  ({time.time()-t0:.0f}s)")
    return float(base_nll), nlls


def run_collection(model, names, inp, tgt, sigmas, n_perturb, eval_slack, tag):
    results = {}
    for sigma in sigmas:
        base_nll, nlls = collect_nll_distribution(
            model, names, inp, tgt, sigma, n_perturb, eval_slack, tag=tag
        )
        results[sigma] = {"base_nll": base_nll, "nlls": nlls.tolist()}
    return results


# ── Plotting ─────────────────────────────────────────────────────────────────
def plot_histograms(pretrained_data, lora_data, lr, eval_slack, output_dir):
    """
    2-row × 3-col histogram grid.
    Row 0 = pretrained, Row 1 = fine-tuned.
    Columns = σ ∈ HIST_SIGMAS.
    Each bar is colored green (below threshold) or red (above threshold).
    """
    fig, axes = plt.subplots(2, 3, figsize=(15, 8), sharey=False)
    row_labels = ["Pretrained", f"LoRA lr={lr:.0e}"]
    datasets   = [pretrained_data, lora_data]

    for row, (label, data) in enumerate(zip(row_labels, datasets)):
        for col, sigma in enumerate(HIST_SIGMAS):
            ax       = axes[row, col]
            entry    = data.get(sigma, {})
            base_nll = entry.get("base_nll", float("nan"))
            nlls     = np.array(entry.get("nlls", []))
            if len(nlls) == 0:
                ax.set_title(f"σ={sigma:.4f}  (no data)")
                continue

            threshold = base_nll + eval_slack
            density   = float(np.mean(nlls <= threshold))
            mean_shift = np.mean(nlls) - base_nll

            # Bin and colour
            bins = np.linspace(nlls.min() - 0.001, nlls.max() + 0.001, 40)
            for i in range(len(bins) - 1):
                mask = (nlls >= bins[i]) & (nlls < bins[i + 1])
                cnt  = mask.sum()
                if cnt == 0:
                    continue
                mid   = (bins[i] + bins[i + 1]) / 2
                color = "#2ca02c" if mid <= threshold else "#d62728"
                ax.bar(mid, cnt, width=(bins[1] - bins[0]) * 0.85,
                       color=color, alpha=0.75, edgecolor="none")

            ax.axvline(base_nll,  color="black",  lw=1.5, ls="--",
                       label=f"base_nll={base_nll:.4f}")
            ax.axvline(threshold, color="#ff7f0e", lw=1.5, ls=":",
                       label=f"threshold (+{eval_slack:.0e})")
            ax.set_title(
                f"σ={sigma:.4f}   density={density:.3f}   "
                f"shift={mean_shift:+.5f}",
                fontsize=9
            )
            ax.set_xlabel("NLL(θ + ε)", fontsize=8)
            ax.set_ylabel("count", fontsize=8)
            ax.tick_params(labelsize=7)

            if row == 0 and col == 0:
                legend_elems = [
                    Line2D([0], [0], color="#2ca02c", lw=6, alpha=0.75, label="pass"),
                    Line2D([0], [0], color="#d62728", lw=6, alpha=0.75, label="fail"),
                    Line2D([0], [0], color="black",   lw=1.5, ls="--", label="base_nll"),
                    Line2D([0], [0], color="#ff7f0e", lw=1.5, ls=":",  label="threshold"),
                ]
                ax.legend(handles=legend_elems, fontsize=7, loc="upper left")

        axes[row, 0].set_ylabel(f"{label}\ncount", fontsize=9)

    fig.suptitle(
        f"NLL Distribution Diagnostic — {args.model}  lr={lr:.0e}\n"
        f"Green = perturbations that pass the certificate threshold, "
        f"Red = fail  (N={args.n_perturb} per σ)",
        fontsize=11
    )
    plt.tight_layout()
    tag = f"{lr:.0e}".replace("-", "n")
    path = os.path.join(output_dir, f"basin_histograms_lr{tag}.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    print(f"Saved: {path}")
    plt.close()


def plot_statistics(pretrained_data, lora_data, lr, eval_slack, sigmas, output_dir):
    """
    Left:  mean NLL shift vs σ (log-log, slope ≈ 2 for convex minimum).
    Right: std of NLL distribution vs σ (log-log).
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

    for label, data, color, marker in [
        ("Pretrained",     pretrained_data, "steelblue", "o"),
        (f"LoRA lr={lr:.0e}", lora_data,   "crimson",   "s"),
    ]:
        sig_vals, shifts, stds = [], [], []
        for sigma in sorted(sigmas):
            entry = data.get(sigma, {})
            nlls  = np.array(entry.get("nlls", []))
            if len(nlls) == 0:
                continue
            base_nll = entry["base_nll"]
            shift    = np.mean(nlls) - base_nll
            std      = float(np.std(nlls))
            if shift > 0:
                sig_vals.append(sigma)
                shifts.append(shift)
                stds.append(std)

        if sig_vals:
            ax1.plot(sig_vals, shifts, color=color, marker=marker,
                     lw=2, ms=6, label=label)
            ax2.plot(sig_vals, stds,   color=color, marker=marker,
                     lw=2, ms=6, label=label)

    # Reference slope-2 line on ax1 (O(σ²) expected for smooth minimum)
    if len(sig_vals) >= 2:
        ref_x = np.array([min(sig_vals), max(sig_vals)])
        scale = shifts[0] / (ref_x[0] ** 2) if shifts else 1.0
        ax1.plot(ref_x, scale * ref_x ** 2, "k--", lw=1, alpha=0.5, label="O(σ²) ref")

    for ax, ylabel, title in [
        (ax1,
         "E[NLL(θ+ε)] − NLL(θ)  (mean shift)",
         "Mean NLL Shift vs σ\n"
         "Slope ≈ 2 (log-log) → convex minimum;  flat → plateau"),
        (ax2,
         "std  NLL(θ+ε)",
         "NLL Distribution Width vs σ\n"
         "Wide at all σ → basin is flat / not a proper minimum"),
    ]:
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("σ  (perturbation scale, log)", fontsize=11)
        ax.set_ylabel(ylabel, fontsize=10)
        ax.set_title(title, fontsize=10)
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3, which="both")

    fig.suptitle(
        f"Basin Geometry — {args.model}  lr={lr:.0e}  "
        f"(N={args.n_perturb} perturbations per σ)",
        fontsize=12
    )
    plt.tight_layout()
    tag = f"{lr:.0e}".replace("-", "n")
    path = os.path.join(output_dir, f"basin_statistics_lr{tag}.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    print(f"Saved: {path}")
    plt.close()


# ── Main ─────────────────────────────────────────────────────────────────────
args = get_args()   # module-level so plot_histograms can read args.model

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.output_dir, exist_ok=True)
    lr_tag = f"{args.lr:.0e}".replace("-", "n")
    json_path = os.path.join(args.output_dir, f"basin_raw_lr{lr_tag}.json")

    print(f"Device: {device}  |  Output: {args.output_dir}")

    from transformers import GPT2Tokenizer
    tokenizer = GPT2Tokenizer.from_pretrained(args.model)
    tokenizer.pad_token = tokenizer.eos_token

    if args.replot and os.path.exists(json_path):
        print("Reloading from JSON (--replot)...")
        with open(json_path) as f:
            saved = json.load(f)
        pretrained_data = {float(k): v for k, v in saved["pretrained"].items()}
        lora_data       = {float(k): v for k, v in saved["lora"].items()}
    else:
        print(f"\nLoading {args.model}...")
        model, _ = load_model(args.model, device)
        names    = get_target_param_names(model, n_blocks=None)
        inp, tgt = load_data(tokenizer, N_TEXT_SAMPLES, SEQ_LEN, device)

        # ── Pretrained NLL distribution ───────────────────────────────────────
        print("\n[Pretrained] Collecting NLL distributions...")
        pretrained_data = run_collection(
            model, names, inp, tgt,
            sigmas=DIAG_SIGMAS, n_perturb=args.n_perturb,
            eval_slack=args.eval_slack, tag="pretrained"
        )

        # ── Fine-tune with LoRA ───────────────────────────────────────────────
        print(f"\n[LoRA lr={args.lr:.0e}] Fine-tuning...")
        finetuned_model, per_param_norm, _ = finetune_lora(
            model, tokenizer,
            lr=args.lr,
            train_steps=args.train_steps,
            batch_size=args.train_batch,
            rank=args.lora_rank,
            alpha=args.lora_alpha,
            device=device,
        )
        ft_names = get_target_param_names(finetuned_model, n_blocks=None)

        # ── Fine-tuned NLL distribution ───────────────────────────────────────
        print(f"\n[LoRA lr={args.lr:.0e}] Collecting NLL distributions...")
        lora_data = run_collection(
            finetuned_model, ft_names, inp, tgt,
            sigmas=DIAG_SIGMAS, n_perturb=args.n_perturb,
            eval_slack=args.eval_slack, tag=f"lora_lr{lr_tag}"
        )

        with open(json_path, "w") as f:
            json.dump({
                "model": args.model, "lr": args.lr,
                "n_perturb": args.n_perturb, "eval_slack": args.eval_slack,
                "per_param_norm": per_param_norm,
                "pretrained": {str(k): v for k, v in pretrained_data.items()},
                "lora":       {str(k): v for k, v in lora_data.items()},
            }, f, indent=2)
        print(f"\nRaw NLLs saved: {json_path}")

    # ── Plots ─────────────────────────────────────────────────────────────────
    plot_histograms(pretrained_data, lora_data, args.lr, args.eval_slack, args.output_dir)
    plot_statistics(pretrained_data, lora_data, args.lr, args.eval_slack, DIAG_SIGMAS, args.output_dir)
    print("\nDone.")


if __name__ == "__main__":
    main()
