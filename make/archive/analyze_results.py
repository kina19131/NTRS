"""
analyze_results.py
==================
Produces richer output from certified density JSON results:
  1. Density curves with visible CI error bars
  2. LaTeX-ready results table
  3. Formal certificate cards per model

Usage:
  python analyze_results.py --results_dir ./results/certified_density/final
"""

import argparse
import json
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from certified_density_experiment import half_max_sigma, certified_radius

MODEL_PARAMS = {"gpt2": 124, "gpt2-medium": 355, "gpt2-large": 774, "gpt2-xl": 1558}
MODEL_ORDER  = ["gpt2", "gpt2-medium", "gpt2-large", "gpt2-xl"]
COLORS       = {"gpt2": "#e41a1c", "gpt2-medium": "#377eb8",
                "gpt2-large": "#4daf4a", "gpt2-xl": "#984ea3"}


def load_results(results_dir):
    results = {}
    for f in os.listdir(results_dir):
        if not f.endswith(".json"):
            continue
        data = json.load(open(os.path.join(results_dir, f)))
        key  = (data["model"], data["condition"])
        results[key] = data["sigma_results"]
    return results


# ── Plot 1: density curves with CI error bars ─────────────────────────────────
def plot_density_errorbars(results, output_dir, target_density=0.40):
    fig, axes = plt.subplots(1, 2, figsize=(16, 6), sharey=True)

    for ax, condition, title in [
        (axes[0], "pretrained",  "Pretrained models"),
        (axes[1], "random-init", "Random-init baseline"),
    ]:
        for model in MODEL_ORDER:
            key = (model, condition)
            if key not in results:
                continue
            rs     = results[key]
            sigmas = np.array([r["sigma"]   for r in rs])
            dens   = np.array([r["density"] for r in rs])
            ci_lo  = np.array([r["ci_lower"] for r in rs])
            ci_hi  = np.array([r["ci_upper"] for r in rs])
            err_lo = dens - ci_lo
            err_hi = ci_hi - dens
            color  = COLORS[model]
            label  = f"{model} ({MODEL_PARAMS[model]}M)"

            ax.errorbar(sigmas, dens,
                        yerr=[err_lo, err_hi],
                        fmt="o-", color=color, lw=2, ms=6,
                        capsize=4, capthick=1.5, elinewidth=1.5,
                        label=label)

        ax.axhline(target_density, color="black", lw=1.2, ls=":",
                   label=f"Target density = {target_density}")
        ax.set_xscale("log")
        ax.set_xlabel("Perturbation scale σ  (log scale)", fontsize=12)
        ax.set_ylabel("C(θ, σ, m)  —  certified density", fontsize=12)
        ax.set_title(title, fontsize=13, fontweight="bold")
        ax.set_ylim([-0.05, 1.1])
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

    fig.suptitle(
        "Weight-Space Certified Density  (N=200, 95% Clopper-Pearson CI)\n"
        "Error bars show exact Binomial CI bounds",
        fontsize=12, fontweight="bold"
    )
    plt.tight_layout()
    path = os.path.join(output_dir, "density_errorbars.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    print(f"Saved: {path}")
    plt.close()


# ── Plot 2: σ½ comparison bar chart ──────────────────────────────────────────
def plot_sigma_half_bars(results, output_dir):
    models  = [m for m in MODEL_ORDER if (m, "pretrained") in results]
    pre_hm  = [half_max_sigma(results[(m, "pretrained")]) for m in models]
    # exclude xl random-init (confounded)
    rand_hm = []
    for m in models:
        if m == "gpt2-xl":
            rand_hm.append(None)
        elif (m, "random-init") in results:
            rand_hm.append(half_max_sigma(results[(m, "random-init")]))
        else:
            rand_hm.append(None)

    x    = np.arange(len(models))
    w    = 0.35
    fig, ax = plt.subplots(figsize=(10, 6))

    bars_pre = ax.bar(x - w/2, pre_hm, w, label="Pretrained",
                      color="crimson", alpha=0.85)
    bars_rnd = ax.bar(x + w/2,
                      [v if v is not None else 0 for v in rand_hm],
                      w, label="Random-init (⚠ xl confounded)",
                      color="steelblue", alpha=0.85)

    # Value labels
    for bar, val in zip(bars_pre, pre_hm):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.002,
                f"{val:.4f}", ha="center", va="bottom", fontsize=8, color="crimson")
    for bar, val in zip(bars_rnd, rand_hm):
        if val is not None:
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.002,
                    f"{val:.4f}", ha="center", va="bottom", fontsize=8, color="steelblue")
        else:
            ax.text(bar.get_x() + bar.get_width()/2, 0.01,
                    "⚠ N/A", ha="center", va="bottom", fontsize=8, color="gray")

    ax.set_xticks(x)
    ax.set_xticklabels([f"{m}\n({MODEL_PARAMS[m]}M)" for m in models], fontsize=10)
    ax.set_ylabel("σ½  (half-max sigma)", fontsize=12)
    ax.set_title(
        "Basin Width σ½: Pretrained vs Random-Init\n"
        "Pretrained σ½ is consistently 40–900x narrower — sharp, well-defined minimum",
        fontsize=11, fontweight="bold"
    )
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3, axis="y")
    plt.tight_layout()
    path = os.path.join(output_dir, "sigma_half_bars.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    print(f"Saved: {path}")
    plt.close()


# ── LaTeX table ───────────────────────────────────────────────────────────────
def print_latex_table(results, target_density=0.40):
    print("\n" + "="*70)
    print("LaTeX TABLE")
    print("="*70)
    print(r"""\begin{table}[t]
\centering
\begin{tabular}{llrrrr}
\toprule
Model & Condition & Params & $\sigma^*$ & $\sigma_{1/2}$ & Peak density \\
\midrule""")

    for model in MODEL_ORDER:
        for condition in ["pretrained", "random-init"]:
            key = (model, condition)
            if key not in results:
                continue
            rs   = results[key]
            cr   = certified_radius(rs, target_density)
            hm   = half_max_sigma(rs)
            peak = max(r["density"] for r in rs)
            note = r" \textsuperscript{†}" if model == "gpt2-xl" and condition == "random-init" else ""
            print(f"  {model} & {condition}{note} & "
                  f"{MODEL_PARAMS.get(model, '?')}M & "
                  f"{cr:.5f} & {hm:.5f} & {peak:.3f} \\\\")
        print(r"  \midrule")

    print(r"""\bottomrule
\end{tabular}
\caption{Weight-space certified density results. $\sigma^*$: max $\sigma$ where
CI$_\text{lower} \geq 0.40$ (95\% Clopper-Pearson). $\sigma_{1/2}$: $\sigma$
where density drops to half its peak (basin width). $N=200$ perturbations per
$\sigma$. \textsuperscript{†}gpt2-xl random-init confounded by scaled init.}
\label{tab:certified_density}
\end{table}""")


# ── Certificate cards ─────────────────────────────────────────────────────────
def print_certificate_cards(results, target_density=0.40, confidence=0.95):
    print("\n" + "="*70)
    print("FORMAL CERTIFICATE STATEMENTS")
    print("="*70)

    for model in MODEL_ORDER:
        key = (model, "pretrained")
        if key not in results:
            continue
        rs = results[key]
        cr = certified_radius(rs, target_density)

        # Find CI bounds at the certified sigma
        cert_result = next((r for r in rs if abs(r["sigma"] - cr) < 1e-9), None)
        if cert_result is None or cr == 0.0:
            print(f"\n{model}: no sigma reached CI_lower >= {target_density}")
            continue

        ppl  = cert_result["base_ppl"]
        lo   = cert_result["ci_lower"]
        hi   = cert_result["ci_upper"]
        dens = cert_result["density"]
        N    = cert_result["N"]

        print(f"""
┌─ CERTIFICATE: {model} ({MODEL_PARAMS.get(model,'?')}M params) ───────────────
│
│  Baseline perplexity:  {ppl:.4f}  (WikiText-2 validation)
│  Certified σ*:         {cr:.5f}
│  Observed density:     {dens:.3f}  ({int(dens*N)}/{N} perturbations succeeded)
│  {int(confidence*100)}% Clopper-Pearson CI:  [{lo:.3f}, {hi:.3f}]
│
│  FORMAL STATEMENT:
│  "With {int(confidence*100)}% confidence, at least {lo:.1%} of Gaussian weight
│   perturbations of scale σ={cr:.5f} preserve the model's perplexity
│   on WikiText-2 (≤ {ppl:.4f})."
│
│  Equivalently: C({model}, σ={cr:.5f}, m={ppl:.2f}) ≥ {lo:.3f}  [{int(confidence*100)}% CI]
└───────────────────────────────────────────────────────────────""")


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--results_dir", default="./results/certified_density/final")
    p.add_argument("--output_dir",  default=None,
                   help="Where to save plots (default: same as results_dir)")
    p.add_argument("--target_density", type=float, default=0.40)
    args = p.parse_args()

    out = args.output_dir or args.results_dir
    os.makedirs(out, exist_ok=True)

    results = load_results(args.results_dir)
    print(f"Loaded {len(results)} conditions from {args.results_dir}")

    plot_density_errorbars(results, out, args.target_density)
    plot_sigma_half_bars(results, out)
    print_latex_table(results, args.target_density)
    print_certificate_cards(results, args.target_density)


if __name__ == "__main__":
    main()
