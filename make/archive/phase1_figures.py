"""
phase1_figures.py
=================
Phase 1: figures that validate and sharpen the existing experimental story.

  fig_A1_trajectory.png       — training trajectory: norm/σ½ growth vs σ½ widening ratio
  fig_A2_density_evolution.png — density curves across training steps (collapse visible)
  fig_A3_multimodel_bar.png   — σ½ before/after + norm/σ½ ratio for all 5 models

Run: python phase1_figures.py
"""

import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.ticker as mticker
import matplotlib.pyplot as plt
import numpy as np

ROOT    = os.path.dirname(os.path.abspath(__file__))
BW_SST2 = os.path.join(ROOT, "results", "basin_widening", "gpt2", "sst2")
LD      = os.path.join(ROOT, "results", "lora_density")
OUT_DIR = os.path.join(ROOT, "results", "figures")
os.makedirs(OUT_DIR, exist_ok=True)

plt.rcParams.update({
    "figure.facecolor": "white",
    "axes.facecolor":   "white",
    "axes.spines.top":  False,
    "axes.spines.right":False,
    "axes.grid":        True,
    "grid.alpha":       0.25,
    "grid.linestyle":   "--",
    "font.family":      "sans-serif",
    "font.size":        11,
    "axes.titlesize":   12,
    "axes.labelsize":   11,
    "legend.fontsize":  9,
    "legend.framealpha":0.9,
})

C_MODEL = {
    "gpt2":           "#2563EB",
    "gpt2-medium":    "#DC2626",
    "gpt2-large":     "#7C3AED",
    "tinyllama-1.1b": "#D97706",
    "llama-3.2-1b":   "#059669",
}
MODEL_LABELS = {
    "gpt2":           "GPT-2\n(124M)",
    "gpt2-medium":    "GPT-2-med\n(354M)",
    "gpt2-large":     "GPT-2-large\n(774M)",
    "tinyllama-1.1b": "TinyLlama\n(1.1B)",
    "llama-3.2-1b":   "Llama-3.2\n(1B)",
}
MODELS = ["gpt2", "gpt2-medium", "gpt2-large", "tinyllama-1.1b", "llama-3.2-1b"]
ALPHA_CI = 0.12


def _save(fig, name):
    path = os.path.join(OUT_DIR, name)
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close()
    print(f"  Saved  {path}")


def _load_json(path):
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


def _sigma_results(d):
    if d is None:
        return None
    if isinstance(d, list):
        return d
    return d.get("sigma_results")


def half_max_sigma(sr):
    if not sr:
        return float("nan")
    dens   = [r["density"] for r in sr]
    sigmas = [r["sigma"]   for r in sr]
    target = max(dens) / 2.0
    for i in range(len(dens) - 1):
        if dens[i] >= target > dens[i + 1]:
            t = (target - dens[i]) / (dens[i + 1] - dens[i])
            return sigmas[i] + t * (sigmas[i + 1] - sigmas[i])
    return sigmas[-1]


def extract(sr):
    return (
        np.array([r["sigma"]    for r in sr]),
        np.array([r["density"]  for r in sr]),
        np.array([r["ci_lower"] for r in sr]),
        np.array([r["ci_upper"] for r in sr]),
    )


# ── Figure A1: Training trajectory ────────────────────────────────────────────
def fig_A1():
    """
    Dual-axis trajectory plot showing how norm/σ½ and σ½ widening ratio
    evolve over training steps. The moment norm crosses σ½_pre is the
    certified ball boundary — shows collapse threshold visually.
    """
    summary = _load_json(os.path.join(BW_SST2, "widening_summary.json"))
    if summary is None:
        print("  fig_A1: widening_summary.json not found — skipping")
        return

    rows           = summary["results"]
    steps          = [r["steps"]          for r in rows]
    norm_over_sh   = [r["norm_over_sh"]   for r in rows]
    widening_ratio = [r["widening_ratio"] for r in rows]
    ppl_ratio      = [r["ppl_ratio"]      for r in rows]

    fig, ax1 = plt.subplots(figsize=(9, 5.5))
    ax2 = ax1.twinx()

    # Green shaded region: inside the certified ball (norm/σ½ < 1)
    ax1.axhspan(0, 1.0, alpha=0.07, color="#16a34a", zorder=0)

    # Red dashed line: certified ball boundary
    ax1.axhline(1.0, color="#dc2626", lw=1.8, ls="--", zorder=3)

    # Left axis — norm/σ½ trajectory
    line1, = ax1.plot(steps, norm_over_sh, color="#1d4ed8", lw=2.5,
                      marker="o", ms=8, zorder=4)
    for s, n in zip(steps, norm_over_sh):
        ax1.annotate(f"{n:.2f}×", xy=(s, n),
                     xytext=(0, 11), textcoords="offset points",
                     ha="center", fontsize=9, color="#1d4ed8", fontweight="bold")

    # Right axis — σ½ widening ratio trajectory
    line2, = ax2.plot(steps, widening_ratio, color="#7c3aed", lw=2.5,
                      marker="s", ms=8, ls="--", zorder=4)
    for s, w in zip(steps, widening_ratio):
        ax2.annotate(f"{w:.2f}×", xy=(s, w),
                     xytext=(0, -17), textcoords="offset points",
                     ha="center", fontsize=9, color="#7c3aed")

    # Red circle at steps where PPL regresses (forgetting)
    for s, pr, n in zip(steps, ppl_ratio, norm_over_sh):
        if pr > 1.005:
            ax1.scatter([s], [n], s=220, zorder=6,
                        facecolors="none", edgecolors="#dc2626", linewidths=2.5)
            ax1.annotate(f"PPL ×{pr:.3f}\n(forgetting)",
                         xy=(s, n), xytext=(-60, 20), textcoords="offset points",
                         ha="center", fontsize=8.5, color="#dc2626",
                         arrowprops=dict(arrowstyle="->", color="#dc2626", lw=1.0))

    ax1.set_xscale("log")
    ax1.xaxis.set_major_locator(mticker.FixedLocator(steps))
    ax1.xaxis.set_major_formatter(mticker.FixedFormatter([str(s) for s in steps]))
    ax1.set_xlabel("Training steps  (log scale)", fontsize=11)
    ax1.set_ylabel("norm / σ½  (dimensionless)", color="#1d4ed8", fontsize=11)
    ax2.set_ylabel("σ½ widening ratio  (σ½_ft / σ½_pre)", color="#7c3aed", fontsize=11)
    ax1.tick_params(axis="y", labelcolor="#1d4ed8")
    ax2.tick_params(axis="y", labelcolor="#7c3aed")
    ax2.spines["right"].set_visible(True)
    ax1.set_ylim(0, max(norm_over_sh) * 1.50)
    ax2.set_ylim(0, max(widening_ratio) * 1.50)

    legend_elements = [
        mpatches.Patch(color="#16a34a", alpha=0.25,
                       label="Inside certified ball  (norm < σ½)"),
        plt.Line2D([0], [0], color="#dc2626", lw=1.8, ls="--",
                   label="Certified ball boundary  (norm = σ½)"),
        plt.Line2D([0], [0], color="#1d4ed8", lw=2.5, marker="o", ms=7,
                   label="norm / σ½  (left axis)"),
        plt.Line2D([0], [0], color="#7c3aed", lw=2.5, marker="s", ms=7, ls="--",
                   label="σ½ widening ratio  (right axis)"),
        plt.Line2D([0], [0], marker="o", color="none", markersize=12,
                   markeredgecolor="#dc2626", markerfacecolor="none",
                   markeredgewidth=2.5, label="PPL regression  (forgetting)"),
    ]
    ax1.legend(handles=legend_elements, fontsize=8.5, loc="upper left", framealpha=0.92)

    ax1.set_title(
        "Training Trajectory  —  GPT-2  |  LoRA rank=8  |  SST-2  |  lr=1e-4\n"
        "Green shading: inside certified ball  ·  "
        "Red dashed: collapse threshold  ·  "
        "norm > σ½ → forgetting",
        fontsize=11,
    )
    fig.tight_layout()
    _save(fig, "fig_A1_trajectory.png")


# ── Figure A2: Density curve evolution across training steps ──────────────────
def fig_A2():
    """
    Overlaid density curves at each step count. Shows how the density profile
    evolves and the visual collapse/shift when the model exits the certified ball
    at steps=5000.
    """
    STEPS_CFG = [
        (None, "pretrained",              "#111827", "-",  "o", 2.8, 1.00),
        (100,  "steps=100   (norm/σ½=0.07)",  "#93c5fd", "-",  "o", 2.0, 0.90),
        (500,  "steps=500   (norm/σ½=0.20)",  "#3b82f6", "-",  "s", 2.0, 0.90),
        (1000, "steps=1000  (norm/σ½=0.29)",  "#16a34a", "-",  "^", 2.0, 0.90),
        (2000, "steps=2000  (norm/σ½=0.92)",  "#f59e0b", "-",  "D", 2.0, 0.90),
        (5000, "steps=5000  (norm/σ½=1.92) ← exits ball",
               "#dc2626", "--", "X", 2.5, 1.00),
    ]

    pre_sr = _sigma_results(_load_json(os.path.join(BW_SST2, "pretrained_density.json")))
    step_srs = {
        steps: _sigma_results(
            _load_json(os.path.join(BW_SST2, f"rank8_steps{steps}_density.json"))
        )
        for (steps, *_) in STEPS_CFG if steps is not None
    }

    sh_pre = half_max_sigma(pre_sr) if pre_sr is not None else None

    fig, ax = plt.subplots(figsize=(9, 5.5))

    for steps, label, color, ls, marker, lw, alpha in STEPS_CFG:
        sr = pre_sr if steps is None else step_srs.get(steps)
        if sr is None:
            continue
        s, d, lo, hi = extract(sr)
        ax.plot(s, d, color=color, lw=lw, marker=marker, ms=5,
                ls=ls, label=label, alpha=alpha, zorder=4)
        ax.fill_between(s, lo, hi, color=color, alpha=ALPHA_CI * alpha)

    # Reference line: σ½_pre (the pre-training basin width as a budget ceiling)
    if sh_pre is not None:
        ax.axvline(sh_pre, color="#111827", lw=1.4, ls=":", alpha=0.55)
        ax.text(sh_pre * 1.08, 0.05,
                f"σ½_pre\n={sh_pre:.5f}", fontsize=8, color="#374151",
                va="bottom", ha="left")

    # σ* threshold reference
    ax.axhline(0.40, color="gray", lw=1.0, ls=":", alpha=0.45)
    ax.text(1.5e-4, 0.415, "σ* threshold (0.40)", fontsize=8,
            color="gray", alpha=0.65)

    ax.set_xscale("log")
    ax.set_xlabel("Perturbation scale σ")
    ax.set_ylabel("Certified density  C(θ, σ)")
    ax.set_ylim(-0.02, 1.10)
    ax.legend(fontsize=8.5, loc="upper right")
    ax.set_title(
        "Density Curve Evolution Across Training Steps\n"
        "GPT-2  |  LoRA rank=8  |  SST-2  |  lr=1e-4  |  N=200, 95% CI  "
        "|  vertical dotted = σ½_pre",
        fontsize=11,
    )
    fig.tight_layout()
    _save(fig, "fig_A2_density_evolution.png")


# ── Figure A3: Multi-model σ½ bar chart with norm overlay ────────────────────
def fig_A3():
    """
    Two-panel figure.
    Top: grouped bars showing σ½ pretrained vs σ½ fine-tuned (lr=1e-4) for all
         five models, with per-param norm shown as a diamond marker.
    Bottom: norm/σ½ ratio bar chart — green below threshold, red above.
    """
    sh_pre_vals, sh_ft_vals, norm_vals, ratio_vals = [], [], [], []
    valid_models = []

    for m in MODELS:
        pre_sr  = _sigma_results(_load_json(os.path.join(LD, m, "pretrained_density.json")))
        ft_sr   = _sigma_results(_load_json(os.path.join(LD, m, "lora_lr1e-04_density.json")))
        norm_d  = _load_json(os.path.join(LD, m, "lora_lr1e-04_norms.json"))
        if pre_sr is None or ft_sr is None or norm_d is None:
            print(f"  fig_A3: missing data for {m} — skipping")
            continue

        sh_pre = half_max_sigma(pre_sr)
        sh_ft  = half_max_sigma(ft_sr)
        norm   = norm_d.get("per_param_norm", float("nan"))
        ratio  = norm / sh_pre if sh_pre > 0 else float("nan")

        sh_pre_vals.append(sh_pre)
        sh_ft_vals.append(sh_ft)
        norm_vals.append(norm)
        ratio_vals.append(ratio)
        valid_models.append(m)

    if not valid_models:
        print("  fig_A3: no data found — skipping")
        return

    x      = np.arange(len(valid_models))
    w      = 0.32
    colors = [C_MODEL[m] for m in valid_models]

    fig, (ax_top, ax_bot) = plt.subplots(
        2, 1, figsize=(10, 8.5),
        gridspec_kw={"height_ratios": [2.2, 1], "hspace": 0.50},
    )

    # ── Top panel: σ½ bars ──────────────────────────────────────────────────
    ax_top.bar(x - w / 2, sh_pre_vals, w, color=colors, alpha=0.92,
               edgecolor="white", linewidth=1.2, label="Pretrained σ½")
    ax_top.bar(x + w / 2, sh_ft_vals, w, color=colors, alpha=0.35,
               edgecolor=colors, linewidth=2.0, label="LoRA σ½  (lr=1e-4)")

    # Per-param norm as a diamond overlay (in the same units as σ½)
    ax_top.scatter(x, norm_vals, s=90, color=colors, marker="D", zorder=5,
                   edgecolors="white", linewidths=0.8,
                   label="per-param update norm  ||ΔW||_pp")

    for xi, (sh_pre, sh_ft, norm, ratio) in enumerate(
            zip(sh_pre_vals, sh_ft_vals, norm_vals, ratio_vals)):
        # Widening % above the fine-tuned bar
        delta_pct = (sh_ft - sh_pre) / sh_pre * 100
        sign  = "+" if delta_pct >= 0 else ""
        color_txt = "#16a34a" if delta_pct >= 0 else "#dc2626"
        ax_top.text(xi + w / 2, sh_ft * 1.05,
                    f"{sign}{delta_pct:.0f}%",
                    ha="center", va="bottom", fontsize=9,
                    fontweight="bold", color=color_txt)
        # norm/σ½ ratio below the diamond
        ax_top.text(xi, norm * 0.82,
                    f"{ratio:.2f}×", ha="center", va="top",
                    fontsize=7.5, color="#4b5563")

    ax_top.set_xticks(x)
    ax_top.set_xticklabels([MODEL_LABELS[m] for m in valid_models], fontsize=10)
    ax_top.set_ylabel("σ½  (weight-space basin width)")
    ax_top.legend(fontsize=9, loc="upper right")
    ax_top.set_title(
        "σ½ Before vs. After LoRA  (lr=1e-4)  —  All 5 Models\n"
        "% = σ½ change  ·  ◆ = per-param update norm  ·  "
        "small ratio (×) = norm well inside ball",
        fontsize=11,
    )

    # ── Bottom panel: norm/σ½ ratio bars ────────────────────────────────────
    ratio_colors = ["#16a34a" if r < 1.0 else "#dc2626" for r in ratio_vals]
    ax_bot.bar(x, ratio_vals, color=ratio_colors, alpha=0.82,
               edgecolor="white", linewidth=1.2)
    ax_bot.axhline(1.0, color="#111827", lw=1.8, ls="--",
                   label="norm = σ½  (ball boundary)")
    for xi, ratio in enumerate(ratio_vals):
        ax_bot.text(xi, ratio + max(ratio_vals) * 0.03,
                    f"{ratio:.2f}×", ha="center", va="bottom",
                    fontsize=9, fontweight="bold", color=ratio_colors[xi])

    ax_bot.set_xticks(x)
    ax_bot.set_xticklabels([MODEL_LABELS[m] for m in valid_models], fontsize=10)
    ax_bot.set_ylabel("norm / σ½")
    ax_bot.set_ylim(0, max(ratio_vals) * 1.40)
    ax_bot.legend(fontsize=9)
    ax_bot.set_title(
        "norm / σ½ Ratio  —  Green: inside certified ball  |  Red: outside",
        fontsize=10,
    )

    fig.suptitle(
        "Multi-Model Summary  —  LoRA lr=1e-4  |  WikiText-2 Certified Density",
        fontsize=12, y=1.01,
    )
    fig.tight_layout()
    _save(fig, "fig_A3_multimodel_bar.png")


# ── Main ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print(f"Output → {OUT_DIR}\n")
    print("Figure A1: training trajectory...")
    fig_A1()
    print("Figure A2: density curve evolution...")
    fig_A2()
    print("Figure A3: multi-model bar chart...")
    fig_A3()
    print("\nPhase 1 figures complete.")
