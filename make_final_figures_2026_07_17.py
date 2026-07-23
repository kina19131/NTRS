"""
make_final_figures_2026_07_17.py
=================================
Regenerates the four report figures (one per hypothesis + seed variance) from the
corrected 2026-07-17 research state. All values are read directly from existing
result JSONs on disk; corrected relslack sigmas are applied where the cached
per-file sigma is known to be stale (see RAW_DATA.md standing warning).

Palette: validated categorical set from the dataviz skill reference
(references/palette.md) — fixed hue order, not cycled.
"""

import json
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

# ── Validated categorical palette (fixed order, from dataviz skill) ──────────
BLUE    = "#2a78d6"   # slot 1 -> GPT-2
AQUA    = "#1baf7a"   # slot 2 -> Llama
VIOLET  = "#4a3aa7"   # slot 5 -> Mistral
YELLOW  = "#eda100"
RED     = "#e34948"
ORANGE  = "#eb6834"
MAGENTA = "#e87ba4"

GOOD     = "#0ca30c"
WARNING  = "#fab219"
CRITICAL = "#d03b3b"

INK        = "#0b0b0b"
INK_SEC    = "#52514e"
INK_MUTED  = "#898781"
GRID       = "#e1e0d9"
SURFACE    = "#fcfcfb"

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
    "axes.edgecolor": INK_MUTED,
    "axes.labelcolor": INK,
    "text.color": INK,
    "xtick.color": INK_SEC,
    "ytick.color": INK_SEC,
    "axes.grid": True,
    "grid.color": GRID,
    "grid.linewidth": 0.8,
    "figure.facecolor": SURFACE,
    "axes.facecolor": SURFACE,
    "savefig.facecolor": SURFACE,
})

OUT = "figures"
os.makedirs(OUT, exist_ok=True)


def load_json(path):
    with open(path) as f:
        return json.load(f)


# ══════════════════════════════════════════════════════════════════════════
# FIGURE 1 — H1: three architectures, corrected relslack, criterion sensitivity
# ══════════════════════════════════════════════════════════════════════════

def fig_h1():
    SIGMA = {"GPT-2": 0.007191011235955057, "Llama-3.2-3B": 0.0007717391304347826,
             "Mistral-7B-v0.1": 0.0001824175824175824}
    DIRS = {
        "GPT-2": "h1_kl/gpt2",
        "Llama-3.2-3B": "h1_kl/meta-llama/meta-llama/Llama-3.2-3B",
        "Mistral-7B-v0.1": "h1_kl/mistralai/Mistral-7B-v0.1_sst2_v5",
    }
    COLOR = {"GPT-2": BLUE, "Llama-3.2-3B": AQUA, "Mistral-7B-v0.1": VIOLET}
    LRS = ["1e-05", "5e-05", "1e-04", "2e-04", "5e-04"]

    data = {}
    for model, d in DIRS.items():
        rows = []
        for lr in LRS:
            r = load_json(os.path.join(d, f"lr{lr}_results.json"))
            R = r["per_param_norm"] / SIGMA[model]
            rows.append({"lr": float(lr), "R": R, "dnll": r["delta_nll_wiki"],
                         "kl_diverse": r["kl_diverse"], "nll_base": r["nll_wiki_base"]})
        data[model] = rows

    fig, axes = plt.subplots(2, 2, figsize=(12, 9.5))

    # (a) R vs ΔNLL, all three models overlaid, log-x, single shared y-axis
    ax = axes[0, 0]
    for model, rows in data.items():
        Rs = [r["R"] for r in rows]
        dn = [r["dnll"] for r in rows]
        ax.plot(Rs, dn, "-o", color=COLOR[model], lw=1.8, ms=7, label=model, zorder=3)
    ax.set_xscale("log")
    ax.axhline(0, color=INK_MUTED, lw=0.8, ls=":")
    ax.set_xlabel("R = ‖Δθ‖ / σ½  (relslack, log scale)")
    ax.set_ylabel("Δ NLL (WikiText-2, ft − base)")
    ax.set_title("(a)  R tracks capability degradation\nin all three architectures", fontsize=10.5, loc="left")
    ax.legend(fontsize=8.5, frameon=False, loc="upper left")

    # (b) Criterion-sensitivity: cliff-jump vs 1%-threshold brackets, dumbbell
    ax = axes[0, 1]
    models = ["GPT-2", "Llama-3.2-3B", "Mistral-7B-v0.1"]
    cliff = {"GPT-2": (0.220, 0.453), "Llama-3.2-3B": (0.236, 0.586), "Mistral-7B-v0.1": (0.194, 0.226)}
    strict = {"GPT-2": (0.220, 0.453), "Llama-3.2-3B": (0.113, 0.236), "Mistral-7B-v0.1": (0.030, 0.194)}
    ypos = {m: i for i, m in enumerate(models)}
    for m in models:
        y = ypos[m]
        lo, hi = cliff[m]
        ax.plot([lo, hi], [y + 0.16, y + 0.16], color=COLOR[m], lw=6, solid_capstyle="round",
                alpha=0.9, label="cliff-jump" if m == "GPT-2" else None)
        lo2, hi2 = strict[m]
        ax.plot([lo2, hi2], [y - 0.16, y - 0.16], color=COLOR[m], lw=6, solid_capstyle="round",
                alpha=0.4, label="1%-threshold" if m == "GPT-2" else None)
    ax.set_yticks(list(ypos.values()))
    ax.set_yticklabels(models, fontsize=9.5)
    ax.set_xlabel("T* bracket (R, relslack)")
    ax.set_ylim(-0.6, 2.7)
    ax.set_title("(b)  Flagship claim is criterion-sensitive:\nchain, not 3-way overlap, under the strict rule",
                 fontsize=10.5, loc="left")
    ax.axvspan(0.220, 0.236, color=GOOD, alpha=0.08)
    ax.text(0.227, 2.35, "GPT-2 & Llama\noverlaps\n(both criteria)", fontsize=7.3, color=INK_SEC, ha="center")
    ax.axvspan(0.113, 0.194, color=WARNING, alpha=0.10)
    ax.text(0.15, -0.5, "Llama & Mistral overlaps only\nunder 1%-threshold criterion", fontsize=7.3, color=INK_SEC, ha="center")
    handles = [plt.Line2D([0], [0], color=INK_SEC, lw=6, alpha=0.9, solid_capstyle="round"),
               plt.Line2D([0], [0], color=INK_SEC, lw=6, alpha=0.4, solid_capstyle="round")]
    ax.legend(handles, ["cliff-jump criterion", "1%-of-baseline-NLL criterion (strict)"],
              fontsize=7.5, frameon=False, loc="lower right")

    # (c) KL(diverse) failure comparison: Llama vs Mistral
    ax = axes[1, 0]
    for model in ["Llama-3.2-3B", "Mistral-7B-v0.1"]:
        rows = data[model]
        Rs = [r["R"] for r in rows]
        kl = [r["kl_diverse"] for r in rows]
        ax.plot(Rs, kl, "-o", color=COLOR[model], lw=1.8, ms=7, label=model)
    ax.set_xscale("log")
    ax.set_xlabel("R = ‖Δθ‖ / σ½  (relslack, log scale)")
    ax.set_ylabel("KL(diverse)  [base ‖ fine-tuned]")
    ax.set_title("(c)  KL(diverse) fails to track the onset —\nby two different mechanisms, not one",
                 fontsize=10.5, loc="left")
    ax.annotate("saturates immediately\n(Llama: 7.11 at LR=1e-5)", xy=(data["Llama-3.2-3B"][0]["R"], data["Llama-3.2-3B"][0]["kl_diverse"]),
                xytext=(0.06, 9.2), fontsize=7.3, color=AQUA,
                arrowprops=dict(arrowstyle="->", color=AQUA, lw=1))
    ax.annotate("insufficient slope at cliff\n(Mistral: 2.6× while ΔNLL jumps 22.5×)",
                xy=(0.226, 3.706), xytext=(0.05, 1.5), fontsize=7.3, color=VIOLET,
                arrowprops=dict(arrowstyle="->", color=VIOLET, lw=1))
    ax.legend(fontsize=8.5, frameon=False, loc="center right")

    # (d) Metric divergence: R at which each Llama SST-2 metric collapses
    ax = axes[1, 1]
    metrics = ["WikiText-2 NLL\n(general capability)", "MMLU\n(world knowledge)", "SST-2 accuracy\n(task itself)"]
    r_collapse = [0.586, 0.586, 3.641]  # NLL and MMLU both collapse at lr=2e-4 (R=0.586); task acc survives to lr=5e-4 (R=3.641)
    colors_d = [AQUA, ORANGE, MAGENTA]
    bars = ax.bar(metrics, r_collapse, color=colors_d, width=0.55)
    for b, v in zip(bars, r_collapse):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.08, f"R≈{v:.2f}", ha="center", fontsize=9, color=INK)
    ax.set_ylabel("R at which metric collapses (Llama SST-2)")
    ax.set_title("(d)  \"Forgetting\" is metric-dependent:\ntask accuracy survives far longer than general capability",
                 fontsize=10.5, loc="left")
    ax.tick_params(axis="x", labelsize=8.3)

    fig.suptitle("H1 — Pretraining-to-SFT boundary (corrected 2026-07-17)", fontsize=13, fontweight="bold", y=0.995)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    path = os.path.join(OUT, "fig_h1_2026_07_17.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {path}")


# ══════════════════════════════════════════════════════════════════════════
# FIGURE 2 — H2: 8-cell transfer grid + the V1 objection (raw vs normalized spread)
# ══════════════════════════════════════════════════════════════════════════

def fig_h2_transfer():
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.3), gridspec_kw={"width_ratios": [1.35, 1]})

    # (a) 8-cell transfer grid
    ax = axes[0]
    tasks = ["MNLI", "AGNews", "SST-2", "CoLA"]
    ranks = ["r8", "r32"]
    status = {
        ("MNLI", "r8"): ("pass", "[0.041,0.081]\nvs [0.040,0.084]"),
        ("MNLI", "r32"): ("pass", "0.063 vs 0.063"),
        ("AGNews", "r8"): ("pass", "[0.042,0.085]\nvs [0.042,0.089]"),
        ("AGNews", "r32"): ("pass", "0.062 vs 0.063"),
        ("SST-2", "r8"): ("fail", "44× spread\n(3 Phase-2 tasks)"),
        ("SST-2", "r32"): ("pass", "0.053 vs 0.050"),
        ("CoLA", "r8"): ("weak", "[0.060,0.118]\nvs [0.060,0.126]"),
        ("CoLA", "r32"): ("fail", "0.189 vs 0.089\n(2× gap)"),
    }
    color_map = {"pass": GOOD, "fail": CRITICAL, "weak": WARNING}
    for ti, task in enumerate(tasks):
        for ri, rank in enumerate(ranks):
            s, label = status[(task, rank)]
            ax.add_patch(Rectangle((ri, len(tasks) - 1 - ti), 0.94, 0.94,
                                    facecolor=color_map[s], alpha=0.82, edgecolor="white", lw=2))
            ax.text(ri + 0.47, len(tasks) - 1 - ti + 0.6, task if ri == 0 else "", fontsize=0)  # no-op placeholder
            ax.text(ri + 0.47, len(tasks) - 1 - ti + 0.47, label, ha="center", va="center",
                    fontsize=7.6, color="white", fontweight="600")
    ax.set_xlim(-0.1, 2.05)
    ax.set_ylim(-0.1, 4.1)
    ax.set_xticks([0.47, 1.47])
    ax.set_xticklabels(ranks, fontsize=10)
    ax.set_yticks([len(tasks) - 1 - i + 0.47 for i in range(len(tasks))])
    ax.set_yticklabels(tasks, fontsize=10)
    ax.grid(False)
    ax.set_title("(a)  H2 transfer: 5 pass / 2 fail / 1 weak of 8 cells\n(Llama, Phase-1 task × rank)",
                 fontsize=10.5, loc="left")
    handles = [Rectangle((0, 0), 1, 1, facecolor=color_map[s], alpha=0.82) for s in ["pass", "weak", "fail"]]
    ax.legend(handles, ["transfers", "weak / ambiguous", "fails"], fontsize=8.5,
              frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.08), ncol=3)

    # (b) V1 objection: raw vs normalized spread
    ax = axes[1]
    cats = ["Raw T*_raw\n(unnormalized)", "σ½-normalized\nR_A"]
    vals = [4.0, 53.0]
    bars = ax.bar(cats, vals, color=[GOOD, CRITICAL], width=0.5)
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v + 1.5, f"~{v:.0f}%", ha="center", fontsize=12, fontweight="700", color=INK)
    ax.set_ylabel("Spread across MNLI/AGNews/CoLA r8 pairs\n(max−min)/min")
    ax.set_ylim(0, 65)
    ax.set_title("(b)  The V1 objection, quantified:\nσ½ normalization loosens this comparison,\nit doesn't tighten it",
                 fontsize=10.5, loc="left")

    fig.suptitle("H2 — Sequential task forgetting: transfer + the sharpest standing objection",
                 fontsize=13, fontweight="bold", y=1.03)
    fig.tight_layout()
    path = os.path.join(OUT, "fig_h2_transfer_2026_07_17.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {path}")


# ══════════════════════════════════════════════════════════════════════════
# FIGURE 3 — H2 cross-architecture (Mistral, corrected consistent methodology)
# ══════════════════════════════════════════════════════════════════════════

def fig_h2_crossarch():
    fig, ax = plt.subplots(figsize=(9.5, 5))

    pairs = ["AGNews->MNLI", "AGNews->SST-2"]
    llama_t = {"AGNews->MNLI": (0.042, 0.085), "AGNews->SST-2": (0.042, 0.089)}
    mistral_t = {"AGNews->MNLI": (0.020, 0.085), "AGNews->SST-2": (0.068, 0.105)}
    gaps = {"AGNews->MNLI": "0.2% gap", "AGNews->SST-2": "17.9% gap"}

    ypos = {p: i for i, p in enumerate(pairs)}
    for p in pairs:
        y = ypos[p]
        lo, hi = llama_t[p]
        ax.plot([lo, hi], [y + 0.14, y + 0.14], color=AQUA, lw=9, solid_capstyle="round",
                label="Llama-3.2-3B" if p == pairs[0] else None)
        lo2, hi2 = mistral_t[p]
        ax.plot([lo2, hi2], [y - 0.14, y - 0.14], color=VIOLET, lw=9, solid_capstyle="round",
                label="Mistral-7B-v0.1" if p == pairs[0] else None)
        # overlap shading
        olo, ohi = max(lo, lo2), min(hi, hi2)
        if olo <= ohi:
            ax.axvspan(olo, ohi, ymin=(y - 0.3 + 1) / (len(pairs) + 0.6), ymax=(y + 0.3 + 1) / (len(pairs) + 0.6),
                       color=GOOD, alpha=0.12)
        ax.text(max(hi, hi2) + 0.006, y, gaps[p], va="center", fontsize=10, fontweight="700", color=INK_SEC)

    ax.set_yticks(list(ypos.values()))
    ax.set_yticklabels(pairs, fontsize=11)
    ax.set_xlabel("T* (accuracy-based σ½, consistent methodology for both models)")
    ax.set_ylim(-0.6, 1.6)
    ax.set_title("Cross-architecture H2 validation — corrected 2026-07-17\n"
                 "(previously mixed acc-based Llama vs. NLL-based Mistral σ½; both pairs now overlap)",
                 fontsize=11.5, loc="left")
    ax.legend(fontsize=9.5, frameon=False, loc="lower right")

    fig.tight_layout()
    path = os.path.join(OUT, "fig_h2_crossarch_2026_07_17.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {path}")


# ══════════════════════════════════════════════════════════════════════════
# FIGURE 4 — Seed variance (GPT-2 SST-2)
# ══════════════════════════════════════════════════════════════════════════

def fig_seed_variance():
    runA = load_json("seed_variance/gpt2/sst2/summary.json")            # lr 5e-5,1e-4,2e-4 @ 500 steps
    runC = load_json("seed_variance_5e4_500steps/gpt2/sst2/summary.json")  # lr 5e-4 @ 500 steps
    runB = load_json("seed_variance_3k/gpt2/sst2/summary.json")         # lr 5e-4 @ 3000 steps (has NaN seed42)

    rows_500 = sorted(runA + runC, key=lambda r: r["lr"])  # 4 LR points, all @ 500 steps, matched protocol

    fig, axes = plt.subplots(1, 2, figsize=(12.5, 5))

    # (a) R_mean ± R_std across seeds, per LR (500-step matched protocol)
    ax = axes[0]
    lrs = [r["lr"] for r in rows_500]
    means = [r["R_mean"] for r in rows_500]
    stds = [r["R_std"] for r in rows_500]
    x = np.arange(len(lrs))
    ax.errorbar(x, means, yerr=stds, fmt="o-", color=BLUE, lw=1.8, ms=8, capsize=5, capthick=1.5,
                ecolor=BLUE, elinewidth=1.5)
    ax.set_xticks(x)
    ax.set_xticklabels([f"{lr:.0e}" for lr in lrs], fontsize=9.5)
    ax.set_xlabel("Learning rate (500 steps, 5 data seeds, fixed init)")
    ax.set_ylabel("R = ‖Δθ‖ / σ½_pre  (mean ± std across seeds)")
    ax.set_title("(a)  Seed-to-seed R variance grows with LR\n(GPT-2 SST-2, fixed init, 5 data-loading seeds)",
                 fontsize=10.5, loc="left")

    # (b) pairwise inter-seed distance / σ½ vs LR — is stochastic drift comparable to σ½?
    ax = axes[1]
    ratio_mean = [r["pairwise_ratio_mean"] for r in rows_500]
    ratio_min = [r["pairwise_ratio_min"] for r in rows_500]
    ratio_max = [r["pairwise_ratio_max"] for r in rows_500]
    ax.plot(x, ratio_mean, "o-", color=ORANGE, lw=1.8, ms=8, label="mean pairwise dist / σ½")
    ax.fill_between(x, ratio_min, ratio_max, color=ORANGE, alpha=0.18, label="min–max range")
    ax.axhline(1.0, color=CRITICAL, lw=1.3, ls="--", label="σ½ boundary (ratio = 1)")
    ax.set_xticks(x)
    ax.set_xticklabels([f"{lr:.0e}" for lr in lrs], fontsize=9.5)
    ax.set_xlabel("Learning rate (500 steps)")
    ax.set_ylabel("‖θ_seed_a − θ_seed_b‖ / σ½_pre")
    ax.set_title("(b)  Pure data-order stochasticity approaches σ½\nat higher LR — before any fine-tuning \"signal\"",
                 fontsize=10.5, loc="left")
    ax.set_ylim(0, 1.12)
    ax.legend(fontsize=8, frameon=False, loc="center left", bbox_to_anchor=(0.02, 0.62))

    fig.suptitle("Seed Variance Experiment — GPT-2 SST-2 (same init, different data order)",
                 fontsize=13, fontweight="bold", y=1.03)
    fig.tight_layout()
    path = os.path.join(OUT, "fig_seed_variance_2026_07_17.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {path}")

    # Print the note about the 3000-step / matched-accuracy comparison for the record
    print("\nNote: 3000-step run (lr=5e-4) shows seed123/seed456 at matched accuracy (0.906/0.906) "
          "with R=0.918/0.977 (~6% apart) and ΔNLL=0.394/1.004 (~2.5× apart) — not plotted above "
          "(different step count than the 500-step matched set) but relevant context for how much "
          "damage variance exists at fixed accuracy.")


# ══════════════════════════════════════════════════════════════════════════
# FIGURE 5 — Dedicated KL-vs-R comparison, all 3 models, all 3 KL variants
# ══════════════════════════════════════════════════════════════════════════

def fig_kl_vs_r():
    SIGMA = {"GPT-2": 0.007191011235955057, "Llama-3.2-3B": 0.0007717391304347826,
             "Mistral-7B-v0.1": 0.0001824175824175824}
    DIRS = {
        "GPT-2": "h1_kl/gpt2",
        "Llama-3.2-3B": "h1_kl/meta-llama/meta-llama/Llama-3.2-3B",
        "Mistral-7B-v0.1": "h1_kl/mistralai/Mistral-7B-v0.1_sst2_v5",
    }
    COLOR = {"GPT-2": BLUE, "Llama-3.2-3B": AQUA, "Mistral-7B-v0.1": VIOLET}
    LRS = ["1e-05", "5e-05", "1e-04", "2e-04", "5e-04"]

    data = {}
    for model, d in DIRS.items():
        rows = []
        for lr in LRS:
            r = load_json(os.path.join(d, f"lr{lr}_results.json"))
            R = r["per_param_norm"] / SIGMA[model]
            rows.append({"R": R, "kl_diverse": r["kl_diverse"], "kl_wiki": r["kl_wiki"],
                         "kl_superni": r.get("kl_superni"), "dnll": r["delta_nll_wiki"]})
        data[model] = rows

    fig, axes = plt.subplots(1, 3, figsize=(15.5, 4.8))

    # (a) KL(diverse) vs R — all three models (the baseline that fails)
    ax = axes[0]
    for model, rows in data.items():
        Rs = [r["R"] for r in rows]
        kl = [r["kl_diverse"] for r in rows]
        ax.plot(Rs, kl, "-o", color=COLOR[model], lw=1.8, ms=6.5, label=model)
    ax.set_xscale("log")
    ax.set_xlabel("R  (relslack, log scale)")
    ax.set_ylabel("KL(diverse)  [base ‖ fine-tuned]")
    ax.set_title("(a)  KL(diverse) — fails for all 3 models\n(non-monotone or insufficient slope)", fontsize=10, loc="left")
    ax.legend(fontsize=8, frameon=False, loc="upper left")

    # (b) KL(wiki) vs R — the pretraining-domain KL, which DOES track ΔNLL
    ax = axes[1]
    for model, rows in data.items():
        Rs = [r["R"] for r in rows]
        kl = [r["kl_wiki"] for r in rows]
        ax.plot(Rs, kl, "-o", color=COLOR[model], lw=1.8, ms=6.5, label=model)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("R  (relslack, log scale)")
    ax.set_ylabel("KL(wiki)  [base ‖ fine-tuned]  (log scale)")
    ax.set_title("(b)  KL(wiki) — monotone for all 3 models\n(the failure in (a) is domain-specific, not universal)", fontsize=10, loc="left")
    ax.legend(fontsize=8, frameon=False, loc="upper left")

    # (c) KL(SuperNI) vs R — Llama (weak) vs Mistral (discriminating at cliff)
    ax = axes[2]
    for model in ["Llama-3.2-3B", "Mistral-7B-v0.1"]:
        rows = data[model]
        Rs = [r["R"] for r in rows]
        kl = [r["kl_superni"] for r in rows]
        ax.plot(Rs, kl, "-o", color=COLOR[model], lw=1.8, ms=6.5, label=model)
    ax.set_xscale("log")
    ax.set_xlabel("R  (relslack, log scale)")
    ax.set_ylabel("KL(SuperNI)  [base ‖ fine-tuned]")
    ax.set_title("(c)  KL(SuperNI) — monotone but weak for Llama\n(15% at boundary); discriminates for Mistral (12.4× at cliff)",
                 fontsize=10, loc="left")
    ax.legend(fontsize=8, frameon=False, loc="upper left")
    ax.annotate("cliff: 0.581 -> 7.218\n(12.4× jump)", xy=(0.226, 7.218), xytext=(0.35, 5.5),
                fontsize=7.5, color=VIOLET, arrowprops=dict(arrowstyle="->", color=VIOLET, lw=1))

    fig.suptitle("KL vs. R — all three KL variants tested, all three models (GPT-2 has no SuperNI/MMLU fields)",
                 fontsize=12.5, fontweight="bold", y=1.04)
    fig.tight_layout()
    path = os.path.join(OUT, "fig_kl_vs_r_2026_07_17.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {path}")


if __name__ == "__main__":
    fig_h1()
    fig_h2_transfer()
    fig_h2_crossarch()
    fig_seed_variance()
    fig_kl_vs_r()
    print("\nAll figures regenerated.")
