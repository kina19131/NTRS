"""
make_ntrs_figures.py
====================
Publication narrative figures for the NTRS σ½ paper.
Tells the story in 3 figures:

  Fig 1 — Foundation: R_A is a monotone forgetting predictor (all 6 pairs)
  Fig 2 — Main claim: T* is a Phase 1 property for multi-class; breaks for binary
  Fig 3 — The binary anomaly: SST-2 Phase 1 is Phase 2-task-dependent (the mechanism)

Run:  python make_ntrs_figures.py
Out:  figures/ntrs/fig1_monotone.png
      figures/ntrs/fig2_t_star_property.png
      figures/ntrs/fig3_binary_anomaly.png
"""

import json, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec

os.makedirs("figures/ntrs", exist_ok=True)

# ── Style ─────────────────────────────────────────────────────────────────────
plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 10,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.25,
    "grid.linewidth": 0.6,
})

C = {"MNLI": "#1f77b4", "AGNews": "#2ca02c", "SST-2": "#d62728"}
MULTI = {"MNLI", "AGNews"}

PAIRS = {
    "MNLI→SST-2":   ("mnli_to_sst2",   "MNLI",   "SST-2",  0.002857, 0.884),
    "MNLI→AGNews":  ("mnli_to_agnews",  "MNLI",   "AGNews", 0.002857, 0.884),
    "AGNews→MNLI":  ("agnews_to_mnli",  "AGNews", "MNLI",   0.002756, 0.905),
    "AGNews→SST-2": ("agnews_to_sst2",  "AGNews", "SST-2",  0.002756, 0.905),
    "SST-2→MNLI":   ("sst2_to_mnli",   "SST-2",  "MNLI",   0.003337, 0.975),
    "SST-2→AGNews": ("sst2_to_agnews",  "SST-2",  "AGNews", 0.003337, 0.975),
}

def load_pair(tag):
    path = f"h2_llama/{tag}/summary_acc_sigma.json"
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)

def t_star(rows, baseline, sigma, rank):
    thresh = baseline * 0.9
    rs = sorted([r for r in rows if r["rank"] == rank],
                key=lambda x: x["final_per_param"])
    last_safe = next((r for r in reversed(rs) if r["final_acc_task_a"] >= thresh), None)
    first_bad = next((r for r in rs if r["final_acc_task_a"] < thresh), None)
    lo = last_safe["final_per_param"] / sigma if last_safe else 0.0
    hi = first_bad["final_per_param"] / sigma if first_bad else None
    return lo, hi


# ── Fig 1: Monotone predictor (foundation) ───────────────────────────────────

def fig1_monotone():
    fig, axes = plt.subplots(2, 3, figsize=(13, 7), sharey=False)
    axes = axes.flatten()

    for ax, (label, (tag, p1, p2, sigma, base)) in zip(axes, PAIRS.items()):
        rows = load_pair(tag)
        thresh = base * 0.9
        color = C[p1]

        if rows is None:
            ax.set_title(f"{label}\n(missing)", fontsize=9)
            continue

        for rank, mk, alpha in [(8, "o", 0.95), (32, "s", 0.55)]:
            rs = sorted([r for r in rows if r["rank"] == rank],
                        key=lambda x: x["final_per_param"])
            xs = [r["final_per_param"] / sigma for r in rs]
            ys = [r["final_acc_task_a"] for r in rs]
            ax.plot(xs, ys, mk + "-", color=color, alpha=alpha,
                    markersize=8, linewidth=1.5, label=f"r{rank}")

        ax.axhline(thresh, color="black", lw=1, ls="--", alpha=0.5,
                   label=f"90% of {base:.2f}")
        ax.axvline(1.0, color="gray", lw=0.8, ls=":", alpha=0.6)

        # T* band shading
        lo8, hi8 = t_star(rows, base, sigma, 8)
        if hi8:
            ax.axvspan(lo8, hi8, alpha=0.12, color=color, label="T* band")

        ax.set_xlim(left=0)
        ax.set_ylim(0.15, 1.05)
        ax.set_xlabel("$R_A = \\|\\Delta\\theta\\| / \\sigma_{½}$", fontsize=9)
        ax.set_ylabel(f"{p1} acc", fontsize=9)
        type_tag = "multi-class" if p1 in MULTI else "binary"
        ax.set_title(f"{label}  [{type_tag} Phase 1]", fontsize=9, color=color)
        ax.legend(fontsize=7.5, loc="lower left")

    fig.suptitle(
        "Fig 1 — $R_A$ is a monotone predictor of forgetting  "
        "(Llama-3.2-3B, all 6 task pairs)\n"
        "Shaded band = T* zone (r8). Vertical dotted = $R_A = 1$ certified bound.",
        fontsize=10, y=1.01
    )
    plt.tight_layout()
    out = "figures/ntrs/fig1_monotone.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out}")


# ── Fig 2: T* as Phase 1 property ────────────────────────────────────────────

def fig2_t_star_property():
    """
    Left panel: T*(r8) for each pair, grouped by Phase 1 model.
                Multi-class Phase 1 → near-constant T* across Phase 2 tasks.
                Binary Phase 1 → wildly different T* by Phase 2 task.
    Right panel: T* consistency metric (max/min ratio within Phase 1 group).
    """
    fig, (ax_main, ax_ratio) = plt.subplots(1, 2, figsize=(12, 5),
                                             gridspec_kw={"width_ratios": [3, 1]})

    # ── Left: bar chart grouped by Phase 1 model ─────────────────────────────
    groups = {
        "MNLI\n(multi-class)":   [("MNLI→SST-2", "→SST-2"), ("MNLI→AGNews", "→AGNews")],
        "AGNews\n(multi-class)": [("AGNews→MNLI", "→MNLI"),  ("AGNews→SST-2", "→SST-2")],
        "SST-2\n(binary)":       [("SST-2→MNLI",  "→MNLI"),  ("SST-2→AGNews", "→AGNews")],
    }
    p1_key = {"MNLI\n(multi-class)": "MNLI",
              "AGNews\n(multi-class)": "AGNews",
              "SST-2\n(binary)": "SST-2"}

    group_centers = np.array([0, 2.5, 5.0])
    offsets = [-0.45, 0.45]
    p2_styles = {
        "→SST-2":  {"hatch": "",    "alpha": 0.85},
        "→AGNews": {"hatch": "///", "alpha": 0.75},
        "→MNLI":   {"hatch": "...", "alpha": 0.75},
    }

    t_star_data = {}  # for ratio panel
    for gc, (g_label, pairs) in zip(group_centers, groups.items()):
        p1 = p1_key[g_label]
        color = C[p1]
        tvals = []
        for offset, (pair_name, p2_label) in zip(offsets, pairs):
            tag, _, _, sigma, base = PAIRS[pair_name]
            rows = load_pair(tag)
            if rows is None:
                continue
            lo, hi = t_star(rows, base, sigma, 8)
            mid = (lo + hi) / 2 if hi else None
            tvals.append(mid or lo)
            style = p2_styles[p2_label]
            bar_height = mid if mid else lo
            bar_x = gc + offset
            ax_main.bar(bar_x, bar_height, width=0.75, color=color,
                        alpha=style["alpha"], hatch=style["hatch"],
                        edgecolor=color, linewidth=1.2, zorder=3)
            if hi:
                err_lo = bar_height - lo
                err_hi = hi - bar_height
                ax_main.errorbar(bar_x, bar_height,
                                 yerr=[[err_lo], [err_hi]],
                                 fmt="none", color="black",
                                 capsize=4, linewidth=1.2, zorder=4)
            elif lo > 0:
                ax_main.annotate(f">{lo:.2f}", (bar_x, lo + 0.008),
                                 ha="center", fontsize=8, color=color)
            ax_main.text(bar_x, -0.022, p2_label, ha="center",
                         fontsize=8, color="gray", rotation=0)

        t_star_data[g_label] = tvals

    ax_main.set_xticks(group_centers)
    ax_main.set_xticklabels(list(groups.keys()), fontsize=9.5)
    ax_main.set_ylabel("$T^*$ (rank 8 LoRA)", fontsize=10)
    ax_main.set_ylim(-0.04, 0.30)
    ax_main.set_xlim(-0.9, 6.0)
    ax_main.axhline(0.062, color="gray", lw=1, ls="--", alpha=0.6)
    ax_main.text(5.85, 0.063, "0.062", fontsize=8, color="gray", va="bottom", ha="right")

    # Vertical separators between groups
    for xsep in [1.35, 3.85]:
        ax_main.axvline(xsep, color="lightgray", lw=1.2, ls="-", zorder=0)

    # Annotations
    ax_main.text(1.25, 0.26, "T* consistent\n(<5% variation)",
                 ha="center", fontsize=8.5, color=C["MNLI"],
                 style="italic", bbox=dict(boxstyle="round,pad=0.2", fc="white", alpha=0.7))
    ax_main.text(3.75, 0.26, "T* consistent\n(<5% variation)",
                 ha="center", fontsize=8.5, color=C["AGNews"],
                 style="italic", bbox=dict(boxstyle="round,pad=0.2", fc="white", alpha=0.7))
    ax_main.text(6.1, 0.26, "T* varies\n>8×  ✗",
                 ha="center", fontsize=8.5, color=C["SST-2"],
                 style="italic", bbox=dict(boxstyle="round,pad=0.2", fc="white", alpha=0.7))

    # Legend
    leg = [mpatches.Patch(fc="gray", alpha=0.85, label="Phase 2 = task A"),
           mpatches.Patch(fc="gray", alpha=0.75, hatch="///", label="Phase 2 = task B")]
    ax_main.legend(handles=leg, fontsize=8, loc="upper left")
    ax_main.set_title("T* for each task pair  (grouped by Phase 1 model, rank 8 LoRA)",
                      fontsize=10)

    # ── Right: consistency ratio panel ───────────────────────────────────────
    labels, ratios, colors = [], [], []
    for g_label, tvals in t_star_data.items():
        if len(tvals) >= 2 and all(v and v > 0 for v in tvals):
            ratio = max(tvals) / min(tvals)
        else:
            ratio = float("nan")
        labels.append(g_label.replace("\n", " "))
        ratios.append(ratio)
        colors.append(C[p1_key[g_label]])

    bars = ax_ratio.barh(labels, ratios, color=colors, alpha=0.75,
                         edgecolor=colors, linewidth=1.2)
    ax_ratio.axvline(1.0, color="gray", lw=0.8, ls="--", alpha=0.5)
    ax_ratio.axvline(1.2, color="orange", lw=1, ls=":", alpha=0.8)
    ax_ratio.set_xlabel("max(T*) / min(T*)\nacross Phase 2 tasks", fontsize=9)
    ax_ratio.set_title("T* stability\n(1.0 = perfect property)", fontsize=9)
    ax_ratio.set_xlim(0, max(r for r in ratios if not np.isnan(r)) * 1.15)
    for i, (r, label) in enumerate(zip(ratios, labels)):
        if not np.isnan(r):
            ax_ratio.text(r + 0.05, i, f"{r:.1f}×", va="center", fontsize=9)

    ax_ratio.text(1.2, -0.6, "20% band", color="orange",
                  fontsize=7.5, rotation=90, va="bottom", alpha=0.8)

    plt.tight_layout()
    out = "figures/ntrs/fig2_t_star_property.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out}")


# ── Fig 3: Binary Phase 1 anomaly ────────────────────────────────────────────

def fig3_binary_anomaly():
    """
    2×2 grid:
      Row 1 = SST-2→MNLI  (Phase 2 where SST-2 Ph1 is SAFE)
      Row 2 = SST-2→AGNews (Phase 2 where SST-2 Ph1 COLLAPSES immediately)
      Cols = rank 8 and rank 32
    Shows how identical Phase 1 model has wildly different T* by Phase 2 task.
    """
    fig, axes = plt.subplots(2, 2, figsize=(10, 7), sharey=False)
    pairs_here = [
        ("SST-2→MNLI",   "sst2_to_mnli",   "MNLI",   0.003337, 0.975),
        ("SST-2→AGNews", "sst2_to_agnews",  "AGNews", 0.003337, 0.975),
    ]
    color = C["SST-2"]
    thresh = 0.975 * 0.9  # 0.8775

    for row, (label, tag, p2, sigma, base) in enumerate(pairs_here):
        rows = load_pair(tag)
        if rows is None:
            for ax in axes[row]:
                ax.set_title(f"{label} (missing)", fontsize=9)
            continue

        for col, rank in enumerate([8, 32]):
            ax = axes[row][col]
            rs = sorted([r for r in rows if r["rank"] == rank],
                        key=lambda x: x["final_per_param"])
            xs = [r["final_per_param"] / sigma for r in rs]
            ys = [r["final_acc_task_a"] for r in rs]

            ax.plot(xs, ys, "o-", color=color, markersize=9, linewidth=2,
                    markeredgecolor="white", markeredgewidth=0.8, zorder=3)

            # Mark safe vs catastrophic
            for x, y, r in zip(xs, ys, rs):
                clr = "green" if y >= thresh else "darkred"
                ax.scatter(x, y, s=80, color=clr, zorder=4, linewidth=0)

            ax.axhline(thresh, color="black", lw=1.2, ls="--", alpha=0.7,
                       label=f"90% of {base:.2f} = {thresh:.3f}")

            lo, hi = t_star(rows, base, sigma, rank)
            if hi:
                mid = (lo + hi) / 2
                ax.axvspan(lo, hi, alpha=0.18, color=color)
                ax.text((lo + hi) / 2, thresh + 0.01,
                        f"T*≈{mid:.3f}", ha="center", fontsize=8.5, color=color,
                        bbox=dict(boxstyle="round,pad=0.15", fc="white", alpha=0.8))
            else:
                ax.annotate(f"T* > {lo:.3f}\n(never catastrophic)",
                            xy=(lo, thresh + 0.02), fontsize=8.5, color="green",
                            bbox=dict(boxstyle="round,pad=0.15", fc="white", alpha=0.8))

            ax.set_xlim(left=0)
            ax.set_ylim(0.15, 1.08)
            ax.set_xlabel("$R_A = \\|\\Delta\\theta\\| / \\sigma_{½}$", fontsize=9)
            ax.set_ylabel("SST-2 acc (Phase 1 task)", fontsize=9)
            ax.set_title(f"{label}  [rank {rank}]", fontsize=9.5,
                         color=("green" if label == "SST-2→MNLI" else "darkred"))
            ax.legend(fontsize=8, loc="lower left")

    # Row labels
    for row, lbl in enumerate(["Phase 2 = MNLI\n→ SST-2 Phase 1 stays safe (rank 8)",
                                "Phase 2 = AGNews\n→ SST-2 Phase 1 collapses immediately (rank 32)"]):
        axes[row][0].set_ylabel(lbl + "\n\nSST-2 acc", fontsize=8.5)

    fig.suptitle(
        "Fig 3 — Binary Phase 1 (SST-2): T* depends on Phase 2 task  "
        "(same Phase 1 model, σ½=0.003337)\n"
        "Green dots = safe (acc ≥ 90% threshold), red dots = catastrophic",
        fontsize=9.5, y=1.01
    )
    plt.tight_layout()
    out = "figures/ntrs/fig3_binary_anomaly.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out}")


# ── Fig 4: σ½ density curves (training geometry) ─────────────────────────────

def fig4_density_curves():
    fig, ax = plt.subplots(figsize=(7, 4.5))

    phase1_data = [
        ("MNLI",   "mnli_to_sst2",   "multi-class (3)"),
        ("AGNews", "agnews_to_mnli",  "multi-class (4)"),
        ("SST-2",  "sst2_to_mnli",   "binary"),
    ]
    for p1, tag, desc in phase1_data:
        path = f"h2_llama/{tag}/phase1_sigma_half_acc.json"
        if not os.path.exists(path):
            continue
        with open(path) as f:
            d = json.load(f)
        sigmas = sorted(float(s) for s in d["densities"])
        dens   = [d["densities"][str(s)] for s in sigmas]
        color  = C[p1]
        ax.plot(sigmas, dens, "o-", color=color, markersize=5.5,
                linewidth=2, label=f"{p1}  [{desc}]  σ½={d['sigma_half']:.4f}")
        ax.axvline(d["sigma_half"], color=color, lw=1, ls=":", alpha=0.6)
        ax.text(d["sigma_half"] + 0.00003, 0.55, f"{d['sigma_half']:.4f}",
                color=color, fontsize=8, va="bottom")

    ax.axhline(0.5, color="gray", lw=0.8, ls="--", alpha=0.6, label="density = 0.5  (σ½ definition)")
    ax.set_xlabel("Perturbation σ (per-parameter weight noise std dev)", fontsize=10)
    ax.set_ylabel("Certified density\n(fraction of perturbations above accuracy threshold)", fontsize=9)
    ax.set_title("Fig 4 — σ½ density curves for all Phase 1 models\n"
                 "MNLI and AGNews are within 4%; SST-2 is 21% wider — "
                 "reflects training regime geometry, not task content",
                 fontsize=9.5)
    ax.legend(fontsize=9)
    ax.set_ylim(-0.05, 1.1)
    ax.set_xlim(left=0)
    plt.tight_layout()
    out = "figures/ntrs/fig4_density_curves.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out}")


# ── Run all ───────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    fig1_monotone()
    fig2_t_star_property()
    fig3_binary_anomaly()
    fig4_density_curves()
    print("\nAll figures in figures/ntrs/")
