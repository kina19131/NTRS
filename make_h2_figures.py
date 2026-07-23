"""
make_h2_figures.py
==================
Generate all H2 publication figures from local h2_llama/ results.

Figures produced:
  figures/h2/fig_t_star_property.png   — T* as Phase 1 property (key claim)
  figures/h2/fig_forgetting_curves.png — R_A vs Phase 1 Acc for all 5 pairs
  figures/h2/fig_density_curves.png    — σ½ density curves for all 3 Phase 1 models
  figures/h2/fig_rank_effect.png       — Rank 8 vs 32 T* comparison
"""

import json, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

os.makedirs("figures/h2", exist_ok=True)

# ── Data ─────────────────────────────────────────────────────────────────────

PAIRS = {
    "MNLI→SST-2":   ("mnli_to_sst2",   "MNLI",   "SST-2",  0.002857,           0.884),
    "MNLI→AGNews":  ("mnli_to_agnews",  "MNLI",   "AGNews", 0.002857,           0.884),
    "AGNews→MNLI":  ("agnews_to_mnli",  "AGNews", "MNLI",   0.002755586419753086, 0.905),
    "AGNews→SST-2": ("agnews_to_sst2",  "AGNews", "SST-2",  0.002755586419753086, 0.905),
    "SST-2→MNLI":   ("sst2_to_mnli",   "SST-2",  "MNLI",   0.003337373737373738, 0.975),
    "SST-2→AGNews": ("sst2_to_agnews",  "SST-2",  "AGNews", 0.003337373737373738, 0.975),
}

PHASE1_COLORS = {"MNLI": "#1f77b4", "AGNews": "#2ca02c", "SST-2": "#d62728"}
TASK2_MARKERS = {"SST-2": "o", "MNLI": "s", "AGNews": "^"}

def load_pair(tag):
    path = f"h2_llama/{tag}/summary_acc_sigma.json"
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)

def load_density(tag):
    path = f"h2_llama/{tag}/phase1_sigma_half_acc.json"
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)

def t_star_band(rows, baseline, acc_slack=0.10):
    thresh = baseline * (1 - acc_slack)
    for rank in [8, 32]:
        rs = sorted([r for r in rows if r["rank"] == rank],
                    key=lambda x: x["R_A_acc"])
        last_safe = next((r for r in reversed(rs)
                          if r["final_acc_task_a"] >= thresh), None)
        first_bad = next((r for r in rs
                          if r["final_acc_task_a"] < thresh), None)
        lo = last_safe["R_A_acc"] if last_safe else 0.0
        hi = first_bad["R_A_acc"] if first_bad else None
        yield rank, lo, hi


# ── Fig 1: T* as Phase 1 property ────────────────────────────────────────────

def fig_t_star_property():
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.5), sharey=False)
    phase1_models = ["MNLI", "AGNews", "SST-2"]
    pair_groups = {
        "MNLI":   [("MNLI→SST-2",   "SST-2"),  ("MNLI→AGNews",  "AGNews")],
        "AGNews": [("AGNews→MNLI",  "MNLI"),    ("AGNews→SST-2", "SST-2")],
        "SST-2":  [("SST-2→MNLI",   "MNLI"),   ("SST-2→AGNews", "AGNews")],
    }

    for ax, p1 in zip(axes, phase1_models):
        pairs_here = pair_groups[p1]
        color = PHASE1_COLORS[p1]
        x_pos = 0
        xticks, xlabels = [], []

        for label, (pair_name, p2_task) in enumerate(pairs_here):
            tag, _, _, sigma, base = PAIRS[pair_name]
            rows = load_pair(tag)
            if rows is None:
                continue

            for rank, lo, hi in t_star_band(rows, base):
                marker_x = x_pos
                if hi is None:
                    ax.annotate(f">{lo:.3f}", xy=(marker_x, lo + 0.01),
                                ha="center", fontsize=8, color=color)
                    ax.bar(marker_x, lo, width=0.6, bottom=0,
                           color=color, alpha=0.3, edgecolor=color, linewidth=1.2)
                else:
                    mid = (lo + hi) / 2
                    err_lo = mid - lo
                    err_hi = hi - mid
                    ax.bar(marker_x, mid, width=0.6,
                           color=color, alpha=0.7 if rank == 8 else 0.35,
                           edgecolor=color, linewidth=1.2)
                    ax.errorbar(marker_x, mid, yerr=[[err_lo], [err_hi]],
                                fmt="none", color="black", capsize=4, linewidth=1.2)

                xtick_label = f"{p2_task}\nr{rank}"
                xticks.append(x_pos)
                xlabels.append(xtick_label)
                x_pos += 1

            x_pos += 0.5  # gap between Phase 2 tasks

        ax.set_xticks(xticks)
        ax.set_xticklabels(xlabels, fontsize=8)
        ax.set_title(f"Phase 1: {p1}\n(σ½={[v[3] for k,v in PAIRS.items() if v[1]==p1][0]:.4f})",
                     fontsize=10, color=color)
        ax.set_ylabel("T* (R_A at onset of forgetting)", fontsize=9)
        ax.set_ylim(0, 0.25)
        ax.axhline(0, color="gray", lw=0.5)
        ax.grid(axis="y", alpha=0.3)

        # Consistency annotation
        if p1 == "MNLI":
            ax.text(0.5, 0.92, "r8 consistent ✓\nr32 varies ×2",
                    transform=ax.transAxes, ha="center", fontsize=7.5,
                    color="gray", style="italic")
        elif p1 == "AGNews":
            ax.text(0.5, 0.92, "Both ranks\nconsistent ✓",
                    transform=ax.transAxes, ha="center", fontsize=7.5,
                    color="gray", style="italic")
        elif p1 == "SST-2":
            ax.text(0.5, 0.92, "NOT Phase 1\nproperty ×8\n(Phase 2 task matters)",
                    transform=ax.transAxes, ha="center", fontsize=7.5,
                    color="gray", style="italic")

    # Legend for rank
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor="gray", alpha=0.7, label="Rank 8"),
        Patch(facecolor="gray", alpha=0.35, label="Rank 32"),
    ]
    fig.legend(handles=legend_elements, loc="lower center", ncol=2,
               fontsize=9, bbox_to_anchor=(0.5, -0.04))
    fig.suptitle("H2: T* as Phase 1 model property  (Llama-3.2-3B, 6 task pairs)",
                 fontsize=12, y=1.02)
    plt.tight_layout()
    out = "figures/h2/fig_t_star_property.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out}")


# ── Fig 2: Forgetting curves for all 5 pairs ─────────────────────────────────

def fig_forgetting_curves():
    fig, axes = plt.subplots(1, 6, figsize=(21, 4), sharey=False)

    for ax, (label, (tag, p1, p2, sigma, base)) in zip(axes, PAIRS.items()):
        rows = load_pair(tag)
        if rows is None:
            ax.set_title(f"{label}\n(missing)", fontsize=9)
            continue

        thresh = base * 0.9
        color = PHASE1_COLORS[p1]

        for rank, marker_style in [(8, "o"), (32, "s")]:
            rs = [r for r in rows if r["rank"] == rank]
            x = [r["R_A_acc"] for r in rs]
            y = [r["final_acc_task_a"] for r in rs]
            ax.scatter(x, y, c=color, marker=marker_style,
                       s=60, alpha=0.9 if rank == 8 else 0.5,
                       label=f"r{rank}", zorder=3)

        ax.axhline(thresh, color="black", lw=0.8, ls="--", alpha=0.6,
                   label=f"90% base ({thresh:.2f})")
        ax.axvline(1.0, color="red", lw=0.8, ls="--", alpha=0.5, label="R_A=1")
        ax.set_xlabel("R_A = ‖Δθ‖ / σ½_A", fontsize=8)
        ax.set_ylabel(f"{p1} Acc (Phase 1 task)", fontsize=8)
        ax.set_title(f"{label}", fontsize=9)
        ax.set_xlim(left=0)
        ax.set_ylim(0.2, 1.0)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=7)

    fig.suptitle("H2: R_A as monotone predictor of forgetting  (Llama-3.2-3B, 6 pairs)",
                 fontsize=11, y=1.02)
    plt.tight_layout()
    out = "figures/h2/fig_forgetting_curves.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out}")


# ── Fig 3: σ½ density curves ─────────────────────────────────────────────────

def fig_density_curves():
    phase1_tags = [
        ("MNLI",   "mnli_to_sst2",  0.884),
        ("AGNews", "agnews_to_mnli", 0.905),
        ("SST-2",  "sst2_to_mnli",  0.975),
    ]

    fig, ax = plt.subplots(figsize=(7, 4))

    for p1, tag, base in phase1_tags:
        d = load_density(tag)
        if d is None:
            continue
        sigmas = sorted(float(s) for s in d["densities"])
        dens   = [d["densities"][str(s)] for s in sigmas]
        color  = PHASE1_COLORS[p1]
        ax.plot(sigmas, dens, "o-", color=color, label=f"{p1} (σ½={d['sigma_half']:.4f})",
                markersize=5, linewidth=1.8)
        ax.axvline(d["sigma_half"], color=color, lw=1, ls=":", alpha=0.7)

    ax.set_xlabel("Perturbation σ", fontsize=10)
    ax.set_ylabel("Certified density  (fraction perturbations above threshold)", fontsize=9)
    ax.set_title("σ½ density curves — all Phase 1 models  (Llama-3.2-3B)", fontsize=11)
    ax.axhline(0.5, color="gray", lw=0.8, ls="--", alpha=0.6, label="density = 0.5 (σ½ definition)")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)
    ax.set_ylim(-0.05, 1.05)

    plt.tight_layout()
    out = "figures/h2/fig_density_curves.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out}")


# ── Fig 4: Rank effect summary ────────────────────────────────────────────────

def fig_rank_effect():
    fig, ax = plt.subplots(figsize=(9, 4.5))

    pair_order = list(PAIRS.keys())
    x = np.arange(len(pair_order))
    width = 0.35

    t_r8, t_r32 = [], []
    t_r8_hi, t_r32_hi = [], []

    for label in pair_order:
        tag, p1, _, sigma, base = PAIRS[label]
        rows = load_pair(tag)
        thresh = base * 0.9

        lo8 = hi8 = lo32 = hi32 = None
        if rows:
            for rank, lo, hi in t_star_band(rows, base):
                if rank == 8:
                    lo8, hi8 = lo, hi
                else:
                    lo32, hi32 = lo, hi

        mid8  = (lo8  + hi8)  / 2 if (lo8  is not None and hi8  is not None) else lo8 or 0
        mid32 = (lo32 + hi32) / 2 if (lo32 is not None and hi32 is not None) else lo32 or 0
        t_r8.append(mid8)
        t_r32.append(mid32)
        t_r8_hi.append(hi8 - mid8 if hi8 else 0.02)
        t_r32_hi.append(hi32 - mid32 if hi32 else 0.02)

    colors = [PHASE1_COLORS[PAIRS[l][1]] for l in pair_order]

    bars8  = ax.bar(x - width/2, t_r8,  width, label="Rank 8",
                    color=colors, alpha=0.85, edgecolor="black", linewidth=0.8)
    bars32 = ax.bar(x + width/2, t_r32, width, label="Rank 32",
                    color=colors, alpha=0.4,  edgecolor="black", linewidth=0.8)

    ax.errorbar(x - width/2, t_r8,  yerr=[np.zeros(len(t_r8)),  t_r8_hi],
                fmt="none", color="black", capsize=3, linewidth=1)
    ax.errorbar(x + width/2, t_r32, yerr=[np.zeros(len(t_r32)), t_r32_hi],
                fmt="none", color="black", capsize=3, linewidth=1)

    ax.set_xticks(x)
    ax.set_xticklabels(pair_order, fontsize=9, rotation=15, ha="right")
    ax.set_ylabel("T*  (R_A at onset of forgetting)", fontsize=10)
    ax.set_title("Rank 8 vs Rank 32 T* across all task pairs", fontsize=11)
    ax.legend(fontsize=9)
    ax.grid(axis="y", alpha=0.3)
    ax.set_ylim(0, 0.25)

    from matplotlib.patches import Patch
    legend_phase1 = [Patch(facecolor=c, label=p1)
                     for p1, c in PHASE1_COLORS.items()]
    ax.legend(handles=legend_phase1 + [
        Patch(facecolor="gray", alpha=0.85, label="Rank 8"),
        Patch(facecolor="gray", alpha=0.4,  label="Rank 32"),
    ], fontsize=8, ncol=2, loc="upper right")

    plt.tight_layout()
    out = "figures/h2/fig_rank_effect.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out}")


# ── Run all ───────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    fig_t_star_property()
    fig_forgetting_curves()
    fig_density_curves()
    fig_rank_effect()
    print("\nDone. All figures in figures/h2/")
