"""
Standalone plot for H2 results from recompute_sigma_half_acc.py output.
Data entered from the printed table.
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Data from recompute_sigma_half_acc.py output
data = [
    # (lr, rank, R_A_acc, R_A_nll, Acc_A)
    (1e-5,  8,  0.0049, 0.0023, 0.8860),
    (5e-5,  8,  0.0193, 0.0091, 0.8800),
    (1e-4,  8,  0.0371, 0.0175, 0.8700),
    (2e-4,  8,  0.0753, 0.0356, 0.7880),
    (5e-4,  8,  0.2369, 0.1119, 0.3580),
    (1e-5, 32,  0.0084, 0.0040, 0.8840),
    (5e-5, 32,  0.0371, 0.0175, 0.8560),
    (1e-4, 32,  0.0770, 0.0363, 0.7520),
    (2e-4, 32,  0.2230, 0.1053, 0.3520),
    (5e-4, 32,  0.6606, 0.3119, 0.3520),
]

sigma_half_acc = 0.003120
sigma_half_nll = 0.006607
phase1_acc     = 0.884   # MNLI Phase 1 accuracy

lrs      = [d[0] for d in data]
ranks    = [d[1] for d in data]
ra_acc   = [d[2] for d in data]
ra_nll   = [d[3] for d in data]
acc_a    = [d[4] for d in data]

# Color by LR, marker by rank
lr_vals   = sorted(set(lrs))
lr_colors = {"1e-05": "#1a237e", "5e-05": "#1565c0",
             "1e-04": "#2e7d32", "2e-04": "#f57f17", "5e-04": "#b71c1c"}
rank_markers = {8: "o", 32: "s"}

def lr_key(lr):
    return f"{lr:.0e}"

fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))

# ── Panel 1: R_A_acc vs MNLI accuracy (the key H2 claim) ────────────────────
ax = axes[0]

# Shade zones
ax.axvspan(0, 0.15,  color="#e8f5e9", alpha=0.40, zorder=0)
ax.axvspan(0.15, 0.8, color="#ffebee", alpha=0.35, zorder=0)

for lr, rank, ra, acc in zip(lrs, ranks, ra_acc, acc_a):
    color  = lr_colors[lr_key(lr)]
    marker = rank_markers[rank]
    ax.scatter(ra, acc, color=color, s=180, marker=marker,
               edgecolors="white", linewidths=0.8, zorder=5)

# Add threshold annotation
ax.axvline(x=0.15, color="#888", linestyle="--", linewidth=1.4, alpha=0.8, zorder=3)
ax.text(0.02, 0.40, "safe zone\n(R_A < 0.15)", ha="left", va="center",
        fontsize=9.5, color="#2e7d32", fontstyle="italic")
ax.text(0.30, 0.40, "forgetting\n(R_A > 0.15)", ha="left", va="center",
        fontsize=9.5, color="#c62828", fontstyle="italic")

# Phase 1 accuracy reference line
ax.axhline(y=phase1_acc, color="#555", linestyle=":", linewidth=1.2, alpha=0.7)
ax.text(0.62, phase1_acc + 0.01, "Phase 1 MNLI acc (88.4%)",
        fontsize=8.5, color="#555", va="bottom")

# Legend: LR colors + rank markers
from matplotlib.lines import Line2D
legend_els = []
for lr, c in zip(lr_vals, ["#1a237e","#1565c0","#2e7d32","#f57f17","#b71c1c"]):
    legend_els.append(Line2D([0],[0], marker="o", color="w",
                              markerfacecolor=c, markersize=10, label=f"lr={lr:.0e}"))
legend_els.append(Line2D([0],[0], marker="o", color="gray",
                          markersize=10, label="rank=8 (●)"))
legend_els.append(Line2D([0],[0], marker="s", color="gray",
                          markersize=10, label="rank=32 (■)"))
ax.legend(handles=legend_els, fontsize=8.5, loc="center right")

ax.set_xlabel("R_A  =  ‖Δθ‖ / σ½_A_acc   (accuracy-based basin radius)", fontsize=11)
ax.set_ylabel("MNLI Accuracy after Phase 2 (Task A retention)", fontsize=11)
ax.set_title("H2: R_A predicts forgetting\n"
             "Phase 2 (SST-2) fine-tuning vs Task A (MNLI) retention",
             fontsize=11, pad=8)
ax.set_xlim(-0.01, 0.72)
ax.set_ylim(0.30, 0.96)
ax.grid(True, alpha=0.25)

# ── Panel 2: R_A_acc vs R_A_nll comparison ───────────────────────────────────
ax = axes[1]

for lr, rank, ra_a, ra_n, acc in zip(lrs, ranks, ra_acc, ra_nll, acc_a):
    color  = lr_colors[lr_key(lr)]
    marker = rank_markers[rank]
    ax.scatter(ra_n, acc, color=color, s=180, marker=marker,
               facecolors="none", edgecolors=color, linewidths=2.0, zorder=5,
               label=f"NLL-based")  # hollow = NLL-based
    ax.scatter(ra_a, acc, color=color, s=180, marker=marker,
               edgecolors="white", linewidths=0.8, zorder=5)  # filled = acc-based

ax.axvline(x=0.15, color="#888", linestyle="--", linewidth=1.4, alpha=0.8)
ax.text(0.16, 0.36, "empirical\nthreshold", ha="left", fontsize=8.5, color="#888")

ax.axhline(y=phase1_acc, color="#555", linestyle=":", linewidth=1.2, alpha=0.7)

# Arrow showing acc-based vs nll-based for worst condition
ax.annotate("", xy=(0.6606, 0.352), xytext=(0.3119, 0.352),
            arrowprops=dict(arrowstyle="->", color="#555", lw=1.5))
ax.text(0.45, 0.365, "acc-based ← NLL-based\n(filled ← hollow)", ha="center",
        fontsize=8, color="#555")

from matplotlib.patches import Patch
legend_el2 = [
    Line2D([0],[0], marker="o", color="w", markerfacecolor="gray",
           markersize=10, label="acc-based σ½ (filled)"),
    Line2D([0],[0], marker="o", color="gray", markerfacecolor="none",
           markersize=10, markeredgewidth=2.0, label="NLL-based σ½ (hollow)"),
    Line2D([0],[0], linestyle="--", color="#888", label="empirical threshold (0.15)"),
]
ax.legend(handles=legend_el2, fontsize=8.5, loc="center right")

ax.set_xlabel("R_A  (acc-based filled, NLL-based hollow)", fontsize=11)
ax.set_ylabel("MNLI Accuracy after Phase 2", fontsize=11)
ax.set_title("Acc-based vs NLL-based σ½\nBoth still predict R_A < 1 at forgetting",
             fontsize=11, pad=8)
ax.set_xlim(-0.01, 0.72)
ax.set_ylim(0.30, 0.96)
ax.grid(True, alpha=0.25)

plt.suptitle(
    "H2 — R_A predicts Task A forgetting, but σ½ computed from isotropic noise\n"
    "overestimates safe budget: fine-tuning gradients are directionally adversarial (R_A_threshold ≈ 0.15, not 1.0)",
    fontsize=10, y=1.01
)
plt.tight_layout()
out = "h2_figure.png"
plt.savefig(out, dpi=180, bbox_inches="tight")
plt.close()
print(f"Saved: {out}")
