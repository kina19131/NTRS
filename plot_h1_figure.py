"""
Standalone plot for H1 key figure.
Reads h1_kl_comp_v2/gpt2/summary.json and writes h1_figure.png.
"""
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

with open("h1_kl_comp_v2/gpt2/summary.json") as f:
    data = json.load(f)

lrs       = [r["lr"]            for r in data]
rs        = [r["R"]             for r in data]
delta_nll = [r["delta_nll_wiki"] for r in data]
kl_wiki   = [r["kl_wiki"]       for r in data]
kl_div    = [r["kl_diverse"]     for r in data]
accs      = [r["acc"]            for r in data]
sigma_half = data[0]["sigma_half_pre"]
nll_base   = data[0]["nll_wiki_base"]

lr_labels = [f"lr={lr:.0e}" for lr in lrs]
colors    = ["#1a237e", "#1565c0", "#2e7d32", "#f57f17", "#b71c1c"]

fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))

# ── Panel 1: R vs Δ NLL  (main H1 claim) ────────────────────────────────────
ax = axes[0]

ax.axvline(x=1.0, color="#d32f2f", linestyle="--", linewidth=1.8,
           alpha=0.85, zorder=1, label="R = 1 threshold")
ax.axhline(y=0.0, color="#555", linestyle=":", linewidth=1.2, alpha=0.6, zorder=1)

for i, (r, dn, lr, lbl) in enumerate(zip(rs, delta_nll, lrs, lr_labels)):
    ax.scatter(r, dn, color=colors[i], s=160, zorder=5,
               edgecolors="white", linewidths=0.8)
    # label offset: avoid overlap
    x_off = 0.3 if r < 15 else -2.0
    y_off = 0.05 if dn >= 0 else -0.20
    ax.annotate(lbl, (r, dn),
                xytext=(r + x_off, dn + y_off),
                fontsize=9, color=colors[i], fontweight="bold",
                va="bottom" if dn >= 0 else "top")

# shade the "safe" region
ax.axvspan(0, 1.0, color="#e8f5e9", alpha=0.45, zorder=0)
ax.axvspan(1.0, 22, color="#ffebee", alpha=0.35, zorder=0)
ax.text(0.50, 4.0, "safe\n(R < 1)", ha="center", va="center",
        fontsize=10, color="#2e7d32", fontstyle="italic")
ax.text(11,   4.0, "degraded\n(R > 1)", ha="center", va="center",
        fontsize=10, color="#c62828", fontstyle="italic")

ax.set_xlabel("R  =  ‖Δθ‖ / σ½  (weight displacement / certified radius)",
              fontsize=11)
ax.set_ylabel(f"Δ NLL  =  NLL_ft − NLL_base  (base NLL = {nll_base:.2f})",
              fontsize=11)
ax.set_title("R  predicts pretrained-capability degradation\n"
             "(Δ NLL > 0  =  capability forgotten)", fontsize=12, pad=8)
ax.set_xlim(-0.5, 22)
ax.set_ylim(-0.5, 5.0)
ax.legend(fontsize=10, loc="upper left")
ax.grid(True, alpha=0.25)

# ── Panel 2: KL vs Δ NLL  (KL as comparator) ────────────────────────────────
ax = axes[1]

ax.axhline(y=0.0, color="#555", linestyle=":", linewidth=1.2, alpha=0.6)

for i, (kw, kd, dn, lbl) in enumerate(zip(kl_wiki, kl_div, delta_nll, lr_labels)):
    ax.scatter(kw, dn, color=colors[i], s=160, marker="o", zorder=5,
               edgecolors="white", linewidths=0.8)
    ax.scatter(kd, dn, color=colors[i], s=160, marker="^", zorder=5,
               edgecolors="white", linewidths=0.8)

# Draw non-monotone arrow for KL diverse at R=5→R=20 point
# lr=2e-4 diverse (4.10, 1.39) → lr=5e-4 diverse (3.53, 4.31)
ax.annotate("", xy=(kl_div[-1], delta_nll[-1]), xytext=(kl_div[-2], delta_nll[-2]),
            arrowprops=dict(arrowstyle="->", color="#b71c1c", lw=1.6))
ax.text(4.3, 2.8, "KL drops\nbut ΔNLL rises\n(non-monotone)", fontsize=8.5,
        color="#b71c1c", ha="left", va="center",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="#fff9c4", alpha=0.85))

wiki_patch    = mpatches.Patch(facecolor="gray", label="KL on WikiText-2 (●)")
diverse_patch = mpatches.Patch(facecolor="gray",
                                label="KL on diverse/instruction prompts (▲)")
ax.legend(handles=[wiki_patch, diverse_patch], fontsize=9, loc="upper left")

# annotate lr labels on wiki points only (avoid clutter)
for i, (kw, dn, lbl) in enumerate(zip(kl_wiki, delta_nll, lr_labels)):
    ax.annotate(lbl, (kw, dn), xytext=(kw + 0.1, dn + 0.1),
                fontsize=8, color=colors[i], fontweight="bold")

ax.set_xlabel("KL( base ‖ fine-tuned )", fontsize=11)
ax.set_ylabel("Δ NLL  (capability degradation)", fontsize=11)
ax.set_title("KL on diverse/instruction prompts is\nnon-monotone — fails to track degradation",
             fontsize=12, pad=8)
ax.set_xlim(-0.1, 5.5)
ax.set_ylim(-0.5, 5.0)
ax.grid(True, alpha=0.25)

plt.suptitle(
    "H1 — σ½ basin radius predicts pretrained-capability degradation at the pretraining→SFT boundary;\n"
    "KL on instruction prompts is a non-monotone, unreliable predictor",
    fontsize=11, y=1.01
)
plt.tight_layout()
out = "h1_figure.png"
plt.savefig(out, dpi=180, bbox_inches="tight")
plt.close()
print(f"Saved: {out}")
