"""
make_h2_tstar_9pairs_figure.py

Fresh H2 figure covering all 9 currently-measured Phase1->Phase2 pairs at
rank 8, pulled directly from the T* values recorded in SUMMARY.md (cross-
experiment summary table + H2-I). Shows the two-part H2 story:
  - AGNews / MNLI / CoLA Phase 1: T* clusters tightly across Phase 2 tasks
    (the "Phase 1 property" finding).
  - SST-2 Phase 1: T* spans a 44x range depending on the Phase 2 task
    (the anisotropic-basin finding).

Run:
  python make_h2_tstar_9pairs_figure.py
Out:
  figures/fig_h2_tstar_9pairs.png
"""

import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

os.makedirs("figures", exist_ok=True)

# ── categorical palette (fixed order, validated) ────────────────────────────
COLOR = {
    "AGNews": "#2a78d6",   # slot 1 blue
    "MNLI":   "#1baf7a",   # slot 2 aqua
    "CoLA":   "#eda100",   # slot 3 yellow
    "SST-2":  "#008300",   # slot 4 green
}
INK          = "#0b0b0b"
INK_SECOND   = "#52514e"
GRID         = "#e5e4e0"
CLUSTER_FILL = "#f0efec"

# ── T*(rank 8) data, from SUMMARY.md (2026-07-05 state) ────────────────────
# (phase1, phase2, low, mid, high)
DATA = [
    ("AGNews", "MNLI",   0.042, 0.064, 0.085),
    ("AGNews", "SST-2",  0.042, 0.065, 0.089),
    ("MNLI",   "AGNews", 0.040, 0.062, 0.084),
    ("MNLI",   "SST-2",  0.041, 0.058, 0.081),
    ("CoLA",   "MNLI",   0.060, 0.089, 0.118),
    ("CoLA",   "AGNews", 0.060, 0.093, 0.126),
    ("SST-2",  "AGNews", 0.036, 0.051, 0.073),
    ("SST-2",  "MNLI",   0.212, 0.532, 0.851),
    ("SST-2",  "CoLA",   0.0046, 0.012, 0.0187),
]

ORDER = ["AGNews", "CoLA", "MNLI", "SST-2"]
rows = sorted(DATA, key=lambda d: (ORDER.index(d[0]), d[1]))

fig, ax = plt.subplots(figsize=(8.8, 6.0), dpi=200)
fig.patch.set_facecolor("white")
ax.set_facecolor("white")

n = len(rows)
ax.set_ylim(-0.8, n - 1 + 1.1)

# Shaded band — visual grouping only, no text
ax.axvspan(0.036, 0.126, color=CLUSTER_FILL, zorder=0)

y_labels = []
for i, (p1, p2, lo, mid, hi) in enumerate(rows):
    y = n - 1 - i
    c = COLOR[p1]
    ax.plot([lo, hi], [y, y], color=c, lw=2, solid_capstyle="round", zorder=2)
    ax.plot(mid, y, "o", color=c, markersize=8, markeredgecolor="white",
             markeredgewidth=1.2, zorder=3)
    label_x = hi * 1.18 if hi < 0.3 else hi * 1.12
    ax.text(label_x, y, f"{mid:.3f}", va="center", ha="left",
            fontsize=9, color=INK)
    y_labels.append(f"{p1} → {p2}")

ax.set_yticks(range(n))
ax.set_yticklabels(y_labels[::-1], fontsize=9.5, color=INK)
ax.set_xscale("log")
ax.set_xlim(0.003, 1.6)
ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{v:g}"))
ax.set_xlabel("T∗ (R_A at forgetting threshold, rank 8)", fontsize=10.5, color=INK)

for spine in ["top", "right"]:
    ax.spines[spine].set_visible(False)
for spine in ["left", "bottom"]:
    ax.spines[spine].set_color(GRID)
ax.tick_params(colors=INK_SECOND)
ax.grid(axis="x", which="major", color=GRID, lw=0.8, zorder=0)
ax.grid(axis="x", which="minor", color=GRID, lw=0.4, zorder=0, alpha=0.6)
ax.set_axisbelow(True)

# Legend — placed in the empty upper-left region (fixed categorical order)
handles = [plt.Line2D([0], [0], marker="o", color=COLOR[k], linestyle="",
                       markersize=8, markeredgecolor="white", markeredgewidth=1)
           for k in ORDER]
ax.legend(handles, [f"Phase 1: {k}" for k in ORDER], loc="upper left",
          bbox_to_anchor=(0.003, 0.985), frameon=False, fontsize=9,
          labelcolor=INK, handletextpad=0.6, borderaxespad=0)

fig.suptitle(
    "H2 forgetting threshold T∗ across all 9 measured task pairs (rank 8)",
    fontsize=13, color=INK, x=0.02, y=0.975, ha="left", fontweight="bold")

plt.tight_layout(rect=[0.0, 0.0, 0.98, 0.93])
out = "figures/fig_h2_tstar_9pairs.png"
plt.savefig(out, dpi=200, bbox_inches="tight", facecolor="white")
print(f"Saved {out}")
