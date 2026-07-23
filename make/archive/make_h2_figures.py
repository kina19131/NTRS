"""
H2 Forgetting Budget figures.
Run: python3 make_h2_figures.py
Saves to outputs_0618_h2/figures/
"""
import json, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D

BASE     = "outputs_0618_h2/gpt2/seq_agnews_to_sst2"
OUT_DIR  = "outputs_0618_h2/figures"
SH_A     = 0.0006363636   # σ½_A from phase1_task_a_density.json
BASE_ACC = 0.8728813559322034
SIGMA_PRE = 0.003031250   # pretrained σ½ (used internally by the trajectory norm)

os.makedirs(OUT_DIR, exist_ok=True)

CONDITIONS = [
    ("lr1e-05_rank8",  "1e-5", 8,  "#4393c3", "o"),
    ("lr1e-05_rank32", "1e-5", 32, "#4393c3", "s"),
    ("lr1e-04_rank8",  "1e-4", 8,  "#f4a582", "o"),
    ("lr1e-04_rank32", "1e-4", 32, "#d6604d", "s"),
    ("lr2e-04_rank8",  "2e-4", 8,  "#b2182b", "o"),
    ("lr2e-04_rank32", "2e-4", 32, "#67001f", "s"),
]

# ── Load data ──────────────────────────────────────────────────────────────────
def load_trajectory(fname):
    with open(f"{BASE}/{fname}_steps2000_trajectory.json") as f:
        t = json.load(f)
    traj = t["trajectory"]
    steps, ra_vals, acc_a, acc_b = [], [], [], []
    for pt in traj:
        total_norm = pt["norm_over_sh_pre"] * SIGMA_PRE
        ra = total_norm / SH_A
        steps.append(pt["step"])
        ra_vals.append(ra)
        acc_a.append(pt["task_a_val_acc"])
        acc_b.append(pt["task_val_acc"])
    final_acc_a = t["seq_task_a_acc_final"]
    return steps, ra_vals, acc_a, acc_b, final_acc_a

def load_final_norm(fname):
    with open(f"{BASE}/{fname}_steps2000_norm.json") as f:
        n = json.load(f)
    return n.get("norm", 0.0)

all_data = {}
for (fname, lr_label, rank, color, marker) in CONDITIONS:
    steps, ra_vals, acc_a, acc_b, final_acc_a = load_trajectory(fname)
    final_norm_file = load_final_norm(fname)
    all_data[fname] = dict(
        lr_label=lr_label, rank=rank, color=color, marker=marker,
        steps=steps, ra_vals=ra_vals, acc_a=acc_a, acc_b=acc_b,
        final_acc_a=final_acc_a,
        final_ra=ra_vals[-1],
    )

# ── Figure 1: Final R_A vs. Task-A forgetting (scatter) ───────────────────────
fig, ax = plt.subplots(figsize=(6, 4.5))

for fname, d in all_data.items():
    delta_a = d["final_acc_a"] - BASE_ACC
    ax.scatter(d["final_ra"], delta_a * 100,
               color=d["color"], marker=d["marker"], s=100, zorder=5,
               edgecolors="k", linewidths=0.6,
               label=f"lr={d['lr_label']}, r={d['rank']}")

ax.axvline(1.0, color="black", lw=1.4, ls="--", zorder=3)
ax.axhline(0.0, color="gray",  lw=0.8, ls=":", zorder=2)
ax.text(1.04, ax.get_ylim()[0] + 1, "R_A = 1\n(budget)", fontsize=8, color="black")

# Shade the "safe zone"
ax.axvspan(0, 1, alpha=0.06, color="#2166ac", zorder=1)
ax.axvspan(1, ax.get_xlim()[1] if ax.get_xlim()[1] > 1 else 3.5, alpha=0.06, color="#d6604d", zorder=1)

ax.set_xlabel(r"$R_A = \|\theta_B - \theta_A\| \,/\, \sigma_{1/2}^A$", fontsize=12)
ax.set_ylabel(r"$\Delta$ Task-A accuracy (pp)", fontsize=12)
ax.set_title("H2: Forgetting budget — Task A accuracy vs. displacement ratio", fontsize=10)
ax.legend(fontsize=7.5, loc="lower left", framealpha=0.9)
ax.set_xlim(0, 3.3)

fig.tight_layout()
path = f"{OUT_DIR}/h2_scatter.png"
fig.savefig(path, dpi=150, bbox_inches="tight")
plt.close()
print(f"Saved: {path}")


# ── Figure 2: Task-A accuracy trajectory vs. R_A ──────────────────────────────
fig, ax = plt.subplots(figsize=(7, 4.5))

for fname, d in all_data.items():
    ax.plot(d["ra_vals"], [a * 100 for a in d["acc_a"]],
            color=d["color"], lw=1.8, alpha=0.9,
            linestyle="--" if d["rank"] == 32 else "-",
            label=f"lr={d['lr_label']}, r={d['rank']}")

ax.axvline(1.0, color="black", lw=1.6, ls="--", zorder=5)
ax.axhline(BASE_ACC * 100, color="gray", lw=0.8, ls=":", zorder=2)
ax.text(1.03, 89.5, "R_A = 1", fontsize=9, color="black", va="top")
ax.text(0.02, 89.5, "Task A baseline", fontsize=8, color="gray", va="top")

ax.set_xlabel(r"$R_A = \|\theta_B - \theta_A\| \,/\, \sigma_{1/2}^A$  (grows as SST-2 training proceeds)",
              fontsize=10)
ax.set_ylabel("Task-A (AGNews) accuracy (%)", fontsize=11)
ax.set_title("H2: Task-A accuracy drops as displacement crosses the certified boundary", fontsize=10)
ax.legend(fontsize=7.5, loc="upper right", framealpha=0.9)
ax.set_xlim(0, 3.3)
ax.set_ylim(50, 92)

fig.tight_layout()
path = f"{OUT_DIR}/h2_trajectory.png"
fig.savefig(path, dpi=150, bbox_inches="tight")
plt.close()
print(f"Saved: {path}")


# ── Figure 3: Task-A accuracy trajectory vs. training step ────────────────────
fig, ax = plt.subplots(figsize=(7, 4.5))

for fname, d in all_data.items():
    ax.plot(d["steps"], [a * 100 for a in d["acc_a"]],
            color=d["color"], lw=1.8, alpha=0.9,
            linestyle="--" if d["rank"] == 32 else "-",
            label=f"lr={d['lr_label']}, r={d['rank']}  (R_A={d['final_ra']:.2f})")

ax.axhline(BASE_ACC * 100, color="gray", lw=0.8, ls=":", zorder=2)
ax.set_xlabel("SST-2 fine-tuning step", fontsize=11)
ax.set_ylabel("Task-A (AGNews) accuracy (%)", fontsize=11)
ax.set_title("H2: Task-A forgetting over SST-2 training (by final R_A)", fontsize=10)
ax.legend(fontsize=7.5, loc="lower left", framealpha=0.9)

fig.tight_layout()
path = f"{OUT_DIR}/h2_steps.png"
fig.savefig(path, dpi=150, bbox_inches="tight")
plt.close()
print(f"Saved: {path}")


# ── Figure 4: Combined 2-panel summary ────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

# Panel A — trajectory vs R_A
ax = axes[0]
for fname, d in all_data.items():
    ax.plot(d["ra_vals"], [a * 100 for a in d["acc_a"]],
            color=d["color"], lw=2.0, alpha=0.9,
            linestyle="--" if d["rank"] == 32 else "-")

ax.axvline(1.0, color="black", lw=1.8, ls="--", zorder=5)
ax.axhline(BASE_ACC * 100, color="gray", lw=0.8, ls=":", zorder=2)
ax.fill_betweenx([50, 92], 0, 1, alpha=0.07, color="#2166ac")
ax.fill_betweenx([50, 92], 1, 3.3, alpha=0.07, color="#d6604d")
ax.text(0.05, 51.5, "Safe zone\n(R_A < 1)", fontsize=8.5, color="#2166ac")
ax.text(1.05, 51.5, "Forgetting zone\n(R_A > 1)", fontsize=8.5, color="#d6604d")
ax.text(1.03, 91.0, "R_A = 1", fontsize=8, color="black")
ax.set_xlabel(r"$R_A = \|\theta_B - \theta_A\| / \sigma_{1/2}^A$", fontsize=11)
ax.set_ylabel("Task-A (AGNews) accuracy (%)", fontsize=11)
ax.set_xlim(0, 3.3)
ax.set_ylim(50, 92)
ax.set_title("(a) Forgetting trajectory vs. displacement ratio", fontsize=10)

# Panel B — final R_A vs delta
ax = axes[1]
legend_handles = []
for fname, d in all_data.items():
    delta_a = (d["final_acc_a"] - BASE_ACC) * 100
    h = ax.scatter(d["final_ra"], delta_a,
                   color=d["color"], marker=d["marker"], s=110, zorder=5,
                   edgecolors="k", linewidths=0.7)
    legend_handles.append(
        Line2D([0], [0], marker=d["marker"], color="w", markerfacecolor=d["color"],
               markersize=8, markeredgecolor="k", markeredgewidth=0.7,
               label=f"lr={d['lr_label']}, r={d['rank']}"))

ax.axvline(1.0, color="black", lw=1.8, ls="--", zorder=3)
ax.axhline(0.0, color="gray",  lw=0.8, ls=":",  zorder=2)
ax.fill_betweenx([-35, 5], 0, 1, alpha=0.07, color="#2166ac")
ax.fill_betweenx([-35, 5], 1, 3.3, alpha=0.07, color="#d6604d")
ax.set_xlabel(r"Final $R_A$", fontsize=11)
ax.set_ylabel(r"$\Delta$ Task-A accuracy (pp)", fontsize=11)
ax.set_title("(b) Final forgetting vs. final displacement ratio", fontsize=10)
ax.legend(handles=legend_handles, fontsize=7.5, loc="lower left", framealpha=0.9)
ax.set_xlim(0, 3.3)
ax.set_ylim(-35, 5)

fig.suptitle(
    r"H2: Forgetting budget — $\|\theta_B - \theta_A\| > \sigma_{1/2}^A$ predicts Task-A catastrophic forgetting",
    fontsize=11, y=1.01)
fig.tight_layout()
path = f"{OUT_DIR}/h2_combined.png"
fig.savefig(path, dpi=150, bbox_inches="tight")
plt.close()
print(f"Saved: {path}")

print("\nDone. All figures in", OUT_DIR)
