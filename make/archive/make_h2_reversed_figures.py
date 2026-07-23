"""
H2 figures — reversed (SST-2→AGNews) + side-by-side comparison with forward.
Run: python3 make_h2_reversed_figures.py
"""
import json, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

SIGMA_PRE = 0.003031250

BASE_FWD = "outputs_0618_h2/gpt2/seq_agnews_to_sst2"
BASE_REV = "outputs_0618_h2/outputs_0618_h2_reversed/gpt2/seq_sst2_to_agnews"
OUT_DIR  = "outputs_0618_h2/figures"
os.makedirs(OUT_DIR, exist_ok=True)

CONDITIONS = [
    ("lr1e-05_rank8",  "1e-5", 8,  "#4393c3", "o"),
    ("lr1e-05_rank32", "1e-5", 32, "#4393c3", "s"),
    ("lr1e-04_rank8",  "1e-4", 8,  "#f4a582", "o"),
    ("lr1e-04_rank32", "1e-4", 32, "#d6604d", "s"),
    ("lr2e-04_rank8",  "2e-4", 8,  "#b2182b", "o"),
    ("lr2e-04_rank32", "2e-4", 32, "#67001f", "s"),
]


def load_sigma_half(phase1_path):
    with open(phase1_path) as f:
        ph = json.load(f)
    sigmas  = [r["sigma"]   for r in ph["sigma_results"]]
    dens    = [r["density"] for r in ph["sigma_results"]]
    for i in range(len(dens) - 1):
        if dens[i] >= 0.5 >= dens[i + 1]:
            t = (0.5 - dens[i]) / (dens[i + 1] - dens[i])
            return sigmas[i] + t * (sigmas[i + 1] - sigmas[i]), ph["task_a_acc"]
    return sigmas[-1], ph["task_a_acc"]


def load_dataset(base, sh_a, base_acc):
    data = {}
    for (fname, lr_label, rank, color, marker) in CONDITIONS:
        with open(f"{base}/{fname}_steps2000_trajectory.json") as f:
            t = json.load(f)
        traj = t["trajectory"]
        ra_vals, acc_a = [], []
        for pt in traj:
            norm = pt["norm_over_sh_pre"] * SIGMA_PRE
            ra_vals.append(norm / sh_a)
            acc_a.append(pt["task_a_val_acc"])
        data[fname] = dict(
            lr_label=lr_label, rank=rank, color=color, marker=marker,
            ra_vals=ra_vals, acc_a=acc_a,
            final_ra=ra_vals[-1],
            final_acc_a=t["seq_task_a_acc_final"],
            delta_a=t["seq_task_a_acc_final"] - base_acc,
        )
    return data


# Load both datasets
sh_fwd, base_fwd = load_sigma_half(f"{BASE_FWD}/phase1_task_a_density.json")
sh_rev, base_rev = load_sigma_half(f"{BASE_REV}/phase1_task_a_density.json")
fwd = load_dataset(BASE_FWD, sh_fwd, base_fwd)
rev = load_dataset(BASE_REV, sh_rev, base_rev)

print(f"Forward  σ½_A={sh_fwd:.6f}  Phase1_acc={base_fwd:.3f}")
print(f"Reversed σ½_A={sh_rev:.6f}  Phase1_acc={base_rev:.3f}")


# ── Figure 1: Reversed trajectory vs R_A ──────────────────────────────────────
fig, ax = plt.subplots(figsize=(7, 4.5))

for fname, d in rev.items():
    ax.plot(d["ra_vals"], [a * 100 for a in d["acc_a"]],
            color=d["color"], lw=1.8, alpha=0.9,
            linestyle="--" if d["rank"] == 32 else "-",
            label=f"lr={d['lr_label']}, r={d['rank']}")

ax.axvline(1.0, color="black", lw=1.6, ls="--", zorder=5, label="R_A = 1 (certified boundary)")
ax.axvline(0.25, color="#666666", lw=1.2, ls=":", zorder=4, label="R_A ≈ 0.25 (empirical threshold)")
ax.axhline(base_rev * 100, color="gray", lw=0.8, ls=":", zorder=2)

ax.fill_betweenx([45, 92], 0, 0.25, alpha=0.10, color="#2166ac")
ax.fill_betweenx([45, 92], 0.25, 1.0, alpha=0.06, color="#f4a582")
ax.fill_betweenx([45, 92], 1.0, 3.2,  alpha=0.07, color="#d6604d")

ax.text(0.02, 47, "No\nforgetting", fontsize=8, color="#2166ac")
ax.text(0.27, 47, "Forgetting\n(R_A<1 !)", fontsize=8, color="#b2182b")
ax.text(1.03, 47, "R_A>1", fontsize=8, color="#d6604d")

ax.set_xlabel(r"$R_A = \|\theta_B - \theta_A\| \,/\, \sigma_{1/2}^A$  (grows as AGNews training proceeds)",
              fontsize=10)
ax.set_ylabel("Task-A (SST-2) accuracy (%)", fontsize=11)
ax.set_title(
    f"Reversed: SST-2→AGNews  "
    r"[$\sigma_{1/2}^A$"
    f"={sh_rev:.4f}, Phase-1 acc={base_rev:.1%}]\n"
    "Forgetting begins at R_A≈0.25 — NLL-based σ½ overestimates safe basin for binary task",
    fontsize=9)
ax.legend(fontsize=7.5, loc="lower left", framealpha=0.9)
ax.set_xlim(0, 3.2)
ax.set_ylim(45, 92)
fig.tight_layout()
fig.savefig(f"{OUT_DIR}/h2_reversed_trajectory.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved: h2_reversed_trajectory.png")


# ── Figure 2: Side-by-side forward vs reversed ─────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))

datasets = [
    (fwd, base_fwd, sh_fwd, "Forward: AGNews → SST-2",
     r"Task-A: AGNews accuracy (%)", 1.0, "#2166ac", "#d6604d",
     "R_A = 1\n(certified boundary)", 50, 92),
    (rev, base_rev, sh_rev, "Reversed: SST-2 → AGNews",
     r"Task-A: SST-2 accuracy (%)", 0.25, "#2166ac", "#d6604d",
     "Empirical\nthreshold ≈ 0.25", 45, 92),
]

for ax, (data, base_acc, sh_a, title, ylabel, thresh, col_safe, col_forg, thresh_label, ylo, yhi) in \
        zip(axes, datasets):

    for fname, d in data.items():
        ax.plot(d["ra_vals"], [a * 100 for a in d["acc_a"]],
                color=d["color"], lw=2.0, alpha=0.85,
                linestyle="--" if d["rank"] == 32 else "-")

    ax.axvline(1.0,    color="black",   lw=1.8, ls="--", zorder=5)
    ax.axvline(thresh, color="#444444", lw=1.3, ls=":",  zorder=4)
    ax.axhline(base_acc * 100, color="gray", lw=0.8, ls=":", zorder=2)

    ax.fill_betweenx([ylo, yhi], 0, thresh, alpha=0.10, color=col_safe)
    ax.fill_betweenx([ylo, yhi], thresh, 3.2, alpha=0.06, color=col_forg)

    ax.text(1.03, ylo + 2, "R_A = 1", fontsize=8, color="black")
    ax.text(thresh + 0.03, yhi - 5, thresh_label, fontsize=7.5, color="#444444")

    ax.set_xlabel(r"$R_A = \|\theta_B - \theta_A\| / \sigma_{1/2}^A$", fontsize=11)
    ax.set_ylabel(ylabel, fontsize=10)
    ax.set_title(title, fontsize=10, fontweight="bold")
    ax.set_xlim(0, 3.2)
    ax.set_ylim(ylo, yhi)

# Shared legend
legend_handles = [
    Line2D([0],[0], color="#4393c3", lw=2, label="lr=1e-5 (circle=r8, sq=r32)"),
    Line2D([0],[0], color="#f4a582", lw=2, label="lr=1e-4"),
    Line2D([0],[0], color="#b2182b", lw=2, label="lr=2e-4"),
    Line2D([0],[0], color="black", lw=1.5, ls="--", label="Certified boundary (R_A=1)"),
    Line2D([0],[0], color="#444444", lw=1.2, ls=":", label="Empirical forgetting onset"),
]
fig.legend(handles=legend_handles, fontsize=8, loc="lower center",
           ncol=5, bbox_to_anchor=(0.5, -0.06), framealpha=0.9)

fig.suptitle(
    "H2 Forgetting Budget: forward vs. reversed task order\n"
    r"Left: σ½ well-calibrated (NLL↔accuracy for 4-class).  "
    r"Right: σ½ overestimates safe region (NLL insensitive for binary task at 76% accuracy).",
    fontsize=9.5, y=1.03)
fig.tight_layout()
fig.savefig(f"{OUT_DIR}/h2_forward_vs_reversed.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved: h2_forward_vs_reversed.png")


# ── Figure 3: Final scatter both datasets overlaid ─────────────────────────────
fig, ax = plt.subplots(figsize=(6.5, 4.5))

for fname, d in fwd.items():
    ax.scatter(d["final_ra"], d["delta_a"] * 100,
               color=d["color"], marker=d["marker"], s=110,
               edgecolors="k", linewidths=0.7, zorder=5)

for fname, d in rev.items():
    ax.scatter(d["final_ra"], d["delta_a"] * 100,
               color=d["color"], marker="D", s=90,
               edgecolors="#555555", linewidths=0.7, zorder=5, alpha=0.75)

ax.axvline(1.0, color="black", lw=1.6, ls="--", zorder=3)
ax.axhline(0.0, color="gray",  lw=0.8, ls=":",  zorder=2)
ax.fill_betweenx([-40, 5], 0, 1, alpha=0.07, color="#2166ac")
ax.fill_betweenx([-40, 5], 1, 3.3, alpha=0.07, color="#d6604d")

# Custom legend
legend_handles = [
    Line2D([0],[0], marker="o", color="w", markerfacecolor="#4393c3", markersize=8,
           markeredgecolor="k", label="Forward (AGNews→SST-2)  lr=1e-5"),
    Line2D([0],[0], marker="o", color="w", markerfacecolor="#d6604d", markersize=8,
           markeredgecolor="k", label="Forward  lr=1e-4/2e-4"),
    Line2D([0],[0], marker="D", color="w", markerfacecolor="#4393c3", markersize=8,
           markeredgecolor="#555", label="Reversed (SST-2→AGNews)  lr=1e-5"),
    Line2D([0],[0], marker="D", color="w", markerfacecolor="#d6604d", markersize=8,
           markeredgecolor="#555", label="Reversed  lr=1e-4/2e-4"),
    Line2D([0],[0], color="black", lw=1.5, ls="--", label="R_A = 1"),
]
ax.legend(handles=legend_handles, fontsize=7.5, loc="lower left", framealpha=0.9)
ax.set_xlabel(r"Final $R_A = \|\theta_B - \theta_A\| / \sigma_{1/2}^A$", fontsize=11)
ax.set_ylabel(r"$\Delta$ Task-A accuracy (pp)", fontsize=11)
ax.set_title(
    "Forward: forgetting concentrated at R_A > 1\n"
    "Reversed: forgetting starts at R_A < 1 (σ½ miscalibration for binary task)",
    fontsize=9)
ax.set_xlim(0, 3.3)
ax.set_ylim(-40, 5)
fig.tight_layout()
fig.savefig(f"{OUT_DIR}/h2_scatter_both.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved: h2_scatter_both.png")

print(f"\nAll figures saved to {OUT_DIR}/")
