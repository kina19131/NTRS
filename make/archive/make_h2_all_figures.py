"""
H2 Forgetting Budget — all figures, updated with medium aggressive data.
Uses sh_certified (density >= 0.5 absolute, interpolated) — NOT half_max_sigma.
"""
import json, os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

OUT = "outputs_0619_h2/figures"
os.makedirs(OUT, exist_ok=True)

# ── helpers ───────────────────────────────────────────────────────────────────
def sh_certified(results, threshold=0.5):
    sigmas = [r["sigma"] for r in results]
    dens   = [r["density"] for r in results]
    for i in range(len(dens) - 1):
        if dens[i] >= threshold and dens[i + 1] < threshold:
            t = (threshold - dens[i]) / (dens[i + 1] - dens[i])
            return sigmas[i] + t * (sigmas[i + 1] - sigmas[i])
    return sigmas[-1] if dens[-1] >= threshold else 0.0

def load_traj(path, sh_a, base_acc):
    with open(path) as f: t = json.load(f)
    sh_pre = t["sigma_half_pre"]
    pts = t["trajectory"]
    ra_vals  = [pt["norm_over_sh_pre"] * sh_pre / sh_a for pt in pts]
    acc_vals = [pt["task_a_val_acc"] for pt in pts]
    return ra_vals, acc_vals, t["seq_task_a_acc_final"]

# ── constants (NLL-based certified radius, density>=0.5) ─────────────────────
SH_A_GPT2   = 0.000636
BASE_ACC_G2 = 0.873
SH_A_MED    = 0.001625   # from aggressive run Phase 1 (NLL-based)
BASE_ACC_MD = 0.901

GPT2_BASE = "outputs_0619_h2/gpt2/seq_agnews_to_sst2"
MED_BASE  = "outputs_0619_h2/outputs_0619_h2_medium/gpt2-medium/seq_agnews_to_sst2"
AGG_BASE  = "outputs_0619_h2/outputs_0619_h2_medium_aggressive/gpt2-medium/seq_agnews_to_sst2"

FNAMES = ["lr1e-05_rank8_steps2000","lr1e-05_rank32_steps2000",
          "lr1e-04_rank8_steps2000","lr1e-04_rank32_steps2000",
          "lr2e-04_rank8_steps2000","lr2e-04_rank32_steps2000"]
COND_COLORS = ["#92c5de","#4393c3","#f4a582","#d6604d","#b2182b","#67001f"]
MARKERS     = ["o","s","o","s","o","s"]
LABELS      = ["1e-5 r8","1e-5 r32","1e-4 r8","1e-4 r32","2e-4 r8","2e-4 r32"]

# load GPT-2 scatter points
with open(f"{GPT2_BASE}/widening_summary.json") as f: g2_summ = json.load(f)
g2_pts = [{"ra": r["norm"]/SH_A_GPT2,
            "da": load_traj(f"{GPT2_BASE}/{fn}_trajectory.json", SH_A_GPT2, BASE_ACC_G2)[2] - BASE_ACC_G2}
          for r, fn in zip(g2_summ["results"], FNAMES)]

# load medium scatter points
with open(f"{MED_BASE}/widening_summary.json") as f: md_summ = json.load(f)
md_pts = [{"ra": r["norm"]/SH_A_MED,
            "da": load_traj(f"{MED_BASE}/{fn}_trajectory.json", SH_A_MED, BASE_ACC_MD)[2] - BASE_ACC_MD}
          for r, fn in zip(md_summ["results"], FNAMES)]

# load trajectories
g2_trajs = [load_traj(f"{GPT2_BASE}/{fn}_trajectory.json",  SH_A_GPT2, BASE_ACC_G2)[:2]
            for fn in FNAMES]
md_trajs = [load_traj(f"{MED_BASE}/{fn}_trajectory.json",   SH_A_MED,  BASE_ACC_MD)[:2]
            for fn in FNAMES]

# medium aggressive trajectory (lr=5e-4 r=32)
agg_ra, agg_acc, _ = load_traj(f"{AGG_BASE}/lr5e-04_rank32_steps2000_trajectory.json",
                                SH_A_MED, BASE_ACC_MD)


# ═══════════════════════════════════════════════════════════════════════════
# Fig 1 — R_A vs ΔA scatter: GPT-2 + medium, all conditions
# ═══════════════════════════════════════════════════════════════════════════
fig, axes = plt.subplots(1, 2, figsize=(13, 5))

for ax, (model_pts, model_trajs, mname, sh_a, base_acc, col) in zip(axes, [
    (g2_pts, g2_trajs, "GPT-2",        SH_A_GPT2, BASE_ACC_G2, "#2166ac"),
    (md_pts, md_trajs, "GPT-2-medium", SH_A_MED,  BASE_ACC_MD, "#b2182b"),
]):
    for i, (pt, marker) in enumerate(zip(model_pts, MARKERS)):
        ax.scatter(pt["ra"], pt["da"]*100, color=COND_COLORS[i],
                   marker=marker, s=140, edgecolors="k", linewidths=0.8, zorder=5)
        ax.annotate(LABELS[i], (pt["ra"], pt["da"]*100),
                    textcoords="offset points", xytext=(6,2), fontsize=7.5)

    ax.axvline(1.0, color="black", lw=2, ls="--", label="R_A = 1")
    ax.axhline(0,   color="gray",  lw=0.8, ls=":")
    ax.fill_betweenx([-70, 8], 0, 1,   alpha=0.07, color="#2166ac")
    ax.fill_betweenx([-70, 8], 1, 10,  alpha=0.07, color="#d6604d")
    ax.set_xlabel(r"$R_A = \|\theta_B - \theta_A\| / \sigma^A_{1/2}$", fontsize=10)
    ax.set_ylabel("Δ Task-A (AGNews) accuracy (pp)", fontsize=10)
    ax.set_title(f"{mname}  [σ½_A = {sh_a:.5f}]", fontsize=10)
    ax.legend(fontsize=9)

# Add aggressive point on medium panel
axes[1].scatter(9.609, -65.1, color="#67001f", marker="*",
                s=220, edgecolors="k", linewidths=0.8, zorder=6)
axes[1].annotate("lr=5e-4 r=32\n(aggressive)", (9.609, -65.1),
                 textcoords="offset points", xytext=(-70, 8), fontsize=7.5,
                 color="#67001f")
axes[0].set_xlim(0, 3.5); axes[0].set_ylim(-40, 8)
axes[1].set_xlim(0, 11);  axes[1].set_ylim(-75, 8)

axes[0].text(0.05, -37, "Safe", fontsize=9, color="#2166ac")
axes[0].text(1.05, -37, "Forgetting zone", fontsize=9, color="#d6604d")
axes[1].text(0.05, -70, "Safe", fontsize=9, color="#2166ac")
axes[1].text(1.05, -70, "Forgetting zone", fontsize=9, color="#d6604d")

fig.suptitle("H2: R_A=1 threshold holds for both GPT-2 and GPT-2-medium\n"
             "R_A<1 → safe; R_A>1 → significant forgetting", fontsize=11)
fig.tight_layout()
fig.savefig(f"{OUT}/fig1_ra_scatter_both_models.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved: fig1_ra_scatter_both_models.png")


# ═══════════════════════════════════════════════════════════════════════════
# Fig 2 — Trajectory: medium aggressive showing R_A=1 transition
# ═══════════════════════════════════════════════════════════════════════════
fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))

# Panel A: GPT-2 trajectories
ax = axes[0]
for i, (traj, marker) in enumerate(zip(g2_trajs, MARKERS)):
    ra_v, acc_v = traj
    ls = "--" if LABELS[i].endswith("32") else "-"
    ax.plot(ra_v, [a*100 for a in acc_v], color=COND_COLORS[i],
            lw=1.8, ls=ls, alpha=0.9, label=LABELS[i])
ax.axvline(1.0, color="black", lw=2, ls="--")
ax.axhline(BASE_ACC_G2*100, color="gray", lw=0.8, ls=":", label="Phase-1 baseline")
ax.fill_betweenx([50,95], 0, 1,   alpha=0.07, color="#2166ac")
ax.fill_betweenx([50,95], 1, 3.5, alpha=0.07, color="#d6604d")
ax.set_xlabel(r"$R_A$  (grows as Task-B training proceeds)", fontsize=10)
ax.set_ylabel("Task-A (AGNews) accuracy (%)", fontsize=10)
ax.set_title(f"GPT-2  [σ½_A={SH_A_GPT2:.5f}]\nClear drop when R_A crosses 1", fontsize=9)
ax.legend(fontsize=7.5, loc="lower left"); ax.set_xlim(0,3.3); ax.set_ylim(50,95)

# Panel B: medium trajectories + aggressive
ax = axes[1]
for i, (traj, marker) in enumerate(zip(md_trajs, MARKERS)):
    ra_v, acc_v = traj
    ls = "--" if LABELS[i].endswith("32") else "-"
    ax.plot(ra_v, [a*100 for a in acc_v], color=COND_COLORS[i],
            lw=1.6, ls=ls, alpha=0.75, label=LABELS[i])

# Aggressive trajectory (special highlight)
ax.plot(agg_ra, [a*100 for a in agg_acc], color="#3d0000",
        lw=2.5, ls="-", zorder=6, label="lr=5e-4 r=32 (aggressive)")
ax.axvline(1.0, color="black", lw=2, ls="--")
ax.axhline(BASE_ACC_MD*100, color="gray", lw=0.8, ls=":", label="Phase-1 baseline")
ax.fill_betweenx([20,95], 0, 1,   alpha=0.07, color="#2166ac")
ax.fill_betweenx([20,95], 1, 10,  alpha=0.07, color="#d6604d")

# Annotate key events on aggressive trajectory
ax.annotate("R_A≈1 → −13.5%", xy=(1.09, 72.5), xytext=(2.5, 80),
            arrowprops=dict(arrowstyle="->", color="black"), fontsize=8)
ax.annotate("R_A≈2 → −51%\n(catastrophic)", xy=(2.47, 38.8), xytext=(4, 55),
            arrowprops=dict(arrowstyle="->", color="#b2182b"), fontsize=8, color="#b2182b")

ax.set_xlabel(r"$R_A$  (grows as Task-B training proceeds)", fontsize=10)
ax.set_ylabel("Task-A (AGNews) accuracy (%)", fontsize=10)
ax.set_title(f"GPT-2-medium  [σ½_A={SH_A_MED:.5f}]\nAggressive run confirms R_A=1 threshold", fontsize=9)
ax.legend(fontsize=7.5, loc="lower left"); ax.set_xlim(0, 10); ax.set_ylim(20, 98)

fig.suptitle("Trajectory: Task-A accuracy during Task-B training — R_A=1 is the critical boundary for both models",
             fontsize=10)
fig.tight_layout()
fig.savefig(f"{OUT}/fig2_trajectory_comparison.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved: fig2_trajectory_comparison.png")


# ═══════════════════════════════════════════════════════════════════════════
# Fig 3 — Combined scatter: both models on same axes
# ═══════════════════════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(8, 5.5))

for i, (pt, marker) in enumerate(zip(g2_pts, MARKERS)):
    ax.scatter(pt["ra"], pt["da"]*100, color=COND_COLORS[i],
               marker=marker, s=130, edgecolors="#2166ac", linewidths=1.5, zorder=5)

for i, (pt, marker) in enumerate(zip(md_pts, MARKERS)):
    ax.scatter(pt["ra"], pt["da"]*100, color=COND_COLORS[i],
               marker=marker, s=130, edgecolors="#b2182b", linewidths=1.5, zorder=5, alpha=0.75)

# Aggressive point
ax.scatter(9.609, -65.1, color=COND_COLORS[5], marker="*",
           s=250, edgecolors="#b2182b", linewidths=1.5, zorder=6)

ax.axvline(1.0, color="black", lw=2, ls="--", label="R_A = 1")
ax.axhline(0,   color="gray",  lw=0.8, ls=":")
ax.fill_betweenx([-70, 8], 0, 1,  alpha=0.08, color="#2166ac")
ax.fill_betweenx([-70, 8], 1, 11, alpha=0.08, color="#d6604d")

legend_handles = [
    Line2D([0],[0], marker="o", color="w", markerfacecolor="gray", markersize=10,
           markeredgecolor="#2166ac", markeredgewidth=2, label="GPT-2"),
    Line2D([0],[0], marker="o", color="w", markerfacecolor="gray", markersize=10,
           markeredgecolor="#b2182b", markeredgewidth=2, label="GPT-2-medium"),
    Line2D([0],[0], color="black", lw=2, ls="--", label="R_A = 1  (predicted threshold)"),
]
ax.legend(handles=legend_handles, fontsize=9)
ax.set_xlabel(r"$R_A = \|\theta_B - \theta_A\| / \sigma^A_{1/2}$", fontsize=11)
ax.set_ylabel("Δ Task-A (AGNews) accuracy (pp)", fontsize=11)
ax.set_title("H2 Forgetting Budget: R_A=1 threshold across model scales\n"
             "Both GPT-2 and GPT-2-medium show forgetting only when R_A > 1", fontsize=10)
ax.set_xlim(-0.1, 11); ax.set_ylim(-72, 8)
ax.text(0.05, -67, "R_A < 1\n→ safe", fontsize=9.5, color="#2166ac", fontweight="bold")
ax.text(1.1,  -67, "R_A > 1\n→ forgetting", fontsize=9.5, color="#d6604d", fontweight="bold")

fig.tight_layout()
fig.savefig(f"{OUT}/fig3_combined_scatter.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved: fig3_combined_scatter.png")


# ═══════════════════════════════════════════════════════════════════════════
# Fig 4 — Raw norm vs ΔA (shows R_A is the right predictor, not raw norm)
# ═══════════════════════════════════════════════════════════════════════════
fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

# Panel A: raw norm
ax = axes[0]
with open(f"{GPT2_BASE}/widening_summary.json") as f: g2s = json.load(f)
with open(f"{MED_BASE}/widening_summary.json") as f:  mds = json.load(f)
for i, (r, marker) in enumerate(zip(g2s["results"], MARKERS)):
    da = g2_pts[i]["da"]
    ax.scatter(r["norm"]*1e3, da*100, color=COND_COLORS[i],
               marker=marker, s=120, edgecolors="#2166ac", linewidths=1.5, zorder=5)
for i, (r, marker) in enumerate(zip(mds["results"], MARKERS)):
    da = md_pts[i]["da"]
    ax.scatter(r["norm"]*1e3, da*100, color=COND_COLORS[i],
               marker=marker, s=120, edgecolors="#b2182b", linewidths=1.5, zorder=5, alpha=0.75)
ax.axhline(0, color="gray", lw=0.8, ls=":")
ax.annotate("Medium moves\n47% MORE here\nbut forgets 5× LESS",
            xy=(3.34, -18.2), xytext=(1.5, -12),
            arrowprops=dict(arrowstyle="->", color="#b2182b"), fontsize=8, color="#b2182b")
ax.annotate("GPT-2 −31%\nat smaller norm",
            xy=(1.92, -31.1), xytext=(0.3, -28),
            arrowprops=dict(arrowstyle="->", color="#2166ac"), fontsize=8, color="#2166ac")
ax.set_xlabel("||Δθ|| per-param norm (×10⁻³)", fontsize=10)
ax.set_ylabel("Δ Task-A accuracy (pp)", fontsize=10)
ax.set_title("Raw displacement: fails cross-model\n(medium moves more, forgets less)", fontsize=9)
ax.legend(handles=[
    Line2D([0],[0], marker="o", color="w", markerfacecolor="gray", markersize=9,
           markeredgecolor="#2166ac", markeredgewidth=2, label="GPT-2"),
    Line2D([0],[0], marker="o", color="w", markerfacecolor="gray", markersize=9,
           markeredgecolor="#b2182b", markeredgewidth=2, label="GPT-2-medium"),
], fontsize=8.5)

# Panel B: R_A (correct predictor)
ax = axes[1]
for i, (pt, marker) in enumerate(zip(g2_pts, MARKERS)):
    ax.scatter(pt["ra"], pt["da"]*100, color=COND_COLORS[i],
               marker=marker, s=120, edgecolors="#2166ac", linewidths=1.5, zorder=5)
for i, (pt, marker) in enumerate(zip(md_pts, MARKERS)):
    ax.scatter(pt["ra"], pt["da"]*100, color=COND_COLORS[i],
               marker=marker, s=120, edgecolors="#b2182b", linewidths=1.5, zorder=5, alpha=0.75)
ax.axvline(1.0, color="black", lw=2, ls="--")
ax.axhline(0,   color="gray",  lw=0.8, ls=":")
ax.fill_betweenx([-40, 8], 0, 1,   alpha=0.08, color="#2166ac")
ax.fill_betweenx([-40, 8], 1, 3.5, alpha=0.08, color="#d6604d")
ax.set_xlabel(r"$R_A = \|\theta_B - \theta_A\| / \sigma^A_{1/2}$", fontsize=10)
ax.set_ylabel("Δ Task-A accuracy (pp)", fontsize=10)
ax.set_title("R_A: correct predictor across models\nBoth align at R_A=1 threshold", fontsize=9)
ax.set_xlim(0, 3.5); ax.set_ylim(-40, 8)
ax.legend(handles=[
    Line2D([0],[0], marker="o", color="w", markerfacecolor="gray", markersize=9,
           markeredgecolor="#2166ac", markeredgewidth=2, label="GPT-2"),
    Line2D([0],[0], marker="o", color="w", markerfacecolor="gray", markersize=9,
           markeredgecolor="#b2182b", markeredgewidth=2, label="GPT-2-medium"),
    Line2D([0],[0], color="black", lw=2, ls="--", label="R_A = 1"),
], fontsize=8.5)

fig.suptitle("Why R_A (not raw ‖Δθ‖) is the right predictor of catastrophic forgetting",
             fontsize=10)
fig.tight_layout()
fig.savefig(f"{OUT}/fig4_norm_vs_ra.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved: fig4_norm_vs_ra.png")

print(f"\nAll figures saved to {OUT}/")
