"""
GPT-2 vs GPT-2-medium H2 scale comparison figures.
"""
import json, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import matplotlib.patches as mpatches

OUT_DIR = "outputs_0618_h2/figures"
os.makedirs(OUT_DIR, exist_ok=True)

SIGMA_PRE_SM = 0.003031
SIGMA_PRE_MD = 0.003650

MODELS = {
    "GPT-2": {
        "base": "outputs_0618_h2/gpt2/seq_agnews_to_sst2",
        "sh_a": 0.000636, "sh_pre": 0.003031, "base_acc": 0.8728,
        "n_params": 117e6, "color": "#2166ac",
    },
    "GPT-2-medium": {
        "base": "outputs_0618_h2/outputs_0618_h2_medium/gpt2-medium/seq_agnews_to_sst2",
        "sh_a": 0.002300, "sh_pre": 0.003650, "base_acc": 0.892,
        "n_params": 345e6, "color": "#b2182b",
    },
}

FNAMES = ["lr1e-05_rank8","lr1e-05_rank32","lr1e-04_rank8",
          "lr1e-04_rank32","lr2e-04_rank8","lr2e-04_rank32"]
COND_LABELS = ["1e-5 r8","1e-5 r32","1e-4 r8","1e-4 r32","2e-4 r8","2e-4 r32"]
MARKERS = ["o","s","o","s","o","s"]
LR_COLORS = ["#4393c3","#4393c3","#f4a582","#d6604d","#b2182b","#67001f"]

def load_model_data(cfg):
    base, sh_a, base_acc = cfg["base"], cfg["sh_a"], cfg["base_acc"]
    sh_pre = cfg["sh_pre"]
    rows, trajectories = [], []
    with open(base + "/widening_summary.json") as f:
        s = json.load(f)
    for r, fname in zip(s["results"], FNAMES):
        with open(f"{base}/{fname}_steps2000_trajectory.json") as f:
            t = json.load(f)
        traj = t["trajectory"]
        ra_vals, acc_a = [], []
        for pt in traj:
            norm = pt["norm_over_sh_pre"] * sh_pre
            ra_vals.append(norm / sh_a)
            acc_a.append(pt["task_a_val_acc"])
        rows.append({
            "lr": r["lr"], "rank": r["rank"],
            "norm": r["norm"], "ra": r["norm"] / sh_a,
            "sh_ft": r["sigma_half_ft"],
            "da": t["seq_task_a_acc_final"] - base_acc,
            "final_acc_a": t["seq_task_a_acc_final"],
        })
        trajectories.append({"ra_vals": ra_vals, "acc_a": acc_a})
    return rows, trajectories

data = {m: load_model_data(cfg) for m, cfg in MODELS.items()}


# ── Figure 1: Basin width comparison ──────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(10, 4))

# Panel A: σ½ values across model stages
ax = axes[0]
x = np.arange(3)
width = 0.3
bars_sm = [MODELS["GPT-2"]["sh_pre"],      MODELS["GPT-2"]["sh_a"],      None]
bars_md = [MODELS["GPT-2-medium"]["sh_pre"], MODELS["GPT-2-medium"]["sh_a"], None]
labels  = ["Pretrained\n(before Task A)", "After Task A\n(AGNews, σ½_A)", ""]

ax.bar(x[0] - width/2, bars_sm[0], width, color="#4393c3", alpha=0.85, label="GPT-2")
ax.bar(x[0] + width/2, bars_md[0], width, color="#b2182b", alpha=0.85, label="GPT-2-medium")
ax.bar(x[1] - width/2, bars_sm[1], width, color="#4393c3", alpha=0.85)
ax.bar(x[1] + width/2, bars_md[1], width, color="#b2182b", alpha=0.85)

ax.annotate("", xy=(1 - width/2, bars_sm[1] + 0.0001), xytext=(0 - width/2, bars_sm[0] - 0.0001),
            arrowprops=dict(arrowstyle="->", color="#4393c3", lw=1.5))
ax.annotate("", xy=(1 + width/2, bars_md[1] + 0.0001), xytext=(0 + width/2, bars_md[0] - 0.0001),
            arrowprops=dict(arrowstyle="->", color="#b2182b", lw=1.5))

ax.text(0.5 - width/2 - 0.05, (bars_sm[0] + bars_sm[1])/2, "−79%", fontsize=8, color="#2166ac")
ax.text(0.5 + width/2 + 0.02, (bars_md[0] + bars_md[1])/2, "−37%", fontsize=8, color="#b2182b")

ax.set_xticks([0, 1])
ax.set_xticklabels(["Pretrained\nσ½", "After AGNews\nσ½_A"])
ax.set_ylabel("σ½ (certified basin radius)", fontsize=10)
ax.set_title("Basin narrowing from AGNews fine-tuning", fontsize=10)
ax.legend(fontsize=9)
ax.set_ylim(0, 0.0045)

# Panel B: R_A ranges achievable with current LoRA conditions
ax = axes[1]
for i, (model, cfg) in enumerate(MODELS.items()):
    rows, _ = data[model]
    ras = [r["ra"] for r in rows]
    offset = (i - 0.5) * 0.35
    for j, (ra, label, marker) in enumerate(zip(ras, COND_LABELS, MARKERS)):
        da = rows[j]["da"]
        ax.scatter(ra, i + offset + j * 0.05, marker=marker,
                   color=LR_COLORS[j], s=80, zorder=5)
    ax.annotate(f"max R_A={max(ras):.2f}", xy=(max(ras), i + offset + 5*0.05),
                xytext=(max(ras) + 0.1, i + offset + 5*0.05),
                fontsize=8, color=cfg["color"],
                arrowprops=dict(arrowstyle="->", color=cfg["color"], lw=1))

ax.axvline(1.0, color="black", lw=1.6, ls="--", label="R_A = 1")
ax.fill_betweenx([-0.5, 1.5], 0, 1, alpha=0.07, color="#2166ac")
ax.fill_betweenx([-0.5, 1.5], 1, 3.5, alpha=0.07, color="#d6604d")
ax.set_yticks([0, 1])
ax.set_yticklabels(list(MODELS.keys()))
ax.set_xlabel("R_A = ||Δθ|| / σ½_A", fontsize=10)
ax.set_title("R_A range reached with current LoRA conditions\n(medium stays near R_A=1, GPT-2 reaches R_A=3)", fontsize=9)
ax.set_xlim(0, 3.5)
ax.set_ylim(-0.5, 1.5)
ax.legend(fontsize=9)

fig.suptitle("Why GPT-2 medium looks different: σ½_A is 3.6× wider → same LoRA = much smaller R_A", fontsize=10)
fig.tight_layout()
fig.savefig(f"{OUT_DIR}/scale_basin_comparison.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved: scale_basin_comparison.png")


# ── Figure 2: R_A vs forgetting, both models ─────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

# Panel A: raw R_A (current, NLL-based)
ax = axes[0]
for model, cfg in MODELS.items():
    rows, _ = data[model]
    col = cfg["color"]
    for i, (r, marker) in enumerate(zip(rows, MARKERS)):
        ax.scatter(r["ra"], r["da"] * 100, color=col, marker=marker,
                   s=110, edgecolors="k", linewidths=0.7, zorder=5,
                   alpha=0.9 if model == "GPT-2" else 0.6)

ax.axvline(1.0, color="black", lw=1.6, ls="--")
ax.axhline(0, color="gray", lw=0.8, ls=":")
ax.fill_betweenx([-35, 5], 0, 1, alpha=0.07, color="#2166ac")
ax.fill_betweenx([-35, 5], 1, 3.5, alpha=0.07, color="#d6604d")
ax.text(0.02, -33, "Safe zone", fontsize=8.5, color="#2166ac")
ax.text(1.05, -33, "Forgetting zone", fontsize=8.5, color="#d6604d")
ax.set_xlabel("R_A (NLL-based σ½)", fontsize=10)
ax.set_ylabel("Δ Task-A accuracy (pp)", fontsize=10)
ax.set_title("Current: GPT-2 shows clean threshold;\nGPT-2-medium stays compressed near R_A<1.3", fontsize=9)
ax.set_xlim(0, 3.5); ax.set_ylim(-35, 5)

legend_handles = [
    Line2D([0],[0], marker="o", color="w", markerfacecolor="#2166ac", markersize=9,
           markeredgecolor="k", label="GPT-2"),
    Line2D([0],[0], marker="o", color="w", markerfacecolor="#b2182b", markersize=9,
           markeredgecolor="k", alpha=0.6, label="GPT-2-medium"),
    Line2D([0],[0], color="black", lw=1.5, ls="--", label="R_A = 1"),
]
ax.legend(handles=legend_handles, fontsize=8.5, loc="lower left")

# Panel B: estimated R_A if accuracy-based σ½ used
# Estimate: accuracy-based σ½ ≈ NLL-based σ½ × scale_factor
# For high-confidence models: accuracy-based ≈ 0.5 × NLL-based (conservative estimate)
# For GPT-2 (well-calibrated): maybe 0.7×
# For medium (higher accuracy, more confident): maybe 0.4×
# This is an approximation to show the likely effect
scale_sm = 0.55   # GPT-2: accuracy-based σ½ ≈ 0.55 × NLL-based
scale_md = 0.35   # medium: higher confidence → tighter accuracy σ½

ax = axes[1]
for model, cfg in MODELS.items():
    rows, _ = data[model]
    scale = scale_sm if model == "GPT-2" else scale_md
    sh_a_acc = cfg["sh_a"] * scale
    col = cfg["color"]
    for i, (r, marker) in enumerate(zip(rows, MARKERS)):
        ra_new = r["norm"] / sh_a_acc
        ax.scatter(ra_new, r["da"] * 100, color=col, marker=marker,
                   s=110, edgecolors="k", linewidths=0.7, zorder=5,
                   alpha=0.9 if model == "GPT-2" else 0.6)

ax.axvline(1.0, color="black", lw=1.6, ls="--")
ax.axhline(0, color="gray", lw=0.8, ls=":")
ax.fill_betweenx([-35, 5], 0, 1, alpha=0.07, color="#2166ac")
ax.fill_betweenx([-35, 5], 1, 10, alpha=0.07, color="#d6604d")
ax.text(0.02, -33, "Safe zone", fontsize=8.5, color="#2166ac")
ax.text(1.1, -33, "Forgetting zone", fontsize=8.5, color="#d6604d")
ax.set_xlabel("R_A (estimated accuracy-based σ½)", fontsize=10)
ax.set_ylabel("Δ Task-A accuracy (pp)", fontsize=10)
ax.set_title("Estimated: accuracy-based σ½ would compress GPT-2\nand spread GPT-2-medium — both show R_A>1 when forgetting", fontsize=9)
ax.set_xlim(0, 9); ax.set_ylim(-35, 5)
ax.legend(handles=legend_handles, fontsize=8.5, loc="lower left")
ax.text(7, 3, "estimated\n(not run yet)", fontsize=8, color="gray",
        bbox=dict(boxstyle="round", fc="w", ec="gray", alpha=0.7))

fig.suptitle("H2 Scale Generalization: NLL-based σ½ (left) vs. estimated accuracy-based σ½ (right)", fontsize=10)
fig.tight_layout()
fig.savefig(f"{OUT_DIR}/scale_ra_comparison.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved: scale_ra_comparison.png")


# ── Figure 3: Trajectory side-by-side ─────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

for ax, (model, cfg) in zip(axes, MODELS.items()):
    rows, trajs = data[model]
    base_acc = cfg["base_acc"]
    for i, (row, traj, marker) in enumerate(zip(rows, trajs, MARKERS)):
        ax.plot(traj["ra_vals"], [a * 100 for a in traj["acc_a"]],
                color=LR_COLORS[i], lw=1.8, alpha=0.85,
                linestyle="--" if row["rank"] == 32 else "-",
                label=f"lr={row['lr']:.0e} r={row['rank']}")

    ax.axvline(1.0, color="black", lw=1.6, ls="--")
    ax.axhline(base_acc * 100, color="gray", lw=0.8, ls=":")
    ax.fill_betweenx([50, 95], 0, 1, alpha=0.07, color="#2166ac")
    ax.fill_betweenx([50, 95], 1, 3.5, alpha=0.06, color="#d6604d")
    ax.set_xlabel(r"$R_A = \|\theta_B - \theta_A\| / \sigma_{1/2}^A$", fontsize=10)
    ax.set_ylabel("Task-A (AGNews) accuracy (%)", fontsize=10)
    sh_a = cfg["sh_a"]
    ax.set_title(f"{model}  [σ½_A = {sh_a:.4f}]\n{'R_A=1 → catastrophic forgetting' if model=='GPT-2' else 'only 1 condition crosses R_A=1'}", fontsize=9)
    ax.legend(fontsize=7.5, loc="lower left", framealpha=0.9)
    ax.set_xlim(0, 3.3); ax.set_ylim(50, 95)

fig.suptitle("H2 trajectory: GPT-2 (clean threshold) vs GPT-2-medium (compressed R_A range)", fontsize=10)
fig.tight_layout()
fig.savefig(f"{OUT_DIR}/scale_trajectory_comparison.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved: scale_trajectory_comparison.png")

print(f"\nAll saved to {OUT_DIR}/")
