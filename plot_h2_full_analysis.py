"""
Full H2 analysis from trajectory + endpoint data.
Reads h2_llama/mnli_to_sst2/summary.json + individual trajectory JSONs.
"""
import json, os, glob
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

data_dir = "h2_llama/mnli_to_sst2"

# ── Load all trajectory files ─────────────────────────────────────────────────
traj_files = sorted(glob.glob(os.path.join(data_dir, "lr*_trajectory.json")))
conditions = []
for f in traj_files:
    with open(f) as fh:
        traj = json.load(fh)
    tag = os.path.basename(f).replace("_trajectory.json","")
    lr_str, rank_str = tag.split("_rank")
    lr   = float(lr_str.replace("lr",""))
    rank = int(rank_str)
    conditions.append({"lr": lr, "rank": rank, "traj": traj, "tag": tag})

# ── Load summary (endpoint data) ──────────────────────────────────────────────
with open(os.path.join(data_dir, "summary.json")) as f:
    summary = json.load(f)

# ── Load acc-based σ½ ─────────────────────────────────────────────────────────
with open(os.path.join(data_dir, "phase1_sigma_half_acc.json")) as f:
    sh_acc = json.load(f)
sigma_half_acc = sh_acc["sigma_half"]

# Recompute R_A using acc-based sigma in summary
for s in summary:
    s["R_A_acc"] = s["final_per_param"] / sigma_half_acc

# Sort conditions
lr_order   = sorted(set(c["lr"]   for c in conditions))
rank_order = [8, 32]
colors_lr  = {1e-5:"#1a237e", 5e-5:"#1565c0", 1e-4:"#2e7d32",
              2e-4:"#f57f17", 5e-4:"#b71c1c"}
ls_rank    = {8:"-", 32:"--"}

phase1_acc = 0.884

# ── Figure layout: 3 panels ───────────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(18, 6))

# ── Panel 1: MNLI Accuracy trajectory over training steps ────────────────────
ax = axes[0]
for c in conditions:
    traj  = c["traj"]
    steps = [t["step"] for t in traj]
    acc_a = [t["acc_task_a"] for t in traj]
    ax.plot(steps, acc_a, color=colors_lr[c["lr"]], ls=ls_rank[c["rank"]],
            lw=1.8, alpha=0.85)
ax.axhline(y=phase1_acc, color="gray", ls=":", lw=1.2, alpha=0.7)
ax.axhline(y=0.85,       color="orange", ls=":", lw=1.2, alpha=0.7,
           label="85% onset threshold")
ax.axhline(y=0.40,       color="red",    ls=":", lw=1.2, alpha=0.6,
           label="40% catastrophic threshold")
ax.text(1850, phase1_acc+0.002, "Phase 1 (88.4%)", fontsize=8, color="gray", ha="right")

legend_els = []
for lr in lr_order:
    legend_els.append(Line2D([0],[0], color=colors_lr[lr], lw=2,
                              label=f"lr={lr:.0e}"))
legend_els.append(Line2D([0],[0], color="gray", lw=2, ls="-",  label="rank=8"))
legend_els.append(Line2D([0],[0], color="gray", lw=2, ls="--", label="rank=32"))
ax.legend(handles=legend_els, fontsize=8, loc="lower left", ncol=2)
ax.set_xlabel("Training Step (Phase 2: SST-2)", fontsize=11)
ax.set_ylabel("MNLI Accuracy (Task A retention)", fontsize=11)
ax.set_title("Task A forgetting during Phase 2 training", fontsize=11, pad=8)
ax.set_ylim(0.30, 0.92)
ax.grid(True, alpha=0.25)

# ── Panel 2: R_A trajectory — when does forgetting trigger? ──────────────────
ax = axes[1]
for c in conditions:
    traj   = c["traj"]
    r_vals = [t["per_param_norm"] / sigma_half_acc for t in traj]
    acc_a  = [t["acc_task_a"] for t in traj]
    ax.plot(r_vals, acc_a, color=colors_lr[c["lr"]], ls=ls_rank[c["rank"]],
            lw=1.8, alpha=0.85)
    # Mark the starting point (step 0)
    ax.scatter(r_vals[0], acc_a[0], color=colors_lr[c["lr"]],
               s=40, zorder=5, marker="o")

ax.axhline(y=phase1_acc, color="gray", ls=":", lw=1.2, alpha=0.7)
ax.axhline(y=0.85, color="orange", ls=":", lw=1.2, alpha=0.7)
ax.axhline(y=0.40, color="red",    ls=":", lw=1.2, alpha=0.6)

# Shade safe zone
ax.axvspan(0, 0.08, color="#e8f5e9", alpha=0.35, zorder=0)
ax.axvspan(0.08, 0.70, color="#ffebee", alpha=0.25, zorder=0)

# Mark empirical catastrophic threshold
ax.axvline(x=0.22, color="#b71c1c", ls="--", lw=1.4, alpha=0.8)
ax.text(0.23, 0.33, "R_A ≈ 0.22\ncatastrophic\nthreshold", fontsize=7.5,
        color="#b71c1c", va="bottom")

ax.set_xlabel("R_A (acc)  =  ‖Δθ‖ / σ½_A_acc  (during training)", fontsize=11)
ax.set_ylabel("MNLI Accuracy (Task A)", fontsize=11)
ax.set_title("R_A trajectory: path through forgetting budget\n"
             "Points = step 0 (training start)", fontsize=10, pad=8)
ax.set_xlim(-0.01, 0.70)
ax.set_ylim(0.30, 0.92)
ax.grid(True, alpha=0.25)
ax.legend(handles=legend_els, fontsize=7.5, loc="lower right", ncol=2)

# ── Panel 3: Endpoint scatter + σ½ density curve ─────────────────────────────
ax = axes[2]
ax2_twin = ax.twinx()

# σ½ density curve (acc-based)
sigmas  = [float(s) for s in sh_acc["densities"].keys()]
dens    = [sh_acc["densities"][s] for s in sh_acc["densities"].keys()]
ax2_twin.plot(sigmas, dens, "k-o", lw=1.5, ms=5, alpha=0.5, label="density curve")
ax2_twin.axhline(y=0.5, color="k", ls=":", lw=1, alpha=0.4)
ax2_twin.axvline(x=sigma_half_acc, color="k", ls=":", lw=1, alpha=0.4)
ax2_twin.set_ylabel("σ½ density (acc-based)", fontsize=9, color="gray")
ax2_twin.set_ylim(-0.05, 1.05)
ax2_twin.tick_params(axis='y', labelcolor='gray')

# Endpoint scatter
for s in summary:
    lr   = s["lr"]
    rank = s["rank"]
    r    = s["R_A_acc"]
    acc  = s["final_acc_task_a"]
    pnorm = s["final_per_param"]
    marker = "o" if rank == 8 else "s"
    ax.scatter(pnorm, acc, color=colors_lr[lr], s=160, marker=marker,
               edgecolors="white", lw=0.8, zorder=10)

ax.axhline(y=phase1_acc, color="gray", ls=":", lw=1.2, alpha=0.7)
ax.axvline(x=sigma_half_acc, color="k", ls="--", lw=1.3, alpha=0.6)
ax.text(sigma_half_acc+0.00005, 0.32, f"σ½_acc={sigma_half_acc:.4f}",
        fontsize=8, color="gray", rotation=90, va="bottom")

# Threshold band
ax.axvspan(0.002114, 0.004229, color="yellow", alpha=0.2, label="σ½ transition zone")
ax.text(0.0022, 0.34, "transition\nzone", fontsize=7.5, color="goldenrod",
        ha="left", va="bottom")

ax.set_xlabel("per-param ‖Δθ‖  (weight displacement, RMS)", fontsize=10)
ax.set_ylabel("MNLI Accuracy (Task A final)", fontsize=10)
ax.set_title("Endpoint: weight displacement vs forgetting\n"
             "σ½ transition zone shown (density: 0.95 → 0.0)", fontsize=10, pad=8)
ax.set_xlim(-0.0001, 0.0023)
ax.set_ylim(0.30, 0.92)
ax.grid(True, alpha=0.25)

legend_scatter = [
    Line2D([0],[0], marker="o", color="w", markerfacecolor="gray", ms=10, label="rank=8 (●)"),
    Line2D([0],[0], marker="s", color="w", markerfacecolor="gray", ms=10, label="rank=32 (■)"),
]
for lr in lr_order:
    legend_scatter.append(Line2D([0],[0], color=colors_lr[lr], lw=2, label=f"lr={lr:.0e}"))
ax.legend(handles=legend_scatter, fontsize=7.5, loc="lower right")

plt.suptitle(
    "H2 full analysis — MNLI→SST-2, Llama-3.2-3B LoRA\n"
    "σ½_A_acc = 0.003120  |  Phase 1 MNLI = 88.4%",
    fontsize=11, y=1.01
)
plt.tight_layout()
out = "h2_full_analysis.png"
plt.savefig(out, dpi=180, bbox_inches="tight")
plt.close()
print(f"Saved: {out}")

# ── Print detailed analysis table ─────────────────────────────────────────────
print("\n" + "─"*90)
print(f"{'LR':>8}  {'Rank':>5}  {'per_param':>11}  {'R_A_acc':>9}  "
      f"{'Acc_A':>7}  {'Acc_B':>7}  {'outcome'}")
print("─"*90)
for s in sorted(summary, key=lambda x: (x["final_per_param"])):
    outcome = ("catastrophic" if s["final_acc_task_a"] < 0.45
               else "significant" if s["final_acc_task_a"] < 0.80
               else "mild" if s["final_acc_task_a"] < 0.87
               else "safe")
    print(f"  {s['lr']:>8.0e}  {s['rank']:>5}  "
          f"{s['final_per_param']:>11.4e}  {s['R_A_acc']:>9.4f}  "
          f"{s['final_acc_task_a']:>7.4f}  {s['final_acc_task_b']:>7.4f}  "
          f"{outcome}")
print("─"*90)

# Key findings
print("\n── Key findings ──")
print(f"  σ½_A_acc = {sigma_half_acc:.6f}")
print(f"  Transition zone: σ = 0.002114 (density=0.953) → 0.004229 (density=0.0)")
safe_max = max(s["final_per_param"] for s in summary if s["final_acc_task_a"] > 0.87)
catastrophic_min = min(s["final_per_param"] for s in summary if s["final_acc_task_a"] < 0.45)
print(f"  Max per_param_norm while SAFE (acc>87%):       {safe_max:.4e}")
print(f"  Min per_param_norm at CATASTROPHIC (acc<45%):  {catastrophic_min:.4e}")
print(f"  Ratio: {catastrophic_min/safe_max:.1f}x more displacement needed for catastrophe")
print(f"  σ½_lower (density=0.953) / catastrophic threshold = {0.002114/catastrophic_min:.1f}x")
print(f"  → Gradient-directed fine-tuning ~{0.002114/catastrophic_min:.0f}x more destructive than isotropic noise")
