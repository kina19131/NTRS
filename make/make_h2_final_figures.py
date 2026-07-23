"""
H2 Forgetting Budget — final comprehensive figures.
Sources:
  GPT-2 forward:   outputs/h2_sequential/gpt2_agnews_to_sst2/
  Medium forward:  outputs/h2_sequential/gpt2med_agnews_to_sst2/
  Reversed GPT-2:  outputs/h2_sequential/gpt2_sst2_to_agnews/
"""
import json, glob, os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

OUT = "figures/h2"
os.makedirs(OUT, exist_ok=True)

# ── helpers ───────────────────────────────────────────────────────────────────
def sh(results, threshold=0.5):
    for i in range(len(results) - 1):
        if results[i]["density"] >= threshold and results[i+1]["density"] < threshold:
            t = (threshold - results[i]["density"]) / (results[i+1]["density"] - results[i]["density"])
            return results[i]["sigma"] + t * (results[i+1]["sigma"] - results[i]["sigma"])
    return results[-1]["sigma"] if results[-1]["density"] >= threshold else 0.0

def load_conditions(base, sh_a, base_acc):
    pts = []
    for nf in sorted(glob.glob(f"{base}/*_norm.json")):
        tag = nf.replace(f"{base}/", "").replace("_norm.json", "")
        n = json.load(open(nf))
        t = json.load(open(f"{base}/{tag}_trajectory.json"))
        norm = n["per_param_norm"]
        ra   = norm / sh_a
        da   = t["seq_task_a_acc_final"] - base_acc
        lr_str = tag.split("_rank")[0].replace("lr","")
        rank   = int(tag.split("_rank")[1].split("_")[0])
        pts.append({"ra": ra, "da": da, "norm": norm, "lr": lr_str, "rank": rank, "tag": tag})
    return sorted(pts, key=lambda x: x["ra"])

# ── load data ─────────────────────────────────────────────────────────────────
# GPT-2 forward
G2_BASE = "outputs/h2_sequential/gpt2_agnews_to_sst2"
p1g = json.load(open(f"{G2_BASE}/phase1_task_a_density.json"))
SH_G2  = sh(p1g["sigma_results"])
BA_G2  = p1g["task_a_acc"]
g2_pts = load_conditions(G2_BASE, SH_G2, BA_G2)

# Medium forward (0620 — all 8 conditions, same Phase 1)
MD_BASE = "outputs/h2_sequential/gpt2med_agnews_to_sst2"
p1m = json.load(open(f"{MD_BASE}/phase1_task_a_density.json"))
SH_MD  = sh(p1m["sigma_results"])
BA_MD  = p1m["task_a_acc"]
md_pts = load_conditions(MD_BASE, SH_MD, BA_MD)

# Medium aggressive trajectory (lr=5e-4 r=32, different Phase1 σ½=0.001625)
AGG_TRAJ = "outputs/h2_sequential/gpt2med_agnews_to_sst2_aggressive/lr5e-04_rank32_steps2000_trajectory.json"
agg_t = json.load(open(AGG_TRAJ))
SH_AGG = agg_t["sigma_half_pre"]
BA_AGG = agg_t["seq_task_a_acc_phase1"]
agg_traj_ra  = [pt["norm_over_sh_pre"] * SH_AGG / SH_MD for pt in agg_t["trajectory"]]
agg_traj_acc = [pt["task_a_val_acc"] for pt in agg_t["trajectory"]]

# Reversed GPT-2 (acc-based)
REV_BASE = "outputs/h2_sequential/gpt2_sst2_to_agnews"
p1r = json.load(open(f"{REV_BASE}/phase1_task_a_density_acc.json"))
SH_REV = sh(p1r["sigma_results"])
BA_REV = p1r["task_a_acc"]
rev_pts = load_conditions(REV_BASE, SH_REV, BA_REV)

# GPT-2 forward trajectory for one key condition (lr=1e-4 r=32, crosses R_A=1)
g2_traj_f  = json.load(open(f"{G2_BASE}/lr1e-04_rank32_steps2000_trajectory.json"))
g2_traj_ra = [pt["norm_over_sh_pre"] * g2_traj_f["sigma_half_pre"] / SH_G2
               for pt in g2_traj_f["trajectory"]]
g2_traj_acc= [pt["task_a_val_acc"] for pt in g2_traj_f["trajectory"]]

md_traj_f   = json.load(open(f"{MD_BASE}/lr2e-04_rank32_steps2000_trajectory.json"))
md_traj_ra  = [pt["norm_over_sh_pre"] * md_traj_f["sigma_half_pre"] / SH_MD
                for pt in md_traj_f["trajectory"]]
md_traj_acc = [pt["task_a_val_acc"] for pt in md_traj_f["trajectory"]]

print(f"GPT-2  σ½_A={SH_G2:.6f}  base_acc={BA_G2:.4f}")
print(f"Medium σ½_A={SH_MD:.6f}  base_acc={BA_MD:.4f}")
print(f"Reversed σ½_A={SH_REV:.6f} (acc-based)  base_acc={BA_REV:.4f}")
print()
for p in g2_pts:  print(f"  GPT-2  R_A={p['ra']:.3f}  ΔA={p['da']*100:+.1f}%  [{p['tag']}]")
print()
for p in md_pts:  print(f"  Medium R_A={p['ra']:.3f}  ΔA={p['da']*100:+.1f}%  [{p['tag']}]")
print()
for p in rev_pts: print(f"  Revers R_A={p['ra']:.4f}  ΔA={p['da']*100:+.1f}%  [{p['tag']}]")

# ── colour helpers ────────────────────────────────────────────────────────────
def pt_color(ra):
    return "#d6604d" if ra > 1.0 else "#2166ac"

def pt_marker(rank):
    return "s" if rank == 32 else "o"

# ═══════════════════════════════════════════════════════════════════════════════
# Fig 1 — Main H2 result: GPT-2 + Medium forward scatter + threshold line
# ═══════════════════════════════════════════════════════════════════════════════
fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))

for ax, (pts, model, sh_a, color_edge) in zip(axes, [
    (g2_pts, "GPT-2",        SH_G2, "#2166ac"),
    (md_pts, "GPT-2-medium", SH_MD, "#b2182b"),
]):
    for p in pts:
        ax.scatter(p["ra"], p["da"]*100,
                   color=pt_color(p["ra"]), marker=pt_marker(p["rank"]),
                   s=150, edgecolors=color_edge, linewidths=1.5, zorder=5)
        label = f"lr={p['lr']} r{p['rank']}"
        offset = (6, 3) if p["ra"] < 4 else (-65, 3)
        ax.annotate(label, (p["ra"], p["da"]*100),
                    textcoords="offset points", xytext=offset, fontsize=7.5)

    ax.axvline(1.0, color="black", lw=2.2, ls="--", label="R_A = 1")
    ax.axhline(0,   color="gray",  lw=0.8, ls=":")
    ax.fill_betweenx([-60, 10], 0, 1,  alpha=0.07, color="#2166ac")
    ax.fill_betweenx([-60, 10], 1, 6,  alpha=0.07, color="#d6604d")
    ax.text(0.05, -55, "Safe\n(R_A < 1)", fontsize=9, color="#2166ac", fontweight="bold")
    ax.text(1.05, -55, "Forgetting\n(R_A > 1)", fontsize=9, color="#d6604d", fontweight="bold")
    ax.set_xlabel(r"$R_A = \|\theta_B - \theta_A\| / \sigma^A_{1/2}$", fontsize=11)
    ax.set_ylabel("Δ Task-A (AGNews) accuracy (pp)", fontsize=11)
    ax.set_title(f"{model}  [σ½_A = {sh_a:.5f}]\n"
                 f"R_A < 1 → safe;  R_A > 1 → forgetting", fontsize=9.5)

# Annotate the key gap-fill points on medium
for p in md_pts:
    if p["lr"] in ["3e-04", "4e-04"]:
        axes[1].scatter(p["ra"], p["da"]*100,
                        color=pt_color(p["ra"]), marker="D",
                        s=180, edgecolors="#67001f", linewidths=2, zorder=6)

axes[0].set_xlim(0, 3.5); axes[0].set_ylim(-40, 10)
axes[1].set_xlim(0, 5.5); axes[1].set_ylim(-60, 10)

legend_els = [
    Line2D([0],[0], marker="o", color="w", markerfacecolor="#2166ac", markersize=9,
           markeredgecolor="black", markeredgewidth=1, label="rank 8"),
    Line2D([0],[0], marker="s", color="w", markerfacecolor="#2166ac", markersize=9,
           markeredgecolor="black", markeredgewidth=1, label="rank 32"),
    Line2D([0],[0], marker="D", color="w", markerfacecolor="#d6604d", markersize=9,
           markeredgecolor="#67001f", markeredgewidth=1.5, label="gap-fill (new)"),
    Line2D([0],[0], color="black", lw=2, ls="--", label="R_A = 1"),
]
axes[1].legend(handles=legend_els, fontsize=8, loc="lower right")
axes[0].legend(handles=legend_els[:2] + [legend_els[3]], fontsize=8, loc="lower right")

fig.suptitle("H2 Forgetting Budget: R_A = 1 is the threshold for catastrophic forgetting\n"
             "Holds consistently across GPT-2 and GPT-2-medium", fontsize=11)
fig.tight_layout()
fig.savefig(f"{OUT}/h2_fig1_forward_scatter.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved: h2_fig1_forward_scatter.png")


# ═══════════════════════════════════════════════════════════════════════════════
# Fig 2 — Combined both models on one axis (H2 cross-model)
# ═══════════════════════════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(9, 6))

for p in g2_pts:
    ax.scatter(p["ra"], p["da"]*100,
               color=pt_color(p["ra"]), marker=pt_marker(p["rank"]),
               s=140, edgecolors="#2166ac", linewidths=1.8, zorder=5)

for p in md_pts:
    ax.scatter(p["ra"], p["da"]*100,
               color=pt_color(p["ra"]), marker=pt_marker(p["rank"]),
               s=140, edgecolors="#b2182b", linewidths=1.8, zorder=5, alpha=0.85)

# Add medium aggressive endpoint
ax.scatter(9.609 * SH_AGG / SH_MD, -65.1,
           color="#67001f", marker="*", s=260, edgecolors="#b2182b", linewidths=1.5, zorder=6)
ax.annotate("med aggressive\nlr=5e-4 r=32", (9.609*SH_AGG/SH_MD, -65.1),
            textcoords="offset points", xytext=(-70, 8), fontsize=7.5, color="#67001f")

ax.axvline(1.0, color="black", lw=2.5, ls="--")
ax.axhline(0, color="gray", lw=0.8, ls=":")
ax.fill_betweenx([-70, 10], 0, 1,   alpha=0.07, color="#2166ac")
ax.fill_betweenx([-70, 10], 1, 5.5, alpha=0.07, color="#d6604d")

legend_els = [
    Line2D([0],[0], marker="o", color="w", markerfacecolor="gray", markersize=10,
           markeredgecolor="#2166ac", markeredgewidth=2, label="GPT-2 (r8)"),
    Line2D([0],[0], marker="s", color="w", markerfacecolor="gray", markersize=10,
           markeredgecolor="#2166ac", markeredgewidth=2, label="GPT-2 (r32)"),
    Line2D([0],[0], marker="o", color="w", markerfacecolor="gray", markersize=10,
           markeredgecolor="#b2182b", markeredgewidth=2, label="GPT-2-medium (r8)"),
    Line2D([0],[0], marker="s", color="w", markerfacecolor="gray", markersize=10,
           markeredgecolor="#b2182b", markeredgewidth=2, label="GPT-2-medium (r32)"),
    Line2D([0],[0], color="#2166ac", lw=0, marker="o",
           markerfacecolor="#2166ac", markersize=9, label="R_A < 1 (safe)"),
    Line2D([0],[0], color="#d6604d", lw=0, marker="o",
           markerfacecolor="#d6604d", markersize=9, label="R_A > 1 (forgetting)"),
    Line2D([0],[0], color="black", lw=2.2, ls="--", label="R_A = 1  (predicted threshold)"),
]
ax.legend(handles=legend_els, fontsize=8.5, loc="lower left")
ax.set_xlabel(r"$R_A = \|\theta_B - \theta_A\| / \sigma^A_{1/2}$", fontsize=12)
ax.set_ylabel("Δ Task-A (AGNews) accuracy (pp)", fontsize=12)
ax.set_title("H2: R_A=1 threshold generalizes across model scale\n"
             "Blue border = GPT-2  |  Red border = GPT-2-medium", fontsize=10)
ax.set_xlim(-0.1, 5.5); ax.set_ylim(-70, 12)
ax.text(0.08, -64, "Safe zone", fontsize=10, color="#2166ac", fontweight="bold")
ax.text(1.08, -64, "Forgetting zone", fontsize=10, color="#d6604d", fontweight="bold")
fig.tight_layout()
fig.savefig(f"{OUT}/h2_fig2_combined_models.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved: h2_fig2_combined_models.png")


# ═══════════════════════════════════════════════════════════════════════════════
# Fig 3 — Trajectory: how forgetting unfolds as R_A grows during training
# ═══════════════════════════════════════════════════════════════════════════════
fig, axes = plt.subplots(1, 2, figsize=(13, 5))

# Panel A: GPT-2 forward key trajectories
ax = axes[0]
for tag, col, ls, lbl in [
    ("lr1e-05_rank32_steps2000", "#92c5de", "-",  "lr=1e-5 r32 (R_A=0.17)"),
    ("lr1e-04_rank8_steps2000",  "#4393c3", "--", "lr=1e-4 r8  (R_A=0.97)"),
    ("lr1e-04_rank32_steps2000", "#d6604d", "-",  "lr=1e-4 r32 (R_A=1.52) ←crosses 1"),
    ("lr2e-04_rank32_steps2000", "#b2182b", "--", "lr=2e-4 r32 (R_A=3.02)"),
]:
    t = json.load(open(f"{G2_BASE}/{tag}_trajectory.json"))
    sh_pre = t["sigma_half_pre"]
    ra_v  = [pt["norm_over_sh_pre"] * sh_pre / SH_G2 for pt in t["trajectory"]]
    acc_v = [pt["task_a_val_acc"] for pt in t["trajectory"]]
    ax.plot(ra_v, [a*100 for a in acc_v], color=col, lw=2, ls=ls, label=lbl)

ax.axvline(1.0, color="black", lw=2, ls="--")
ax.axhline(BA_G2*100, color="gray", lw=0.8, ls=":", label=f"baseline {BA_G2*100:.1f}%")
ax.fill_betweenx([60, 92], 0, 1,   alpha=0.07, color="#2166ac")
ax.fill_betweenx([60, 92], 1, 3.5, alpha=0.07, color="#d6604d")
ax.set_xlabel(r"$R_A$  (grows as Task-B training proceeds)", fontsize=10)
ax.set_ylabel("Task-A (AGNews) accuracy (%)", fontsize=10)
ax.set_title(f"GPT-2 forward — forgetting onset at R_A=1", fontsize=10)
ax.legend(fontsize=8, loc="lower left"); ax.set_xlim(0, 3.3); ax.set_ylim(60, 94)

# Panel B: Medium — two trajectories + aggressive
ax = axes[1]
for tag, col, ls, lbl in [
    ("lr1e-04_rank32_steps2000", "#4393c3", "--", f"lr=1e-4 r32 (R_A=0.77)"),
    ("lr2e-04_rank32_steps2000", "#d6604d", "-",  f"lr=2e-4 r32 (R_A=1.76) ←crosses 1"),
]:
    t = json.load(open(f"{MD_BASE}/{tag}_trajectory.json"))
    sh_pre = t["sigma_half_pre"]
    ra_v  = [pt["norm_over_sh_pre"] * sh_pre / SH_MD for pt in t["trajectory"]]
    acc_v = [pt["task_a_val_acc"] for pt in t["trajectory"]]
    ax.plot(ra_v, [a*100 for a in acc_v], color=col, lw=2, ls=ls, label=lbl)

# Aggressive trajectory (re-scaled to current σ½_A)
ax.plot(agg_traj_ra, [a*100 for a in agg_traj_acc],
        color="#67001f", lw=2.5, label="lr=5e-4 r32 (R_A=9.6, aggressive)")

ax.axvline(1.0, color="black", lw=2, ls="--")
ax.axhline(BA_MD*100, color="gray", lw=0.8, ls=":", label=f"baseline {BA_MD*100:.1f}%")
ax.fill_betweenx([20, 94], 0, 1,   alpha=0.07, color="#2166ac")
ax.fill_betweenx([20, 94], 1, 5.5, alpha=0.07, color="#d6604d")
ax.set_xlabel(r"$R_A$  (grows as Task-B training proceeds)", fontsize=10)
ax.set_ylabel("Task-A (AGNews) accuracy (%)", fontsize=10)
ax.set_title(f"GPT-2-medium forward — same R_A=1 threshold", fontsize=10)
ax.legend(fontsize=8, loc="lower left"); ax.set_xlim(0, 5.5); ax.set_ylim(20, 96)

fig.suptitle("Trajectory: Task-A accuracy during Task-B training — forgetting accelerates at R_A=1",
             fontsize=10)
fig.tight_layout()
fig.savefig(f"{OUT}/h2_fig3_trajectories.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved: h2_fig3_trajectories.png")


# ═══════════════════════════════════════════════════════════════════════════════
# Fig 4 — Reversed experiment: why R_A stays << 1 (SST-2 basin is wide)
# ═══════════════════════════════════════════════════════════════════════════════
fig, axes = plt.subplots(1, 2, figsize=(13, 5))

# Panel A: reversed scatter (all R_A << 1)
ax = axes[0]
for p in rev_pts:
    ax.scatter(p["ra"], p["da"]*100,
               color="#2166ac", marker=pt_marker(p["rank"]),
               s=140, edgecolors="navy", linewidths=1.5, zorder=5)
    ax.annotate(f"lr={p['lr']} r{p['rank']}", (p["ra"], p["da"]*100),
                textcoords="offset points", xytext=(5, 2), fontsize=7.5)

ax.axvline(1.0, color="black", lw=2, ls="--", label="R_A = 1")
ax.axhline(0, color="gray", lw=0.8, ls=":")
ax.fill_betweenx([-12, 5], 0, 0.12, alpha=0.12, color="#2166ac")
ax.text(0.002, -10, "All points here\nR_A << 1\n(safe zone)", fontsize=9,
        color="#2166ac", fontweight="bold")
ax.annotate("Threshold at R_A=1\n(not reachable\nwith these LRs)",
            xy=(1.0, -5), xytext=(0.55, -10),
            arrowprops=dict(arrowstyle="->", color="black"), fontsize=8.5)
ax.set_xlabel(r"$R_A = \|\theta_B - \theta_A\| / \sigma^A_{1/2}$  (acc-based)", fontsize=10)
ax.set_ylabel("Δ Task-A (SST-2) accuracy (pp)", fontsize=10)
ax.set_title(f"GPT-2 Reversed (SST-2→AGNews)\nσ½_A={SH_REV:.5f} (acc-based)  —  all R_A << 1",
             fontsize=9.5)
ax.set_xlim(-0.005, 0.12); ax.set_ylim(-12, 5)
ax.legend(fontsize=9)

# Panel B: σ½_A comparison — why reversed has much larger σ½_A
ax = axes[1]
tasks  = ["AGNews→SST-2\n(GPT-2, NLL)", "AGNews→SST-2\n(Medium, NLL)",
          "SST-2→AGNews\n(GPT-2, acc)"]
sh_vals = [SH_G2, SH_MD, SH_REV]
colors  = ["#2166ac", "#b2182b", "#4d9221"]
bars = ax.bar(tasks, [v*1e3 for v in sh_vals], color=colors,
              edgecolor="black", linewidth=0.8, width=0.5, alpha=0.85)
for bar, v in zip(bars, sh_vals):
    ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.1,
            f"{v:.5f}", ha="center", va="bottom", fontsize=9, fontweight="bold")

ax.set_ylabel("σ½_A  (×10⁻³)", fontsize=11)
ax.set_title("σ½_A comparison: SST-2 basin is 49× wider than AGNews\n"
             "→ impossible to reach R_A=1 with normal LRs in reversed experiment",
             fontsize=9.5)
ax.set_ylim(0, max(sh_vals)*1e3 * 1.3)

# Annotate the ratio
ax.annotate(f"49×\nlarger", xy=(2, SH_REV*1e3*0.6),
            xytext=(1.3, SH_REV*1e3*0.5),
            arrowprops=dict(arrowstyle="->", color="black"), fontsize=10,
            color="#4d9221", fontweight="bold")

fig.suptitle("Reversed Experiment: SST-2→AGNews — σ½_A (acc-based) too large to reach R_A=1\n"
             "Suggests SST-2 basin is much wider than AGNews — consistent with pretraining bias",
             fontsize=10)
fig.tight_layout()
fig.savefig(f"{OUT}/h2_fig4_reversed.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved: h2_fig4_reversed.png")

print(f"\nAll H2 final figures saved to {OUT}/")
