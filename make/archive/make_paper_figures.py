"""
make_paper_figures.py
=====================
Five paper-ready figures from outputs_0615_v2.

  fig1_phase_transition.png  — Trajectory: norm/σ½_pre vs task accuracy + WikiText-2 PPL
  fig2_goldilocks.png        — LR sweep scatter: all models/tasks, boundary at norm/σ½=1
  fig3_rank_sweep.png        — Rank sweep: r/d vs norm/σ½_pre and forgetting (2-panel)
  fig4_density_scale.png     — Pretrained density curves: GPT-2 vs Llama-3B
  fig5_widening_summary.png  — σ½ widening (filtered to valid conditions)

Run: python make_paper_figures.py
"""

import json, os, math
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D

# ── Paths ─────────────────────────────────────────────────────────────────────
ROOT   = os.path.dirname(os.path.abspath(__file__))
V2     = os.path.join(ROOT, "outputs_0615_v2")
OUTDIR = os.path.join(ROOT, "results", "paper_figures")
os.makedirs(OUTDIR, exist_ok=True)

LLAMA_HIDDEN = 3072   # Llama-3.2-3B hidden dim

# ── Style ──────────────────────────────────────────────────────────────────────
plt.rcParams.update({
    "figure.facecolor":  "white",
    "axes.facecolor":    "white",
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "axes.grid":         True,
    "grid.alpha":        0.22,
    "grid.linestyle":    "--",
    "font.family":       "sans-serif",
    "font.size":         11,
    "axes.titlesize":    12,
    "axes.labelsize":    11,
    "legend.fontsize":   9,
    "legend.framealpha": 0.92,
})

C_GPT2  = "#2563EB"
C_LLAMA = "#D97706"
C_BOUND = "#DC2626"
C_PPL   = "#7C3AED"
C_GREEN = "#16A34A"


# ── Helpers ───────────────────────────────────────────────────────────────────
def jload(path):
    with open(path) as f:
        return json.load(f)

def half_max_sigma(sr):
    dens   = [r["density"] for r in sr]
    sigmas = [r["sigma"]   for r in sr]
    target = max(dens) / 2.0
    for i in range(len(dens) - 1):
        if dens[i] >= target > dens[i+1]:
            t = (target - dens[i]) / (dens[i+1] - dens[i])
            return sigmas[i] + t * (sigmas[i+1] - sigmas[i])
    return sigmas[-1]

def _save(fig, name):
    p = os.path.join(OUTDIR, name)
    fig.savefig(p, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {p}")


# ══════════════════════════════════════════════════════════════════════════════
# FIG 1 — Phase Transition Trajectory
# GPT-2 SST-2, lr=2e-4, rank=8, 2000 steps
# ══════════════════════════════════════════════════════════════════════════════
def fig1_phase_transition():
    traj_path = os.path.join(V2, "basin_widening_traj_v2", "gpt2", "sst2",
                             "rank8_steps2000_trajectory.json")
    dens_path = os.path.join(V2, "basin_widening_traj_v2", "gpt2", "sst2",
                             "pretrained_density.json")
    d    = jload(traj_path)
    traj = d["trajectory"]
    pretrained_ppl = math.exp(jload(dens_path)["sigma_results"][0]["base_ppl"])
    sh_pre = d["sigma_half_pre"]

    x_norm = [t["norm_over_sh_pre"] for t in traj]
    y_acc  = [t["task_val_acc"] * 100 for t in traj]
    y_ppl  = [t["wikitext2_ppl"]      for t in traj]
    steps  = [t["step"]               for t in traj]

    fig, ax1 = plt.subplots(figsize=(8, 5))
    ax2 = ax1.twinx()
    ax2.spines["right"].set_visible(True)

    # Safe zone
    ax1.axvspan(0, 1.0, alpha=0.09, color=C_GREEN, zorder=0)
    ax1.axvline(1.0, color=C_BOUND, lw=1.8, ls="--", zorder=5,
                label="norm/σ½ = 1  (certified boundary)")

    # Accuracy (left)
    ax1.plot(x_norm, y_acc, color=C_GPT2, lw=2.5, marker="o", ms=5.5, zorder=6,
             label="SST-2 accuracy (%)")
    ax1.set_ylim(40, 102)
    ax1.set_ylabel("SST-2 Accuracy (%)", color=C_GPT2)
    ax1.tick_params(axis="y", labelcolor=C_GPT2)

    # WikiText-2 PPL (right)
    ax2.plot(x_norm, y_ppl, color=C_PPL, lw=2.0, marker="s", ms=4.5, ls="--",
             zorder=6, label="WikiText-2 PPL")
    ax2.axhline(pretrained_ppl, color=C_PPL, lw=1.0, ls=":", alpha=0.5,
                label=f"Pretrained PPL = {pretrained_ppl:.0f}")
    ax2.set_ylim(55, 100)
    ax2.set_ylabel("WikiText-2 PPL", color=C_PPL)
    ax2.tick_params(axis="y", labelcolor=C_PPL)

    # Step annotations at key waypoints
    annotate_steps = {100, 800, 1400, 2000}
    for t_data in traj:
        if t_data["step"] in annotate_steps:
            xi = t_data["norm_over_sh_pre"]
            yi = t_data["task_val_acc"] * 100
            ax1.annotate(f"step {t_data['step']}",
                         xy=(xi, yi),
                         xytext=(xi + 0.15, yi - 6),
                         fontsize=7.5, color="#374151",
                         arrowprops=dict(arrowstyle="-", color="#9CA3AF", lw=0.8))

    # Zone labels
    ax1.text(0.42, 43, "inside ball\n(no task learning,\nno forgetting)",
             fontsize=8, color="#166534", ha="center",
             bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.8, ec="#BBF7D0"))
    ax1.text(2.1, 43, "outside ball\n(task learning +\nforgetting begins)",
             fontsize=8, color="#991B1B", ha="center",
             bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.8, ec="#FCA5A5"))

    ax1.set_xlabel("‖ΔW‖ / σ½_pretrained  (normalized LoRA update norm)")
    ax1.set_title(
        "Phase Transition at the Certified Boundary\n"
        f"GPT-2  |  SST-2  |  lr=2e-4, rank=8, 2000 steps  |  σ½_pre = {sh_pre:.5f}",
        fontsize=10)

    lines1, l1 = ax1.get_legend_handles_labels()
    lines2, l2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, l1 + l2, loc="upper left", fontsize=9)
    fig.tight_layout()
    _save(fig, "fig1_phase_transition.png")


# ══════════════════════════════════════════════════════════════════════════════
# FIG 2 — Goldilocks Zone: all models × tasks
# Two panels: task_acc and ppl_ratio_actual vs norm/σ½_pre
# ══════════════════════════════════════════════════════════════════════════════
def fig2_goldilocks():
    DATASETS = [
        (os.path.join(V2, "basin_widening_v2", "gpt2", "sst2",
                      "widening_summary.json"),
         "GPT-2 / SST-2",   C_GPT2,  "o",  "-"),
        (os.path.join(V2, "basin_widening_v2", "gpt2", "agnews",
                      "widening_summary.json"),
         "GPT-2 / AG News", C_GPT2,  "s",  "--"),
        (os.path.join(V2, "basin_widening_v2", "meta-llama_Llama-3.2-3B", "sst2",
                      "widening_summary.json"),
         "Llama-3B / SST-2",C_LLAMA, "o",  "-"),
    ]
    PPL_CAP = 5.0

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    all_handles = []

    for (path, label, color, marker, ls) in DATASETS:
        rows = jload(path)["results"]
        rows_valid = [r for r in rows if r.get("ppl_ratio_actual", 999) <= 100]
        rows_valid.sort(key=lambda r: r["norm_over_sh"])

        x      = [r["norm_over_sh"]      for r in rows_valid]
        y_acc  = [r["task_val_acc"] * 100 for r in rows_valid]
        y_ppl  = [min(r["ppl_ratio_actual"], PPL_CAP) for r in rows_valid]
        broken = [r for r in rows if r.get("ppl_ratio_actual", 0) > 100]

        ax1.scatter(x, y_acc, c=color, marker=marker, s=80, zorder=6,
                    edgecolors="white", lw=0.8, alpha=0.92)
        ax1.plot(x, y_acc, color=color, lw=1.3, ls=ls, alpha=0.45, zorder=4)

        ax2.scatter(x, y_ppl, c=color, marker=marker, s=80, zorder=6,
                    edgecolors="white", lw=0.8, alpha=0.92)
        ax2.plot(x, y_ppl, color=color, lw=1.3, ls=ls, alpha=0.45, zorder=4)

        for r in broken:
            xi  = r["norm_over_sh"]
            ppl = r["ppl_ratio_actual"]
            ax2.scatter([xi], [PPL_CAP], c=color, marker="^", s=100, zorder=7,
                        edgecolors="white", lw=0.8, alpha=0.92)
            lbl = f"×{ppl:.0f}" if ppl < 1e4 else f"×{ppl/1e3:.0f}k"
            ax2.annotate(lbl, xy=(xi, PPL_CAP), xytext=(xi + 0.15, PPL_CAP - 0.2),
                         fontsize=7.5, color=color)

        all_handles.append(
            Line2D([0],[0], color=color, marker=marker, markersize=6,
                   ls=ls, lw=1.5, label=label))

    for ax in (ax1, ax2):
        ax.axvspan(0, 1.0, alpha=0.09, color=C_GREEN, zorder=0)
        ax.axvline(1.0, color=C_BOUND, lw=1.6, ls="--", alpha=0.75, zorder=5)
        ax.set_xlabel("‖ΔW‖ / σ½_pretrained  (normalized update norm)")

    ax1.set_ylabel("Task Accuracy (%)")
    ax1.set_title("Task Accuracy vs. Normalized Update Norm")
    ax1.set_ylim(30, 105)
    ax1.text(0.42, 33, "inside\nball", fontsize=8, color="#166534", ha="center")
    ax1.text(2.8,  33, "outside\nball", fontsize=8, color="#991B1B", ha="center")

    ax2.axhline(1.0, color="#6B7280", lw=0.9, ls=":", alpha=0.6)
    ax2.set_ylabel(f"WikiText-2 PPL ratio (true)  [capped at {PPL_CAP}×, ▲ = off-chart]")
    ax2.set_title("Forgetting vs. Normalized Update Norm")
    ax2.set_ylim(0.8, PPL_CAP + 0.5)

    all_handles += [
        Line2D([0],[0], color=C_BOUND, lw=1.6, ls="--", alpha=0.75,
               label="norm/σ½ = 1  (certified boundary)"),
        mpatches.Patch(fc=C_GREEN, alpha=0.25, label="certified safe zone  (norm/σ½ < 1)"),
    ]
    fig.legend(handles=all_handles, loc="lower center", ncol=len(all_handles),
               fontsize=9, bbox_to_anchor=(0.5, -0.05), framealpha=0.95)

    fig.suptitle(
        "Goldilocks Zone: Task Accuracy and Forgetting Across Models and Tasks\n"
        "Llama-3B: 94.8% SST-2 accuracy inside the ball  |  "
        "GPT-2: 85.5% AG News accuracy inside the ball  |  "
        "Forgetting begins only outside the ball",
        y=1.03, fontsize=10)
    fig.tight_layout()
    _save(fig, "fig2_goldilocks.png")


# ══════════════════════════════════════════════════════════════════════════════
# FIG 3 — Rank Sweep: Llama-3B SST-2 at lr=1e-4  [2-panel]
# Panel A: norm/σ½_pre vs rank  |  Panel B: task_acc + PPL vs rank
# ══════════════════════════════════════════════════════════════════════════════
def fig3_rank_sweep():
    path = os.path.join(V2, "basin_widening_rank_sweep_v2",
                        "meta-llama_Llama-3.2-3B", "sst2", "widening_summary.json")
    rows = jload(path)["results"]

    ranks   = [r["rank"]                   for r in rows]
    rd      = [r["rank"]/LLAMA_HIDDEN*100  for r in rows]
    norm_sh = [r["norm_over_sh"]           for r in rows]
    acc     = [r["task_val_acc"] * 100     for r in rows]
    ppl_raw = [r["ppl_ratio_actual"]       for r in rows]

    thresh_idx = next((i for i, v in enumerate(norm_sh) if v >= 1.0), None)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

    # ── Panel A: norm/σ½_pre bar chart ────────────────────────────────────
    bar_colors = [C_GREEN if v < 1.0 else "#F87171" for v in norm_sh]
    bars = ax1.bar(range(len(ranks)), norm_sh, color=bar_colors, alpha=0.78,
                   edgecolor="white", zorder=4)
    ax1.axhline(1.0, color=C_BOUND, lw=2.0, ls="--", zorder=5,
                label="certified boundary (norm/σ½ = 1)")
    ax1.set_xticks(range(len(ranks)))
    ax1.set_xticklabels([f"r={r}\n({d:.1f}%)" for r, d in zip(ranks, rd)],
                        fontsize=8.5)
    ax1.set_ylabel("‖ΔW‖ / σ½_pretrained")
    ax1.set_title("LoRA Rank Controls the Certified Boundary\n"
                  "Llama-3.2-3B  |  SST-2  |  lr=1e-4  |  500 steps")
    ax1.legend(fontsize=9)
    for b, v in zip(bars, norm_sh):
        ax1.text(b.get_x() + b.get_width()/2, v + 0.08, f"{v:.2f}",
                 ha="center", fontsize=8.5, color="#1F2937", fontweight="bold")

    # Shade "outside ball" region
    if thresh_idx is not None:
        ax1.axvspan(thresh_idx - 0.5, len(ranks) - 0.5, alpha=0.07, color=C_BOUND)
        ax1.text(thresh_idx + 0.5, ax1.get_ylim()[1] * 0.95, "outside ball →",
                 fontsize=8, color=C_BOUND)

    # ── Panel B: accuracy + PPL vs rank ──────────────────────────────────
    ax2_r = ax2.twinx()
    ax2_r.spines["right"].set_visible(True)

    ax2.plot(range(len(ranks)), acc, color=C_GPT2, lw=2.2, marker="o", ms=7,
             zorder=6, label="SST-2 accuracy (%)")
    ax2.set_ylim(70, 102)
    ax2.set_ylabel("SST-2 Accuracy (%)", color=C_GPT2)
    ax2.tick_params(axis="y", labelcolor=C_GPT2)

    # PPL ratio: plot normally up to rank 512, annotate rank=768 separately
    ppl_plot_idx = list(range(len(ranks)))
    ppl_cap      = 200.0
    ppl_disp     = [min(p, ppl_cap) for p in ppl_raw]

    ax2_r.plot(ppl_plot_idx, ppl_disp, color=C_PPL, lw=1.8, marker="s", ms=6,
               ls="--", zorder=6, label="WikiText-2 PPL ratio")
    ax2_r.axhline(1.0, color=C_PPL, lw=0.8, ls=":", alpha=0.4)
    ax2_r.set_yscale("log")
    ax2_r.set_ylim(0.9, ppl_cap * 2)
    ax2_r.set_ylabel("WikiText-2 PPL ratio (log scale)", color=C_PPL)
    ax2_r.tick_params(axis="y", labelcolor=C_PPL)

    # Annotate the clipped rank=768 point
    for i, (p, pd) in enumerate(zip(ppl_raw, ppl_disp)):
        if p > ppl_cap:
            lbl = f"×{p/1000:.0f}k" if p > 1000 else f"×{p:.0f}"
            ax2_r.annotate(lbl, xy=(i, ppl_cap), xytext=(i - 0.7, ppl_cap * 1.5),
                           fontsize=8, color=C_PPL,
                           arrowprops=dict(arrowstyle="->", color=C_PPL, lw=0.9))

    ax2.set_xticks(list(range(len(ranks))))
    ax2.set_xticklabels([f"r={r}\n({d:.1f}%)" for r, d in zip(ranks, rd)],
                        fontsize=8.5)
    ax2.set_title("Task Accuracy and Forgetting vs. Rank\n"
                  "Higher rank → exits certified ball → forgetting accumulates")

    if thresh_idx is not None:
        ax2.axvspan(thresh_idx - 0.5, len(ranks) - 0.5, alpha=0.06, color=C_BOUND)

    lines1, l1 = ax2.get_legend_handles_labels()
    lines2, l2 = ax2_r.get_legend_handles_labels()
    ax2.legend(lines1 + lines2, l1 + l2, loc="lower left", fontsize=9)

    fig.tight_layout()
    _save(fig, "fig3_rank_sweep.png")


# ══════════════════════════════════════════════════════════════════════════════
# FIG 4 — Scale Comparison: pretrained density curves GPT-2 vs Llama-3B
# ══════════════════════════════════════════════════════════════════════════════
def fig4_density_scale():
    paths = {
        "GPT-2 (124M)": (os.path.join(V2, "basin_widening_v2", "gpt2",
                                       "sst2", "pretrained_density.json"), C_GPT2),
        "Llama-3.2-3B": (os.path.join(V2, "basin_widening_v2",
                                       "meta-llama_Llama-3.2-3B", "sst2",
                                       "pretrained_density.json"),         C_LLAMA),
    }

    fig, ax = plt.subplots(figsize=(7, 5))
    shs = {}

    for label, (path, color) in paths.items():
        sr   = jload(path)["sigma_results"]
        sigs = [r["sigma"]    for r in sr]
        dens = [r["density"]  for r in sr]
        lo   = [r["ci_lower"] for r in sr]
        hi   = [r["ci_upper"] for r in sr]
        sh   = half_max_sigma(sr)
        shs[label] = sh

        ax.plot(sigs, dens, color=color, lw=2.5, marker="o", ms=5,
                label=f"{label}  (σ½ = {sh:.5f})")
        ax.fill_between(sigs, lo, hi, color=color, alpha=0.12)
        ax.axvline(sh, color=color, lw=1.3, ls="--", alpha=0.6)
        ax.text(sh * 1.15, 0.54, f"σ½ = {sh:.4f}", color=color, fontsize=8.5)

    ax.axhline(0.5, color="#9CA3AF", lw=0.8, ls=":", alpha=0.6,
               label="C = max_density / 2  (σ½ threshold)")
    ax.set_xscale("log")
    ax.set_xlabel("Perturbation scale σ")
    ax.set_ylabel("Certified density  C(θ, σ)")
    ax.set_title(
        "Pretrained Basin Width vs. Model Scale\n"
        "N=200 perturbations  |  WikiText-2 eval  |  95% CI",
        fontsize=10)
    ax.legend(fontsize=9.5)
    ax.set_ylim(-0.03, 1.08)

    ratio = shs["GPT-2 (124M)"] / shs["Llama-3.2-3B"]
    sh_l  = shs["Llama-3.2-3B"]
    ax.annotate(
        f"Llama-3B basin {ratio:.1f}×\ntighter than GPT-2",
        xy=(sh_l, 0.40), xytext=(sh_l * 4, 0.22),
        fontsize=9, color="#374151",
        arrowprops=dict(arrowstyle="->", color="#6B7280", lw=1.2))

    fig.tight_layout()
    _save(fig, "fig4_density_scale.png")


# ══════════════════════════════════════════════════════════════════════════════
# FIG 5 — Widening Ratio (valid conditions only)
# ══════════════════════════════════════════════════════════════════════════════
def fig5_widening_summary():
    DATASETS = [
        (os.path.join(V2, "basin_widening_v2", "gpt2", "sst2",
                      "widening_summary.json"),          "GPT-2 / SST-2",   C_GPT2,  "o",  "-"),
        (os.path.join(V2, "basin_widening_v2", "gpt2", "agnews",
                      "widening_summary.json"),          "GPT-2 / AG News", C_GPT2,  "s",  "--"),
        (os.path.join(V2, "basin_widening_v2", "meta-llama_Llama-3.2-3B", "sst2",
                      "widening_summary.json"),          "Llama-3B / SST-2",C_LLAMA, "o",  "-"),
    ]

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.axhline(1.0, color="#9CA3AF", lw=1.5, ls="--", alpha=0.7,
               label="no change (ratio = 1.0)")
    ax.axvline(1.0, color=C_BOUND, lw=1.6, ls="--", alpha=0.7,
               label="norm/σ½ = 1  (certified boundary)")
    ax.axvspan(0, 1.0, alpha=0.08, color=C_GREEN, zorder=0)

    for (path, label, color, marker, ls) in DATASETS:
        rows  = jload(path)["results"]
        valid = [r for r in rows
                 if r.get("ppl_ratio_actual", 999) < 3.0
                 and r.get("norm_over_sh", 999) < 6.0
                 and r.get("widening_ratio", 999) < 10.0]
        valid.sort(key=lambda r: r["norm_over_sh"])
        x = [r["norm_over_sh"]   for r in valid]
        y = [r["widening_ratio"]  for r in valid]
        ax.scatter(x, y, color=color, marker=marker, s=80, alpha=0.88,
                   edgecolors="white", lw=0.8, zorder=5)
        ax.plot(x, y, color=color, lw=1.3, ls=ls, alpha=0.45, zorder=4,
                label=label)

    ax.set_xlabel("‖ΔW‖ / σ½_pretrained  (normalized update norm)")
    ax.set_ylabel("σ½ widening ratio  (σ½_ft / σ½_pre)")
    ax.set_title(
        "σ½ Widening: How Fine-Tuning Changes the Certified Basin Width\n"
        "Filtered to valid conditions (ppl_ratio < 3×, widening < 10×)  |  "
        "Widening is modest: 1.0–1.5× inside the ball",
        fontsize=10)
    ax.set_ylim(0.7, 3.0)
    ax.legend(fontsize=9)
    fig.tight_layout()
    _save(fig, "fig5_widening_summary.png")


# ── Main ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print(f"Output → {OUTDIR}\n")
    print("Fig 1: Phase transition trajectory...")
    fig1_phase_transition()
    print("Fig 2: Goldilocks zone (all models/tasks)...")
    fig2_goldilocks()
    print("Fig 3: Rank sweep (Llama-3B, 2-panel)...")
    fig3_rank_sweep()
    print("Fig 4: Density scale comparison...")
    fig4_density_scale()
    print("Fig 5: Widening ratio summary...")
    fig5_widening_summary()
    print("\nDone.")
