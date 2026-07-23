"""
make_pres_figures.py
Generates dark-themed, presentation-quality figures for ntrs_presentation2.pptx.
All figures saved to results/pres_figures/.
"""

import json, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
from matplotlib.patches import FancyBboxPatch

ROOT    = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(ROOT, "results", "pres_figures")
os.makedirs(OUT_DIR, exist_ok=True)

# ── Dark theme matching the slide palette ────────────────────────────────────
BG      = "#1a1a2e"
BG2     = "#0d1b2a"
LIGHT   = "#f5f5f5"
MID     = "#b0b8c8"
ACCENT  = "#e94f37"
ACCENT2 = "#397bbf"
GREEN   = "#43c59e"
YELLOW  = "#f5c518"

MODEL_COLORS = {
    "GPT-2 (124M)":       "#397bbf",
    "Qwen2.5-3B (2.5B)":  "#D97706",
    "Llama-3.2-3B (3B)":  "#7C3AED",
    "Llama-3.1-8B (8B)":  "#e94f37",
}

DARK_RC = {
    "figure.facecolor":  BG,
    "axes.facecolor":    BG2,
    "axes.edgecolor":    MID,
    "axes.labelcolor":   LIGHT,
    "axes.titlecolor":   LIGHT,
    "xtick.color":       MID,
    "ytick.color":       MID,
    "text.color":        LIGHT,
    "grid.color":        "#2a3a5a",
    "grid.linestyle":    "--",
    "grid.alpha":        0.5,
    "legend.facecolor":  "#111827",
    "legend.edgecolor":  MID,
    "legend.labelcolor": LIGHT,
    "font.family":       "sans-serif",
    "font.size":         13,
    "axes.titlesize":    15,
    "axes.labelsize":    13,
    "legend.fontsize":   11,
}

SOURCES = [
    ("GPT-2 (124M)",      "outputs_0606/lr_sweep/gpt2/sst2/widening_summary.json"),
    ("Qwen2.5-3B (2.5B)", "outputs_0607/qwen3b_lr_sweep/Qwen_Qwen2.5-3B/sst2/widening_summary.json"),
    ("Llama-3.2-3B (3B)", "outputs_0607/llama3b_lr_sweep/meta-llama_Llama-3.2-3B/sst2/widening_summary.json"),
    ("Llama-3.1-8B (8B)", "outputs_0607/llama8b_lr_sweep/meta-llama_Meta-Llama-3.1-8B/sst2/widening_summary.json"),
]

# Full FT (rank=768) sources — different fine-tuning method
FULLFT_SOURCES = [
    ("GPT-2 (124M)",
     "outputs_0605/full_ft_baseline/gpt2/sst2/widening_summary.json"),
    ("Llama-3.2-3B (3B)",
     "outputs_0609/outputs_0609/llama3b_fullft_lr_sweep/meta-llama_Llama-3.2-3B/sst2/widening_summary.json"),
]

# Steps sweep + trajectory (Llama-3B, rank=8)
STEPS_SOURCE      = "outputs_0610/llama3b_steps_sweep/meta-llama_Llama-3.2-3B/sst2/widening_summary.json"
TRAJ_SOURCE       = "outputs_0610/meta-llama_Llama-3.2-3B/sst2/widening_summary.json"
TRAJ_CURVE_SOURCE = "outputs_0610/llama3b_trajectory/meta-llama_Llama-3.2-3B/sst2/rank8_steps2000_trajectory.json"
TRAJ_CURVE_WS     = "outputs_0610/llama3b_trajectory/meta-llama_Llama-3.2-3B/sst2/widening_summary.json"

DENSITY_SOURCES = [
    ("GPT-2 (124M)",
     "outputs_0606/lr_sweep/gpt2/sst2/pretrained_density.json",
     "outputs_0606/lr_sweep/gpt2/sst2/lr1e-04_rank8_steps500_density.json",
     "1e-4"),
    ("Qwen2.5-3B (2.5B)",
     "outputs_0607/qwen3b_lr_sweep/Qwen_Qwen2.5-3B/sst2/pretrained_density.json",
     "outputs_0607/qwen3b_lr_sweep/Qwen_Qwen2.5-3B/sst2/lr1e-04_rank8_steps500_density.json",
     "1e-4"),
    ("Llama-3.2-3B (3B)",
     "outputs_0607/llama3b_lr_sweep/meta-llama_Llama-3.2-3B/sst2/pretrained_density.json",
     "outputs_0607/llama3b_lr_sweep/meta-llama_Llama-3.2-3B/sst2/lr1e-04_rank8_steps500_density.json",
     "1e-4"),
    ("Llama-3.1-8B (8B)",
     "outputs_0607/llama8b_lr_sweep/meta-llama_Meta-Llama-3.1-8B/sst2/pretrained_density.json",
     "outputs_0607/llama8b_lr_sweep/meta-llama_Meta-Llama-3.1-8B/sst2/lr1e-04_rank8_steps500_density.json",
     "1e-4"),
]

def _save(fig, name):
    path = os.path.join(OUT_DIR, name)
    fig.savefig(path, dpi=160, bbox_inches="tight", facecolor=BG)
    plt.close()
    print(f"Saved {path}")

def load_density_sr(path):
    full = path if os.path.isabs(path) else os.path.join(ROOT, path)
    if not os.path.exists(full):
        return None
    d = json.load(open(full))
    sr = d.get("sigma_results", None)
    return sr if isinstance(sr, list) else (d if isinstance(d, list) else None)

def sigma_half(sr):
    if sr is None: return float("nan")
    dens   = [r["density"] for r in sr]
    sigmas = [r["sigma"]   for r in sr]
    peak = max(dens)
    target = peak / 2.0
    for i in range(len(dens)-1):
        if dens[i] >= target > dens[i+1]:
            t = (target-dens[i])/(dens[i+1]-dens[i])
            return sigmas[i] + t*(sigmas[i+1]-sigmas[i])
    return sigmas[-1]


# ── Fig 1: 4-panel density curves (experiment verification) ──────────────────
def fig_density_grid():
    plt.rcParams.update(DARK_RC)
    fig, axes = plt.subplots(2, 2, figsize=(14, 8))
    fig.suptitle(
        "Certified Density C(θ, σ)  —  Pretrained vs LoRA lr=1e-4  (all 4 models)",
        fontsize=16, color=LIGHT, y=1.01
    )

    for ax, (label, pre_path, ft_path, lr_label) in zip(axes.flat, DENSITY_SOURCES):
        color = MODEL_COLORS[label]
        ax.set_facecolor(BG2)
        for spine in ax.spines.values():
            spine.set_edgecolor(MID)

        pre_sr = load_density_sr(pre_path)
        ft_sr  = load_density_sr(ft_path)

        sh_vals = {}
        for sr, ls, lbl, alpha, key in [
            (pre_sr, "-",  "Pretrained",       1.0, "pre"),
            (ft_sr,  "--", f"LoRA lr={lr_label}", 0.9, "ft"),
        ]:
            if sr is None: continue
            sigmas = [r["sigma"]   for r in sr]
            dens   = [r["density"] for r in sr]
            cis_lo = [r.get("ci_lower", r["density"]) for r in sr]
            cis_hi = [r.get("ci_upper", r["density"]) for r in sr]
            ax.plot(sigmas, dens, ls=ls, lw=2.5, color=color,
                    marker="o", ms=4, label=lbl, alpha=alpha)
            ax.fill_between(sigmas, cis_lo, cis_hi, color=color, alpha=0.15)

            sh = sigma_half(sr)
            if np.isfinite(sh):
                sh_vals[key] = sh
                ax.axvline(sh, color=color, lw=1.2, ls=":",
                           alpha=0.75 if ls == "-" else 0.4)

        # σ½ annotations: pre at y=0.04, ft at y=0.44 to avoid overlap when values are similar
        if "pre" in sh_vals:
            ax.text(sh_vals["pre"], 0.04, f"σ½_pre\n{sh_vals['pre']:.2e}",
                    ha="left", va="bottom", fontsize=7.5, color=color,
                    alpha=0.9, rotation=0,
                    bbox=dict(boxstyle="round,pad=0.15", fc=BG, ec="none", alpha=0.7))
        if "ft" in sh_vals:
            ax.text(sh_vals["ft"], 0.44, f"σ½_ft\n{sh_vals['ft']:.2e}",
                    ha="left", va="bottom", fontsize=7.5, color=color,
                    alpha=0.7, rotation=0,
                    bbox=dict(boxstyle="round,pad=0.15", fc=BG, ec="none", alpha=0.7))

        ax.set_xscale("log")
        ax.set_xlabel("Perturbation scale σ", fontsize=11, color=MID)
        ax.set_ylabel("C(θ, σ)  — certified density", fontsize=11, color=MID)
        ax.set_title(label, fontsize=13, color=color, fontweight="bold")
        ax.axhline(0.5, color=YELLOW, lw=0.8, ls=":", alpha=0.5, label="density=0.5")
        ax.legend(fontsize=9, loc="upper right")
        ax.grid(True)
        ax.set_ylim(-0.03, 1.05)
        ax.tick_params(colors=MID)

    fig.tight_layout()
    _save(fig, "fig1_density_grid.png")


# ── Fig 2: LR × Model status matrix ─────────────────────────────────────────
def fig_lr_matrix():
    plt.rcParams.update(DARK_RC)

    model_names = [label for label, _ in SOURCES]
    lrs = [1e-5, 5e-5, 1e-4, 2e-4, 5e-4, 1e-3]
    lr_labels = ["1e-5", "5e-5", "1e-4", "2e-4", "5e-4", "1e-3"]

    # Build data matrix
    ratios = np.zeros((len(model_names), len(lrs)))
    nsh    = np.zeros((len(model_names), len(lrs)))
    ppls   = np.zeros((len(model_names), len(lrs)))

    for i, (label, path) in enumerate(SOURCES):
        d = json.load(open(os.path.join(ROOT, path)))
        for j, lr in enumerate(lrs):
            r = next((x for x in d["results"] if abs(x["lr"]-lr)/lr < 0.01), None)
            if r:
                ratios[i,j] = r["widening_ratio"]
                nsh[i,j]    = r["norm_over_sh"]
                ppls[i,j]   = r["ppl_ratio"]

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle("Experiment Matrix: All 4 Models × 6 Learning Rates",
                 fontsize=16, color=LIGHT, y=1.02)

    def heatmap(ax, data, title, fmt, cmap, vmin, vmax, annotate_boundary=False):
        ax.set_facecolor(BG2)
        im = ax.imshow(data, cmap=cmap, vmin=vmin, vmax=vmax, aspect="auto")
        ax.set_xticks(range(len(lrs)))
        ax.set_xticklabels(lr_labels, fontsize=11, color=LIGHT)
        ax.set_yticks(range(len(model_names)))
        ax.set_yticklabels(model_names, fontsize=11, color=LIGHT)
        ax.set_title(title, fontsize=13, color=LIGHT, pad=10)
        ax.set_xlabel("Learning Rate", color=MID)
        cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.ax.yaxis.set_tick_params(color=MID)
        plt.setp(cbar.ax.yaxis.get_ticklabels(), color=MID)

        for i in range(data.shape[0]):
            for j in range(data.shape[1]):
                v = data[i,j]
                txt = fmt.format(v)
                # bold if boundary cell
                weight = "bold" if annotate_boundary and abs(nsh[i,j]-1.0) < 0.5 else "normal"
                fc = "#000000" if v > (vmin+vmax)*0.55 else LIGHT
                ax.text(j, i, txt, ha="center", va="center",
                        fontsize=10, color=fc, fontweight=weight)

        if annotate_boundary:
            # draw boundary where norm/sh crosses 1
            for i in range(data.shape[0]):
                for j in range(len(lrs)-1):
                    if (nsh[i,j] < 1.0) != (nsh[i,j+1] < 1.0):
                        ax.plot([j+0.5, j+0.5], [i-0.5, i+0.5],
                                color=ACCENT, lw=2.5)

    # Clamp ratios for display (collapse values)
    ratios_disp = np.clip(ratios, 0.5, 3.0)
    heatmap(axes[0], ratios_disp, "σ½ Widening Ratio\n(σ½_ft / σ½_pre, capped at 3×)",
            "{:.2f}", "RdYlGn", 0.5, 3.0, annotate_boundary=True)

    heatmap(axes[1], np.log10(np.clip(nsh, 0.03, 20)),
            "Update Size\n(log₁₀  ‖ΔW‖/σ½_pre)",
            "{:.1f}", "YlOrRd", -1.5, 1.3, annotate_boundary=False)
    # add boundary line on panel 2
    for i in range(nsh.shape[0]):
        for j in range(len(lrs)-1):
            if (nsh[i,j] < 1.0) != (nsh[i,j+1] < 1.0):
                axes[1].plot([j+0.5, j+0.5], [i-0.5, i+0.5],
                             color=ACCENT, lw=2.5)

    heatmap(axes[2], ppls, "PPL Ratio\n(ppl_ft / ppl_pre)",
            "{:.3f}", "RdYlGn_r", 0.98, 1.12, annotate_boundary=True)

    # Legend for boundary line
    for ax in axes:
        ax.plot([], [], color=ACCENT, lw=2.5, label="ball boundary\n(norm/σ½=1)")
        ax.legend(fontsize=9, loc="lower right")

    fig.tight_layout()
    _save(fig, "fig2_lr_matrix.png")


# ── Fig 3: σ½ Scaling Law bar chart (dark) ──────────────────────────────────
def fig_sigma_scaling():
    plt.rcParams.update(DARK_RC)
    fig, ax = plt.subplots(figsize=(10, 5.5))
    ax.set_facecolor(BG2)
    for sp in ax.spines.values(): sp.set_edgecolor(MID)

    labels  = [l for l,_ in SOURCES]
    sh_pre  = []
    ppl_pre = []
    for label, path in SOURCES:
        d = json.load(open(os.path.join(ROOT, path)))
        sh_pre.append(d["results"][0]["sigma_half_pre"])
        ppl_pre.append(d["results"][0]["ppl_pre"])

    colors = [MODEL_COLORS[l] for l in labels]
    x = np.arange(len(labels))
    bars = ax.bar(x, sh_pre, color=colors, alpha=0.85, width=0.55,
                  edgecolor=BG, linewidth=1.5)

    for bar, sh, ppl, label in zip(bars, sh_pre, ppl_pre, labels):
        ax.text(bar.get_x() + bar.get_width()/2,
                bar.get_height() + max(sh_pre)*0.012,
                f"σ½ = {sh:.5f}\nPPL = {ppl:.2f}",
                ha="center", va="bottom", fontsize=10, color=LIGHT)

    # log-linear trend
    log_p = np.log10([124, 2500, 3000, 8000])
    coeffs = np.polyfit(log_p, sh_pre, 1)
    xs = np.linspace(log_p[0]-0.1, log_p[-1]+0.1, 100)
    ys = np.polyval(coeffs, xs)
    xplot = np.interp(10**xs, [124, 2500, 3000, 8000], x)
    ax.plot(xplot, ys, color=YELLOW, lw=2, ls="--", alpha=0.7,
            label="log-linear trend")

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=12, color=LIGHT)
    ax.set_ylabel("σ½_pre  (certified basin width)", fontsize=13, color=MID)
    ax.set_title(
        "Pretrained Basin Width (σ½) Shrinks with Model Scale\n"
        "Larger / better-trained models sit in tighter weight-space minima",
        fontsize=14, color=LIGHT
    )
    ax.legend(fontsize=10)
    ax.grid(True, axis="y")
    ax.tick_params(colors=MID)
    _save(fig, "fig3_sigma_scaling.png")


# ── Fig 4: PPL vs norm/σ½ — the phase boundary (dark, enhanced) ─────────────
def fig_phase_boundary():
    plt.rcParams.update(DARK_RC)
    fig, ax = plt.subplots(figsize=(14, 7.5))
    fig.subplots_adjust(top=0.88, right=0.72)
    ax.set_facecolor(BG2)
    for sp in ax.spines.values(): sp.set_edgecolor(MID)

    from matplotlib.lines import Line2D

    # LoRA rank=8 points — circles; annotate LR only on clearly outside-ball visible points
    for label, path in SOURCES:
        d = json.load(open(os.path.join(ROOT, path)))
        color = MODEL_COLORS[label]
        pts = [(r["norm_over_sh"], r["ppl_ratio"], r["lr"])
               for r in d["results"] if r["ppl_ratio"] < 5.0]
        if not pts: continue
        ns, pp, lrs_pts = zip(*pts)
        ax.scatter(ns, pp, color=color, s=110, zorder=5, alpha=0.9,
                   marker="o", label=label, edgecolors=BG2, linewidths=0.8)
        # only annotate points clearly outside the ball and within ylim
        for n, p, lr in zip(ns, pp, lrs_pts):
            if n > 1.0 and p < 1.14:
                ax.annotate(f"{lr:.0e}", (n, p),
                            textcoords="offset points", xytext=(5, 6),
                            fontsize=8, color=color, alpha=0.85)

    # Full FT rank=768 points — triangles
    for label, path in FULLFT_SOURCES:
        fpath = os.path.join(ROOT, path)
        if not os.path.exists(fpath): continue
        d = json.load(open(fpath))
        color = MODEL_COLORS[label]
        pts = [(r["norm_over_sh"], r["ppl_ratio"], r.get("lr", "?"))
               for r in d["results"] if r["ppl_ratio"] < 5.0]
        if not pts: continue
        ns, pp, tags = zip(*pts)
        ax.scatter(ns, pp, color=color, s=180, zorder=6, alpha=0.95,
                   marker="^", edgecolors=LIGHT, linewidths=1.2,
                   label=f"_ft_{label}")
        offsets = [(8, 8), (-45, 8), (8, -14), (-45, -14)]
        for i, (n, p, tag) in enumerate(zip(ns, pp, tags)):
            ann = f"{tag:.0e}" if isinstance(tag, float) else str(tag)
            dx, dy = offsets[i % len(offsets)]
            ax.annotate(ann, (n, p),
                        textcoords="offset points", xytext=(dx, dy),
                        fontsize=9, color=color, alpha=0.9,
                        fontweight="bold")

    # Steps sweep points — diamonds (Llama-3B, lr=1e-4, vary steps)
    color_3b = MODEL_COLORS["Llama-3.2-3B (3B)"]
    steps_fp = os.path.join(ROOT, STEPS_SOURCE)
    if os.path.exists(steps_fp):
        d_steps = json.load(open(steps_fp))
        for r in d_steps["results"]:
            if r["ppl_ratio"] < 5.0:
                ax.scatter([r["norm_over_sh"]], [r["ppl_ratio"]],
                           color=color_3b, s=85, zorder=6, alpha=0.85,
                           marker="D", edgecolors=BG2, linewidths=0.5,
                           label="_steps_lora")

    # Trajectory endpoint — star (Llama-3B, lr=2e-4, steps=2000)
    traj_fp = os.path.join(ROOT, TRAJ_SOURCE)
    if os.path.exists(traj_fp):
        d_traj = json.load(open(traj_fp))
        for r in d_traj["results"]:
            if r["ppl_ratio"] < 5.0:
                ax.scatter([r["norm_over_sh"]], [r["ppl_ratio"]],
                           color=color_3b, s=230, zorder=7, alpha=0.95,
                           marker="*", edgecolors=LIGHT, linewidths=0.8,
                           label="_traj_lora")
                ax.annotate("lr=2e-4\n2000 steps",
                            (r["norm_over_sh"], r["ppl_ratio"]),
                            textcoords="offset points", xytext=(8, 6),
                            fontsize=8, color=color_3b, alpha=0.9)

    # Shaded zones
    ax.axvspan(0, 1, color=GREEN, alpha=0.08, zorder=0)
    ax.axvspan(1, 20, color=ACCENT, alpha=0.06, zorder=0)
    ax.axvline(1.0, color=ACCENT, lw=2.5, ls="--", alpha=0.8)
    ax.axhline(1.0, color=LIGHT, lw=0.8, ls=":", alpha=0.35)

    # Zone labels — placed in lower part of shaded bands, clear of legend & data
    ax.text(0.055, 0.978, "SAFE\nZONE", ha="center", va="bottom", fontsize=12,
            color=GREEN, fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.3", fc=BG, ec=GREEN, alpha=0.7, lw=1.2))
    ax.text(4.5, 0.978, "PPL\nDEGRADES", ha="center", va="bottom", fontsize=12,
            color=ACCENT, fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.3", fc=BG, ec=ACCENT, alpha=0.7, lw=1.2))

    # Clean legend: model colors + method markers separate
    model_handles, model_labels = ax.get_legend_handles_labels()
    # filter out the _ft_ pseudo-labels
    model_handles = [h for h, l in zip(model_handles, model_labels) if not l.startswith("_")]
    model_labels  = [l for l in model_labels if not l.startswith("_")]

    method_handles = [
        Line2D([0],[0], marker="o", color="w", markerfacecolor=LIGHT,
               markersize=10, label="● LoRA rank=8  (vary LR)"),
        Line2D([0],[0], marker="^", color="w", markerfacecolor=LIGHT,
               markeredgecolor=LIGHT, markersize=11, label="▲ Full FT (all params)"),
        Line2D([0],[0], marker="D", color="w", markerfacecolor=LIGHT,
               markersize=9, label="◆ LoRA rank=8  (vary steps, Llama-3B)"),
        Line2D([0],[0], marker="*", color="w", markerfacecolor=LIGHT,
               markersize=12, label="★ LoRA lr=2e-4  (trajectory, Llama-3B)"),
        Line2D([0],[0], color=ACCENT, lw=2.5, ls="--",
               label="certified ball boundary"),
    ]
    leg1 = ax.legend(model_handles, model_labels,
                     fontsize=11, loc="upper left",
                     title="Model  (color)", title_fontsize=10,
                     bbox_to_anchor=(0.01, 0.99))
    ax.add_artist(leg1)
    # Place method legend outside the axes to the right — no overlap
    ax.legend(method_handles, [h.get_label() for h in method_handles],
              fontsize=10, loc="upper left",
              title="Method  (marker)", title_fontsize=10,
              bbox_to_anchor=(1.02, 0.97), borderaxespad=0)

    ax.set_xscale("log")
    ax.set_xlabel("‖ΔW‖ / σ½_pre  (update norm relative to certified radius)",
                  fontsize=13, color=MID)
    ax.set_ylabel("PPL ratio  (ppl_ft / ppl_pre)", fontsize=13, color=MID)
    ax.set_xlim(0.025, 12)
    ax.set_ylim(0.975, 1.16)
    ax.set_title(
        "PPL Preservation vs Certified Ball Position  —  LoRA & Full FT\n"
        "PPL ≈ 1 inside ball, rises outside  (33 conditions, 4 models, 2 methods, vary LR + steps  —  zero exceptions)",
        fontsize=14, color=LIGHT, pad=14
    )
    ax.grid(True)
    ax.tick_params(colors=MID)
    _save(fig, "fig4_phase_boundary.png")


# ── Fig 7: Method comparison — Llama-3B LoRA vs Full FT ─────────────────────
def fig_method_comparison():
    plt.rcParams.update(DARK_RC)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6.5))
    fig.subplots_adjust(top=0.84, wspace=0.32)
    for ax in (ax1, ax2):
        ax.set_facecolor(BG2)
        for sp in ax.spines.values(): sp.set_edgecolor(MID)

    color = MODEL_COLORS["Llama-3.2-3B (3B)"]

    # ── Left: density curves comparison ──────────────────────────────────────
    pre_sr = load_density_sr(
        "outputs_0607/llama3b_lr_sweep/meta-llama_Llama-3.2-3B/sst2/pretrained_density.json")
    lora_sr = load_density_sr(
        "outputs_0607/llama3b_lr_sweep/meta-llama_Llama-3.2-3B/sst2/lr1e-04_rank8_steps500_density.json")
    fullft_sr = load_density_sr(
        "outputs_0609/outputs_0609/llama3b_fullft_lr_sweep/meta-llama_Llama-3.2-3B/sst2/lr1e-05_rank768_steps500_density.json")

    for sr, ls, lbl, c, mk in [
        (pre_sr,    "-",  "Pretrained",                        LIGHT,  "o"),
        (lora_sr,   "--", "LoRA rank=8, lr=1e-4 (norm/σ½=0.22)",   color,  "s"),
        (fullft_sr, ":",  "Full FT (all params), lr=1e-5, norm/σ½=0.37", YELLOW, "^"),
    ]:
        if sr is None: continue
        s  = [r["sigma"]   for r in sr]
        d  = [r["density"] for r in sr]
        lo = [r.get("ci_lower", r["density"]) for r in sr]
        hi = [r.get("ci_upper", r["density"]) for r in sr]
        ax1.plot(s, d, ls=ls, lw=2.5, color=c, marker=mk, ms=4, label=lbl)
        ax1.fill_between(s, lo, hi, color=c, alpha=0.10)
        sh = sigma_half(sr)
        if np.isfinite(sh):
            ax1.axvline(sh, color=c, lw=1.2, ls=":", alpha=0.6)

    ax1.set_xscale("log"); ax1.grid(True); ax1.set_ylim(-0.03, 1.08)
    ax1.set_xlabel("σ", fontsize=12, color=MID)
    ax1.set_ylabel("C(θ, σ)  — certified density", fontsize=12, color=MID)
    ax1.set_title("Llama-3.2-3B — Density Curves by Method\n"
                  "Dotted verticals = σ½  |  Full FT shifts σ½ further right",
                  fontsize=12, color=LIGHT, pad=10)
    ax1.legend(fontsize=9.5, loc="upper right",
               handlelength=2.5, labelspacing=0.6)
    ax1.tick_params(colors=MID)

    # ── Right: method comparison scatter (Llama-3B only) ─────────────────────
    d_lora = json.load(open(os.path.join(ROOT,
        "outputs_0607/llama3b_lr_sweep/meta-llama_Llama-3.2-3B/sst2/widening_summary.json")))
    lora_pts = [(r["norm_over_sh"], r["widening_ratio"], r["lr"])
                for r in d_lora["results"] if r["ppl_ratio"] < 1.10]

    d_ft = json.load(open(os.path.join(ROOT,
        "outputs_0609/outputs_0609/llama3b_fullft_lr_sweep/meta-llama_Llama-3.2-3B/sst2/widening_summary.json")))
    ft_pts = [(r["norm_over_sh"], r["widening_ratio"], r["lr"])
              for r in d_ft["results"] if r["ppl_ratio"] < 1.10]

    # Alternating annotation offsets to avoid overlap
    lora_offsets = [(8, 6), (8, -14), (-50, 6), (-50, -14), (8, 6), (8, -14)]
    ft_offsets   = [(10, 8), (-52, 8), (10, -14)]

    for pts, mk, lbl, c, offsets in [
        (lora_pts, "o", "LoRA rank=8",                    color,  lora_offsets),
        (ft_pts,   "^", "LoRA rank=768  (r/d = 25%)",     YELLOW, ft_offsets),
    ]:
        if not pts: continue
        ns, rs, lrs = zip(*pts)
        ax2.scatter(ns, rs, color=c, s=150, marker=mk, zorder=5,
                    label=lbl, edgecolors=BG2, linewidths=0.8)
        for i, (n, r, lr) in enumerate(zip(ns, rs, lrs)):
            dx, dy = offsets[i % len(offsets)]
            ax2.annotate(f"{lr:.0e}", (n, r),
                         textcoords="offset points", xytext=(dx, dy),
                         fontsize=9, color=c, alpha=0.9)

    ax2.axhline(1.0, color=LIGHT, lw=1.5, ls="--", alpha=0.45, label="no change (ratio=1)")
    ax2.axvline(1.0, color=ACCENT, lw=1.5, ls="--", alpha=0.6, label="ball boundary (norm/σ½=1)")
    ax2.set_xlabel("‖ΔW‖ / σ½_pre", fontsize=12, color=MID)
    ax2.set_ylabel("σ½ widening ratio  (σ½_ft / σ½_pre)", fontsize=12, color=MID)
    ax2.set_title("Llama-3B — σ½ Widening by Method\n"
                  "High-rank LoRA (r=768) widens; LoRA rank=8 flat or narrows",
                  fontsize=12, color=LIGHT, pad=10)
    ax2.legend(fontsize=10, loc="upper right", labelspacing=0.6)
    ax2.grid(True); ax2.tick_params(colors=MID)
    ax2.set_xlim(-0.05, 2.3)
    ax2.set_ylim(0.75, 1.45)

    fig.suptitle("Same Model, Same Norm/σ½ Budget — Different Fine-Tuning Method",
                 fontsize=15, color=LIGHT)
    _save(fig, "fig7_method_comparison.png")


# ── Fig 5: Widening ratio vs LR — 4 model lines (dark) ──────────────────────
def fig_widening_vs_lr():
    plt.rcParams.update(DARK_RC)
    fig, ax = plt.subplots(figsize=(11, 6.5))
    ax.set_facecolor(BG2)
    for sp in ax.spines.values(): sp.set_edgecolor(MID)

    ax.axhline(1.0, color=LIGHT, lw=1.8, ls="--", alpha=0.5,
               label="ratio = 1  (no change)")
    ax.fill_between([5e-6, 2e-3], 0.4, 1.0,
                    color=ACCENT, alpha=0.05)
    ax.fill_between([5e-6, 2e-3], 1.0, 5.5,
                    color=GREEN, alpha=0.05)

    # shared counter so annotations from different models at the same LR stack upward
    offchart_count = {}
    for label, path in SOURCES:
        d = json.load(open(os.path.join(ROOT, path)))
        color = MODEL_COLORS[label]
        pts = sorted([(r["lr"], r["widening_ratio"], r["norm_over_sh"], r["ppl_ratio"])
                      for r in d["results"]], key=lambda x: x[0])
        lrs_l  = [p[0] for p in pts]
        rats   = [min(p[1], 5.0) for p in pts]
        raw    = [p[1] for p in pts]
        inside = [p[2] < 1.0 for p in pts]

        ax.plot(lrs_l, rats, color=color, lw=2.5, marker="o", ms=7, label=label)

        # Open markers = outside ball; stagger off-chart annotations vertically
        for lr, rat, ins, raw_r in zip(lrs_l, rats, inside, raw):
            if not ins:
                ax.plot(lr, rat, marker="o", ms=12, color=color,
                        markerfacecolor="none", markeredgewidth=2.2, zorder=8)
            if raw_r > 5.0:
                idx = offchart_count.get(lr, 0)
                offchart_count[lr] = idx + 1
                dy = 8 + idx * 16
                ax.annotate(f"→{raw_r:.0f}×",
                            xy=(lr, 5.0), xytext=(0, dy),
                            textcoords="offset points",
                            ha="center", fontsize=9, color=color, fontweight="bold")

    ax.set_xscale("log")
    ax.set_ylim(0.4, 5.8)
    ax.set_xlabel("Learning Rate", fontsize=13, color=MID)
    ax.set_ylabel("σ½ Widening Ratio  (σ½_ft / σ½_pre)", fontsize=13, color=MID)
    ax.set_title(
        "Basin Widening Ratio vs Learning Rate  (LoRA rank=8, SST-2)\n"
        "Open marker = norm/σ½ > 1 (escaped certified ball)",
        fontsize=14, color=LIGHT
    )
    ax.legend(fontsize=11, loc="upper left")
    ax.grid(True)
    ax.tick_params(colors=MID)

    # Annotate regions
    ax.text(6e-6, 5.2, "widening", fontsize=11, color=GREEN, alpha=0.7)
    ax.text(6e-6, 0.55, "narrowing", fontsize=11, color=ACCENT, alpha=0.7)
    _save(fig, "fig5_widening_vs_lr.png")


# ── Fig 6: GPT-2 full FT vs LoRA density comparison ─────────────────────────
def fig_full_ft_comparison():
    plt.rcParams.update(DARK_RC)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5.5))

    for ax in (ax1, ax2):
        ax.set_facecolor(BG2)
        for sp in ax.spines.values(): sp.set_edgecolor(MID)

    color = MODEL_COLORS["GPT-2 (124M)"]

    # LoRA rank=8 vs pretrained (LR=1e-4)
    pre_sr = load_density_sr("outputs_0606/lr_sweep/gpt2/sst2/pretrained_density.json")
    ft_sr  = load_density_sr("outputs_0606/lr_sweep/gpt2/sst2/lr1e-04_rank8_steps500_density.json")

    for sr, ls, lbl, c in [
        (pre_sr, "-",  "Pretrained", LIGHT),
        (ft_sr,  "--", "LoRA rank=8  (lr=1e-4)", color),
    ]:
        if sr:
            s = [r["sigma"] for r in sr]; d = [r["density"] for r in sr]
            lo = [r.get("ci_lower", r["density"]) for r in sr]
            hi = [r.get("ci_upper", r["density"]) for r in sr]
            ax1.plot(s, d, ls=ls, lw=2.5, color=c, marker="o", ms=5, label=lbl)
            ax1.fill_between(s, lo, hi, color=c, alpha=0.12)
            sh = sigma_half(sr)
            if np.isfinite(sh):
                ax1.axvline(sh, color=c, lw=1.2, ls=":", alpha=0.6)

    ax1.set_xscale("log"); ax1.grid(True); ax1.set_ylim(-0.03, 1.05)
    ax1.set_xlabel("σ", fontsize=12, color=MID)
    ax1.set_ylabel("C(θ, σ)", fontsize=12, color=MID)
    ax1.set_title("GPT-2: Pretrained vs LoRA rank=8\n(widens basin)", fontsize=13, color=LIGHT)
    ax1.legend(fontsize=10)
    ax1.tick_params(colors=MID)

    # Full FT (rank=768) vs pretrained
    ft_full_sr  = load_density_sr("outputs_0605/full_ft_baseline/gpt2/sst2/rank768_steps500_density.json")
    pre_full_sr = load_density_sr("outputs_0605/full_ft_baseline/gpt2/sst2/pretrained_density.json")

    for sr, ls, lbl, c in [
        (pre_full_sr or pre_sr, "-",  "Pretrained", LIGHT),
        (ft_full_sr,             "--", "Full FT (all params), lr=1e-4", YELLOW),
    ]:
        if sr:
            s = [r["sigma"] for r in sr]; d = [r["density"] for r in sr]
            lo = [r.get("ci_lower", r["density"]) for r in sr]
            hi = [r.get("ci_upper", r["density"]) for r in sr]
            ax2.plot(s, d, ls=ls, lw=2.5, color=c, marker="o", ms=5, label=lbl)
            ax2.fill_between(s, lo, hi, color=c, alpha=0.12)
            sh = sigma_half(sr)
            if np.isfinite(sh):
                ax2.axvline(sh, color=c, lw=1.2, ls=":", alpha=0.6)

    ax2.set_xscale("log"); ax2.grid(True); ax2.set_ylim(-0.03, 1.05)
    ax2.set_xlabel("σ", fontsize=12, color=MID)
    ax2.set_ylabel("C(θ, σ)", fontsize=12, color=MID)
    ax2.set_title("GPT-2: Pretrained vs Full FT (all params)\n(barely changes basin  → 1.002×)", fontsize=13, color=LIGHT)
    ax2.legend(fontsize=10)
    ax2.tick_params(colors=MID)

    fig.suptitle("Low-rank Constraint Drives Basin Widening  —  Same LR, Different Rank",
                 fontsize=14, color=LIGHT, y=1.02)
    fig.tight_layout()
    _save(fig, "fig6_full_ft_comparison.png")


# ── Fig 8: Null-space vs subspace vs isotropic (3-way density) ───────────────
def fig_nullspace_comparison():
    plt.rcParams.update(DARK_RC)
    fig, ax = plt.subplots(figsize=(11, 6.5))
    fig.subplots_adjust(top=0.87)
    ax.set_facecolor(BG2)
    for sp in ax.spines.values(): sp.set_edgecolor(MID)

    BASE = os.path.join(ROOT, "archive/outputs_0604/outputs_0604/lora_density")

    curves = [
        ("pretrained_density.json",   LIGHT,   "-",  "o", "Pretrained  (isotropic)"),
        ("lora_lr1e-04_density.json",  ACCENT2, "--", "s", "LoRA lr=1e-4  —  isotropic noise"),
        ("lora_lr1e-04_subspace.json", GREEN,   "-.", "^", "LoRA lr=1e-4  —  subspace noise  (LoRA directions)"),
        ("lora_lr1e-04_nullspace.json",YELLOW,  ":",  "D", "LoRA lr=1e-4  —  null-space noise  (complement)"),
    ]

    sh_vals = {}
    for fname, color, ls, mk, label in curves:
        sr = load_density_sr(os.path.join(BASE, fname))
        if sr is None: continue
        s  = [r["sigma"]   for r in sr]
        d  = [r["density"] for r in sr]
        lo = [r.get("ci_lower", r["density"]) for r in sr]
        hi = [r.get("ci_upper", r["density"]) for r in sr]
        ax.plot(s, d, ls=ls, lw=2.5, color=color, marker=mk, ms=5, label=label)
        ax.fill_between(s, lo, hi, color=color, alpha=0.10)
        sh = sigma_half(sr)
        if np.isfinite(sh):
            sh_vals[label] = sh
            ax.axvline(sh, color=color, lw=1.0, ls=":", alpha=0.5)

    # σ½ result annotations — stacked on right side, clear of curves
    y_pos = [0.76, 0.66, 0.56, 0.46]
    for i, (fname, color, ls, mk, label) in enumerate(curves):
        if label not in sh_vals: continue
        sh = sh_vals[label]
        pre_sh = sh_vals.get("Pretrained  (isotropic)", sh)
        ratio = sh / pre_sh
        ax.text(0.98, y_pos[i],
                "σ½=%.3e  (%.3f×)" % (sh, ratio),
                transform=ax.transAxes, ha="right", va="top",
                fontsize=10, color=color,
                bbox=dict(boxstyle="round,pad=0.2", fc=BG, ec=color, alpha=0.8, lw=0.8))

    ax.set_xscale("log")
    ax.set_xlabel("σ  (per-element noise std)", fontsize=13, color=MID)
    ax.set_ylabel("C(θ, σ)  — certified density", fontsize=13, color=MID)
    ax.set_ylim(-0.03, 1.08)
    ax.grid(True)
    ax.tick_params(colors=MID)
    ax.legend(fontsize=10, loc="lower left", labelspacing=0.7)
    ax.set_title(
        "GPT-2  —  Noise Direction Comparison  (LoRA rank=8, lr=1e-4)\n"
        "Null-space ≈ isotropic   |   Subspace slightly tighter",
        fontsize=13, color=LIGHT, pad=12
    )
    _save(fig, "fig8_nullspace_comparison.png")


# ── Fig 9: Per-model bar chart — σ½ ratio + norm overlay (all models lr=1e-4) ─
def fig_per_model_bar():
    plt.rcParams.update(DARK_RC)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6.0))
    fig.subplots_adjust(top=0.85, wspace=0.35)
    for ax in (ax1, ax2):
        ax.set_facecolor(BG2)
        for sp in ax.spines.values(): sp.set_edgecolor(MID)

    lora_data = [
        ("GPT-2\n(124M)",       "#397bbf", "outputs_0606/lr_sweep/gpt2/sst2/widening_summary.json"),
        ("Qwen2.5-3B\n(2.5B)", "#D97706", "outputs_0607/qwen3b_lr_sweep/Qwen_Qwen2.5-3B/sst2/widening_summary.json"),
        ("Llama-3.2-3B\n(3B)", "#7C3AED", "outputs_0607/llama3b_lr_sweep/meta-llama_Llama-3.2-3B/sst2/widening_summary.json"),
        ("Llama-3.1-8B\n(8B)", "#e94f37", "outputs_0607/llama8b_lr_sweep/meta-llama_Meta-Llama-3.1-8B/sst2/widening_summary.json"),
    ]

    # Also full FT at best inside-ball LR
    fullft_data = {
        "GPT-2\n(124M)":       ("outputs_0605/full_ft_baseline/gpt2/sst2/widening_summary.json", 1e-4),
        "Llama-3.2-3B\n(3B)": ("outputs_0609/outputs_0609/llama3b_fullft_lr_sweep/meta-llama_Llama-3.2-3B/sst2/widening_summary.json", 1e-5),
    }

    x = np.arange(len(lora_data))
    bar_w = 0.35

    ratios_lora, norms_lora = [], []
    ratios_ft,   norms_ft   = [], []

    for label, color, path in lora_data:
        d = json.load(open(os.path.join(ROOT, path)))
        r = next((r for r in d["results"] if abs(r["lr"]-1e-4) < 1e-10), None)
        if r:
            ratios_lora.append(r["widening_ratio"])
            norms_lora.append(r["norm_over_sh"])
        else:
            ratios_lora.append(float("nan"))
            norms_lora.append(float("nan"))

        ft_info = fullft_data.get(label)
        if ft_info:
            fpath, flr = ft_info
            fd = json.load(open(os.path.join(ROOT, fpath)))
            # GPT-2 baseline has no 'lr' key; Llama sweep does
            fr = next((r for r in fd["results"]
                       if abs(r.get("lr", flr) - flr) < flr * 0.1), None)
            ratios_ft.append(fr["widening_ratio"] if fr else float("nan"))
            norms_ft.append(fr["norm_over_sh"]    if fr else float("nan"))
        else:
            ratios_ft.append(float("nan"))
            norms_ft.append(float("nan"))

    colors = [c for _, c, _ in lora_data]
    labels = [l for l, _, _ in lora_data]

    # ── Left: widening ratio bars ──────────────────────────────────────────
    bars_lora = ax1.bar(x - bar_w/2, ratios_lora, bar_w,
                        color=colors, alpha=0.85, label="LoRA rank=8 lr=1e-4",
                        edgecolor=BG2, linewidth=1.2)
    # Full FT bars (hatched, only where data exists)
    ft_x = [i for i, v in enumerate(ratios_ft) if not np.isnan(v)]
    ft_y = [v for v in ratios_ft if not np.isnan(v)]
    ft_c = [colors[i] for i in ft_x]
    ax1.bar([x[i] + bar_w/2 for i in ft_x], ft_y, bar_w,
            color=ft_c, alpha=0.5, hatch="////",
            edgecolor=LIGHT, linewidth=0.8, label="LoRA rank=768 (best inside-ball LR)")

    ax1.axhline(1.0, color=LIGHT, lw=1.5, ls="--", alpha=0.45, label="ratio=1 (no change)")

    # Value labels on bars
    for i, v in enumerate(ratios_lora):
        if not np.isnan(v):
            ax1.text(x[i] - bar_w/2, v + 0.01, "%.3f×" % v,
                     ha="center", va="bottom", fontsize=9, color=colors[i], fontweight="bold")
    for i, v in zip(ft_x, ft_y):
        ax1.text(x[i] + bar_w/2, v + 0.01, "%.3f×" % v,
                 ha="center", va="bottom", fontsize=9, color=colors[i], alpha=0.85)

    # Mark models with no Full FT data
    for i, v in enumerate(ratios_ft):
        if np.isnan(v):
            ax1.text(x[i] + bar_w/2, 0.67, "no\nFull FT\ndata",
                     ha="center", va="bottom", fontsize=7.5, color=MID,
                     alpha=0.7, style="italic")

    ax1.set_xticks(x); ax1.set_xticklabels(labels, fontsize=10)
    ax1.set_ylabel("σ½ widening ratio  (σ½_ft / σ½_pre)", fontsize=12, color=MID)
    ax1.set_ylim(0.65, 1.35)
    ax1.set_title("Basin Widening at lr=1e-4\n"
                  "LoRA rank=8 (solid)  |  LoRA rank=768 hatched (GPT-2 + Llama-3B only)",
                  fontsize=12, color=LIGHT, pad=10)
    ax1.legend(fontsize=9, loc="upper right")
    ax1.grid(True, axis="y"); ax1.tick_params(colors=MID)

    # ── Right: norm/σ½ with ball boundary ──────────────────────────────────
    ax2.bar(x - bar_w/2, norms_lora, bar_w, color=colors, alpha=0.85,
            label="LoRA rank=8", edgecolor=BG2, linewidth=1.2)
    ft_n = [v for v in norms_ft if not np.isnan(v)]
    ax2.bar([x[i] + bar_w/2 for i in ft_x], ft_n, bar_w,
            color=ft_c, alpha=0.5, hatch="////",
            edgecolor=LIGHT, linewidth=0.8, label="LoRA rank=768 (best inside-ball LR)")

    ax2.axhline(1.0, color=ACCENT, lw=2.0, ls="--", alpha=0.8,
                label="certified ball boundary (norm/σ½=1)")

    for i, v in enumerate(norms_lora):
        if not np.isnan(v):
            ax2.text(x[i] - bar_w/2, v + 0.01, "%.2f" % v,
                     ha="center", va="bottom", fontsize=9, color=colors[i], fontweight="bold")
    for i, v in zip(ft_x, ft_n):
        ax2.text(x[i] + bar_w/2, v + 0.01, "%.2f" % v,
                 ha="center", va="bottom", fontsize=9, color=colors[i], alpha=0.85)

    ax2.set_xticks(x); ax2.set_xticklabels(labels, fontsize=10)
    ax2.set_ylabel("‖ΔW‖ / σ½_pre  (update relative to certified radius)", fontsize=12, color=MID)
    ax2.set_ylim(0, 0.85)
    ax2.set_title("All 4 Models Stay Inside Certified Ball at lr=1e-4\n"
                  "Red line = certified ball boundary (norm/σ½=1)",
                  fontsize=12, color=LIGHT, pad=10)
    # Mark models with no Full FT data on right panel too
    for i, v in enumerate(norms_ft):
        if np.isnan(v):
            ax2.text(x[i] + bar_w/2, 0.02, "no data",
                     ha="center", va="bottom", fontsize=7.5, color=MID,
                     alpha=0.7, style="italic")

    ax2.legend(fontsize=9, loc="upper right")
    ax2.grid(True, axis="y"); ax2.tick_params(colors=MID)

    fig.suptitle("Per-Model Summary at lr=1e-4  —  LoRA rank=8 and LoRA rank=768  (SST-2, 500 steps)\n"
                 "rank=768 data available for GPT-2 and Llama-3B only",
                 fontsize=13, color=LIGHT)
    _save(fig, "fig9_per_model_bar.png")


# ── Fig 10: Steps progression — norm accumulation vs training duration ─────────
def fig_steps_progression():
    plt.rcParams.update(DARK_RC)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 7.0))
    fig.subplots_adjust(top=0.79, wspace=0.36)
    for ax in (ax1, ax2):
        ax.set_facecolor(BG2)
        for sp in ax.spines.values():
            sp.set_edgecolor(MID)

    color = MODEL_COLORS["Llama-3.2-3B (3B)"]

    # Steps sweep data (lr=1e-4)
    steps_fp = os.path.join(ROOT, STEPS_SOURCE)
    traj_fp  = os.path.join(ROOT, TRAJ_SOURCE)

    if not os.path.exists(steps_fp):
        print(f"  [skip fig10] missing {STEPS_SOURCE}")
        plt.close()
        return

    d_steps = json.load(open(steps_fp))
    pts = sorted([(r["steps"], r["norm_over_sh"], r["widening_ratio"], r["ppl_ratio"])
                  for r in d_steps["results"]], key=lambda x: x[0])
    steps = [p[0] for p in pts]
    norms = [p[1] for p in pts]
    ppls  = [p[3] for p in pts]

    # Trajectory endpoint (lr=2e-4, steps=2000) — for PPL panel
    traj_n, traj_ppl = None, None
    if os.path.exists(traj_fp):
        d_traj = json.load(open(traj_fp))
        r = d_traj["results"][0]
        traj_n, traj_ppl = r["norm_over_sh"], r["ppl_ratio"]

    # lr=2e-4 trajectory at matching step counts (200, 500, 1000, 2000) from curve JSON
    traj_curve_fp = os.path.join(ROOT, TRAJ_CURVE_SOURCE)
    traj2_steps, traj2_norms = [], []
    if os.path.exists(traj_curve_fp):
        d_curve = json.load(open(traj_curve_fp))
        match_steps = set(steps)   # {200, 500, 1000, 2000}
        by_step = {e["step"]: e["norm_over_sh_pre"] for e in d_curve["trajectory"]}
        for s in sorted(match_steps):
            if s in by_step:
                traj2_steps.append(s)
                traj2_norms.append(by_step[s])

    # ── Left: norm/σ½ vs steps ────────────────────────────────────────────────
    ax1.plot(steps, norms, color=color, lw=2.5, marker="o", ms=8,
             label="lr=1e-4  (all inside ball)", zorder=5)
    ax1.fill_between(steps, 0, norms, color=color, alpha=0.07)
    for s, n in zip(steps, norms):
        ax1.annotate(f"{n:.3f}", (s, n),
                     textcoords="offset points", xytext=(0, 10),
                     fontsize=9.5, color=color, ha="center")

    # lr=2e-4 matched line — replaces lone star on left panel
    if traj2_steps:
        inside = [n < 1.0 for n in traj2_norms]
        ax1.plot(traj2_steps, traj2_norms, color=ACCENT, lw=2.5,
                 ls="--", zorder=6, label="lr=2e-4  (exits ball at step 500)")
        for i, (s, n, ins) in enumerate(zip(traj2_steps, traj2_norms, inside)):
            ax1.scatter([s], [n], s=100, marker="o", zorder=7,
                        color=ACCENT if ins else "none",
                        edgecolors=ACCENT, linewidths=2.2)
            offsets = [(0, 10), (0, -18), (0, 10), (0, 12)]
            dx, dy = offsets[i % len(offsets)]
            ax1.annotate(f"{n:.3f}", (s, n),
                         textcoords="offset points", xytext=(dx, dy),
                         fontsize=9.5, color=ACCENT, ha="center", fontweight="bold")
    elif traj_n is not None:
        # fallback: single star if trajectory curve not available
        ax1.scatter([2000], [traj_n], color=ACCENT, s=220, marker="*", zorder=7,
                    label="lr=2e-4  (outside ball)", edgecolors=LIGHT, linewidths=0.8)
        ax1.annotate(f"{traj_n:.3f}", (2000, traj_n),
                     textcoords="offset points", xytext=(0, 12),
                     fontsize=10, color=ACCENT, ha="center", fontweight="bold")

    ax1.axhline(1.0, color=ACCENT, lw=2.0, ls="--", alpha=0.85,
                label="certified ball boundary (norm/σ½=1)")
    ax1.axhspan(0, 1, color=GREEN, alpha=0.07, zorder=0)
    ax1.axhspan(1, 3.0, color=ACCENT, alpha=0.05, zorder=0)
    ax1.text(100, 0.13, "SAFE ZONE", color=GREEN, fontsize=11, fontweight="bold", alpha=0.85)
    if traj_n is not None:
        ax1.text(100, 1.18, "PPL DEGRADES", color=ACCENT, fontsize=11, fontweight="bold", alpha=0.85)

    ax1.set_xlabel("Training Steps", fontsize=13, color=MID)
    ax1.set_ylabel("‖ΔW‖ / σ½_pre  (update norm / certified radius)", fontsize=12, color=MID)
    ax1.set_title("Norm Grows Sub-linearly with Steps  (steps^~0.7, 4 pts)\n"
                  "lr=1e-4: inside ball throughout  |  lr=2e-4: exits at step 500",
                  fontsize=12, color=LIGHT, pad=10)
    ax1.legend(fontsize=10, loc="upper left", labelspacing=0.6)
    ax1.set_xlim(0, 2300)
    ax1.set_ylim(0, 2.6)
    ax1.grid(True)
    ax1.tick_params(colors=MID)

    # ── Right: PPL ratio vs steps ─────────────────────────────────────────────
    ax2.plot(steps, ppls, color=color, lw=2.5, marker="o", ms=8,
             label="lr=1e-4", zorder=5)
    # Alternate above/below so adjacent labels don't overlap
    ppl_offsets = [(8, 8), (-8, 8), (8, -14), (-8, -14)]
    for i, (s, p) in enumerate(zip(steps, ppls)):
        dx, dy = ppl_offsets[i % len(ppl_offsets)]
        ax2.annotate(f"{p:.4f}", (s, p),
                     textcoords="offset points", xytext=(dx, dy),
                     fontsize=8.5, color=color, ha="center")

    if traj_ppl is not None:
        ax2.scatter([2000], [traj_ppl], color=ACCENT, s=220, marker="*", zorder=7,
                    label=f"lr=2e-4  endpoint only  (ppl={traj_ppl:.3f})",
                    edgecolors=LIGHT, linewidths=0.8)
        ax2.annotate(f"ppl={traj_ppl:.3f}\n(step 2000 only\n— PPL not tracked\nper checkpoint)",
                     (2000, traj_ppl),
                     textcoords="offset points", xytext=(-85, 12),
                     fontsize=8.5, color=ACCENT, ha="left", fontweight="bold")

    ax2.axhline(1.0, color=LIGHT, lw=1.5, ls="--", alpha=0.45,
                label="PPL ratio = 1.0  (no change)")
    ax2.axhspan(0.99, 1.015, color=GREEN, alpha=0.06, zorder=0)

    ax2.set_xlabel("Training Steps", fontsize=13, color=MID)
    ax2.set_ylabel("PPL ratio  (ppl_ft / ppl_pre)", fontsize=12, color=MID)
    ax2.set_title("PPL Stable at lr=1e-4 Regardless of Steps\n"
                  "lr=2e-4 endpoint: +6.3%  (intermediate PPL not measured)",
                  fontsize=12, color=LIGHT, pad=10)
    ax2.legend(fontsize=10, loc="upper left", labelspacing=0.6)
    ax2.set_xlim(0, 2300)
    ax2.set_ylim(0.990, 1.080)
    ax2.grid(True)
    ax2.tick_params(colors=MID)

    fig.suptitle(
        "F2 Validated Across Training Duration — Ball Boundary Predicts PPL Safety\n"
        "Llama-3.2-3B, SST-2, LoRA rank=8  |  Budget estimate: ~2400 steps (linear) to ~3000 steps (√steps model)  —  4 pts, exponent uncertain",
        fontsize=13, color=LIGHT, y=0.97
    )
    _save(fig, "fig10_steps_progression.png")


# ── Fig 11: Training trajectory — σ½ and norm/σ½ over 20 checkpoints ─────────
def fig_trajectory_dynamic():
    plt.rcParams.update(DARK_RC)

    traj_fp = os.path.join(ROOT, TRAJ_CURVE_SOURCE)
    ws_fp   = os.path.join(ROOT, TRAJ_CURVE_WS)
    if not os.path.exists(traj_fp):
        print(f"  [skip fig11] missing {TRAJ_CURVE_SOURCE}")
        return

    with open(traj_fp) as f:
        traj = json.load(f)
    sigma_pre = traj["sigma_half_pre"]
    entries   = traj["trajectory"]

    steps     = [e["step"]           for e in entries]
    sigma_fts = [e["sigma_half"]      for e in entries]
    nsh_pre   = [e["norm_over_sh_pre"] for e in entries]
    nsh_ft    = [e["per_param_norm"] / e["sigma_half"] for e in entries]
    wr        = [s / sigma_pre for s in sigma_fts]

    ppl_ratio = None
    if os.path.exists(ws_fp):
        with open(ws_fp) as f:
            ws = json.load(f)
        ppl_ratio = ws["results"][0]["ppl_ratio"]

    color = MODEL_COLORS["Llama-3.2-3B (3B)"]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6.5))
    fig.subplots_adjust(top=0.81, wspace=0.38)
    for ax in (ax1, ax2):
        ax.set_facecolor(BG2)
        for sp in ax.spines.values():
            sp.set_edgecolor(MID)

    # ── Left: basin widening ratio at each checkpoint ────────────────────────
    ax1.axhline(1.0, color=LIGHT, lw=1.5, ls="--", alpha=0.5,
                label="ratio = 1  (no widening)")
    ax1.scatter(steps, wr, color=GREEN, s=60, zorder=6)
    ax1.plot(steps, wr, color=GREEN, lw=2.0, alpha=0.7, zorder=5,
             label="σ½_ft / σ½_pre  (widening ratio)")
    ax1.fill_between(steps, 1.0, wr, color=GREEN, alpha=0.10, zorder=0)

    ax1.annotate(f"{wr[0]:.2f}×\n(step 100)",
                 (steps[0], wr[0]), textcoords="offset points", xytext=(8, 8),
                 fontsize=10, color=GREEN, fontweight="bold")
    ax1.annotate(f"{wr[-1]:.2f}×\n(step 2000,\ncheckpoint)",
                 (steps[-1], wr[-1]), textcoords="offset points", xytext=(-60, 12),
                 fontsize=9, color=GREEN)

    ax1.set_xlabel("Training Steps", fontsize=13, color=MID)
    ax1.set_ylabel("σ½_ft / σ½_pre", fontsize=13, color=MID)
    ax1.set_title("Basin Widens at Step 100 and Stays Wide\n"
                  "All 20/20 checkpoints wider than pretrained σ½",
                  fontsize=12, color=LIGHT, pad=10)
    ax1.set_xlim(0, 2250)
    ax1.set_ylim(0.5, 2.4)
    ax1.legend(fontsize=10, loc="lower right")
    ax1.grid(True); ax1.tick_params(colors=MID)

    # ── Right: norm/σ½_ft and norm/σ½_pre over steps ─────────────────────────
    ax2.axhspan(0, 1.0, color=GREEN, alpha=0.07, zorder=0)
    ax2.axhline(1.0, color=ACCENT, lw=2.0, ls="--", alpha=0.85,
                label="ball boundary (= 1)")
    ax2.text(80, 0.08, "SAFE ZONE", color=GREEN, fontsize=11,
             fontweight="bold", alpha=0.85)

    ax2.plot(steps, nsh_pre, color=ACCENT, lw=1.6, ls="--", alpha=0.55,
             label="norm / σ½_pre  (pretrained ball)")
    ax2.plot(steps, nsh_ft,  color=ACCENT2, lw=2.5, marker="o", ms=6, zorder=5,
             label="norm / σ½_ft  (finetuned ball)")
    ax2.fill_between(steps, 0, nsh_ft, color=ACCENT2, alpha=0.07, zorder=0)

    if ppl_ratio is not None:
        ax2.scatter([steps[-1]], [nsh_ft[-1]], color=ACCENT2, s=200,
                    marker="*", zorder=7, edgecolors=LIGHT, linewidths=0.8)
        ax2.annotate(
            f"norm/σ½_ft = {nsh_ft[-1]:.2f}\nPPL ratio = {ppl_ratio:.3f}  (+{(ppl_ratio-1)*100:.1f}%)",
            (steps[-1], nsh_ft[-1]),
            textcoords="offset points", xytext=(-105, 20),
            fontsize=9.5, color=ACCENT2, fontweight="bold",
            arrowprops=dict(arrowstyle="->", color=ACCENT2, lw=1.1))

    ax2.set_xlabel("Training Steps", fontsize=13, color=MID)
    ax2.set_ylabel("norm / σ½", fontsize=13, color=MID)
    ax2.set_title("norm/σ½_ft < 1 Through Step ~1000\n"
                  "Pretrained ball crossed at step ~500; finetuned ball holds longer",
                  fontsize=12, color=LIGHT, pad=10)
    ax2.legend(fontsize=10, loc="upper left")
    ax2.set_xlim(0, 2250)
    ax2.set_ylim(0, 2.6)
    ax2.grid(True); ax2.tick_params(colors=MID)

    fig.suptitle(
        "Training Trajectory — Llama-3.2-3B, LoRA rank=8, lr=2e-4, SST-2  "
        "(checkpoint every 100 steps)\n"
        "Basin widens immediately; norm/σ½_ft self-consistent safety reference throughout training",
        fontsize=13, color=LIGHT, y=0.97
    )
    _save(fig, "fig11_trajectory_dynamic.png")


if __name__ == "__main__":
    print("Generating presentation figures...\n")
    fig_density_grid()
    fig_lr_matrix()
    fig_sigma_scaling()
    fig_phase_boundary()
    fig_widening_vs_lr()
    fig_full_ft_comparison()
    fig_method_comparison()
    fig_nullspace_comparison()
    fig_per_model_bar()
    fig_steps_progression()
    fig_trajectory_dynamic()
    print("\nAll done.")
