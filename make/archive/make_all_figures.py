"""
make_all_figures.py
===================
Comprehensive figures from all three output directories.

  outputs_0615_v2             — GPT-2 + Llama-3B, SST-2 + AG News
  outputs_0617_h1             — GPT-2 family (medium/large/xl), 5 tasks
  outputs_0616_seqForgetting  — sequential forgetting (Phase 1 failure diagnostic)

Generated figures  →  results/all_figures/
  f1_sigma_half_comparison.png     — σ½_pre across all 5 models
  f2_pretrained_density.png        — density curves: all 5 models
  f3_h1_ppl_vs_R.png               — H1: WikiText-2 PPL ratio vs R (rank=8)
  f4_h1_acc_vs_R.png               — H1: task accuracy vs R (rank=8)
  f5_h1_stability_boundary.png     — H1: R at PPL>1.01/1.05/1.10 per (model,task)
  f6_h1_widening_ratio.png         — H1: σ½_ft/σ½_pre vs R
  f7_h1_dynamics.png               — H1: σ½ + acc over training steps
  f8_rank_sweep.png                — rank sweep: GPT-2 and Llama-3B
  f9_seq_phase1_failure.png        — H2 diagnostic: Phase 1 failure
  f10_seq_trajectories.png         — H2 diagnostic: Phase 2 forgetting trajectories

Run: python make_all_figures.py
"""

import json, os, glob, math
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D

# ── Paths ─────────────────────────────────────────────────────────────────────
ROOT = os.path.dirname(os.path.abspath(__file__))
V2   = os.path.join(ROOT, "outputs_0615_v2")
H1   = os.path.join(ROOT, "outputs_0617_h1")
SEQ  = os.path.join(ROOT, "outputs_0616_seqForgetting")
OUT  = os.path.join(ROOT, "results", "all_figures")
os.makedirs(OUT, exist_ok=True)

# ── Style ─────────────────────────────────────────────────────────────────────
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

MODEL_COLOR = {
    "gpt2":                       "#1D4ED8",
    "gpt2-medium":                "#3B82F6",
    "gpt2-large":                 "#93C5FD",
    "gpt2-xl":                    "#BFDBFE",
    "meta-llama/Llama-3.2-3B":   "#DC2626",
}
MODEL_LABEL = {
    "gpt2":                       "GPT-2 (124M)",
    "gpt2-medium":                "GPT-2-med (354M)",
    "gpt2-large":                 "GPT-2-large (774M)",
    "gpt2-xl":                    "GPT-2-xl (1.5B)",
    "meta-llama/Llama-3.2-3B":   "Llama-3.2-3B (3B)",
}
TASK_MARKER = {
    "sst2": "o", "agnews": "s", "mnli": "^", "dbpedia": "D", "yahoo": "P",
}
TASK_LABEL = {
    "sst2":    "SST-2 (2-cls)",
    "agnews":  "AG News (4-cls)",
    "mnli":    "MNLI (3-cls)",
    "dbpedia": "DBPedia (14-cls)",
    "yahoo":   "Yahoo (10-cls)",
}
TASK_LS = {
    "sst2": "-", "agnews": "--", "mnli": "-.", "dbpedia": ":", "yahoo": "--",
}

RANK_FILTER = 8
C_BOUND = "#EF4444"
C_SAFE  = "#10B981"
C_PPL   = "#7C3AED"


# ── Helpers ───────────────────────────────────────────────────────────────────
def jload(path):
    with open(path) as f:
        return json.load(f)

def half_max_sigma(sr):
    dens, sigs = [r["density"] for r in sr], [r["sigma"] for r in sr]
    target = max(dens) / 2.0
    for i in range(len(dens) - 1):
        if dens[i] >= target > dens[i + 1]:
            t = (target - dens[i]) / (dens[i + 1] - dens[i])
            return sigs[i] + t * (sigs[i + 1] - sigs[i])
    return sigs[-1]

def _save(fig, name):
    p = os.path.join(OUT, name)
    fig.savefig(p, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {p}")

def _is_valid(v):
    return v is not None and not (isinstance(v, float) and math.isnan(v))


# ── H1 data: all (model, task) widening summaries ────────────────────────────
H1_SOURCES = [
    (os.path.join(V2, "basin_widening_v2", "gpt2", "sst2",
                  "widening_summary.json"),                      "gpt2",                     "sst2"),
    (os.path.join(V2, "basin_widening_v2", "gpt2", "agnews",
                  "widening_summary.json"),                      "gpt2",                     "agnews"),
    (os.path.join(V2, "basin_widening_v2", "meta-llama_Llama-3.2-3B", "sst2",
                  "widening_summary.json"),                      "meta-llama/Llama-3.2-3B",  "sst2"),
    (os.path.join(H1, "gpt2-medium", "agnews",
                  "widening_summary.json"),                      "gpt2-medium",              "agnews"),
    (os.path.join(H1, "gpt2-medium", "sst2",
                  "widening_summary.json"),                      "gpt2-medium",              "sst2"),
    (os.path.join(H1, "gpt2-medium", "mnli",
                  "widening_summary.json"),                      "gpt2-medium",              "mnli"),
    (os.path.join(H1, "gpt2-medium", "dbpedia",
                  "widening_summary.json"),                      "gpt2-medium",              "dbpedia"),
    (os.path.join(H1, "gpt2-medium", "yahoo",
                  "widening_summary.json"),                      "gpt2-medium",              "yahoo"),
    (os.path.join(H1, "gpt2-large",  "agnews",
                  "widening_summary.json"),                      "gpt2-large",               "agnews"),
    (os.path.join(H1, "gpt2-xl",     "agnews",
                  "widening_summary.json"),                      "gpt2-xl",                  "agnews"),
]

def load_h1_records():
    """Load all H1 results filtered to RANK_FILTER. Returns list of dicts."""
    records = []
    for path, model, task in H1_SOURCES:
        if not os.path.exists(path):
            print(f"  [missing] {path}")
            continue
        summ = jload(path)
        for row in summ["results"]:
            if row["rank"] != RANK_FILTER:
                continue
            ppl = row.get("ppl_ratio_actual") or row.get("ppl_ratio")
            acc = row.get("task_val_acc")
            if not _is_valid(ppl) or not _is_valid(acc):
                continue
            records.append({
                "model":    model,
                "task":     task,
                "lr":       row["lr"],
                "R":        row["norm_over_sh"],
                "norm":     row["norm"],
                "sh_pre":   row["sigma_half_pre"],
                "sh_ft":    row["sigma_half_ft"],
                "ppl":      ppl,
                "ppl_nll":  row["ppl_ratio"],
                "acc":      acc,
                "widening": row.get("widening_ratio", float("nan")),
            })
    return records


# ══════════════════════════════════════════════════════════════════════════════
# Fig 1: σ½_pre across all 5 models
# ══════════════════════════════════════════════════════════════════════════════
def fig1_sigma_half_comparison():
    PRETRAINED = [
        ("gpt2",                     os.path.join(V2, "basin_widening_v2", "gpt2", "sst2", "pretrained_density.json")),
        ("gpt2-medium",              os.path.join(H1, "gpt2-medium", "agnews", "pretrained_density.json")),
        ("gpt2-large",               os.path.join(H1, "gpt2-large",  "agnews", "pretrained_density.json")),
        ("gpt2-xl",                  os.path.join(H1, "gpt2-xl",     "agnews", "pretrained_density.json")),
        ("meta-llama/Llama-3.2-3B",  os.path.join(V2, "basin_widening_v2", "meta-llama_Llama-3.2-3B", "sst2", "pretrained_density.json")),
    ]
    PARAMS = {"gpt2": 124, "gpt2-medium": 354, "gpt2-large": 774, "gpt2-xl": 1558,
              "meta-llama/Llama-3.2-3B": 3000}

    models, sh_vals, colors, labels = [], [], [], []
    for model, path in PRETRAINED:
        if not os.path.exists(path):
            continue
        sr  = jload(path)["sigma_results"]
        sh  = half_max_sigma(sr)
        models.append(model)
        sh_vals.append(sh)
        colors.append(MODEL_COLOR[model])
        labels.append(f"{MODEL_LABEL[model]}\n({PARAMS[model]}M params)")

    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(models))
    bars = ax.bar(x, sh_vals, color=colors, alpha=0.82, width=0.55,
                  edgecolor="white", linewidth=1.2)

    for bar, sh in zip(bars, sh_vals):
        ax.text(bar.get_x() + bar.get_width() / 2, sh * 1.04,
                f"{sh:.5f}", ha="center", fontsize=9, fontweight="bold")

    # Annotate Llama vs GPT-2 family difference
    gpt2_avg = np.mean([sh_vals[i] for i, m in enumerate(models) if "Llama" not in m])
    llama_sh  = next(sh_vals[i] for i, m in enumerate(models) if "Llama" in m)
    ax.axhline(gpt2_avg, color="#6B7280", lw=1.2, ls="--", alpha=0.5,
               label=f"GPT-2 family avg = {gpt2_avg:.5f}")

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel("σ½  (half-peak perturbation scale)")
    ax.set_title(
        "σ½_pre Across Model Scales\n"
        "GPT-2 family: nearly constant despite 12× parameter scaling.  "
        "Llama-3B: 3–4× smaller (sharper pretrained minimum)",
        fontsize=11)
    ax.legend(fontsize=9)
    ax.set_ylim(0, max(sh_vals) * 1.3)
    fig.tight_layout()
    _save(fig, "f1_sigma_half_comparison.png")


# ══════════════════════════════════════════════════════════════════════════════
# Fig 2: Pretrained density curves, all 5 models
# ══════════════════════════════════════════════════════════════════════════════
def fig2_pretrained_density():
    PRETRAINED = [
        ("gpt2",                    os.path.join(V2, "basin_widening_v2", "gpt2", "sst2", "pretrained_density.json")),
        ("gpt2-medium",             os.path.join(H1, "gpt2-medium", "agnews", "pretrained_density.json")),
        ("gpt2-large",              os.path.join(H1, "gpt2-large",  "agnews", "pretrained_density.json")),
        ("gpt2-xl",                 os.path.join(H1, "gpt2-xl",     "agnews", "pretrained_density.json")),
        ("meta-llama/Llama-3.2-3B", os.path.join(V2, "basin_widening_v2", "meta-llama_Llama-3.2-3B", "sst2", "pretrained_density.json")),
    ]

    fig, ax = plt.subplots(figsize=(9, 5))

    for model, path in PRETRAINED:
        if not os.path.exists(path):
            continue
        sr   = jload(path)["sigma_results"]
        sigs = [r["sigma"]    for r in sr]
        dens = [r["density"]  for r in sr]
        lo   = [r["ci_lower"] for r in sr]
        hi   = [r["ci_upper"] for r in sr]
        sh   = half_max_sigma(sr)
        c    = MODEL_COLOR[model]
        lbl  = MODEL_LABEL[model]

        ax.plot(sigs, dens, color=c, lw=2.2, marker="o", ms=4,
                label=f"{lbl}  (σ½={sh:.5f})")
        ax.fill_between(sigs, lo, hi, color=c, alpha=0.10)
        ax.axvline(sh, color=c, lw=1.0, ls=":", alpha=0.50)

    ax.axhline(0.5, color="#9CA3AF", lw=0.8, ls="--", alpha=0.6,
               label="half-max threshold (σ½ definition)")
    ax.set_xscale("log")
    ax.set_xlabel("Perturbation scale σ")
    ax.set_ylabel("Certified density  C(θ, σ)")
    ax.set_title("Pretrained Certified Density — All Models\n"
                 "N=200 perturbations, 95% CI  |  dotted verticals = σ½ per model",
                 fontsize=11)
    ax.legend(fontsize=8.5, loc="upper right")
    ax.set_ylim(-0.03, 1.08)
    fig.tight_layout()
    _save(fig, "f2_pretrained_density.png")


# ══════════════════════════════════════════════════════════════════════════════
# Fig 3: H1 — PPL ratio vs R
# ══════════════════════════════════════════════════════════════════════════════
def fig3_h1_ppl_vs_R(records):
    combos = sorted({(r["model"], r["task"]) for r in records},
                    key=lambda x: (x[0], x[1]))

    fig, ax = plt.subplots(figsize=(11, 6))
    ax.axvspan(0, 1.0, color=C_SAFE, alpha=0.08, zorder=0)
    ax.axvspan(1.0, 100, color=C_BOUND, alpha=0.05, zorder=0)
    ax.axvline(1.0, color=C_BOUND, lw=2.0, ls="--", alpha=0.8,
               label="R = 1  (certified boundary)", zorder=5)
    ax.axhline(1.0, color="#6B7280", lw=0.8, ls=":", alpha=0.5)

    for model, task in combos:
        pts = sorted([(r["R"], r["ppl"]) for r in records
                      if r["model"] == model and r["task"] == task])
        if not pts:
            continue
        Rs, ppls = zip(*pts)
        c  = MODEL_COLOR.get(model, "#6B7280")
        mk = TASK_MARKER.get(task, "o")
        ls = TASK_LS.get(task, "-")
        short = model.split("/")[-1]
        lbl = f"{short} / {TASK_LABEL.get(task, task)}"
        ax.plot(Rs, ppls, ls, color=c, marker=mk, markersize=7,
                linewidth=1.8, label=lbl)

    ax.set_xscale("log")
    ax.set_xlabel(r"$R = \|\Delta\theta\| \,/\, \sigma_{1/2}^{\mathrm{pre}}$", fontsize=12)
    ax.set_ylabel("WikiText-2 PPL ratio  (PPL_ft / PPL_pre)", fontsize=11)
    ppl_max = max(r["ppl"] for r in records)
    ax.set_ylim(0.96, min(ppl_max * 1.05, 20.0))
    ax.set_title(
        "H1: Does σ½ Normalize the Stability Boundary?  (rank=8)\n"
        "Curves do NOT collapse — stability boundary varies 20× across tasks/models",
        fontsize=11)
    ax.legend(fontsize=8, loc="upper left", ncol=2)
    ax.text(0.3, ax.get_ylim()[0] * 1.04, "inside ball", fontsize=9,
            color="#065F46", ha="center")
    ax.text(4.0, ax.get_ylim()[0] * 1.04, "outside ball", fontsize=9,
            color="#991B1B", ha="center")
    fig.tight_layout()
    _save(fig, "f3_h1_ppl_vs_R.png")


# ══════════════════════════════════════════════════════════════════════════════
# Fig 4: H1 — Task accuracy vs R
# ══════════════════════════════════════════════════════════════════════════════
def fig4_h1_acc_vs_R(records):
    combos = sorted({(r["model"], r["task"]) for r in records},
                    key=lambda x: (x[0], x[1]))

    fig, ax = plt.subplots(figsize=(11, 6))
    ax.axvspan(0, 1.0, color=C_SAFE, alpha=0.08, zorder=0)
    ax.axvspan(1.0, 100, color=C_BOUND, alpha=0.05, zorder=0)
    ax.axvline(1.0, color=C_BOUND, lw=2.0, ls="--", alpha=0.8,
               label="R = 1  (certified boundary)", zorder=5)

    for model, task in combos:
        pts = sorted([(r["R"], r["acc"]) for r in records
                      if r["model"] == model and r["task"] == task])
        if not pts:
            continue
        Rs, accs = zip(*pts)
        c  = MODEL_COLOR.get(model, "#6B7280")
        mk = TASK_MARKER.get(task, "o")
        ls = TASK_LS.get(task, "-")
        short = model.split("/")[-1]
        lbl = f"{short} / {TASK_LABEL.get(task, task)}"
        ax.plot(Rs, accs, ls, color=c, marker=mk, markersize=7,
                linewidth=1.8, label=lbl)

    ax.set_xscale("log")
    ax.set_xlabel(r"$R = \|\Delta\theta\| \,/\, \sigma_{1/2}^{\mathrm{pre}}$", fontsize=12)
    ax.set_ylabel("Downstream task accuracy", fontsize=11)
    ax.set_ylim(0.15, 1.05)
    ax.set_title(
        "H1: Task Accuracy vs Normalized Update Radius  (rank=8)\n"
        "Different tasks require very different R to succeed — no universal R_success",
        fontsize=11)
    ax.legend(fontsize=8, loc="lower right", ncol=2)
    ax.text(0.3, 0.17, "inside ball\n(small update)", fontsize=8.5,
            color="#065F46", ha="center")
    ax.text(4.0, 0.17, "outside ball\n(large update)", fontsize=8.5,
            color="#991B1B", ha="center")
    fig.tight_layout()
    _save(fig, "f4_h1_acc_vs_R.png")


# ══════════════════════════════════════════════════════════════════════════════
# Fig 5: H1 — Stability boundary (R at PPL > threshold) per (model, task)
# ══════════════════════════════════════════════════════════════════════════════
def fig5_h1_stability_boundary(records):
    THRESHOLDS = [1.01, 1.05, 1.10]
    THRESH_COLOR = ["#FCD34D", "#FB923C", "#EF4444"]
    combos = sorted({(r["model"], r["task"]) for r in records},
                    key=lambda x: (MODEL_LABEL.get(x[0], x[0]), x[1]))

    fig, ax = plt.subplots(figsize=(13, 5))

    boundaries = {thr: [] for thr in THRESHOLDS}
    x_labels = []

    for model, task in combos:
        pts = sorted([(r["R"], r["ppl"]) for r in records
                      if r["model"] == model and r["task"] == task])
        if not pts:
            continue
        Rs, ppls = zip(*pts)
        short = model.split("/")[-1]
        x_labels.append(f"{short}\n{task}")
        for thr in THRESHOLDS:
            exceed = [R for R, p in zip(Rs, ppls) if p > thr]
            boundaries[thr].append(min(exceed) if exceed else max(Rs) * 1.3)

    x = np.arange(len(x_labels))
    w = 0.26
    offsets = [-w, 0, w]
    for thr, color, off, label in zip(
            THRESHOLDS, THRESH_COLOR, offsets,
            ["PPL > 1.01 (1%)", "PPL > 1.05 (5%)", "PPL > 1.10 (10%)"]):
        bars = ax.bar(x + off, boundaries[thr], w, color=color,
                      alpha=0.85, edgecolor="white", label=label)
        for bar, val in zip(bars, boundaries[thr]):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.05,
                    f"{val:.2f}", ha="center", fontsize=7.5, rotation=45)

    ax.axhline(1.0, color=C_BOUND, lw=1.5, ls="--", alpha=0.7,
               label="R = 1 (certified boundary)")
    ax.set_xticks(x)
    ax.set_xticklabels(x_labels, fontsize=8.5)
    ax.set_ylabel("R at which PPL ratio first exceeds threshold")
    ax.set_title(
        "H1: Stability Boundary Varies 20× Across Models and Tasks  (rank=8)\n"
        "If H1 held, all bars would cluster near R=1",
        fontsize=11)
    ax.legend(fontsize=9, loc="upper right")
    fig.tight_layout()
    _save(fig, "f5_h1_stability_boundary.png")


# ══════════════════════════════════════════════════════════════════════════════
# Fig 6: H1 — σ½ widening ratio vs R
# ══════════════════════════════════════════════════════════════════════════════
def fig6_h1_widening_ratio(records):
    combos = sorted({(r["model"], r["task"]) for r in records},
                    key=lambda x: (x[0], x[1]))

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.axhline(1.0, color="#9CA3AF", lw=1.2, ls="--", alpha=0.7,
               label="No change  (ratio = 1.0)")
    ax.axvline(1.0, color=C_BOUND, lw=1.8, ls="--", alpha=0.7,
               label="R = 1  (certified boundary)")
    ax.axvspan(0, 1.0, color=C_SAFE, alpha=0.07, zorder=0)

    for model, task in combos:
        pts = sorted(
            [(r["R"], r["widening"]) for r in records
             if r["model"] == model and r["task"] == task
             and _is_valid(r["widening"]) and r["widening"] < 10.0],
            key=lambda x: x[0])
        if not pts:
            continue
        Rs, wids = zip(*pts)
        c  = MODEL_COLOR.get(model, "#6B7280")
        mk = TASK_MARKER.get(task, "o")
        short = model.split("/")[-1]
        ax.scatter(Rs, wids, color=c, marker=mk, s=70, alpha=0.85,
                   edgecolors="white", lw=0.7, zorder=5)
        ax.plot(Rs, wids, color=c, lw=1.0, alpha=0.35, zorder=4)

    # Legend: one entry per model + one per task
    model_handles = [
        Line2D([0], [0], color=MODEL_COLOR[m], lw=2.5,
               label=MODEL_LABEL[m])
        for m in MODEL_COLOR if any(r["model"] == m for r in records)
    ]
    task_handles = [
        Line2D([0], [0], marker=TASK_MARKER[t], color="gray",
               lw=0, markersize=7, label=TASK_LABEL[t])
        for t in TASK_MARKER if any(r["task"] == t for r in records)
    ]
    ax.legend(handles=model_handles + task_handles, fontsize=8, ncol=2,
              loc="upper left")
    ax.set_xscale("log")
    ax.set_xlabel(r"$R = \|\Delta\theta\| \,/\, \sigma_{1/2}^{\mathrm{pre}}$", fontsize=12)
    ax.set_ylabel("σ½ widening ratio  (σ½_ft / σ½_pre)")
    ax.set_ylim(0.7, 5.0)
    ax.set_title(
        "H1: Does Fine-Tuning Widen the Certified Basin?  (rank=8)\n"
        "Widening is modest inside the ball (1.0–1.5×); grows for large R",
        fontsize=11)
    fig.tight_layout()
    _save(fig, "f6_h1_widening_ratio.png")


# ══════════════════════════════════════════════════════════════════════════════
# Fig 7: H1 — σ½ and accuracy over training steps (trajectory)
# ══════════════════════════════════════════════════════════════════════════════
def fig7_h1_dynamics():
    """
    Left column: step-by-step trajectory for GPT-2/SST-2 (the only run with
    per-step density tracking; other 0617_h1 runs have empty trajectory arrays).
    Right column: before/after σ½ comparison across all (model, task) at a
    representative LR (1e-4), using final values from widening_summary.
    """
    traj_path = os.path.join(V2, "basin_widening_traj_v2", "gpt2", "sst2",
                             "rank8_steps2000_trajectory.json")

    fig = plt.figure(figsize=(16, 9))
    gs  = fig.add_gridspec(2, 2, width_ratios=[1, 1.4], hspace=0.45, wspace=0.38)
    ax_top_l = fig.add_subplot(gs[0, 0])
    ax_bot_l = fig.add_subplot(gs[1, 0])
    ax_top_r = fig.add_subplot(gs[0, 1])
    ax_bot_r = fig.add_subplot(gs[1, 1])

    # ── Left: step-by-step trajectory ────────────────────────────────────────
    if os.path.exists(traj_path):
        d      = jload(traj_path)
        traj   = d["trajectory"]
        sh_pre = d["sigma_half_pre"]
        steps  = [t["step"]            for t in traj]
        sh_rat = [t["sigma_half"] / sh_pre for t in traj]
        R      = [t["norm_over_sh_pre"] for t in traj]
        acc    = [t["task_val_acc"] * 100 for t in traj]
        ppl    = [t["wikitext2_ppl"]   for t in traj]

        # σ½ ratio + R over steps
        ax_top_l.plot(steps, sh_rat, color=MODEL_COLOR["gpt2"], lw=2.2,
                      marker="o", ms=5, label="σ½_ft / σ½_pre")
        ax_top_l.axhline(1.0, color="#9CA3AF", lw=0.8, ls="--", alpha=0.6)
        ax_R = ax_top_l.twinx()
        ax_R.spines["right"].set_visible(True)
        ax_R.plot(steps, R, color="#F59E0B", lw=1.8, marker="s", ms=4,
                  ls="--", label="R = ‖Δθ‖/σ½_pre")
        ax_R.axhline(1.0, color=C_BOUND, lw=0.8, ls=":", alpha=0.5)
        ax_R.set_ylabel("R", fontsize=9, color="#F59E0B")
        ax_R.tick_params(axis="y", labelcolor="#F59E0B")
        ax_top_l.set_ylabel("σ½_ft / σ½_pre")
        ax_top_l.set_title("GPT-2 / SST-2  (lr=2e-4, rank=8)\nσ½ ratio and R over steps")
        lines1, l1 = ax_top_l.get_legend_handles_labels()
        lines2, l2 = ax_R.get_legend_handles_labels()
        ax_top_l.legend(lines1 + lines2, l1 + l2, fontsize=8, loc="upper left")

        # Accuracy + PPL over steps
        ax_bot_l.plot(steps, acc, color=MODEL_COLOR["gpt2"], lw=2.2,
                      marker="o", ms=5, label="SST-2 acc (%)")
        ax_bot_l.set_ylim(40, 102)
        ax_bot_l.set_ylabel("Task accuracy (%)", color=MODEL_COLOR["gpt2"])
        ax_bot_l.tick_params(axis="y", labelcolor=MODEL_COLOR["gpt2"])
        ax_ppl = ax_bot_l.twinx()
        ax_ppl.spines["right"].set_visible(True)
        ax_ppl.plot(steps, ppl, color=C_PPL, lw=1.8, marker="s", ms=4,
                    ls="--", label="WikiText-2 PPL")
        ax_ppl.set_ylabel("WikiText-2 PPL", fontsize=9, color=C_PPL)
        ax_ppl.tick_params(axis="y", labelcolor=C_PPL)
        ax_bot_l.set_xlabel("Training step")
        ax_bot_l.set_title("SST-2 Accuracy + WikiText-2 PPL over steps")
        lines1, l1 = ax_bot_l.get_legend_handles_labels()
        lines2, l2 = ax_ppl.get_legend_handles_labels()
        ax_bot_l.legend(lines1 + lines2, l1 + l2, fontsize=8, loc="lower right")

    # ── Right: before/after σ½ across all (model, task) at lr≈1e-4 ──────────
    BA_SOURCES = [
        (os.path.join(V2, "basin_widening_v2", "gpt2", "sst2",
                      "widening_summary.json"),                     "gpt2",                    "sst2"),
        (os.path.join(V2, "basin_widening_v2", "gpt2", "agnews",
                      "widening_summary.json"),                     "gpt2",                    "agnews"),
        (os.path.join(H1, "gpt2-medium", "agnews", "widening_summary.json"), "gpt2-medium",   "agnews"),
        (os.path.join(H1, "gpt2-medium", "sst2",   "widening_summary.json"), "gpt2-medium",   "sst2"),
        (os.path.join(H1, "gpt2-large",  "agnews", "widening_summary.json"), "gpt2-large",    "agnews"),
        (os.path.join(H1, "gpt2-xl",     "agnews", "widening_summary.json"), "gpt2-xl",       "agnews"),
        (os.path.join(V2, "basin_widening_v2", "meta-llama_Llama-3.2-3B",
                      "sst2", "widening_summary.json"),             "meta-llama/Llama-3.2-3B", "sst2"),
    ]
    TARGET_LR = 1e-4

    labels_ba, sh_pre_vals, sh_ft_vals, colors_ba = [], [], [], []
    for path, model, task in BA_SOURCES:
        if not os.path.exists(path):
            continue
        rows = jload(path)["results"]
        row  = next((r for r in rows if abs(r["lr"] - TARGET_LR) < 1e-8
                     and r["rank"] == RANK_FILTER), None)
        if row is None:
            continue
        short = model.split("/")[-1]
        labels_ba.append(f"{short}\n{task}")
        sh_pre_vals.append(row["sigma_half_pre"])
        sh_ft_vals.append(row["sigma_half_ft"])
        colors_ba.append(MODEL_COLOR.get(model, "#6B7280"))

    x_ba = np.arange(len(labels_ba))
    w_ba = 0.35

    # σ½ before/after bars
    bars_pre = ax_top_r.bar(x_ba - w_ba / 2, sh_pre_vals, w_ba,
                             color=colors_ba, alpha=0.85, edgecolor="white",
                             label="Pretrained σ½")
    bars_ft  = ax_top_r.bar(x_ba + w_ba / 2, sh_ft_vals,  w_ba,
                             color=colors_ba, alpha=0.40, edgecolor=colors_ba,
                             linewidth=1.5, label="Fine-tuned σ½  (lr=1e-4)")
    ax_top_r.set_xticks(x_ba)
    ax_top_r.set_xticklabels(labels_ba, fontsize=7.5)
    ax_top_r.set_ylabel("σ½")
    ax_top_r.set_title("σ½ Before vs After Fine-Tuning  (lr=1e-4, rank=8)\nDark=pretrained, light=fine-tuned")
    ax_top_r.legend(fontsize=8)

    # Widening ratio bars
    wid_vals = [ft / pre if pre > 0 else 1.0
                for pre, ft in zip(sh_pre_vals, sh_ft_vals)]
    bar_c_wid = [C_SAFE if w < 1.5 else C_BOUND for w in wid_vals]
    bars_w = ax_bot_r.bar(x_ba, wid_vals, 0.55, color=bar_c_wid,
                           alpha=0.80, edgecolor="white")
    ax_bot_r.axhline(1.0, color="#9CA3AF", lw=1.2, ls="--", alpha=0.7,
                      label="No change (ratio=1.0)")
    for bar, v in zip(bars_w, wid_vals):
        ax_bot_r.text(bar.get_x() + bar.get_width() / 2, v + 0.02,
                       f"{v:.2f}×", ha="center", fontsize=8.5)
    ax_bot_r.set_xticks(x_ba)
    ax_bot_r.set_xticklabels(labels_ba, fontsize=7.5)
    ax_bot_r.set_ylabel("σ½ widening ratio  (ft / pre)")
    ax_bot_r.set_title("σ½ Widening Ratio at lr=1e-4  (rank=8)\nGreen < 1.5×  (modest)  |  Red ≥ 1.5×")
    ax_bot_r.legend(fontsize=8)

    fig.suptitle(
        "Training Dynamics  |  Left: step-by-step trajectory (GPT-2/SST-2 only — other runs lack per-step tracking)"
        "\n  Right: before/after σ½ comparison across all 7 (model, task) conditions at lr=1e-4",
        y=1.01, fontsize=10)
    _save(fig, "f7_h1_dynamics.png")


# ══════════════════════════════════════════════════════════════════════════════
# Fig 8: Rank sweep — GPT-2 (AG News) and Llama-3B (SST-2)
# ══════════════════════════════════════════════════════════════════════════════
def fig8_rank_sweep():
    SWEEP_SOURCES = [
        (os.path.join(V2, "rank_sweep_v2", "gpt2", "agnews",
                      "widening_summary.json"),                      "gpt2",                    "agnews", "lr=1e-4",
         "GPT-2 / AG News rank sweep"),
        (os.path.join(V2, "basin_widening_rank_sweep_v2",
                      "meta-llama_Llama-3.2-3B", "sst2",
                      "widening_summary.json"),                      "meta-llama/Llama-3.2-3B", "sst2",   "lr=1e-4",
         "Llama-3.2-3B / SST-2 rank sweep"),
    ]
    LLAMA_D = 3072

    fig, axes = plt.subplots(2, 2, figsize=(14, 9))

    for row_idx, (path, model, task, lr_lbl, title) in enumerate(SWEEP_SOURCES):
        if not os.path.exists(path):
            continue
        summ  = jload(path)
        rows  = sorted(summ["results"], key=lambda r: r["rank"])
        ranks = [r["rank"] for r in rows]
        R_vals = [r["norm_over_sh"] for r in rows]
        accs   = [r["task_val_acc"] * 100 for r in rows]
        ppls   = [r["ppl_ratio_actual"] for r in rows]
        wids   = [r["widening_ratio"] for r in rows]
        c = MODEL_COLOR.get(model, "#6B7280")

        # Left: R and widening vs rank (bar + line)
        ax_l = axes[row_idx][0]
        bar_c = [C_SAFE if v < 1.0 else C_BOUND for v in R_vals]
        ax_l.bar(range(len(ranks)), R_vals, color=bar_c, alpha=0.75,
                 edgecolor="white")
        ax_l.axhline(1.0, color=C_BOUND, lw=1.8, ls="--",
                     label="R=1 (boundary)")
        ax2 = ax_l.twinx()
        ax2.spines["right"].set_visible(True)
        wids_plot = [w for w in wids if _is_valid(w) and w < 50]
        ax2.plot(range(len(wids_plot)), wids_plot, color="#F59E0B", lw=2.0,
                 marker="D", ms=5, label="σ½ widening ratio")
        ax2.set_ylabel("σ½_ft / σ½_pre", color="#F59E0B", fontsize=9)
        ax2.tick_params(axis="y", labelcolor="#F59E0B")
        ax_l.set_xticks(range(len(ranks)))
        ax_l.set_xticklabels([f"r={r}" for r in ranks], fontsize=8.5)
        ax_l.set_ylabel("R = ‖Δθ‖ / σ½_pre")
        ax_l.set_title(f"{title}\nR and widening vs rank")
        ax_l.legend(fontsize=8.5, loc="upper left")
        for i, v in enumerate(R_vals):
            ax_l.text(i, v + 0.03, f"{v:.2f}", ha="center", fontsize=8)

        # Right: accuracy + PPL vs rank
        ax_r = axes[row_idx][1]
        ax_r.plot(range(len(ranks)), accs, color=c, lw=2.2, marker="o", ms=7,
                  label="Task acc (%)")
        ax_r.set_ylim(30, 105)
        ax_r.set_ylabel("Task accuracy (%)", color=c, fontsize=9)
        ax_r.tick_params(axis="y", labelcolor=c)
        ax_ppl = ax_r.twinx()
        ax_ppl.spines["right"].set_visible(True)
        ppl_cap = 300.0
        ppls_plot = [min(p, ppl_cap) if _is_valid(p) else 1.0 for p in ppls]
        ax_ppl.plot(range(len(ranks)), ppls_plot, color=C_PPL, lw=1.8,
                    marker="s", ms=5, ls="--", label="PPL ratio")
        ax_ppl.axhline(1.0, color=C_PPL, lw=0.8, ls=":", alpha=0.4)
        ax_ppl.set_yscale("log")
        ax_ppl.set_ylim(0.9, ppl_cap * 3)
        ax_ppl.set_ylabel("WikiText-2 PPL ratio (log)", color=C_PPL, fontsize=9)
        ax_ppl.tick_params(axis="y", labelcolor=C_PPL)
        ax_r.set_xticks(range(len(ranks)))
        ax_r.set_xticklabels([f"r={r}" for r in ranks], fontsize=8.5)
        ax_r.set_title(f"Accuracy and PPL vs rank")
        lines_l, ll = ax_r.get_legend_handles_labels()
        lines_r, lr = ax_ppl.get_legend_handles_labels()
        ax_r.legend(lines_l + lines_r, ll + lr, fontsize=8, loc="upper left")
        # Annotate off-chart PPL
        for i, (p, pd) in enumerate(zip(ppls, ppls_plot)):
            if _is_valid(p) and p >= ppl_cap:
                ax_ppl.annotate(f"×{p/1000:.0f}k" if p > 1000 else f"×{p:.0f}",
                                xy=(i, ppl_cap), xytext=(i - 0.5, ppl_cap * 1.8),
                                fontsize=8, color=C_PPL,
                                arrowprops=dict(arrowstyle="->", color=C_PPL, lw=0.8))

    fig.suptitle("Rank Sweep: Effect of LoRA Rank on Update Norm, Widening, Accuracy, PPL\n"
                 "Green bars = inside certified ball (R<1);  Red = outside",
                 y=1.01, fontsize=11)
    fig.tight_layout()
    _save(fig, "f8_rank_sweep.png")


# ══════════════════════════════════════════════════════════════════════════════
# Fig 9: H2 diagnostic — Phase 1 failure
# ══════════════════════════════════════════════════════════════════════════════
def fig9_seq_phase1_failure():
    seq_dir = os.path.join(SEQ, "gpt2", "seq_agnews_to_sst2")
    norm_files = sorted(glob.glob(os.path.join(seq_dir, "*_norm.json")))

    if not norm_files:
        print("  [skipping f9 — no seqForgetting norm files found]")
        return

    conditions, task_b_accs, task_a_accs, R_vals, sigma_half_vals = [], [], [], [], []
    for nf in norm_files:
        d   = jload(nf)
        tag = os.path.basename(nf).replace("_norm.json", "")
        sh  = d.get("sigma_half_pre", float("nan"))
        norm = d.get("per_param_norm", float("nan"))
        R    = norm / sh if sh > 0 else float("nan")
        conditions.append(tag.replace("_steps1000", ""))
        task_b_accs.append(d.get("task_val_acc", float("nan")) * 100)
        task_a_accs.append(d.get("task_a_val_acc_final", float("nan")) * 100)
        R_vals.append(R)
        sigma_half_vals.append(sh)

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    # Panel A: Task A acc (should be ~80%+ if Phase 1 worked)
    ax = axes[0]
    colors = [C_BOUND] * len(conditions)
    bars = ax.bar(range(len(conditions)), task_a_accs, color=colors, alpha=0.75,
                  edgecolor="white")
    ax.axhline(70.0, color="#059669", lw=2.0, ls="--",
               label="70% guard threshold")
    ax.axhline(25.0, color="#9CA3AF", lw=1.0, ls=":",
               label="Random baseline (4-class)")
    ax.set_xticks(range(len(conditions)))
    ax.set_xticklabels(conditions, fontsize=7.5, rotation=30, ha="right")
    ax.set_ylabel("Task A (AG News) accuracy  (%)")
    ax.set_ylim(0, 100)
    ax.set_title("Phase 1 Task A Accuracy\n(Expected ≥70%; Got ~44% — FAILED)")
    ax.legend(fontsize=8.5)
    for bar, v in zip(bars, task_a_accs):
        ax.text(bar.get_x() + bar.get_width() / 2, v + 1.5,
                f"{v:.1f}%", ha="center", fontsize=8)

    # Panel B: σ½_A (inflated due to Phase 1 failure)
    ax = axes[1]
    sh_colors = [C_BOUND if s > 0.001 else "#3B82F6" for s in sigma_half_vals]
    bars = ax.bar(range(len(conditions)), sigma_half_vals, color=sh_colors,
                  alpha=0.75, edgecolor="white")
    ax.axhline(0.0006, color="#3B82F6", lw=2.0, ls="--",
               label="Expected σ½_A ≈ 0.0006 (normal GPT-2)")
    ax.set_xticks(range(len(conditions)))
    ax.set_xticklabels(conditions, fontsize=7.5, rotation=30, ha="right")
    ax.set_ylabel("σ½_A  (Phase 1 certified ball radius)")
    ax.set_title("Phase 1 σ½_A\n(Expected ~0.0006; Got 0.018 — 30× inflated!)")
    ax.legend(fontsize=8.5)
    for bar, v in zip(bars, sigma_half_vals):
        ax.text(bar.get_x() + bar.get_width() / 2, v * 1.02,
                f"{v:.4f}", ha="center", fontsize=8)

    # Panel C: Task B acc (SST-2 — Phase 2 still learned something)
    ax = axes[2]
    bars = ax.bar(range(len(conditions)), task_b_accs,
                  color="#3B82F6", alpha=0.75, edgecolor="white")
    ax.axhline(50.0, color="#9CA3AF", lw=1.0, ls=":",
               label="Random baseline (2-class)")
    ax.set_xticks(range(len(conditions)))
    ax.set_xticklabels(conditions, fontsize=7.5, rotation=30, ha="right")
    ax.set_ylabel("Task B (SST-2) accuracy  (%)")
    ax.set_ylim(0, 100)
    ax.set_title("Phase 2 Task B Accuracy (SST-2)\n(Phase 2 did learn, despite bad Phase 1)")
    ax.legend(fontsize=8.5)
    for bar, v in zip(bars, task_b_accs):
        ax.text(bar.get_x() + bar.get_width() / 2, v + 1.5,
                f"{v:.1f}%", ha="center", fontsize=8)

    fig.suptitle(
        "H2 Diagnostic: Phase 1 Failure  (GPT-2, AG News → SST-2)\n"
        "Phase 1 used lr=1e-5 (too small) → Task A accuracy = 43.9% → σ½_A = 0.018 (30× inflated)"
        " → R_A is meaningless",
        y=1.02, fontsize=11)
    fig.tight_layout()
    _save(fig, "f9_seq_phase1_failure.png")


# ══════════════════════════════════════════════════════════════════════════════
# Fig 10: H2 diagnostic — Phase 2 forgetting trajectories
# ══════════════════════════════════════════════════════════════════════════════
def fig10_seq_trajectories():
    seq_dir = os.path.join(SEQ, "gpt2", "seq_agnews_to_sst2")
    traj_files = sorted(glob.glob(os.path.join(seq_dir, "*_trajectory.json")))

    if not traj_files:
        print("  [skipping f10 — no seqForgetting trajectory files found]")
        return

    # Parse (lr, rank) from filename; arrange in 2 rows (rank8 / rank32) × 3 LR cols
    def parse_key(fn):
        base = os.path.basename(fn)
        parts = base.replace("_trajectory.json", "").split("_")
        lr = float(parts[0].replace("lr", "").replace("e-", "e-"))
        rank = int(parts[1].replace("rank", ""))
        return lr, rank

    # Build a sorted 2D dict: {rank: {lr: filepath}}
    from collections import defaultdict
    grid = defaultdict(dict)
    for tf in traj_files:
        try:
            lr, rank = parse_key(tf)
            grid[rank][lr] = tf
        except Exception:
            pass

    ranks_sorted = sorted(grid.keys())    # [8, 32]
    all_lrs      = sorted({lr for d in grid.values() for lr in d})  # [1e-5, 1e-4, 2e-4]
    nrows, ncols = len(ranks_sorted), len(all_lrs)

    fig, axes = plt.subplots(nrows, ncols, figsize=(15, 7), sharey="row")
    if nrows == 1:
        axes = [axes]

    for ri, rank in enumerate(ranks_sorted):
        for ci, lr in enumerate(all_lrs):
            ax = axes[ri][ci]
            tf = grid[rank].get(lr)
            if tf is None:
                ax.set_visible(False)
                continue
            d    = jload(tf)
            traj = d["trajectory"]
            steps = [t["step"]          for t in traj]
            accB  = [t["task_val_acc"] * 100   for t in traj]
            accA  = [t["task_a_val_acc"] * 100 for t in traj]
            sigma = [t["sigma_half"]           for t in traj]
            R_A   = [t["norm_over_sh_pre"]     for t in traj]
            sh_pre = d["sigma_half_pre"]   # 0.018 — inflated

            ax.plot(steps, accB, color="#3B82F6", lw=2.0, marker="o", ms=4,
                    label="Task B (SST-2)" if ri == 0 and ci == 0 else None)
            ax.plot(steps, accA, color=C_BOUND, lw=2.0, marker="s", ms=4,
                    ls="--",
                    label="Task A (AG News)" if ri == 0 and ci == 0 else None)
            ax.axhline(25.0, color="#9CA3AF", lw=0.7, ls=":", alpha=0.6)
            ax.axhline(50.0, color="#9CA3AF", lw=0.7, ls=":", alpha=0.6)
            ax.set_ylim(0, 100)
            ax.set_title(f"rank={rank}, lr={lr:.0e}", fontsize=9)
            ax.tick_params(labelsize=8)

            # Secondary: R_A (on right axis)
            ax2 = ax.twinx()
            ax2.spines["right"].set_visible(True)
            ax2.plot(steps, R_A, color="#F59E0B", lw=1.2, ls=":", alpha=0.8,
                     label="R_A (inflated)" if ri == 0 and ci == 0 else None)
            ax2.axhline(1.0, color=C_BOUND, lw=0.6, ls="--", alpha=0.4)
            ax2.set_ylim(0, max(R_A) * 2 + 0.01)
            ax2.set_ylabel("R_A", fontsize=7.5, color="#F59E0B")
            ax2.tick_params(axis="y", labelcolor="#F59E0B", labelsize=7)

            if ci == 0:
                ax.set_ylabel(f"Rank={rank}\nAccuracy (%)", fontsize=8.5)
            if ri == nrows - 1:
                ax.set_xlabel("Phase 2 step", fontsize=8.5)

    # Shared legend
    legend_elements = [
        Line2D([0], [0], color="#3B82F6", lw=2.0, marker="o", ms=5,
               label="Task B — SST-2 (being trained)"),
        Line2D([0], [0], color=C_BOUND, lw=2.0, marker="s", ms=5, ls="--",
               label="Task A — AG News (flat at ~44%: Phase 1 failed)"),
        Line2D([0], [0], color="#F59E0B", lw=1.5, ls=":",
               label="R_A = ‖Δθ‖/σ½_A  (σ½_A = 0.018 inflated — R meaningless)"),
    ]
    fig.legend(handles=legend_elements, loc="lower center", ncol=3, fontsize=9.5,
               bbox_to_anchor=(0.5, -0.04))
    fig.suptitle(
        "H2 Diagnostic: Phase 2 Forgetting Trajectories  (GPT-2, AG News→SST-2)\n"
        "Task A accuracy is flat at ~44% — not because Phase 2 caused forgetting,\n"
        "but because Phase 1 never converged (lr=1e-5 too small)",
        y=1.02, fontsize=10)
    fig.tight_layout()
    _save(fig, "f10_seq_trajectories.png")


# ── Main ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print(f"Output → {OUT}\n")

    print("Loading H1 records...")
    records = load_h1_records()
    print(f"  Loaded {len(records)} records from {len({(r['model'],r['task']) for r in records})} (model,task) combos\n")

    print("Fig 1: σ½_pre comparison (all models)...")
    fig1_sigma_half_comparison()

    print("Fig 2: Pretrained density curves...")
    fig2_pretrained_density()

    print("Fig 3: H1 — PPL ratio vs R...")
    fig3_h1_ppl_vs_R(records)

    print("Fig 4: H1 — Task accuracy vs R...")
    fig4_h1_acc_vs_R(records)

    print("Fig 5: H1 — Stability boundary per (model, task)...")
    fig5_h1_stability_boundary(records)

    print("Fig 6: H1 — Widening ratio vs R...")
    fig6_h1_widening_ratio(records)

    print("Fig 7: H1 — Training dynamics (trajectory)...")
    fig7_h1_dynamics()

    print("Fig 8: Rank sweep (GPT-2 + Llama-3B)...")
    fig8_rank_sweep()

    print("Fig 9: H2 diagnostic — Phase 1 failure...")
    fig9_seq_phase1_failure()

    print("Fig 10: H2 diagnostic — Phase 2 forgetting trajectories...")
    fig10_seq_trajectories()

    print("\nDone.")
