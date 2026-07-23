"""
plot_scale_results.py
=====================
Analyzes LR-sweep widening_summary.json files across model scales and
produces the core paper figures for the certified basin geometry story.

Figures:
  scale_A_sigma_half.png       — σ½_pre vs model (scaling law)
  scale_B_widening_vs_lr.png   — widening ratio × LR per model (4 lines)
  scale_C_norm_vs_lr.png       — norm/σ½ × LR, with ball boundary
  scale_D_ppl_vs_norm.png      — PPL ratio vs norm/σ½ (phase boundary plot)
  scale_E_summary_table.png    — printable summary table

Run:
  python plot_scale_results.py
"""

import json, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

ROOT    = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(ROOT, "results", "scale_figures")
os.makedirs(OUT_DIR, exist_ok=True)

plt.rcParams.update({
    "figure.facecolor": "white",
    "axes.facecolor":   "white",
    "axes.spines.top":  False,
    "axes.spines.right": False,
    "axes.grid":        True,
    "grid.alpha":       0.25,
    "grid.linestyle":   "--",
    "font.family":      "sans-serif",
    "font.size":        11,
    "axes.titlesize":   12,
    "axes.labelsize":   11,
    "legend.fontsize":  9,
    "legend.framealpha": 0.92,
})

# ── Data sources ──────────────────────────────────────────────────────────────
# Each entry: (display_name, params_M, color, summary_json_path)
SOURCES = [
    (
        "GPT-2 (124M)", 124, "#2563EB",
        "outputs_0606/lr_sweep/gpt2/sst2/widening_summary.json",
    ),
    (
        "Qwen2.5-3B (2.5B)", 2500, "#D97706",
        "outputs_0607/qwen3b_lr_sweep/Qwen_Qwen2.5-3B/sst2/widening_summary.json",
    ),
    (
        "Llama-3.2-3B (3B)", 3000, "#7C3AED",
        "outputs_0607/llama3b_lr_sweep/meta-llama_Llama-3.2-3B/sst2/widening_summary.json",
    ),
    (
        "Llama-3.1-8B (8B)", 8000, "#DC2626",
        "outputs_0607/llama8b_lr_sweep/meta-llama_Meta-Llama-3.1-8B/sst2/widening_summary.json",
    ),
]

# Full FT control (GPT-2 rank=768 at lr=1e-4)
FULL_FT = "outputs_0605/full_ft_baseline/gpt2/sst2/widening_summary.json"


def load(path):
    full = os.path.join(ROOT, path)
    if not os.path.exists(full):
        print(f"  [missing] {path}")
        return None
    with open(full) as f:
        return json.load(f)


def _save(fig, name):
    path = os.path.join(OUT_DIR, name)
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close()
    print(f"Saved {path}")


# ── Load ──────────────────────────────────────────────────────────────────────
records = []   # list of dicts with all per-LR data
model_meta = []  # (label, params, color, sh_pre, ppl_pre)

for label, params, color, path in SOURCES:
    d = load(path)
    if d is None:
        continue
    sh_pre = d["results"][0]["sigma_half_pre"]
    ppl_pre = d["results"][0]["ppl_pre"]
    model_meta.append((label, params, color, sh_pre, ppl_pre))
    for r in d["results"]:
        records.append({
            "label":   label,
            "params":  params,
            "color":   color,
            "lr":      r["lr"],
            "ratio":   r["widening_ratio"],
            "norm_sh": r["norm_over_sh"],
            "ppl":     r["ppl_ratio"],
            "sh_pre":  sh_pre,
            "sh_ft":   r["sigma_half_ft"],
            "inside":  r["norm_over_sh"] < 1.0,
        })

model_meta.sort(key=lambda x: x[1])  # sort by params


# ── Figure A: σ½_pre scaling law ─────────────────────────────────────────────
def fig_A():
    fig, ax = plt.subplots(figsize=(7, 4.5))

    labels = [m[0] for m in model_meta]
    params = [m[1] for m in model_meta]
    colors = [m[2] for m in model_meta]
    sh_pre = [m[3] for m in model_meta]
    ppl_pre = [m[4] for m in model_meta]

    x = np.arange(len(labels))
    bars = ax.bar(x, sh_pre, color=colors, alpha=0.85, width=0.55,
                  edgecolor="white", linewidth=1.5)

    for bar, sh, ppl in zip(bars, sh_pre, ppl_pre):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + max(sh_pre) * 0.012,
                f"σ½={sh:.5f}\n(ppl={ppl:.2f})",
                ha="center", va="bottom", fontsize=8)

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9, rotation=10, ha="right")
    ax.set_ylabel("σ½_pre  (certified basin width at pretrained θ)")
    ax.set_title(
        "Pretrained Basin Width Shrinks with Model Scale\n"
        "σ½ measures sensitivity: smaller = tighter minimum"
    )
    # Log-scale trend line
    log_p = np.log10(params)
    coeffs = np.polyfit(log_p, sh_pre, 1)
    p_range = np.linspace(min(log_p), max(log_p), 100)
    ax.plot(np.interp(10**p_range, params, x), np.polyval(coeffs, p_range),
            color="#374151", lw=1.5, ls="--", alpha=0.5, label="log-linear trend")
    ax.legend(fontsize=9)
    _save(fig, "scale_A_sigma_half.png")


# ── Figure B: Widening ratio vs LR (4 model lines) ───────────────────────────
def fig_B():
    fig, ax = plt.subplots(figsize=(9, 5.5))

    # Shade "safe zone" background (norm/σ½ < 1 for a representative model)
    # We draw a note instead since the threshold LR differs per model

    ax.axhline(1.0, color="#374151", lw=1.5, ls="--", alpha=0.6,
               label="ratio = 1 (no widening)")
    ax.axhline(1.05, color="#16A34A", lw=0.8, ls=":", alpha=0.4)

    # Collapse points (ppl_ratio > 2) → open markers
    for label, params, color, sh_pre, _ in model_meta:
        pts = [(r["lr"], r["ratio"], r["inside"], r["ppl"] > 1.5)
               for r in records if r["label"] == label]
        pts.sort(key=lambda x: x[0])

        lrs_all  = [p[0] for p in pts]
        rats_all = [p[1] for p in pts]

        # Clip absurd collapse values for readability
        rats_clipped = [min(r, 5.0) for r in rats_all]
        collapsed    = [r > 5.0 for r in rats_all]

        ax.plot(lrs_all, rats_clipped, color=color, lw=2.2, marker="o",
                ms=6, label=label)

        # Dashed connector for collapsed points (plotted at 5.0 ceiling)
        for lr, rat, clip in zip(lrs_all, rats_all, collapsed):
            if clip:
                ax.annotate(f"→{rat:.0f}×",
                            xy=(lr, 5.0), xytext=(0, 8),
                            textcoords="offset points", ha="center",
                            fontsize=7.5, color=color)

        # Open markers = outside certified ball
        for lr, rat, inside, _ in pts:
            if not inside:
                rc = min(rat, 5.0)
                ax.plot(lr, rc, marker="o", ms=9, color=color,
                        markerfacecolor="none", markeredgewidth=2.0, zorder=6)

    # Legend extras
    open_patch = mpatches.Circle((0, 0), radius=0.1, fill=False,
                                 edgecolor="gray", linewidth=1.8)
    ax.legend(
        *ax.get_legend_handles_labels(),
        title="Open marker = outside certified ball (norm/σ½ > 1)",
        fontsize=9, title_fontsize=8,
    )

    ax.set_xscale("log")
    ax.set_xlabel("Learning rate")
    ax.set_ylabel("Widening ratio  (σ½_ft / σ½_pre)")
    ax.set_ylim(0.5, 5.5)
    ax.set_title(
        "Basin Widening Ratio vs Learning Rate  (LoRA rank=8, SST-2)\n"
        "Open markers = update escaped certified ball  |  line cap at 5× for readability"
    )
    _save(fig, "scale_B_widening_vs_lr.png")


# ── Figure C: norm/σ½ vs LR ───────────────────────────────────────────────────
def fig_C():
    fig, ax = plt.subplots(figsize=(9, 5))

    ax.axhline(1.0, color="#374151", lw=2.0, ls="--", alpha=0.7,
               label="norm/σ½ = 1  (certified ball boundary)")
    ax.fill_between([1e-6, 2e-3], 0, 1, color="#D1FAE5", alpha=0.35, label="inside ball")
    ax.fill_between([1e-6, 2e-3], 1, 20, color="#FEE2E2", alpha=0.25, label="outside ball")

    for label, params, color, sh_pre, _ in model_meta:
        pts = [(r["lr"], r["norm_sh"]) for r in records if r["label"] == label]
        pts.sort()
        lrs  = [p[0] for p in pts]
        nrms = [p[1] for p in pts]
        ax.plot(lrs, nrms, color=color, lw=2.2, marker="o", ms=6, label=label)

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Learning rate")
    ax.set_ylabel("‖ΔW‖ / σ½_pre  (update norm relative to certified radius)")
    ax.set_ylim(0.03, 30)
    ax.set_title(
        "Update Size vs Certified Ball  (LoRA rank=8, SST-2)\n"
        "Green = inside certified ball; Red = model has escaped pretrained basin"
    )
    ax.legend(fontsize=9)
    _save(fig, "scale_C_norm_vs_lr.png")


# ── Figure D: PPL ratio vs norm/σ½ ───────────────────────────────────────────
def fig_D():
    fig, ax = plt.subplots(figsize=(8, 5))

    ax.axvline(1.0, color="#374151", lw=2.0, ls="--", alpha=0.7,
               label="certified ball boundary")
    ax.axhline(1.0, color="#6B7280", lw=0.8, ls=":", alpha=0.5)
    ax.fill_betweenx([0.7, 6.0], 0, 1, color="#D1FAE5", alpha=0.3)
    ax.fill_betweenx([0.7, 6.0], 1, 20, color="#FEE2E2", alpha=0.2)

    # Exclude catastrophic collapses (ppl > 5×) for readability
    for label, params, color, sh_pre, _ in model_meta:
        pts = [(r["norm_sh"], r["ppl"]) for r in records
               if r["label"] == label and r["ppl"] < 5.0]
        if not pts:
            continue
        ns, pp = zip(*pts)
        ax.scatter(ns, pp, color=color, s=70, label=label, zorder=5, alpha=0.9)

    ax.set_xscale("log")
    ax.set_xlabel("‖ΔW‖ / σ½_pre  (certified ball position)")
    ax.set_ylabel("PPL ratio  (ppl_ft / ppl_pre)")
    ax.set_xlim(0.03, 15)
    ax.set_ylim(0.97, 1.15)
    ax.set_title(
        "PPL Degradation vs Certified Ball Position\n"
        "PPL is stable when norm/σ½ < 1; degrades when update escapes the ball"
    )
    ax.legend(fontsize=9)

    # Annotate safe / unsafe regions
    ax.text(0.35, 1.135, "Safe zone\n(inside ball)", ha="center", fontsize=9,
            color="#16A34A", fontweight="bold")
    ax.text(4.0, 1.135, "Degraded zone\n(outside ball)", ha="center", fontsize=9,
            color="#DC2626", fontweight="bold")
    _save(fig, "scale_D_ppl_vs_norm.png")


# ── Print summary table ────────────────────────────────────────────────────────
def print_table():
    lrs = sorted(set(r["lr"] for r in records))
    labels = [m[0] for m in model_meta]
    print()
    print("=" * 105)
    print(f"{'Model':<22} {'LR':>7}  {'σ½_ratio':>10}  {'norm/σ½':>9}  {'PPL_ratio':>10}  {'Safe?':>6}  Verdict")
    print("=" * 105)
    for label, params, color, sh_pre, ppl_pre in model_meta:
        print(f"\n{label} (σ½_pre={sh_pre:.6f}, ppl_pre={ppl_pre:.4f})")
        for r in sorted([x for x in records if x["label"] == label], key=lambda x: x["lr"]):
            lr   = r["lr"]
            rat  = r["ratio"]
            nsh  = r["norm_sh"]
            ppl  = r["ppl"]
            ins  = "IN " if r["inside"] else "OUT"
            if ppl > 1.10:
                v = "COLLAPSE"
            elif rat > 1.05 and r["inside"]:
                v = "✓ widen (safe)"
            elif rat > 1.05:
                v = "widen (outside)"
            elif rat < 0.97:
                v = "narrowing"
            else:
                v = "~flat"
            print(f"  {'':20} {lr:>7.0e}  {rat:>10.4f}  {nsh:>9.3f}  {ppl:>10.4f}  {ins:>6}  {v}")
    print()

    # σ½ scaling summary
    print("σ½_pre scaling (pretrained certified basin width):")
    for label, params, color, sh_pre, ppl_pre in model_meta:
        print(f"  {label:<25}  σ½={sh_pre:.6f}  ppl={ppl_pre:.4f}")

    # Max safe widening per model (best ratio where norm/σ½ < 1 AND ppl < 1.05)
    print("\nBest safe widening (norm/σ½ < 1, ppl_ratio < 1.05):")
    for label, params, color, sh_pre, ppl_pre in model_meta:
        safe = [r for r in records
                if r["label"] == label and r["inside"] and r["ppl"] < 1.05]
        if safe:
            best = max(safe, key=lambda x: x["ratio"])
            print(f"  {label:<25}  {best['ratio']:.4f}×  (lr={best['lr']:.0e}, norm/σ½={best['norm_sh']:.3f})")
        else:
            print(f"  {label:<25}  — (no safe widening found)")
    print()


if __name__ == "__main__":
    print_table()
    print(f"Generating figures → {OUT_DIR}\n")
    fig_A()
    fig_B()
    fig_C()
    fig_D()
    print("\nAll done.")
