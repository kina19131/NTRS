#!/usr/bin/env python3
"""
make_batch1_figures.py

Four publication-quality figures from Batch 1 + existing H2 results.

Figures produced:
  fig1_sst2_mnli_forgetting_curve.png  -- SST-2→MNLI r8 vs r32, 10× rank gap
  fig2_t_star_overview.png             -- T* bar chart across all pairs
  fig3_t_star_raw_h1_vs_h2.png        -- Absolute displacement at T*, H1 vs H2
  fig4_mnli_agnews_rank_comparison.png -- Non-monotone rank pattern (r8/r16/r32)

Run:
  python make_batch1_figures.py [--out_dir figures/]
"""

import json, glob, os, argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyArrowPatch

# ── colour palette (colour-blind friendly) ────────────────────────────────────
C8  = "#1f77b4"   # rank 8  – blue
C16 = "#ff7f0e"   # rank 16 – orange
C32 = "#2ca02c"   # rank 32 – green
C4  = "#9467bd"   # rank 4  – purple
CGREY = "#7f7f7f"

# ── constants ─────────────────────────────────────────────────────────────────
SIGMA = {"sst2": 0.003337, "mnli": 0.002857, "agnews": 0.002756, "cola": 0.001944}
H1_RAW_LOW  = 0.22 * 0.000772   # 1.70e-4
H1_RAW_HIGH = 0.58 * 0.000772   # 4.48e-4

BASE = os.path.join(os.path.dirname(__file__), "h2_llama")


# ── helpers ───────────────────────────────────────────────────────────────────

def load_pair(task_a, task_b, rank=None):
    """Return sorted list of (lr, per_param, R_A_acc, acc_a) for a pair+rank."""
    d = os.path.join(BASE, f"{task_a}_to_{task_b}")
    meta = json.load(open(os.path.join(d, "phase1_meta.json")))
    baseline = meta["acc_phase1"]
    sigma = SIGMA[task_a]
    rows = []
    pattern = f"lr*_rank{rank}_result.json" if rank else "lr*_rank*_result.json"
    for path in glob.glob(os.path.join(d, pattern)):
        r = json.load(open(path))
        if rank is None and "rank" not in r:
            continue
        rows.append((r["lr"], r["final_per_param"],
                     r["final_per_param"] / sigma, r["final_acc_task_a"],
                     r.get("rank", rank), baseline))
    return sorted(rows, key=lambda x: (x[4], x[0]))


def t_star_bracket(rows, baseline):
    """Return (T*_lo, T*_hi) in R_A units given sorted (lr, pp, ra, acc, ...) rows."""
    threshold = 0.9 * baseline
    safe, forget = None, None
    for lr, pp, ra, acc, *_ in sorted(rows, key=lambda x: x[0]):
        if acc >= threshold:
            safe = ra
        elif forget is None:
            forget = ra
            break
    return safe, forget


# ─────────────────────────────────────────────────────────────────────────────
# FIGURE 1 — SST-2 → MNLI forgetting curve (r8 vs r32)
# ─────────────────────────────────────────────────────────────────────────────

def fig1(out_dir):
    fig, ax = plt.subplots(figsize=(6.5, 4.5))

    for rank, color, ls, label in [(8, C8, "-", "rank 8"), (32, C32, "--", "rank 32")]:
        rows = load_pair("sst2", "mnli", rank=rank)
        xs = [r[2] for r in rows]   # R_A_acc
        ys = [r[3] for r in rows]   # SST-2 acc
        ax.plot(xs, ys, color=color, ls=ls, marker="o", ms=6, label=label, zorder=3)

    # threshold line
    baseline = load_pair("sst2", "mnli", rank=8)[0][5]
    thr = 0.9 * baseline
    ax.axhline(thr, color="red", ls=":", lw=1.5, label=f"90% threshold ({thr:.3f})")

    # T* brackets
    r8_rows  = load_pair("sst2", "mnli", rank=8)
    r32_rows = load_pair("sst2", "mnli", rank=32)
    bl = r8_rows[0][5]

    lo8, hi8   = t_star_bracket(r8_rows,  bl)
    lo32, hi32 = t_star_bracket(r32_rows, bl)

    for lo, hi, color, ypos, label in [
        (lo8, hi8, C8, 0.55, r"$T^*$ r8"),
        (lo32, hi32, C32, 0.48, r"$T^*$ r32"),
    ]:
        if lo and hi:
            ax.axvspan(lo, hi, alpha=0.12, color=color)
            ax.annotate("", xy=(hi, ypos), xytext=(lo, ypos),
                        arrowprops=dict(arrowstyle="<->", color=color, lw=1.5))
            mid = 0.5 * (lo + hi)
            ax.text(mid, ypos + 0.02, label, ha="center", fontsize=9, color=color)

    # annotation: 10× gap
    if lo8 and lo32 and hi8 and hi32:
        mid8  = 0.5 * (lo8  + hi8)
        mid32 = 0.5 * (lo32 + hi32)
        ax.annotate(f"~10× gap in R_A", xy=(mid8, 0.70),
                    fontsize=9, ha="center",
                    arrowprops=dict(arrowstyle="-", lw=1, color="black"),
                    xytext=(mid8, 0.70))

    ax.set_xscale("log")
    ax.set_xlim(3e-3, 5.0)
    ax.set_ylim(0.35, 1.0)
    ax.set_xlabel(r"$R_A$ = ‖Δθ‖ / σ½$_A$ (log scale)", fontsize=11)
    ax.set_ylabel("SST-2 Phase 1 accuracy", fontsize=11)
    ax.set_title("SST-2 → MNLI: forgetting curve (V5 closed)", fontsize=12)
    ax.legend(fontsize=9, loc="lower left")
    ax.grid(True, which="both", alpha=0.25)

    # secondary annotation: rank gap
    r8_mid  = 0.5 * (lo8  + hi8)  if (lo8 and hi8)   else None
    r32_mid = 0.5 * (lo32 + hi32) if (lo32 and hi32) else None
    if r8_mid and r32_mid:
        ratio = r8_mid / r32_mid
        ax.text(0.97, 0.50,
                f"T*(r8)/T*(r32) ≈ {ratio:.0f}×\n"
                f"T*(r8) mid = {r8_mid:.2f}\n"
                f"T*(r32) mid = {r32_mid:.2f}",
                transform=ax.transAxes, ha="right", va="center",
                fontsize=8.5, bbox=dict(boxstyle="round", fc="white", alpha=0.8))

    plt.tight_layout()
    path = os.path.join(out_dir, "fig1_sst2_mnli_forgetting_curve.png")
    plt.savefig(path, dpi=180)
    plt.close()
    print(f"Saved {path}")


# ─────────────────────────────────────────────────────────────────────────────
# FIGURE 2 — T* overview: all pairs, rank 8
# ─────────────────────────────────────────────────────────────────────────────

def fig2(out_dir):
    # Data: (label, phase1_task, T*_lo, T*_hi)
    pairs_data = [
        ("MNLI→SST-2",    "mnli",    0.040, 0.082),
        ("MNLI→AGNews",   "mnli",    0.040, 0.084),
        ("AGNews→MNLI",   "agnews",  0.041, 0.083),
        ("AGNews→SST-2",  "agnews",  0.042, 0.089),
        ("CoLA→MNLI",     "cola",    0.059, 0.118),
        ("CoLA→AGNews",   "cola",    0.060, 0.126),
        ("SST-2→AGNews",  "sst2",    0.018, 0.035),
        ("SST-2→MNLI",    "sst2",    0.212, 0.851),
    ]

    task_color = {"mnli": "#4e79a7", "agnews": "#f28e2b",
                  "cola": "#59a14f", "sst2": "#e15759"}

    labels = [p[0] for p in pairs_data]
    mids   = [0.5 * (p[2] + p[3]) for p in pairs_data]
    errs_lo = [0.5 * (p[3] - p[2]) for p in pairs_data]   # symmetric half-width
    errs_hi = errs_lo
    colors = [task_color[p[1]] for p in pairs_data]

    fig, ax = plt.subplots(figsize=(8, 4.5))
    y = np.arange(len(labels))
    bars = ax.barh(y, mids, xerr=[errs_lo, errs_hi],
                   color=colors, alpha=0.75, ecolor="black", capsize=4, height=0.6)

    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=10)
    ax.set_xlabel(r"$T^*$ (R$_A$ at forgetting threshold, rank 8)", fontsize=11)
    ax.set_title("Forgetting threshold T* across all task pairs (rank 8)", fontsize=12)
    ax.axvline(0, color="black", lw=0.8)

    # legend for Phase 1 tasks
    patches = [mpatches.Patch(color=c, label=t.upper())
               for t, c in task_color.items()]
    ax.legend(handles=patches, title="Phase 1 task",
              fontsize=8.5, loc="lower right")

    # annotate the SST-2→MNLI anomaly
    sst2_mnli_idx = labels.index("SST-2→MNLI")
    ax.text(mids[sst2_mnli_idx] + errs_hi[sst2_mnli_idx] + 0.05,
            sst2_mnli_idx, "~20× anomaly", va="center", fontsize=9, color="#e15759")

    ax.grid(True, axis="x", alpha=0.3)
    plt.tight_layout()
    path = os.path.join(out_dir, "fig2_t_star_overview.png")
    plt.savefig(path, dpi=180)
    plt.close()
    print(f"Saved {path}")


# ─────────────────────────────────────────────────────────────────────────────
# FIGURE 3 — T*_raw: H1 vs H2, SST-2 as outlier (log y-axis)
# ─────────────────────────────────────────────────────────────────────────────

def fig3(out_dir):
    # T*_raw midpoints for each pair (rank 8)
    # raw = per_param at T* midpoint = 0.5*(safe_pp + forget_pp)
    cluster_data = [
        ("MNLI→SST-2",   0.5 * (1.16e-4 + 2.35e-4)),
        ("MNLI→AGNews",  0.5 * (1.15e-4 + 2.39e-4)),
        ("AGNews→MNLI",  0.5 * (1.14e-4 + 2.27e-4)),
        ("AGNews→SST-2", 0.5 * (1.16e-4 + 2.44e-4)),
        ("CoLA→MNLI",    0.5 * (1.16e-4 + 2.30e-4)),
        ("CoLA→AGNews",  0.5 * (1.17e-4 + 2.44e-4)),
        ("SST-2→AGNews", 0.5 * (6.09e-5 + 1.17e-4)),   # low outlier (r8)
        ("SST-2→MNLI",   0.5 * (7.09e-4 + 2.84e-3)),   # high outlier
    ]

    labels = [d[0] for d in cluster_data]
    vals   = [d[1] for d in cluster_data]
    colors_list = (["#4e79a7"] * 2 + ["#f28e2b"] * 2 + ["#59a14f"] * 2
                   + ["#e15759", "#e15759"])

    fig, ax = plt.subplots(figsize=(8, 4.5))
    y = np.arange(len(labels))
    ax.barh(y, vals, color=colors_list, alpha=0.75, height=0.6)
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=10)
    ax.set_xscale("log")
    ax.set_xlabel(r"$T^*_\mathrm{raw}$ = per-param ‖Δθ‖ at forgetting (rank 8)", fontsize=11)
    ax.set_title(r"Absolute displacement at $T^*$: H1 vs H2", fontsize=12)

    # H1 reference band
    ax.axvspan(H1_RAW_LOW, H1_RAW_HIGH, alpha=0.12, color="grey",
               label=f"H1 range [{H1_RAW_LOW:.1e}, {H1_RAW_HIGH:.1e}]")
    ax.axvline(H1_RAW_LOW,  color="grey", ls="--", lw=1.2)
    ax.axvline(H1_RAW_HIGH, color="grey", ls="--", lw=1.2)

    # cluster annotation
    ax.text(1.78e-4, len(labels) - 0.3,
            r"H2 cluster ≈ 1.75×10⁻⁴", fontsize=8.5, color="#4e79a7",
            ha="center", va="top")

    # SST-2→MNLI annotation
    sst2_idx = labels.index("SST-2→MNLI")
    ax.text(vals[sst2_idx] * 1.3, sst2_idx,
            "10× above cluster", fontsize=8.5, color="#e15759", va="center")

    task_color = {"mnli": "#4e79a7", "agnews": "#f28e2b",
                  "cola": "#59a14f", "sst2": "#e15759"}
    patches = [mpatches.Patch(color=c, label=t.upper())
               for t, c in task_color.items()]
    patches.append(mpatches.Patch(color="grey", alpha=0.3, label="H1 range"))
    ax.legend(handles=patches, title="Phase 1 task", fontsize=8.5, loc="lower right")

    ax.grid(True, axis="x", alpha=0.3, which="both")
    plt.tight_layout()
    path = os.path.join(out_dir, "fig3_t_star_raw_h1_vs_h2.png")
    plt.savefig(path, dpi=180)
    plt.close()
    print(f"Saved {path}")


# ─────────────────────────────────────────────────────────────────────────────
# FIGURE 4 — MNLI → AGNews rank comparison (r8 / r16 / r32)
# ─────────────────────────────────────────────────────────────────────────────

def fig4(out_dir):
    fig, ax = plt.subplots(figsize=(6.5, 4.5))

    rank_cfg = [(8, C8, "-", "rank 8"), (16, C16, "-.", "rank 16 (non-monotone)"), (32, C32, "--", "rank 32")]

    baseline = None
    for rank, color, ls, label in rank_cfg:
        rows = load_pair("mnli", "agnews", rank=rank)
        if not rows:
            continue
        if baseline is None:
            baseline = rows[0][5]
        xs = sorted([r[2] for r in rows])
        # rebuild in R_A order
        rows_sorted = sorted(rows, key=lambda r: r[2])
        xs = [r[2] for r in rows_sorted]
        ys = [r[3] for r in rows_sorted]
        ax.plot(xs, ys, color=color, ls=ls, marker="o", ms=6, label=label, zorder=3)

    if baseline:
        thr = 0.9 * baseline
        ax.axhline(thr, color="red", ls=":", lw=1.5, label=f"90% threshold ({thr:.3f})")

    # annotate T* midpoints on the x-axis
    t_star_vals = {8: (0.040, 0.084), 16: (0.058, 0.123), 32: (0.042, 0.084)}
    y_annot = 0.735
    for rank, (lo, hi), color in [(8, t_star_vals[8], C8), (16, t_star_vals[16], C16), (32, t_star_vals[32], C32)]:
        ax.axvspan(lo, hi, alpha=0.10, color=color)

    ax.set_xscale("log")
    ax.set_xlim(4e-3, 1.0)
    ax.set_ylim(0.28, 0.96)
    ax.set_xlabel(r"$R_A$ = ‖Δθ‖ / σ½$_A$ (log scale)", fontsize=11)
    ax.set_ylabel("MNLI Phase 1 accuracy", fontsize=11)
    ax.set_title("MNLI → AGNews: non-monotone rank pattern", fontsize=12)
    ax.legend(fontsize=9)
    ax.grid(True, which="both", alpha=0.25)

    # table of T* midpoints
    table_text = (
        "T* midpoints:\n"
        f"  r8:  {0.5*(0.040+0.084):.3f}\n"
        f"  r16: {0.5*(0.058+0.123):.3f}  ← highest\n"
        f"  r32: {0.5*(0.042+0.084):.3f}"
    )
    ax.text(0.03, 0.25, table_text, transform=ax.transAxes,
            fontsize=8.5, va="bottom", family="monospace",
            bbox=dict(boxstyle="round", fc="white", alpha=0.85))

    plt.tight_layout()
    path = os.path.join(out_dir, "fig4_mnli_agnews_rank_comparison.png")
    plt.savefig(path, dpi=180)
    plt.close()
    print(f"Saved {path}")


# ─────────────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir", default="figures")
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    fig1(args.out_dir)
    fig2(args.out_dir)
    fig3(args.out_dir)
    fig4(args.out_dir)
    print("Done — 4 figures written.")


if __name__ == "__main__":
    main()
