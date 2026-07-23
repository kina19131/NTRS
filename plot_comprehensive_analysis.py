"""
plot_comprehensive_analysis.py
================================
Creates the main analysis figures from all current experiment data.

Figures produced:
  figures/fig1_h1_gpt2.png         -- H1: R vs ΔNLL, KL comparison (GPT-2)
  figures/fig2_h2_cross_task.png   -- H2: cross-task R_A vs forgetting (3 pairs)
  figures/fig3_directional_vuln.png -- T* summary + directional vulnerability gap

Run:
  python plot_comprehensive_analysis.py
"""

import json
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D

os.makedirs("figures", exist_ok=True)

# ─── color scheme ───────────────────────────────────────────────────────────
LR_COLORS = {
    1e-5:  "#2196F3",   # blue
    5e-5:  "#4CAF50",   # green
    1e-4:  "#FF9800",   # orange
    2e-4:  "#F44336",   # red
    5e-4:  "#9C27B0",   # purple
    1e-3:  "#795548",   # brown
}
RANK_MARKERS = {4: "s", 8: "o", 16: "^", 32: "D"}
RANK_LABELS  = {8: "rank 8", 32: "rank 32"}
SIGMA_ACC      = 0.002857   # refined MNLI acc-based σ½
SIGMA_ACC_SST2 = 0.003337   # SST-2 acc-based σ½ (from phase1_sigma_half_acc.json)


# ══════════════════════════════════════════════════════════════════════════════
# Figure 1 — H1: R vs ΔNLL vs KL (GPT-2)
# ══════════════════════════════════════════════════════════════════════════════

def _load_h1(path):
    with open(path) as f:
        d = json.load(f)
    return {
        "lrs":        [x["lr"]                   for x in d],
        "Rs":         [x["R"]                    for x in d],
        "delta_nll":  [x["delta_nll_wiki"]       for x in d],
        "kl_wiki":    [x["kl_wiki"]              for x in d],
        "kl_div":     [x["kl_diverse"]           for x in d],
        "kl_superni": [x.get("kl_superni")       for x in d],  # None if absent (GPT-2)
        "mmlu_acc":   [x.get("mmlu_acc")         for x in d],  # None if absent (GPT-2)
        "accs":       [x["acc"]                  for x in d],
        "sigma_pre":  d[0]["sigma_half_pre"],
        "task":       d[0].get("task", "sst2"),
    }


def _h1_panel_r_vs_y(ax, d, y_key, ylabel, cfg, colors, extra_legend=None, annotate_boundary=None):
    """Shared helper: scatter R vs some y metric."""
    ys = d[y_key]
    for r, y, lr, c in zip(d["Rs"], ys, d["lrs"], colors):
        if y is None:
            continue
        ax.scatter(r, y, color=c, s=160, zorder=4,
                   edgecolors="white", linewidth=0.8, label=f"lr={lr:.0e}")
    ax.axvspan(cfg["T_lo"], cfg["T_hi"], alpha=0.15, color="red",
               label=f"T* [{cfg['T_lo']:.3f},{cfg['T_hi']:.3f}]")
    ax.axvline(1.0, color="red", lw=1.2, ls="--", alpha=0.5, label="R=1")
    ax.set_xlabel("R = ‖Δθ‖ / σ½_pre  (log scale)", fontsize=9)
    ax.set_ylabel(ylabel, fontsize=9)
    ax.set_xscale("log")
    ax.set_xlim(*cfg["xlim"])
    ax.legend(fontsize=7, framealpha=0.9)
    ax.grid(True, alpha=0.25)
    if annotate_boundary:
        ax.annotate(annotate_boundary,
                    xy=(cfg["T_hi"], ax.get_ylim()[1] * 0.6),
                    fontsize=7, color="#c0392b",
                    bbox=dict(facecolor="white", alpha=0.8, edgecolor="#c0392b",
                              boxstyle="round,pad=0.3"))
    if extra_legend:
        ax.legend(handles=extra_legend, fontsize=7.5, framealpha=0.9)


def plot_h1(
    path_gpt2="h1_kl/gpt2/summary.json",
    path_llama="h1_kl/meta-llama/meta-llama/Llama-3.2-3B/summary.json",
    out="figures/fig1_h1_comparison.png"
):
    gpt2  = _load_h1(path_gpt2)
    llama = _load_h1(path_llama)

    models = [
        ("GPT-2 (117M)\nσ½_pre=0.000603  3000 steps", gpt2,
         {"T_lo": 1.025, "T_hi": 2.626, "ylim_nll": (-0.2, 5.0), "xlim": (0.1, 30),
          "has_extended": False}),
        ("Llama-3.2-3B\nσ½_pre=0.000576  2000 steps  [SST-2 fine-tuning]", llama,
         {"T_lo": 0.316, "T_hi": 0.784, "ylim_nll": (-0.2, 10.0), "xlim": (0.03, 6),
          "has_extended": True}),
    ]

    fig, axes = plt.subplots(2, 4, figsize=(20, 10))
    fig.suptitle(
        "H1: R = ‖Δθ‖/σ½_pre as capability predictor  |  KL-diverse fails; KL-SuperNI monotone but poor threshold discrimination\n"
        "σ½_pre ≈ 0.000576–0.000603 for both models (within 4.5% despite 26× size)",
        fontsize=11, fontweight="bold"
    )

    for row, (label, d, cfg) in enumerate(models):
        colors = [LR_COLORS.get(lr, "#666") for lr in d["lrs"]]

        # ── Col 0: R vs ΔNLL (main claim) ───────────────────────────────────
        ax = axes[row][0]
        for r, dn, lr, c in zip(d["Rs"], d["delta_nll"], d["lrs"], colors):
            ax.scatter(r, dn, color=c, s=160, zorder=4,
                       edgecolors="white", linewidth=0.8, label=f"lr={lr:.0e}")
        ax.axvspan(cfg["T_lo"], cfg["T_hi"], alpha=0.15, color="red",
                   label=f"T* [{cfg['T_lo']:.3f},{cfg['T_hi']:.3f}]")
        ax.axvline(1.0, color="red", lw=1.2, ls="--", alpha=0.7, label="R=1 (certified)")
        ax.axhline(0.0, color="gray", lw=0.8, ls=":", alpha=0.6)
        ax.fill_betweenx([cfg["ylim_nll"][0], 0], 0, 1.0, color="green", alpha=0.06, zorder=0)
        ax.set_xlabel("R = ‖Δθ‖ / σ½_pre  (log scale)", fontsize=9)
        ax.set_ylabel("Δ NLL  (wiki_ft − wiki_base)", fontsize=9)
        ax.set_title(f"{label}\n✓ R is monotone predictor", fontsize=9, color="#1a5276")
        ax.set_xscale("log")
        ax.set_xlim(*cfg["xlim"])
        ax.set_ylim(*cfg["ylim_nll"])
        ax.legend(fontsize=7, framealpha=0.9)
        ax.grid(True, alpha=0.25)

        # ── Col 1: KL-diverse vs ΔNLL (fails) ───────────────────────────────
        ax = axes[row][1]
        for kw, kd, dn, lr, c in zip(d["kl_wiki"], d["kl_div"], d["delta_nll"],
                                      d["lrs"], colors):
            ax.scatter(kw, dn, color=c, s=130, marker="o", zorder=4,
                       edgecolors="white", linewidth=0.8)
            ax.scatter(kd, dn, color=c, s=130, marker="^", zorder=4,
                       edgecolors="white", linewidth=0.8, alpha=0.7)
        ax.axhline(0.0, color="gray", lw=0.8, ls=":", alpha=0.6)
        legend_els = [
            Line2D([0],[0], marker="o", color="gray", markersize=8,
                   label="KL(wiki) — tracks ΔNLL", linestyle="None"),
            Line2D([0],[0], marker="^", color="gray", markersize=8,
                   label="KL(diverse) — non-monotone ✗", linestyle="None"),
        ]
        ax.legend(handles=legend_els, fontsize=8, framealpha=0.9)
        ax.set_xlabel("KL(base ‖ fine-tuned)", fontsize=9)
        ax.set_ylabel("Δ NLL", fontsize=9)
        ax.set_title("✗ KL-diverse non-monotone:\nfails as forgetting alarm", fontsize=9, color="#922b21")
        ax.set_ylim(*cfg["ylim_nll"])
        ax.grid(True, alpha=0.25)

        # ── Col 2: R vs KL-SuperNI (monotone but poor discrimination) ────────
        ax = axes[row][2]
        if not cfg["has_extended"] or all(v is None for v in d["kl_superni"]):
            ax.set_facecolor("#f5f5f5")
            ax.text(0.5, 0.5, "Not measured\nfor GPT-2",
                    ha="center", va="center", fontsize=11, color="gray",
                    transform=ax.transAxes)
            ax.set_title("KL-SuperNI vs R\n(GPT-2: N/A)", fontsize=9, color="gray")
            ax.set_xticks([]); ax.set_yticks([])
        else:
            for r, ks, lr, c in zip(d["Rs"], d["kl_superni"], d["lrs"], colors):
                if ks is not None:
                    ax.scatter(r, ks, color=c, s=160, zorder=4,
                               edgecolors="white", linewidth=0.8, label=f"lr={lr:.0e}")
            ax.axvspan(cfg["T_lo"], cfg["T_hi"], alpha=0.15, color="red",
                       label=f"T* zone")
            ax.axvline(1.0, color="red", lw=1.0, ls="--", alpha=0.5)
            # Annotate the poor discrimination at the boundary
            ax.annotate("ΔNLL jumps 32× here\nKL-SuperNI: +15% only",
                        xy=(cfg["T_hi"], 3.0), fontsize=7.5, color="#c0392b",
                        bbox=dict(facecolor="white", alpha=0.85, edgecolor="#c0392b",
                                  boxstyle="round,pad=0.3"))
            ax.set_xlabel("R = ‖Δθ‖ / σ½_pre  (log scale)", fontsize=9)
            ax.set_ylabel("KL-SuperNI (instruction prompts)", fontsize=9)
            ax.set_title("⚠ KL-SuperNI: monotone but\npoor threshold discrimination", fontsize=9, color="#7d6608")
            ax.set_xscale("log")
            ax.set_xlim(*cfg["xlim"])
            ax.legend(fontsize=7, framealpha=0.9)
            ax.grid(True, alpha=0.25)

        # ── Col 3: R vs MMLU (drops at same threshold as NLL) ────────────────
        ax = axes[row][3]
        if not cfg["has_extended"] or all(v is None for v in d["mmlu_acc"]):
            ax.set_facecolor("#f5f5f5")
            ax.text(0.5, 0.5, "Not measured\nfor GPT-2",
                    ha="center", va="center", fontsize=11, color="gray",
                    transform=ax.transAxes)
            ax.set_title("R vs MMLU accuracy\n(GPT-2: N/A)", fontsize=9, color="gray")
            ax.set_xticks([]); ax.set_yticks([])
        else:
            for r, mm, lr, c in zip(d["Rs"], d["mmlu_acc"], d["lrs"], colors):
                if mm is not None:
                    ax.scatter(r, mm, color=c, s=160, zorder=4,
                               edgecolors="white", linewidth=0.8, label=f"lr={lr:.0e}")
            ax.axvspan(cfg["T_lo"], cfg["T_hi"], alpha=0.15, color="red",
                       label=f"T* zone")
            ax.axvline(1.0, color="red", lw=1.0, ls="--", alpha=0.5)
            ax.set_xlabel("R = ‖Δθ‖ / σ½_pre  (log scale)", fontsize=9)
            ax.set_ylabel("MMLU zero-shot accuracy", fontsize=9)
            ax.set_title("✓ MMLU drops at same T*\n(factual knowledge degradation)", fontsize=9, color="#1a5276")
            ax.set_xscale("log")
            ax.set_xlim(*cfg["xlim"])
            ax.set_ylim(0.0, 0.75)
            ax.legend(fontsize=7, framealpha=0.9)
            ax.grid(True, alpha=0.25)

    fig.text(
        0.5, 0.01,
        "σ½_pre (GPT-2) = 0.000603   |   σ½_pre (Llama-3.2-3B) = 0.000576   →   within 4.5% despite 26× model size\n"
        "KL-diverse: non-monotone for both models.  "
        "KL-SuperNI: monotone but 15% change where ΔNLL jumps 32× — cannot detect threshold.  "
        "MMLU: confirms R predicts knowledge degradation, not just perplexity.",
        ha="center", fontsize=8.5, style="italic", color="#333",
        bbox=dict(facecolor="#fff9e6", alpha=0.9, edgecolor="#e0c060", boxstyle="round,pad=0.4")
    )

    plt.tight_layout(rect=[0, 0.08, 1, 1])
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out}")


def plot_h1_mnli(
    path_sst2="h1_kl/meta-llama/meta-llama/Llama-3.2-3B/summary.json",
    path_mnli="h1_kl/meta-llama/meta-llama/Llama-3.2-3B_mnli/summary.json",
    out="figures/fig1_h1_kl_v6.png"
):
    """V6 closure figure: KL-diverse fails for BOTH SST-2 and MNLI fine-tuning."""
    sst2 = _load_h1(path_sst2)
    mnli = _load_h1(path_mnli)

    cfgs = {
        "sst2": {"T_lo": 0.316, "T_hi": 0.784, "ylim_nll": (-0.2, 10.0), "xlim": (0.03, 6),
                 "label": "Llama fine-tuned on SST-2"},
        "mnli": {"T_lo": 0.312, "T_hi": 0.711, "ylim_nll": (-0.5, 9.0), "xlim": (0.03, 6),
                 "label": "Llama fine-tuned on MNLI"},
    }

    fig, axes = plt.subplots(2, 4, figsize=(20, 10))
    fig.suptitle(
        "V6 Closure — KL-diverse failure is NOT SST-2-specific: same pattern for MNLI fine-tuning\n"
        "Llama-3.2-3B  |  Both rows: KL-diverse non-monotone, KL-SuperNI poor threshold discrimination",
        fontsize=11, fontweight="bold"
    )

    for row, (task, d) in enumerate([("sst2", sst2), ("mnli", mnli)]):
        cfg = cfgs[task]
        colors = [LR_COLORS.get(lr, "#666") for lr in d["lrs"]]
        task_label = cfg["label"]

        # Col 0: R vs ΔNLL
        ax = axes[row][0]
        for r, dn, lr, c in zip(d["Rs"], d["delta_nll"], d["lrs"], colors):
            ax.scatter(r, dn, color=c, s=160, zorder=4,
                       edgecolors="white", linewidth=0.8, label=f"lr={lr:.0e}")
        ax.axvspan(cfg["T_lo"], cfg["T_hi"], alpha=0.15, color="red",
                   label=f"T* [{cfg['T_lo']:.3f},{cfg['T_hi']:.3f}]")
        ax.axvline(1.0, color="red", lw=1.0, ls="--", alpha=0.5)
        ax.axhline(0.0, color="gray", lw=0.8, ls=":", alpha=0.6)
        ax.set_xlabel("R = ‖Δθ‖ / σ½_pre  (log scale)", fontsize=9)
        ax.set_ylabel("Δ NLL  (wiki_ft − wiki_base)", fontsize=9)
        ax.set_title(f"{task_label}\n✓ R is monotone predictor", fontsize=9, color="#1a5276")
        ax.set_xscale("log"); ax.set_xlim(*cfg["xlim"]); ax.set_ylim(*cfg["ylim_nll"])
        ax.legend(fontsize=7, framealpha=0.9); ax.grid(True, alpha=0.25)

        # Col 1: KL-diverse vs ΔNLL
        ax = axes[row][1]
        for kw, kd, dn, lr, c in zip(d["kl_wiki"], d["kl_div"], d["delta_nll"],
                                      d["lrs"], colors):
            ax.scatter(kw, dn, color=c, s=130, marker="o", zorder=4,
                       edgecolors="white", linewidth=0.8)
            ax.scatter(kd, dn, color=c, s=130, marker="^", zorder=4,
                       edgecolors="white", linewidth=0.8, alpha=0.7)
        ax.axhline(0.0, color="gray", lw=0.8, ls=":", alpha=0.6)
        legend_els = [
            Line2D([0],[0], marker="o", color="gray", markersize=8,
                   label="KL(wiki) — tracks ΔNLL", linestyle="None"),
            Line2D([0],[0], marker="^", color="gray", markersize=8,
                   label="KL(diverse) — non-monotone ✗", linestyle="None"),
        ]
        ax.legend(handles=legend_els, fontsize=8, framealpha=0.9)
        ax.set_xlabel("KL(base ‖ fine-tuned)", fontsize=9)
        ax.set_ylabel("Δ NLL", fontsize=9)
        ax.set_title("✗ KL-diverse non-monotone\n(confirmed for this task too)", fontsize=9, color="#922b21")
        ax.set_ylim(*cfg["ylim_nll"]); ax.grid(True, alpha=0.25)

        # Col 2: R vs KL-SuperNI
        ax = axes[row][2]
        for r, ks, lr, c in zip(d["Rs"], d["kl_superni"], d["lrs"], colors):
            if ks is not None:
                ax.scatter(r, ks, color=c, s=160, zorder=4,
                           edgecolors="white", linewidth=0.8, label=f"lr={lr:.0e}")
        ax.axvspan(cfg["T_lo"], cfg["T_hi"], alpha=0.15, color="red", label="T* zone")
        ax.axvline(1.0, color="red", lw=1.0, ls="--", alpha=0.5)
        if task == "sst2":
            annot = "ΔNLL +32× at boundary\nKL-SuperNI: +15% only"
        else:
            annot = "ΔNLL +198× at boundary\nKL-SuperNI: +3× only"
        ax.annotate(annot, xy=(cfg["T_hi"], ax.get_ylim()[1] * 0.5 if ax.get_ylim()[1] > 1 else 5),
                    fontsize=7.5, color="#7d6608",
                    bbox=dict(facecolor="white", alpha=0.85, edgecolor="#7d6608",
                              boxstyle="round,pad=0.3"))
        ax.set_xlabel("R = ‖Δθ‖ / σ½_pre  (log scale)", fontsize=9)
        ax.set_ylabel("KL-SuperNI (instruction prompts)", fontsize=9)
        ax.set_title("⚠ KL-SuperNI: mostly monotone\nbut poor threshold discrimination", fontsize=9, color="#7d6608")
        ax.set_xscale("log"); ax.set_xlim(*cfg["xlim"])
        ax.legend(fontsize=7, framealpha=0.9); ax.grid(True, alpha=0.25)

        # Col 3: R vs MMLU
        ax = axes[row][3]
        for r, mm, lr, c in zip(d["Rs"], d["mmlu_acc"], d["lrs"], colors):
            if mm is not None:
                ax.scatter(r, mm, color=c, s=160, zorder=4,
                           edgecolors="white", linewidth=0.8, label=f"lr={lr:.0e}")
        ax.axvspan(cfg["T_lo"], cfg["T_hi"], alpha=0.15, color="red", label="T* zone")
        ax.axvline(1.0, color="red", lw=1.0, ls="--", alpha=0.5)
        ax.set_xlabel("R = ‖Δθ‖ / σ½_pre  (log scale)", fontsize=9)
        ax.set_ylabel("MMLU zero-shot accuracy", fontsize=9)
        ax.set_title("✓ MMLU drops at same T*\n(factual knowledge)", fontsize=9, color="#1a5276")
        ax.set_xscale("log"); ax.set_xlim(*cfg["xlim"]); ax.set_ylim(0.0, 0.75)
        ax.legend(fontsize=7, framealpha=0.9); ax.grid(True, alpha=0.25)

    fig.text(
        0.5, 0.01,
        "V6 CLOSED: KL-diverse fails for BOTH SST-2 and MNLI fine-tuning — failure is a property of KL on instruction prompts, not a quirk of SST-2.\n"
        "KL-SuperNI is monotone but discriminates the safe/forget boundary poorly (ΔNLL jumps 32–198× where KL-SuperNI changes 15–3×).",
        ha="center", fontsize=8.5, style="italic", color="#333",
        bbox=dict(facecolor="#fff9e6", alpha=0.9, edgecolor="#e0c060", boxstyle="round,pad=0.4")
    )

    plt.tight_layout(rect=[0, 0.07, 1, 1])
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out}")


# ══════════════════════════════════════════════════════════════════════════════
# Figure 2 — H2: Cross-task-pair comparison
# ══════════════════════════════════════════════════════════════════════════════

def load_h2(path):
    with open(path) as f:
        return json.load(f)

def load_h2_dir(dir_path):
    """Load individual lr*_rank*_result.json files from a directory (no summary.json)."""
    import glob
    results = []
    for fpath in sorted(glob.glob(os.path.join(dir_path, "lr*_rank*_result.json"))):
        with open(fpath) as f:
            results.append(json.load(f))
    return results

def _baseline_acc(data):
    """MNLI pre-ft baseline = trajectory step 0 acc_task_a."""
    return data[0]["trajectory"][0]["acc_task_a"]

def plot_h2_cross_task(
    path_a="h2_llama/mnli_to_sst2/summary.json",
    path_b="h2_llama/mnli_to_agnews/summary.json",
    path_c="h2_llama/sst2_to_mnli/summary.json",
    path_d="h2_llama/sst2_to_cola",
    out="figures/fig2_h2_cross_task.png"
):
    da = load_h2(path_a)
    db = load_h2(path_b)
    dc = load_h2(path_c)
    dd = load_h2_dir(path_d)

    # Compute acc-based R_A for H2-A and H2-B
    def extract(data, sigma, use_nll_r=False):
        rows = []
        for d in data:
            if use_nll_r:
                R = d["final_R_A"]
            else:
                R = d["final_per_param"] / sigma
            rows.append({
                "lr":    d["lr"],
                "rank":  d["rank"],
                "R":     R,
                "acc_A": d["final_acc_task_a"],
                "per_p": d["final_per_param"],
            })
        return rows

    rows_a = extract(da, SIGMA_ACC)
    rows_b = extract(db, SIGMA_ACC)
    rows_c = extract(dc, SIGMA_ACC_SST2)  # acc-based σ½ now valid for SST-2
    rows_d = extract(dd, SIGMA_ACC_SST2)  # sst2_to_cola — same Phase 1 σ½

    base_mnli  = _baseline_acc(da)   # 0.884 for MNLI
    base_sst2  = _baseline_acc(dc)   # 0.960 for SST-2
    base_sst2d = _baseline_acc(dd)   # 0.960 for SST-2 (sst2_to_cola Phase 1)

    # Determine T* bands (last "safe" → first "catastrophic")
    thresh_mnli = base_mnli * 0.88  # 88% retention = ~10% absolute drop
    thresh_sst2 = base_sst2 * 0.88

    def find_t_star(rows, thresh):
        safe_max, cat_min = None, None
        for r in sorted(rows, key=lambda x: x["R"]):
            if r["acc_A"] >= thresh:
                safe_max = r["R"]
            else:
                if cat_min is None:
                    cat_min = r["R"]
        return safe_max, cat_min

    t_a = find_t_star(rows_a, thresh_mnli)
    t_b = find_t_star(rows_b, thresh_mnli)
    t_c = find_t_star(rows_c, thresh_sst2)
    t_d = find_t_star(rows_d, thresh_sst2)

    fig, axes = plt.subplots(1, 4, figsize=(22, 5), sharey=False)
    fig.suptitle("H2 (Llama-3.2-3B): R_A as pre-run predictor of sequential fine-tuning forgetting",
                 fontsize=12, fontweight="bold")

    datasets = [
        (axes[0], rows_a, base_mnli, t_a, "MNLI → SST-2\n(Phase 1 = MNLI, R_A uses acc-based σ½)",
         "MNLI Retention (Phase 1 accuracy)"),
        (axes[1], rows_b, base_mnli, t_b, "MNLI → AGNews\n(Phase 1 = MNLI, R_A uses acc-based σ½)",
         "MNLI Retention (Phase 1 accuracy)"),
        (axes[2], rows_c, base_sst2, t_c,
         "SST-2 → MNLI\n(Phase 1 = SST-2, acc-based σ½=0.003337)\n⚠ rank-invariance breaks",
         "SST-2 Retention (Phase 1 accuracy)"),
        (axes[3], rows_d, base_sst2d, t_d,
         "SST-2 → CoLA\n(Phase 1 = SST-2, acc-based σ½=0.003337)\n⚠ extreme fragility: r32 forgets at all LRs",
         "SST-2 Retention (Phase 1 accuracy)"),
    ]

    for ax, rows, baseline, t_band, title, ylabel in datasets:
        # T* band
        if t_band[0] is not None and t_band[1] is not None:
            ax.axvspan(t_band[0], t_band[1], alpha=0.18, color="red",
                       label=f"T* zone [{t_band[0]:.3f}, {t_band[1]:.3f}]")

        # Plot points
        for row in rows:
            c = LR_COLORS.get(row["lr"], "#888888")
            m = RANK_MARKERS.get(row["rank"], "x")
            ax.scatter(row["R"], row["acc_A"], color=c, marker=m,
                       s=150, zorder=4, edgecolors="white", linewidth=0.8)

        # Reference lines
        ax.axhline(baseline, color="gray", lw=1.2, ls="--", alpha=0.7,
                   label=f"Baseline ({baseline:.3f})")
        ax.axhline(baseline * 0.88, color="black", lw=0.8, ls=":",
                   alpha=0.5, label="88% retention threshold")
        ax.axvline(1.0, color="darkred", lw=1.2, ls="-.", alpha=0.5,
                   label="R_A = 1 (theory)")

        ax.set_xscale("log")
        ax.set_xlabel("R_A = ‖Δθ‖ / σ½_A", fontsize=11)
        ax.set_ylabel(ylabel, fontsize=10)
        ax.set_title(title, fontsize=10, pad=6)
        ax.set_ylim(0.28, 1.0)
        ax.grid(True, alpha=0.25)

    # Shared legend for LR/rank
    lr_patches = [mpatches.Patch(color=c, label=f"lr={lr:.0e}")
                  for lr, c in LR_COLORS.items()]
    rank_handles = [
        Line2D([0],[0], marker="o", color="gray", markersize=8,
               linestyle="None", label="rank 8"),
        Line2D([0],[0], marker="D", color="gray", markersize=8,
               linestyle="None", label="rank 32"),
    ]
    all_handles = lr_patches + rank_handles
    fig.legend(handles=all_handles, loc="lower center", ncol=8,
               fontsize=9, framealpha=0.9, bbox_to_anchor=(0.5, -0.01))

    # Add T* annotation text
    t_a_mid = (t_a[0] * t_a[1]) ** 0.5 if (t_a[0] and t_a[1]) else None
    t_b_mid = (t_b[0] * t_b[1]) ** 0.5 if (t_b[0] and t_b[1]) else None
    t_c_mid = (t_c[0] * t_c[1]) ** 0.5 if (t_c[0] and t_c[1]) else None
    t_d_mid = (t_d[0] * t_d[1]) ** 0.5 if (t_d[0] and t_d[1]) else None

    note = (f"T* zone midpoints:  MNLI→SST-2: R_A≈{t_a_mid:.3f}  |  "
            f"MNLI→AGNews: R_A≈{t_b_mid:.3f}  |  SST-2→MNLI: R_A≈{t_c_mid:.3f}  |  "
            f"SST-2→CoLA: R_A≈{t_d_mid:.4f}  (44× range across SST-2 Phase 2 tasks)\n"
            "Rank-invariance: rank 8 and rank 32 cluster by LR, not by rank — "
            "R_A absorbs the rank effect (breaks for SST-2 Phase 1).")
    fig.text(0.01, -0.08, note, fontsize=8.5, style="italic", color="#333333",
             bbox=dict(facecolor="#fff9e6", alpha=0.9, edgecolor="#e0c060",
                       boxstyle="round,pad=0.4"))

    plt.tight_layout(rect=[0, 0.06, 1, 1])
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out}")

    return t_a, t_b, t_c, t_d


# ══════════════════════════════════════════════════════════════════════════════
# Figure 3 — Directional Vulnerability & T* Summary
# ══════════════════════════════════════════════════════════════════════════════

def plot_directional_vulnerability(
    t_a, t_b, t_c, t_d=None,
    sigma_acc=0.002857,
    out="figures/fig3_directional_vuln.png"
):
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle("Directional Vulnerability: Why T* ≪ 1 (the key mechanistic finding)",
                 fontsize=12, fontweight="bold")

    # ── Panel 1: T* summary — all 4 pairs (acc-based) ───────────────────────
    ax = axes[0]
    # T* for SST-2→MNLI is rank-dependent — show rank 32 (where T* is defined)
    t_c_rank32 = (0.0356, 0.0704)  # rank 32 only with σ½=0.003337

    rows_info = [
        ("MNLI→SST-2\n(acc σ½, rank-invariant)", t_a, "#E53935"),
        ("MNLI→AGNews\n(acc σ½, rank-invariant)", t_b, "#E53935"),
        ("SST-2→MNLI\n(acc σ½, rank 32 only;\nrank 8 T*>0.21)", t_c_rank32, "#FF6F00"),
        ("SST-2→CoLA\n(acc σ½, r8 only safe;\nr32 forgets at all LRs)", t_d, "#9C27B0"),
    ]

    for i, (label, band, color) in enumerate(rows_info):
        if band is None or band[0] is None or band[1] is None:
            continue
        lo, hi = min(band), max(band)
        mid = (lo * hi) ** 0.5
        err_lo = mid - lo
        err_hi = hi - mid
        ax.barh(i, mid, xerr=[[err_lo], [err_hi]], color=color, alpha=0.78,
                height=0.5, capsize=6, error_kw={"elinewidth": 2, "capthick": 2})
        ax.text(hi * 1.15, i, f"T*≈{mid:.4f}  (1/T*≈{1/mid:.0f}×)",
                va="center", ha="left", fontsize=9, color=color, fontweight="bold")

    ax.axvline(1.0, color="navy", lw=1.5, ls="--", label="R_A = 1 (certified threshold)")
    ax.fill_betweenx([-0.5, 3.5], 0.003, 1.0, color="green", alpha=0.06, zorder=0)
    ax.set_yticks([0, 1, 2, 3])
    ax.set_yticklabels(["MNLI→SST-2\n(rank-invariant)", "MNLI→AGNews\n(rank-invariant)",
                        "SST-2→MNLI\n(rank 32 only)", "SST-2→CoLA\n(extreme fragility)"],
                       fontsize=9)
    ax.set_xlabel("R_A at threshold (log scale)", fontsize=11)
    ax.set_title("T* values across all 4 task pairs (acc-based)\n"
                 "T* ≪ 1: directed gradients far more destructive than isotropic noise",
                 fontsize=10)
    ax.set_xscale("log")
    ax.set_xlim(0.003, 1.5)
    ax.legend(fontsize=9, loc="lower right")
    ax.grid(True, alpha=0.25, axis="x")

    # ── Panel 2: Directional Vulnerability Concept ───────────────────────────
    ax = axes[1]

    # Use MNLI values for the illustration
    t_mid_acc = (t_a[0] * t_a[1]) ** 0.5 if (t_a[0] and t_a[1]) else 0.13

    # Bar heights: σ½_acc (isotropic tolerance) vs T*_pp (effective directed tolerance)
    t_star_pp = t_mid_acc * sigma_acc     # actual per-param displacement at T*
    d_vuln = sigma_acc / t_star_pp        # directional vulnerability factor

    bar_vals = [sigma_acc, t_star_pp]
    bar_labels = [
        f"σ½_acc = {sigma_acc:.4f}\n(certified isotropic radius)\n\nRandom Gaussian noise\nat this amplitude\ndrops accuracy to half",
        f"T* · σ½ = {t_star_pp:.4f}\n(effective directed radius)\n\nGradient-directed update\nat this amplitude\ncauses catastrophic forgetting"
    ]
    bar_colors = ["#1976D2", "#D32F2F"]
    bars = ax.bar([0, 1], bar_vals, color=bar_colors, width=0.5, alpha=0.8,
                  edgecolor="white", linewidth=1.5)

    # Gap annotation
    ax.annotate("", xy=(1, t_star_pp), xytext=(0, sigma_acc),
                arrowprops=dict(arrowstyle="<->", color="black", lw=1.5))
    ax.text(0.5, (sigma_acc + t_star_pp) / 2, f"{d_vuln:.1f}×\ngap",
            ha="center", va="center", fontsize=12, fontweight="bold", color="black",
            bbox=dict(facecolor="white", alpha=0.9, edgecolor="gray", boxstyle="round"))

    # Labels
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["Isotropic noise\n(σ½ certified)", "Gradient-directed\nfine-tuning"], fontsize=10)
    ax.set_ylabel("Per-parameter displacement magnitude", fontsize=10)
    ax.set_title(f"Directional Vulnerability Factor ≈ {d_vuln:.1f}×\n"
                 "(MNLI→SST-2 / MNLI→AGNews)",
                 fontsize=10)
    ax.set_xlim(-0.5, 1.5)
    ax.set_ylim(0, sigma_acc * 1.4)
    ax.grid(True, alpha=0.25, axis="y")

    # Add interpretation text
    for bar, label, color in zip(bars, bar_labels, bar_colors):
        y = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, y + 0.0001,
                label, ha="center", va="bottom", fontsize=7.5, color=color,
                style="italic")

    key_msg = (
        f"Implication: σ½ is a LOOSE bound for gradient-directed fine-tuning.\n"
        f"Gradients concentrate displacement in destructive subspaces.\n"
        f"Directional vulnerability ≈ {d_vuln:.1f}× is consistent across rank 8 & 32 (rank-invariant)."
    )
    fig.text(0.5, -0.04, key_msg, ha="center", fontsize=9, style="italic", color="#333",
             bbox=dict(facecolor="#e8f5e9", alpha=0.9, edgecolor="#81C784",
                       boxstyle="round,pad=0.4"))

    plt.tight_layout(rect=[0, 0.06, 1, 1])
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out}")


# ══════════════════════════════════════════════════════════════════════════════
# Figure 4 — σ½ density curve (certified basin shape)
# ══════════════════════════════════════════════════════════════════════════════

def plot_density_curve(
    path="h2_llama/mnli_to_sst2/phase1_sigma_half_acc.json",
    out="figures/fig4_density_curve.png"
):
    with open(path) as f:
        d = json.load(f)

    sigmas   = sorted([float(k) for k in d["densities"].keys()])
    densities = [d["densities"][str(s)] for s in sigmas]
    sh_acc   = d["sigma_half"]

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(sigmas, densities, "o-", color="#1976D2", lw=2, markersize=8,
            markeredgecolor="white", markeredgewidth=0.8, label="Certified density")
    ax.axvline(sh_acc, color="red", lw=1.5, ls="--",
               label=f"σ½_acc = {sh_acc:.5f}")
    ax.axhline(0.5, color="gray", lw=0.8, ls=":", alpha=0.7, label="Density = 0.5")

    # Shade refined zone
    refine_sigmas = [s for s in sigmas if 0.002 < s < 0.004]
    refine_dens   = [d["densities"][str(s)] for s in refine_sigmas]
    if refine_sigmas:
        ax.fill_between(refine_sigmas, 0, refine_dens, alpha=0.15, color="orange",
                        label="Refined zone (7 added points)")

    ax.set_xlabel("Perturbation radius σ", fontsize=11)
    ax.set_ylabel("Certified density (fraction passing)", fontsize=11)
    ax.set_title("MNLI Phase 1 certified density curve (acc-based)\n"
                 "σ½_acc = 0.002857 (refined from coarse 0.003120)", fontsize=10)
    ax.set_ylim(-0.05, 1.1)
    ax.legend(fontsize=9, framealpha=0.9)
    ax.grid(True, alpha=0.3)

    fig.text(0.5, -0.04,
             "Each point: fraction of 150 Gaussian-perturbed models maintaining ≥90% of baseline MNLI accuracy.\n"
             "Refinement added 7 σ points in the 0.0021–0.0042 transition zone, reducing uncertainty from ±30% to ±5%.",
             ha="center", fontsize=8, style="italic",
             bbox=dict(facecolor="#f5f5f5", alpha=0.8, edgecolor="#ccc", boxstyle="round,pad=0.3"))

    plt.tight_layout(rect=[0, 0.1, 1, 1])
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out}")


# ══════════════════════════════════════════════════════════════════════════════
# Figure 5 — Trajectory comparison (H2-A rank 8 vs rank 32 at each LR)
# ══════════════════════════════════════════════════════════════════════════════

def plot_trajectories(
    path="h2_llama/mnli_to_sst2/summary.json",
    out="figures/fig5_trajectories.png"
):
    with open(path) as f:
        data = json.load(f)

    # Group by LR
    by_lr = {}
    for d in data:
        lr = d["lr"]
        by_lr.setdefault(lr, {})[d["rank"]] = d

    lrs_to_plot = [1e-5, 1e-4, 2e-4, 5e-4]
    fig, axes = plt.subplots(2, 2, figsize=(13, 8))
    axes = axes.flatten()

    for ax, lr in zip(axes, lrs_to_plot):
        for rank, d in sorted(by_lr.get(lr, {}).items()):
            traj = d["trajectory"]
            steps = [t["step"] for t in traj]
            R_A   = [t["R_A"]  for t in traj]
            acc_A = [t["acc_task_a"] for t in traj]

            color = LR_COLORS[lr]
            ls = "-" if rank == 8 else "--"
            label = f"rank {rank}"
            ax.plot(R_A, acc_A, ls, color=color, lw=2, marker=".", markersize=6,
                    markevery=3, label=label)
            # Mark final point
            if R_A:
                ax.scatter([R_A[-1]], [acc_A[-1]], color=color, s=120, zorder=5,
                           edgecolors="black", linewidth=1, marker=RANK_MARKERS[rank])

        baseline = 0.884
        T_STAR = 0.083   # empirical T* for MNLI Phase 1
        ax.axvline(1.0, color="red", lw=1.2, ls="--", alpha=0.7,
                   label="R_A = 1  (certified bound)")
        ax.axvline(T_STAR, color="darkorange", lw=1.2, ls=":",
                   label=f"T* ≈ {T_STAR}  (empirical threshold)")
        ax.axhline(baseline, color="gray", lw=0.9, ls="--", alpha=0.7,
                   label=f"Phase 1 baseline  ({baseline})")
        ax.axhline(baseline * 0.88, color="black", lw=0.7, ls=":", alpha=0.6,
                   label=f"88% retention  ({baseline*0.88:.3f})")
        ax.set_xscale("log")
        ax.set_xlabel("R_A = ‖Δθ‖ / σ½_A  (grows during Phase 2 training)", fontsize=10)
        ax.set_ylabel("MNLI Accuracy  (Phase 1 task, monitored during Phase 2)", fontsize=10)
        ax.set_title(f"MNLI→SST-2  lr={lr:.0e}\n"
                     "Each point = one eval checkpoint; forgetting visible before training ends",
                     fontsize=10)
        ax.legend(fontsize=8.5, framealpha=0.9)
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0.28, 1.0)

    plt.suptitle("H2 Trajectories: MNLI forgetting during SST-2 fine-tuning\n"
                 "Rank-invariance: rank 8 & 32 overlap when R_A is on x-axis",
                 fontsize=12, fontweight="bold")
    plt.tight_layout()
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out}")


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("Generating figures...")

    print("\n[1/6] H1 GPT-2 + Llama comparison (updated with KL-SuperNI + MMLU)...")
    plot_h1()

    print("\n[2/6] H1 V6 closure (KL failure for SST-2 and MNLI)...")
    plot_h1_mnli()

    print("\n[3/6] H2 cross-task scatter...")
    t_a, t_b, t_c, t_d = plot_h2_cross_task()

    print(f"\n  T* zones:")
    print(f"    MNLI→SST-2  (acc): [{t_a[0]:.4f}, {t_a[1]:.4f}]  mid={((t_a[0]*t_a[1])**0.5):.4f}")
    print(f"    MNLI→AGNews (acc): [{t_b[0]:.4f}, {t_b[1]:.4f}]  mid={((t_b[0]*t_b[1])**0.5):.4f}")
    if t_c[0] and t_c[1]:
        print(f"    SST-2→MNLI  (acc, rank-32 T*): [{t_c[0]:.4f}, {t_c[1]:.4f}]  mid={((t_c[0]*t_c[1])**0.5):.4f}")
    if t_d[0] and t_d[1]:
        print(f"    SST-2→CoLA  (acc): [{t_d[0]:.4f}, {t_d[1]:.4f}]  mid={((t_d[0]*t_d[1])**0.5):.4f}")

    print("\n[4/6] Directional vulnerability...")
    plot_directional_vulnerability(t_a, t_b, t_c, t_d)

    print("\n[5/6] Density curve...")
    plot_density_curve()

    print("\n[6/6] Trajectories...")
    plot_trajectories()

    print("\nAll figures saved to figures/")
