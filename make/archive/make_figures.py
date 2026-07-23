"""
make_figures.py
===============
Generates all paper figures from results/lora_density/.

  fig1_pretrained_scaling.png   — density curves, all 5 models (pretrained)
  fig2_lora_before_after.png    — before/after LoRA lr=1e-4, GPT-2 family
  fig3_phase_transition.png     — phase transition matrix: model × LR
  fig4_norm_predictor.png       — norm/σ½ predictor at lr=1e-4 vs lr=5e-4
  fig5_sigma_half_summary.png   — σ½ bar chart: all 5 models
  fig6_subspace_comparison.png  — isotropic vs LoRA-subspace noise

Run: python make_figures.py
"""

import json, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

ROOT    = os.path.dirname(os.path.abspath(__file__))
LD      = os.path.join(ROOT, "results", "lora_density")
OUT_DIR = os.path.join(ROOT, "results", "lora_density", "figures")
os.makedirs(OUT_DIR, exist_ok=True)

plt.rcParams.update({
    "figure.facecolor": "white",
    "axes.facecolor":   "white",
    "axes.spines.top":  False,
    "axes.spines.right":False,
    "axes.grid":        True,
    "grid.alpha":       0.25,
    "grid.linestyle":   "--",
    "font.family":      "sans-serif",
    "font.size":        11,
    "axes.titlesize":   12,
    "axes.labelsize":   11,
    "legend.fontsize":  9,
    "legend.framealpha":0.9,
})

C = {
    "gpt2":           "#2563EB",
    "gpt2-medium":    "#DC2626",
    "gpt2-large":     "#7C3AED",
    "tinyllama-1.1b": "#D97706",
    "llama-3.2-1b":   "#059669",
}
MODEL_LABELS = {
    "gpt2":           "GPT-2 (124M)",
    "gpt2-medium":    "GPT-2-med (354M)",
    "gpt2-large":     "GPT-2-large (774M)",
    "tinyllama-1.1b": "TinyLlama (1.1B)",
    "llama-3.2-1b":   "Llama-3.2 (1B)",
}
MODEL_LABELS_MULTILINE = {
    "gpt2":           "GPT-2\n(124M)",
    "gpt2-medium":    "GPT-2-med\n(354M)",
    "gpt2-large":     "GPT-2-large\n(774M)",
    "tinyllama-1.1b": "TinyLlama\n(1.1B)",
    "llama-3.2-1b":   "Llama-3.2\n(1B)",
}
MODELS = ["gpt2", "gpt2-medium", "gpt2-large", "tinyllama-1.1b", "llama-3.2-1b"]
LRS    = [1e-4, 5e-4, 1e-3, 5e-3]
ALPHA_CI = 0.12

STATUS_COLORS = {
    "preserved":  "#16a34a",
    "partial":    "#f59e0b",
    "collapsed":  "#dc2626",
    "forgetting": "#9ca3af",
}


# ── Helpers ───────────────────────────────────────────────────────────────────
def load_density(path):
    if not os.path.exists(path):
        return None
    with open(path) as f:
        d = json.load(f)
    return d.get("sigma_results", d)

def load_norm(path):
    if not os.path.exists(path):
        return float("nan")
    with open(path) as f:
        return json.load(f).get("per_param_norm", float("nan"))

def half_max_sigma(sr):
    if sr is None:
        return float("nan")
    dens   = [r["density"] for r in sr]
    sigmas = [r["sigma"]   for r in sr]
    target = max(dens) / 2.0
    for i in range(len(dens) - 1):
        if dens[i] >= target > dens[i+1]:
            t = (target - dens[i]) / (dens[i+1] - dens[i])
            return sigmas[i] + t * (sigmas[i+1] - sigmas[i])
    return sigmas[-1]

def certified_radius(sr, target=0.40):
    if sr is None:
        return 0.0
    valid = [r for r in sr if r["ci_lower"] >= target]
    return max((r["sigma"] for r in valid), default=0.0)

def get_base_ppl(sr):
    if sr is None:
        return float("nan")
    return sr[0].get("base_ppl", float("nan"))

def extract(sr):
    return (
        np.array([r["sigma"]    for r in sr]),
        np.array([r["density"]  for r in sr]),
        np.array([r["ci_lower"] for r in sr]),
        np.array([r["ci_upper"] for r in sr]),
    )

def cert_status(ft_sr, pre_sr, forgetting_factor=2.0):
    """
    Classify one (model, lr) result into a status string.
    Uses σ½ when pretrained σ* = 0 (Llama-family), otherwise uses σ*.
    """
    pre_ppl = get_base_ppl(pre_sr)
    ft_ppl  = get_base_ppl(ft_sr)
    if np.isnan(ft_ppl) or np.isnan(pre_ppl) or ft_ppl > pre_ppl * forgetting_factor:
        return "forgetting"

    pre_star = certified_radius(pre_sr)
    ft_star  = certified_radius(ft_sr)

    if pre_star > 0:
        if ft_star >= pre_star:
            return "preserved"
        elif ft_star > 0:
            return "partial"
        else:
            return "collapsed"
    else:
        # pretrained σ* not achieved in our grid — use σ½ as proxy
        pre_sh = half_max_sigma(pre_sr)
        ft_sh  = half_max_sigma(ft_sr)
        if np.isnan(pre_sh) or pre_sh == 0:
            return "collapsed"
        ratio = ft_sh / pre_sh
        if ratio >= 0.85:
            return "preserved"
        elif ft_sh > 0:
            return "partial"
        else:
            return "collapsed"


# ── Load all data ─────────────────────────────────────────────────────────────
def load_all():
    data = {}
    for m in MODELS:
        data[m] = {"pretrained": load_density(f"{LD}/{m}/pretrained_density.json")}
        for lr in LRS:
            tag = f"lr{lr:.0e}"
            data[m][lr]           = load_density(f"{LD}/{m}/lora_{tag}_density.json")
            data[m][f"norm_{lr}"] = load_norm(f"{LD}/{m}/lora_{tag}_norms.json")
    # Subspace data
    data["gpt2_subspace"] = {
        "pretrained": load_density(f"{LD}/gpt2_subspace/pretrained_density.json"),
        "isotropic":  load_density(f"{LD}/gpt2_subspace/lora_lr1e-04_density.json"),
        "subspace":   load_density(f"{LD}/gpt2_subspace/lora_lr1e-04_subspace.json"),
    }
    return data


def _save(fig, name):
    path = os.path.join(OUT_DIR, name)
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close()
    print(f"Saved {path}")


# ── Fig 1: Pretrained density scaling (all 5 models) ─────────────────────────
def fig1(data):
    fig, ax = plt.subplots(figsize=(8, 5))
    for m in MODELS:
        sr = data[m]["pretrained"]
        if sr is None:
            continue
        s, d, lo, hi = extract(sr)
        ax.plot(s, d, color=C[m], lw=2.2, marker="o", ms=4,
                label=MODEL_LABELS[m])
        ax.fill_between(s, lo, hi, color=C[m], alpha=ALPHA_CI)
        sh = half_max_sigma(sr)
        if not np.isnan(sh):
            ax.axvline(sh, color=C[m], lw=1.0, ls=":", alpha=0.45)

    ax.set_xscale("log")
    ax.set_xlabel("Perturbation scale σ")
    ax.set_ylabel("Certified density  C(θ, σ)")
    ax.set_title("Pretrained Certified Density Across Architectures\n"
                 "N=200, 95% CI  |  dotted lines mark σ½ per model")
    ax.legend(fontsize=9)
    ax.set_ylim(-0.02, 1.05)
    _save(fig, "fig1_pretrained_scaling.png")


# ── Fig 2: Before/after LoRA lr=1e-4, GPT-2 family ───────────────────────────
def fig2(data):
    gpt2_models = ["gpt2", "gpt2-medium", "gpt2-large"]
    fig, axes = plt.subplots(1, 3, figsize=(18, 5), sharey=True)

    for ax, m in zip(axes, gpt2_models):
        pre  = data[m]["pretrained"]
        lora = data[m][1e-4]
        if pre is None or lora is None:
            continue
        color = C[m]
        s_p, d_p, lo_p, hi_p = extract(pre)
        s_l, d_l, lo_l, hi_l = extract(lora)
        sh_pre  = half_max_sigma(pre)
        sh_lora = half_max_sigma(lora)

        ax.plot(s_p, d_p, color=color, lw=2.2, marker="o", ms=5, ls="-",
                label=f"Pretrained  (σ½={sh_pre:.5f})")
        ax.fill_between(s_p, lo_p, hi_p, color=color, alpha=ALPHA_CI)
        ax.plot(s_l, d_l, color=color, lw=2.2, marker="s", ms=5, ls="--",
                label=f"LoRA lr=1e-4  (σ½={sh_lora:.5f})")
        ax.fill_between(s_l, lo_l, hi_l, color=color, alpha=ALPHA_CI)
        ax.set_xscale("log")
        ax.set_xlabel("Perturbation scale σ")
        ax.set_title(MODEL_LABELS[m])
        ax.legend(fontsize=8.5)
        ax.set_ylim(-0.02, 1.05)

    axes[0].set_ylabel("Certified density  C(θ, σ)")
    fig.suptitle("Certified Density Before vs. After LoRA  (lr=1e-4)\n"
                 "GPT-2 family  |  N=200, 95% CI, WikiText-2", y=1.01)
    fig.tight_layout()
    _save(fig, "fig2_lora_before_after.png")


# ── Fig 3: Phase transition matrix ───────────────────────────────────────────
def fig3(data):
    nrows, ncols = len(MODELS), len(LRS)
    fig, ax = plt.subplots(figsize=(11, 6))

    for i, m in enumerate(MODELS):
        pre_sr = data[m]["pretrained"]
        pre_star = certified_radius(pre_sr)
        pre_ppl  = get_base_ppl(pre_sr)
        row = nrows - 1 - i   # top = first model

        for j, lr in enumerate(LRS):
            ft_sr  = data[m][lr]
            status = cert_status(ft_sr, pre_sr)
            color  = STATUS_COLORS[status]

            rect = mpatches.FancyBboxPatch(
                [j - 0.44, row - 0.44], 0.88, 0.88,
                boxstyle="round,pad=0.04",
                facecolor=color, alpha=0.80, edgecolor="white", linewidth=2,
            )
            ax.add_patch(rect)

            # Cell text: σ* value or status label
            ft_ppl  = get_base_ppl(ft_sr)
            ft_star = certified_radius(ft_sr)
            if status == "forgetting":
                ppl_ratio = ft_ppl / pre_ppl if pre_ppl > 0 else float("nan")
                cell_text = f"forget\nppl×{ppl_ratio:.1f}"
            elif ft_star > 0:
                cell_text = f"σ*={ft_star:.4f}"
            else:
                cell_text = "σ*=0\n(collapsed)"

            ax.text(j, row, cell_text, ha="center", va="center",
                    fontsize=8, fontweight="bold", color="white",
                    multialignment="center")

    ax.set_xlim(-0.5, ncols - 0.5)
    ax.set_ylim(-0.5, nrows - 0.5)
    ax.set_xticks(range(ncols))
    ax.set_xticklabels([f"lr={lr:.0e}" for lr in LRS], fontsize=12)
    ax.set_yticks(range(nrows))
    ax.set_yticklabels(
        [MODEL_LABELS[m] for m in reversed(MODELS)], fontsize=11
    )
    ax.grid(False)

    patches = [mpatches.Patch(color=STATUS_COLORS[k], alpha=0.80,
                               label=k.capitalize())
               for k in STATUS_COLORS]
    ax.legend(handles=patches, loc="upper right", fontsize=10,
              framealpha=0.95, edgecolor="gray")
    ax.set_title("Phase Transition: Certificate Status vs. Learning Rate × Model\n"
                 "Green=preserved  |  Amber=partial  |  Red=collapsed  |  Gray=catastrophic forgetting",
                 fontsize=11)
    fig.tight_layout()
    _save(fig, "fig3_phase_transition.png")


# ── Fig 4: norm/σ½ predictor at lr=1e-4 vs lr=5e-4 ──────────────────────────
def fig4(data):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

    for ax, lr, title in [
        (ax1, 1e-4, "lr = 1e-4  (safe regime)\nAll norms < σ½ → cert preserved"),
        (ax2, 5e-4, "lr = 5e-4  (collapse threshold)\nnorm > σ½ → cert collapses / forgetting"),
    ]:
        ratios = []
        colors = []
        labels = []
        hatch_flags = []

        for m in MODELS:
            pre_sr = data[m]["pretrained"]
            ft_sr  = data[m][lr]
            norm   = data[m][f"norm_{lr}"]
            sh_pre = half_max_sigma(pre_sr)

            ratio = norm / sh_pre if (not np.isnan(norm) and not np.isnan(sh_pre) and sh_pre > 0) else float("nan")
            status = cert_status(ft_sr, pre_sr)

            ratios.append(ratio)
            colors.append(C[m] if status != "forgetting" else "#9ca3af")
            labels.append(MODEL_LABELS_MULTILINE[m])
            hatch_flags.append(status == "forgetting")

        x = np.arange(len(MODELS))
        bars = ax.bar(x, ratios, color=colors, alpha=0.82, width=0.55,
                      edgecolor="white", linewidth=1.2)

        # Hatching for forgetting cases
        for bar, is_fgt in zip(bars, hatch_flags):
            if is_fgt:
                bar.set_hatch("///")
                bar.set_edgecolor("#6b7280")

        ax.axhline(1.0, color="black", lw=1.8, ls="--",
                   label="norm = σ½  (phase boundary)")

        for bar, ratio in zip(bars, ratios):
            if not np.isnan(ratio):
                ax.text(bar.get_x() + bar.get_width()/2,
                        ratio + max(ratios) * 0.03,
                        f"{ratio:.2f}×", ha="center", fontsize=9,
                        fontweight="bold")

        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=9)
        ax.set_ylabel("LoRA update norm  /  pretrained σ½")
        ax.set_title(title, fontsize=10)
        ax.legend(fontsize=9)
        ax.set_ylim(0, max(r for r in ratios if not np.isnan(r)) * 1.35)

    fig.suptitle("norm/σ½ Ratio as Phase Transition Predictor\n"
                 "Ratio < 1 → update inside certified ball → cert preserved; "
                 "ratio > 1 → outside ball → cert collapses",
                 y=1.02, fontsize=11)
    fig.tight_layout()
    _save(fig, "fig4_norm_predictor.png")


# ── Fig 5: σ½ bar chart, all 5 models ────────────────────────────────────────
def fig5(data):
    sh_pre  = [half_max_sigma(data[m]["pretrained"]) for m in MODELS]
    sh_lora = [half_max_sigma(data[m][1e-4]) for m in MODELS]
    norms   = [data[m]["norm_0.0001"] for m in MODELS]
    norm_ratios = [
        n / sp if (not np.isnan(n) and not np.isnan(sp) and sp > 0) else float("nan")
        for n, sp in zip(norms, sh_pre)
    ]

    x = np.arange(len(MODELS))
    w = 0.35
    colors = [C[m] for m in MODELS]

    fig, ax = plt.subplots(figsize=(10, 5))
    b1 = ax.bar(x - w/2, sh_pre,  w, color=colors, alpha=0.90, label="Pretrained")
    b2 = ax.bar(x + w/2, sh_lora, w, color=colors, alpha=0.38,
                edgecolor=colors, linewidth=1.8, label="LoRA lr=1e-4")

    for bar in list(b1) + list(b2):
        h = bar.get_height()
        if not np.isnan(h):
            ax.text(bar.get_x() + bar.get_width()/2, h * 1.03,
                    f"{h:.5f}", ha="center", va="bottom", fontsize=7.5, rotation=45)

    for bar, ratio in zip(b2, norm_ratios):
        if not np.isnan(ratio):
            ax.text(bar.get_x() + bar.get_width()/2,
                    bar.get_height() + max(sh_pre) * 0.13,
                    f"norm/σ½\n={ratio:.2f}×",
                    ha="center", fontsize=7, color="#374151")

    ax.set_xticks(x)
    ax.set_xticklabels([MODEL_LABELS_MULTILINE[m] for m in MODELS], fontsize=10)
    ax.set_ylabel("σ½  (half-peak perturbation scale)")
    ax.set_title("σ½ Before vs. After LoRA Fine-Tuning  (lr=1e-4)\n"
                 "Basin width preserved across all architectures at safe learning rate")
    ax.legend()
    _save(fig, "fig5_sigma_half_summary.png")


# ── Fig 6: Subspace vs isotropic (GPT-2, lr=1e-4) ────────────────────────────
def fig6(data):
    sub = data["gpt2_subspace"]
    fig, ax = plt.subplots(figsize=(7, 5))
    color = C["gpt2"]

    for sr, label, ls, marker in [
        (sub["pretrained"], "Pretrained",           "-",  "o"),
        (sub["isotropic"],  "LoRA isotropic noise", "--", "s"),
        (sub["subspace"],   "LoRA subspace noise",  ":",  "^"),
    ]:
        if sr is None:
            continue
        s, d, lo, hi = extract(sr)
        ax.plot(s, d, color=color, lw=2.2, marker=marker, ms=5, ls=ls,
                label=label, alpha=(1.0 if ls != ":" else 0.75))
        ax.fill_between(s, lo, hi, color=color, alpha=ALPHA_CI * 0.6)

    for sr, ls, label in [
        (sub["pretrained"], "-",  f"σ½ pretrained = {half_max_sigma(sub['pretrained']):.5f}"),
        (sub["isotropic"],  "--", f"σ½ isotropic  = {half_max_sigma(sub['isotropic']):.5f}"),
        (sub["subspace"],   ":",  f"σ½ subspace   = {half_max_sigma(sub['subspace']):.5f}"),
    ]:
        if sr is not None:
            sh = half_max_sigma(sr)
            ax.axvline(sh, color=color, lw=1.0, ls=ls, alpha=0.35, label=label)

    ax.set_xscale("log")
    ax.set_xlabel("Perturbation scale σ")
    ax.set_ylabel("Certified density  C(θ, σ)")
    ax.set_title("Isotropic vs. LoRA-Subspace Noise  —  GPT-2  (124M, lr=1e-4)\n"
                 "Subspace projection does not improve certified density")
    ax.legend(fontsize=9)
    ax.set_ylim(-0.02, 1.05)
    _save(fig, "fig6_subspace_comparison.png")


# ── Main ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print(f"Loading data from {LD}...")
    data = load_all()

    # Print quick summary table
    print("\nQuick summary  (σ½ pretrained  |  σ* pretrained):")
    for m in MODELS:
        sr  = data[m]["pretrained"]
        sh  = half_max_sigma(sr)
        st  = certified_radius(sr)
        ppl = get_base_ppl(sr)
        print(f"  {MODEL_LABELS[m]:22s}  σ½={sh:.6f}  σ*={st:.6f}  ppl={ppl:.3f}")

    print(f"\nGenerating figures → {OUT_DIR}\n")
    fig1(data)
    fig2(data)
    fig3(data)
    fig4(data)
    fig5(data)
    fig6(data)
    print("\nAll done.")
