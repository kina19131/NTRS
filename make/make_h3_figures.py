"""
H3 LoRA Subspace Certification figures.
Data: outputs/h1_basin_widening/gpt2_lora_rank8/  (run1, also used as h3_run1)
"""
import json, os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

OUT = "figures/h1"
os.makedirs(OUT, exist_ok=True)
BASE = "outputs/h1_basin_widening/gpt2_lora_rank8"

def sh(results, threshold=0.5):
    for i in range(len(results) - 1):
        if results[i]["density"] >= threshold and results[i+1]["density"] < threshold:
            t = (threshold - results[i]["density"]) / (results[i+1]["density"] - results[i]["density"])
            return results[i]["sigma"] + t * (results[i+1]["sigma"] - results[i]["sigma"])
    return results[-1]["sigma"] if results[-1]["density"] >= threshold else 0.0

def load(path):
    d = json.load(open(path))
    return d.get("sigma_results", d.get("results", []))

pre_r = load(f"{BASE}/pretrained_density.json")
iso1  = load(f"{BASE}/lora_lr1e-04_density.json")
sub1  = load(f"{BASE}/lora_lr1e-04_subspace.json")
null1 = load(f"{BASE}/lora_lr1e-04_nullspace.json")
iso5  = load(f"{BASE}/lora_lr5e-04_density.json")
sub5  = load(f"{BASE}/lora_lr5e-04_subspace.json")
null5 = load(f"{BASE}/lora_lr5e-04_nullspace.json")

sh_pre  = sh(pre_r)
sh_iso1 = sh(iso1)
sh_sub1 = sh(sub1)
sh_null1= sh(null1)
# lr=5e-4 never crosses 0.5 — density too low even at smallest σ

COLORS = {
    "pretrained": "#444444",
    "isotropic":  "#2166ac",
    "subspace":   "#d6604d",
    "nullspace":  "#4dac26",
}

sigmas1 = [r["sigma"] for r in iso1]
sigmas5 = [r["sigma"] for r in iso5]

# ═══════════════════════════════════════════════════════════════════════════
# Fig 1 — Density curves: lr=1e-4 (the informative condition)
# ═══════════════════════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(8, 5))

ax.plot([r["sigma"] for r in pre_r], [r["density"] for r in pre_r],
        color=COLORS["pretrained"], lw=2, ls=":", label="Pretrained (no fine-tuning)")
ax.plot(sigmas1, [r["density"] for r in iso1],
        color=COLORS["isotropic"], lw=2.5, label="LoRA isotropic  (σ½=0.000756)")
ax.plot(sigmas1, [r["density"] for r in null1],
        color=COLORS["nullspace"], lw=2.5, ls="--", label="LoRA null-space  (σ½=0.000750)")
ax.plot(sigmas1, [r["density"] for r in sub1],
        color=COLORS["subspace"], lw=2.5, ls="-.", label="LoRA subspace    (σ½=0.000667)")

ax.axhline(0.5, color="black", lw=1, ls="--", alpha=0.5, label="σ½ threshold (density=0.5)")
ax.axvline(sh_iso1,  color=COLORS["isotropic"], lw=1.2, ls=":", alpha=0.7)
ax.axvline(sh_null1, color=COLORS["nullspace"],  lw=1.2, ls=":", alpha=0.7)
ax.axvline(sh_sub1,  color=COLORS["subspace"],   lw=1.2, ls=":", alpha=0.7)

ax.set_xscale("log")
ax.set_xlabel("σ  (perturbation scale)", fontsize=11)
ax.set_ylabel("Certified density  C(θ, σ)", fontsize=11)
ax.set_title("H3: LoRA subspace vs isotropic vs null-space  [GPT-2, lr=1e-4]\n"
             "Null-space ≈ isotropic; LoRA update subspace is the sharpest direction",
             fontsize=10)
ax.legend(fontsize=9, loc="upper right")
ax.set_xlim(7e-5, 3e-3)
ax.set_ylim(-0.02, 1.05)

# Annotate σ½ values
ax.annotate(f"σ½={sh_iso1:.5f}", xy=(sh_iso1, 0.5), xytext=(sh_iso1*1.5, 0.62),
            fontsize=8, color=COLORS["isotropic"],
            arrowprops=dict(arrowstyle="->", color=COLORS["isotropic"]))
ax.annotate(f"σ½={sh_sub1:.5f}", xy=(sh_sub1, 0.5), xytext=(sh_sub1*0.35, 0.35),
            fontsize=8, color=COLORS["subspace"],
            arrowprops=dict(arrowstyle="->", color=COLORS["subspace"]))

fig.tight_layout()
fig.savefig(f"{OUT}/h3_density_curves_lr1e4.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved: h3_density_curves_lr1e4.png")


# ═══════════════════════════════════════════════════════════════════════════
# Fig 2 — σ½ bar chart + H1 context
# ═══════════════════════════════════════════════════════════════════════════
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Panel A: σ½ bar chart for lr=1e-4
ax = axes[0]
labels_bar = ["Pretrained", "LoRA\nisotropic", "LoRA\nnull-space", "LoRA\nsubspace"]
vals       = [sh_pre, sh_iso1, sh_null1, sh_sub1]
colors_bar = [COLORS["pretrained"], COLORS["isotropic"], COLORS["nullspace"], COLORS["subspace"]]
bars = ax.bar(labels_bar, [v*1e4 for v in vals], color=colors_bar, width=0.55,
              edgecolor="black", linewidth=0.8, alpha=0.88)
for bar, v in zip(bars, vals):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
            f"{v:.5f}", ha="center", va="bottom", fontsize=8.5, fontweight="bold")

ax.set_ylabel("σ½  (×10⁻⁴)", fontsize=11)
ax.set_title("Certified radius σ½ by perturbation type\n[GPT-2, lr=1e-4, rank=8]", fontsize=10)

# Shading: LoRA WIDENS basin vs pretrained (H1)
ax.annotate("", xy=(1, sh_iso1*1e4), xytext=(0, sh_pre*1e4),
            arrowprops=dict(arrowstyle="->", color="navy", lw=1.5))
ax.text(0.5, (sh_pre + sh_iso1)/2*1e4 + 0.05, "+57%\n(H1)", ha="center", fontsize=8.5,
        color="navy", fontweight="bold")

# Subspace arrow showing it's narrower
ax.annotate("", xy=(3, sh_sub1*1e4), xytext=(1, sh_iso1*1e4),
            arrowprops=dict(arrowstyle="->", color=COLORS["subspace"], lw=1.5))
ax.text(2.0, (sh_iso1 + sh_sub1)/2*1e4 + 0.05, "−12%\n(H3)", ha="center", fontsize=8.5,
        color=COLORS["subspace"], fontweight="bold")

ax.set_ylim(0, max(vals)*1e4 * 1.35)

# Panel B: density curves for lr=5e-4 (the collapsed condition)
ax = axes[1]
ax.plot(sigmas5, [r["density"] for r in iso5],
        color=COLORS["isotropic"], lw=2.5, label="Isotropic")
ax.plot(sigmas5, [r["density"] for r in null5],
        color=COLORS["nullspace"], lw=2.5, ls="--", label="Null-space")
ax.plot(sigmas5, [r["density"] for r in sub5],
        color=COLORS["subspace"], lw=2.5, ls="-.", label="Subspace")

# Pretrained for reference
ax.plot([r["sigma"] for r in pre_r], [r["density"] for r in pre_r],
        color=COLORS["pretrained"], lw=1.5, ls=":", label="Pretrained", alpha=0.7)

ax.axhline(0.5, color="black", lw=1, ls="--", alpha=0.5, label="σ½ threshold")
ax.set_xscale("log")
ax.set_xlabel("σ", fontsize=11)
ax.set_ylabel("Certified density", fontsize=11)
ax.set_title("LR=5e-4: basin too sharp to measure\n"
             "All three start below 0.5 — σ½ = 0 for all", fontsize=10)
ax.legend(fontsize=8.5)
ax.set_xlim(7e-5, 3e-3)
ax.set_ylim(-0.02, 1.05)

ax.text(0.5, 0.5, "BASIN TOO NARROW\n(aggressive fine-tuning\ncollapsed basin)",
        transform=ax.transAxes, ha="center", va="center",
        fontsize=12, color="#b2182b", fontweight="bold", alpha=0.35)

fig.suptitle("H1 + H3 Combined: LoRA widens basin overall (H1 ✓), "
             "but update subspace is the sharpest direction (H3 revised)",
             fontsize=10)
fig.tight_layout()
fig.savefig(f"{OUT}/h3_sigma_half_bars.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved: h3_sigma_half_bars.png")


# ═══════════════════════════════════════════════════════════════════════════
# Fig 3 — Conceptual summary: what the geometry looks like
# ═══════════════════════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(9, 5))

# Three curves side by side for interpretation
ax.plot(sigmas1, [r["density"] for r in iso1],
        color=COLORS["isotropic"], lw=2.5, label=f"Isotropic  σ½={sh_iso1:.5f}")
ax.plot(sigmas1, [r["density"] for r in null1],
        color=COLORS["nullspace"], lw=2.5, ls="--", label=f"Null-space  σ½={sh_null1:.5f}  ≈ isotropic")
ax.plot(sigmas1, [r["density"] for r in sub1],
        color=COLORS["subspace"], lw=2.5, ls="-.", label=f"Subspace  σ½={sh_sub1:.5f}  ← sharpest")
ax.plot([r["sigma"] for r in pre_r], [r["density"] for r in pre_r],
        color=COLORS["pretrained"], lw=1.5, ls=":", label=f"Pretrained  σ½={sh_pre:.5f}")

ax.axhline(0.5, color="black", lw=1, ls="--", alpha=0.4)
ax.set_xscale("log")
ax.set_xlabel("σ", fontsize=12)
ax.set_ylabel("Certified density  C(θ, σ)", fontsize=12)
ax.set_xlim(7e-5, 3e-3); ax.set_ylim(-0.02, 1.05)

# Interpretation box
ax.text(0.58, 0.60,
        "Geometry interpretation:\n"
        "• Null-space ≈ Isotropic\n"
        "  → flat directions (not trained)\n"
        "• Subspace < Isotropic\n"
        "  → LoRA update directions are\n"
        "     the curved (sharp) directions\n"
        "• Fine-tuning concentrates\n"
        "  curvature in LoRA subspace",
        transform=ax.transAxes, fontsize=9,
        bbox=dict(boxstyle="round,pad=0.5", facecolor="#fffde7", edgecolor="#ccc", alpha=0.9))

ax.legend(fontsize=9, loc="upper right")
ax.set_title("H3 Finding: LoRA update subspace = direction of maximum curvature\n"
             "[GPT-2, lr=1e-4, rank=8]", fontsize=10)
fig.tight_layout()
fig.savefig(f"{OUT}/h3_geometry_interpretation.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved: h3_geometry_interpretation.png")

print(f"\nAll H3 figures saved to {OUT}/")
print(f"\nKey numbers (lr=1e-4):")
print(f"  Pretrained  σ½ = {sh_pre:.6f}")
print(f"  Isotropic   σ½ = {sh_iso1:.6f}  ({sh_iso1/sh_pre:.2f}× pretrained)  ← H1")
print(f"  Null-space  σ½ = {sh_null1:.6f}  ({sh_null1/sh_iso1:.2f}× isotropic)")
print(f"  Subspace    σ½ = {sh_sub1:.6f}  ({sh_sub1/sh_iso1:.2f}× isotropic)  ← H3 result")
