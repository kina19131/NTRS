"""
make_research_plan.py
Generates a slideshow (PNG series + PDF) tracking hypotheses, experiments, and status.
Regenerate any time experiments update: python make_research_plan.py
"""

import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
import matplotlib.image as mpimg

BASE  = os.path.dirname(os.path.abspath(__file__))
OUT   = os.path.join(BASE, "results", "research_plan")
os.makedirs(OUT, exist_ok=True)

W, H   = 16, 9          # slide dimensions (inches)
BG     = "#FAFAFA"
TITLE_COLOR  = "#1E293B"
BODY_COLOR   = "#334155"
GREEN   = "#16A34A"
YELLOW  = "#D97706"
RED     = "#DC2626"
BLUE    = "#2563EB"
LIGHT   = "#F1F5F9"
BORDER  = "#CBD5E1"

def new_slide(title, subtitle=None):
    fig = plt.figure(figsize=(W, H))
    fig.patch.set_facecolor(BG)
    # top bar
    fig.add_axes([0, 0.88, 1, 0.12]).set_axis_off()
    fig.axes[-1].set_facecolor("#1E293B")
    fig.text(0.04, 0.92, title,   fontsize=22, fontweight="bold",
             color="white", va="center", transform=fig.transFigure)
    if subtitle:
        fig.text(0.04, 0.875, subtitle, fontsize=12, color="#94A3B8",
                 va="center", transform=fig.transFigure)
    # footer
    fig.text(0.97, 0.015, "NTRS — Neural Thickets × Randomized Smoothing",
             fontsize=8, color="#94A3B8", ha="right", transform=fig.transFigure)
    return fig


def status_chip(ax, x, y, label, color, fig):
    ax.text(x, y, f"  {label}  ", transform=fig.transFigure,
            fontsize=9, color="white", va="center", ha="left",
            bbox=dict(boxstyle="round,pad=0.3", facecolor=color, edgecolor="none"))


def text_block(fig, x, y, lines, fontsize=11, color=BODY_COLOR, spacing=0.045):
    for i, line in enumerate(lines):
        fig.text(x, y - i * spacing, line, fontsize=fontsize,
                 color=color, va="top", transform=fig.transFigure)


slides = []

# ══════════════════════════════════════════════════════════════════════════════
# SLIDE 1 — Overview
# ══════════════════════════════════════════════════════════════════════════════
fig = new_slide("NTRS Research Tracker", "Hypotheses · Experiments · Status · Redirections")

# Three hypothesis boxes
box_configs = [
    (0.04,  "#DBEAFE", "#2563EB", "H1  [ACTIVE]",
     "σ½ as Universal Adaptation Scale",
     "Does R = ‖Δθ‖/σ½ collapse PPL curves\nacross models and tasks?"),
    (0.37, "#DCFCE7", "#16A34A", "H2  [PAUSED]",
     "Forgetting Budget",
     "Does ‖θ_B − θ_A‖/σ½_A > 1 predict\nTask A catastrophic forgetting?"),
    (0.70, "#F3F4F6", "#6B7280", "H3  [DEFERRED]",
     "Model Merging",
     "Does d < σ½_A + σ½_B predict\nsuccessful weight-space merging?"),
]
for x, bg, border, tag, title, desc in box_configs:
    ax = fig.add_axes([x, 0.20, 0.28, 0.62])
    ax.set_facecolor(bg)
    for spine in ax.spines.values():
        spine.set_color(border); spine.set_linewidth(2)
    ax.set_xticks([]); ax.set_yticks([])
    ax.text(0.08, 0.88, tag,   transform=ax.transAxes,
            fontsize=10, color=border, fontweight="bold")
    ax.text(0.08, 0.72, title, transform=ax.transAxes,
            fontsize=13, color=TITLE_COLOR, fontweight="bold", wrap=True)
    ax.text(0.08, 0.52, desc,  transform=ax.transAxes,
            fontsize=10, color=BODY_COLOR, va="top")

fig.text(0.04, 0.16,
    "Priority: H1 establishes whether σ½ is a meaningful unit. "
    "H2 and H3 build on top of H1.",
    fontsize=10, color="#64748B", style="italic")
slides.append(("01_overview", fig))


# ══════════════════════════════════════════════════════════════════════════════
# SLIDE 2 — Background: what is σ½ and R
# ══════════════════════════════════════════════════════════════════════════════
fig = new_slide("Background: σ½ and R", "The core geometric quantities")

ax = fig.add_axes([0.04, 0.10, 0.44, 0.72])
ax.set_facecolor(LIGHT); ax.set_axis_off()
ax.set_xlim(0, 1); ax.set_ylim(0, 1)
defs = [
    ("σ½  (certified basin radius)",
     "Largest σ where ≥50% of random Gaussian\n"
     "perturbations N(0, σ²I) keep WikiText-2 NLL\n"
     "within eval_slack of the pretrained model."),
    ("C(θ, σ)  (certified density)",
     "Fraction of N random perturbations of size σ\n"
     "that preserve NLL ≤ NLL(θ) + eval_slack.\n"
     "σ½ = σ where C(θ,σ) = 0.5 × max(C)."),
    ("R  (normalized update radius)",
     "R = ‖Δθ‖ / σ½\n"
     "Measures how far fine-tuning moved the weights\n"
     "in units of the certified basin radius.\n"
     "R < 1 → inside ball.  R > 1 → outside ball."),
]
y = 0.93
for title, body in defs:
    ax.text(0.05, y,       "▸ " + title, fontsize=11, fontweight="bold", color=TITLE_COLOR, va="top")
    ax.text(0.07, y-0.065, body,          fontsize=9.5, color=BODY_COLOR, va="top")
    y -= 0.31

# Right: key numbers table
ax2 = fig.add_axes([0.54, 0.10, 0.42, 0.72])
ax2.set_facecolor(LIGHT); ax2.set_axis_off()
ax2.text(0.05, 0.95, "Measured σ½_pre values", fontsize=12,
         fontweight="bold", color=TITLE_COLOR, va="top", transform=ax2.transAxes)
rows = [
    ("GPT-2  (124M)",       "0.000601", "0.0001"),
    ("GPT-2-medium (355M)", "0.000806", "0.0001"),
    ("GPT-2-large  (774M)", "0.000601", "0.0001"),
    ("GPT-2-xl  (1.5B)",    "0.000665", "0.0001"),
    ("Llama-3B  (3B)",      "0.000200", "0.0001"),
]
hdrs = ["Model", "σ½_pre", "eval_slack"]
ys = [0.83, 0.73, 0.63, 0.53, 0.43, 0.33]
for j, h in enumerate(hdrs):
    ax2.text(0.04 + j*0.33, ys[0], h, fontsize=9.5, fontweight="bold",
             color=TITLE_COLOR, va="top", transform=ax2.transAxes)
for i, (m, sh, sl) in enumerate(rows):
    c = BLUE if "Llama" in m else BODY_COLOR
    for j, val in enumerate([m, sh, sl]):
        ax2.text(0.04 + j*0.33, ys[i+1], val, fontsize=9.5, color=c,
                 va="top", transform=ax2.transAxes)

ax2.text(0.05, 0.16,
    "★ σ½_pre nearly constant across GPT-2 family\n"
    "   despite 12× parameter scaling.\n"
    "★ Llama-3B has 3× smaller σ½ despite being\n"
    "   24× larger — better training → sharper min.",
    fontsize=9, color=BLUE, va="top", transform=ax2.transAxes,
    bbox=dict(boxstyle="round", facecolor="#DBEAFE", edgecolor=BLUE, alpha=0.5))
slides.append(("02_background", fig))


# ══════════════════════════════════════════════════════════════════════════════
# SLIDE 3 — H1 Hypothesis
# ══════════════════════════════════════════════════════════════════════════════
fig = new_slide("H1: σ½ as Universal Adaptation Scale", "The core claim and what it predicts")

ax = fig.add_axes([0.04, 0.10, 0.44, 0.72])
ax.set_facecolor("#DBEAFE"); ax.set_axis_off()
claims = [
    ("Claim (weak)",
     "Fine-tuning breaks pretraining quality when\n‖Δθ‖/σ½ exceeds some constant C*.\nσ½ is a meaningful normalization."),
    ("Claim (strong — the paper claim)",
     "C* ≈ 1 universally across:\n  · model sizes\n  · tasks\n  · learning rates\n  · LoRA ranks\nσ½ is the natural unit of adaptation distance."),
    ("What collapse looks like",
     "Plot PPL ratio vs R for all (model, task, LR).\nAll curves overlap → C* is universal.\nBefore normalization: 100× spread in raw ‖Δθ‖.\nAfter normalization: curves collapse near R=1."),
    ("What falsifies it",
     "Stability boundaries differ by >2× across\nconditions after σ½ normalization."),
]
y = 0.93
for title, body in claims:
    ax.text(0.04, y,       "▸ " + title, fontsize=10.5, fontweight="bold",
            color=TITLE_COLOR, va="top")
    ax.text(0.06, y-0.065, body, fontsize=9.5, color=BODY_COLOR, va="top")
    y -= 0.24

ax2 = fig.add_axes([0.54, 0.10, 0.42, 0.72])
ax2.set_facecolor("#DBEAFE"); ax2.set_axis_off()
ax2.text(0.05, 0.95, "Analogies that motivate the claim",
         fontsize=11, fontweight="bold", color=TITLE_COLOR, va="top",
         transform=ax2.transAxes)
analogies = [
    ("Chinchilla scaling laws",
     "Previously: model size and tokens were\nincomparable. Chinchilla found the right\nnormalization (FLOPs ratio)."),
    ("μP parameterization",
     "Previously: hyperparameters didn't transfer\nacross model widths. μP found the correct\nscaling for each tensor type."),
    ("This work (if H1 holds)",
     "Previously: ‖Δθ‖ from different models is\nincomparable. σ½ normalization makes them\ncomparable → σ½ is the natural distance unit."),
]
y = 0.82
for title, body in analogies:
    ax2.text(0.05, y,       "▸ " + title, fontsize=10, fontweight="bold",
             color=BLUE, va="top", transform=ax2.transAxes)
    ax2.text(0.07, y-0.06,  body, fontsize=9, color=BODY_COLOR, va="top",
             transform=ax2.transAxes)
    y -= 0.27
slides.append(("03_h1_hypothesis", fig))


# ══════════════════════════════════════════════════════════════════════════════
# SLIDE 4 — H1 Experiments table
# ══════════════════════════════════════════════════════════════════════════════
fig = new_slide("H1: Experiments Run", "63 trajectories across 5 models and 5 tasks")

ax = fig.add_axes([0.04, 0.08, 0.92, 0.76])
ax.set_facecolor(LIGHT); ax.set_axis_off()

# Table
col_x   = [0.02, 0.22, 0.38, 0.52, 0.68, 0.82]
col_hdrs= ["Model", "Task", "LR sweep", "Status", "Addresses", "Source dir"]
rows_data = [
    ("GPT-2 (124M)",       "SST-2",   "6 LRs", "DONE",
     "H1 cross-task\n(capacity floor)", "outputs_0615_v2"),
    ("GPT-2 (124M)",       "AG News", "6 LRs", "DONE",
     "H1 cross-task\n(primary task)",   "outputs_0615_v2"),
    ("GPT-2-medium (355M)","AG News", "6 LRs", "DONE",
     "H1 cross-model\n(medium scale)",  "outputs_0617_h1"),
    ("GPT-2-medium (355M)","SST-2",   "6 LRs", "DONE",
     "H1 cross-task",                   "outputs_0617_h1"),
    ("GPT-2-medium (355M)","MNLI",    "6 LRs", "DONE",
     "H1 cross-task\n(hard task)",      "outputs_0617_h1"),
    ("GPT-2-medium (355M)","DBPedia", "6 LRs", "DONE",
     "H1 cross-task\n(14-class)",       "outputs_0617_h1"),
    ("GPT-2-medium (355M)","Yahoo",   "6 LRs", "DONE",
     "H1 cross-task\n(10-class)",       "outputs_0617_h1"),
    ("GPT-2-large (774M)", "AG News", "6 LRs", "DONE",
     "H1 cross-model\n(large scale)",   "outputs_0617_h1"),
    ("GPT-2-xl (1.5B)",    "AG News", "6 LRs", "DONE",
     "H1 cross-model\n(xl scale)",      "outputs_0617_h1"),
    ("Llama-3B",           "SST-2",   "6 LRs", "DONE",
     "H1 cross-arch\n(diff. family)",   "outputs_0615_v2"),
]
status_colors = {"DONE": GREEN, "RUNNING": YELLOW, "PENDING": "#94A3B8", "BLOCKED": RED}

y0 = 0.93
ax.text(col_x[0], y0, col_hdrs[0], fontsize=9.5, fontweight="bold", color=TITLE_COLOR, va="top", transform=ax.transAxes)
for j in range(1, len(col_hdrs)):
    ax.text(col_x[j], y0, col_hdrs[j], fontsize=9.5, fontweight="bold", color=TITLE_COLOR, va="top", transform=ax.transAxes)

row_h = 0.081
for i, (model, task, lrs, status, addresses, src) in enumerate(rows_data):
    y = y0 - (i+1)*row_h
    bg = "#F8FAFC" if i % 2 == 0 else "#EFF6FF"
    rect = mpatches.FancyBboxPatch((0, y - 0.005), 1.0, row_h - 0.008,
                                    boxstyle="round,pad=0.002",
                                    facecolor=bg, edgecolor="none",
                                    transform=ax.transAxes)
    ax.add_patch(rect)
    sc = status_colors.get(status, "#94A3B8")
    vals = [model, task, lrs, "", addresses, src]
    for j, v in enumerate(vals):
        ax.text(col_x[j], y + row_h*0.55, v, fontsize=8.5, color=BODY_COLOR,
                va="center", transform=ax.transAxes)
    # status chip
    ax.text(col_x[3] + 0.01, y + row_h*0.55, status, fontsize=8,
            color="white", va="center", transform=ax.transAxes,
            bbox=dict(boxstyle="round,pad=0.25", facecolor=sc, edgecolor="none"))

slides.append(("04_h1_experiments", fig))


# ══════════════════════════════════════════════════════════════════════════════
# SLIDE 5 — H1 Key Figure
# ══════════════════════════════════════════════════════════════════════════════
fig = new_slide("H1: Results — Normalization Diagnostic",
                "Before vs after σ½ normalization (63 trajectories, rank=8)")

img_path = os.path.join(BASE, "results", "figures_h1", "h1_normalization_diagnostic.png")
if os.path.exists(img_path):
    img = mpimg.imread(img_path)
    ax = fig.add_axes([0.03, 0.09, 0.72, 0.76])
    ax.imshow(img); ax.set_axis_off()
else:
    ax = fig.add_axes([0.03, 0.09, 0.72, 0.76])
    ax.text(0.5, 0.5, "[Figure not found — run make_h1_curve.py first]",
            ha="center", va="center", fontsize=12, color="red")
    ax.set_axis_off()

ax2 = fig.add_axes([0.77, 0.09, 0.21, 0.76])
ax2.set_facecolor(LIGHT); ax2.set_axis_off()
ax2.text(0.05, 0.97, "What to read", fontsize=11, fontweight="bold",
         color=TITLE_COLOR, va="top", transform=ax2.transAxes)
callouts = [
    (GREEN,  "✓  After norm: all\n   conditions near\n   PPL≈1.0 for small R"),
    (GREEN,  "✓  Before norm:\n   3 orders of mag\n   spread. After: ~1-2×"),
    (RED,    "✗  Curves do NOT\n   collapse onto one\n   universal function"),
    (YELLOW, "⚠  Llama-3B spikes\n   at R≈7; GPT-2\n   family is milder"),
    (RED,    "✗  Stability boundary\n   ranges R=0.18\n   to R=3.79 (20×)"),
]
y = 0.88
for color, text in callouts:
    ax2.text(0.05, y, text, fontsize=9, color=color, va="top",
             transform=ax2.transAxes,
             bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                       edgecolor=color, alpha=0.8))
    y -= 0.19
slides.append(("05_h1_figure", fig))


# ══════════════════════════════════════════════════════════════════════════════
# SLIDE 6 — H1 Findings
# ══════════════════════════════════════════════════════════════════════════════
fig = new_slide("H1: Findings", "What the 63 trajectories show")

ax = fig.add_axes([0.04, 0.08, 0.44, 0.76])
ax.set_facecolor(LIGHT); ax.set_axis_off()
ax.text(0.04, 0.96, "σ½_pre across GPT-2 family", fontsize=11,
        fontweight="bold", color=TITLE_COLOR, va="top", transform=ax.transAxes)
scale_rows = [
    ("GPT-2 (124M)",        "0.000601", "—"),
    ("GPT-2-medium (355M)", "0.000806", "1.34×"),
    ("GPT-2-large (774M)",  "0.000601", "1.00×"),
    ("GPT-2-xl (1.5B)",     "0.000665", "1.11×"),
    ("Llama-3B (3B)",       "0.000200", "0.33×"),
]
hdrs2 = ["Model", "σ½_pre", "ratio to base"]
ys2 = [0.85] + [0.85 - (i+1)*0.12 for i in range(len(scale_rows))]
for j, h in enumerate(hdrs2):
    ax.text(0.04 + j*0.32, ys2[0], h, fontsize=9, fontweight="bold",
            color=TITLE_COLOR, va="top", transform=ax.transAxes)
for i, row in enumerate(scale_rows):
    c = BLUE if "Llama" in row[0] else BODY_COLOR
    for j, val in enumerate(row):
        ax.text(0.04 + j*0.32, ys2[i+1], val, fontsize=9, color=c,
                va="top", transform=ax.transAxes)

ax.text(0.04, 0.15,
    "⚠  12× parameter scale → only 1.3× σ½ variation.\n"
    "    σ½ is NOT a model capacity measure.",
    fontsize=9, color=RED, va="top", transform=ax.transAxes,
    bbox=dict(boxstyle="round", facecolor="#FEF2F2", edgecolor=RED, alpha=0.6))

ax2 = fig.add_axes([0.54, 0.08, 0.42, 0.76])
ax2.set_facecolor(LIGHT); ax2.set_axis_off()
ax2.text(0.04, 0.96, "Stability boundary R* (PPL>1.05)",
         fontsize=11, fontweight="bold", color=TITLE_COLOR, va="top",
         transform=ax2.transAxes)
boundary_rows = [
    ("gpt2-medium / AG News",   "0.42",  RED),
    ("gpt2-medium / Yahoo",     "0.46",  RED),
    ("gpt2-xl     / AG News",   "0.56",  RED),
    ("gpt2-large  / AG News",   "1.18",  YELLOW),
    ("gpt2-medium / DBPedia",   "1.21",  YELLOW),
    ("gpt2  /      SST-2",      "2.61",  GREEN),
    ("gpt2-medium / MNLI",      "3.79",  GREEN),
    ("gpt2-medium / SST-2",     ">max",  GREEN),
]
ys3 = [0.84 - i*0.094 for i in range(len(boundary_rows))]
for i, (label, val, c) in enumerate(boundary_rows):
    ax2.text(0.04, ys3[i], label, fontsize=9, color=BODY_COLOR, va="top",
             transform=ax2.transAxes)
    ax2.text(0.72, ys3[i], val, fontsize=9, fontweight="bold", color=c,
             va="top", transform=ax2.transAxes)

ax2.text(0.04, 0.08,
    "20× spread in stability boundary.\n"
    "H1 strong claim (universal R≈1) NOT supported.",
    fontsize=9, color=RED, va="top", transform=ax2.transAxes,
    bbox=dict(boxstyle="round", facecolor="#FEF2F2", edgecolor=RED, alpha=0.6))
slides.append(("06_h1_findings", fig))


# ══════════════════════════════════════════════════════════════════════════════
# SLIDE 7 — H1 Redirection
# ══════════════════════════════════════════════════════════════════════════════
fig = new_slide("H1: Redirection", "What the data is pointing toward instead")

sections = [
    ("#FEF2F2", RED,   "✗  What failed",
     [
         "Universal R=1 boundary: NOT supported.",
         "σ½ scales with model capacity: NOT supported.",
         "Stability boundaries vary 20× across tasks/models.",
         "σ½_pre nearly constant across GPT-2 family.",
     ]),
    ("#F0FDF4", GREEN, "✓  What holds",
     [
         "σ½ normalization reduces raw norm spread (100×→20×).",
         "All conditions safe for very small R (PPL≈1.0 region exists).",
         "Llama-3B has smaller σ½ — more capable models sit at\n  sharper minima. σ½ tracks minimum sharpness.",
         "gpt2-xl already knows AG News (89.6%) → fine-tuning\n  is gratuitous → PPL blows up at R=0.18. σ½ predicted\n  this; task accuracy alone did not.",
     ]),
    ("#FFFBEB", YELLOW, "→  Redirection: σ½ as sharpness indicator",
     [
         "New framing: σ½ ∝ 1/sharpness of pretrained minimum.",
         "Better-trained models (Llama) → sharper minima → smaller σ½.",
         "Adaptation succeeds iff R_success < R_stability.",
         "R_success depends on task-representation alignment.",
         "Test: does σ½ correlate with top Hessian eigenvalue?",
     ]),
]
ys_start = [0.60, 0.36, 0.12]
for (bg, border, header, bullets), y0 in zip(sections, ys_start):
    ax = fig.add_axes([0.04, y0, 0.92, 0.22])
    ax.set_facecolor(bg); ax.set_axis_off()
    for sp in ax.spines.values():
        sp.set_color(border); sp.set_linewidth(1.5)
    ax.text(0.01, 0.88, header, fontsize=11, fontweight="bold",
            color=border, va="top", transform=ax.transAxes)
    x_pos = 0.02
    for b in bullets:
        ax.text(x_pos, 0.62 - bullets.index(b)*0.20,
                "• " + b, fontsize=9.5, color=TITLE_COLOR,
                va="top", transform=ax.transAxes)
        x_pos = 0.02
slides.append(("07_h1_redirection", fig))


# ══════════════════════════════════════════════════════════════════════════════
# SLIDE 8 — H2 Hypothesis
# ══════════════════════════════════════════════════════════════════════════════
fig = new_slide("H2: Forgetting Budget  [PAUSED]",
                "Does the certified ball around θ_A predict Task A forgetting?")

ax = fig.add_axes([0.04, 0.08, 0.44, 0.76])
ax.set_facecolor("#DCFCE7"); ax.set_axis_off()
h2_content = [
    ("Claim",
     "After sequential fine-tuning A→B:\n"
     "Task A forgetting occurs when\n"
     "‖θ_B − θ_A‖ / σ½_A > 1\n"
     "and NOT before."),
    ("Stronger claim",
     "Forgetting curves F(R_A) align across:\n"
     "  · model sizes\n"
     "  · task pairs (A→B)\n"
     "  · learning rates\n"
     "The threshold is sharp at R_A = 1."),
    ("Why it matters",
     "Continual learning: safe update budget.\n"
     "RLHF: σ½ is the weight-space analog\n"
     "of the KL budget (π_ref distance).\n"
     "SFT practitioners: when to expect forgetting."),
    ("σ½_A definition",
     "σ½ certified around θ_A using\n"
     "Task A classification cross-entropy\n"
     "(NOT WikiText-2) as the loss.\n"
     "eval_slack = 1e-3 for clf NLL."),
]
y = 0.94
for title, body in h2_content:
    ax.text(0.04, y,       "▸ " + title, fontsize=10.5, fontweight="bold",
            color=TITLE_COLOR, va="top")
    ax.text(0.06, y-0.065, body, fontsize=9.5, color=BODY_COLOR, va="top")
    y -= 0.235

ax2 = fig.add_axes([0.54, 0.08, 0.42, 0.76])
ax2.set_facecolor("#DCFCE7"); ax2.set_axis_off()
ax2.text(0.04, 0.96, "Experimental design", fontsize=11, fontweight="bold",
         color=TITLE_COLOR, va="top", transform=ax2.transAxes)
design = [
    "Phase 1: Fine-tune on Task A.",
    "Measure σ½_A (using Task A clf loss).",
    "Save θ_A.",
    "",
    "Phase 2: Fine-tune on Task B from θ_A.",
    "Every track_interval steps:",
    "  · Measure ‖θ_t − θ_A‖ / σ½_A",
    "  · Measure Task A accuracy",
    "  · Measure Task B accuracy",
    "",
    "Plot Task A accuracy vs R_A.",
    "Hypothesis: drop occurs when R_A crosses 1.",
    "",
    "Task pairs planned:",
    "  GPT-2:    AG News → SST-2",
    "  Llama-3B: SST-2 → AG News",
]
for i, line in enumerate(design):
    ax2.text(0.04, 0.88 - i*0.057, line, fontsize=9, color=BODY_COLOR,
             va="top", transform=ax2.transAxes)
slides.append(("08_h2_hypothesis", fig))


# ══════════════════════════════════════════════════════════════════════════════
# SLIDE 9 — H2 Status
# ══════════════════════════════════════════════════════════════════════════════
fig = new_slide("H2: Status  [PAUSED — bug fixed, re-run needed]",
                "What broke, what was fixed, what's pending")

sections_h2 = [
    ("#FEF2F2", RED, "✗  Attempt 1 failed (outputs_0616_seqForgetting)",
     [
         "Phase 1 used lr=args.lr[0] = 1e-5 instead of intended 1e-4.",
         "GPT-2 / AG News Phase 1 accuracy: 43.9% (near random for 4-class).",
         "σ½_A = 0.018 (SPURIOUS — 30× larger than σ½_pre). Ball is meaningless.",
         "Density curve non-monotonic: base_nll=1.607 > random NLL (ln4=1.386).",
         "All Phase 2 conditions: norm/σ½_A < 0.082. Hypothesis untestable.",
     ]),
    ("#F0FDF4", GREEN, "✓  Bugs fixed in basin_widening_experiment.py",
     [
         "--lr_a arg added AND wired into Phase 1 finetune() call.",
         "--rank_a arg similarly decoupled from Phase 2 sweep.",
         "Guard added: raises RuntimeError if Phase 1 accuracy < 70%.",
         "_get_task_clf_batch() now uses TASK_REGISTRY (handles all tasks).",
         "Phase 1 always re-runs (no cache): classifier + density stay consistent.",
     ]),
    ("#FFFBEB", YELLOW, "→  Pending: re-run with corrected commands",
     [
         "GPT-2:    --task_a agnews --task sst2  --lr_a 1e-4 --train_steps_a 2000",
         "Llama-3B: --task_a sst2   --task agnews --lr_a 5e-5 --train_steps_a 500",
         "Expected Phase 1 accuracy: >80% before proceeding.",
         "DEPENDENCY: H2 is paused until H1 redirection is resolved.",
         "  (If σ½ doesn't predict stability, σ½_A predicting forgetting is weaker.)",
     ]),
]
ys_h2 = [0.60, 0.36, 0.12]
for (bg, border, header, bullets), y0 in zip(sections_h2, ys_h2):
    ax = fig.add_axes([0.04, y0, 0.92, 0.22])
    ax.set_facecolor(bg); ax.set_axis_off()
    for sp in ax.spines.values():
        sp.set_color(border); sp.set_linewidth(1.5)
    ax.text(0.01, 0.88, header, fontsize=10.5, fontweight="bold",
            color=border, va="top", transform=ax.transAxes)
    for bi, b in enumerate(bullets):
        ax.text(0.02, 0.62 - bi*0.20, "• " + b, fontsize=9,
                color=TITLE_COLOR, va="top", transform=ax.transAxes)
slides.append(("09_h2_status", fig))


# ══════════════════════════════════════════════════════════════════════════════
# SLIDE 10 — Open Questions and Decision Tree
# ══════════════════════════════════════════════════════════════════════════════
fig = new_slide("Open Questions & Next Decision Points",
                "What to answer before choosing direction")

ax = fig.add_axes([0.04, 0.08, 0.56, 0.76])
ax.set_facecolor(LIGHT); ax.set_axis_off()
ax.text(0.03, 0.96, "Priority questions", fontsize=12, fontweight="bold",
        color=TITLE_COLOR, va="top", transform=ax.transAxes)
questions = [
    (BLUE,   "Q1 [H1 core]",
     "Does σ½ correlate with top Hessian eigenvalue?\n"
     "If yes → σ½ = certified sharpness measure.\n"
     "Test: compute dominant Hessian eigenvalue for each\n"
     "GPT-2 family model, compare to σ½_pre."),
    (BLUE,   "Q2 [H1 redirection]",
     "Can R_success be estimated from θ_pre + task examples\n"
     "WITHOUT running fine-tuning?\n"
     "If yes → σ½ becomes predictive, not just descriptive.\n"
     "Approach: zero-shot gradient norm in task direction."),
    (GREEN,  "Q3 [H2 prerequisite]",
     "Does Phase 1 converge cleanly with --lr_a 1e-4?\n"
     "Cheap to check: run one condition and verify\n"
     "Phase 1 accuracy >80% before full sweep."),
    (YELLOW, "Q4 [framing]",
     "Is the sharpness story more defensible than\n"
     "the universal boundary story?\n"
     "Depends on Q1. If σ½ ≠ sharpness, need new framing."),
]
y = 0.86
for color, tag, body in questions:
    ax.text(0.03, y, tag, fontsize=10, fontweight="bold", color=color,
            va="top", transform=ax.transAxes)
    ax.text(0.03, y - 0.055, body, fontsize=9, color=BODY_COLOR,
            va="top", transform=ax.transAxes)
    y -= 0.24

ax2 = fig.add_axes([0.64, 0.08, 0.32, 0.76])
ax2.set_facecolor(LIGHT); ax2.set_axis_off()
ax2.text(0.05, 0.96, "Decision tree", fontsize=12, fontweight="bold",
         color=TITLE_COLOR, va="top", transform=ax2.transAxes)
tree = [
    ("Q1: σ½ ∝ sharpness?", BLUE),
    ("  YES → σ½ = certified sharpness.\n  Reframe paper around this.", GREEN),
    ("  NO  → σ½ is neither capacity\n  nor sharpness. Find new frame.", RED),
    ("", BODY_COLOR),
    ("Q2: R_success predictable?", BLUE),
    ("  YES → full predictive theory.\n  Top-conference result.", GREEN),
    ("  NO  → σ½ descriptive only.\n  Workshop / short paper.", YELLOW),
    ("", BODY_COLOR),
    ("Q3: H2 Phase 1 converges?", BLUE),
    ("  YES → run full H2 sweep.", GREEN),
    ("  NO  → debug further or\n  switch to Llama-3B only.", RED),
]
y = 0.88
for text, color in tree:
    ax2.text(0.05, y, text, fontsize=8.5, color=color,
             va="top", transform=ax2.transAxes)
    y -= 0.092
slides.append(("10_open_questions", fig))


# ══════════════════════════════════════════════════════════════════════════════
# Save all slides
# ══════════════════════════════════════════════════════════════════════════════
paths = []
for name, fig in slides:
    p = os.path.join(OUT, f"{name}.png")
    fig.savefig(p, dpi=150, bbox_inches="tight", facecolor=BG)
    plt.close(fig)
    paths.append(p)
    print(f"  Saved {p}")

# Combined PDF
try:
    from matplotlib.backends.backend_pdf import PdfPages
    pdf_path = os.path.join(OUT, "ntrs_research_plan.pdf")
    with PdfPages(pdf_path) as pdf:
        for p in paths:
            img = mpimg.imread(p)
            fig2, ax = plt.subplots(figsize=(W, H))
            fig2.patch.set_facecolor(BG)
            ax.imshow(img); ax.set_axis_off()
            plt.tight_layout(pad=0)
            pdf.savefig(fig2, bbox_inches="tight")
            plt.close(fig2)
    print(f"\nPDF saved: {pdf_path}")
except Exception as e:
    print(f"PDF skipped: {e}")

print(f"\n{len(slides)} slides saved to {OUT}/")
