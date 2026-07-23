"""
H1 Universal Adaptation Curve
x = ||Δθ|| / σ½   (normalized update radius R)
y = WikiText-2 PPL ratio  (NLL_ft / NLL_pre — "retained pretraining quality")

Run with all outputs dirs you want to include, e.g.:
  python make_h1_curve.py --dirs outputs_0615_v2 outputs_0617_h1

Reads every widening_summary.json found under --dirs, aggregates all trajectories,
and plots (1) before-normalization, (2) after-normalization, and (3) a combined
two-panel figure for the paper.
"""

import argparse
import json
import os
import glob

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BASE = os.path.dirname(os.path.abspath(__file__))


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--dirs", nargs="+",
                   default=["outputs_0615_v2"],
                   help="Output directories to scan for widening_summary.json files.")
    p.add_argument("--out", default="results/figures_h1",
                   help="Directory to save figures.")
    p.add_argument("--rank", type=int, default=8,
                   help="Which LoRA rank to plot (default 8).")
    return p.parse_args()


# ── Load all summary files ─────────────────────────────────────────────────────
def load_all_records(dirs, rank_filter):
    """
    Returns list of dicts with:
      model, task, lr, R, norm, sigma_half, ppl_ratio, ppl_ratio_actual, task_acc
    One entry per (model, task, lr) condition matching rank_filter.
    """
    records = []
    for d in dirs:
        pattern = os.path.join(BASE, d, "**/widening_summary.json")
        for path in glob.glob(pattern, recursive=True):
            with open(path) as f:
                summ = json.load(f)
            model = summ["model"]
            task  = summ["task"]
            for row in summ["results"]:
                if row["rank"] != rank_filter:
                    continue
                records.append({
                    "model":            model,
                    "task":             task,
                    "lr":               row["lr"],
                    "R":                row["norm_over_sh"],
                    "norm":             row["norm"],
                    "sigma_half":       row["sigma_half_pre"],
                    "ppl_ratio":        row["ppl_ratio"],        # NLL_ft / NLL_pre
                    "ppl_ratio_actual": row["ppl_ratio_actual"], # exp(ΔNLL), true PPL ratio
                    "task_acc":         row["task_val_acc"],
                    "source":           path,
                })
    return records


# ── Assign visual style ───────────────────────────────────────────────────────
MODEL_COLOR = {
    "gpt2":                             "#1D4ED8",  # dark blue
    "gpt2-medium":                      "#2563EB",  # blue
    "gpt2-large":                       "#60A5FA",  # light blue
    "gpt2-xl":                          "#93C5FD",  # very light blue
    "meta-llama/Llama-3.2-3B":         "#DC2626",  # red
    "meta-llama/Llama-3.2-1B":         "#F87171",  # light red
    "EleutherAI/pythia-160m":           "#7C3AED",  # purple
    "EleutherAI/pythia-410m":           "#A855F7",
    "EleutherAI/pythia-1b":             "#C084FC",
    "EleutherAI/pythia-1.4b":          "#DDD6FE",
}
TASK_MARKER = {
    "sst2":    "o",
    "agnews":  "s",
    "mnli":    "^",
    "dbpedia": "D",
    "yahoo":   "P",
    "wikitext2": "x",
}
TASK_LS = {
    "sst2":      "-",
    "agnews":    "--",
    "mnli":      "-.",
    "dbpedia":   ":",
    "yahoo":     "--",
    "wikitext2": "-",
}

def _style(model, task):
    color  = MODEL_COLOR.get(model, "#6B7280")
    marker = TASK_MARKER.get(task, "o")
    ls     = TASK_LS.get(task, "-")
    short_model = model.split("/")[-1]
    label  = f"{short_model} / {task}"
    return color, marker, ls, label


def _group(records, model, task):
    pts = [(r["R"], r["ppl_ratio"], r["task_acc"], r["norm"])
           for r in records if r["model"] == model and r["task"] == task]
    if not pts:
        return None
    pts.sort()
    Rs, ppls, accs, norms = zip(*pts)
    return list(Rs), list(ppls), list(accs), list(norms)


# ── Figures ───────────────────────────────────────────────────────────────────
def make_figures(records, out_dir, rank_filter):
    os.makedirs(out_dir, exist_ok=True)

    combos = sorted({(r["model"], r["task"]) for r in records},
                    key=lambda x: (x[0], x[1]))
    if not combos:
        print("No data found — check --dirs and --rank.")
        return

    print(f"Found {len(records)} data points across {len(combos)} (model, task) combinations:")
    for m, t in combos:
        pts = [r for r in records if r["model"] == m and r["task"] == t]
        print(f"  {m:42s} / {t:10s}  {len(pts)} LR conditions")

    # ── Figure 1: Normalization diagnostic (before vs after) ─────────────────
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

    for model, task in combos:
        g = _group(records, model, task)
        if g is None:
            continue
        Rs, ppls, accs, _ = g
        color, marker, ls, label = _style(model, task)
        sh = [r["sigma_half"] for r in records
              if r["model"] == model and r["task"] == task][0]
        raw_norms = [r * sh for r in Rs]

        ax1.plot(raw_norms, ppls, ls, color=color, marker=marker,
                 markersize=6, linewidth=1.6, label=label)
        ax2.plot(Rs, ppls, ls, color=color, marker=marker,
                 markersize=6, linewidth=1.6, label=label)

    for ax in (ax1, ax2):
        ax.axhline(y=1.0, color="gray", linewidth=0.6, alpha=0.4)
        ax.set_ylabel("WikiText-2 NLL ratio  (ft / pre)", fontsize=10)
        ax.grid(True, alpha=0.3, which="both")
        ax.set_ylim(0.97, max(5.0, max(r["ppl_ratio"] for r in records) * 1.05))

    ax1.set_xscale("log")
    ax1.set_xlabel("‖Δθ‖  (raw parameter distance)", fontsize=10)
    ax1.set_title("Before σ½ normalization", fontsize=11)
    ax1.legend(fontsize=7.5, loc="upper left", ncol=1)

    ax2.axvline(x=1.0, color="black", linestyle=":", linewidth=1.5, alpha=0.8)
    ax2.axvspan(0, 1.0, color="#DCFCE7", alpha=0.35, label="Inside σ½ ball (R<1)")
    ax2.axvspan(1.0, 200, color="#FEF2F2", alpha=0.25, label="Outside (R>1)")
    ax2.set_xscale("log")
    ax2.set_xlabel(r"$R = \|\Delta\theta\| \,/\, \sigma_{1/2}$", fontsize=10)
    ax2.set_title(r"After σ½ normalization  ($R = \|\Delta\theta\|/\sigma_{1/2}$)", fontsize=11)
    ax2.legend(fontsize=7.5, loc="upper left", ncol=1)

    fig.suptitle(f"H1: Does σ½ normalization collapse the curves? (rank={rank_filter})",
                 fontsize=12, y=1.01)
    fig.tight_layout()
    out = os.path.join(out_dir, "h1_normalization_diagnostic.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved {out}")

    # ── Figure 2: Main two-panel (PPL ratio + task acc vs R) ─────────────────
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

    for model, task in combos:
        g = _group(records, model, task)
        if g is None:
            continue
        Rs, ppls, accs, _ = g
        color, marker, ls, label = _style(model, task)
        ax1.plot(Rs, ppls, ls, color=color, marker=marker, markersize=6,
                 linewidth=1.6, label=label)
        ax2.plot(Rs, accs, ls, color=color, marker=marker, markersize=6,
                 linewidth=1.6, label=label)

    for ax, ylabel, ylim in [
        (ax1, "WikiText-2 NLL ratio  (ft / pre)",
         (0.97, max(5.0, max(r["ppl_ratio"] for r in records) * 1.05))),
        (ax2, "Downstream task accuracy", (0.2, 1.05)),
    ]:
        ax.axvline(x=1.0, color="black", linestyle=":", linewidth=1.5, alpha=0.8)
        ax.axvspan(0, 1.0, color="#DCFCE7", alpha=0.35)
        ax.axvspan(1.0, 200, color="#FEF2F2", alpha=0.25)
        ax.set_xscale("log")
        ax.set_xlabel(r"$R = \|\Delta\theta\| \,/\, \sigma_{1/2}$", fontsize=10)
        ax.set_ylabel(ylabel, fontsize=10)
        ax.set_ylim(*ylim)
        ax.grid(True, alpha=0.3, which="both")
        if ylabel.startswith("WikiText"):
            ax.axhline(y=1.0, color="gray", linewidth=0.6, alpha=0.4)

    ax1.legend(fontsize=7.5, loc="upper left", ncol=1)
    ax2.legend(fontsize=7.5, loc="lower right", ncol=1)
    ax1.set_title("Pretraining retention vs R", fontsize=11)
    ax2.set_title("Task accuracy vs R", fontsize=11)

    n_traj = len(records)
    n_models = len({r["model"] for r in records})
    n_tasks  = len({r["task"]  for r in records})
    fig.suptitle(
        rf"H1: $\sigma_{{1/2}}$ as universal adaptation scale"
        f"  —  {n_traj} trajectories, {n_models} models, {n_tasks} tasks  (rank={rank_filter})",
        fontsize=11, y=1.01)
    fig.tight_layout()
    out = os.path.join(out_dir, "h1_two_panel.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved {out}")

    # ── Figure 3: R boundary per (model, task) ────────────────────────────────
    print(f"\n=== R where NLL ratio first exceeds threshold ===")
    thresholds = [1.01, 1.05, 1.10]
    for model, task in combos:
        g = _group(records, model, task)
        if g is None:
            continue
        Rs, ppls, accs, _ = g
        sh = [r["sigma_half"] for r in records if r["model"] == model and r["task"] == task][0]
        short = model.split("/")[-1]
        print(f"  {short:20s} / {task:10s}  σ½={sh:.5f}")
        for thr in thresholds:
            exceed = [r for r, p in zip(Rs, ppls) if p > thr]
            first  = f"{min(exceed):.2f}" if exceed else ">max"
            print(f"    PPL_ratio > {thr:.2f}  at R = {first}")


if __name__ == "__main__":
    args = parse_args()
    records = load_all_records(args.dirs, args.rank)
    make_figures(records, os.path.join(BASE, args.out), args.rank)
