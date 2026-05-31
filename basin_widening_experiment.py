"""
basin_widening_experiment.py
============================
Tests whether LoRA basin widening (σ½ increases after lr=1e-4 fine-tuning) is:

  A. Task-agnostic:   does it hold when fine-tuning ON WikiText-2 (same eval task)?
                      If yes → can't be orthogonality artifact.
  B. Rank-dependent:  does σ½ widening grow monotonically with LoRA rank?
                      If yes → LoRA's rank structure is the mechanism.
  C. Steps-robust:    does σ½ keep widening as training progresses?
                      If yes → not just "barely moved."

These three checks together support the claim:
  "LoRA's low-rank constraint acts as an implicit flattening regularizer,
   consistently widening the pretrained certified basin across tasks and ranks."

Usage:
  # Rank ablation — SST-2 task
  python basin_widening_experiment.py --model gpt2 --task sst2 \\
      --ranks 1 2 4 8 16 32 --lr 1e-4 --train_steps 500

  # Rank ablation — WikiText-2 CLM (rules out task-orthogonality)
  python basin_widening_experiment.py --model gpt2 --task wikitext2 \\
      --ranks 1 2 4 8 16 32 --lr 1e-4 --train_steps 500

  # Steps ablation (rank fixed at 8)
  python basin_widening_experiment.py --model gpt2 --task sst2 \\
      --ranks 8 --lr 1e-4 --train_steps 100 500 1000 2000

  # Reload saved results and regenerate summary only
  python basin_widening_experiment.py --model gpt2 --task sst2 \\
      --ranks 1 2 4 8 16 32 --replot

Output:
  results/basin_widening/{model}/{task}/
    rank{r}_steps{n}_density.json   — full density curve post-LoRA
    pretrained_density.json         — baseline (computed once, reused)
    widening_summary.json           — [{rank, steps, sigma_half_pre, sigma_half_ft, ratio, norm}, ...]
    widening_vs_rank.png            — key figure: widening ratio vs rank
    density_curves.png              — all density curves overlaid
"""

import argparse
import copy
import json
import os
import sys

import numpy as np
import torch
import torch.nn as nn
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(__file__))
from certified_density_experiment import (
    load_data, estimate_density, half_max_sigma,
)
from lora_density_experiment import (
    _default_target_modules, _load_model_auto,
    _get_target_names_auto, _get_hidden_size, _get_tokenizer,
    add_lora, load_sst2,
)


# ── Args ──────────────────────────────────────────────────────────────────────
def get_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model",  default="gpt2")
    p.add_argument("--task",   default="sst2", choices=["sst2", "wikitext2"],
                   help="Fine-tuning task. 'wikitext2' uses CLM on WikiText-2 train "
                        "split — same distribution as eval, ruling out task-orthogonality.")
    p.add_argument("--ranks",  nargs="+", type=int, default=[1, 2, 4, 8, 16, 32],
                   help="LoRA ranks to sweep")
    p.add_argument("--lora_alpha",   type=int,   default=None,
                   help="LoRA alpha. Default: None = set equal to rank (scale=1). "
                        "Set explicitly to fix alpha across all ranks (conflates "
                        "rank with effective LR — avoid for rank ablation).")
    p.add_argument("--lr",           type=float, default=1e-4)
    p.add_argument("--train_steps",  nargs="+", type=int, default=[500],
                   help="Training steps (multiple values → steps ablation)")
    p.add_argument("--train_batch",  type=int,   default=16)
    p.add_argument("--sigmas", nargs="+", type=float,
                   default=[0.0001, 0.0002, 0.0005, 0.001, 0.002, 0.005,
                            0.01,  0.02,   0.05,   0.1,   0.2])
    p.add_argument("--n_perturb",      type=int,   default=200)
    p.add_argument("--n_text_samples", type=int,   default=128)
    p.add_argument("--seq_len",        type=int,   default=128)
    p.add_argument("--eval_slack",     type=float, default=1e-4)
    p.add_argument("--finetune_fp32",  action="store_true")
    p.add_argument("--grad_clip",      type=float, default=1.0)
    p.add_argument("--replot",         action="store_true",
                   help="Skip fine-tuning; reload existing JSONs and regenerate plots.")
    p.add_argument("--output_dir",     default="./results/basin_widening")
    return p.parse_args()


# ── WikiText-2 CLM data loader ────────────────────────────────────────────────
def load_wikitext2_clm(tokenizer, batch_size, max_len=128, seed=42):
    """
    Load WikiText-2 *train* split for causal LM fine-tuning.
    Using CLM objective on WikiText-2 means fine-tuning and eval share the
    same loss surface — any σ½ widening cannot be explained by task orthogonality.
    """
    from datasets import load_dataset
    ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
    text = "\n\n".join(t for t in ds["text"] if t.strip())

    tokens = tokenizer.encode(text)
    chunks = [tokens[i : i + max_len]
              for i in range(0, len(tokens) - max_len, max_len)]

    rng = np.random.default_rng(seed)
    rng.shuffle(chunks)

    input_ids = torch.tensor(chunks, dtype=torch.long)
    dataset   = torch.utils.data.TensorDataset(input_ids)
    return torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=True, drop_last=True
    )


# ── Fine-tuning ───────────────────────────────────────────────────────────────
def finetune(base_model, tokenizer, task, lr, train_steps, batch_size,
             rank, alpha, device, target_modules, finetune_fp32=False,
             grad_clip=1.0, seed=42):
    """
    Fine-tune with LoRA. Task selects the training objective:
      sst2      — classification head on SST-2 (cross-entropy, 2 classes)
      wikitext2 — causal LM on WikiText-2 train (no classifier head)

    alpha=None resolves to rank (scale=1), keeping effective LR constant
    across the rank ablation so only subspace dimensionality varies.

    Returns (merged_model, per_param_norm, resolved_alpha).
    """
    alpha = rank if alpha is None else alpha
    torch.manual_seed(seed)
    print(f"\n  Fine-tuning | task={task}  rank={rank}  alpha={alpha}  "
          f"lr={lr:.1e}  steps={train_steps}  fp32={finetune_fp32}", flush=True)

    pretrained_snapshot = {n: p.data.clone().float()
                           for n, p in base_model.named_parameters()}

    model_copy = copy.deepcopy(base_model)
    if finetune_fp32:
        model_copy = model_copy.float()
    peft_model = add_lora(model_copy, rank, alpha, target_modules)

    if task == "sst2":
        hidden_size = _get_hidden_size(base_model)
        classifier  = nn.Linear(hidden_size, 2).to(device)
        params      = list(peft_model.parameters()) + list(classifier.parameters())
        loader      = load_sst2(tokenizer, batch_size)
        loss_fn     = nn.CrossEntropyLoss()
    else:  # wikitext2 CLM
        classifier  = None
        params      = list(peft_model.parameters())
        loader      = load_wikitext2_clm(tokenizer, batch_size)
        loss_fn     = None

    opt = torch.optim.AdamW([p for p in params if p.requires_grad], lr=lr)

    peft_model.train()
    if classifier is not None:
        classifier.train()

    step = 0
    while step < train_steps:
        for batch in loader:
            if step >= train_steps:
                break

            if task == "sst2":
                input_ids, labels = batch
                input_ids = input_ids.to(device)
                labels    = labels.to(device)
                out     = peft_model(input_ids, output_hidden_states=True)
                hidden  = out.hidden_states[-1][:, -1, :].float()
                loss    = loss_fn(classifier(hidden), labels)
            else:
                input_ids = batch[0].to(device)
                out       = peft_model(input_ids, labels=input_ids)
                loss      = out.loss

            if torch.isnan(loss):
                print(f"    step {step}: NaN loss, skipping", flush=True)
                opt.zero_grad()
                step += 1
                continue

            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(params, grad_clip)
            opt.step()
            step += 1

            if step % 100 == 0:
                print(f"    step {step}/{train_steps}  loss={loss.item():.4f}",
                      flush=True)

    merged = peft_model.merge_and_unload().eval()

    total_sq, total_n = 0.0, 0
    for name, param in merged.named_parameters():
        if name in pretrained_snapshot:
            delta   = param.data.float() - pretrained_snapshot[name]
            total_sq += delta.norm().item() ** 2
            total_n  += delta.numel()
    per_param_norm = float((total_sq / max(total_n, 1)) ** 0.5)
    print(f"  Per-parameter norm: {per_param_norm:.6f}", flush=True)

    return merged, per_param_norm, alpha


# ── Density sweep ─────────────────────────────────────────────────────────────
def run_density(model, names, inp, tgt, sigmas, n_perturb, eval_slack, tag):
    results = []
    for sigma in sigmas:
        r = estimate_density(model, names, inp, tgt, sigma,
                             N=n_perturb, eval_slack=eval_slack, tag=tag)
        results.append(r)
    return results


# ── Summary JSON ──────────────────────────────────────────────────────────────
def build_summary(records):
    """records: list of dicts with keys rank, steps, pre, ft, norm."""
    out = []
    for rec in records:
        sh_pre = half_max_sigma(rec["pre"])
        sh_ft  = half_max_sigma(rec["ft"])
        ratio  = sh_ft / sh_pre if sh_pre > 0 else float("nan")
        ppl_pre = rec["pre"][0]["base_ppl"]
        ppl_ft  = rec["ft"][0]["base_ppl"]
        out.append({
            "rank":           rec["rank"],
            "steps":          rec["steps"],
            "sigma_half_pre": sh_pre,
            "sigma_half_ft":  sh_ft,
            "widening_ratio": ratio,
            "norm":           rec["norm"],
            "norm_over_sh":   rec["norm"] / sh_pre if sh_pre > 0 else float("nan"),
            "ppl_pre":        ppl_pre,
            "ppl_ft":         ppl_ft,
            "ppl_ratio":      ppl_ft / ppl_pre if ppl_pre > 0 else float("nan"),
        })
    return out


# ── Plots ─────────────────────────────────────────────────────────────────────
def plot_widening_vs_rank(summary, output_dir, model, task):
    """
    Key figure: σ½ widening ratio vs LoRA rank.
    One line per step count. Horizontal line at 1.0 = no change.
    """
    from collections import defaultdict
    by_steps = defaultdict(list)
    for rec in summary:
        by_steps[rec["steps"]].append(rec)
    for v in by_steps.values():
        v.sort(key=lambda r: r["rank"])

    _, axes = plt.subplots(1, 2, figsize=(14, 5))

    colors = plt.cm.Blues(np.linspace(0.45, 0.9, len(by_steps)))
    for (n_steps, recs), color in zip(sorted(by_steps.items()), colors):
        ranks   = [r["rank"]           for r in recs]
        ratios  = [r["widening_ratio"] for r in recs]
        norms   = [r["norm_over_sh"]   for r in recs]
        axes[0].plot(ranks, ratios, marker="o", color=color, lw=2,
                     label=f"steps={n_steps}")
        axes[1].plot(ranks, norms,  marker="s", color=color, lw=2,
                     label=f"steps={n_steps}")

    axes[0].axhline(1.0, color="gray", lw=1.5, ls="--", label="no change")
    axes[0].set_xlabel("LoRA rank", fontsize=12)
    axes[0].set_ylabel("σ½ widening ratio  (σ½_after / σ½_before)", fontsize=11)
    axes[0].set_title(f"Basin Widening vs LoRA Rank\n{model}  task={task}", fontsize=12)
    axes[0].legend(fontsize=9)
    axes[0].grid(True, alpha=0.3)
    axes[0].set_xscale("log", base=2)

    axes[1].axhline(1.0, color="gray", lw=1.5, ls="--", label="norm = σ½")
    axes[1].set_xlabel("LoRA rank", fontsize=12)
    axes[1].set_ylabel("norm(ΔW) / σ½_pretrained", fontsize=11)
    axes[1].set_title(f"Update Norm / σ½ vs Rank\n(should be << 1 for widening to hold)",
                      fontsize=12)
    axes[1].legend(fontsize=9)
    axes[1].grid(True, alpha=0.3)
    axes[1].set_xscale("log", base=2)

    plt.tight_layout()
    path = os.path.join(output_dir, "widening_vs_rank.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    print(f"Saved: {path}")
    plt.close()


def plot_density_curves(pre_results, all_ft_results, output_dir, model, task):
    """Overlay all density curves: pretrained + each (rank, steps) condition."""
    _, ax = plt.subplots(figsize=(12, 6))

    sigmas = [r["sigma"] for r in pre_results]
    ax.plot(sigmas, [r["density"] for r in pre_results],
            color="black", lw=2.5, marker="o", ms=4,
            label="pretrained", zorder=10)
    ax.fill_between(sigmas,
                    [r["ci_lower"] for r in pre_results],
                    [r["ci_upper"] for r in pre_results],
                    color="black", alpha=0.10)

    ranks_all = sorted(set(r["rank"]   for r in all_ft_results))
    steps_all = sorted(set(r["steps"]  for r in all_ft_results))
    colors = plt.cm.Reds(np.linspace(0.3, 0.9, len(ranks_all)))

    for rank, color in zip(ranks_all, colors):
        for steps in steps_all:
            match = [r for r in all_ft_results
                     if r["rank"] == rank and r["steps"] == steps]
            if not match:
                continue
            ft = match[0]["ft"]
            ls = "-" if steps == steps_all[0] else "--"
            ax.plot([r["sigma"] for r in ft],
                    [r["density"] for r in ft],
                    color=color, lw=1.5, ls=ls, marker="s", ms=3,
                    label=f"r={rank} s={steps}")

    ax.set_xscale("log")
    ax.set_xlabel("σ", fontsize=12)
    ax.set_ylabel("C(θ, σ)  — certified density", fontsize=12)
    ax.set_title(f"Density Curves: Pretrained vs LoRA (lr=1e-4)\n"
                 f"{model}  task={task}", fontsize=12)
    ax.legend(fontsize=8, ncol=3)
    ax.grid(True, alpha=0.3)
    ax.axhline(0.40, color="gray", lw=1, ls=":")
    plt.tight_layout()
    path = os.path.join(output_dir, "density_curves.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    print(f"Saved: {path}")
    plt.close()


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    args   = get_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    out_dir = os.path.join(args.output_dir, args.model.replace("/", "_"), args.task)
    os.makedirs(out_dir, exist_ok=True)
    print(f"Device: {device}  |  Output: {out_dir}")

    tokenizer      = _get_tokenizer(args.model)
    target_modules = _default_target_modules(args.model)

    print(f"\nLoading {args.model}...")
    model, _ = _load_model_auto(args.model, device)
    names    = _get_target_names_auto(model, args.model)
    inp, tgt = load_data(tokenizer, args.n_text_samples, args.seq_len, device)

    # ── Pretrained density (compute once) ─────────────────────────────────────
    pre_json = os.path.join(out_dir, "pretrained_density.json")
    if os.path.exists(pre_json):
        print("Reloading pretrained density...")
        with open(pre_json) as f:
            pre_results = json.load(f)["sigma_results"]
    else:
        print("\n[Pretrained] Density sweep...")
        pre_results = run_density(model, names, inp, tgt,
                                  args.sigmas, args.n_perturb,
                                  args.eval_slack, tag="pretrained")
        with open(pre_json, "w") as f:
            json.dump({"model": args.model, "condition": "pretrained",
                       "sigma_results": pre_results}, f, indent=2)

    sh_pre = half_max_sigma(pre_results)
    print(f"\nPretrained σ½ = {sh_pre:.6f}  "
          f"ppl = {pre_results[0]['base_ppl']:.4f}")

    # ── Sweep ranks × steps ───────────────────────────────────────────────────
    records     = []   # for summary JSON and plots
    ft_meta     = []   # for density overlay plot

    for steps in args.train_steps:
        for rank in args.ranks:
            tag      = f"rank{rank}_steps{steps}"
            ft_json  = os.path.join(out_dir, f"{tag}_density.json")
            nrm_json = os.path.join(out_dir, f"{tag}_norm.json")

            if args.replot and os.path.exists(ft_json):
                print(f"Reloading {tag}...")
                with open(ft_json)  as f: ft_data  = json.load(f)
                with open(nrm_json) as f: nrm_data = json.load(f)
                ft_results  = ft_data["sigma_results"]
                per_param_norm = nrm_data["per_param_norm"]
            else:
                ft_model, per_param_norm, resolved_alpha = finetune(
                    model, tokenizer,
                    task         = args.task,
                    lr           = args.lr,
                    train_steps  = steps,
                    batch_size   = args.train_batch,
                    rank         = rank,
                    alpha        = args.lora_alpha,
                    device       = device,
                    target_modules = target_modules,
                    finetune_fp32  = args.finetune_fp32,
                    grad_clip      = args.grad_clip,
                )
                ft_names = _get_target_names_auto(ft_model, args.model)

                print(f"\n[{tag}] Density sweep...")
                ft_results = run_density(ft_model, ft_names, inp, tgt,
                                         args.sigmas, args.n_perturb,
                                         args.eval_slack, tag=tag)
                del ft_model
                if device.type == "cuda":
                    torch.cuda.empty_cache()

                with open(ft_json, "w") as f:
                    json.dump({"model": args.model, "task": args.task,
                               "rank": rank, "alpha": resolved_alpha,
                               "steps": steps, "lr": args.lr,
                               "per_param_norm": per_param_norm,
                               "sigma_results": ft_results}, f, indent=2)
                with open(nrm_json, "w") as f:
                    json.dump({"rank": rank, "steps": steps,
                               "per_param_norm": per_param_norm,
                               "sigma_half_pre": sh_pre}, f, indent=2)

            sh_ft = half_max_sigma(ft_results)
            ratio = sh_ft / sh_pre if sh_pre > 0 else float("nan")
            print(f"  {tag}: σ½ {sh_pre:.6f} → {sh_ft:.6f}  "
                  f"ratio={ratio:.3f}  norm/σ½={per_param_norm/sh_pre:.3f}")

            records.append({"rank": rank, "steps": steps,
                            "pre": pre_results, "ft": ft_results,
                            "norm": per_param_norm})
            ft_meta.append({"rank": rank, "steps": steps,
                            "ft": ft_results, "norm": per_param_norm})

    # ── Summary JSON ──────────────────────────────────────────────────────────
    summary = build_summary(records)
    sum_json = os.path.join(out_dir, "widening_summary.json")
    with open(sum_json, "w") as f:
        json.dump({"model": args.model, "task": args.task,
                   "lr": args.lr, "results": summary}, f, indent=2)
    print(f"\nSaved summary: {sum_json}")

    # Print table
    print(f"\n{'rank':>6}  {'steps':>6}  {'σ½_pre':>10}  {'σ½_ft':>10}  "
          f"{'ratio':>7}  {'norm/σ½':>8}  {'ppl_ratio':>10}")
    print("-" * 68)
    for rec in sorted(summary, key=lambda r: (r["steps"], r["rank"])):
        print(f"  {rec['rank']:>4}  {rec['steps']:>6}  "
              f"{rec['sigma_half_pre']:>10.6f}  {rec['sigma_half_ft']:>10.6f}  "
              f"{rec['widening_ratio']:>7.3f}  "
              f"{rec['norm_over_sh']:>8.3f}  "
              f"{rec['ppl_ratio']:>10.4f}")

    # ── Plots ─────────────────────────────────────────────────────────────────
    plot_widening_vs_rank(summary, out_dir, args.model, args.task)
    plot_density_curves(pre_results, ft_meta, out_dir, args.model, args.task)

    print(f"\nDone. Results in: {out_dir}")


if __name__ == "__main__":
    main()
