"""
recompute_sigma_half_acc.py
============================
Recompute σ½_A using classification ACCURACY instead of NLL.

Why: NLL-based σ½_A (0.006607) is 10x too large for Llama because the
language model is robust to Gaussian noise, but the MNLI classification
representation is much more fragile. R_A = per_param_norm / σ½_NLL gives
values far below 1 even when catastrophic forgetting occurs.

Fix: σ½_A_acc = the perturbation σ where 50% of Gaussian perturbations
still keep MNLI classification accuracy above the success threshold
(default: 90% of Phase 1 baseline accuracy).

This script:
  1. Loads Phase 1 checkpoint (model + classifier) from the H2 output dir
  2. Sweeps σ, evaluating MNLI accuracy after each perturbation
  3. Finds σ½_A_acc via half-max interpolation
  4. Reloads all Phase 2 result JSONs, recomputes R_A = per_param_norm / σ½_A_acc
  5. Saves new summary + plots — no Phase 1 or Phase 2 retraining needed

Usage:
  python recompute_sigma_half_acc.py
  python recompute_sigma_half_acc.py --task_a mnli --task_b sst2
  python recompute_sigma_half_acc.py --n_perturb 100 --n_eval 200
"""

import argparse
import json
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(__file__))
from certified_density_experiment import (
    save_weights, restore_weights, perturb_weights,
)
from lora_density_experiment import (
    _load_model_auto, _get_target_names_auto,
    _get_hidden_size, _get_tokenizer, add_lora,
)


def get_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model",     default="meta-llama/Llama-3.2-3B")
    p.add_argument("--task_a",    default="mnli", choices=["mnli", "sst2", "agnews", "cola"])
    p.add_argument("--task_b",    default="sst2", choices=["mnli", "sst2", "agnews", "cola"])
    p.add_argument("--h2_dir",    default="./results/h2_llama",
                   help="Root output dir used by h2_sequential_llama.py")
    p.add_argument("--n_perturb", type=int, default=150,
                   help="Perturbations per sigma (more = tighter estimate)")
    p.add_argument("--n_eval",    type=int, default=200,
                   help="MNLI validation examples per accuracy check (keep small for speed)")
    p.add_argument("--seq_len",   type=int, default=128)
    p.add_argument("--batch_size",type=int, default=16)
    p.add_argument("--acc_slack", type=float, default=0.10,
                   help="Success = accuracy >= baseline_acc * (1 - acc_slack). "
                        "Default 0.10 = within 10pp of Phase 1 accuracy.")
    p.add_argument("--seed",      type=int, default=42)
    p.add_argument("--refine",    action="store_true",
                   help="Add fine-grained sigma points in the cached result's transition zone. "
                        "Loads model, runs new perturbations, merges with cached densities.")
    p.add_argument("--n_refine",  type=int, default=7,
                   help="Number of new sigma points to add in transition zone (--refine only).")
    p.add_argument("--force",     action="store_true",
                   help="Ignore cached sigma_half_acc.json and recompute from scratch.")
    return p.parse_args()


# ── Data ──────────────────────────────────────────────────────────────────────

def load_mnli_val(tokenizer, n_samples, seq_len, batch_size, seed):
    from datasets import load_dataset
    ds = load_dataset("glue", "mnli", split="validation_matched")
    ds = ds.shuffle(seed=seed).select(range(min(n_samples, len(ds))))

    ids, labs = [], []
    for ex in ds:
        text = f"premise: {ex['premise']} hypothesis: {ex['hypothesis']}"
        enc  = tokenizer(text, truncation=True, max_length=seq_len,
                         padding="max_length", return_tensors="pt")
        ids.append(enc["input_ids"].squeeze(0))
        labs.append(ex["label"])

    dataset = torch.utils.data.TensorDataset(
        torch.stack(ids), torch.tensor(labs, dtype=torch.long)
    )
    return torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=False, drop_last=False
    )


def load_sst2_val(tokenizer, n_samples, seq_len, batch_size, seed):
    from datasets import load_dataset
    ds = load_dataset("glue", "sst2", split="validation")
    ds = ds.shuffle(seed=seed).select(range(min(n_samples, len(ds))))

    ids, labs = [], []
    for ex in ds:
        enc = tokenizer(ex["sentence"], truncation=True, max_length=seq_len,
                        padding="max_length", return_tensors="pt")
        ids.append(enc["input_ids"].squeeze(0))
        labs.append(ex["label"])

    dataset = torch.utils.data.TensorDataset(
        torch.stack(ids), torch.tensor(labs, dtype=torch.long)
    )
    return torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=False, drop_last=False
    )


def load_agnews_val(tokenizer, n_samples, seq_len, batch_size, seed):
    from datasets import load_dataset
    ds = load_dataset("ag_news", split="test")
    ds = ds.shuffle(seed=seed).select(range(min(n_samples, len(ds))))

    ids, labs = [], []
    for ex in ds:
        enc = tokenizer(ex["text"], truncation=True, max_length=seq_len,
                        padding="max_length", return_tensors="pt")
        ids.append(enc["input_ids"].squeeze(0))
        labs.append(ex["label"])

    dataset = torch.utils.data.TensorDataset(
        torch.stack(ids), torch.tensor(labs, dtype=torch.long)
    )
    return torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=False, drop_last=False
    )


def load_cola_val(tokenizer, n_samples, seq_len, batch_size, seed):
    from datasets import load_dataset
    ds = load_dataset("glue", "cola", split="validation")
    ds = ds.shuffle(seed=seed).select(range(min(n_samples, len(ds))))

    ids, labs = [], []
    for ex in ds:
        enc = tokenizer(ex["sentence"], truncation=True, max_length=seq_len,
                        padding="max_length", return_tensors="pt")
        ids.append(enc["input_ids"].squeeze(0))
        labs.append(ex["label"])

    dataset = torch.utils.data.TensorDataset(
        torch.stack(ids), torch.tensor(labs, dtype=torch.long)
    )
    return torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=False, drop_last=False
    )


def load_val_for_task(task, tokenizer, n_samples, seq_len, batch_size, seed):
    """Load validation data for the given task name."""
    if task == "mnli":
        return load_mnli_val(tokenizer, n_samples, seq_len, batch_size, seed)
    elif task == "sst2":
        return load_sst2_val(tokenizer, n_samples, seq_len, batch_size, seed)
    elif task == "agnews":
        return load_agnews_val(tokenizer, n_samples, seq_len, batch_size, seed)
    elif task == "cola":
        return load_cola_val(tokenizer, n_samples, seq_len, batch_size, seed)
    else:
        raise ValueError(f"Unknown task: {task}")


# ── Accuracy evaluation ───────────────────────────────────────────────────────

@torch.no_grad()
def eval_accuracy(model, classifier, loader, device):
    model.eval()
    classifier.eval()
    correct, total = 0, 0
    for ids, labs in loader:
        ids  = ids.to(device)
        labs = labs.to(device)
        out  = model(ids, output_hidden_states=True)
        h    = out.hidden_states[-1][:, -1, :].float()
        pred = classifier(h).argmax(dim=-1)
        correct += (pred == labs).sum().item()
        total   += labs.size(0)
    return correct / max(total, 1)


# ── Accuracy-based density estimation ────────────────────────────────────────

def estimate_density_acc(model, classifier, target_names, val_loader,
                          sigma, N, success_threshold, device, seed_offset=0):
    """
    density(σ) = fraction of N random Gaussian perturbations (magnitude σ)
    for which MNLI accuracy stays >= success_threshold.
    """
    snapshot  = save_weights(model, target_names)
    n_success = 0
    t0        = time.time()

    for i in range(N):
        if i % 20 == 0:
            print(f"    σ={sigma:.5f}  {i}/{N}  {time.time()-t0:.0f}s",
                  end="\r", flush=True)
        perturb_weights(model, seed=seed_offset + i, sigma=sigma, names=target_names)
        acc = eval_accuracy(model, classifier, val_loader, device)
        if acc >= success_threshold:
            n_success += 1
        restore_weights(model, snapshot, target_names)

    print(flush=True)
    density = n_success / N
    print(f"    σ={sigma:.5f}  density={density:.3f}  "
          f"(success≥{success_threshold:.3f}, {N} perturbs, {time.time()-t0:.0f}s)",
          flush=True)
    return {"sigma": sigma, "density": density, "ci_lower": density}


def half_max_sigma_from_results(density_results):
    """σ where density drops to half its peak, by linear interpolation."""
    sigmas = [r["sigma"]   for r in density_results]
    dens   = [r["density"] for r in density_results]
    peak   = max(dens)
    target = peak / 2.0
    for i in range(len(dens) - 1):
        if dens[i] >= target > dens[i + 1]:
            t = (target - dens[i]) / (dens[i + 1] - dens[i])
            return sigmas[i] + t * (sigmas[i + 1] - sigmas[i])
    return sigmas[-1]


# ── Plotting ──────────────────────────────────────────────────────────────────

def plot_threshold_summary(all_results, sigma_half_acc, sigma_half_nll,
                            out_path, task_a, task_b):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    colors = plt.cm.tab10(np.linspace(0, 1, len(all_results)))
    for i, r in enumerate(all_results):
        label = f"lr={r['lr']:.0e} r={r['rank']}"
        for ax, key, xlabel in [
            (axes[0], "R_A_acc", "R_A_acc = ‖Δθ‖ / σ½_A_acc"),
            (axes[1], "R_A_nll", "R_A_nll = ‖Δθ‖ / σ½_A_nll  (original)"),
        ]:
            ax.scatter(r[key], r["final_acc_task_a"],
                       color=colors[i], s=120, zorder=3, label=label)

    for ax, title in [
        (axes[0], f"With σ½_acc={sigma_half_acc:.6f}  (accuracy-based)"),
        (axes[1], f"With σ½_nll={sigma_half_nll:.6f}  (NLL-based, original)"),
    ]:
        ax.axvline(x=1.0, color="red", linestyle="--", alpha=0.8, label="R_A=1")
        ax.set_xlabel(ax.get_xlabel() if hasattr(ax, '_xlabel') else "", fontsize=11)
        ax.set_ylabel(f"{task_a} Accuracy (after Phase 2)", fontsize=11)
        ax.set_title(title, fontsize=10)
        ax.legend(fontsize=7, bbox_to_anchor=(1.01, 1), loc="upper left")
        ax.grid(True, alpha=0.3)

    axes[0].set_xlabel("R_A_acc = ‖Δθ‖ / σ½_A_acc", fontsize=11)
    axes[1].set_xlabel("R_A_nll = ‖Δθ‖ / σ½_A_nll  (original)", fontsize=11)

    plt.suptitle(f"H2: R_A threshold comparison — {task_a}→{task_b}  Llama-3.2-3B",
                 fontsize=12, y=1.01)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out_path}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    args   = get_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(args.seed)

    tag     = f"{args.task_a}_to_{args.task_b}"
    run_dir = os.path.join(args.h2_dir, tag)

    model_path = os.path.join(run_dir, "phase1_model.pt")
    cls_path   = os.path.join(run_dir, "phase1_classifier.pt")
    meta_path  = os.path.join(run_dir, "phase1_meta.json")
    nll_path   = os.path.join(run_dir, "phase1_sigma_half.json")
    acc_path   = os.path.join(run_dir, "phase1_sigma_half_acc.json")
    summary_path = os.path.join(run_dir, "summary.json")

    for path in [model_path, cls_path, meta_path, nll_path, summary_path]:
        if not os.path.exists(path):
            print(f"ERROR: missing {path}")
            print("Run h2_sequential_llama.py first to generate Phase 1 checkpoint.")
            return 1

    print(f"\n{'='*60}")
    print(f"  Recompute σ½_A using accuracy — {tag}")
    print(f"  n_perturb={args.n_perturb}  n_eval={args.n_eval}  acc_slack={args.acc_slack}")
    print(f"{'='*60}\n")

    # ── Load Phase 1 metadata ─────────────────────────────────────────────────
    with open(meta_path) as f:
        meta = json.load(f)
    with open(nll_path) as f:
        nll_result = json.load(f)

    sigma_half_nll  = nll_result["sigma_half"]
    baseline_acc    = meta.get("acc_phase1", 0.836)
    n_classes       = meta["n_classes"]
    success_threshold = baseline_acc * (1 - args.acc_slack)

    print(f"  Phase 1 {args.task_a.upper()} accuracy:  {baseline_acc:.4f}")
    print(f"  Success threshold:      {success_threshold:.4f}  "
          f"(baseline × {1-args.acc_slack:.2f})")
    print(f"  σ½_A_nll (original):   {sigma_half_nll:.6f}\n")

    # ── Load model + classifier ───────────────────────────────────────────────
    print("Loading Phase 1 model...")
    tokenizer = _get_tokenizer(args.model)
    model, _  = _load_model_auto(args.model, device)

    if meta.get("save_format") == "lora_only":
        lora_tgt = meta.get("lora_target_modules", ["q_proj", "v_proj", "k_proj", "o_proj"])
        peft_m = add_lora(model, meta["rank"], meta.get("alpha", 16),
                          target_modules=lora_tgt)
        peft_m.load_state_dict(
            torch.load(model_path, map_location=device, weights_only=False), strict=False
        )
        model = peft_m.merge_and_unload().eval()
    else:
        model.load_state_dict(
            torch.load(model_path, map_location=device, weights_only=False)
        )
    model.eval()

    hidden_size = _get_hidden_size(model)
    classifier  = nn.Linear(hidden_size, n_classes).to(device)
    classifier.load_state_dict(
        torch.load(cls_path, map_location=device, weights_only=False)
    )
    classifier.eval()

    target_names = _get_target_names_auto(model, args.model)

    # ── Validation loader — MUST match task_a (the Phase 1 task) ─────────────
    print(f"Loading {args.n_eval} {args.task_a.upper()} validation examples...")
    val_loader = load_val_for_task(
        args.task_a, tokenizer, args.n_eval, args.seq_len, args.batch_size, seed=args.seed
    )

    # Confirm baseline accuracy on this subset
    baseline_acc_check = eval_accuracy(model, classifier, val_loader, device)
    print(f"  Baseline accuracy on {args.n_eval} samples: {baseline_acc_check:.4f}")
    success_threshold = baseline_acc_check * (1 - args.acc_slack)
    print(f"  Adjusted success threshold: {success_threshold:.4f}\n")

    # ── Load or compute σ½_A_acc ──────────────────────────────────────────────
    if os.path.exists(acc_path) and not args.force:
        with open(acc_path) as f:
            acc_result = json.load(f)
        sigma_half_acc = acc_result["sigma_half"]

        if args.refine:
            # ── Refine: add fine-grained sigma points in the transition zone ──
            existing = sorted((float(s), d) for s, d in acc_result["densities"].items())
            peak     = max(d for _, d in existing)
            target   = peak / 2.0

            lo_sigma = hi_sigma = lo_dens = hi_dens = None
            for i in range(len(existing) - 1):
                s_lo, d_lo = existing[i]
                s_hi, d_hi = existing[i + 1]
                if d_lo >= target > d_hi:
                    lo_sigma, hi_sigma = s_lo, s_hi
                    lo_dens,  hi_dens  = d_lo, d_hi
                    break

            if lo_sigma is None:
                print("  Could not find transition zone — density may be flat. Skipping refine.")
            else:
                print(f"  Transition zone: σ=[{lo_sigma:.6f}, {hi_sigma:.6f}]  "
                      f"density=[{lo_dens:.3f}, {hi_dens:.3f}]")
                existing_sigmas = {s for s, _ in existing}
                new_sigmas = [
                    s for s in np.linspace(lo_sigma, hi_sigma, args.n_refine + 2)[1:-1]
                    if not any(abs(s - e) < 1e-9 for e in existing_sigmas)
                ]
                print(f"  Adding {len(new_sigmas)} sigma points: "
                      f"{[f'{s:.6f}' for s in new_sigmas]}\n")

                for sigma in new_sigmas:
                    res = estimate_density_acc(
                        model, classifier, target_names, val_loader,
                        sigma=sigma, N=args.n_perturb,
                        success_threshold=success_threshold,
                        device=device, seed_offset=args.seed,
                    )
                    existing_sigmas.add(sigma)
                    existing.append((sigma, res["density"]))

                existing.sort()
                density_results_full = [{"sigma": s, "density": d} for s, d in existing]
                old_sh     = sigma_half_acc
                sigma_half_acc = half_max_sigma_from_results(density_results_full)

                print(f"\n  σ½_A_acc updated: {old_sh:.6f} → {sigma_half_acc:.6f}")
                acc_result["sigma_half"] = sigma_half_acc
                acc_result["densities"]  = {str(s): d for s, d in existing}
                with open(acc_path, "w") as f:
                    json.dump(acc_result, f, indent=2)
                print(f"  Saved: {acc_path}")
        else:
            print(f"  Loaded cached σ½_A_acc = {sigma_half_acc:.6f}")
            print("  (Use --refine to add fine-grained sigma points in the transition zone.)")
    else:
        if args.force and os.path.exists(acc_path):
            print("  --force: ignoring cached result, recomputing from scratch.\n")

        # Sigma range: geometric sweep, then stop early when density < 0.05
        s0     = sigma_half_nll / 100
        sigmas = [s0 * (2 ** k) for k in range(10)]
        sigmas = [s for s in sigmas if s <= sigma_half_nll * 2]
        print(f"  Sigma sweep: {[f'{s:.5f}' for s in sigmas]}")
        print(f"  Perturbations per σ: {args.n_perturb}")
        print(f"  Eval examples per perturbation: {args.n_eval}\n")

        density_results = []
        for sigma in sigmas:
            res = estimate_density_acc(
                model, classifier, target_names, val_loader,
                sigma=sigma, N=args.n_perturb,
                success_threshold=success_threshold,
                device=device, seed_offset=args.seed,
            )
            density_results.append(res)
            if res["density"] < 0.05:
                print(f"  Density below 0.05 at σ={sigma:.5f}, stopping sweep early.")
                break

        sigma_half_acc = half_max_sigma_from_results(density_results)
        print(f"\n  σ½_A_acc = {sigma_half_acc:.6f}")
        print(f"  σ½_A_nll = {sigma_half_nll:.6f}")
        print(f"  Ratio σ½_nll / σ½_acc = {sigma_half_nll / sigma_half_acc:.2f}x\n")

        acc_result = {
            "sigma_half":        sigma_half_acc,
            "sigma_half_nll":    sigma_half_nll,
            "baseline_acc":      baseline_acc_check,
            "success_threshold": success_threshold,
            "acc_slack":         args.acc_slack,
            "densities": {str(r["sigma"]): r["density"] for r in density_results},
        }
        with open(acc_path, "w") as f:
            json.dump(acc_result, f, indent=2)
        print(f"  Saved: {acc_path}")

    # ── Recompute R_A for all Phase 2 conditions ──────────────────────────────
    print("\nRecomputing R_A for all Phase 2 conditions...")
    with open(summary_path) as f:
        all_results = json.load(f)

    updated = []
    for r in all_results:
        per_param_norm = r["final_per_param"]
        r["R_A_acc"] = per_param_norm / sigma_half_acc if sigma_half_acc > 0 else float("nan")
        r["R_A_nll"] = per_param_norm / sigma_half_nll if sigma_half_nll > 0 else float("nan")
        updated.append(r)

    # ── Print updated table ───────────────────────────────────────────────────
    print(f"\n{'─'*75}")
    print(f"{'LR':>8}  {'Rank':>5}  {'R_A_acc':>9}  {'R_A_nll':>9}  "
          f"{'Acc_A':>7}  {'Forgot(acc)?':>13}")
    print(f"{'─'*75}")
    for r in sorted(updated, key=lambda x: (x["rank"], x["lr"])):
        forgot = "YES" if r["R_A_acc"] > 1.0 else "no"
        print(f"  {r['lr']:>8.0e}  {r['rank']:>5}  {r['R_A_acc']:>9.4f}  "
              f"{r['R_A_nll']:>9.4f}  {r['final_acc_task_a']:>7.4f}  {forgot:>13}")
    print(f"{'─'*75}")
    print(f"\n  σ½_A_acc = {sigma_half_acc:.6f}")
    print(f"  σ½_A_nll = {sigma_half_nll:.6f}  ({sigma_half_nll/sigma_half_acc:.1f}x larger)\n")

    # ── Save updated summary + plots ──────────────────────────────────────────
    updated_summary_path = os.path.join(run_dir, "summary_acc_sigma.json")
    with open(updated_summary_path, "w") as f:
        json.dump(updated, f, indent=2)

    plot_threshold_summary(
        updated, sigma_half_acc, sigma_half_nll,
        out_path=os.path.join(run_dir, "threshold_comparison_acc_vs_nll.png"),
        task_a=args.task_a, task_b=args.task_b,
    )

    return 0


if __name__ == "__main__":
    sys.exit(main())
