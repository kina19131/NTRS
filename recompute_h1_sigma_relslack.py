"""
recompute_h1_sigma_relslack.py
==============================
Recompute σ½_pre for GPT-2 and Llama-3.2-3B using a STANDARDIZED
relative eval_slack = 1% of baseline WikiText-2 NLL (instead of the
original fixed absolute slacks: 1e-4 for GPT-2, 0.01 for Llama).

This addresses the eval_slack inconsistency that weakens the σ½ scale-
invariance claim in H1.

Usage:
    CUDA_VISIBLE_DEVICES=0 python recompute_h1_sigma_relslack.py --model gpt2
    CUDA_VISIBLE_DEVICES=1 python recompute_h1_sigma_relslack.py --model meta-llama/Llama-3.2-3B

Output:
    h1_kl/<model>/pretrained_density_relslack.json
    Prints comparison table: old σ½ (fixed slack) vs new σ½ (1% relative slack)
"""

import argparse, json, os, torch
import numpy as np

from lora_density_experiment import _load_model_auto, _get_tokenizer, _get_target_names_auto
from certified_density_experiment import half_max_sigma_adaptive
from h1_kl_comparison import _get_lora_scoped_names

RELATIVE_SLACK = 0.01   # 1% of baseline NLL


def load_wikitext_nll_data(tokenizer, n_samples=50, seq_len=128, device="cpu", seed=42):
    from datasets import load_dataset
    torch.manual_seed(seed)
    ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    text = "\n".join(ds["text"])
    enc = tokenizer(text, return_tensors="pt", truncation=False)["input_ids"][0]
    step = seq_len + 1
    seqs = []
    for i in range(0, len(enc) - step, step):
        seqs.append(enc[i : i + step])
        if len(seqs) >= n_samples:
            break
    return torch.stack(seqs).to(device)


def compute_nll(model, ids, device):
    model.eval()
    inp, tgt = ids[:, :-1], ids[:, 1:]
    with torch.no_grad():
        out = model(inp)
        logits = out.logits if hasattr(out, "logits") else out[0]
        loss = torch.nn.functional.cross_entropy(
            logits.reshape(-1, logits.size(-1)), tgt.reshape(-1)
        )
    return loss.item()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="gpt2",
                   help="HF model name: 'gpt2' or 'meta-llama/Llama-3.2-3B'")
    p.add_argument("--n_perturb", type=int, default=150)
    p.add_argument("--n_samples", type=int, default=50)
    p.add_argument("--seq_len",   type=int, default=128)
    p.add_argument("--seed",      type=int, default=42)
    p.add_argument("--out_dir",   default=None,
                   help="Override output dir (default: h1_kl/<model>/)")
    p.add_argument("--sigma_scope", default="lora", choices=["lora", "block"],
                   help="Parameter scope perturbed to calibrate σ½. 'lora' "
                        "(default): only LoRA's actual target modules — "
                        "matches what fine-tuning displaces, fixes the scope "
                        "mismatch flagged in SUMMARY.md. 'block': legacy "
                        "whole-decoder-block scope used by every σ½ value "
                        "currently in SUMMARY.md/pretrained_density_relslack.json; "
                        "kept only to reproduce those numbers.")
    args = p.parse_args()

    if args.sigma_scope == "lora":
        print("=" * 70)
        print("  sigma_scope=lora (DEFAULT as of 2026-07-21)")
        print("  This does NOT reproduce the numbers in SUMMARY.md /")
        print("  pretrained_density_relslack.json — those were all computed")
        print("  with the legacy whole-decoder-block scope.")
        print("  Pass --sigma_scope block to reproduce historical numbers.")
        print("=" * 70)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    print(f"Model: {args.model}")

    # Output directory
    model_tag = args.model.replace("/", os.sep)
    out_dir   = args.out_dir or os.path.join("h1_kl", model_tag)
    os.makedirs(out_dir, exist_ok=True)

    # Load original σ½ for comparison
    orig_path = os.path.join(out_dir, "pretrained_density.json")
    orig_sigma = None
    orig_slack = None
    if os.path.exists(orig_path):
        with open(orig_path) as f:
            orig = json.load(f)
        orig_sigma = orig["sigma_half"]
        orig_slack = 0.01 if "llama" in args.model.lower() else 1e-4

    # Load pretrained model (no fine-tuning)
    print("\nLoading pretrained model...")
    model, _ = _load_model_auto(args.model, device)
    tokenizer = _get_tokenizer(args.model)
    block_names = _get_target_names_auto(model, args.model)
    target_names = (_get_lora_scoped_names(model, args.model)
                    if args.sigma_scope == "lora" else block_names)
    assert len(target_names) > 0, (
        f"sigma_scope='{args.sigma_scope}' matched ZERO parameters for "
        f"{args.model} — before spending any GPU time, check _lora_targets() "
        f"module names against this model's actual named_parameters()."
    )
    # numel-based, not tensor-count-based — see h1_kl_comparison.py for why
    # tensor-count alone is a misleading fraction of perturbed weight-mass.
    n_target_params = sum(p.numel() for n, p in model.named_parameters()
                          if n in target_names)
    n_block_params  = sum(p.numel() for n, p in model.named_parameters()
                          if n in block_names)
    n_total_params  = sum(p.numel() for _, p in model.named_parameters())
    print(f"  sigma_scope={args.sigma_scope}: perturbing {n_target_params:,}/"
          f"{n_total_params:,} params ({100*n_target_params/n_total_params:.1f}%) "
          f"across {len(target_names)} tensors")
    if args.sigma_scope == "lora":
        print(f"  for reference: whole-block scope would be {n_block_params:,} "
              f"params ({100*n_block_params/n_total_params:.1f}%) — lora scope "
              f"is {100*n_target_params/n_block_params:.1f}% of that")

    # Compute baseline WikiText-2 NLL
    print("Computing baseline WikiText-2 NLL...")
    ids = load_wikitext_nll_data(tokenizer, args.n_samples, args.seq_len, device, args.seed)
    nll_base = compute_nll(model, ids, device)
    print(f"  NLL_baseline = {nll_base:.5f}")

    # Standardized relative slack
    eval_slack = RELATIVE_SLACK * nll_base
    print(f"  Relative slack = {RELATIVE_SLACK*100:.0f}% × {nll_base:.4f} = {eval_slack:.5f}")
    if orig_slack is not None:
        print(f"  Original fixed slack = {orig_slack:.5g}  "
              f"({orig_slack/nll_base*100:.3f}% of baseline NLL)")

    # Run density estimation across sigma grid (same base grid as original).
    # This grid was calibrated for whole-block perturbation. A narrower
    # 'lora' scope perturbs fewer independent parameters, so σ½ can land
    # higher than this grid reaches — half_max_sigma_adaptive extends the top
    # (capped) and warns loudly instead of silently returning a clipped value
    # indistinguishable from a real measurement.
    base_sigmas = [0.0001, 0.0002, 0.0005, 0.001, 0.002, 0.005, 0.01, 0.02, 0.05]
    inp_data = ids[:, :-1]
    tgt_data = ids[:, 1:]

    print(f"\nEstimating density ({len(base_sigmas)}+ sigma values × {args.n_perturb} perturbations)...")
    new_sigma, density_results, was_capped = half_max_sigma_adaptive(
        model, target_names, inp_data, tgt_data,
        base_sigmas=base_sigmas, N=args.n_perturb,
        eval_slack=eval_slack, tag=f"{args.model}/{args.sigma_scope}",
    )
    print(f"\nσ½ (rel slack {RELATIVE_SLACK*100:.0f}%) = {new_sigma:.6f}"
          + ("  [LOWER BOUND — see warning above]" if was_capped else ""))

    # Save. 'lora' scope writes to its own filename — it must never overwrite
    # pretrained_density_relslack.json, which is the block-scope value every
    # T*/R number in SUMMARY.md is currently computed from.
    out_filename = ("pretrained_density_relslack.json" if args.sigma_scope == "block"
                     else "pretrained_density_relslack_lorascope.json")
    out_path = os.path.join(out_dir, out_filename)
    with open(out_path, "w") as f:
        json.dump({
            "model": args.model,
            "sigma_half": new_sigma,
            "relative_slack": RELATIVE_SLACK,
            "eval_slack_abs": eval_slack,
            "nll_baseline": nll_base,
            "sigma_half_original": orig_sigma,
            "original_fixed_slack": orig_slack,
            "sigma_scope": args.sigma_scope,
            "sigma_half_is_lower_bound": was_capped,
            "n_target_params": n_target_params,
            "n_total_params": n_total_params,
            "densities": {
                str(r["sigma"]): r["ci_lower"] if "ci_lower" in r else r.get("density", r.get("fraction", 0))
                for r in density_results
            },
        }, f, indent=2)
    print(f"Saved: {out_path}")

    # Comparison summary
    print("\n── Comparison ──────────────────────────────────────────")
    print(f"  {'Metric':<30} {'Original':>12} {'Rel-slack 1%':>12}")
    slack_pct = f"{orig_slack/nll_base*100:.3f}%" if orig_slack is not None else "N/A"
    print(f"  {'eval_slack (abs)':<30} {str(orig_slack) if orig_slack is not None else 'N/A':>12} {eval_slack:>12.5f}")
    print(f"  {'eval_slack (% of NLL)':<30} {slack_pct:>12} {RELATIVE_SLACK*100:>11.1f}%")
    print(f"  {'σ½_pre':<30} {orig_sigma if orig_sigma is not None else 'N/A':>12} {new_sigma:>12.6f}")
    if orig_sigma is not None:
        print(f"  {'Change in σ½':<30} {'—':>12} {(new_sigma-orig_sigma)/orig_sigma*100:>+11.1f}%")
    print("────────────────────────────────────────────────────────")


if __name__ == "__main__":
    main()
