"""
seed_variance_experiment.py
============================
Experiment proposed by Linyi Li (2026-07-14):

Train two (or more) LoRA models from the SAME pretrained initialization
with DIFFERENT data-loading seeds, then measure whether the weight
difference between runs (‖θ_s1 − θ_s2‖ per-param) is bounded by σ½_pre.

Framing: "counterintuitive but universal bound" — if σ½ bounds not just
forgetting from random perturbations but also the inherent stochasticity
of fine-tuning from data ordering, it is a universal scale for the
pre-training basin.

Same init = same LoRA A/B initialization (fixed torch seed before add_lora).
Different data = different ds.shuffle seed AND different DataLoader generator.

Outputs (in seed_variance/<model_tag>/<task>/):
  lr<X>_seed<Y>_result.json    — per-seed metrics (acc, per_param_norm, etc.)
  lr<X>_pairwise.json          — pairwise ‖θ_s1 − θ_s2‖ for all seed pairs
  summary.json                 — one row per LR: mean/max pairwise dist vs σ½

Usage:
  python seed_variance_experiment.py
  python seed_variance_experiment.py --model meta-llama/Llama-3.2-3B --steps 500
  python seed_variance_experiment.py --lrs 1e-5 5e-5 1e-4 --n_seeds 5
"""

import argparse
import copy
import json
import os
import time

import numpy as np
import torch
import torch.nn as nn

# Re-use helpers from existing scripts — read-only imports, no modifications.
from h1_kl_comparison import (
    compute_per_param_norm,
    load_wikitext_for_kl, _lora_targets,
)
from lora_density_experiment import (
    add_lora, _load_model_auto, _get_tokenizer,
    _get_hidden_size,
)
from certified_density_experiment import compute_nll_loss


def load_sigma_data(tokenizer, n_samples=50, seq_len=128, device="cpu"):
    """WikiText-2 test split, sequential windows — same protocol as recompute_h1_sigma_relslack.py."""
    from datasets import load_dataset
    ds   = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    text = "\n".join(ds["text"])
    enc  = tokenizer(text, return_tensors="pt", truncation=False)["input_ids"][0]
    step = seq_len + 1
    seqs = []
    for i in range(0, len(enc) - step, step):
        seqs.append(enc[i : i + step])
        if len(seqs) >= n_samples:
            break
    ids = torch.stack(seqs)
    return ids[:, :-1].to(device), ids[:, 1:].to(device)


def get_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model",      default="gpt2")
    p.add_argument("--task",       default="sst2", choices=["sst2"])
    p.add_argument("--lrs",        nargs="+", type=float,
                   default=[5e-5, 1e-4, 2e-4])
    p.add_argument("--rank",       type=int,   default=8)
    p.add_argument("--alpha",      type=int,   default=16)
    p.add_argument("--steps",      type=int,   default=500)
    p.add_argument("--batch_size", type=int,   default=16)
    p.add_argument("--grad_accum", type=int,   default=1)
    p.add_argument("--grad_clip",  type=float, default=1.0)
    p.add_argument("--n_seeds",    type=int,   default=5,
                   help="Number of data seeds to run per LR")
    p.add_argument("--data_seeds", nargs="+", type=int,
                   default=[42, 123, 456, 789, 1337],
                   help="Data-loading seeds (overrides n_seeds if provided)")
    p.add_argument("--init_seed",  type=int,   default=0,
                   help="Fixed seed for LoRA weight initialization (same across all runs)")
    p.add_argument("--output_dir", default="./seed_variance")
    p.add_argument("--sigma_file", default=None,
                   help="Path to pretrained_density*.json with sigma_half. "
                        "Auto-detected from h1_kl/ if not given.")
    p.add_argument("--sigma_half", type=float, default=None,
                   help="σ½_pre value directly (skips file lookup). "
                        "e.g. 7.191e-3 for GPT-2, 7.717e-4 for Llama-3.2-3B, "
                        "1.824e-4 for Mistral-7B-v0.1")
    p.add_argument("--force",      action="store_true")
    return p.parse_args()


# ── σ½ auto-detection ─────────────────────────────────────────────────────────

def find_sigma_file(model: str):
    """
    Return path to an existing pretrained_density*.json for this model.
    Prefers the relslack version. Raises if nothing found.
    """
    # Try common patterns from h1_kl directory
    candidates = [
        f"h1_kl/gpt2/pretrained_density_relslack.json",
        f"h1_kl/meta-llama/meta-llama/Llama-3.2-3B/pretrained_density_relslack.json",
        f"h1_kl/mistralai/Mistral-7B-v0.1_sst2_v5/pretrained_density.json",
    ]
    model_lower = model.lower()
    for c in candidates:
        if not os.path.exists(c):
            continue
        if "gpt2" in c and "gpt2" in model_lower:
            return c
        if "llama" in c and "llama" in model_lower:
            return c
        if "mistral" in c and "mistral" in model_lower:
            return c
    raise FileNotFoundError(
        f"No pretrained_density*.json found for {model}. Pass --sigma_file."
    )


def load_sigma_half(path: str):
    with open(path) as f:
        j = json.load(f)
    return j["sigma_half"]


# ── Data loading with explicit seed ──────────────────────────────────────────

def load_sst2_seeded(tokenizer, batch_size, data_seed, max_len=128):
    """
    Same as lora_density_experiment.load_sst2 but with an explicit DataLoader
    generator seed, separate from the global torch RNG (so it doesn't interfere
    with the LoRA initialization seed).
    """
    from datasets import load_dataset
    ds = load_dataset("glue", "sst2", split="train")
    ds = ds.shuffle(seed=data_seed)          # dataset-level order

    input_ids_list, label_list = [], []
    for text, label in zip(ds["sentence"], ds["label"]):
        enc = tokenizer(text, truncation=True, max_length=max_len,
                        padding="max_length", return_tensors="pt")
        input_ids_list.append(enc["input_ids"].squeeze(0))
        label_list.append(label)

    dataset = torch.utils.data.TensorDataset(
        torch.stack(input_ids_list),
        torch.tensor(label_list, dtype=torch.long),
    )
    gen = torch.Generator()
    gen.manual_seed(data_seed)               # DataLoader batch-shuffle seed
    return torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=True,
        drop_last=True, generator=gen,
    )


def load_mnli_seeded(tokenizer, batch_size, data_seed, max_length=128):
    from datasets import load_dataset
    ds = load_dataset("glue", "mnli", split="train")
    ds = ds.select(range(min(4000, len(ds))))
    ds = ds.shuffle(seed=data_seed)

    all_ids, all_labels = [], []
    for ex in ds:
        text = f"Premise: {ex['premise']} Hypothesis: {ex['hypothesis']}"
        enc = tokenizer(text, truncation=True, max_length=max_length,
                        padding="max_length", return_tensors="pt")
        all_ids.append(enc["input_ids"].squeeze(0))
        all_labels.append(ex["label"])

    dataset = torch.utils.data.TensorDataset(
        torch.stack(all_ids),
        torch.tensor(all_labels, dtype=torch.long),
    )
    gen = torch.Generator()
    gen.manual_seed(data_seed)
    return torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=True,
        drop_last=True, generator=gen,
    )


# ── Fine-tuning with separated init/data seeds ───────────────────────────────

def finetune_fixed_init(base_model, tokenizer, lr, steps, batch_size, rank,
                        alpha, grad_clip, device, init_seed, data_seed,
                        model_name="gpt2", task="sst2", grad_accum=1):
    """
    Fine-tune base_model with a fixed LoRA initialization seed (init_seed)
    but a variable data-loading seed (data_seed).

    Returns the merged model (CPU) and a linear classifier.
    """
    # ── Fixed init: LoRA weights initialized with init_seed ──────────────────
    torch.manual_seed(init_seed)
    model_copy = copy.deepcopy(base_model).to(device)
    peft_model = add_lora(model_copy, rank, alpha,
                          target_modules=_lora_targets(model_name))

    hidden_size = _get_hidden_size(base_model)
    num_classes = 3 if task == "mnli" else 2
    classifier  = nn.Linear(hidden_size, num_classes).to(device)

    # ── Variable data: seeded DataLoader, independent of init_seed ───────────
    if task == "mnli":
        loader = load_mnli_seeded(tokenizer, batch_size, data_seed)
    else:
        loader = load_sst2_seeded(tokenizer, batch_size, data_seed)

    params  = list(peft_model.parameters()) + list(classifier.parameters())
    opt     = torch.optim.AdamW([p for p in params if p.requires_grad], lr=lr)
    loss_fn = nn.CrossEntropyLoss()

    peft_model.train()
    classifier.train()

    loader_iter = iter(loader)
    def next_batch():
        nonlocal loader_iter
        try:
            return next(loader_iter)
        except StopIteration:
            loader_iter = iter(loader)
            return next(loader_iter)

    step = 0
    while step < steps:
        opt.zero_grad()
        accum_loss = 0.0
        skip = False
        for _ in range(grad_accum):
            input_ids, labels = next_batch()
            input_ids = input_ids.to(device)
            labels    = labels.to(device)
            out    = peft_model(input_ids, output_hidden_states=True)
            hidden = out.hidden_states[-1][:, -1, :].float()
            logits = classifier(hidden)
            loss   = loss_fn(logits, labels) / grad_accum
            if torch.isnan(loss):
                skip = True
                break
            loss.backward()
            accum_loss += loss.item()
        if skip:
            step += 1
            continue
        nn.utils.clip_grad_norm_(params, grad_clip)
        opt.step()
        step += 1
        if step % 100 == 0:
            print(f"      step {step}/{steps}  loss={accum_loss:.4f}", flush=True)

    merged = peft_model.merge_and_unload().eval().cpu()
    classifier = classifier.cpu()
    return merged, classifier


# ── Accuracy ─────────────────────────────────────────────────────────────────

@torch.no_grad()
def eval_sst2_accuracy(model, classifier, tokenizer, device, n_samples=500):
    from datasets import load_dataset
    ds = load_dataset("glue", "sst2", split="validation")
    ds = ds.select(range(min(n_samples, len(ds))))
    model.eval(); classifier.eval()
    correct, total = 0, 0
    for ex in ds:
        enc = tokenizer(ex["sentence"], truncation=True, max_length=128,
                        padding="max_length", return_tensors="pt")
        out    = model(enc["input_ids"].to(device), output_hidden_states=True)
        hidden = out.hidden_states[-1][:, -1, :].float()
        pred   = classifier(hidden).argmax(dim=-1).item()
        correct += int(pred == ex["label"])
        total   += 1
    return correct / max(total, 1)


# ── Pairwise distance ─────────────────────────────────────────────────────────

def pairwise_per_param_norm(state_dict_a, state_dict_b):
    """Per-param RMS of (θ_a − θ_b)."""
    total_sq, total_d = 0.0, 0
    for name, pa in state_dict_a.items():
        if name in state_dict_b:
            delta     = pa.float() - state_dict_b[name].float()
            total_sq += delta.norm().item() ** 2
            total_d  += delta.numel()
    return float((total_sq / max(total_d, 1)) ** 0.5)


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    args   = get_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    data_seeds = args.data_seeds[:args.n_seeds]

    # ── Output dir ────────────────────────────────────────────────────────────
    model_tag = args.model.split("/")[-1]
    out_dir   = os.path.join(args.output_dir, model_tag, args.task)
    os.makedirs(out_dir, exist_ok=True)

    # ── σ½_pre ────────────────────────────────────────────────────────────────
    if args.sigma_half is not None:
        sigma_half_pre = args.sigma_half
        sigma_path     = "cli-arg"
    else:
        sigma_path     = args.sigma_file or find_sigma_file(args.model)
        sigma_half_pre = load_sigma_half(sigma_path)
    print(f"\n  σ½_pre = {sigma_half_pre:.6e}  (from {sigma_path})")

    # ── Load base model ───────────────────────────────────────────────────────
    print(f"  Loading {args.model}...")
    base_model, _ = _load_model_auto(args.model, device)
    tokenizer     = _get_tokenizer(args.model)

    # Base NLL for reference
    inp_s, tgt_s = load_sigma_data(tokenizer, 50, 128, device)
    nll_base = compute_nll_loss(base_model, inp_s, tgt_s)
    print(f"  Base NLL (test split, n=50) = {nll_base:.5f}")

    wiki_ids = load_wikitext_for_kl(tokenizer, 200, 64, device, seed=0)
    wiki_inp = wiki_ids[:, :-1]
    wiki_tgt = wiki_ids[:, 1:]
    nll_wiki_base = compute_nll_loss(base_model, wiki_inp, wiki_tgt)

    large_model = "gpt2" not in args.model.lower()
    if large_model:
        base_model = base_model.cpu()
        torch.cuda.empty_cache()

    # ── LR sweep ─────────────────────────────────────────────────────────────
    all_summary = []

    for lr in args.lrs:
        lr_tag = f"lr{lr:.0e}"
        print(f"\n{'='*60}")
        print(f"  {lr_tag}  init_seed={args.init_seed}  data_seeds={data_seeds}")
        print(f"{'='*60}")

        seed_state_dicts = {}   # data_seed → CPU state dict
        seed_results     = []

        for data_seed in data_seeds:
            out_path = os.path.join(out_dir, f"{lr_tag}_seed{data_seed}_result.json")
            weights_path = os.path.join(out_dir, f"{lr_tag}_seed{data_seed}_weights.pt")

            if os.path.exists(weights_path) and os.path.exists(out_path) and not args.force:
                print(f"  [seed={data_seed}] Loading cached result + weights...")
                with open(out_path) as f:
                    result = json.load(f)
                seed_state_dicts[data_seed] = torch.load(weights_path, map_location="cpu")
                seed_results.append(result)
                continue

            print(f"\n  [seed={data_seed}] Fine-tuning (init_seed={args.init_seed})...")
            t0 = time.time()

            if large_model:
                base_model = base_model.to(device)

            ft_model, classifier = finetune_fixed_init(
                base_model, tokenizer, lr=lr, steps=args.steps,
                batch_size=args.batch_size, rank=args.rank, alpha=args.alpha,
                grad_clip=args.grad_clip, device=device,
                init_seed=args.init_seed, data_seed=data_seed,
                model_name=args.model, task=args.task,
                grad_accum=args.grad_accum,
            )
            # ft_model and classifier are already on CPU from finetune_fixed_init

            if large_model:
                base_model = base_model.cpu()
                torch.cuda.empty_cache()

            # Displacement from base (σ½ normalized)
            base_for_norm = base_model.to("cpu")
            per_param_norm = compute_per_param_norm(base_for_norm, ft_model)
            R = per_param_norm / sigma_half_pre

            # Task accuracy
            ft_model = ft_model.to(device)
            classifier = classifier.to(device)
            acc = eval_sst2_accuracy(ft_model, classifier, tokenizer, device)

            # NLL
            nll_ft = compute_nll_loss(ft_model, wiki_inp, wiki_tgt)
            delta_nll = nll_ft - nll_wiki_base

            ft_model = ft_model.cpu()
            classifier = classifier.cpu()
            torch.cuda.empty_cache()

            elapsed = time.time() - t0
            print(f"  [seed={data_seed}] R={R:.5f}  acc={acc:.4f}  "
                  f"Δnll={delta_nll:+.4f}  ({elapsed:.0f}s)")

            # Save weight state dict for pairwise comparison
            sd = {n: p.data.clone() for n, p in ft_model.named_parameters()}
            torch.save(sd, weights_path)
            seed_state_dicts[data_seed] = sd

            result = {
                "lr":              lr,
                "data_seed":       data_seed,
                "init_seed":       args.init_seed,
                "rank":            args.rank,
                "steps":           args.steps,
                "per_param_norm":  per_param_norm,
                "sigma_half_pre":  sigma_half_pre,
                "R":               R,
                "acc":             acc,
                "delta_nll_wiki":  delta_nll,
                "elapsed_s":       elapsed,
            }
            with open(out_path, "w") as f:
                json.dump(result, f, indent=2)
            seed_results.append(result)

        # ── Pairwise distances ────────────────────────────────────────────────
        pairwise_path = os.path.join(out_dir, f"{lr_tag}_pairwise.json")

        if os.path.exists(pairwise_path) and not args.force:
            print(f"\n  [{lr_tag}] Loading cached pairwise distances...")
            with open(pairwise_path) as f:
                pairwise = json.load(f)
        else:
            print(f"\n  [{lr_tag}] Computing pairwise distances...")
            pairwise = []
            seeds_list = list(seed_state_dicts.keys())
            for i in range(len(seeds_list)):
                for j in range(i + 1, len(seeds_list)):
                    s1, s2 = seeds_list[i], seeds_list[j]
                    dist = pairwise_per_param_norm(
                        seed_state_dicts[s1], seed_state_dicts[s2]
                    )
                    r_ratio = dist / sigma_half_pre
                    print(f"    seed{s1} vs seed{s2}: "
                          f"dist={dist:.6e}  dist/σ½={r_ratio:.5f}")
                    pairwise.append({
                        "seed_a":      s1,
                        "seed_b":      s2,
                        "pairwise_dist": dist,
                        "dist_over_sigma_half": r_ratio,
                    })
            with open(pairwise_path, "w") as f:
                json.dump(pairwise, f, indent=2)

        dists   = [p["pairwise_dist"] for p in pairwise]
        ratios  = [p["dist_over_sigma_half"] for p in pairwise]
        accs    = [r["acc"] for r in seed_results]
        norms   = [r["per_param_norm"] for r in seed_results]
        Rs      = [r["R"] for r in seed_results]

        lr_summary = {
            "lr":               lr,
            "sigma_half_pre":   sigma_half_pre,
            "n_seeds":          len(data_seeds),
            "data_seeds":       data_seeds,
            "init_seed":        args.init_seed,
            "acc_mean":         float(np.mean(accs)),
            "acc_std":          float(np.std(accs)),
            "R_mean":           float(np.mean(Rs)),
            "R_std":            float(np.std(Rs)),
            "per_param_norm_mean": float(np.mean(norms)),
            "per_param_norm_std":  float(np.std(norms)),
            "pairwise_dist_mean":  float(np.mean(dists)),
            "pairwise_dist_min":   float(np.min(dists)),
            "pairwise_dist_max":   float(np.max(dists)),
            "pairwise_ratio_mean": float(np.mean(ratios)),
            "pairwise_ratio_min":  float(np.min(ratios)),
            "pairwise_ratio_max":  float(np.max(ratios)),
        }
        all_summary.append(lr_summary)

        print(f"\n  [{lr_tag}] σ½_pre={sigma_half_pre:.4e}")
        print(f"  [{lr_tag}] Displacement from base: "
              f"R_mean={lr_summary['R_mean']:.4f} ± {lr_summary['R_std']:.4f}")
        print(f"  [{lr_tag}] Pairwise inter-seed dist: "
              f"mean={lr_summary['pairwise_dist_mean']:.4e}  "
              f"max={lr_summary['pairwise_dist_max']:.4e}  "
              f"max/σ½={lr_summary['pairwise_ratio_max']:.5f}")

    # ── Save summary + print table ────────────────────────────────────────────
    summary_path = os.path.join(out_dir, "summary.json")
    with open(summary_path, "w") as f:
        json.dump(all_summary, f, indent=2)

    print(f"\n\n{'='*90}")
    print(f"  SEED VARIANCE RESULTS  model={args.model}  task={args.task}")
    print(f"  σ½_pre = {sigma_half_pre:.4e}  (from {sigma_path})")
    print(f"  init_seed = {args.init_seed}  data_seeds = {data_seeds}")
    print(f"{'='*90}")
    print(f"  {'LR':>8}  {'R_mean':>8}  {'R_std':>7}  "
          f"{'acc_mean':>9}  {'dist_mean':>12}  {'dist_max':>12}  "
          f"{'max/σ½':>8}  verdict")
    print(f"  {'─'*88}")
    for s in all_summary:
        verdict = "BOUNDED" if s["pairwise_ratio_max"] < 1.0 else "EXCEEDS"
        print(f"  {s['lr']:>8.0e}  {s['R_mean']:>8.4f}  {s['R_std']:>7.4f}  "
              f"{s['acc_mean']:>9.4f}  {s['pairwise_dist_mean']:>12.4e}  "
              f"{s['pairwise_dist_max']:>12.4e}  "
              f"{s['pairwise_ratio_max']:>8.5f}  {verdict}")
    print(f"{'='*90}\n")
    print(f"Summary saved: {summary_path}")


if __name__ == "__main__":
    main()
