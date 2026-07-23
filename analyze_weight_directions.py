"""
analyze_weight_directions.py
============================
Tests the "representational overlap" hypothesis:
Does Phase 2 fine-tuning move weights in a direction that overlaps with
the Phase 1 fine-tuning direction? If so, does high overlap predict forgetting?

Method (Tier 0a from experiment plan):
  Δθ_A = Phase 1 LoRA weights (saved in phase1_model.pt)
         These are the LoRA adapter tensors that represent how Phase 1
         fine-tuning moved the model from base weights.

  g_B   = Gradient of Phase 2 loss w.r.t. LoRA parameters at initialization,
         evaluated at the Phase 1 merged model (θ_phase1).
         This is the direction Phase 2 training wants to move the weights.
         (Proxy for Δθ_B direction; exact Δθ_B would need saved Phase 2 adapters.)

  cos_sim(Δθ_A, -g_B) = alignment between Phase 1 direction and Phase 2 movement.
         High positive value → Phase 2 undoes Phase 1 → predicts forgetting.
         Near zero → orthogonal → predicts Phase 1 retention.

Usage:
    CUDA_VISIBLE_DEVICES=0 python analyze_weight_directions.py \
        --h2_dir ./results/h2_llama \
        --base_model meta-llama/Llama-3.2-3B

Output: direction_overlap_results.json  +  console table
"""

import argparse, json, os, torch

from lora_density_experiment import _load_model_auto, _get_tokenizer, add_lora, _default_target_modules


PAIRS = [
    ("mnli_to_sst2",   "mnli",   "sst2",   "MNLI",   "SST-2",  "stable"),
    ("mnli_to_agnews", "mnli",   "agnews", "MNLI",   "AGNews", "stable"),
    ("agnews_to_mnli", "agnews", "mnli",   "AGNews", "MNLI",   "stable"),
    ("agnews_to_sst2", "agnews", "sst2",   "AGNews", "SST-2",  "stable"),
    ("sst2_to_mnli",   "sst2",   "mnli",   "SST-2",  "MNLI",   "robust_r8"),
    ("sst2_to_agnews", "sst2",   "agnews", "SST-2",  "AGNews", "collapses"),
]


def load_task_batch(tokenizer, task, n_samples=64, max_len=128, seed=42, device="cpu"):
    from datasets import load_dataset
    torch.manual_seed(seed)
    if task == "mnli":
        ds = load_dataset("glue", "mnli", split="validation_matched")
        def fmt(ex): return f"premise: {ex['premise']} hypothesis: {ex['hypothesis']}"
    elif task == "sst2":
        ds = load_dataset("glue", "sst2", split="validation")
        def fmt(ex): return ex["sentence"]
    elif task == "agnews":
        ds = load_dataset("ag_news", split="test")
        def fmt(ex): return ex["text"]
    elif task == "cola":
        ds = load_dataset("glue", "cola", split="validation")
        def fmt(ex): return ex["sentence"]
    else:
        raise ValueError(task)

    ds = ds.shuffle(seed=seed).select(range(min(n_samples, len(ds))))
    ids, labels = [], []
    for ex in ds:
        enc = tokenizer(fmt(ex), truncation=True, max_length=max_len,
                        padding="max_length", return_tensors="pt")
        ids.append(enc["input_ids"].squeeze(0))
        labels.append(ex["label"])
    return torch.stack(ids).to(device), torch.tensor(labels, dtype=torch.long).to(device)


def lora_flat_vector(lora_state_dict):
    """Concatenate all LoRA A and B weights into a single flat vector."""
    parts = []
    for k in sorted(lora_state_dict.keys()):
        parts.append(lora_state_dict[k].float().cpu().flatten())
    return torch.cat(parts)


def compute_grad_vector(model, lora_module, task, tokenizer, device,
                        n_samples=64, max_len=128, micro_batch=4):
    """
    Compute gradient of task loss w.r.t. LoRA parameters at current (init) state.
    Uses micro-batching to avoid OOM from activation storage during backward.
    Returns flat gradient vector in same space as lora_flat_vector().
    """
    from torch import nn
    n_classes = {"mnli": 3, "sst2": 2, "agnews": 4, "cola": 2}[task]
    hidden = model.config.hidden_size

    torch.manual_seed(0)
    clf = nn.Linear(hidden, n_classes).to(device)

    ids, labs = load_task_batch(tokenizer, task, n_samples, max_len, device=device)

    lora_module.train()
    clf.train()
    lora_module.zero_grad()
    clf.zero_grad()

    # Accumulate gradients over micro-batches — avoids storing 28 layers of
    # [n_samples, seq_len, 8192] activations simultaneously
    n_total = ids.shape[0]
    for i in range(0, n_total, micro_batch):
        batch_ids = ids[i : i + micro_batch]
        batch_labs = labs[i : i + micro_batch]
        out = lora_module(batch_ids, output_hidden_states=True)
        hidden_states = out.hidden_states[-1][:, -1, :].float()
        logits = clf(hidden_states)
        loss = nn.functional.cross_entropy(logits, batch_labs)
        # Scale so total gradient = mean over all samples
        (loss * batch_ids.shape[0] / n_total).backward()

    grads = {}
    for name, param in lora_module.named_parameters():
        if "lora_" in name and param.grad is not None:
            grads[name] = param.grad.float().cpu().clone()

    lora_module.eval()
    lora_module.zero_grad()
    return lora_flat_vector(grads) if grads else None


def cosine_sim(v1, v2):
    v1 = v1.float()
    v2 = v2.float()
    return (torch.dot(v1, v2) / (v1.norm() * v2.norm() + 1e-12)).item()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--h2_dir",      default="./results/h2_llama")
    p.add_argument("--base_model",  default="meta-llama/Llama-3.2-3B")
    p.add_argument("--n_samples",   type=int, default=128)
    p.add_argument("--max_len",     type=int, default=128)
    p.add_argument("--out",         default="direction_overlap_results.json")
    args = p.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    print(f"Loading base model on CPU: {args.base_model}")

    # Load base on CPU — we'll only move the merged model to GPU per-pair
    base_model, _ = _load_model_auto(args.base_model, "cpu")
    tokenizer = _get_tokenizer(args.base_model)
    default_tgt = _default_target_modules(args.base_model)  # short names for PEFT

    results = []

    for tag, task_a, task_b, p1_name, p2_name, expected in PAIRS:
        run_dir = os.path.join(args.h2_dir, tag)
        model_path = os.path.join(run_dir, "phase1_model.pt")
        meta_path  = os.path.join(run_dir, "phase1_meta.json")

        if not os.path.exists(model_path):
            print(f"\n[SKIP] {tag}: no phase1_model.pt")
            continue

        print(f"\n── {p1_name}→{p2_name} ──────────────────────────────────")

        with open(meta_path) as f:
            meta = json.load(f)
        rank = meta["rank"]
        alpha = meta.get("alpha", 16)
        # lora_target_modules in meta are short module names (q_proj etc.) from
        # newer saves; old local saves don't have it so fall back to defaults
        lora_tgt = meta.get("lora_target_modules", default_tgt)

        # ── Load Phase 1 LoRA weights (Δθ_A) — stays on CPU ─────────────────
        raw_sd = torch.load(model_path, map_location="cpu", weights_only=False)
        # New saves (lora_only format) have 'lora_' in all keys.
        # Old saves stored the full merged model — no LoRA keys, incompatible.
        lora_sd = {k: v for k, v in raw_sd.items() if "lora_" in k}
        if not lora_sd:
            print(f"  [SKIP] {tag}: old full-model save format, no LoRA keys — "
                  f"re-run Phase 1 on server to get lora_only checkpoint")
            continue
        v_A = lora_flat_vector(lora_sd)
        print(f"  Phase 1 LoRA vector dim: {v_A.shape[0]}  norm: {v_A.norm():.4f}")

        # ── Build merged Phase 1 model on CPU, then move to GPU ──────────────
        import copy
        model_cpu = copy.deepcopy(base_model)          # CPU copy
        peft_cpu  = add_lora(model_cpu, rank, alpha, target_modules=lora_tgt)
        peft_cpu.load_state_dict(lora_sd, strict=False)
        merged_cpu = peft_cpu.merge_and_unload()       # merge on CPU
        del model_cpu, peft_cpu                        # free CPU copy

        merged = merged_cpu.to(device).eval()          # only this goes to GPU
        del merged_cpu
        torch.cuda.empty_cache()

        # ── Add fresh Phase 2 LoRA and compute gradient direction ─────────────
        peft_p2 = add_lora(merged, rank, alpha, target_modules=lora_tgt)
        g_B = compute_grad_vector(merged, peft_p2, task_b, tokenizer, device,
                                  args.n_samples, args.max_len)

        if g_B is None:
            print(f"  WARNING: no LoRA gradients found, skipping")
            del merged, peft_p2
            torch.cuda.empty_cache()
            continue

        print(f"  Phase 2 grad vector dim: {g_B.shape[0]}  norm: {g_B.norm():.4f}")

        # ── Cosine similarities ───────────────────────────────────────────────
        cos_same = cosine_sim(v_A, g_B)
        cos_undo = cosine_sim(v_A, -g_B)

        print(f"  cos(Δθ_A,  g_B) = {cos_same:+.4f}  "
              f"{'↑ Phase 2 reinforces Phase 1' if cos_same > 0.05 else '≈0 orthogonal' if abs(cos_same) < 0.05 else '↓ Phase 2 opposes Phase 1'}")
        print(f"  cos(Δθ_A, -g_B) = {cos_undo:+.4f}  "
              f"{'⚠ Phase 2 UNDOES Phase 1' if cos_undo > 0.05 else '≈0 orthogonal (safe)' if abs(cos_undo) < 0.05 else 'Phase 2 amplifies Phase 1'}")
        print(f"  T* behavior: {expected}")

        results.append({
            "pair": f"{p1_name}→{p2_name}",
            "tag": tag,
            "task_a": task_a, "task_b": task_b,
            "p1_type": "binary" if task_a == "sst2" else "multi-class",
            "t_star_behavior": expected,
            "cos_A_gB":  cos_same,
            "cos_A_neg_gB": cos_undo,
            "v_A_norm": v_A.norm().item(),
            "g_B_norm": g_B.norm().item(),
        })

        del merged, peft_p2
        torch.cuda.empty_cache()

    # ── Print summary table ───────────────────────────────────────────────────
    print("\n" + "=" * 72)
    print(f"{'Pair':<22} {'T* behavior':<14} {'cos(Δθ_A, g_B)':>16} {'Prediction':<20}")
    print("=" * 72)
    for r in results:
        cos = r["cos_A_gB"]
        pred = "overlap→forgetting" if cos < -0.02 else ("orthogonal→safe" if abs(cos) < 0.02 else "reinforce→safe")
        print(f"{r['pair']:<22} {r['t_star_behavior']:<14} {cos:>+16.4f}  {pred:<20}")
    print("=" * 72)

    print("\nOverlap hypothesis prediction:")
    print("  SST-2→AGNews (collapses) should have most negative cos(Δθ_A, g_B)")
    print("  SST-2→MNLI (robust_r8)  should have cos(Δθ_A, g_B) near 0")
    print("  Multi-class pairs (stable) should all be near 0")

    with open(args.out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved: {args.out}")


if __name__ == "__main__":
    main()
