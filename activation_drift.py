"""
activation_drift.py
====================
Test whether activation drift on SST-2 inputs explains the MNLI r8 anomaly.

Key comparison: MNLI r8 vs MNLI r32, starting from the same SST-2 theta_A*.
  MNLI r8:  T* at R_A ~ 0.4   (resistant)
  MNLI r32: T* at R_A ~ 0.05  (normal)

Both T* checkpoints are LoRA weights trained ON TOP OF theta_A*. The saved
LoRA must be applied to theta_A* — not base_model — to reconstruct the correct
model state at the forgetting boundary.

Metric:
  raw_drift[l]  = mean_x ||h(x; theta_T*)[l] - h(x; theta_A*)[l]|| / ||h(x; theta_A*)[l]||
  norm_drift[l] = raw_drift[l] / R_A_at_T*    (drift per unit displacement)

Capacity story:   norm_drift(r32) > norm_drift(r8)  — r32 more efficiently damages SST-2
Anisotropy story: norm_drift(r32) ≈ norm_drift(r8)  — same geometry, same damage rate

Requires: T* checkpoints from h2_sequential_llama.py --save_tstar_checkpoints
  h2_llama/sst2_to_mnli/{tag}_tstar_lora.pt
  h2_llama/sst2_to_mnli/{tag}_tstar_meta.json

Run:
  python3 activation_drift.py 2>&1 | tee logs/activation_drift.log

Output: activation_drift_results.json
"""

import glob
import json
import os
import copy

import torch
import numpy as np

BASE_MODEL_NAME  = "meta-llama/Llama-3.2-3B"
PHASE1_SST2_LORA = "results/h2_llama/sst2_to_mnli/phase1_model.pt"
TSTAR_DIR        = "results/h2_llama/sst2_to_mnli"

LORA_RANK_P1     = 8
TASK_SEED_OFFSET = {"sst2": 0, "mnli": 1000, "agnews": 2000, "cola": 3000}
SEED     = 42
N_SAMPLES  = 128
SEQ_LEN    = 128
BATCH_SIZE = 8


def discover_tstar_checkpoints(directory):
    lora_files = sorted(glob.glob(os.path.join(directory, "*_tstar_lora.pt")))
    results = []
    for lp in lora_files:
        meta_p = lp.replace("_tstar_lora.pt", "_tstar_meta.json")
        if not os.path.exists(meta_p):
            print(f"  WARNING: no meta for {lp}, skipping")
            continue
        with open(meta_p) as f:
            meta = json.load(f)
        label = os.path.basename(lp).replace("_tstar_lora.pt", "")
        results.append({
            "label":       label,
            "lora_path":   lp,
            "rank":        meta.get("rank"),
            "lr":          meta.get("lr"),
            "task_b":      meta.get("task_b", "mnli"),
            "tstar_R_A":   meta.get("tstar_R_A"),
            "tstar_acc_a": meta.get("tstar_acc_a"),
        })
    return results


def load_sst2_inputs(tokenizer, n_samples, seq_len, seed=42):
    from datasets import load_dataset
    import random
    ds = load_dataset("glue", "sst2", split="validation")
    items = list(ds)
    random.Random(seed).shuffle(items)
    all_ids = []
    for item in items[:n_samples]:
        enc = tokenizer(item["sentence"], max_length=seq_len, truncation=True,
                        padding="max_length", return_tensors="pt")
        all_ids.append(enc["input_ids"].squeeze(0))
    return torch.stack(all_ids)


def apply_lora_and_merge(theta_base, lora_sd, rank, task_name):
    """
    Apply a saved LoRA state dict to a copy of theta_base and return merged model.
    theta_base can be any model (base Llama or theta_A*).
    The saved lora_sd includes both A and B matrices, so they are loaded exactly
    as trained — no seed dependence.
    """
    from peft import get_peft_model, LoraConfig, TaskType
    model = copy.deepcopy(theta_base)
    torch.manual_seed(SEED + TASK_SEED_OFFSET.get(task_name, 9999))
    config = LoraConfig(
        task_type=TaskType.FEATURE_EXTRACTION,
        r=rank, lora_alpha=rank * 2,
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
        lora_dropout=0.0, bias="none",
    )
    peft_model = get_peft_model(model, config)
    # Loads both A and B — checkpoint A overwrites random init
    peft_model.load_state_dict(lora_sd, strict=False)
    return peft_model.merge_and_unload().eval()


@torch.no_grad()
def get_activations(model, input_ids, batch_size, device):
    all_hidden = None
    for i in range(0, len(input_ids), batch_size):
        batch = input_ids[i:i+batch_size].to(device)
        out = model(batch, output_hidden_states=True)
        batch_hidden = [h[:, -1, :].cpu().float() for h in out.hidden_states]
        if all_hidden is None:
            all_hidden = batch_hidden
        else:
            all_hidden = [torch.cat([a, b], dim=0) for a, b in zip(all_hidden, batch_hidden)]
    return all_hidden  # list of (N, hidden), one per layer


def layer_drift(ref_hidden, task_hidden):
    drifts = []
    for h_ref, h_task in zip(ref_hidden, task_hidden):
        diff = torch.norm(h_task - h_ref, dim=-1)
        ref  = torch.norm(h_ref, dim=-1).clamp(min=1e-8)
        drifts.append((diff / ref).mean().item())
    return np.array(drifts)


def main():
    from transformers import AutoTokenizer, AutoModelForCausalLM

    device = ("cuda" if torch.cuda.is_available()
              else "mps" if hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
              else "cpu")
    print(f"Device: {device}")

    checkpoints = discover_tstar_checkpoints(TSTAR_DIR)
    if not checkpoints:
        print(f"ERROR: No T* checkpoints found in {TSTAR_DIR}")
        print("Run h2_sequential_llama.py --task_a sst2 --task_b mnli --skip_phase1 first.")
        return

    print(f"\nFound {len(checkpoints)} T* checkpoints:")
    for c in checkpoints:
        print(f"  {c['label']:30s}  rank={c['rank']}  lr={c['lr']:.0e}"
              f"  T*_R_A={c['tstar_R_A']:.4f}")

    print(f"\nLoading {BASE_MODEL_NAME} ...")
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_NAME)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_NAME, torch_dtype=torch.float16, device_map=device
    )
    base_model.eval()
    n_layers = base_model.config.num_hidden_layers
    print(f"  Layers: {n_layers}")

    # ── Build theta_A* (SST-2 Phase 1) ────────────────────────────────────────
    print("\nBuilding theta_A* (SST-2 Phase 1) ...")
    p1_lora_sd = torch.load(PHASE1_SST2_LORA, map_location="cpu")
    theta_A_star = apply_lora_and_merge(base_model, p1_lora_sd, LORA_RANK_P1, "sst2")
    # Keep theta_A_star alive — T* LoRAs are applied on top of it

    print(f"Loading {N_SAMPLES} SST-2 val inputs ...")
    input_ids = load_sst2_inputs(tokenizer, N_SAMPLES, SEQ_LEN, SEED)

    print("Computing reference activations (theta_A*) ...")
    ref_hidden = get_activations(theta_A_star, input_ids, BATCH_SIZE, device)
    print(f"  Done: {len(ref_hidden)} layers x {ref_hidden[0].shape[0]} samples")

    # ── Per-checkpoint activation drift ───────────────────────────────────────
    # T* LoRA is applied ON TOP OF theta_A* — not base_model
    results = []
    for ckpt in checkpoints:
        task_b = ckpt.get("task_b", "mnli")
        print(f"\nProcessing {ckpt['label']} (rank={ckpt['rank']}, T*_R_A={ckpt['tstar_R_A']:.4f}) ...")
        lora_sd    = torch.load(ckpt["lora_path"], map_location="cpu")
        tstar_model = apply_lora_and_merge(theta_A_star, lora_sd, ckpt["rank"], task_b)
        task_hidden = get_activations(tstar_model, input_ids, BATCH_SIZE, device)
        del tstar_model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        drift = layer_drift(ref_hidden, task_hidden)
        tstar_R_A = ckpt["tstar_R_A"] or 1e-8

        entry = {
            **ckpt,
            "mean_drift":      float(drift.mean()),
            "median_drift":    float(np.median(drift)),
            "norm_drift":      float(drift.mean()) / max(tstar_R_A, 1e-8),
            "per_layer_drift": drift.tolist(),
            "top5_layers":     np.argsort(drift)[::-1][:5].tolist(),
        }
        results.append(entry)
        print(f"  raw_drift={entry['mean_drift']:.4f}  "
              f"norm_drift={entry['norm_drift']:.4f}  "
              f"T*_R_A={tstar_R_A:.4f}")

    # ── Summary ───────────────────────────────────────────────────────────────
    print("\n" + "=" * 72)
    print("ACTIVATION DRIFT at T* (relative to theta_A*)")
    print("  raw_drift  = ||h(theta_T*) - h(theta_A*)|| / ||h(theta_A*)||")
    print("  norm_drift = raw_drift / R_A_at_T*  (drift per unit displacement)")
    print()
    print("Capacity story:   norm_drift(r32) > norm_drift(r8)")
    print("Anisotropy story: norm_drift(r32) ≈ norm_drift(r8)")
    print("=" * 72)
    print(f"  {'Label':28s}  {'rank':5}  {'T*_R_A':8}  {'raw_drift':10}  {'norm_drift':12}")
    for r in sorted(results, key=lambda x: (x["rank"], x["lr"] or 0)):
        print(f"  {r['label']:28s}  r{r['rank']:<4}  {r['tstar_R_A']:.4f}    "
              f"{r['mean_drift']:.4f}      {r['norm_drift']:.4f}")

    # Key test: r8 vs r32
    r8_entries  = [r for r in results if r["rank"] == 8]
    r32_entries = [r for r in results if r["rank"] == 32]

    if r8_entries and r32_entries:
        # Pick condition where both ranks have similar enough LR to be comparable
        # (or just pick one per rank with lowest LR that reaches T*)
        r8  = min(r8_entries,  key=lambda x: x["tstar_R_A"] or 999)
        r32 = min(r32_entries, key=lambda x: x["tstar_R_A"] or 999)

        print(f"\nKey comparison (MNLI):")
        print(f"  r8:  raw={r8['mean_drift']:.4f}  norm={r8['norm_drift']:.4f}  "
              f"T*_R_A={r8['tstar_R_A']:.4f}")
        print(f"  r32: raw={r32['mean_drift']:.4f}  norm={r32['norm_drift']:.4f}  "
              f"T*_R_A={r32['tstar_R_A']:.4f}")

        raw_ratio  = r32["mean_drift"] / max(r8["mean_drift"], 1e-8)
        norm_ratio = r32["norm_drift"] / max(r8["norm_drift"], 1e-8)
        print(f"\n  raw  drift ratio r32/r8 = {raw_ratio:.3f}  "
              f"({'expected < 1 since r32 displaces less' if raw_ratio < 1 else ''})")
        print(f"  norm drift ratio r32/r8 = {norm_ratio:.3f}")
        if norm_ratio > 1.2:
            verdict = "CAPACITY story: r32 causes more SST-2 damage per unit displacement"
        elif norm_ratio < 0.8:
            verdict = "ANISOTROPY story: r32 causes less damage per unit displacement"
        else:
            verdict = "INCONCLUSIVE: norm_drift ratio within 20% — neither story clearly supported"
        print(f"  Interpretation: {verdict}")
        print()
        print("  NOTE: norm_drift assumes drift scales linearly with displacement.")
        print(f"  r32 T*_R_A={r32['tstar_R_A']:.3f} vs r8 T*_R_A={r8['tstar_R_A']:.3f} "
              f"({r8['tstar_R_A']/max(r32['tstar_R_A'],1e-8):.1f}× difference).")
        print("  If inconclusive, interpret as: cannot distinguish stories at this resolution,")
        print("  not as evidence they are equivalent.")

        # Layer-level: where does r32 show higher norm_drift than r8?
        nd8  = np.array(r8["per_layer_drift"])  / max(r8["tstar_R_A"], 1e-8)
        nd32 = np.array(r32["per_layer_drift"]) / max(r32["tstar_R_A"], 1e-8)
        gaps = nd32 - nd8
        top_gap = np.argsort(gaps)[::-1][:10]
        print(f"\nTop 10 layers where r32 has higher norm_drift than r8:")
        for l in top_gap:
            print(f"  layer {l:3d}: r8={nd8[l]:.4f}  r32={nd32[l]:.4f}  gap={gaps[l]:+.4f}")

    out = {
        "description": (
            "Activation drift on SST-2 inputs at T* checkpoint. "
            "T* LoRA applied on top of theta_A* (SST-2 Phase 1). "
            "norm_drift = raw_drift / R_A_at_T*."
        ),
        "n_samples": N_SAMPLES,
        "n_layers":  n_layers,
        "results":   results,
    }
    with open("activation_drift_results.json", "w") as f:
        json.dump(out, f, indent=2)
    print("\nSaved: activation_drift_results.json")


if __name__ == "__main__":
    main()
