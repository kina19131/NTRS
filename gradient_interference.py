"""
gradient_interference.py
=========================
Test the capacity vs. anisotropy story for SST-2 Phase 1 anomaly.

Question: Does Phase 2 fine-tuning gradient anti-align with SST-2's loss gradient
at the Phase 1 checkpoint? If so, Phase 2 actively damages SST-2 representations
(interference). If not, forgetting is due to something else (e.g., capacity limit).

Method:
  1. Load Llama base + SST-2 Phase 1 LoRA -> theta_A*
  2. Load SST-2 val data + Phase 1 classifier
  3. Compute g_SST2 = grad(L_SST2, lora_params) at theta_A*
     Convert to effective full-weight gradient per layer:
       g_eff_SST2[l] = dL/dB_l @ A_l  (shape: m x r -> m x n via A_l)
       (B=0 not relevant here -- B_l has been trained, not zero)
  4. For each (Phase 2 task, rank):
     a. Apply fresh LoRA(rank) on top of theta_A*, seed per task
     b. Compute g_Phase2 = grad(L_Phase2, new_lora_B) at theta_A* (B=0)
        g_eff_Phase2[l] = dL/dB_l @ A_new_l  (initial direction; B=0 so only B grad)
     c. Per-layer cosine: cos(g_eff_SST2[l], g_eff_Phase2[l])
     d. Aggregate (mean cosine)

Prediction:
  If INTERFERENCE drives forgetting:
    cos(g_SST2, g_CoLA)  < 0  (most anti-aligned -> forgets at lowest T*)
    cos(g_SST2, g_AGNews) < 0 (intermediate)
    cos(g_SST2, g_MNLI_r8) ~ 0 (orthogonal -> doesn't forget until large displacement)
    cos(g_SST2, g_MNLI_r32) < 0 (more anti-aligned -> r32 forgets at normal T*)

  If CAPACITY drives the MNLI anomaly:
    cos(g_SST2, g_MNLI_r8)  ~ 0  (r8 can't reach SST-2's subspace)
    cos(g_SST2, g_MNLI_r32) < 0  (r32 CAN reach it -> more interference)
    This would confirm "capacity story" not "anisotropic basin story"

Run on GPU machine (requires loading full Llama-3.2-3B ~6GB fp16):
  python gradient_interference.py

Output: gradient_interference_results.json
"""

import torch
import torch.nn as nn
import numpy as np
import json
import copy

# ── Paths ─────────────────────────────────────────────────────────────────────
PHASE1_SST2_MODEL = "results/h2_llama/sst2_to_mnli/phase1_model.pt"
PHASE1_SST2_CLSF  = "results/h2_llama/sst2_to_mnli/phase1_classifier.pt"
BASE_MODEL_NAME   = "meta-llama/Llama-3.2-3B"

PHASE2_TASKS = ["cola", "agnews", "mnli"]
PHASE2_RANKS = [8, 32]

TASK_SEED_OFFSET = {"sst2": 0, "mnli": 1000, "agnews": 2000, "cola": 3000}
SEED = 42

N_GRAD_SAMPLES = 64    # SST-2 val samples for gradient estimation
N_PHASE2_SAMPLES = 64  # Phase 2 task samples for gradient estimation
SEQ_LEN = 128
BATCH_SIZE = 8


# ── Imports (deferred to allow script to parse without heavy deps) ─────────────
def setup_imports():
    global AutoTokenizer, AutoModelForCausalLM, get_peft_model, LoraConfig, TaskType
    from transformers import AutoTokenizer, AutoModelForCausalLM
    from peft import get_peft_model, LoraConfig, TaskType


# ── Data loading (reuse from h2_sequential_llama.py pattern) ──────────────────
def load_task_data(tokenizer, task, split, seq_len, n_samples, seed=42):
    from datasets import load_dataset
    import random
    rng = random.Random(seed)

    ds_map = {
        "sst2":   ("glue", "sst2",   lambda x: (x["sentence"], x["label"])),
        "mnli":   ("glue", "mnli",   lambda x: (x["premise"] + " [SEP] " + x["hypothesis"], x["label"])),
        "agnews": ("ag_news", None,   lambda x: (x["text"], x["label"])),
        "cola":   ("glue", "cola",   lambda x: (x["sentence"], x["label"])),
    }
    name, subset, extractor = ds_map[task]
    split_name = "validation_matched" if task == "mnli" and split == "validation" else split
    ds = load_dataset(name, subset, split=split_name) if subset else load_dataset(name, split=split_name)

    items = list(ds)
    rng.shuffle(items)
    items = items[:n_samples]

    all_ids, all_labels = [], []
    for item in items:
        text, label = extractor(item)
        enc = tokenizer(text, max_length=seq_len, truncation=True,
                        padding="max_length", return_tensors="pt")
        all_ids.append(enc["input_ids"].squeeze(0))
        all_labels.append(label)

    input_ids = torch.stack(all_ids)
    labels    = torch.tensor(all_labels, dtype=torch.long)
    n_classes = len(set(all_labels))
    return input_ids, labels, n_classes


def compute_effective_grad(model, classifier, input_ids, labels, device, n_batch=8):
    """
    Compute gradient of task loss w.r.t. LoRA B parameters at current model state.
    Returns dict: layer_base_key -> dL/dB (same shape as B, m x r)
    """
    model.eval()
    classifier.eval()

    total_loss = torch.tensor(0.0, device=device)
    n = 0
    for i in range(0, len(input_ids), n_batch):
        ids  = input_ids[i:i+n_batch].to(device)
        labs = labels[i:i+n_batch].to(device)
        out  = model(ids, output_hidden_states=True)
        hidden = out.hidden_states[-1][:, -1, :].float()
        loss = nn.CrossEntropyLoss()(classifier(hidden), labs)
        total_loss = total_loss + loss * len(ids)
        n += len(ids)

    (total_loss / n).backward()

    grads = {}
    for name, param in model.named_parameters():
        if "lora_B" in name and param.grad is not None:
            base = name.replace(".lora_B.default.weight", "")
            grads[base] = param.grad.detach().cpu().clone()

    model.zero_grad()
    return grads


def effective_weight_grad(grad_B: torch.Tensor, A: torch.Tensor) -> torch.Tensor:
    """
    Convert LoRA B gradient to effective full-weight gradient direction.
    dL/dW_eff = dL/dB @ A  (shape: m x n via (m x r) @ (r x n))
    This is the direction in weight space LoRA would move (for fixed A).
    """
    return grad_B.float() @ A.float()  # (m, n)


def cosine_similarity_frob(M1: torch.Tensor, M2: torch.Tensor) -> float:
    """Frobenius inner product cosine between two matrices."""
    dot  = (M1 * M2).sum().item()
    n1   = M1.norm(p="fro").item()
    n2   = M2.norm(p="fro").item()
    if n1 < 1e-10 or n2 < 1e-10:
        return float("nan")
    return dot / (n1 * n2)


def add_lora(model, rank, task_name, seed):
    from peft import get_peft_model, LoraConfig, TaskType
    torch.manual_seed(seed + TASK_SEED_OFFSET.get(task_name, 9999))
    config = LoraConfig(
        task_type=TaskType.FEATURE_EXTRACTION,
        r=rank, lora_alpha=rank * 2,
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
        lora_dropout=0.0, bias="none",
    )
    return get_peft_model(model, config)


def main():
    setup_imports()

    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"Device: {device}")

    # ── Load base model ────────────────────────────────────────────────────────
    print(f"Loading {BASE_MODEL_NAME}...")
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_NAME)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_NAME, torch_dtype=torch.float16, device_map=device
    )

    # ── Apply SST-2 Phase 1 LoRA ───────────────────────────────────────────────
    print("Applying SST-2 Phase 1 LoRA...")
    torch.manual_seed(SEED + TASK_SEED_OFFSET["sst2"])
    from peft import get_peft_model, LoraConfig, TaskType
    config_p1 = LoraConfig(
        task_type=TaskType.FEATURE_EXTRACTION,
        r=8, lora_alpha=16,
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
        lora_dropout=0.0, bias="none",
    )
    sst2_model = get_peft_model(base_model, config_p1)
    sst2_model.load_state_dict(
        torch.load(PHASE1_SST2_MODEL, map_location=device), strict=False
    )

    hidden_size = sst2_model.config.hidden_size
    sst2_clf = nn.Linear(hidden_size, 2).to(device)
    sst2_clf.load_state_dict(torch.load(PHASE1_SST2_CLSF, map_location=device))

    # Extract SST-2 Phase 1 A matrices (needed to compute effective grad)
    sst2_A = {}
    for name, param in sst2_model.named_parameters():
        if "lora_A" in name:
            base = name.replace(".lora_A.default.weight", "")
            sst2_A[base] = param.detach().cpu().clone()

    # ── SST-2 gradient at theta_A* ────────────────────────────────────────────
    print("Computing SST-2 gradient at Phase 1 checkpoint...")
    print(f"  Loading SST-2 val data ({N_GRAD_SAMPLES} samples)...")
    sst2_ids, sst2_labels, _ = load_task_data(
        tokenizer, "sst2", "validation", SEQ_LEN, N_GRAD_SAMPLES, seed=SEED
    )

    # Enable grad for LoRA parameters only
    for name, p in sst2_model.named_parameters():
        p.requires_grad = "lora_" in name

    grads_sst2_B = compute_effective_grad(
        sst2_model, sst2_clf.to(device), sst2_ids, sst2_labels, device, BATCH_SIZE
    )

    # Effective weight gradients for SST-2
    eff_sst2 = {}
    for base, gB in grads_sst2_B.items():
        if base in sst2_A:
            eff_sst2[base] = effective_weight_grad(gB, sst2_A[base])

    print(f"  SST-2 gradient computed for {len(eff_sst2)} layers")

    # Merge SST-2 LoRA into base model -> theta_A*
    print("Merging SST-2 LoRA into base model (theta_A*)...")
    theta_A_star = sst2_model.merge_and_unload().eval()

    # ── Phase 2 gradients ─────────────────────────────────────────────────────
    results = {}

    for task_b in PHASE2_TASKS:
        print(f"\nLoading Phase 2 task data: {task_b} ({N_PHASE2_SAMPLES} samples)...")
        p2_ids, p2_labels, n_classes_b = load_task_data(
            tokenizer, task_b, "train", SEQ_LEN, N_PHASE2_SAMPLES, seed=SEED
        )

        task_results = {}
        for rank in PHASE2_RANKS:
            print(f"  Computing Phase 2 gradient: {task_b}, rank={rank}...")

            # Fresh LoRA on top of theta_A* with task-specific seed
            model_b = copy.deepcopy(theta_A_star)
            model_b = add_lora(model_b, rank, task_b, SEED)

            # Use default kaiming init — zero weights kill the gradient path
            torch.manual_seed(SEED + TASK_SEED_OFFSET.get(task_b, 9999) + rank)
            clf_b = nn.Linear(hidden_size, n_classes_b).to(device)

            # Enable grad for Phase 2 LoRA B params only
            for name, p in model_b.named_parameters():
                p.requires_grad = "lora_B" in name

            # Extract Phase 2 A matrices
            p2_A = {}
            for name, param in model_b.named_parameters():
                if "lora_A" in name:
                    base = name.replace(".lora_A.default.weight", "")
                    p2_A[base] = param.detach().cpu().clone()

            grads_p2_B = compute_effective_grad(
                model_b, clf_b, p2_ids, p2_labels, device, BATCH_SIZE
            )

            # Effective weight gradients for Phase 2
            eff_p2 = {}
            for base, gB in grads_p2_B.items():
                if base in p2_A:
                    eff_p2[base] = effective_weight_grad(gB, p2_A[base])

            # Per-layer cosine similarity between SST-2 and Phase 2 effective grads
            common_layers = sorted(set(eff_sst2.keys()) & set(eff_p2.keys()))
            cos_per_layer = {}
            for layer in common_layers:
                cos = cosine_similarity_frob(eff_sst2[layer], eff_p2[layer])
                cos_per_layer[layer] = cos

            valid_cos = [v for v in cos_per_layer.values() if not np.isnan(v)]
            mean_cos  = float(np.mean(valid_cos)) if valid_cos else float("nan")
            med_cos   = float(np.median(valid_cos)) if valid_cos else float("nan")
            neg_frac  = sum(1 for v in valid_cos if v < 0) / len(valid_cos) if valid_cos else float("nan")

            print(f"    mean_cos={mean_cos:.4f}  median={med_cos:.4f}  "
                  f"frac_negative={neg_frac:.3f}  n_layers={len(valid_cos)}")

            task_results[f"rank{rank}"] = {
                "mean_cos":     mean_cos,
                "median_cos":   med_cos,
                "frac_negative": neg_frac,
                "n_layers":     len(valid_cos),
            }

            del model_b, clf_b

        results[task_b] = task_results

    # ── Summary table ──────────────────────────────────────────────────────────
    print("\n" + "=" * 65)
    print("GRADIENT INTERFERENCE: cos(grad_SST2, grad_Phase2) at theta_A*")
    print("Negative = Phase 2 gradient anti-aligns with SST2 gradient (interference)")
    print("=" * 65)
    print(f"  {'Task':8s}  {'rank':6s}  {'mean_cos':10s}  {'frac<0':8s}  T* (SST-2 Phase1)")
    T_STAR = {"cola": 0.006, "agnews": 0.016, "mnli_r8": 0.122, "mnli_r32": "cluster"}
    for task in PHASE2_TASKS:
        for rank_key in [f"rank{r}" for r in PHASE2_RANKS]:
            r = results[task][rank_key]
            rank_num = rank_key.replace("rank", "")
            tstar_key = f"{task}_r{rank_num}" if task == "mnli" else task
            tstar = T_STAR.get(tstar_key, T_STAR.get(task, "?"))
            print(f"  {task:8s}  r{rank_num:4s}  {r['mean_cos']:+.4f}      {r['frac_negative']:.3f}     T*~{tstar}")

    print()
    print("Interpretation:")
    print("  Capacity story supported if: MNLI r8 mean_cos ~ 0, MNLI r32 << 0")
    print("  Anisotropy story supported if: CoLA << 0, MNLI r8 ~ 0, MNLI r32 ~ 0")

    # ── Save ──────────────────────────────────────────────────────────────────
    out = {
        "description": "Gradient interference at SST-2 Phase 1 checkpoint",
        "n_grad_samples": N_GRAD_SAMPLES,
        "n_phase2_samples": N_PHASE2_SAMPLES,
        "results": results,
    }
    with open("gradient_interference_results.json", "w") as f:
        json.dump(out, f, indent=2)
    print("\nSaved: gradient_interference_results.json")


if __name__ == "__main__":
    main()
