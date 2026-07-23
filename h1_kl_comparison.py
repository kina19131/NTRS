"""
h1_kl_comparison.py
====================
H1 critical experiment: R = ‖Δθ‖/σ½ vs KL(base‖ft) as discriminators of
safe vs. degraded fine-tuning at the pretraining→SFT boundary.

For each LR in [1e-5, 5e-5, 1e-4, 2e-4, 5e-4], fine-tune GPT-2 on SST-2 then
measure:
  R          = per_param_norm / sigma_half_pretrained
  acc        = SST-2 validation accuracy
  kl_wiki    = KL(base‖ft) on WikiText-2 validation (Option A)
  kl_diverse = KL(base‖ft) on 50 fixed diverse prompts (Option B)

Expected result: R separates safe (high acc, R < 1) from degraded (low acc,
R > 1). Both KL measures are flat across conditions — they do not discriminate.

Output:
  results/h1_kl/{model}/
    lr{X}_results.json    — per-LR metrics
    summary.json          — all conditions
    kl_vs_r_comparison.png — two-panel key figure

Usage:
  python h1_kl_comparison.py
  python h1_kl_comparison.py --lrs 1e-5 5e-5 1e-4 2e-4 5e-4 --steps 500
  python h1_kl_comparison.py --replot  # regenerate figures from saved JSONs
"""

import argparse
import copy
import itertools
import json
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(__file__))
from certified_density_experiment import (
    save_weights, restore_weights, perturb_weights,
    load_data, compute_nll_loss, estimate_density, half_max_sigma_adaptive,
)
from lora_density_experiment import (
    load_sst2, add_lora, _load_model_auto, _get_target_names_auto,
    _get_hidden_size, _get_tokenizer,
)

# ── Fixed diverse prompts for Option B ───────────────────────────────────────
# 50 prompts spanning factual, reasoning, sentiment, summarization, coding,
# instruction-following — matches the diversity of RL's Razor's evaluation set.
DIVERSE_PROMPTS = [
    # Factual
    "The capital of France is",
    "Water is composed of hydrogen and",
    "The speed of light in vacuum is approximately",
    "Albert Einstein was born in",
    "The Great Wall of China was built during the",
    "Photosynthesis converts sunlight into",
    "The human genome contains approximately",
    "The largest planet in our solar system is",
    "DNA stands for",
    "The boiling point of water at sea level is",
    # Reasoning
    "If all mammals are warm-blooded, and dogs are mammals, then dogs are",
    "A train travels 60 miles per hour. In two hours it will have traveled",
    "If it rains, the ground gets wet. The ground is wet, so",
    "All prime numbers greater than 2 are",
    "The next number in the sequence 2, 4, 8, 16 is",
    # Sentiment / opinion
    "The movie was absolutely terrible because",
    "I loved the restaurant because the food was",
    "The product review was overwhelmingly positive, noting that",
    "Customers complained that the service was",
    "The concert exceeded all expectations when",
    # Summarization / completion
    "In summary, the main argument of the paper is",
    "To conclude, the experiment demonstrated that",
    "The key finding from the study was",
    "Based on the evidence presented, we can conclude that",
    "The most important takeaway from this analysis is",
    # Instruction-following
    "Translate the following sentence to French: Hello, how are you?",
    "Summarize the following text in one sentence:",
    "List three advantages of renewable energy:",
    "Explain the concept of machine learning in simple terms:",
    "What are the steps to bake a chocolate cake?",
    # Science / technical
    "The second law of thermodynamics states that",
    "In quantum mechanics, the uncertainty principle was formulated by",
    "Machine learning models are trained by minimizing",
    "The backpropagation algorithm computes gradients by",
    "Neural networks with many layers are called",
    # History / culture
    "The French Revolution began in the year",
    "Shakespeare wrote the play Hamlet around",
    "The Industrial Revolution started in",
    "The United Nations was founded after",
    "The first programmable computer was developed by",
    # General knowledge
    "The process of cell division is called",
    "Antibiotics work by",
    "Inflation in economics refers to",
    "The stock market crashed in October",
    "Climate change is primarily caused by",
    # Language / writing
    "A metaphor is a figure of speech that",
    "The passive voice is formed by",
    "A haiku is a form of poetry with",
    "The Oxford comma refers to",
    "Grammar rules exist to ensure that",
]
assert len(DIVERSE_PROMPTS) == 50


def _lora_targets(model_name):
    """Return LoRA target module names for a given model family."""
    n = model_name.lower()
    if any(x in n for x in ("llama", "mistral", "gemma", "qwen", "falcon")):
        return ["q_proj", "v_proj", "k_proj", "o_proj"]
    return ["c_attn", "c_proj"]  # GPT-2 default


def _get_lora_scoped_names(model, model_name):
    """
    Parameter names for σ½ calibration, restricted to LoRA's actual target
    modules (q/k/v/o_proj for Llama/Mistral, c_attn/c_proj for GPT-2) rather
    than _get_target_names_auto's whole-decoder-block scope (attention + MLP +
    layernorms).

    Moved here 2026-07-21 from hessian_eigenvalue.py (where it was added
    2026-07-20 for the Hessian probe) so σ½ itself can use the same scoping —
    the open validity question in SUMMARY.md ("Open validity questions: σ½/LoRA
    scope mismatch"): σ½ was certifying robustness over a much larger parameter
    set than LoRA fine-tuning ever actually displaces. hessian_eigenvalue.py
    now imports this from here instead of defining its own copy, so there is
    exactly one definition of "LoRA-scoped" shared by both checks.

    Note LoRA target module names differ by architecture, so this scope's
    fraction of total params is NOT the same across GPT-2/Llama/Mistral —
    GPT-2's target name "c_proj" collides with both attn.c_proj and mlp.c_proj
    under substring matching, so GPT-2's LoRA already touches part of the MLP;
    Llama/Mistral's q/k/v/o_proj names have no such collision (attention only).

    Also excludes bias parameters: every add_lora() call in this codebase uses
    LoraConfig(bias="none"), so biases are never updated by fine-tuning in any
    of the three architectures. Only GPT-2's Conv1D layers actually have bias
    parameters here (Llama/Mistral's q/k/v/o_proj have bias=False in the base
    model), so leaving bias in would be a second, GPT-2-only scope leak on top
    of the c_proj/MLP collision above — both would additionally widen GPT-2's
    "LoRA-scoped" set relative to what its LoRA fine-tuning actually touches.
    """
    lora_names = set(_lora_targets(model_name))
    block_names = _get_target_names_auto(model, model_name)
    # Match against any dotted path COMPONENT, not the last one (which is
    # "weight"/"bias") — e.g. "model.layers.5.self_attn.q_proj.weight" must
    # match on "q_proj", the second-to-last component.
    return {n for n in block_names
            if (lora_names & set(n.split("."))) and not n.endswith(".bias")}


def get_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model",    default="gpt2")
    p.add_argument("--task",    default="sst2", choices=["sst2", "mnli"],
                   help="Downstream task for Phase 1 fine-tuning")
    p.add_argument("--lrs",     nargs="+", type=float,
                   default=[1e-5, 5e-5, 1e-4, 2e-4, 5e-4])
    p.add_argument("--rank",    type=int, default=8)
    p.add_argument("--alpha",   type=int, default=16)
    p.add_argument("--steps",   type=int, default=500)
    p.add_argument("--batch_size",      type=int,   default=16)
    p.add_argument("--grad_clip",       type=float, default=1.0)
    p.add_argument("--n_perturb",       type=int,   default=200,
                   help="Perturbations for σ½ estimation")
    p.add_argument("--n_text_samples",  type=int,   default=128)
    p.add_argument("--seq_len",         type=int,   default=128)
    p.add_argument("--target_density",  type=float, default=0.40)
    p.add_argument("--eval_slack",      type=float, default=None,
                   help="NLL tolerance for σ½ density estimation. "
                        "Default: 1e-4 for GPT-2, 0.01 for Llama (set automatically).")
    p.add_argument("--sigma_scope", default="lora", choices=["lora", "block"],
                   help="Parameter scope perturbed to calibrate σ½. 'lora' "
                        "(default): only LoRA's actual target modules "
                        "(q/k/v/o_proj or c_attn/c_proj) — matches what "
                        "fine-tuning displaces. 'block': legacy whole-decoder-"
                        "block scope (attention+MLP+layernorms) used by every "
                        "σ½ value in SUMMARY.md before 2026-07-21; kept only "
                        "to reproduce old numbers. The two scopes write to "
                        "different cache filenames so they never collide.")
    p.add_argument("--kl_seq_len",      type=int,   default=64,
                   help="Sequence length for KL computation (shorter = faster)")
    p.add_argument("--kl_n_wiki",       type=int,   default=200,
                   help="WikiText-2 sequences for Option A KL")
    p.add_argument("--seed",        type=int, default=42)
    p.add_argument("--output_dir",  default="./results/h1_kl")
    p.add_argument("--replot",  action="store_true",
                   help="Skip training; reload JSONs and regenerate figures")
    p.add_argument("--force",   action="store_true",
                   help="Ignore cached per-LR JSONs and rerun all conditions")
    p.add_argument("--n_superni", type=int, default=200,
                   help="Number of SuperNI prompts for kl_superni metric")
    p.add_argument("--n_mmlu",    type=int, default=200,
                   help="Number of MMLU questions for accuracy eval")
    return p.parse_args()


# ── KL computation ────────────────────────────────────────────────────────────

@torch.no_grad()
def compute_kl(base_model, ft_model, input_ids, batch_size=4):
    """
    KL(P_base ‖ P_ft) averaged over tokens and sequences.

    Measures how surprised the fine-tuned model is at what the base model
    predicts — large value means distributional shift, small means none.
    Token-level average: avoids sequence-length confounds.
    """
    base_model.eval()
    ft_model.eval()
    total_kl, n_batches = 0.0, 0

    for i in range(0, len(input_ids), batch_size):
        batch = input_ids[i : i + batch_size]
        logits_base = base_model(batch).logits.float()  # (B, T, V)
        logits_ft   = ft_model(batch).logits.float()

        log_p = F.log_softmax(logits_base, dim=-1)
        log_q = F.log_softmax(logits_ft,   dim=-1)
        p     = log_p.exp()

        # KL per token (B, T), then mean over B and T
        kl_tokens = (p * (log_p - log_q)).sum(dim=-1)
        total_kl += kl_tokens.mean().item()
        n_batches += 1

    return total_kl / max(n_batches, 1)


def encode_diverse_prompts(tokenizer, prompts, seq_len, device):
    """Tokenize the 50 diverse prompts, pad/truncate to seq_len."""
    ids = []
    for text in prompts:
        enc = tokenizer(
            text, truncation=True, max_length=seq_len,
            padding="max_length", return_tensors="pt",
        )
        ids.append(enc["input_ids"].squeeze(0))
    return torch.stack(ids).to(device)


def load_superni_prompts(tokenizer, n, seq_len, device, seed=42,
                         local_path=None):
    """
    Load n SuperNI-style prompts (definition + instance input).
    Priority: (1) local_path JSON, (2) superni_prompts.json next to script,
              (3) allenai/natural-instructions on HF Hub, (4) DIVERSE_PROMPTS fallback.
    """
    # Try local file first (most reliable on air-gapped servers)
    script_dir  = os.path.dirname(os.path.abspath(__file__))
    default_local = os.path.join(script_dir, "superni_prompts.json")
    for path in [local_path, default_local]:
        if path and os.path.exists(path):
            with open(path) as f:
                texts = json.load(f)
            print(f"  [SuperNI] Loaded {len(texts)} prompts from {path}")
            break
    else:
        # Try HuggingFace Hub
        try:
            from datasets import load_dataset
            ds = load_dataset("allenai/natural-instructions", split="train", streaming=True)
            texts = []
            for ex in itertools.islice(ds, n * 4):
                defn = ex.get("Definition", [""])
                defn = defn[0] if isinstance(defn, list) and defn else str(defn)
                instances = ex.get("Instances", [])
                inp = instances[0]["input"] if instances else ""
                text = f"{defn.strip()} {inp.strip()}"
                if len(text) > 20:
                    texts.append(text[:512])
                if len(texts) >= n:
                    break
            if len(texts) < n:
                raise ValueError(f"Only got {len(texts)} examples")
            print(f"  [SuperNI] Loaded {len(texts)} prompts from HuggingFace Hub")
        except Exception as e:
            print(f"  [SuperNI] HF unavailable ({e}), falling back to DIVERSE_PROMPTS")
            texts = list(DIVERSE_PROMPTS)

    rng = np.random.default_rng(seed)
    if len(texts) > n:
        idxs  = rng.choice(len(texts), size=n, replace=False)
        texts = [texts[i] for i in idxs]

    ids = []
    for text in texts:
        enc = tokenizer(text, truncation=True, max_length=seq_len,
                        padding="max_length", return_tensors="pt")
        ids.append(enc["input_ids"].squeeze(0))
    return torch.stack(ids).to(device)


@torch.no_grad()
def eval_mmlu_accuracy(model, tokenizer, device, n_samples=200, seed=42):
    """
    Zero-shot MMLU accuracy: score each A/B/C/D choice by next-token log-prob.
    Uses cais/mmlu (all subjects, test split).
    """
    import random
    from datasets import load_dataset
    random.seed(seed)

    ds = load_dataset("cais/mmlu", "all", split="test")
    indices = random.sample(range(len(ds)), min(n_samples, len(ds)))
    ds = ds.select(indices)

    model.eval()
    correct, total = 0, 0
    choice_letters = ["A", "B", "C", "D"]

    for ex in ds:
        question = ex["question"]
        options  = ex["choices"]
        label    = ex["answer"]  # 0–3

        prompt = f"Question: {question}\n"
        for letter, opt in zip(choice_letters, options):
            prompt += f"{letter}. {opt}\n"
        prompt += "Answer:"

        prompt_ids = tokenizer(prompt, return_tensors="pt",
                               truncation=True, max_length=512)["input_ids"].to(device)
        prompt_len = prompt_ids.shape[1]

        best_score, best_idx = float("-inf"), 0
        for i, letter in enumerate(choice_letters[:len(options)]):
            full_ids = tokenizer(
                prompt + " " + letter, return_tensors="pt",
                truncation=True, max_length=514,
            )["input_ids"].to(device)
            out = model(full_ids)
            # log-prob of the answer letter token at position prompt_len-1
            logits    = out.logits[0, prompt_len - 1, :]
            log_probs = F.log_softmax(logits, dim=-1)
            tok_ids   = tokenizer.encode(" " + letter, add_special_tokens=False)
            if tok_ids:
                score = log_probs[tok_ids[-1]].item()
                if score > best_score:
                    best_score, best_idx = score, i

        correct += int(best_idx == label)
        total   += 1

    return correct / max(total, 1)


def load_mnli(tokenizer, batch_size, max_length=128):
    from datasets import load_dataset
    from torch.utils.data import DataLoader, TensorDataset
    ds = load_dataset("glue", "mnli", split="train")
    ds = ds.select(range(min(4000, len(ds))))
    all_ids, all_labels = [], []
    for ex in ds:
        text = f"Premise: {ex['premise']} Hypothesis: {ex['hypothesis']}"
        enc = tokenizer(text, truncation=True, max_length=max_length,
                        padding="max_length", return_tensors="pt")
        all_ids.append(enc["input_ids"].squeeze(0))
        all_labels.append(ex["label"])
    input_ids = torch.stack(all_ids)
    labels = torch.tensor(all_labels, dtype=torch.long)
    return DataLoader(TensorDataset(input_ids, labels), batch_size=batch_size, shuffle=True)


def load_wikitext_for_kl(tokenizer, n_samples, seq_len, device, seed=42):
    """Load WikiText-2 validation sequences for Option A KL."""
    from datasets import load_dataset
    ds     = load_dataset("wikitext", "wikitext-2-raw-v1", split="validation")
    tokens = tokenizer.encode("\n".join(ds["text"]))
    rng    = np.random.default_rng(seed)
    starts = rng.integers(0, max(1, len(tokens) - seq_len - 1), size=n_samples)
    chunks = [tokens[s : s + seq_len] for s in starts]
    return torch.tensor(chunks, dtype=torch.long, device=device)


# ── Evaluation on SST-2 ───────────────────────────────────────────────────────

@torch.no_grad()
def eval_sst2_accuracy(model, classifier, tokenizer, device, n_samples=500):
    from datasets import load_dataset
    ds = load_dataset("glue", "sst2", split="validation")
    ds = ds.select(range(min(n_samples, len(ds))))

    model.eval()
    classifier.eval()
    correct, total = 0, 0

    for ex in ds:
        enc = tokenizer(
            ex["sentence"], truncation=True, max_length=128,
            padding="max_length", return_tensors="pt",
        )
        input_ids = enc["input_ids"].to(device)
        out    = model(input_ids, output_hidden_states=True)
        hidden = out.hidden_states[-1][:, -1, :].float()
        logits = classifier(hidden)
        pred   = logits.argmax(dim=-1).item()
        correct += int(pred == ex["label"])
        total   += 1

    return correct / max(total, 1)


@torch.no_grad()
def eval_mnli_accuracy(model, classifier, tokenizer, device, n_samples=500):
    from datasets import load_dataset
    ds = load_dataset("glue", "mnli", split="validation_matched")
    ds = ds.select(range(min(n_samples, len(ds))))

    model.eval()
    classifier.eval()
    correct, total = 0, 0

    for ex in ds:
        text = f"Premise: {ex['premise']} Hypothesis: {ex['hypothesis']}"
        enc = tokenizer(text, truncation=True, max_length=128,
                        padding="max_length", return_tensors="pt")
        input_ids = enc["input_ids"].to(device)
        out    = model(input_ids, output_hidden_states=True)
        hidden = out.hidden_states[-1][:, -1, :].float()
        logits = classifier(hidden)
        pred   = logits.argmax(dim=-1).item()
        correct += int(pred == ex["label"])
        total   += 1

    return correct / max(total, 1)


# ── Fine-tuning ───────────────────────────────────────────────────────────────

def finetune_one_lr(base_model, tokenizer, lr, steps, batch_size, rank, alpha,
                    grad_clip, device, seed, model_name="gpt2", task="sst2"):
    """Fine-tune a fresh copy of base_model on the given task at this LR. Returns merged model."""
    torch.manual_seed(seed)
    model_copy = copy.deepcopy(base_model)
    peft_model = add_lora(model_copy, rank, alpha,
                          target_modules=_lora_targets(model_name))

    hidden_size = _get_hidden_size(base_model)
    num_classes = 3 if task == "mnli" else 2
    classifier  = nn.Linear(hidden_size, num_classes).to(device)

    params  = list(peft_model.parameters()) + list(classifier.parameters())
    opt     = torch.optim.AdamW([p for p in params if p.requires_grad], lr=lr)
    loss_fn = nn.CrossEntropyLoss()
    loader  = load_mnli(tokenizer, batch_size) if task == "mnli" else load_sst2(tokenizer, batch_size)

    peft_model.train()
    classifier.train()
    step = 0

    while step < steps:
        for input_ids, labels in loader:
            if step >= steps:
                break
            input_ids = input_ids.to(device)
            labels    = labels.to(device)

            out    = peft_model(input_ids, output_hidden_states=True)
            hidden = out.hidden_states[-1][:, -1, :].float()
            logits = classifier(hidden)
            loss   = loss_fn(logits, labels)

            if torch.isnan(loss):
                step += 1
                continue

            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(params, grad_clip)
            opt.step()
            step += 1

            if step % 100 == 0:
                print(f"    step {step}/{steps}  loss={loss.item():.4f}", flush=True)

    merged = peft_model.merge_and_unload().eval()
    return merged, classifier


# ── Per-parameter norm ────────────────────────────────────────────────────────

def compute_per_param_norm(base_model, ft_model):
    """RMS per-parameter weight displacement ‖Δθ‖ / sqrt(d)."""
    total_sq, total_d = 0.0, 0
    base_params = dict(base_model.named_parameters())
    for name, param in ft_model.named_parameters():
        if name in base_params:
            delta      = param.data.float() - base_params[name].data.float()
            total_sq  += delta.norm().item() ** 2
            total_d   += delta.numel()
    return float((total_sq / max(total_d, 1)) ** 0.5)


# ── Plotting ──────────────────────────────────────────────────────────────────

def plot_comparison(summary, out_path):
    """
    Two-panel figure:
      Left:  R vs Δ NLL (wiki NLL of ft - base)  ← pretrained capability degradation
      Right: KL (both options) vs Δ NLL           ← KL as comparator

    H1 prediction: R tracks Δ NLL (R > 1 → capability degradation);
    KL is either saturated or fails to track the same degradation.
    """
    lrs      = [r["lr"]           for r in summary]
    rs       = [r["R"]            for r in summary]
    accs     = [r["acc"]          for r in summary]
    kl_wiki   = [r["kl_wiki"]                        for r in summary]
    kl_div    = [r["kl_diverse"]                      for r in summary]
    kl_sni    = [r.get("kl_superni", float("nan"))    for r in summary]
    delta_nll = [r.get("delta_nll_wiki", float("nan")) for r in summary]
    mmlu_accs = [r.get("mmlu_acc", float("nan"))       for r in summary]

    fig, axes = plt.subplots(1, 4, figsize=(22, 5))
    colors = plt.cm.viridis(np.linspace(0.1, 0.9, len(lrs)))

    # Panel 1: R vs Δ NLL (primary H1 claim)
    ax1 = axes[0]
    for i, (r, dn, lr) in enumerate(zip(rs, delta_nll, lrs)):
        ax1.scatter(r, dn, color=colors[i], s=120, zorder=3, label=f"lr={lr:.0e}")
    ax1.axvline(x=1.0, color="red", linestyle="--", alpha=0.7, label="R=1 threshold")
    ax1.axhline(y=0.0, color="gray", linestyle=":", alpha=0.5)
    ax1.set_xlabel("R = ‖Δθ‖ / σ½", fontsize=12)
    ax1.set_ylabel("Δ NLL (wiki)  =  NLL_ft − NLL_base", fontsize=11)
    ax1.set_title("R predicts capability degradation\n(Δ NLL > 0 = forgetting)", fontsize=11)
    ax1.legend(fontsize=8)
    ax1.grid(True, alpha=0.3)

    # Panel 2: KL vs Δ NLL (comparator) — wiki, diverse, SuperNI
    ax2 = axes[1]
    for i, (kw, kd, ks, dn, lr) in enumerate(zip(kl_wiki, kl_div, kl_sni, delta_nll, lrs)):
        ax2.scatter(kw, dn, color=colors[i], s=120, marker="o", zorder=3,
                    label=f"lr={lr:.0e} (wiki)")
        ax2.scatter(kd, dn, color=colors[i], s=80,  marker="^", zorder=3,
                    label=f"lr={lr:.0e} (diverse)")
        if not np.isnan(ks):
            ax2.scatter(ks, dn, color=colors[i], s=80, marker="s", zorder=3,
                        label=f"lr={lr:.0e} (SuperNI)")
    ax2.axhline(y=0.0, color="gray", linestyle=":", alpha=0.5)
    ax2.set_xlabel("KL(base ‖ fine-tuned)", fontsize=12)
    ax2.set_ylabel("Δ NLL (wiki)", fontsize=11)
    ax2.set_title("KL vs capability degradation\n○=wiki  △=diverse  □=SuperNI", fontsize=11)
    ax2.legend(fontsize=6, ncol=2)
    ax2.grid(True, alpha=0.3)

    # Panel 3: R vs MMLU accuracy (world knowledge degradation)
    ax3 = axes[2]
    has_mmlu = any(not np.isnan(m) for m in mmlu_accs)
    for i, (r, m, lr) in enumerate(zip(rs, mmlu_accs, lrs)):
        if not np.isnan(m):
            ax3.scatter(r, m, color=colors[i], s=120, zorder=3, label=f"lr={lr:.0e}")
    ax3.axvline(x=1.0, color="red", linestyle="--", alpha=0.7, label="R=1 threshold")
    ax3.set_xlabel("R = ‖Δθ‖ / σ½", fontsize=12)
    ax3.set_ylabel("MMLU Accuracy (zero-shot)", fontsize=11)
    ax3.set_title("R predicts world knowledge loss\n(MMLU = factual knowledge, not fluency)",
                  fontsize=11)
    if not has_mmlu:
        ax3.text(0.5, 0.5, "MMLU not computed", transform=ax3.transAxes,
                 ha="center", va="center", fontsize=10, color="gray")
    ax3.legend(fontsize=8)
    ax3.grid(True, alpha=0.3)

    # Panel 4: R vs task accuracy (sanity check — task was actually learned)
    ax4 = axes[3]
    task_label = summary[0].get("task", "sst2").upper()
    for i, (r, acc, lr) in enumerate(zip(rs, accs, lrs)):
        ax4.scatter(r, acc, color=colors[i], s=120, zorder=3, label=f"lr={lr:.0e}")
    ax4.axvline(x=1.0, color="red", linestyle="--", alpha=0.7, label="R=1 threshold")
    ax4.set_xlabel("R = ‖Δθ‖ / σ½", fontsize=12)
    ax4.set_ylabel(f"{task_label} Accuracy", fontsize=11)
    ax4.set_title(f"Task learning sanity check\n({task_label} acc — not H1's primary claim)",
                  fontsize=11)
    ax4.legend(fontsize=8)
    ax4.grid(True, alpha=0.3)

    fig.suptitle("H1: σ½ basin radius vs KL as pretrained-capability discriminators",
                 fontsize=13, y=1.02)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved figure: {out_path}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    args   = get_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(args.seed)

    model_tag = args.model if args.task == "sst2" else f"{args.model}_{args.task}"
    out_dir = os.path.join(args.output_dir, model_tag)
    os.makedirs(out_dir, exist_ok=True)

    summary_path = os.path.join(out_dir, "summary.json")

    if args.replot:
        with open(summary_path) as f:
            summary = json.load(f)
        plot_comparison(summary, os.path.join(out_dir, "kl_vs_r_comparison.png"))
        return

    if args.sigma_scope == "lora":
        print("=" * 70)
        print("  sigma_scope=lora (DEFAULT as of 2026-07-21)")
        print("  This does NOT reproduce the numbers in SUMMARY.md /")
        print("  pretrained_density.json — those were all computed with the")
        print("  legacy whole-decoder-block scope.")
        print("  Pass --sigma_scope block to reproduce historical numbers.")
        print("=" * 70)

    print(f"\n{'='*60}")
    print(f"  H1 KL Comparison — {args.model}  task={args.task}")
    print(f"  LRs: {args.lrs}")
    print(f"  rank={args.rank}  steps={args.steps}  device={device}")
    print(f"{'='*60}\n")

    # ── Load base model + tokenizer ───────────────────────────────────────────
    print("Loading base model...")
    base_model, _ = _load_model_auto(args.model, device)
    tokenizer     = _get_tokenizer(args.model)

    # sigma_scope controls which parameters σ½ is calibrated over. 'lora'
    # (default) restricts the perturbation to LoRA's own target modules, so σ½
    # certifies robustness over the same subspace fine-tuning actually moves
    # in — fixes the scope mismatch flagged in SUMMARY.md ("σ½ certifies a
    # much larger parameter space than LoRA ever displaces"). 'block' is the
    # legacy whole-decoder-block scope every previously-reported σ½ used.
    block_names = _get_target_names_auto(base_model, args.model)
    target_names = (_get_lora_scoped_names(base_model, args.model)
                    if args.sigma_scope == "lora" else block_names)
    assert len(target_names) > 0, (
        f"sigma_scope='{args.sigma_scope}' matched ZERO parameters for "
        f"{args.model} — before spending any GPU time, check _lora_targets() "
        f"module names against this model's actual named_parameters()."
    )
    # numel-based, not tensor-count-based: a handful of matched weight matrices
    # can outweigh hundreds of small tensors, so "N tensors matched" alone is a
    # misleading fraction. This is what actually determines how much of the
    # model's weight-mass σ½ is being calibrated over.
    n_target_params = sum(p.numel() for n, p in base_model.named_parameters()
                          if n in target_names)
    n_block_params  = sum(p.numel() for n, p in base_model.named_parameters()
                          if n in block_names)
    n_total_params  = sum(p.numel() for _, p in base_model.named_parameters())
    print(f"  sigma_scope={args.sigma_scope}: perturbing {n_target_params:,}/"
          f"{n_total_params:,} params ({100*n_target_params/n_total_params:.1f}%) "
          f"across {len(target_names)} tensors")
    if args.sigma_scope == "lora":
        print(f"  for reference: whole-block scope would be {n_block_params:,} "
              f"params ({100*n_block_params/n_total_params:.1f}%) — lora scope "
              f"is {100*n_target_params/n_block_params:.1f}% of that")

    # ── Set eval_slack if not specified: standardize to 1% of this model's own
    # baseline WikiText-2 NLL (same methodology as recompute_h1_sigma_relslack.py).
    # A fixed absolute slack (the old default) makes sigma_half incomparable
    # across models with different baseline NLL scales — this was the root
    # cause of the retracted "sigma_half scale-invariance" claim for GPT-2 vs
    # Llama (see SUMMARY.md). Computing it here means every new model gets a
    # correctly standardized slack automatically, with no separate pre-step.
    if args.eval_slack is None:
        RELATIVE_SLACK = 0.01
        inp_slack, tgt_slack = load_data(
            tokenizer, args.n_text_samples, args.seq_len, device, seed=args.seed
        )
        nll_baseline   = compute_nll_loss(base_model, inp_slack, tgt_slack)
        args.eval_slack = RELATIVE_SLACK * nll_baseline
        print(f"  NLL_baseline = {nll_baseline:.5f}")
        print(f"  eval_slack auto-set to {args.eval_slack:.5f} "
              f"({RELATIVE_SLACK*100:.0f}% of baseline NLL) for {args.model}")

    # ── Compute σ½ of pretrained model (once, reused for all LRs) ────────────
    print("\nComputing σ½ of pretrained model...")
    # Separate filename per scope: every σ½ in SUMMARY.md before 2026-07-21 was
    # computed with sigma_scope='block' and cached as "pretrained_density.json"
    # (or "..._relslack.json" via recompute_h1_sigma_relslack.py). The new
    # default ('lora') must never silently load or overwrite that cache — it
    # is a different, non-comparable quantity — so it gets its own filename.
    density_filename = ("pretrained_density.json" if args.sigma_scope == "block"
                         else "pretrained_density_lorascope.json")
    sigma_pretrained_path = os.path.join(out_dir, density_filename)
    if os.path.exists(sigma_pretrained_path):
        with open(sigma_pretrained_path) as f:
            pre_result = json.load(f)
        sigma_half_pre = pre_result["sigma_half"]
        print(f"  Loaded cached σ½ = {sigma_half_pre:.6f}")
    else:
        inp_pre, tgt_pre = load_data(
            tokenizer, args.n_text_samples, args.seq_len, device, seed=args.seed
        )
        base_sigmas = [0.0001, 0.0002, 0.0005, 0.001, 0.002, 0.005, 0.01, 0.02, 0.05]
        # This grid was calibrated for whole-block perturbation (~1e-4 to 7e-3
        # observed σ½ across GPT-2/Llama/Mistral). A narrower 'lora' scope
        # perturbs fewer independent parameters, so aggregate NLL degradation
        # per unit σ is smaller and σ½ can land higher — possibly outside this
        # grid. half_max_sigma_adaptive extends the top of the grid (capped)
        # and warns loudly rather than silently returning a clipped value that
        # would look like a real measurement. See its docstring for why.
        sigma_half_pre, density_results, was_capped = half_max_sigma_adaptive(
            base_model, target_names, inp_pre, tgt_pre,
            base_sigmas=base_sigmas, N=args.n_perturb,
            eval_slack=args.eval_slack, tag=f"{args.model}/{args.sigma_scope}",
        )
        print(f"  σ½ = {sigma_half_pre:.6f}" + ("  [LOWER BOUND — see warning above]" if was_capped else ""))
        with open(sigma_pretrained_path, "w") as f:
            json.dump({
                "sigma_half": sigma_half_pre,
                "sigma_half_is_lower_bound": was_capped,
                "densities":  {str(r["sigma"]): r["ci_lower"] for r in density_results},
                "sigma_scope": args.sigma_scope,
                "n_target_params": n_target_params,
                "n_total_params": n_total_params,
            }, f, indent=2)

    # ── Prepare KL evaluation data (load once) ────────────────────────────────
    print("\nPreparing KL evaluation data...")
    wiki_ids     = load_wikitext_for_kl(tokenizer, args.kl_n_wiki,
                                        args.kl_seq_len, device, seed=args.seed)
    diverse_ids  = encode_diverse_prompts(tokenizer, DIVERSE_PROMPTS,
                                          args.kl_seq_len, device)
    superni_ids  = load_superni_prompts(tokenizer, args.n_superni,
                                        args.kl_seq_len, device, seed=args.seed)
    print(f"  WikiText-2: {len(wiki_ids)} seqs")
    print(f"  Diverse:    {len(diverse_ids)} prompts")
    print(f"  SuperNI:    {len(superni_ids)} prompts")

    # ── Base model WikiText-2 NLL ─────────────────────────────────────────────
    # σ½ was computed from WikiText-2 NLL, so pretrained capability = this NLL.
    # Δ NLL = nll_ft - nll_base is the H1 quality signal (degradation measure).
    wiki_inp = wiki_ids[:, :-1]
    wiki_tgt = wiki_ids[:, 1:]
    nll_wiki_base = compute_nll_loss(base_model, wiki_inp, wiki_tgt)
    print(f"  Base WikiText-2 NLL: {nll_wiki_base:.4f}")

    # ── LR sweep ──────────────────────────────────────────────────────────────
    summary = []
    for lr in args.lrs:
        lr_tag   = f"lr{lr:.0e}"
        out_path = os.path.join(out_dir, f"{lr_tag}_results.json")

        if os.path.exists(out_path) and not args.force:
            with open(out_path) as f:
                result = json.load(f)
            if "mmlu_acc" in result and "kl_superni" in result:
                print(f"\n  [{lr_tag}] Loading cached result (complete)...")
                summary.append(result)
                continue
            else:
                print(f"\n  [{lr_tag}] Cached result missing mmlu_acc/kl_superni — rerunning...")

        print(f"\n{'─'*50}")
        print(f"  [{lr_tag}] Fine-tuning...")
        t0 = time.time()

        ft_model, classifier = finetune_one_lr(
            base_model, tokenizer, lr=lr, steps=args.steps,
            batch_size=args.batch_size, rank=args.rank, alpha=args.alpha,
            grad_clip=args.grad_clip, device=device, seed=args.seed,
            model_name=args.model, task=args.task,
        )

        # R = ‖Δθ‖ / σ½_pre
        per_param_norm = compute_per_param_norm(base_model, ft_model)
        R = per_param_norm / sigma_half_pre if sigma_half_pre > 0 else float("nan")

        # Task accuracy
        if args.task == "mnli":
            acc = eval_mnli_accuracy(ft_model, classifier, tokenizer, device)
        else:
            acc = eval_sst2_accuracy(ft_model, classifier, tokenizer, device)

        # KL Option A — WikiText-2
        kl_wiki = compute_kl(base_model, ft_model, wiki_ids)

        # KL Option B — diverse prompts
        kl_diverse = compute_kl(base_model, ft_model, diverse_ids)

        # KL Option C — SuperNI prompts
        kl_superni = compute_kl(base_model, ft_model, superni_ids)

        # WikiText-2 NLL of fine-tuned model (primary H1 quality signal)
        nll_wiki_ft = compute_nll_loss(ft_model, wiki_inp, wiki_tgt)
        delta_nll   = nll_wiki_ft - nll_wiki_base

        # MMLU zero-shot accuracy
        mmlu_acc = eval_mmlu_accuracy(ft_model, tokenizer, device,
                                      n_samples=args.n_mmlu, seed=args.seed)

        elapsed = time.time() - t0
        print(f"  [{lr_tag}] R={R:.4f}  acc={acc:.4f}  "
              f"Δnll={delta_nll:+.4f}  kl_wiki={kl_wiki:.4f}  "
              f"kl_diverse={kl_diverse:.4f}  kl_superni={kl_superni:.4f}  "
              f"mmlu={mmlu_acc:.4f}  ({elapsed:.0f}s)")

        result = {
            "lr":             lr,
            "rank":           args.rank,
            "steps":          args.steps,
            "task":           args.task,
            "per_param_norm": per_param_norm,
            "sigma_half_pre": sigma_half_pre,
            "R":              R,
            "acc":            acc,
            "nll_wiki_base":  nll_wiki_base,
            "nll_wiki_ft":    nll_wiki_ft,
            "delta_nll_wiki": delta_nll,
            "kl_wiki":        kl_wiki,
            "kl_diverse":     kl_diverse,
            "kl_superni":     kl_superni,
            "mmlu_acc":       mmlu_acc,
            "elapsed_s":      elapsed,
        }
        with open(out_path, "w") as f:
            json.dump(result, f, indent=2)
        summary.append(result)

    # ── Save summary + plot ───────────────────────────────────────────────────
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSummary saved: {summary_path}")

    plot_comparison(summary, os.path.join(out_dir, "kl_vs_r_comparison.png"))

    # ── Print table ───────────────────────────────────────────────────────────
    print(f"\n{'─'*110}")
    print(f"{'LR':>8}  {'R':>8}  {'Acc':>6}  {'ΔNLL':>7}  "
          f"{'KL(wiki)':>10}  {'KL(diverse)':>12}  {'KL(SuperNI)':>12}  {'MMLU':>6}")
    print(f"{'─'*110}")
    for r in summary:
        dn   = r.get("delta_nll_wiki", float("nan"))
        ksni = r.get("kl_superni",     float("nan"))
        mmlu = r.get("mmlu_acc",       float("nan"))
        print(f"  {r['lr']:>8.0e}  {r['R']:>8.4f}  {r['acc']:>6.4f}  "
              f"{dn:>+7.4f}  {r['kl_wiki']:>10.4f}  {r['kl_diverse']:>12.4f}  "
              f"{ksni:>12.4f}  {mmlu:>6.4f}")
    print(f"{'─'*110}\n")


if __name__ == "__main__":
    main()
