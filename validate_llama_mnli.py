"""
validate_llama_mnli.py
======================
Validation check: can Llama-3.2-3B reach >80% accuracy on MNLI with LoRA
in 500 steps at lr=1e-4, rank=8?

Run this BEFORE committing to the full H2 sweep. Takes ~20-30 min on an A40.
Pass/fail printed at the end.

Usage:
  python validate_llama_mnli.py
  python validate_llama_mnli.py --model meta-llama/Llama-3.2-3B --steps 500
  python validate_llama_mnli.py --steps 1000 --lr 5e-5   # if 500 steps falls short
"""

import argparse
import os
import sys
import time

import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(__file__))
from lora_density_experiment import (
    _load_model_auto,
    _get_hidden_size,
    _get_tokenizer,
    add_lora,
)

TARGET_ACCURACY = 0.80
MNLI_LABEL2ID  = {"entailment": 0, "neutral": 1, "contradiction": 2}


def get_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model",      default="meta-llama/Llama-3.2-3B")
    p.add_argument("--steps",      type=int,   default=500)
    p.add_argument("--lr",         type=float, default=1e-4)
    p.add_argument("--rank",       type=int,   default=8)
    p.add_argument("--alpha",      type=int,   default=16)
    p.add_argument("--batch_size", type=int,   default=16)
    p.add_argument("--max_len",    type=int,   default=128)
    p.add_argument("--eval_samples", type=int, default=500,
                   help="Validation examples to eval on (MNLI matched dev set)")
    p.add_argument("--grad_clip",  type=float, default=1.0)
    p.add_argument("--seed",       type=int,   default=42)
    p.add_argument("--output_dir", default="./results/validate_llama_mnli")
    return p.parse_args()


# ── Data ──────────────────────────────────────────────────────────────────────

def load_mnli(tokenizer, split, max_len, batch_size, n_samples=None, seed=42):
    """
    Load MNLI (matched) from GLUE. Returns a DataLoader of (input_ids, label).
    Label: 0=entailment, 1=neutral, 2=contradiction.

    Prompt format: "premise: {p} hypothesis: {h}" — minimal, no chat template,
    consistent with how we'll format during the H2 sweep.
    """
    from datasets import load_dataset
    ds = load_dataset("glue", "mnli", split=split)
    if n_samples is not None:
        ds = ds.shuffle(seed=seed).select(range(min(n_samples, len(ds))))

    input_ids_list, label_list = [], []
    for ex in ds:
        text = f"premise: {ex['premise']} hypothesis: {ex['hypothesis']}"
        enc  = tokenizer(
            text,
            truncation=True,
            max_length=max_len,
            padding="max_length",
            return_tensors="pt",
        )
        input_ids_list.append(enc["input_ids"].squeeze(0))
        label_list.append(ex["label"])

    input_ids = torch.stack(input_ids_list)
    labels    = torch.tensor(label_list, dtype=torch.long)

    dataset = torch.utils.data.TensorDataset(input_ids, labels)
    loader  = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=(split == "train"), drop_last=False
    )
    print(f"  MNLI {split}: {len(dataset)} examples, {len(loader)} batches")
    return loader


# ── Training ──────────────────────────────────────────────────────────────────

def train(model, classifier, loader, steps, lr, grad_clip, device):
    target_modules = ["q_proj", "v_proj", "k_proj", "o_proj"]
    peft_model = add_lora(model, rank=args.rank, alpha=args.alpha,
                          target_modules=target_modules)

    params  = list(peft_model.parameters()) + list(classifier.parameters())
    opt     = torch.optim.AdamW([p for p in params if p.requires_grad], lr=lr)
    loss_fn = nn.CrossEntropyLoss()

    peft_model.train()
    classifier.train()
    step = 0
    t0   = time.time()

    while step < steps:
        for input_ids, labels in loader:
            if step >= steps:
                break
            input_ids = input_ids.to(device)
            labels    = labels.to(device)

            out          = peft_model(input_ids, output_hidden_states=True)
            hidden       = out.hidden_states[-1][:, -1, :].float()  # last token, fp32
            logits       = classifier(hidden)
            loss         = loss_fn(logits, labels)

            if torch.isnan(loss):
                step += 1
                continue

            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(params, grad_clip)
            opt.step()
            step += 1

            if step % 50 == 0:
                elapsed = time.time() - t0
                print(f"  step {step:4d}/{steps}  loss={loss.item():.4f}  "
                      f"elapsed={elapsed:.0f}s", flush=True)

    peft_model.eval()
    classifier.eval()
    return peft_model.merge_and_unload(), classifier


# ── Evaluation ────────────────────────────────────────────────────────────────

@torch.no_grad()
def evaluate(model, classifier, loader, device):
    model.eval()
    classifier.eval()
    correct, total = 0, 0
    for input_ids, labels in loader:
        input_ids = input_ids.to(device)
        labels    = labels.to(device)
        out       = model(input_ids, output_hidden_states=True)
        hidden    = out.hidden_states[-1][:, -1, :].float()
        logits    = classifier(hidden)
        preds     = logits.argmax(dim=-1)
        correct  += (preds == labels).sum().item()
        total    += labels.size(0)
    return correct / max(total, 1)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    global args
    args   = get_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"  Validation: {args.model} on MNLI")
    print(f"  steps={args.steps}  lr={args.lr:.1e}  rank={args.rank}")
    print(f"  device={device}")
    print(f"{'='*60}\n")

    # ── Load model and tokenizer ──────────────────────────────────────────────
    print("Loading model...")
    model, _ = _load_model_auto(args.model, device)
    tokenizer = _get_tokenizer(args.model)
    hidden_size = _get_hidden_size(model)

    # ── Data ─────────────────────────────────────────────────────────────────
    print("\nLoading MNLI...")
    train_loader = load_mnli(tokenizer, "train",
                             args.max_len, args.batch_size, seed=args.seed)
    val_loader   = load_mnli(tokenizer, "validation_matched",
                             args.max_len, args.batch_size,
                             n_samples=args.eval_samples, seed=args.seed)

    # ── Classifier head (3-class) ─────────────────────────────────────────────
    classifier = nn.Linear(hidden_size, 3).to(device)

    # ── Train ─────────────────────────────────────────────────────────────────
    print(f"\nFine-tuning for {args.steps} steps...")
    t_start = time.time()
    merged_model, classifier = train(
        model, classifier, train_loader,
        steps=args.steps, lr=args.lr, grad_clip=args.grad_clip, device=device,
    )
    train_time = time.time() - t_start

    # ── Evaluate ──────────────────────────────────────────────────────────────
    print("\nEvaluating on MNLI validation_matched...")
    acc = evaluate(merged_model, classifier, val_loader, device)

    # ── Report ────────────────────────────────────────────────────────────────
    passed = acc >= TARGET_ACCURACY
    status = "PASS" if passed else "FAIL"

    print(f"\n{'='*60}")
    print(f"  Result: {status}")
    print(f"  Accuracy:  {acc:.4f} ({acc*100:.1f}%)")
    print(f"  Threshold: {TARGET_ACCURACY:.0%}")
    print(f"  Train time: {train_time:.0f}s")
    print(f"{'='*60}\n")

    if not passed:
        print("  Suggestions if FAIL:")
        print("    - Try --steps 1000 (double training)")
        print("    - Try --lr 5e-5 (more conservative)")
        print("    - Try --rank 16 (more LoRA capacity)")
        print("    - Try a different task pair (SST-2 or AGNews as Phase 1)")

    import json
    result = {
        "model":     args.model,
        "steps":     args.steps,
        "lr":        args.lr,
        "rank":      args.rank,
        "accuracy":  acc,
        "threshold": TARGET_ACCURACY,
        "passed":    passed,
        "train_time_s": train_time,
    }
    out_path = os.path.join(args.output_dir, "validation_result.json")
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"  Result saved to {out_path}")

    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
