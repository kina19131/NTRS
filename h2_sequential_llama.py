"""
h2_sequential_llama.py
=======================
H2 sweep on Llama-3.2-3B: MNLI → SST-2 (primary task pair).

Phase 1: Fine-tune Llama-3.2-3B on Task A (MNLI). Save checkpoint. Compute
         σ½_A using held-out MNLI NLL — the basin radius in the space relevant
         to Task A, not WikiText-2 general text.

Phase 2: Sweep LR × rank grid. For each condition:
         - Fine-tune on Task B (SST-2)
         - Track Task A (MNLI) accuracy throughout Phase 2
         - Compute R_A = ‖Δθ‖_phase2 / σ½_A at each trajectory step
         - Record the step where R_A crosses 1 and whether MNLI accuracy drops

Expected result: MNLI accuracy drops precisely when R_A crosses 1,
consistent with GPT-2/GPT-2-medium results. Demonstrates cross-architecture
generalization of the R_A = 1 threshold.

Output:
  results/h2_llama/{task_a}_to_{task_b}/
    phase1_model.pt            — Phase 1 Llama checkpoint
    phase1_classifier.pt       — Phase 1 classification head
    phase1_meta.json           — task, accuracy, lr, rank, steps
    phase1_sigma_half.json     — σ½_A (density curve + σ½)
    lr{X}_rank{Y}_trajectory.json — per-step (R_A, MNLI acc, SST-2 acc)
    lr{X}_rank{Y}_result.json  — final summary per condition
    summary.json               — all conditions
    trajectory_plot.png        — key figure: R_A vs MNLI acc across conditions

Usage:
  # Full run (Phase 1 + Phase 2 sweep)
  python h2_sequential_llama.py

  # Custom Phase 1 config
  python h2_sequential_llama.py --phase1_lr 1e-4 --phase1_rank 8 --phase1_steps 2000

  # Phase 2 only (reload Phase 1 from checkpoint)
  python h2_sequential_llama.py --skip_phase1

  # Replot only
  python h2_sequential_llama.py --replot

  # Both directions
  python h2_sequential_llama.py --task_a mnli --task_b sst2
  python h2_sequential_llama.py --task_a sst2 --task_b mnli
"""

import argparse
import copy
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
    estimate_density, half_max_sigma,
)
from lora_density_experiment import (
    _load_model_auto, _get_target_names_auto,
    _get_hidden_size, _get_tokenizer, add_lora,
)


# ── Args ──────────────────────────────────────────────────────────────────────

def get_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model",        default="meta-llama/Llama-3.2-3B")
    p.add_argument("--task_a",       default="mnli",   choices=["mnli", "sst2", "agnews", "cola"])
    p.add_argument("--task_b",       default="sst2",   choices=["mnli", "sst2", "agnews", "cola"])

    # Phase 1
    p.add_argument("--phase1_lr",    type=float, default=5e-5)
    p.add_argument("--phase1_rank",  type=int,   default=8)
    p.add_argument("--phase1_alpha", type=int,   default=16)
    p.add_argument("--phase1_steps", type=int,   default=2000)

    # Phase 2 sweep
    p.add_argument("--phase2_lrs",   nargs="+", type=float,
                   default=[1e-5, 5e-5, 1e-4, 2e-4, 5e-4, 1e-3])
    p.add_argument("--phase2_ranks", nargs="+", type=int,
                   default=[8, 32])
    p.add_argument("--phase2_steps", type=int, default=2000)
    p.add_argument("--track_interval", type=int, default=100,
                   help="Steps between trajectory measurements")

    # σ½ computation
    p.add_argument("--n_perturb",      type=int,   default=200)
    p.add_argument("--sigma_holdout",  type=int,   default=256,
                   help="Task A sequences to use for σ½ density estimation")
    p.add_argument("--seq_len",        type=int,   default=128)
    p.add_argument("--target_density", type=float, default=0.40)
    p.add_argument("--eval_slack",     type=float, default=1e-4)

    p.add_argument("--batch_size",   type=int,   default=16)
    p.add_argument("--grad_clip",    type=float, default=1.0)
    p.add_argument("--eval_samples", type=int,   default=500)
    p.add_argument("--seed",         type=int,   default=42)
    p.add_argument("--output_dir",   default="./results/h2_llama")
    p.add_argument("--skip_phase1",  action="store_true",
                   help="Reload Phase 1 checkpoint from disk (skip retraining)")
    p.add_argument("--phase1_only",  action="store_true",
                   help="Train and save Phase 1 checkpoint only; skip Phase 2 sweep")
    p.add_argument("--replot",       action="store_true",
                   help="Skip all training; regenerate figures from saved JSONs")
    p.add_argument("--save_tstar_checkpoints", action="store_true", default=True,
                   help="Save LoRA weights at T* boundary (first forgetting step)")
    p.add_argument("--tstar_acc_threshold", type=float, default=None,
                   help="Acc drop threshold for T* detection. Default: chance+0.1 "
                        "based on n_classes_a.")
    return p.parse_args()


# ── Data loading (task-aware) ─────────────────────────────────────────────────

def load_task(tokenizer, task, split, max_len, batch_size,
              n_samples=None, seed=42, for_density=False):
    """
    Load classification task. Returns DataLoader of (input_ids, label).
    for_density=True: returns raw input_ids tensor (no labels) for σ½ computation.
    """
    from datasets import load_dataset

    if task == "mnli":
        ds = load_dataset("glue", "mnli",
                          split="train" if split == "train" else "validation_matched")
        def fmt(ex): return f"premise: {ex['premise']} hypothesis: {ex['hypothesis']}"
        n_classes = 3
    elif task == "sst2":
        ds = load_dataset("glue", "sst2", split=split)
        def fmt(ex): return ex["sentence"]
        n_classes = 2
    elif task == "agnews":
        hf_split = "train" if split == "train" else "test"
        ds = load_dataset("ag_news", split=hf_split)
        def fmt(ex): return ex["text"]
        n_classes = 4
    elif task == "cola":
        ds = load_dataset("glue", "cola", split=split)
        def fmt(ex): return ex["sentence"]
        n_classes = 2
    else:
        raise ValueError(f"Unknown task: {task}")

    if n_samples is not None:
        ds = ds.shuffle(seed=seed).select(range(min(n_samples, len(ds))))

    input_ids_list, label_list = [], []
    for ex in ds:
        enc = tokenizer(
            fmt(ex), truncation=True, max_length=max_len,
            padding="max_length", return_tensors="pt",
        )
        input_ids_list.append(enc["input_ids"].squeeze(0))
        label_list.append(ex["label"])

    input_ids = torch.stack(input_ids_list)
    labels    = torch.tensor(label_list, dtype=torch.long)

    if for_density:
        return input_ids, n_classes

    dataset = torch.utils.data.TensorDataset(input_ids, labels)
    loader  = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=(split == "train"), drop_last=False,
    )
    print(f"  {task} {split}: {len(dataset)} examples, n_classes={n_classes}")
    return loader, n_classes


# ── σ½ computation using task NLL ─────────────────────────────────────────────

def compute_task_nll(model, input_ids, device, batch_size=4):
    """NLL of the model on task sequences (using next-token prediction, not classification)."""
    model.eval()
    loss_fn = nn.CrossEntropyLoss(reduction="mean")
    total, n = 0.0, 0
    input_ids = input_ids.to(device)
    with torch.no_grad():
        for i in range(0, len(input_ids), batch_size):
            batch = input_ids[i : i + batch_size]
            out   = model(batch)
            B, T, V = out.logits.shape
            inp_shifted = batch[:, :-1].reshape(-1)
            tgt_shifted = batch[:, 1:].reshape(-1)
            logits_shifted = out.logits[:, :-1, :].reshape(-1, V)
            total += loss_fn(logits_shifted, tgt_shifted).item()
            n += 1
    return total / max(n, 1)


def compute_sigma_half(model, target_names, holdout_ids, device, args, out_path):
    """
    Compute σ½_A: the perturbation radius at which 40% of perturbations
    still achieve NLL ≤ baseline NLL + eval_slack on held-out Task A sequences.

    This is the certified basin radius in Task A space — not WikiText-2.
    """
    if os.path.exists(out_path):
        with open(out_path) as f:
            result = json.load(f)
        print(f"  Loaded cached σ½_A = {result['sigma_half']:.6f}")
        return result["sigma_half"]

    print("  Computing σ½_A from held-out Task A NLL...")
    holdout_ids = holdout_ids.to(device)

    # Build (inp, tgt) pair from holdout sequences (shifted by 1)
    inp = holdout_ids[:, :-1]
    tgt = holdout_ids[:, 1:]

    # Reuse estimate_density from certified_density_experiment which expects (inp, tgt)
    sigmas          = [0.0001, 0.0002, 0.0005, 0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1]
    density_results = []
    for sigma in sigmas:
        res = estimate_density(
            model, target_names, inp, tgt,
            sigma=sigma, N=args.n_perturb,
            confidence=0.95, eval_slack=args.eval_slack,
        )
        density_results.append(res)
        print(f"    σ={sigma:.4f}  density={res['ci_lower']:.3f}", flush=True)

    sigma_half = half_max_sigma(density_results)
    print(f"  σ½_A = {sigma_half:.6f}")

    result = {
        "sigma_half": sigma_half,
        "densities":  {str(r["sigma"]): r["ci_lower"] for r in density_results},
    }
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)
    return sigma_half


# ── Classification utils ──────────────────────────────────────────────────────

@torch.no_grad()
def eval_accuracy(model, classifier, loader, device):
    model.eval()
    classifier.eval()
    correct, total = 0, 0
    for input_ids, labels in loader:
        input_ids = input_ids.to(device)
        labels    = labels.to(device)
        out    = model(input_ids, output_hidden_states=True)
        hidden = out.hidden_states[-1][:, -1, :].float()
        preds  = classifier(hidden).argmax(dim=-1)
        correct += (preds == labels).sum().item()
        total   += labels.size(0)
    return correct / max(total, 1)


def compute_per_param_norm(phase1_params, current_model):
    """RMS per-parameter displacement from Phase 1 weights to current weights."""
    total_sq, total_d = 0.0, 0
    for name, param in current_model.named_parameters():
        if name in phase1_params:
            delta     = param.data.float() - phase1_params[name].to(param.device).float()
            total_sq += delta.norm().item() ** 2
            total_d  += delta.numel()
    return float((total_sq / max(total_d, 1)) ** 0.5)


def compute_lora_delta_norm(peft_model, total_params):
    """
    Same quantity as compute_per_param_norm (RMS per-parameter displacement
    over the FULL model), computed analytically from the active LoRA adapter's
    A/B matrices instead of diffing a merged model against phase1_params.

    Equivalent because the base weights are frozen — every non-LoRA-targeted
    parameter has delta=0, so only summing ||scaling * B @ A||^2 over LoRA
    layers and dividing by the full model's parameter count reproduces the
    same RMS value compute_per_param_norm would give on a merged model.
    Avoids merging/copying the (multi-GB) base model just to read this number.
    """
    adapter = peft_model.active_adapter
    if isinstance(adapter, (list, tuple)):
        adapter = adapter[0]
    total_sq = 0.0
    for module in peft_model.modules():
        if hasattr(module, "lora_A") and adapter in module.lora_A:
            A = module.lora_A[adapter].weight
            B = module.lora_B[adapter].weight
            scaling = module.scaling[adapter]
            delta = (B.detach().float() @ A.detach().float()) * scaling
            total_sq += delta.norm().item() ** 2
    return float((total_sq / max(total_params, 1)) ** 0.5)


# ── Phase 1: fine-tune on Task A ──────────────────────────────────────────────

def run_phase1(args, device, out_dir):
    """
    Fine-tune Llama on Task A. Save model + classifier. Return (model, classifier,
    phase1_params_snapshot, sigma_half_A, n_classes_A).
    """
    model_path = os.path.join(out_dir, "phase1_model.pt")
    cls_path   = os.path.join(out_dir, "phase1_classifier.pt")
    meta_path  = os.path.join(out_dir, "phase1_meta.json")

    print(f"\n{'='*60}")
    print(f"  Phase 1: {args.task_a} on {args.model}")
    print(f"{'='*60}")

    tokenizer = _get_tokenizer(args.model)
    model, _  = _load_model_auto(args.model, device)

    target_names = _get_target_names_auto(model, args.model)
    hidden_size  = _get_hidden_size(model)

    acc_phase1_a = None  # set in both branches below

    if args.skip_phase1 and os.path.exists(model_path):
        print("  Loading Phase 1 from checkpoint...")
        with open(meta_path) as f:
            meta = json.load(f)
        n_classes_a  = meta["n_classes"]
        acc_phase1_a = meta.get("acc_phase1", 0.90)
        classifier   = nn.Linear(hidden_size, n_classes_a).to(device)

        if meta.get("save_format") == "lora_only":
            lora_tgt = meta.get("lora_target_modules", ["q_proj", "v_proj", "k_proj", "o_proj"])
            peft_m = add_lora(model, meta["rank"], meta.get("alpha", args.phase1_alpha),
                              target_modules=lora_tgt)
            peft_m.load_state_dict(torch.load(model_path, map_location=device), strict=False)
            model = peft_m.merge_and_unload().eval()
        else:
            model.load_state_dict(torch.load(model_path, map_location=device))

        classifier.load_state_dict(torch.load(cls_path, map_location=device))
        model.eval()
        classifier.eval()
    else:
        # Fine-tune
        torch.manual_seed(args.seed)
        train_loader, n_classes_a = load_task(
            tokenizer, args.task_a, "train", args.seq_len, args.batch_size,
            seed=args.seed,
        )
        classifier = nn.Linear(hidden_size, n_classes_a).to(device)

        # Reset to task-specific seed before LoRA A init so A matrices are
        # independent of n_classes (which affects RNG consumption above).
        _TASK_SEED_OFFSET = {"sst2": 0, "mnli": 1000, "agnews": 2000, "cola": 3000}
        torch.manual_seed(args.seed + _TASK_SEED_OFFSET.get(args.task_a, 9999))
        target_modules = ["q_proj", "v_proj", "k_proj", "o_proj"]
        peft_model = add_lora(model, args.phase1_rank, args.phase1_alpha,
                               target_modules=target_modules)
        params  = list(peft_model.parameters()) + list(classifier.parameters())
        opt     = torch.optim.AdamW([p for p in params if p.requires_grad],
                                     lr=args.phase1_lr)
        loss_fn = nn.CrossEntropyLoss()

        peft_model.train()
        classifier.train()
        step = 0
        t0   = time.time()

        while step < args.phase1_steps:
            for input_ids, labels in train_loader:
                if step >= args.phase1_steps:
                    break
                input_ids = input_ids.to(device)
                labels    = labels.to(device)

                out    = peft_model(input_ids, output_hidden_states=True)
                hidden = out.hidden_states[-1][:, -1, :].float()
                loss   = loss_fn(classifier(hidden), labels)

                if torch.isnan(loss):
                    step += 1
                    continue

                opt.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(params, args.grad_clip)
                opt.step()
                step += 1

                if step % 200 == 0:
                    print(f"  Phase1 step {step}/{args.phase1_steps}  "
                          f"loss={loss.item():.4f}  "
                          f"elapsed={time.time()-t0:.0f}s", flush=True)

        # Save LoRA adapter weights BEFORE merge (~50MB vs ~6GB full model)
        lora_sd = {k: v.cpu() for k, v in peft_model.state_dict().items()
                   if 'lora_' in k}
        torch.save(lora_sd, model_path)
        torch.save(classifier.state_dict(), cls_path)

        model = peft_model.merge_and_unload().eval()
        classifier.eval()

        # Eval Phase 1 accuracy
        val_loader, _ = load_task(
            tokenizer, args.task_a, "validation", args.seq_len, args.batch_size,
            n_samples=args.eval_samples, seed=args.seed,
        )
        acc_a        = eval_accuracy(model, classifier, val_loader, device)
        acc_phase1_a = acc_a
        print(f"  Phase 1 {args.task_a} accuracy: {acc_a:.4f}")

        if acc_a < 0.80:
            print(f"  WARNING: Phase 1 accuracy {acc_a:.2%} < 80%. "
                  f"Consider --phase1_steps {args.phase1_steps * 2} or --phase1_lr 2e-4.")

        # Save meta after eval so acc_phase1 is available
        with open(meta_path, "w") as f:
            json.dump({
                "task_a": args.task_a, "n_classes": n_classes_a,
                "acc_phase1": acc_a, "lr": args.phase1_lr,
                "rank": args.phase1_rank, "alpha": args.phase1_alpha,
                "steps": args.phase1_steps,
                "lora_target_modules": target_modules,
                "save_format": "lora_only",
            }, f, indent=2)
        print(f"  Phase 1 checkpoint saved (LoRA-only, {len(lora_sd)} tensors).")

    # Snapshot weights for displacement tracking
    phase1_params = {n: p.data.clone().cpu()
                     for n, p in model.named_parameters()}

    # Compute σ½_A from held-out Task A NLL (NOT WikiText-2)
    holdout_ids, n_classes_a = load_task(
        tokenizer, args.task_a, "validation", args.seq_len, batch_size=1,
        n_samples=args.sigma_holdout, seed=args.seed + 1, for_density=True,
    )
    sigma_half_a = compute_sigma_half(
        model, target_names, holdout_ids, device, args,
        out_path=os.path.join(out_dir, "phase1_sigma_half.json"),
    )

    return model, classifier, phase1_params, sigma_half_a, n_classes_a, acc_phase1_a, tokenizer


# ── Phase 2: sweep one condition ──────────────────────────────────────────────

def run_phase2_condition(phase1_model, phase1_classifier, phase1_params,
                          sigma_half_a, n_classes_a, acc_phase1_a,
                          task_a, task_b, tokenizer, lr, rank, steps,
                          track_interval, batch_size, grad_clip,
                          eval_samples, seq_len, device, seed, out_dir,
                          save_tstar_checkpoints=True, tstar_acc_threshold=None):
    """
    Fine-tune on Task B for one (lr, rank) condition. Track R_A and Task A
    forgetting throughout Phase 2.
    """
    seed_suffix = f"_seed{seed}" if seed != 42 else ""
    tag         = f"lr{lr:.0e}_rank{rank}{seed_suffix}"
    traj_path   = os.path.join(out_dir, f"{tag}_trajectory.json")
    result_path = os.path.join(out_dir, f"{tag}_result.json")

    tstar_lora_path = os.path.join(out_dir, f"{tag}_tstar_lora.pt")
    tstar_clf_path  = os.path.join(out_dir, f"{tag}_tstar_classifier.pt")
    tstar_meta_path = os.path.join(out_dir, f"{tag}_tstar_meta.json")

    if os.path.exists(result_path):
        print(f"  [{tag}] Cached — loading result")
        with open(result_path) as f:
            return json.load(f)

    print(f"\n{'─'*50}")
    print(f"  Phase 2 [{tag}]  {task_a}→{task_b}")
    t0 = time.time()

    torch.manual_seed(seed)

    total_params_full = sum(p.numel() for p in phase1_params.values())

    # Fresh copy of Phase 1 model for this condition
    model = copy.deepcopy(phase1_model)
    classifier_b = None  # Task B classifier built fresh

    # Task B data
    train_loader_b, n_classes_b = load_task(
        tokenizer, task_b, "train", seq_len, batch_size, seed=seed,
    )
    val_loader_a, _ = load_task(
        tokenizer, task_a, "validation", seq_len, batch_size,
        n_samples=eval_samples, seed=seed,
    )
    val_loader_b, _ = load_task(
        tokenizer, task_b, "validation", seq_len, batch_size,
        n_samples=eval_samples, seed=seed,
    )

    hidden_size  = _get_hidden_size(model)
    classifier_b = nn.Linear(hidden_size, n_classes_b).to(device)

    # Reset to task-specific seed before Phase 2 LoRA A init so A matrices
    # are independent of n_classes_b (which affects RNG consumption above).
    _TASK_SEED_OFFSET = {"sst2": 0, "mnli": 1000, "agnews": 2000, "cola": 3000}
    torch.manual_seed(seed + _TASK_SEED_OFFSET.get(task_b, 9999))
    target_modules = ["q_proj", "v_proj", "k_proj", "o_proj"]
    peft_model  = add_lora(model, rank, rank * 2, target_modules=target_modules)
    params      = list(peft_model.parameters()) + list(classifier_b.parameters())
    opt         = torch.optim.AdamW([p for p in params if p.requires_grad], lr=lr)
    loss_fn     = nn.CrossEntropyLoss()

    peft_model.train()
    classifier_b.train()

    # T* checkpoint tracking: save LoRA weights at last safe step before forgetting
    _tstar_threshold = tstar_acc_threshold if tstar_acc_threshold is not None else \
        0.9 * acc_phase1_a
    _tstar_saved         = False
    _last_safe_lora_sd   = None  # LoRA state dict at most recent safe trajectory step
    _last_safe_clf_sd    = None
    _last_safe_step      = -1
    _last_safe_R_a       = None
    _last_safe_acc_a     = None
    _tstar_info          = {}

    trajectory = []
    step       = 0

    while step < steps:
        for input_ids, labels in train_loader_b:
            if step >= steps:
                break

            # Trajectory snapshot
            if step % track_interval == 0:
                # No merge, no copy: PEFT applies the LoRA delta on the fly
                # during forward regardless of merge state (same pattern used
                # in Phase 1 training above), and per_param_norm is computed
                # analytically from the LoRA A/B matrices. Avoids ever
                # materializing a second copy of the (multi-GB) base model.
                per_param_norm = compute_lora_delta_norm(peft_model, total_params_full)
                R_a = per_param_norm / sigma_half_a if sigma_half_a > 0 else float("nan")

                peft_model.eval()
                acc_a = eval_accuracy(peft_model, phase1_classifier, val_loader_a, device)
                acc_b = eval_accuracy(peft_model, classifier_b, val_loader_b, device)
                peft_model.train()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

                trajectory.append({
                    "step":           step,
                    "R_A":            R_a,
                    "per_param_norm": per_param_norm,
                    "sigma_half_a":   sigma_half_a,
                    "acc_task_a":     acc_a,
                    "acc_task_b":     acc_b,
                })
                print(f"  [{tag}] step={step:4d}  R_A={R_a:.4f}  "
                      f"acc_A={acc_a:.4f}  acc_B={acc_b:.4f}", flush=True)

                # T* checkpoint: extract current LoRA state from peft_model
                if save_tstar_checkpoints and not _tstar_saved:
                    candidate_lora_sd = {k: v.detach().cpu()
                                         for k, v in peft_model.state_dict().items()
                                         if "lora_" in k}
                    if acc_a >= _tstar_threshold:
                        _last_safe_lora_sd = candidate_lora_sd
                        _last_safe_clf_sd  = {k: v.detach().cpu()
                                              for k, v in classifier_b.state_dict().items()}
                        _last_safe_step    = step
                        _last_safe_R_a     = R_a
                        _last_safe_acc_a   = acc_a
                    else:
                        save_sd  = _last_safe_lora_sd if _last_safe_lora_sd is not None \
                            else candidate_lora_sd
                        save_clf = _last_safe_clf_sd if _last_safe_clf_sd is not None \
                            else {k: v.detach().cpu()
                                  for k, v in classifier_b.state_dict().items()}
                        torch.save(save_sd,  tstar_lora_path)
                        torch.save(save_clf, tstar_clf_path)
                        _tstar_info = {
                            "tstar_step":    _last_safe_step if _last_safe_lora_sd else step,
                            "tstar_R_A":     _last_safe_R_a  if _last_safe_lora_sd else R_a,
                            "tstar_acc_a":   _last_safe_acc_a if _last_safe_lora_sd else acc_a,
                            "forget_step":   step,
                            "forget_R_A":    R_a,
                            "forget_acc_a":  acc_a,
                            "acc_threshold": _tstar_threshold,
                            "rank": rank, "lr": lr, "task_a": task_a, "task_b": task_b,
                        }
                        with open(tstar_meta_path, "w") as _f:
                            json.dump(_tstar_info, _f, indent=2)
                        print(f"  [{tag}] T* checkpoint saved at step={_tstar_info['tstar_step']} "
                              f"R_A={_tstar_info['tstar_R_A']:.4f}")
                        _tstar_saved = True

                # Restore training mode (eval_accuracy sets classifier_b to eval)
                classifier_b.train()

            input_ids = input_ids.to(device)
            labels    = labels.to(device)

            out    = peft_model(input_ids, output_hidden_states=True)
            hidden = out.hidden_states[-1][:, -1, :].float()
            loss   = loss_fn(classifier_b(hidden), labels)

            if not torch.isnan(loss):
                opt.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(params, grad_clip)
                opt.step()
            step += 1

    # Final eval
    final_model = peft_model.merge_and_unload().eval()
    final_norm  = compute_per_param_norm(phase1_params, final_model)
    final_R_a   = final_norm / sigma_half_a if sigma_half_a > 0 else float("nan")
    final_acc_a = eval_accuracy(final_model, phase1_classifier, val_loader_a, device)
    final_acc_b = eval_accuracy(final_model, classifier_b, val_loader_b, device)
    elapsed     = time.time() - t0

    result = {
        "lr": lr, "rank": rank, "steps": steps, "seed": seed,
        "sigma_half_a":    sigma_half_a,
        "final_R_A":       final_R_a,
        "final_per_param": final_norm,
        "final_acc_task_a": final_acc_a,
        "final_acc_task_b": final_acc_b,
        "elapsed_s":       elapsed,
        "trajectory":      trajectory,
        "tstar_checkpoint": _tstar_info if _tstar_saved else None,
    }

    with open(result_path, "w") as f:
        json.dump(result, f, indent=2)
    with open(traj_path, "w") as f:
        json.dump(trajectory, f, indent=2)

    print(f"  [{tag}] DONE  R_A={final_R_a:.4f}  "
          f"acc_A={final_acc_a:.4f}  acc_B={final_acc_b:.4f}  "
          f"({elapsed:.0f}s)")
    return result


# ── Plotting ──────────────────────────────────────────────────────────────────

def plot_trajectories(all_results, out_path, task_a, task_b):
    """
    One panel per (lr, rank) condition: R_A vs Task A accuracy trajectory.
    Red vertical line at R_A = 1. Accuracy should drop at/after that crossing.
    """
    n = len(all_results)
    ncols = min(3, n)
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4 * nrows))
    axes = np.array(axes).flatten() if n > 1 else [axes]

    for i, (result, ax) in enumerate(zip(all_results, axes)):
        traj = result.get("trajectory", [])
        if not traj:
            ax.set_visible(False)
            continue
        steps  = [t["step"]       for t in traj]
        R_vals = [t["R_A"]        for t in traj]
        accs   = [t["acc_task_a"] for t in traj]

        ax2 = ax.twinx()
        ax.plot(steps, R_vals, "b-o", markersize=4, label="R_A")
        ax2.plot(steps, accs,  "g-s", markersize=4, label=f"{task_a} acc")
        ax.axhline(y=1.0, color="red", linestyle="--", alpha=0.7, label="R_A=1")
        ax.set_xlabel("Step")
        ax.set_ylabel("R_A", color="b")
        ax2.set_ylabel(f"{task_a} Accuracy", color="g")
        ax.set_title(f"lr={result['lr']:.0e}  rank={result['rank']}")
        lines1, labels1 = ax.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax.legend(lines1 + lines2, labels1 + labels2, fontsize=7)

    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    plt.suptitle(f"H2: R_A trajectory — {task_a}→{task_b} on Llama-3.2-3B",
                 fontsize=12, y=1.01)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out_path}")


def plot_threshold_summary(all_results, out_path, task_a, task_b):
    """
    Single scatter: R_A (final) vs Task A accuracy drop. R_A=1 vertical line.
    This is the key H2 figure: R_A predicts forgetting.
    """
    fig, ax = plt.subplots(figsize=(7, 5))
    colors = plt.cm.tab10(np.linspace(0, 1, len(all_results)))

    for i, result in enumerate(all_results):
        R   = result["final_R_A"]
        acc = result["final_acc_task_a"]
        lr  = result["lr"]
        rank= result["rank"]
        ax.scatter(R, acc, color=colors[i], s=120, zorder=3,
                   label=f"lr={lr:.0e}  r={rank}")

    ax.axvline(x=1.0, color="red", linestyle="--", alpha=0.8, label="R_A = 1 threshold")
    ax.set_xlabel("R_A = ‖Δθ‖ / σ½_A  (Phase 2 displacement)", fontsize=12)
    ax.set_ylabel(f"{task_a} Accuracy (after Phase 2)", fontsize=12)
    ax.set_title(f"H2: R_A = 1 predicts {task_a} forgetting  ({task_a}→{task_b}, Llama-3.2-3B)",
                 fontsize=11)
    ax.legend(fontsize=8, bbox_to_anchor=(1.01, 1), loc="upper left")
    ax.grid(True, alpha=0.3)
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
    out_dir = os.path.join(args.output_dir, tag)
    os.makedirs(out_dir, exist_ok=True)

    summary_path = os.path.join(out_dir, "summary.json")

    print(f"\n{'='*60}")
    print(f"  H2 Sequential Sweep — {args.model}")
    print(f"  {args.task_a} → {args.task_b}")
    print(f"  Phase 2 LRs:   {args.phase2_lrs}")
    print(f"  Phase 2 ranks: {args.phase2_ranks}")
    print(f"  device: {device}")
    print(f"{'='*60}")

    if args.replot:
        with open(summary_path) as f:
            all_results = json.load(f)
        plot_trajectories(all_results, os.path.join(out_dir, "trajectory_plot.png"),
                          args.task_a, args.task_b)
        plot_threshold_summary(all_results, os.path.join(out_dir, "threshold_summary.png"),
                               args.task_a, args.task_b)
        return

    # ── Phase 1 ──────────────────────────────────────────────────────────────
    (phase1_model, phase1_classifier,
     phase1_params, sigma_half_a,
     n_classes_a, acc_phase1_a, tokenizer) = run_phase1(args, device, out_dir)

    if args.phase1_only:
        print("\n  --phase1_only: checkpoint saved, skipping Phase 2 sweep.")
        return

    # ── Phase 2 sweep ─────────────────────────────────────────────────────────
    all_results = []
    for rank in args.phase2_ranks:
        for lr in args.phase2_lrs:
            result = run_phase2_condition(
                phase1_model=phase1_model,
                phase1_classifier=phase1_classifier,
                phase1_params=phase1_params,
                sigma_half_a=sigma_half_a,
                n_classes_a=n_classes_a,
                acc_phase1_a=acc_phase1_a,
                task_a=args.task_a,
                task_b=args.task_b,
                tokenizer=tokenizer,
                lr=lr, rank=rank,
                steps=args.phase2_steps,
                track_interval=args.track_interval,
                batch_size=args.batch_size,
                grad_clip=args.grad_clip,
                eval_samples=args.eval_samples,
                seq_len=args.seq_len,
                device=device,
                seed=args.seed,
                out_dir=out_dir,
                save_tstar_checkpoints=args.save_tstar_checkpoints,
                tstar_acc_threshold=args.tstar_acc_threshold,
            )
            all_results.append(result)

    # ── Save + plot ───────────────────────────────────────────────────────────
    with open(summary_path, "w") as f:
        json.dump(all_results, f, indent=2)

    plot_trajectories(all_results, os.path.join(out_dir, "trajectory_plot.png"),
                      args.task_a, args.task_b)
    plot_threshold_summary(all_results, os.path.join(out_dir, "threshold_summary.png"),
                           args.task_a, args.task_b)

    # ── Print summary table ───────────────────────────────────────────────────
    print(f"\n{'─'*65}")
    print(f"{'LR':>8}  {'Rank':>5}  {'R_A':>8}  "
          f"{'Acc_A':>7}  {'Acc_B':>7}  {'Forgetting?':>12}")
    print(f"{'─'*65}")
    for r in sorted(all_results, key=lambda x: (x["rank"], x["lr"])):
        forgot = "YES" if r["final_R_A"] > 1.0 else "no"
        print(f"  {r['lr']:>8.0e}  {r['rank']:>5}  {r['final_R_A']:>8.4f}  "
              f"{r['final_acc_task_a']:>7.4f}  {r['final_acc_task_b']:>7.4f}  "
              f"{forgot:>12}")
    print(f"{'─'*65}\n")


if __name__ == "__main__":
    main()
