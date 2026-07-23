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
import matplotlib.gridspec as gridspec

sys.path.insert(0, os.path.dirname(__file__))
from certified_density_experiment import (
    load_data, estimate_density, half_max_sigma,
    compute_nll_loss, save_weights, restore_weights, perturb_weights,
)
from lora_density_experiment import (
    _default_target_modules, _load_model_auto,
    _get_target_names_auto, _get_hidden_size, _get_tokenizer,
    add_lora, load_sst2, _compute_grad_norm,
)


# ── Args ──────────────────────────────────────────────────────────────────────
def get_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model",  default="gpt2")
    p.add_argument("--task",   default="sst2",
                   choices=["sst2", "wikitext2", "agnews", "mnli", "dbpedia", "yahoo"],
                   help="Fine-tuning task. 'wikitext2' uses CLM on WikiText-2 train "
                        "split — same distribution as eval, ruling out task-orthogonality. "
                        "'agnews' uses AG News 4-class topic classification.")
    p.add_argument("--ranks",  nargs="+", type=int, default=[1, 2, 4, 8, 16, 32],
                   help="LoRA ranks to sweep")
    p.add_argument("--lora_alpha",   type=int,   default=None,
                   help="LoRA alpha. Default: None = set equal to rank (scale=1). "
                        "Set explicitly to fix alpha across all ranks (conflates "
                        "rank with effective LR — avoid for rank ablation).")
    p.add_argument("--lr",           nargs="+", type=float, default=[1e-4],
                   help="Learning rate(s). Pass multiple values for an LR sweep "
                        "(e.g. --lr 1e-5 5e-5 1e-4 2e-4 5e-4 1e-3).")
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
    p.add_argument("--force",          action="store_true",
                   help="Re-run all experiments even if output files already exist.")
    p.add_argument("--replot",         action="store_true",
                   help="Skip fine-tuning; reload existing JSONs and regenerate plots.")
    p.add_argument("--quantize",       action="store_true",
                   help="After each fine-tuning run, simulate INT8 and INT4 absmax "
                        "quantization and record WikiText-2 PPL degradation.")
    p.add_argument("--classifier_lr", type=float, default=None,
                   help="Learning rate for the classification head (sst2/agnews). "
                        "Default: max(lr, 1e-3). Set explicitly to override. "
                        "Keeps the classifier convergent at all LoRA LR settings.")
    p.add_argument("--task_a_eval_slack", type=float, default=1e-3,
                   help="eval_slack for the Task A classification density (σ½_A). "
                        "Default 1e-3 — 10× wider than WikiText-2 slack because "
                        "classification cross-entropy has higher natural variance. "
                        "σ½_A is the largest σ where P[XEnt(θ+ε) ≤ XEnt(θ_A)+slack] ≥ 0.5.")
    p.add_argument("--task_a_metric", default="nll", choices=["nll", "acc"],
                   help="Certification metric for σ½_A. 'nll' uses cross-entropy slack "
                        "(original). 'acc' uses accuracy drop: pass iff acc(θ+ε) ≥ "
                        "acc(θ_A) − task_a_acc_delta. Accuracy-based is better calibrated "
                        "for binary tasks (SST-2) and high-confidence models.")
    p.add_argument("--task_a_acc_delta", type=float, default=0.05,
                   help="Accuracy drop tolerance for accuracy-based σ½_A certification "
                        "(--task_a_metric acc). Default 0.05 = 5 percentage points.")
    p.add_argument("--skip_phase1_finetune", action="store_true",
                   help="Skip Phase 1 fine-tuning and load θ_A from a previously saved "
                        "phase1_model.pt checkpoint. Useful for re-running the density "
                        "sweep with a different --task_a_metric without re-training.")
    p.add_argument("--task_a", default=None,
                   choices=["sst2", "agnews", "mnli", "dbpedia", "yahoo"],
                   help="Sequential forgetting mode. Fine-tune on task_a first (Phase 1), "
                        "then fine-tune on --task (task B, Phase 2) starting from task_a "
                        "weights. Phase 2 trajectory tracks task_a accuracy at each "
                        "checkpoint to measure catastrophic forgetting.")
    p.add_argument("--train_steps_a", type=int, default=None,
                   help="Training steps for Phase 1 (task_a). "
                        "Default: same as train_steps[0].")
    p.add_argument("--lr_a", type=float, default=None,
                   help="Learning rate for Phase 1 (task_a) fine-tuning. "
                        "Default: lr[0]. Use this to decouple Phase 1 LR from "
                        "the Phase 2 LR sweep (e.g. --lr_a 1e-4 --lr 1e-5 1e-4 2e-4).")
    p.add_argument("--rank_a", type=int, default=None,
                   help="LoRA rank for Phase 1 (task_a) fine-tuning. "
                        "Default: ranks[0]. Decouples Phase 1 rank from Phase 2 sweep.")
    p.add_argument("--output_dir",     default="./results/basin_widening")
    p.add_argument("--track_interval", type=int, default=0,
                   help="If >0, record σ½, norms, and ppl every N steps during "
                        "training (uses a quick N=50 density sweep). "
                        "Saved as rank{r}_steps{n}_trajectory.json.")
    p.add_argument("--landscape_grid", type=int, default=0,
                   help="If >0, generate a 2D NLL landscape plot after each "
                        "fine-tuning run. Grid resolution (e.g. 20 = 20×20).")
    p.add_argument("--landscape_sigma_mult", type=float, default=3.0,
                   help="Sweep ±landscape_sigma_mult × σ½_pre in each direction "
                        "for the landscape plot.")
    return p.parse_args()


# ── Norm helpers ─────────────────────────────────────────────────────────────
def _compute_norms(merged_model, pretrained_snapshot):
    """
    Returns (per_param_norm, max_spectral_norm) for the weight delta ΔW.

    per_param_norm   — RMS per-weight Frobenius norm: sqrt(Σ‖ΔW‖²_F / d_total)
                       same units as σ, directly comparable to σ½.

    max_spectral_norm — largest singular value of ΔW across all 2-D weight
                        matrices that were modified by LoRA.  Measures the
                        maximum amplification any input direction can receive
                        through the update; a tighter behavioral indicator than
                        the Frobenius norm because it is not diluted by the
                        many near-zero singular values that LoRA leaves untouched.

    Name normalisation: during trajectory checkpoints the model is accessed via
    peft_model.base_model.model after merge_adapter(), which keeps PEFT's
    .base_layer wrapper in place.  This adds a ".base_layer" segment to target-
    module parameter names (e.g. "attn.c_attn.base_layer.weight" instead of
    "attn.c_attn.weight").  We strip that segment so the lookup matches the
    snapshot captured from the original unwrapped model.
    """
    total_sq, total_n = 0.0, 0
    max_spec = 0.0
    for name, param in merged_model.named_parameters():
        # Normalise name: strip the PEFT base_layer wrapper added during checkpoints
        lookup = name.replace(".base_layer.", ".")
        snap_key = name if name in pretrained_snapshot else lookup
        if snap_key not in pretrained_snapshot:
            continue
        delta = param.data.float() - pretrained_snapshot[snap_key]
        total_sq += delta.norm().item() ** 2
        total_n  += delta.numel()
        if delta.dim() >= 2 and delta.norm().item() > 1e-12:
            sv = torch.linalg.svdvals(delta.reshape(delta.shape[0], -1))
            max_spec = max(max_spec, sv[0].item())
    per_param_norm = float((total_sq / max(total_n, 1)) ** 0.5)
    return per_param_norm, max_spec


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


# ── AG News data loader ───────────────────────────────────────────────────────
def load_agnews(tokenizer, batch_size, max_len=128, split="train", seed=42):
    """Load AG News for 4-class topic classification (World/Sports/Business/Sci-Tech)."""
    from datasets import load_dataset
    ds = load_dataset("ag_news", split=split)
    ds = ds.shuffle(seed=seed)

    input_ids_list, label_list = [], []
    for ex in ds:
        enc = tokenizer(ex["text"], truncation=True, max_length=max_len,
                        padding="max_length", return_tensors="pt")
        input_ids_list.append(enc["input_ids"].squeeze(0))
        label_list.append(ex["label"])

    input_ids = torch.stack(input_ids_list)
    labels_t  = torch.tensor(label_list, dtype=torch.long)
    dataset   = torch.utils.data.TensorDataset(input_ids, labels_t)
    return torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=True, drop_last=True
    )


def _tok_and_stack(tokenizer, texts, labels, max_len, batch_size, shuffle=True):
    """Shared tokenize-and-DataLoader helper for all classification loaders."""
    input_ids_list = []
    for text in texts:
        enc = tokenizer(text, truncation=True, max_length=max_len,
                        padding="max_length", return_tensors="pt")
        input_ids_list.append(enc["input_ids"].squeeze(0))
    input_ids = torch.stack(input_ids_list)
    labels_t  = torch.tensor(labels, dtype=torch.long)
    dataset   = torch.utils.data.TensorDataset(input_ids, labels_t)
    return torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=shuffle, drop_last=True
    )


def load_mnli(tokenizer, batch_size, max_len=128, split="train", seed=42):
    """MNLI: 3-class (entailment/neutral/contradiction). Premise ++ hypothesis."""
    from datasets import load_dataset
    eval_split = "validation_matched" if split != "train" else "train"
    ds = load_dataset("glue", "mnli", split=eval_split)
    ds = ds.shuffle(seed=seed)
    texts  = [f"{ex['premise']} {ex['hypothesis']}" for ex in ds]
    labels = [ex["label"] for ex in ds]
    return _tok_and_stack(tokenizer, texts, labels, max_len, batch_size)


def load_dbpedia(tokenizer, batch_size, max_len=128, split="train", seed=42):
    """DBPedia: 14-class ontology classification. Title ++ content."""
    from datasets import load_dataset
    ds = load_dataset("dbpedia_14", split=split)
    ds = ds.shuffle(seed=seed)
    texts  = [f"{ex['title']} {ex['content']}" for ex in ds]
    labels = [ex["label"] for ex in ds]
    return _tok_and_stack(tokenizer, texts, labels, max_len, batch_size)


def load_yahoo(tokenizer, batch_size, max_len=128, split="train", seed=42):
    """Yahoo Answers Topics: 10-class Q&A topic classification."""
    from datasets import load_dataset
    ds = load_dataset("yahoo_answers_topics", split=split)
    ds = ds.shuffle(seed=seed)
    texts  = [f"{ex['question_title']} {ex['question_content']}" for ex in ds]
    labels = [ex["topic"] for ex in ds]
    return _tok_and_stack(tokenizer, texts, labels, max_len, batch_size)


# ── Task registry ─────────────────────────────────────────────────────────────
# Maps task name -> (num_classes, loader_fn, eval_split)
# loader_fn(tokenizer, batch_size, max_len, split, seed) -> DataLoader[(ids, labels)]
# CLM tasks (wikitext2) are handled separately — they don't have classifiers.
_CLF_TASKS = {
    "sst2":    (2,  load_sst2,    "validation"),
    "agnews":  (4,  load_agnews,  "test"),
    "mnli":    (3,  load_mnli,    "validation_matched"),
    "dbpedia": (14, load_dbpedia, "test"),
    "yahoo":   (10, load_yahoo,   "test"),
}


# ── Task accuracy evaluation ──────────────────────────────────────────────────
def _evaluate_task_accuracy(model, classifier, tokenizer, task, device):
    """Evaluate classification accuracy on the task validation set.
    Accepts either a peft_model or a bare base model (e.g. peft_model.base_model.model
    during trajectory checkpoints after merge_adapter()).
    Must be called before merge_and_unload() — classifier is discarded after merge."""
    if classifier is None or task not in _CLF_TASKS:
        return float("nan")

    _, loader_fn, eval_split = _CLF_TASKS[task]
    val_loader = loader_fn(tokenizer, batch_size=64, split=eval_split)

    model.eval()
    classifier.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for input_ids, labels in val_loader:
            input_ids, labels = input_ids.to(device), labels.to(device)
            out    = model(input_ids, output_hidden_states=True)
            hidden = out.hidden_states[-1][:, -1, :].float()
            preds  = classifier(hidden).argmax(dim=-1)
            correct += (preds == labels).sum().item()
            total   += labels.size(0)
    return correct / total if total > 0 else float("nan")


# ── Quantization simulation ───────────────────────────────────────────────────
def _simulate_quantization(model, bits):
    """Return a deepcopy of model with all weight matrices absmax-quantized to `bits` bits."""
    import copy
    q_model = copy.deepcopy(model)
    with torch.no_grad():
        for param in q_model.parameters():
            if param.data.dim() >= 2:
                orig_dtype = param.data.dtype
                fp = param.data.float()
                max_val = fp.abs().max()
                if max_val > 0:
                    scale = (2 ** (bits - 1) - 1) / max_val
                    param.data = ((fp * scale).round() / scale).to(orig_dtype)
    return q_model


# ── Fine-tuning ───────────────────────────────────────────────────────────────
def finetune(base_model, tokenizer, task, lr, train_steps, batch_size,
             rank, alpha, device, target_modules, finetune_fp32=False,
             grad_clip=1.0, seed=42,
             track_interval=0, track_inp=None, track_tgt=None,
             track_names=None, track_eval_slack=1e-4, track_sh_pre=0.0,
             classifier_lr=None,
             seq_task_a=None, seq_classifier_a=None):
    """
    Fine-tune with LoRA. Task selects the training objective:
      sst2      — classification head on SST-2 (cross-entropy, 2 classes)
      wikitext2 — causal LM on WikiText-2 train (no classifier head)

    alpha=None resolves to rank (scale=1), keeping effective LR constant
    across the rank ablation so only subspace dimensionality varies.

    track_interval: if >0, pause every N steps to run a quick σ½ sweep
                    (N=50 perturbations, 8 σ values) and record the trajectory.
                    Requires track_inp, track_tgt, track_names to be provided.

    Returns (merged_model, per_param_norm, spectral_norm, resolved_alpha,
             trajectory, loss_history, task_val_acc, classifier).
    trajectory is a list of dicts [{step, sigma_half, per_param_norm,
    spectral_norm, norm_over_sh_pre, wikitext2_ppl, task_val_acc,
    task_a_val_acc}], empty when track_interval=0.
    task_a_val_acc is only populated when seq_task_a and seq_classifier_a are set.
    loss_history is a list of dicts [{step, loss}] recorded every 10 steps,
    always-on regardless of track_interval.
    classifier is the trained linear head (nn.Linear) for classification tasks,
    or None for CLM tasks. Returned so sequential mode can evaluate task A
    forgetting during task B training.
    """
    alpha = rank if alpha is None else alpha
    torch.manual_seed(seed)
    print(f"\n  Fine-tuning | task={task}  rank={rank}  alpha={alpha}  "
          f"lr={lr:.1e}  steps={train_steps}  fp32={finetune_fp32}", flush=True)

    # Only snapshot LoRA target params (not all params).
    # For 8B models snapshotting all params in fp32 = 32 GB VRAM; target-only = ~5 GB.
    pretrained_snapshot = {
        n: p.data.clone().float()
        for n, p in base_model.named_parameters()
        if any(t in n for t in (target_modules or []))
    }

    model_copy = copy.deepcopy(base_model)
    if finetune_fp32:
        model_copy = model_copy.float()
    peft_model = add_lora(model_copy, rank, alpha, target_modules)

    if task in _CLF_TASKS:
        num_classes, loader_fn, _ = _CLF_TASKS[task]
        hidden_size = _get_hidden_size(base_model)
        classifier  = nn.Linear(hidden_size, num_classes).to(device)
        params      = list(peft_model.parameters()) + list(classifier.parameters())
        loader      = loader_fn(tokenizer, batch_size)
        loss_fn     = nn.CrossEntropyLoss()
    else:  # wikitext2 CLM
        classifier  = None
        params      = list(peft_model.parameters())
        loader      = load_wikitext2_clm(tokenizer, batch_size)
        loss_fn     = None

    if classifier is not None:
        clf_lr   = classifier_lr if classifier_lr is not None else max(lr, 1e-3)
        lora_params = [p for p in peft_model.parameters() if p.requires_grad]
        clf_params  = list(classifier.parameters())
        opt = torch.optim.AdamW([
            {"params": lora_params, "lr": lr},
            {"params": clf_params,  "lr": clf_lr},
        ])
        print(f"  optimizer: LoRA lr={lr:.1e}  classifier lr={clf_lr:.1e}", flush=True)
    else:
        opt = torch.optim.AdamW([p for p in params if p.requires_grad], lr=lr)

    peft_model.train()
    if classifier is not None:
        classifier.train()

    # Quick σ values used for in-training checkpoints (8 points, coarser grid)
    _TRACK_SIGMAS = [1e-4, 5e-4, 1e-3, 2e-3, 5e-3, 1e-2, 5e-2, 0.1]
    _do_track = (track_interval > 0
                 and track_inp is not None
                 and track_tgt is not None
                 and track_names is not None)
    trajectory   = []
    loss_history = []  # always-on: step, loss every 10 steps

    step = 0
    while step < train_steps:
        for batch in loader:
            if step >= train_steps:
                break

            if task in _CLF_TASKS:
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

            if step % 10 == 0:
                loss_history.append({"step": step, "loss": float(loss.item())})
            if step % 100 == 0:
                print(f"    step {step}/{train_steps}  loss={loss.item():.4f}",
                      flush=True)

            # ── Per-step σ½ checkpoint ────────────────────────────────────────
            # Uses merge_adapter/unmerge_adapter (in-place) instead of
            # copy.deepcopy + merge_and_unload, saving ~16-32 GB VRAM on large models.
            if _do_track and step % track_interval == 0:
                peft_model.eval()
                with torch.no_grad():
                    peft_model.merge_adapter()          # merge LoRA into base weights in-place
                    # Access the underlying base model for measurement
                    base = peft_model.base_model.model
                    pp_norm, spec_norm = _compute_norms(base, pretrained_snapshot)
                    base_names = {n for n, _ in base.named_parameters()
                                  if n in track_names}
                    grad_norm_ckpt = _compute_grad_norm(base, base_names,
                                                        track_inp, track_tgt)
                    quick_results = []
                    for sig in _TRACK_SIGMAS:
                        r = estimate_density(base, base_names, track_inp, track_tgt,
                                             sig, N=50, eval_slack=track_eval_slack,
                                             tag=f"track/step{step}")
                        quick_results.append(r)
                    sh_ckpt = half_max_sigma(quick_results)

                    # WikiText-2 PPL at this checkpoint (cheap: single forward pass)
                    wikitext2_nll_ckpt = compute_nll_loss(base, track_inp, track_tgt)
                    wikitext2_ppl_ckpt = float(np.exp(wikitext2_nll_ckpt))

                    # Task B accuracy at this checkpoint
                    task_acc_ckpt = _evaluate_task_accuracy(
                        base, classifier, tokenizer, task, device)

                    # Task A forgetting (sequential mode only)
                    task_a_acc_ckpt = float("nan")
                    if seq_task_a is not None and seq_classifier_a is not None:
                        task_a_acc_ckpt = _evaluate_task_accuracy(
                            base, seq_classifier_a, tokenizer, seq_task_a, device)

                    peft_model.unmerge_adapter()        # restore original weights
                if device.type == "cuda":
                    torch.cuda.empty_cache()
                trajectory.append({
                    "step":             step,
                    "sigma_half":       sh_ckpt,
                    "per_param_norm":   pp_norm,
                    "spectral_norm":    spec_norm,
                    "grad_norm":        grad_norm_ckpt,
                    "norm_over_sh_pre": (pp_norm / track_sh_pre
                                         if track_sh_pre > 0 else float("nan")),
                    "norm_over_sh_ft":  (pp_norm / sh_ckpt
                                         if sh_ckpt > 0 else float("nan")),
                    "wikitext2_ppl":    wikitext2_ppl_ckpt,
                    "task_val_acc":     task_acc_ckpt,
                    "task_a_val_acc":   task_a_acc_ckpt,
                })
                seq_str = (f"  task_a_acc={task_a_acc_ckpt:.4f}"
                           if seq_task_a else "")
                print(f"    [track step={step}]  σ½={sh_ckpt:.6f}  "
                      f"norm/σ½_pre={pp_norm/track_sh_pre:.3f}  "
                      f"wikitext2_ppl={wikitext2_ppl_ckpt:.4f}  "
                      f"task_acc={task_acc_ckpt:.4f}{seq_str}", flush=True)
                peft_model.train()

    task_val_acc = _evaluate_task_accuracy(peft_model, classifier, tokenizer, task, device)
    if not np.isnan(task_val_acc):
        print(f"  {task} val accuracy = {task_val_acc:.4f}", flush=True)

    merged = peft_model.merge_and_unload().eval()
    per_param_norm, spectral_norm = _compute_norms(merged, pretrained_snapshot)
    print(f"  per_param_norm={per_param_norm:.6f}  "
          f"spectral_norm={spectral_norm:.6f}", flush=True)

    return merged, per_param_norm, spectral_norm, alpha, trajectory, loss_history, task_val_acc, classifier


# ── Density sweep ─────────────────────────────────────────────────────────────
def run_density(model, names, inp, tgt, sigmas, n_perturb, eval_slack, tag):
    results = []
    for sigma in sigmas:
        r = estimate_density(model, names, inp, tgt, sigma,
                             N=n_perturb, eval_slack=eval_slack, tag=tag)
        results.append(r)
    return results


# ── Task-A classification density (for sequential forgetting experiment) ──────
def _get_task_clf_batch(tokenizer, task, max_samples=128, device=None):
    """
    Return a fixed (input_ids, labels) batch for classification density.
    Using a fixed batch makes σ½_A a deterministic function of θ_A, so
    the threshold is reproducible across runs.
    """
    _, loader_fn, eval_split = _CLF_TASKS[task]
    loader = loader_fn(tokenizer, batch_size=max_samples, split=eval_split, seed=42)
    inp_ids, labels = next(iter(loader))
    if device is not None:
        inp_ids, labels = inp_ids.to(device), labels.to(device)
    return inp_ids, labels


def compute_clf_nll(model, classifier, inp, labels):
    """
    Mean cross-entropy (nats) of the frozen classifier head on a fixed batch.
    Used instead of WikiText-2 NLL so that σ½_A directly certificates
    Task A classification performance, not an unrelated language-modeling loss.
    """
    model.eval()
    classifier.eval()
    with torch.no_grad():
        out    = model(inp, output_hidden_states=True)
        hidden = out.hidden_states[-1][:, -1, :].float()
        loss   = nn.functional.cross_entropy(classifier(hidden), labels)
    return float(loss.item())


def estimate_density_clf(model, classifier, names, inp, labels,
                         sigma, N, eval_slack, tag=""):
    """
    C_clf(θ, σ) = P[XEnt(θ+ε) ≤ XEnt(θ) + eval_slack]   ε ~ N(0, σ²I).

    Mirrors estimate_density() but uses classification cross-entropy so that
    σ½_A is the certified radius for Task A performance, not WikiText-2 NLL.
    The two quantities are independent: a model can have a wide WikiText-2 ball
    but a narrow Task-A ball (or vice versa), so using the wrong one would make
    the forgetting-budget hypothesis untestable.
    """
    base_nll  = compute_clf_nll(model, classifier, inp, labels)
    threshold = base_nll + eval_slack
    snapshot  = save_weights(model, names)
    n_better  = 0
    for i in range(N):
        perturb_weights(model, seed=i, sigma=sigma, names=names)
        if compute_clf_nll(model, classifier, inp, labels) <= threshold:
            n_better += 1
        restore_weights(model, snapshot, names)

    # Wilson 95% CI (no external dependency)
    z       = 1.96
    p       = n_better / max(N, 1)
    denom   = 1 + z**2 / N
    center  = (p + z**2 / (2 * N)) / denom
    half    = (z / denom) * (p * (1 - p) / N + z**2 / (4 * N**2)) ** 0.5
    return {
        "sigma":      sigma,
        "density":    p,
        "ci_lower":   max(0.0, center - half),
        "ci_upper":   min(1.0, center + half),
        "n_better":   n_better,
        "N":          N,
        "base_nll":   base_nll,
        "eval_slack": eval_slack,
        "tag":        tag,
    }


def run_density_clf(model, classifier, names, inp, labels,
                    sigmas, n_perturb, eval_slack, tag):
    """Full sigma sweep using Task A classification NLL."""
    results = []
    for sigma in sigmas:
        r = estimate_density_clf(model, classifier, names, inp, labels,
                                 sigma, n_perturb, eval_slack, tag=tag)
        results.append(r)
        print(f"  [{tag}] σ={sigma:.5f}  "
              f"density={r['density']:.3f}  base_nll={r['base_nll']:.4f}",
              flush=True)
    return results


def compute_clf_acc(model, classifier, inp, labels):
    """Accuracy of frozen classifier head on a fixed batch."""
    model.eval()
    classifier.eval()
    with torch.no_grad():
        out    = model(inp, output_hidden_states=True)
        hidden = out.hidden_states[-1][:, -1, :].float()
        preds  = classifier(hidden).argmax(dim=-1)
    return (preds == labels).float().mean().item()


def estimate_density_clf_acc(model, classifier, names, inp, labels,
                              sigma, N, acc_delta, tag=""):
    """
    C_clf_acc(θ, σ) = P[acc(θ+ε) ≥ acc(θ) − acc_delta]   ε ~ N(0, σ²I).

    Accuracy-based certification. Directly matches the forgetting metric and
    avoids NLL overestimation for binary/low-confidence tasks.
    """
    base_acc  = compute_clf_acc(model, classifier, inp, labels)
    threshold = base_acc - acc_delta
    snapshot  = save_weights(model, names)
    n_better  = 0
    for i in range(N):
        perturb_weights(model, seed=i, sigma=sigma, names=names)
        if compute_clf_acc(model, classifier, inp, labels) >= threshold:
            n_better += 1
        restore_weights(model, snapshot, names)
    z       = 1.96
    p       = n_better / max(N, 1)
    denom   = 1 + z**2 / N
    center  = (p + z**2 / (2 * N)) / denom
    half    = (z / denom) * (p * (1 - p) / N + z**2 / (4 * N**2)) ** 0.5
    return {
        "sigma":     sigma,
        "density":   p,
        "ci_lower":  max(0.0, center - half),
        "ci_upper":  min(1.0, center + half),
        "n_better":  n_better,
        "N":         N,
        "base_acc":  base_acc,
        "acc_delta": acc_delta,
        "tag":       tag,
    }


def run_density_clf_acc(model, classifier, names, inp, labels,
                        sigmas, n_perturb, acc_delta, tag):
    """Full sigma sweep using Task A classification accuracy."""
    results = []
    for sigma in sigmas:
        r = estimate_density_clf_acc(model, classifier, names, inp, labels,
                                     sigma, n_perturb, acc_delta, tag=tag)
        results.append(r)
        print(f"  [{tag}] σ={sigma:.5f}  "
              f"density={r['density']:.3f}  base_acc={r['base_acc']:.4f}",
              flush=True)
    return results


# ── Summary JSON ──────────────────────────────────────────────────────────────
def build_summary(records):
    """records: list of dicts with keys rank, steps, pre, ft, norm, spectral_norm, grad_norm."""
    out = []
    for rec in records:
        sh_pre = half_max_sigma(rec["pre"])
        sh_ft  = half_max_sigma(rec["ft"])
        ratio  = sh_ft / sh_pre if sh_pre > 0 else float("nan")
        # "base_ppl" = WikiText-2 NLL key; "base_nll" = Task-A clf NLL key
        ppl_pre = rec["pre"][0].get("base_ppl", rec["pre"][0].get("base_nll", float("nan")))
        ppl_ft  = rec["ft"][0].get("base_ppl",  rec["ft"][0].get("base_nll",  float("nan")))
        spec    = rec.get("spectral_norm", float("nan"))
        grad    = rec.get("grad_norm", float("nan"))
        # grad_x_sh: expected NLL change (nats) when perturbing by one σ½.
        # Dimensionally consistent (nats) and directly comparable to eval_slack.
        # If grad_x_sh >> eval_slack the slope at θ_ft is steep enough that a
        # σ½-sized perturbation will likely exceed the NLL threshold.
        _g_finite = not np.isnan(grad) and sh_pre > 0
        qr = rec.get("quant_results", {})
        out.append({
            "rank":              rec["rank"],
            "steps":             rec["steps"],
            "lr":                rec.get("lr", float("nan")),
            "sigma_half_pre":    sh_pre,
            "sigma_half_ft":     sh_ft,
            "widening_ratio":    ratio,
            "norm":              rec["norm"],
            "norm_over_sh":      rec["norm"] / sh_pre if sh_pre > 0 else float("nan"),
            "spectral_norm":     spec,
            "spec_over_sh":      spec / sh_pre if sh_pre > 0 else float("nan"),
            "grad_norm":         grad,
            "grad_x_sh":         grad * sh_pre if _g_finite else float("nan"),
            "ppl_pre":           ppl_pre,       # base_nll (nats), NOT actual PPL
            "ppl_ft":            ppl_ft,        # base_nll (nats), NOT actual PPL
            "ppl_ratio":         ppl_ft / ppl_pre if ppl_pre > 0 else float("nan"),  # NLL ratio
            "ppl_ratio_actual":  float(np.exp(ppl_ft - ppl_pre)) if ppl_pre > 0 else float("nan"),  # true PPL_ft/PPL_pre
            "task_val_acc":      rec.get("task_val_acc", float("nan")),
            "ppl_int8":          qr.get("ppl_int8",        float("nan")),
            "ppl_ratio_int8":    qr.get("ppl_ratio_int8",  float("nan")),
            "ppl_int4":          qr.get("ppl_int4",        float("nan")),
            "ppl_ratio_int4":    qr.get("ppl_ratio_int4",  float("nan")),
        })
    return out


# ── Weight landscape ──────────────────────────────────────────────────────────
def plot_weight_landscape(pretrained_snapshot, finetuned_model, names,
                          inp, tgt, sh_pre, n_grid=20, sigma_mult=3.0,
                          sh_ft=None, eval_slack=1e-4, save_path=None):
    """
    Three-panel loss-landscape figure.

    Panel 1 — 2D contour (top-down, fine scale ±sigma_mult×σ½):
      Shows the basin shape around θ_ft. Two circles mark the certified basin
      width before (σ½_pre, blue) and after (σ½_ft, orange) fine-tuning —
      widening is directly visible as the larger orange circle.

    Panel 2 — 3D surface (same fine scale):
      Shows basin depth and asymmetry.

    Panel 3 — 1D trajectory θ_pre → θ_ft → beyond (two scales):
      Main plot: full ‖ΔW‖ scale, shows where the model started and landed.
      Inset: zoomed ±5×σ½ around θ_ft, shows the certified region (green) and
      the NLL rising back above the threshold = certification collapse.

    d1 = (θ_ft − θ_pre) / ‖θ_ft − θ_pre‖  (LoRA update direction)
    d2 = random unit vector ⊥ d1 (Gram-Schmidt)
    θ_ft = origin (0, 0) throughout.
    """
    # Flatten ΔW over target parameters in a consistent order
    param_list = [(n, p) for n, p in finetuned_model.named_parameters()
                  if n in names]
    if not param_list:
        print("  plot_weight_landscape: no target parameters — skipping")
        return

    parts = []
    for name, param in param_list:
        if name in pretrained_snapshot:
            parts.append((param.data.float() - pretrained_snapshot[name]).flatten())
        else:
            parts.append(torch.zeros(param.numel()))
    delta_vec  = torch.cat(parts)          # shape: (total_d,)
    norm_delta = delta_vec.norm()

    if norm_delta < 1e-12:
        print("  plot_weight_landscape: ΔW ≈ 0 — model unchanged, skipping")
        return

    d1 = delta_vec / norm_delta            # unit vector: LoRA update direction
    d2 = torch.randn_like(d1)
    d2 = d2 - (d2 @ d1) * d1              # Gram-Schmidt orthogonalise
    d2 = d2 / d2.norm()                    # unit vector: random orthogonal

    # ── Fine 2-D grid centred at θ_ft ────────────────────────────────────────
    nd     = norm_delta.item()
    sweep  = sigma_mult * sh_pre
    alphas = np.linspace(-sweep, sweep, n_grid)
    betas  = np.linspace(-sweep, sweep, n_grid)
    nll_grid = np.full((n_grid, n_grid), np.nan)

    # ── Coarse 1-D trajectory: θ_pre → θ_ft → beyond (full ‖ΔW‖ scale) ──────
    n_traj       = max(n_grid * 2, 40)
    traj_alphas  = np.linspace(-nd * 1.15, nd * 0.55, n_traj)
    nll_traj     = np.full(n_traj, np.nan)

    # ── Fine 1-D trajectory: zoomed around θ_ft (±5×σ½, for inset) ──────────
    n_fine         = 30
    fine_alphas    = np.linspace(-5 * sh_pre, 5 * sh_pre, n_fine)
    nll_fine       = np.full(n_fine, np.nan)

    snapshot = save_weights(finetuned_model, names)

    def _apply_and_measure(alpha_val, beta_val=0.0):
        noise_vec = alpha_val * d1 + beta_val * d2
        offset = 0
        with torch.no_grad():
            for _, param in param_list:
                d_    = param.numel()
                chunk = noise_vec[offset:offset + d_].reshape(param.shape)
                param.data.copy_((param.data.float() + chunk.to(param.device))
                                  .to(param.dtype))
                offset += d_
        nll = compute_nll_loss(finetuned_model, inp, tgt)
        restore_weights(finetuned_model, snapshot, names)
        return nll

    print(f"  Landscape: {n_grid}×{n_grid} fine 2D grid  range=±{sweep:.5f} ...",
          flush=True)
    for i, alpha in enumerate(alphas):
        for j, beta in enumerate(betas):
            nll_grid[j, i] = _apply_and_measure(alpha, beta)
        if (i + 1) % max(1, n_grid // 4) == 0:
            print(f"    {i+1}/{n_grid} cols done", flush=True)

    print(f"  Landscape: coarse trajectory  {n_traj} pts  "
          f"θ_pre({-nd:.2e})→θ_ft(0)→beyond({nd*0.55:.2e}) ...", flush=True)
    for i, alpha in enumerate(traj_alphas):
        nll_traj[i] = _apply_and_measure(alpha)

    print(f"  Landscape: fine trajectory  {n_fine} pts  "
          f"±{5*sh_pre:.2e} around θ_ft ...", flush=True)
    for i, alpha in enumerate(fine_alphas):
        nll_fine[i] = _apply_and_measure(alpha)

    # ── Save grid for later replotting ────────────────────────────────────────
    if save_path:
        grid_json = save_path.replace(".png", "_grid.json")
        with open(grid_json, "w") as _f:
            json.dump({
                "nll_grid":    nll_grid.tolist(),
                "alphas":      alphas.tolist(),
                "betas":       betas.tolist(),
                "sweep":       sweep,
                "sigma_mult":  sigma_mult,
                "sh_pre":      sh_pre,
                "sh_ft":       sh_ft,
                "eval_slack":  eval_slack,
                "norm_delta":  nd,
                "traj_alphas": traj_alphas.tolist(),
                "nll_traj":    nll_traj.tolist(),
                "fine_alphas": fine_alphas.tolist(),
                "nll_fine":    nll_fine.tolist(),
            }, _f)

    # ── Smooth fine grid ──────────────────────────────────────────────────────
    import mpl_toolkits.mplot3d  # registers projection='3d'
    try:
        from scipy.ndimage import gaussian_filter as _gf
        nll_smooth = _gf(nll_grid, sigma=0.9)
    except ImportError:
        nll_smooth = nll_grid
    nll_finite = np.where(np.isfinite(nll_smooth), nll_smooth,
                          float(np.nanmax(nll_smooth)))
    A, B = np.meshgrid(alphas, betas)
    mid  = n_grid // 2

    # ΔNLL = NLL − NLL_ft  (θ_ft sits at 0 on the y-axis)
    ft_i_c = int(np.argmin(np.abs(traj_alphas)))
    ft_i_f = int(np.argmin(np.abs(fine_alphas)))
    nll_at_ft   = float(nll_traj[ft_i_c]) if np.isfinite(nll_traj[ft_i_c]) else float(nll_fine[ft_i_f])
    dnll_traj   = nll_traj  - nll_at_ft
    dnll_fine   = nll_fine  - nll_at_ft

    # ── Figure layout ─────────────────────────────────────────────────────────
    fig = plt.figure(figsize=(17, 5.8))
    gs  = gridspec.GridSpec(1, 3, figure=fig, wspace=0.42,
                            left=0.05, right=0.97, top=0.88, bottom=0.12)

    # ── Panel 1: 2D contour with σ½ circles ──────────────────────────────────
    ax2d = fig.add_subplot(gs[0])
    vmin, vmax = float(np.nanmin(nll_smooth)), float(np.nanmax(nll_smooth))
    levels = np.linspace(vmin, vmax, 25)
    cf = ax2d.contourf(A, B, nll_smooth, levels=levels, cmap="RdYlGn_r")
    ax2d.contour(A, B, nll_smooth, levels=levels,
                 colors="k", linewidths=0.35, alpha=0.4)
    fig.colorbar(cf, ax=ax2d, label="NLL", format="%.4f")
    # θ_ft at centre
    ax2d.scatter([0], [0], s=200, color="#2563EB", marker="*",
                 zorder=6, label="θ_ft")
    # σ½ certified-basin circles
    circle_t = np.linspace(0, 2 * np.pi, 200)
    ax2d.plot(sh_pre * np.cos(circle_t), sh_pre * np.sin(circle_t),
              color="#3B82F6", lw=2, ls="--",
              label=f"σ½_pre = {sh_pre:.4f}")
    if sh_ft is not None:
        ax2d.plot(sh_ft * np.cos(circle_t), sh_ft * np.sin(circle_t),
                  color="#F97316", lw=2, ls="--",
                  label=f"σ½_ft  = {sh_ft:.4f}")
    ax2d.axvline(0, color="white", lw=0.6, ls="--", alpha=0.3)
    ax2d.axhline(0, color="white", lw=0.6, ls="--", alpha=0.3)
    ax2d.set_xlabel("α  (LoRA update direction d1)", fontsize=9)
    ax2d.set_ylabel("β  (random orthogonal d2)", fontsize=9)
    ax2d.set_title("2D Contour  (top-down)\nCircles = certified basin width σ½", fontsize=9)
    ax2d.legend(fontsize=7.5, loc="upper right")

    # ── Scale-context inset: schematic showing θ_pre→θ_ft relative to visible window ──
    # The 2D contour covers ±sweep ≈ ±3×σ½. The actual move ‖ΔW‖ is nd/sh_pre times larger.
    # This inset communicates that relationship without distorting the main contour.
    axsc = ax2d.inset_axes([0.01, 0.01, 0.40, 0.26])  # bottom-left corner
    axsc.set_xlim(0, 1)
    axsc.set_ylim(0, 1)
    axsc.set_facecolor("#F8FAFC")
    axsc.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
    for spine in axsc.spines.values():
        spine.set_linewidth(0.5)
    # Draw trajectory arrow: θ_pre (left) → θ_ft (right)
    axsc.annotate("", xy=(0.88, 0.52), xytext=(0.12, 0.52),
                  xycoords="axes fraction", textcoords="axes fraction",
                  arrowprops=dict(arrowstyle="-|>", color="#111827",
                                  lw=1.2, mutation_scale=8))
    axsc.scatter([0.12], [0.52], s=28, color="#111827", marker="o",
                 transform=axsc.transAxes, zorder=4)
    axsc.scatter([0.88], [0.52], s=40, color="#2563EB", marker="*",
                 transform=axsc.transAxes, zorder=4)
    axsc.text(0.12, 0.75, "θ_pre", ha="center", fontsize=5.5,
              transform=axsc.transAxes, color="#111827")
    axsc.text(0.88, 0.75, "θ_ft", ha="center", fontsize=5.5,
              transform=axsc.transAxes, color="#2563EB")
    axsc.text(0.50, 0.88, f"‖ΔW‖ = {nd/sh_pre:.0f}×σ½",
              ha="center", fontsize=5.5, transform=axsc.transAxes, color="#111827")
    # Highlight the "visible window" as a tiny shaded rectangle near θ_ft
    vis_frac = (sweep / nd) * 0.76   # fraction of arrow that the visible region spans
    rect_x = 0.88 - vis_frac / 2
    rect_x = max(rect_x, 0.80)       # keep it near θ_ft
    from matplotlib.patches import FancyBboxPatch
    highlight = FancyBboxPatch((rect_x, 0.25), vis_frac, 0.54,
                               boxstyle="square,pad=0", linewidth=1.0,
                               edgecolor="#F97316", facecolor="#FEF3C7",
                               alpha=0.7, transform=axsc.transAxes, zorder=3)
    axsc.add_patch(highlight)
    axsc.text(rect_x + vis_frac / 2, 0.12, "← map", ha="center",
              fontsize=5, transform=axsc.transAxes, color="#B45309")

    # ── Panel 2: 3D surface ───────────────────────────────────────────────────
    ax3d = fig.add_subplot(gs[1], projection="3d")
    surf = ax3d.plot_surface(A, B, nll_finite, cmap="RdYlGn_r",
                             alpha=0.75, linewidth=0, antialiased=True,
                             rstride=1, cstride=1)
    # Project the fine 1D trajectory onto the surface (β=0 slice)
    traj_in_view = np.isfinite(nll_fine) & (np.abs(fine_alphas) <= sweep)
    if traj_in_view.any():
        _fa = fine_alphas[traj_in_view]
        _nf = nll_fine[traj_in_view]
        _nf_sm = _nf + 0.00005   # lift slightly above surface so it's visible
        ax3d.plot(_fa, np.zeros_like(_fa), _nf_sm,
                  color="#7C3AED", lw=2.5, zorder=10, label="trajectory")
    ax3d.scatter([0], [0], [nll_finite[mid, mid]],
                 s=100, color="#2563EB", marker="*", zorder=11, depthshade=False)
    ax3d.set_xlabel("α (LoRA dir)", fontsize=7, labelpad=1)
    ax3d.set_ylabel("β (⊥)", fontsize=7, labelpad=1)
    ax3d.set_zlabel("NLL", fontsize=7, labelpad=1)
    ax3d.tick_params(labelsize=6)
    ax3d.set_title(f"3D Basin  (purple = trajectory slice)\nsweep ±{sigma_mult:.0f}×σ½",
                   fontsize=9, pad=4)
    fig.colorbar(surf, ax=ax3d, shrink=0.45, pad=0.08,
                 label="NLL", format="%.4f")

    # ── Panel 3: θ_pre → θ_ft → beyond  (main + zoom inset) ─────────────────
    ax1d = fig.add_subplot(gs[2])

    # Main plot: full ‖ΔW‖ scale
    valid_c = np.isfinite(dnll_traj)
    ax1d.plot(traj_alphas[valid_c], dnll_traj[valid_c],
              color="#7C3AED", lw=2.5, label="ΔNLL along d1")
    pre_i = int(np.argmin(np.abs(traj_alphas - (-nd))))
    ax1d.scatter([traj_alphas[pre_i]], [dnll_traj[pre_i]],
                 s=120, color="#111827", marker="o", zorder=5, label="θ_pre")
    ax1d.scatter([traj_alphas[ft_i_c]], [dnll_traj[ft_i_c]],
                 s=150, color="#2563EB", marker="*", zorder=5, label="θ_ft")
    ax1d.axhline(eval_slack, color="#16A34A", lw=1.3, ls="--",
                 label=f"cert. threshold (+{eval_slack:.0e})")
    ax1d.axhline(0, color="#2563EB", lw=0.7, ls=":", alpha=0.4)
    ax1d.axvline(0,   color="#2563EB", lw=0.8, ls=":", alpha=0.5)
    ax1d.axvline(-nd, color="#111827", lw=0.8, ls=":", alpha=0.5)
    ax1d.set_xlabel("α  (LoRA update direction, raw units)", fontsize=9)
    ax1d.set_ylabel("ΔNLL  =  NLL − NLL_ft", fontsize=9)
    ax1d.set_title("θ_pre → θ_ft → beyond\nΔNLL = 0 at θ_ft by definition", fontsize=9)
    ax1d.legend(fontsize=7.5, loc="upper left")
    ax1d.grid(True, alpha=0.2)

    # Inset: zoomed ±5×σ½ around θ_ft — shows certification and collapse
    axins = ax1d.inset_axes([0.52, 0.38, 0.46, 0.55])
    valid_f = np.isfinite(dnll_fine)
    axins.plot(fine_alphas[valid_f], dnll_fine[valid_f],
               color="#7C3AED", lw=2)
    axins.fill_between(fine_alphas[valid_f], 0, eval_slack,
                       where=(dnll_fine[valid_f] >= -eval_slack * 0.5) &
                             (dnll_fine[valid_f] <= eval_slack),
                       alpha=0.30, color="#16A34A")
    axins.axhline(eval_slack, color="#16A34A", lw=1.2, ls="--")
    axins.axhline(0, color="#2563EB", lw=0.7, ls=":", alpha=0.5)
    axins.scatter([0], [0], s=80, color="#2563EB", marker="*", zorder=5)
    axins.axvline(-sh_pre, color="#3B82F6", lw=1.1, ls=":")
    axins.axvline(sh_pre,  color="#3B82F6", lw=1.1, ls=":",
                  label=f"±σ½_pre")
    if sh_ft is not None:
        axins.axvline(-sh_ft, color="#F97316", lw=1.1, ls=":")
        axins.axvline(sh_ft,  color="#F97316", lw=1.1, ls=":",
                      label=f"±σ½_ft")
    axins.set_xlim(-5 * sh_pre, 5 * sh_pre)
    y_lo = min(-eval_slack * 0.5, float(np.nanmin(dnll_fine)))
    y_hi = max(eval_slack * 4, float(np.nanmax(dnll_fine)) * 0.5)
    axins.set_ylim(y_lo, y_hi)
    axins.set_title("zoom  ±5×σ½", fontsize=7)
    axins.tick_params(labelsize=6)
    axins.legend(fontsize=6, loc="upper right")
    axins.text(0, eval_slack * 1.15, "certified\nregion ↓",
               ha="center", fontsize=6, color="#16A34A")
    ax1d.indicate_inset_zoom(axins, edgecolor="#888888", alpha=0.6)

    fig.suptitle(
        f"Loss Landscape  |  ★ θ_ft = origin  "
        f"|  sweep ±{sigma_mult:.1f}×σ½={sweep:.2e}  "
        f"|  ‖ΔW‖={nd:.2e}  ({nd/sh_pre:.0f}×σ½)",
        fontsize=10, y=0.97,
    )
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"  Saved landscape: {save_path}")
    plt.close()


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

    # ── Sequential forgetting mode: Phase 1 (Task A) ─────────────────────────
    # Hypothesis: ‖θ_t − θ_A‖ / σ½_A > 1  ⟺  Task A accuracy degrades.
    # σ½_A is computed from Task A classification NLL (not WikiText-2), so the
    # certified ball directly bounds Task A performance, not an unrelated loss.
    classifier_a     = None
    task_a_inp       = None   # fixed batch for Task A density / forgetting eval
    task_a_labels    = None
    task_a_acc_phase1 = None  # recorded in trj_json for self-contained analysis
    if args.task_a is not None:
        if args.track_interval == 0:
            raise ValueError(
                "--task_a (sequential forgetting mode) requires --track_interval > 0 "
                "to record Task A forgetting during Phase 2 training. "
                "Suggested: --track_interval 50"
            )

        steps_a = args.train_steps_a if args.train_steps_a is not None else args.train_steps[0]
        seq_out_dir = os.path.join(
            args.output_dir,
            args.model.replace("/", "_"),
            f"seq_{args.task_a}_to_{args.task}")
        os.makedirs(seq_out_dir, exist_ok=True)
        out_dir = seq_out_dir

        metric_label = (f"clf_acc (δ={args.task_a_acc_delta:.2f})"
                        if args.task_a_metric == "acc"
                        else f"clf_nll (slack={args.task_a_eval_slack:.0e})")
        print(f"\n{'='*60}")
        print(f"SEQUENTIAL FORGETTING  Phase 1: {args.task_a}  steps={steps_a}")
        print(f"  σ½_A metric: {metric_label}")
        print(f"{'='*60}")

        phase1_ckpt_path = os.path.join(out_dir, "phase1_model.pt")
        phase1_clf_path  = os.path.join(out_dir, "phase1_classifier.pt")
        phase1_meta_path = os.path.join(out_dir, "phase1_meta.json")

        if args.skip_phase1_finetune:
            # Load θ_A and classifier from a previous Phase 1 run.
            if not (os.path.exists(phase1_ckpt_path) and
                    os.path.exists(phase1_clf_path) and
                    os.path.exists(phase1_meta_path)):
                raise FileNotFoundError(
                    "--skip_phase1_finetune requires phase1_model.pt, "
                    "phase1_classifier.pt, and phase1_meta.json in the output dir. "
                    f"Not found in: {out_dir}")
            print(f"  [skip_phase1_finetune] Loading θ_A from {phase1_ckpt_path}")
            model.load_state_dict(torch.load(phase1_ckpt_path, map_location=device))
            with open(phase1_meta_path) as f:
                phase1_meta = json.load(f)
            task_a_acc_phase1 = phase1_meta["task_a_acc"]
            num_classes_a, _, _ = _CLF_TASKS[args.task_a]
            hidden_size_a = _get_hidden_size(model)
            classifier_a = nn.Linear(hidden_size_a, num_classes_a).to(device)
            classifier_a.load_state_dict(
                torch.load(phase1_clf_path, map_location=device))
            classifier_a.eval()
            names = _get_target_names_auto(model, args.model)
            print(f"  [Phase 1 loaded]  task_a accuracy = {task_a_acc_phase1:.4f}")
        else:
            _lr_a   = args.lr_a   if args.lr_a   is not None else args.lr[0]
            _rank_a = args.rank_a if args.rank_a is not None else args.ranks[0]
            print(f"  Phase 1 LR={_lr_a}  rank={_rank_a}  steps={steps_a}")
            (model, _, _, _, _, _, task_a_acc_phase1, classifier_a) = finetune(
                model, tokenizer,
                task           = args.task_a,
                lr             = _lr_a,
                train_steps    = steps_a,
                batch_size     = args.train_batch,
                rank           = _rank_a,
                alpha          = args.lora_alpha,
                device         = device,
                target_modules = target_modules,
                finetune_fp32  = args.finetune_fp32,
                grad_clip      = args.grad_clip,
                classifier_lr  = args.classifier_lr,
            )
            names = _get_target_names_auto(model, args.model)
            print(f"  [Phase 1 fine-tune done]  task_a accuracy = {task_a_acc_phase1:.4f}")

            _min_phase1_acc = 0.70
            if task_a_acc_phase1 < _min_phase1_acc:
                raise RuntimeError(
                    f"Phase 1 ({args.task_a}) accuracy {task_a_acc_phase1:.4f} < {_min_phase1_acc}. "
                    f"The certified ball σ½_A will be spuriously large and the experiment "
                    f"will not be informative. Increase --train_steps_a or use --lr_a 1e-4.")

            # Save θ_A checkpoint so --skip_phase1_finetune can reload it later.
            torch.save(model.state_dict(), phase1_ckpt_path)
            torch.save(classifier_a.state_dict(), phase1_clf_path)
            with open(phase1_meta_path, "w") as f:
                json.dump({"task_a": args.task_a, "task_a_acc": task_a_acc_phase1,
                           "lr_a": _lr_a, "rank_a": _rank_a, "steps_a": steps_a}, f)
            print(f"  Saved Phase 1 checkpoint → {phase1_ckpt_path}")

        # Fixed eval batch: same 128 examples every time → deterministic σ½_A
        task_a_inp, task_a_labels = _get_task_clf_batch(
            tokenizer, args.task_a,
            max_samples=128, device=device)

        # Compute σ½_A using the selected metric
        if args.task_a_metric == "acc":
            print(f"\n[Phase 1] σ½_A density sweep  "
                  f"(acc-based, δ={args.task_a_acc_delta:.2f}, N={args.n_perturb})...")
            pre_results = run_density_clf_acc(
                model, classifier_a, names,
                task_a_inp, task_a_labels,
                args.sigmas, args.n_perturb, args.task_a_acc_delta,
                tag="phase1_task_a_clf_acc")
            phase1_eval_key = "task_a_clf_acc"
        else:
            print(f"\n[Phase 1] σ½_A density sweep  "
                  f"(NLL, slack={args.task_a_eval_slack:.0e}, N={args.n_perturb})...")
            pre_results = run_density_clf(
                model, classifier_a, names,
                task_a_inp, task_a_labels,
                args.sigmas, args.n_perturb, args.task_a_eval_slack,
                tag="phase1_task_a_clf")
            phase1_eval_key = "task_a_clf_nll"

        phase1_suffix = f"_{args.task_a_metric}" if args.task_a_metric != "nll" else ""
        phase1_json = os.path.join(out_dir, f"phase1_task_a_density{phase1_suffix}.json")
        with open(phase1_json, "w") as f:
            json.dump({
                "model":         args.model,
                "condition":     f"task_a={args.task_a}",
                "eval":          phase1_eval_key,
                "eval_slack":    args.task_a_eval_slack,
                "acc_delta":     args.task_a_acc_delta,
                "task_a_acc":    task_a_acc_phase1,
                "sigma_results": pre_results,
            }, f, indent=2)
        print(f"  Saved: {phase1_json}")

        sh_pre = half_max_sigma(pre_results)
        print(f"\n{'='*60}")
        print(f"[Phase 1 complete]")
        print(f"  σ½_A ({args.task_a_metric}) = {sh_pre:.6f}")
        print(f"  task_a accuracy            = {task_a_acc_phase1:.4f}")
        print(f"\nSEQUENTIAL FORGETTING  Phase 2: task_b={args.task}")
        print(f"  Starting from θ_A  |  tracking norm relative to θ_A")
        print(f"  norm_over_sh_pre in trajectory = ‖θ_t − θ_A‖ / σ½_A")
        print(f"  Hypothesis: task_a_val_acc drops when norm_over_sh_pre crosses 1.0")
        print(f"{'='*60}\n")

    # ── Pretrained density (compute once) ─────────────────────────────────────
    pre_json = os.path.join(out_dir, "pretrained_density.json")
    if args.task_a is not None:
        # Sequential mode: σ½_pre is already set to σ½_A above; skip this block.
        pass
    elif os.path.exists(pre_json) and not args.force:
        print("Reloading pretrained density...")
        with open(pre_json) as f:
            pre_results = json.load(f)["sigma_results"]
        sh_pre = half_max_sigma(pre_results)
        print(f"\nPretrained σ½ = {sh_pre:.6f}  "
              f"ppl = {pre_results[0]['base_ppl']:.4f}")
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

    # ── Sweep lr × ranks × steps ─────────────────────────────────────────────
    records     = []   # for summary JSON and plots
    ft_meta     = []   # for density overlay plot
    multi_lr    = len(args.lr) > 1

    for lr in args.lr:
        lr_tag = f"{lr:.0e}"   # e.g. "1e-04"
        for steps in args.train_steps:
            for rank in args.ranks:
                tag = (f"lr{lr_tag}_rank{rank}_steps{steps}" if multi_lr
                       else f"rank{rank}_steps{steps}")
                ft_json  = os.path.join(out_dir, f"{tag}_density.json")
                nrm_json = os.path.join(out_dir, f"{tag}_norm.json")
                trj_json = os.path.join(out_dir, f"{tag}_trajectory.json")

                if os.path.exists(ft_json) and os.path.exists(nrm_json) and not args.force:
                    print(f"Reloading {tag}...")
                    with open(ft_json)  as f: ft_data  = json.load(f)
                    with open(nrm_json) as f: nrm_data = json.load(f)
                    ft_results     = ft_data["sigma_results"]
                    per_param_norm = nrm_data["per_param_norm"]
                    spectral_norm  = nrm_data.get("spectral_norm", float("nan"))
                    grad_norm      = nrm_data.get("grad_norm", float("nan"))
                    task_val_acc   = nrm_data.get("task_val_acc", float("nan"))
                    quant_results  = nrm_data.get("quant_results", {})
                else:
                    (ft_model, per_param_norm, spectral_norm,
                     resolved_alpha, trajectory, loss_history,
                     task_val_acc, _) = finetune(
                        model, tokenizer,
                        task         = args.task,
                        lr           = lr,
                        train_steps  = steps,
                        batch_size   = args.train_batch,
                        rank         = rank,
                        alpha        = args.lora_alpha,
                        device       = device,
                        target_modules = target_modules,
                        finetune_fp32  = args.finetune_fp32,
                        grad_clip      = args.grad_clip,
                        track_interval   = args.track_interval,
                        track_inp        = inp if args.track_interval > 0 else None,
                        track_tgt        = tgt if args.track_interval > 0 else None,
                        track_names      = names if args.track_interval > 0 else None,
                        track_eval_slack = args.eval_slack,
                        track_sh_pre     = sh_pre,
                        classifier_lr    = args.classifier_lr,
                        seq_task_a       = args.task_a,
                        seq_classifier_a = classifier_a,
                    )
                    ft_names = _get_target_names_auto(ft_model, args.model)

                    # Measure Task A forgetting at end of Phase 2
                    task_a_acc_final = float("nan")
                    if args.task_a is not None and classifier_a is not None:
                        task_a_acc_final = _evaluate_task_accuracy(
                            ft_model, classifier_a, tokenizer, args.task_a, device)
                        print(f"  [Sequential] End of Phase 2  "
                              f"task_a_acc={task_a_acc_final:.4f}  "
                              f"task_b_acc={task_val_acc:.4f}", flush=True)

                    # Save trajectory — include sequential metadata so the JSON
                    # is self-contained for hypothesis evaluation
                    with open(trj_json, "w") as f:
                        json.dump({"model": args.model, "task": args.task,
                                   "rank": rank, "steps": steps, "lr": lr,
                                   "track_interval": args.track_interval,
                                   "sigma_half_pre": sh_pre,
                                   # Sequential forgetting metadata
                                   "seq_task_a":          args.task_a,
                                   "seq_task_a_eval_slack": args.task_a_eval_slack
                                       if args.task_a else None,
                                   "seq_task_a_acc_phase1": task_a_acc_phase1
                                       if args.task_a else None,
                                   "seq_task_a_acc_final":  task_a_acc_final,
                                   "loss_history": loss_history,
                                   "trajectory": trajectory}, f, indent=2)
                    print(f"  Saved trajectory+loss: {trj_json}")

                    print(f"\n[{tag}] Density sweep...")
                    ft_results = run_density(ft_model, ft_names, inp, tgt,
                                             args.sigmas, args.n_perturb,
                                             args.eval_slack, tag=tag)

                    # Weight landscape
                    if args.landscape_grid > 0:
                        land_path = os.path.join(out_dir, f"{tag}_landscape.png")
                        pretrained_snapshot = {n: p.data.clone().float()
                                               for n, p in model.named_parameters()}
                        _sh_ft = half_max_sigma(ft_results)
                        plot_weight_landscape(
                            pretrained_snapshot, ft_model, ft_names,
                            inp, tgt, sh_pre,
                            n_grid     = args.landscape_grid,
                            sigma_mult = args.landscape_sigma_mult,
                            sh_ft      = _sh_ft,
                            eval_slack = args.eval_slack,
                            save_path  = land_path,
                        )

                    grad_norm = _compute_grad_norm(ft_model, ft_names, inp, tgt)
                    print(f"  grad_norm={grad_norm:.6f}", flush=True)

                    quant_results = {}
                    if args.quantize:
                        ppl_ft_base = float(np.exp(compute_nll_loss(ft_model, inp, tgt)))
                        for bits in (8, 4):
                            q_model = _simulate_quantization(ft_model, bits)
                            ppl_q   = float(np.exp(compute_nll_loss(q_model, inp, tgt)))
                            quant_results[f"ppl_int{bits}"] = ppl_q
                            quant_results[f"ppl_ratio_int{bits}"] = (
                                ppl_q / ppl_ft_base if ppl_ft_base > 0 else float("nan"))
                            print(f"  INT{bits}: ppl={ppl_q:.4f}  ratio={quant_results[f'ppl_ratio_int{bits}']:.4f}",
                                  flush=True)
                            del q_model
                        if device.type == "cuda":
                            torch.cuda.empty_cache()

                    del ft_model
                    if device.type == "cuda":
                        torch.cuda.empty_cache()

                    with open(ft_json, "w") as f:
                        json.dump({"model": args.model, "task": args.task,
                                   "rank": rank, "alpha": resolved_alpha,
                                   "steps": steps, "lr": lr,
                                   "per_param_norm": per_param_norm,
                                   "spectral_norm":  spectral_norm,
                                   "sigma_results": ft_results}, f, indent=2)
                    with open(nrm_json, "w") as f:
                        json.dump({"rank": rank, "steps": steps, "lr": lr,
                                   "per_param_norm":       per_param_norm,
                                   "spectral_norm":        spectral_norm,
                                   "grad_norm":            grad_norm,
                                   "sigma_half_pre":       sh_pre,
                                   "task_val_acc":         task_val_acc,
                                   "task_a_val_acc_final": task_a_acc_final,
                                   "quant_results":        quant_results}, f, indent=2)

                sh_ft = half_max_sigma(ft_results)
                ratio = sh_ft / sh_pre if sh_pre > 0 else float("nan")
                spec_ratio = spectral_norm / sh_pre if sh_pre > 0 else float("nan")
                grad_x_sh = (grad_norm * sh_pre
                             if (sh_pre > 0 and not np.isnan(grad_norm)) else float("nan"))
                print(f"  {tag}: σ½ {sh_pre:.6f} → {sh_ft:.6f}  "
                      f"ratio={ratio:.3f}  norm/σ½={per_param_norm/sh_pre:.3f}  "
                      f"spec/σ½={spec_ratio:.3f}  grad×σ½={grad_x_sh:.5f}nats")

                records.append({"rank": rank, "steps": steps, "lr": lr,
                                "pre": pre_results, "ft": ft_results,
                                "norm": per_param_norm,
                                "spectral_norm":  spectral_norm,
                                "grad_norm":      grad_norm,
                                "task_val_acc":   task_val_acc,
                                "quant_results":  quant_results})
                ft_meta.append({"rank": rank, "steps": steps, "lr": lr,
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
          f"{'ratio':>7}  {'norm/σ½':>8}  {'ppl_ratio':>10}  "
          f"{'task_acc':>9}  {'int8_ratio':>10}  {'int4_ratio':>10}")
    print("-" * 110)
    for rec in sorted(summary, key=lambda r: (r["steps"], r["rank"])):
        print(f"  {rec['rank']:>4}  {rec['steps']:>6}  "
              f"{rec['sigma_half_pre']:>10.6f}  {rec['sigma_half_ft']:>10.6f}  "
              f"{rec['widening_ratio']:>7.3f}  "
              f"{rec['norm_over_sh']:>8.3f}  "
              f"{rec['ppl_ratio']:>10.4f}  "
              f"{rec.get('task_val_acc', float('nan')):>9.4f}  "
              f"{rec.get('ppl_ratio_int8', float('nan')):>10.4f}  "
              f"{rec.get('ppl_ratio_int4', float('nan')):>10.4f}")

    # ── Plots ─────────────────────────────────────────────────────────────────
    plot_widening_vs_rank(summary, out_dir, args.model, args.task)
    plot_density_curves(pre_results, ft_meta, out_dir, args.model, args.task)

    print(f"\nDone. Results in: {out_dir}")


if __name__ == "__main__":
    main()
