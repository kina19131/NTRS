"""
hessian_eigenvalue.py
=====================
Q1: Does σ½ correlate with the top Hessian eigenvalue λ_max?

For each pretrained model:
  1. Load σ½ from existing pretrained_density.json (already computed).
  2. Compute λ_max via power iteration using Hessian-vector products.
  3. Plot σ½ vs 1/√λ_max — theory predicts a linear relationship through origin.

Theory: at a quadratic minimum, the certified density C(θ,σ) ≈ 0.5 when the
perturbation energy matches eval_slack, i.e.:
    ½ λ_max σ½² ≈ eval_slack  →  σ½ ≈ √(2·eval_slack / λ_max)

If this holds, σ½ is a certified sharpness proxy: cheaper to compute than
Hessian eigenvalues (~200 forward passes vs. ~100 forward+backward passes each
needing create_graph=True), and probabilistically interpretable.

Usage:
  python3 hessian_eigenvalue.py
  python3 hessian_eigenvalue.py --models gpt2 gpt2-medium gpt2-large gpt2-xl
  python3 hessian_eigenvalue.py --models meta-llama/Llama-3.2-3B

  # Run on GPU 1 while H2 experiment runs on GPU 0:
  CUDA_VISIBLE_DEVICES=1 python3 hessian_eigenvalue.py
"""

import argparse
import contextlib
import json
import os
import sys
import time

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(__file__))
from lora_density_experiment import (
    _load_model_auto, _get_tokenizer,
)
from certified_density_experiment import load_data, half_max_sigma
from h1_kl_comparison import _get_lora_scoped_names

BASE = os.path.dirname(os.path.abspath(__file__))

# ── Existing σ½ data (from prior density experiments) ────────────────────────
# Maps model name → list of pretrained_density*.json paths.
# Multiple paths per model = average σ½ across tasks (σ½ is a model property,
# not a task property — variance across tasks shows measurement noise).
#
# 2026-07-20: repointed to the current relslack-corrected files (the ones cited
# throughout RAW_DATA.md/SUMMARY.md) — the original paths here (outputs_0615_v2/,
# outputs_0617_h1/, results/basin_widening_v2/) were an older, pre-relslack
# experiment layout and do not exist in this repo; every one of them silently
# resolved to "no σ½ data found — skipping" (see load_sigma_half below), so this
# script has never actually produced a result. Only the 3 models used in H1/H2
# are listed now (gpt2-medium/large/xl were a separate, abandoned scaling probe
# with no current data and aren't part of this project's H1/H2 comparison).
SIGMA_SOURCES = {
    "gpt2": [
        "h1_kl/gpt2/pretrained_density_relslack.json",
    ],
    "meta-llama/Llama-3.2-3B": [
        "h1_kl/meta-llama/Llama-3.2-3B/pretrained_density_relslack.json",
    ],
    "mistralai/Mistral-7B-v0.1": [
        # Same frozen pretrained checkpoint as sst2_v5 (confirmed via matching
        # nll_wiki_base, see SUMMARY.md) — v3's relslack file is the canonical
        # one cited throughout RAW_DATA.md/SUMMARY.md's Key Numbers table.
        "h1_kl/mistralai/Mistral-7B-v0.1_sst2_v3/pretrained_density_relslack.json",
    ],
}

MODEL_N_PARAMS = {
    "gpt2":                       124e6,
    "gpt2-medium":                354e6,
    "gpt2-large":                 774e6,
    "gpt2-xl":                    1.5e9,
    "meta-llama/Llama-3.2-3B":    3e9,
    "mistralai/Mistral-7B-v0.1":  7e9,
}

MODEL_COLOR = {
    "gpt2":                       "#1D4ED8",
    "gpt2-medium":                "#2563EB",
    "gpt2-large":                 "#60A5FA",
    "gpt2-xl":                    "#93C5FD",
    "meta-llama/Llama-3.2-3B":    "#DC2626",
    "mistralai/Mistral-7B-v0.1":  "#EA580C",
}


# _get_lora_scoped_names now lives in h1_kl_comparison.py (moved 2026-07-21 so
# σ½ calibration and this Hessian probe share one definition of "LoRA-scoped").
# Originally added here 2026-07-20 for two independent reasons, still true:
# (1) OOM — the whole-block scope needs fp32 grad buffers for ~7B params on
#     Mistral (two full-size buffers for hvp_fd's ±eps gradients), which
#     filled a 44GB GPU by itself before any perturbation was even applied.
# (2) Relevance — this directly answers the more useful question: is the
#     checkpoint curved (a real basin) in the SPECIFIC subspace fine-tuning
#     actually moves in, rather than in a much larger space dominated by
#     parameters LoRA never touches.


# ── Args ──────────────────────────────────────────────────────────────────────
def get_args():
    p = argparse.ArgumentParser()
    p.add_argument("--models", nargs="+",
                   default=["gpt2", "meta-llama/Llama-3.2-3B",
                            "mistralai/Mistral-7B-v0.1"])
    p.add_argument("--n_iter", type=int, default=24,
                   help="Lanczos steps (was 'max power iterations' before the "
                        "2026-07-20 rewrite — see lanczos_spectrum). Needs far "
                        "fewer than the old power-iteration scheme (20-30 is "
                        "typically enough for the extremal eigenvalues of a "
                        "large, sparse-ish spectrum), and each step stores one "
                        "more full-size basis vector on CPU for reorthogonalization "
                        "— e.g. for Mistral's ~1.3B LoRA-scoped target params in "
                        "bf16, 24 steps is roughly 24 * 2.6GB ≈ 63GB of CPU RAM. "
                        "Reduce if the pod is CPU-RAM constrained.")
    p.add_argument("--tol", type=float, default=0.2,
                   help="Relative-agreement threshold for the eps-doubling "
                        "stability check (FD/large-model path only): the top "
                        "positive eigenvalue is computed at both eps and 2*eps, "
                        "and if they disagree by more than this fraction the "
                        "result is reported as unstable (NaN) rather than "
                        "silently trusting a single Lanczos run's own internal "
                        "convergence — that trust is exactly what let GPT-2's "
                        "old Phase-2 bug and Llama/Mistral's FD noise through "
                        "undetected before this rewrite.")
    p.add_argument("--hessian_batch", type=int, default=4,
                   help="Examples per HVP call. Kept small to avoid OOM from "
                        "create_graph=True. 4 is sufficient for stable power iteration.")
    p.add_argument("--eps", type=float, default=1e-4,
                   help="Base finite-difference step size (FD/large-model path "
                        "only; ignored for double-backward). Internally rescaled "
                        "by sqrt(n_target_params) — see hvp_fd docstring — so "
                        "this value means the same thing regardless of model "
                        "size. Also re-run at 2x this value for the stability "
                        "check (see --tol).")
    p.add_argument("--n_text_samples", type=int, default=128,
                   help="Samples loaded for WikiText-2 eval (same as density experiments). "
                        "Only the first --hessian_batch are used for HVP.")
    p.add_argument("--seq_len", type=int, default=128)
    p.add_argument("--eval_slack", type=float, default=None,
                   help="Fallback only, used if a model's density JSON has no "
                        "eval_slack_abs field. Normally each model's own "
                        "eval_slack_abs (1% of its own baseline NLL, relslack "
                        "convention) is read from its density JSON directly — "
                        "passing one shared value here for all models would "
                        "reintroduce the exact mismatched-eval_slack bug that "
                        "the relslack correction (RAW_DATA.md/SUMMARY.md) fixed "
                        "for σ½ itself, since baseline NLL differs ~1.6x across "
                        "GPT-2/Llama/Mistral.")
    p.add_argument("--out", default="results/hessian_eigenvalue")
    p.add_argument("--force", action="store_true",
                   help="Recompute λ_max even if already in results.json.")
    return p.parse_args()


# ── Load σ½ from existing density JSONs ──────────────────────────────────────
def load_sigma_half(model_name):
    """
    Average σ½ (and its eval_slack_abs) across all available pretrained density
    files for this model.

    2026-07-20: rewritten for the current relslack file schema, which stores
    "sigma_half" and "eval_slack_abs" directly (see recompute_h1_sigma_relslack.py
    / h1_kl_comparison.py output). The original version here expected a raw
    "sigma_results" list (estimate_density()'s own per-sigma output, as produced
    by lora_density_experiment.py's older pretrained_density.json format) and
    called half_max_sigma() to recompute σ½ from scratch — that key doesn't
    exist in any current file, so this always KeyError'd/was unreachable in
    practice (masked by every SIGMA_SOURCES path also being wrong at the time).
    Falls back to the old "sigma_results" format if "sigma_half" isn't present,
    so this still works against any genuinely old-format file.
    """
    paths = SIGMA_SOURCES.get(model_name, [])
    sh_values, slack_values = [], []
    for rel in paths:
        path = os.path.join(BASE, rel)
        if not os.path.exists(path):
            print(f"  [warn] missing: {rel}")
            continue
        with open(path) as f:
            data = json.load(f)
        if "sigma_half" in data:
            sh = data["sigma_half"]
            slack = data.get("eval_slack_abs")
        else:
            sh = half_max_sigma(data["sigma_results"])
            slack = None
        sh_values.append(sh)
        if slack is not None:
            slack_values.append(slack)
        print(f"  σ½ = {sh:.6f}  (eval_slack_abs={slack})  ← {rel}")
    if not sh_values:
        return None, None, None
    mean_sh    = float(np.mean(sh_values))
    std_sh     = float(np.std(sh_values))
    mean_slack = float(np.mean(slack_values)) if slack_values else None
    if len(sh_values) > 1:
        print(f"  σ½ mean = {mean_sh:.6f}  std = {std_sh:.6f}")
    return mean_sh, std_sh, mean_slack


# ── Hessian-vector product ────────────────────────────────────────────────────
def hvp(model, param_list, inp, tgt, v):
    """
    Hessian-vector product via double-backward (create_graph=True).
    Works for small/medium models. Requires flash attention disabled.
    OOMs on large models (>2B params) because the backward graph is huge.
    """
    model.zero_grad()

    # Disable flash/efficient attention — they don't implement backward-of-backward.
    # Use the new sdpa_kernel API (PyTorch 2.4+); fall back to deprecated API if needed.
    if inp.is_cuda:
        try:
            from torch.nn.attention import sdpa_kernel, SDPBackend
            ctx = sdpa_kernel([SDPBackend.MATH])
        except (ImportError, AttributeError):
            ctx = torch.backends.cuda.sdp_kernel(
                enable_flash=False, enable_mem_efficient=False, enable_math=True)
    else:
        ctx = contextlib.nullcontext()
    with ctx:
        loss = model(inp, labels=tgt).loss
        # First backward: keep graph so we can differentiate through it
        grads = torch.autograd.grad(loss, param_list, create_graph=True)

    # g · v (scalar) — v is detached so this doesn't extend the graph further
    gv = sum((g * vi).sum() for g, vi in zip(grads, v))

    # Second backward: ∂(g·v)/∂θ = Hv
    Hv = torch.autograd.grad(gv, param_list, retain_graph=False)
    return [hvi.detach() for hvi in Hv]


def hvp_fd(model, param_list, inp, tgt, v, eps=1e-4):
    """
    Hessian-vector product via finite differences: Hv ≈ (∇L(θ+ε'v) − ∇L(θ−ε'v)) / (2ε').

    Perturbs from one snapshot of the true parameter values (see the second
    fix note below for why it's a snapshot-and-restore now, not sequential
    in-place shifts). v must be normalised (unit vector, over ALL of param_list combined) by the
    caller; may be any dtype. Returns Hv in fp32 (central-difference needs fp32
    to avoid cancellation).

    2026-07-20 fix — bf16 underflow, confirmed live on Llama-3.2-3B: ε' = eps *
    √d (d = total elements across param_list), NOT eps directly. v is globally
    unit-normalized, so each coordinate has magnitude ~1/√d — for a several-
    billion-parameter model, a literal `eps * v` step is astronomically small
    (e.g. d≈3e9, eps=1e-4 → per-coordinate step ≈1.8e-9), far below bf16's
    representable increment (~1e-4 to 1e-5 at typical weight magnitude): θ+εv
    and θ-εv round to the IDENTICAL bf16 tensor, the two gradients come back
    bit-identical, Hv=0 exactly — a numerical-underflow artifact, not a real
    "zero curvature" finding (this is exactly what produced Llama's bogus
    "Hv ≈ 0 at iter 1" / λ_max≈0 result). Scaling by √d keeps the actual
    per-coordinate perturbation at a fixed ~eps regardless of model size —
    matching certified_density_experiment.py's own convention (perturb_weights
    adds sigma directly per-parameter, no 1/√d shrinkage). This doesn't change
    what the estimate approximates: only ε'·v ever enters the perturbation, and
    the same ε' is used in the final division, so this is still a standard
    central-difference estimate of Hv for the (unrescaled) v the caller passed in.

    The ± perturbation itself is computed in fp32 (mirrors perturb_weights) so
    the arithmetic doesn't compound rounding error beyond what the final bf16
    storage already imposes.

    2026-07-20 second fix — inexact restore, confirmed by inspection: the
    previous version perturbed via THREE SEQUENTIAL in-place shifts
    (θ→θ+ε'v→θ-ε'v→θ), each individually casting fp32→bf16 for storage. Three
    separate bf16 round-trips do not cancel to bit-for-bit the original value
    — each rounds independently (up to half a bf16 ULP), so the "restored" θ
    differs from the true original by a small residual, NOT exactly zero.
    Since ε' is deliberately sized to be only just above bf16's representable
    increment, this residual isn't negligible relative to it. This function is
    called ~n_iter times per Lanczos run (lanczos_spectrum calls it once per
    step), so with the old sequential-shift approach this residual would
    compound across iterations, meaning LATER Lanczos steps measure curvature
    at a slightly different (drifted) point than EARLIER ones — violating the
    fixed-operator assumption the whole method depends on, and plausibly a
    real contributor to the old power-iteration scheme's chaotic, never-
    converging behavior on Llama/Mistral (bf16+FD) while GPT-2's exact
    double-backward path (no perturb/restore cycle at all) converged cleanly.

    Fixed by snapshotting the true θ ONCE, computing θ±ε'v independently from
    that snapshot (not via sequential shifts of an already-shifted state), and
    restoring via a direct bit-for-bit copy_ — no arithmetic, no rounding, no
    possible drift, regardless of how many times this function is called.
    """
    d = sum(vi.numel() for vi in v)
    eps_eff = eps * (d ** 0.5)

    orig = [p.data.clone() for p in param_list]

    def _set(alpha):
        with torch.no_grad():
            for p, o, vi in zip(param_list, orig, v):
                p_f32 = o.float()
                p_f32.add_(vi.float(), alpha=alpha)
                p.data.copy_(p_f32.to(p.dtype))

    # θ_orig → θ_orig + ε'v  (from the snapshot, not from a prior shifted state)
    _set(eps_eff)
    model.zero_grad()
    model(inp, labels=tgt).loss.backward()
    grad_plus = [p.grad.clone() for p in param_list]

    # θ_orig → θ_orig − ε'v  (also from the snapshot, independently)
    _set(-eps_eff)
    model.zero_grad()
    model(inp, labels=tgt).loss.backward()
    grad_minus = [p.grad.clone() for p in param_list]

    # Exact restore — bit-for-bit copy, no rounding, no drift
    with torch.no_grad():
        for p, o in zip(param_list, orig):
            p.data.copy_(o)
    model.zero_grad()

    # fp32 subtraction to avoid bf16 cancellation; grad buffers freed immediately after
    return [(gp.float() - gm.float()) / (2 * eps_eff)
            for gp, gm in zip(grad_plus, grad_minus)]


# ── Lanczos iteration ─────────────────────────────────────────────────────────
#
# 2026-07-20: replaces the old power-iteration + shift-and-restart scheme
# (see git history). That scheme had two confirmed failures in its first real
# run: (1) GPT-2's Phase 2 "converged" in exactly 2 iterations to exactly the
# shift value — a fresh random restart's Rayleigh quotient on (H+shift·I) is
# dominated by the shift term for generic directions when shift >> the true
# λ_max^+, so the iterate barely moves and the relative-change convergence
# check false-positives almost immediately; (2) Llama/Mistral's finite-
# difference HVP oscillated chaotically for the full 100-iteration budget
# without ever converging (bf16 forward/backward compute noise dominating a
# single power-iteration trajectory that has nothing to average it against).
#
# Lanczos fixes both structurally rather than symptomatically: it builds an
# orthogonal Krylov basis from repeated HVP calls and extracts eigenvalues
# from a small (n_iter x n_iter) tridiagonal projection. Both the most
# negative AND most positive eigenvalues fall out of ONE run (no separate
# shift/restart phase needed — this is exactly what neural-net Hessian-
# spectrum papers use it for, e.g. Ghorbani et al., Papyan). It's also more
# robust to noisy individual HVP evaluations than plain power iteration,
# since the final extraction aggregates information across every Krylov
# direction rather than committing to one evolving vector — though this
# alone doesn't guarantee immunity to bf16 compute noise, hence the explicit
# eps-doubling stability check in top_eigenvalue below.

def _vec_dot(a, b):
    return float(sum((ai.float() * bi.float()).sum() for ai, bi in zip(a, b)).item())

def _vec_norm(a):
    return _vec_dot(a, a) ** 0.5

def _vec_axpy(a, alpha, b):
    """a += alpha * b, in place. a/b may differ in dtype (e.g. fp32 Hv, bf16 v)."""
    for ai, bi in zip(a, b):
        ai.add_(bi.to(ai.dtype), alpha=alpha)

def _vec_scale_(a, alpha):
    for ai in a:
        ai.mul_(alpha)

def _to_cpu(a):
    return [ai.detach().to("cpu") for ai in a]

def _to_device(a, device, dtype):
    return [ai.to(device=device, dtype=dtype) for ai in a]


def lanczos_spectrum(model, param_list, inp, tgt, n_iter, hvp_fn, v0=None):
    """
    Lanczos tridiagonalization for the extremal eigenvalues of the Hessian
    restricted to param_list, using only Hessian-vector products (hvp_fn).

    Full reorthogonalization against every previous basis vector (needed for
    numerical stability — Lanczos without it is known to produce spurious
    "ghost" eigenvalues from loss of orthogonality). Basis vectors are stored
    on CPU between uses (only moved to GPU transiently for a dot product) to
    keep peak GPU memory bounded — for a several-hundred-million-parameter
    target set, keeping n_iter such vectors on GPU simultaneously would risk
    the same class of OOM already hit once with the old whole-block scope.

    v0: optional starting vector (list of tensors matching param_list's shapes
    and count). If given, used (a clone, unit-normalized) instead of a fresh
    random vector. Needed so top_eigenvalue's eps-doubling stability check
    compares two runs differing ONLY in eps — confirmed as a real confound
    2026-07-21 on Llama: without a shared v0, the eps and 2*eps runs each
    start from an independently random Krylov vector, so part of any observed
    disagreement between them could be ordinary Lanczos-restart variance
    (a 24-step Krylov subspace's quality depends on how well the random start
    happens to overlap the true eigenvectors) rather than eps-driven bf16
    noise — muddying exactly the verdict the stability check exists to give.

    Returns eigvals: the Ritz values (approximate eigenvalues of the full
    Hessian, restricted to the target params), sorted ascending. Empty if the
    Krylov subspace collapsed at step 0 (Hv≈0 for the very first vector).
    """
    v_dtype = next(iter(param_list)).dtype
    device  = param_list[0].device
    if v0 is None:
        v = [torch.randn_like(p, dtype=v_dtype) for p in param_list]
    else:
        v = [vi.clone().to(device=device, dtype=v_dtype) for vi in v0]
    _vec_scale_(v, 1.0 / _vec_norm(v))

    basis_cpu = []
    alphas, betas = [], []
    v_prev, beta_prev = None, 0.0
    t_start = time.time()

    for j in range(n_iter):
        t0 = time.time()
        Hv = hvp_fn(model, param_list, inp, tgt, v)  # fp32
        alpha_j = _vec_dot(Hv, v)
        alphas.append(alpha_j)
        print(f"    step {j+1:2d}/{n_iter}  alpha={alpha_j:10.4f}  "
              f"({time.time()-t0:.1f}s this step, {time.time()-t_start:.0f}s total)",
              flush=True)

        w = [hvi.clone() for hvi in Hv]
        _vec_axpy(w, -alpha_j, v)
        if v_prev is not None:
            _vec_axpy(w, -beta_prev, v_prev)

        # Full reorthogonalization against every basis vector so far,
        # INCLUDING the current v_j (floating-point error in the alpha_j
        # subtraction above can leave a residual component along v_j itself).
        basis_cpu.append(_to_cpu(v))
        for vb_cpu in basis_cpu:
            vb = _to_device(vb_cpu, device, v_dtype)
            coeff = _vec_dot(w, vb)
            _vec_axpy(w, -coeff, vb)
            del vb

        beta_j = _vec_norm(w)
        if beta_j < 1e-10:
            print(f"  Lanczos: basis collapsed at step {j+1} (beta≈0) — "
                  f"stopping with {j+1} Krylov vectors.")
            break
        betas.append(beta_j)

        v_prev, beta_prev = v, beta_j
        v = [(wi / beta_j).to(v_dtype) for wi in w]

    m = len(alphas)
    if m == 0:
        return np.array([])
    T = np.zeros((m, m))
    for i in range(m):
        T[i, i] = alphas[i]
        if i < m - 1:
            T[i, i + 1] = T[i + 1, i] = betas[i]
    eigvals = np.linalg.eigvalsh(T)
    return eigvals


def top_eigenvalue(model, names, inp, tgt, n_iter, tol, use_fd=False, eps=1e-4):
    """
    Extremal eigenvalues (most negative + largest positive) of the Hessian
    restricted to `names`, via Lanczos (see lanczos_spectrum and the module
    comment above for why this replaced power iteration).

    use_fd=True (large models, bf16): runs Lanczos twice — at eps and 2*eps —
    and requires the resulting top positive eigenvalue to agree within `tol`
    (relative) before trusting it. This is a direct empirical check that the
    measurement isn't dominated by bf16 finite-difference noise, rather than
    trusting Lanczos's own internal machinery alone — exactly the kind of
    check that would have caught Llama/Mistral's oscillating, non-convergent
    old power-iteration runs before reporting a number from them.

    Returns (lambda_max_positive, lambda_min, n_iters, stable). stable=False
    means the eps-agreement check failed (use_fd only); lambda_max_positive
    is NaN in that case rather than a number that looked plausible but wasn't
    actually verified.
    """
    param_list = [p for n, p in model.named_parameters() if n in names]
    if not param_list:
        raise ValueError("No parameters match target names — check model loading.")
    for p in param_list:
        p.requires_grad_(True)

    # Fixed starting vector shared by every run() call below (in particular
    # both the eps and 2*eps stability-check runs) — see lanczos_spectrum's
    # v0 docstring for why: without this, the two compared runs would each
    # start from an independently random Krylov vector, confounding the
    # stability check's verdict with ordinary Lanczos-restart variance on
    # top of whatever eps-driven noise it's meant to isolate.
    v_dtype = next(iter(param_list)).dtype
    v0 = [torch.randn_like(p, dtype=v_dtype) for p in param_list]

    def run(eps_val):
        if use_fd:
            fn = lambda m, pl, i, t, v: hvp_fd(m, pl, i, t, v, eps=eps_val)
        else:
            fn = hvp
        eigvals = lanczos_spectrum(model, param_list, inp, tgt, n_iter, fn, v0=v0)
        if len(eigvals) == 0:
            return float("nan"), float("nan")
        lam_min = float(eigvals[0])
        positive = eigvals[eigvals > 0]
        lam_max_pos = float(positive[-1]) if len(positive) else 0.0
        return lam_min, lam_max_pos

    print(f"  Lanczos ({'FD' if use_fd else 'double-backward'}, {n_iter} steps)...")
    lam_min, lam_max_pos = run(eps)
    print(f"  λ_min={lam_min:.4f}  λ_max^+={lam_max_pos:.6g}")

    stable = True
    if use_fd:
        lam_min2, lam_max_pos2 = run(2 * eps)
        print(f"  Stability check (2x eps): λ_min={lam_min2:.4f}  "
              f"λ_max^+={lam_max_pos2:.6g}")
        denom = max(abs(lam_max_pos), abs(lam_max_pos2), 1e-12)
        rel = abs(lam_max_pos - lam_max_pos2) / denom
        stable = rel < tol
        if not stable:
            print(f"  WARNING: λ_max^+ changed {rel*100:.0f}% when eps doubled "
                  f"(threshold {tol*100:.0f}%) — measurement is eps-sensitive, "
                  f"likely dominated by bf16 compute noise rather than real "
                  f"curvature signal. NOT reporting a number for this model; "
                  f"try a larger --hessian_batch or accept this as inconclusive.")

    if not stable:
        return float("nan"), float("nan"), n_iter, False
    return lam_max_pos, lam_min, n_iter, True


# ── Figure ────────────────────────────────────────────────────────────────────
def make_figure(results, out_dir):
    """
    Two-panel scatter: σ½ vs λ_max.
    Left  — log-log (theory: slope = -0.5 if σ½ ∝ 1/√λ_max)
    Right — σ½ vs 1/√λ_max linear (theory: straight line through origin)

    2026-07-20: no longer takes a single shared eval_slack. GPT-2/Llama/Mistral
    each have their own eval_slack_abs (relslack: 1% of their own baseline NLL,
    which differs ~1.6x across the three) — drawing one theory curve from a
    single slack value would mix them incorrectly, the same class of bug the
    relslack correction fixed for σ½ itself. Each result already carries its
    own correctly-computed theory_sigma_half (see main()); plotted per-model
    as reference points instead of one shared line. The empirical fit (slope,
    R²) is unaffected either way — it doesn't depend on slack.
    """
    # Drop results with invalid λ_max (≤ 0, NaN, inf) — power iteration
    # can return negative values if the Hessian restricted to target params
    # is not PSD at that checkpoint (e.g. model not at a true local minimum).
    valid   = [r for r in results
               if np.isfinite(r["lambda_max"]) and r["lambda_max"] > 0
               and np.isfinite(r["sigma_half"]) and r["sigma_half"] > 0]
    if len(valid) < len(results):
        dropped = [r["model"] for r in results if r not in valid]
        print(f"  [warn] Dropping {dropped} from plot — invalid λ_max")
    if len(valid) < 2:
        print("  Not enough valid points to plot. Check λ_max values in results.json.")
        return

    models  = [r["model"]       for r in valid]
    sh      = np.array([r["sigma_half"]  for r in valid])
    lam     = np.array([r["lambda_max"]  for r in valid])
    colors  = [MODEL_COLOR.get(m, "#6B7280") for m in models]
    labels  = [m.split("/")[-1] for m in models]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # ── Panel 1: log-log ─────────────────────────────────────────────────────
    ax1.scatter(lam, sh, c=colors, s=140, zorder=5,
                edgecolors="white", linewidths=0.8)
    for x, y, lab in zip(lam, sh, labels):
        ax1.annotate(lab, (x, y), textcoords="offset points",
                     xytext=(7, 4), fontsize=8.5)

    x_lo, x_hi = lam.min() * 0.7, lam.max() * 1.4
    x_fit = np.linspace(x_lo, x_hi, 100)

    # Each model's own theoretically-predicted σ½ (√(2·its own slack/λ), from
    # main()) plotted as a per-model reference point rather than one shared
    # curve, since slack differs by model.
    theory_sh_pts = np.array([r["theory_sigma_half"] for r in valid])
    ax1.scatter(lam, theory_sh_pts, c=colors, s=70, marker="x", zorder=6,
                label="theory (own eval_slack)" if len(valid) else None)

    if len(lam) >= 2:
        log_lam, log_sh = np.log(lam), np.log(sh)
        slope, intercept = np.polyfit(log_lam, log_sh, 1)
        y_fit = np.exp(intercept) * x_fit ** slope
        r2 = np.corrcoef(log_lam, log_sh)[0, 1] ** 2
        ax1.plot(x_fit, y_fit, "k--", lw=1.5,
                 label=f"fit: slope={slope:.2f}  R²={r2:.3f}\n(theory slope = −0.50)")

    ax1.set_xscale("log")
    ax1.set_yscale("log")
    ax1.set_xlabel("λ_max  (top Hessian eigenvalue)", fontsize=10)
    ax1.set_ylabel("σ½  (certified density half-max)", fontsize=10)
    ax1.set_title("σ½ vs λ_max  (log-log)\nSlope ≈ −0.5 if σ½ ∝ 1/√λ_max", fontsize=10)
    ax1.legend(fontsize=8.5)
    ax1.grid(True, alpha=0.3, which="both")

    # ── Panel 2: σ½ vs 1/√λ (linear, should be straight line through origin) ─
    inv_sqrt_lam = 1.0 / np.sqrt(lam)
    ax2.scatter(inv_sqrt_lam, sh, c=colors, s=140, zorder=5,
                edgecolors="white", linewidths=0.8)
    for x, y, lab in zip(inv_sqrt_lam, sh, labels):
        ax2.annotate(lab, (x, y), textcoords="offset points",
                     xytext=(7, 4), fontsize=8.5)

    x2_fit = np.linspace(0, inv_sqrt_lam.max() * 1.3, 100)
    # Per-model theory points again (not one shared-slope line — see docstring).
    ax2.scatter(1.0 / np.sqrt(lam), theory_sh_pts, c=colors, s=70, marker="x",
                zorder=6, label="theory (own eval_slack)")

    if len(lam) >= 2:
        coeffs = np.polyfit(inv_sqrt_lam, sh, 1)
        y_fit2 = coeffs[0] * x2_fit + coeffs[1]
        r2_lin = np.corrcoef(inv_sqrt_lam, sh)[0, 1] ** 2
        ax2.plot(x2_fit, y_fit2, "k--", lw=1.5,
                 label=f"fit: slope={coeffs[0]:.4f}  intercept={coeffs[1]:.4f}"
                       f"\nR²={r2_lin:.3f}")
        # Force-through-origin fit
        slope0 = float(np.dot(inv_sqrt_lam, sh) / np.dot(inv_sqrt_lam, inv_sqrt_lam))
        ax2.plot(x2_fit, slope0 * x2_fit, color="#16A34A", lw=1.5, ls="--",
                 label=f"origin fit: slope={slope0:.4f}")

    ax2.set_xlim(left=0)
    ax2.set_ylim(bottom=0)
    ax2.set_xlabel("1 / √λ_max", fontsize=10)
    ax2.set_ylabel("σ½", fontsize=10)
    ax2.set_title("σ½ vs 1/√λ_max  (linear)\nLinear through origin = quadratic landscape",
                  fontsize=10)
    ax2.legend(fontsize=8)
    ax2.grid(True, alpha=0.3)

    fig.suptitle(
        "Q1: Certified Sharpness — σ½ vs Top Hessian Eigenvalue",
        fontsize=12)
    fig.tight_layout()

    path = os.path.join(out_dir, "q1_sigma_vs_hessian.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved figure: {path}")


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    args   = get_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = os.path.join(BASE, args.out)
    os.makedirs(out_dir, exist_ok=True)
    print(f"Device: {device}  |  Output: {out_dir}\n")

    # Load existing results for incremental runs
    results_path = os.path.join(out_dir, "results.json")
    results = []
    done_models = set()
    if os.path.exists(results_path) and not args.force:
        with open(results_path) as f:
            results = json.load(f)
        done_models = {r["model"] for r in results}
        if done_models:
            print(f"Resuming — already computed: {sorted(done_models)}\n")

    for model_name in args.models:
        if model_name in done_models:
            print(f"Skipping {model_name} (already in results.json). "
                  f"Use --force to recompute.\n")
            continue

        print(f"{'='*60}")
        print(f"Model: {model_name}")
        print(f"{'='*60}")

        # Load σ½ (and its own eval_slack_abs) from prior experiments
        sh, sh_std, model_eval_slack = load_sigma_half(model_name)
        if sh is None:
            print(f"  No σ½ data found — skipping.\n")
            continue
        if model_eval_slack is None:
            if args.eval_slack is None:
                print(f"  No eval_slack_abs in density JSON and no --eval_slack "
                      f"fallback given — skipping.\n")
                continue
            print(f"  [warn] density JSON has no eval_slack_abs; using --eval_slack "
                  f"fallback ({args.eval_slack}) instead.")
            model_eval_slack = args.eval_slack

        # Load pretrained model
        print(f"\nLoading {model_name}...")
        model, _ = _load_model_auto(model_name, device)

        # Large models (>2B params) use finite-difference HVP to avoid the
        # create_graph=True backward graph, which exceeds 44 GB on Llama-3B.
        # FD only needs two gradient passes (no graph), so bf16 is safe.
        # Small models use double-backward in fp32 (no flash-attention double-bwd).
        n_total = sum(p.numel() for p in model.parameters())
        use_fd = n_total > 2_000_000_000
        if use_fd:
            # Keep bf16 to save ~6 GB; FD gradients accumulate in fp32 internally
            print(f"  dtype: bf16 ({n_total/1e6:.0f}M params, using FD-HVP)")
            # LoRA-scoped target params are spread across every layer, so
            # backward still needs a gradient path through the WHOLE network
            # depth (same activation cost as full fine-tuning) even though
            # only a fraction of parameters are targeted. Confirmed 2026-07-2X:
            # Llama-3B OOM'd on its very first HVP call once the drift-fix's
            # extra orig-snapshot buffer (~1.4GB) was added to an already
            # ~100%-saturated 44GB budget (the earlier Mistral OOM showed the
            # same near-total saturation). Gradient checkpointing trades
            # recompute for activation memory (typically 2-5x reduction) and
            # is the standard fix for exactly this shape of problem, rather
            # than reverting the correctness fix. use_cache=False is required
            # alongside it — the KV cache the model would otherwise try to
            # build/retain is irrelevant here and wastes memory.
            if hasattr(model, "gradient_checkpointing_enable"):
                # use_reentrant=False (non-reentrant checkpointing) — the
                # older reentrant default has a known gradient-flow gotcha in
                # exactly our scenario: only a scattered subset of parameters
                # (LoRA-target attention projections) has requires_grad=True,
                # while the hidden-state activations flowing between layers
                # and the surrounding MLP/layernorm weights don't. Reentrant
                # mode's recompute-and-backward can fail to reconnect
                # gradients or stall when a checkpointed segment's own direct
                # inputs lack a requires_grad=True tensor, unlike normal eager
                # autograd (or non-reentrant checkpointing) which tracks
                # requires_grad transitively through the whole graph. This is
                # PyTorch's own current recommendation, not just a guess.
                model.gradient_checkpointing_enable(
                    gradient_checkpointing_kwargs={"use_reentrant": False})
            model.config.use_cache = False
            # HF's decoder layers gate checkpointing on `self.training` (most
            # implementations use `if self.gradient_checkpointing and self.training`)
            # — model.eval() below would silently make the enable() call above a
            # no-op, with no error, just no memory savings. Use train() instead,
            # but explicitly zero every Dropout module's probability so the
            # forward pass stays deterministic (this measurement needs a frozen,
            # non-stochastic model either way) rather than relying on the
            # assumption that these particular checkpoints' configs already
            # default attention_dropout/hidden_dropout to 0.
            model.train()
            for m in model.modules():
                if isinstance(m, torch.nn.Dropout):
                    m.p = 0.0
        else:
            model = model.float()
            print(f"  dtype: fp32 ({n_total/1e6:.0f}M params, using double-backward HVP)")
            model.eval()

        tokenizer = _get_tokenizer(model_name)
        inp, tgt  = load_data(tokenizer, args.n_text_samples, args.seq_len, device)

        # Small batch for HVP — create_graph=True stores the full backward graph,
        # so memory scales with batch size. 4 examples is enough for stable
        # power iteration; using the full density batch (128) causes OOM on Llama-3B.
        hvp_inp = inp[:args.hessian_batch]
        hvp_tgt = tgt[:args.hessian_batch]

        # LoRA-scoped, not the whole decoder block — see _get_lora_scoped_names
        # docstring: fixes the Mistral OOM (whole-block fp32 grad buffers on
        # ~7B params) and tests curvature in the subspace fine-tuning actually
        # moves in, not one dominated by parameters LoRA never touches.
        names     = _get_lora_scoped_names(model, model_name)

        n_target = sum(p.numel() for n, p in model.named_parameters() if n in names)
        n_total  = sum(p.numel() for p in model.parameters())
        print(f"  Target params (LoRA-scoped): {n_target:,} / {n_total:,}  "
              f"({100 * n_target / n_total:.1f}%)", flush=True)

        # Compute extremal eigenvalues via Lanczos
        print(f"\nLanczos ({args.n_iter} steps, hessian_batch={args.hessian_batch}, "
              f"method={'FD' if use_fd else 'double-backward'}"
              f"{f', eps={args.eps}, stability tol={args.tol}' if use_fd else ''})...")
        lam_max, lam_min, n_iters, stable = top_eigenvalue(
            model, names, hvp_inp, hvp_tgt, args.n_iter, args.tol,
            use_fd=use_fd, eps=args.eps)

        if not stable:
            print(f"  Skipping σ½/theory comparison for {model_name} — "
                  f"eps-stability check failed (see WARNING above), not a "
                  f"trustworthy λ_max.\n")
            result = {
                "model":               model_name,
                "n_params":            MODEL_N_PARAMS.get(model_name, n_total),
                "sigma_half":          sh,
                "sigma_half_std":      sh_std,
                "lambda_max":          float("nan"),
                "lambda_min":          lam_min,
                "inv_sqrt_lambda":     float("nan"),
                "theory_sigma_half":   float("nan"),
                "ratio_meas_theory":   float("nan"),
                "n_iters_converged":   n_iters,
                "eval_slack":          model_eval_slack,
                "n_text_samples":      args.n_text_samples,
                "stable":              False,
            }
            results.append(result)
            with open(results_path, "w") as f:
                json.dump(results, f, indent=2)
            del model
            if device.type == "cuda":
                torch.cuda.empty_cache()
            continue

        if lam_max <= 0:
            print(f"  WARNING: λ_max = {lam_max:.4f} ≤ 0. "
                  f"No positive eigenvalue found within {args.n_iter} Lanczos "
                  f"steps — the checkpoint may be indefinite/flat throughout "
                  f"this subspace, or more steps are needed.")
            if lam_min < 0:
                print(f"  λ_min = {lam_min:.4f} (confirmed negative — genuine "
                      f"saddle-point curvature in this subspace, independent "
                      f"of whether λ_max^+ was found).")

        # Uses this model's OWN eval_slack (relslack: 1% of its own baseline NLL),
        # not one shared value — GPT-2/Llama/Mistral baseline NLL differ ~1.6x,
        # so a single global slack here would reintroduce the same mismatched-
        # eval_slack bug the relslack correction fixed for σ½ itself.
        theory_sh = np.sqrt(2 * model_eval_slack / lam_max) if lam_max > 0 else float("nan")
        ratio     = sh / theory_sh if theory_sh > 0 else float("nan")

        print(f"\n  σ½ (measured)       = {sh:.6f}  (std={sh_std:.6f})")
        print(f"  λ_min               = {lam_min:.4f}"
              f"{'  (confirmed saddle point)' if lam_min < 0 else ''}")
        print(f"  λ_max^+             = {lam_max:.4f}")
        if lam_max > 0:
            print(f"  1/√λ_max            = {1/np.sqrt(lam_max):.6f}")
            print(f"  σ½ (theory)         = √(2·slack/λ) = {theory_sh:.6f}")
            print(f"  measured / theory   = {ratio:.3f}  "
                  f"(1.0 = perfectly quadratic landscape)")

        result = {
            "model":               model_name,
            "n_params":            MODEL_N_PARAMS.get(model_name, n_total),
            "sigma_half":          sh,
            "sigma_half_std":      sh_std,
            "lambda_max":          lam_max,
            "lambda_min":          lam_min,
            "inv_sqrt_lambda":     1.0 / np.sqrt(lam_max) if lam_max > 0 else float("nan"),
            "theory_sigma_half":   theory_sh,
            "ratio_meas_theory":   ratio,
            "n_iters_converged":   n_iters,
            "eval_slack":          model_eval_slack,
            "n_text_samples":      args.n_text_samples,
            "stable":              True,
        }
        results.append(result)

        # Save incrementally so partial runs are preserved
        with open(results_path, "w") as f:
            json.dump(results, f, indent=2)
        print(f"  Saved: {results_path}\n")

        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    # Summary table
    print(f"\n{'='*90}")
    print(f"  {'Model':32s}  {'σ½':>10}  {'λ_min':>9}  {'λ_max^+':>10}  "
          f"{'1/√λ':>10}  {'meas/theory':>12}  {'stable':>7}")
    print(f"  {'-'*85}")
    for r in results:
        print(f"  {r['model']:32s}  {r['sigma_half']:10.6f}  "
              f"{r.get('lambda_min', float('nan')):9.4f}  {r['lambda_max']:10.4f}  "
              f"{r['inv_sqrt_lambda']:10.6f}  {r['ratio_meas_theory']:12.3f}  "
              f"{str(r.get('stable', True)):>7}")

    if len(results) >= 2:
        make_figure(results, out_dir)

    print(f"\nDone. Results in: {out_dir}")


if __name__ == "__main__":
    main()
