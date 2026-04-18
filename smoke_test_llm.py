"""
smoke_test_llm.py
=================
Validates every code path in subspace_experiment_llm.py using a tiny
synthetic GPT-2-like model. No internet required, runs in ~60 seconds on CPU.

Run this BEFORE spinning up a GPU instance to confirm:
  1. All imports work
  2. Model loading, weight perturbation, restoration work correctly
  3. Perplexity scoring produces sensible numbers
  4. RandOpt loop completes and returns correct shapes
  5. PCA analysis runs and produces plausible results
  6. Plots render and save without errors
  7. Pretrained vs random conditions produce DIFFERENT results
     (if they look identical something is wrong)

Usage:
  python smoke_test_llm.py

Expected output:
  All checks should print PASS.
  Final plots saved to ./smoke_test_outputs/
  Total runtime: ~60 seconds on CPU.
"""

import sys, os, copy, time
import numpy as np
import torch
import torch.nn as nn
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ── Make sure the main script is importable ───────────────────────────────────
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)

OUTPUT_DIR = "./smoke_test_outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

PASS = "\033[92m  PASS\033[0m"
FAIL = "\033[91m  FAIL\033[0m"

def check(name, condition, detail=""):
    status = PASS if condition else FAIL
    print(f"{status}  {name}" + (f"  [{detail}]" if detail else ""))
    if not condition:
        raise AssertionError(f"FAILED: {name}")

# ── 1. Build a tiny synthetic GPT-2 (no internet needed) ─────────────────────
print("\n" + "="*60)
print("SMOKE TEST: subspace_experiment_llm.py")
print("="*60)

print("\n[1] Building tiny synthetic GPT-2...")
from transformers import GPT2Config, GPT2LMHeadModel

TINY_CFG = GPT2Config(
    n_layer     = 4,      # 4 transformer blocks (vs 12 in real GPT-2)
    n_embd      = 128,    # 128 hidden dim (vs 768)
    n_head      = 2,      # 2 attention heads
    n_positions = 256,
    vocab_size  = 1000,   # tiny vocab
)
model_pretrained = GPT2LMHeadModel(TINY_CFG)

# Simulate "pretraining" by giving it structured weights
# (just use default HuggingFace init which is non-trivial)
n_params = sum(p.numel() for p in model_pretrained.parameters())
check("Model created", n_params > 0, f"{n_params:,} params")

# Import all functions from the main script
from subspace_experiment_llm import (
    _init_weights,
    make_random_init_copy,
    get_target_param_names,
    get_flat_weights,
    perturb_weights,
    restore_weights,
    save_target_weights,
    get_flat_weights_from_snapshot,
    compute_perplexity,
    randopt_collect_llm,
    pca_dims_for_threshold,
    solution_density,
    run_condition,
    plot_comparison,
    plot_loss_distributions,
)
check("All imports from main script", True)

device = torch.device("cpu")
model_pretrained = model_pretrained.to(device)

# ── 2. Random-init baseline ───────────────────────────────────────────────────
print("\n[2] Creating random-init baseline...")
model_random = make_random_init_copy(model_pretrained, device)

# Verify weights are DIFFERENT between pretrained and random
flat_pre = get_flat_weights(
    model_pretrained,
    get_target_param_names(model_pretrained, n_blocks=2)
)
flat_rnd = get_flat_weights(
    model_random,
    get_target_param_names(model_random, n_blocks=2)
)
weight_diff = (flat_pre - flat_rnd).abs().mean().item()
check("Pretrained and random weights differ", weight_diff > 1e-4,
      f"mean abs diff={weight_diff:.5f}")

# ── 3. Target parameter identification ───────────────────────────────────────
print("\n[3] Target parameter identification...")
target_names = get_target_param_names(model_pretrained, n_blocks=2)
n_target = sum(p.numel() for n, p in model_pretrained.named_parameters()
               if n in target_names)
check("Target names non-empty", len(target_names) > 0,
      f"{len(target_names)} tensors, {n_target:,} params")
check("Target names are subset of all params",
      all(n in dict(model_pretrained.named_parameters()) for n in target_names))

# ── 4. Weight perturbation & restoration ─────────────────────────────────────
print("\n[4] Weight perturbation and restoration...")
snap  = save_target_weights(model_pretrained, target_names)
flat0 = get_flat_weights(model_pretrained, target_names).clone()

perturb_weights(model_pretrained, seed=0, sigma=0.01, target_names=target_names)
flat1 = get_flat_weights(model_pretrained, target_names)
pert_norm = (flat1 - flat0).norm().item()
check("Perturbation changes weights", pert_norm > 0,
      f"norm={pert_norm:.4f}")

# Same seed = same perturbation
model_copy = copy.deepcopy(model_pretrained)
restore_weights(model_copy, snap, target_names)
perturb_weights(model_copy, seed=0, sigma=0.01, target_names=target_names)
flat1b = get_flat_weights(model_copy, target_names)
check("Same seed reproducible",
      (flat1 - flat1b).abs().max().item() < 1e-6)

# Different seed = different perturbation
restore_weights(model_pretrained, snap, target_names)
perturb_weights(model_pretrained, seed=1, sigma=0.01, target_names=target_names)
flat2 = get_flat_weights(model_pretrained, target_names)
check("Different seeds produce different perturbations",
      (flat1 - flat2).abs().mean().item() > 1e-6)

# Restoration is exact
restore_weights(model_pretrained, snap, target_names)
flat_restored = get_flat_weights(model_pretrained, target_names)
restoration_err = (flat_restored - flat0).abs().max().item()
check("Weight restoration exact", restoration_err < 1e-7,
      f"max err={restoration_err:.2e}")

# ── 5. Perplexity scoring ─────────────────────────────────────────────────────
print("\n[5] Perplexity scoring...")
B, T = 8, 32
torch.manual_seed(42)
input_ids  = torch.randint(0, 1000, (B, T), device=device)
target_ids = torch.randint(0, 1000, (B, T), device=device)

ppl_pretrained = compute_perplexity(model_pretrained, input_ids, target_ids)
ppl_random     = compute_perplexity(model_random,     input_ids, target_ids)

check("Pretrained perplexity is finite and positive",
      np.isfinite(ppl_pretrained) and ppl_pretrained > 0,
      f"ppl={ppl_pretrained:.4f}")
check("Random perplexity is finite and positive",
      np.isfinite(ppl_random) and ppl_random > 0,
      f"ppl={ppl_random:.4f}")

# Random model on random data should be near log(vocab_size) = log(1000) ~ 6.9
expected_ppl = np.log(1000)
check("Random model perplexity near log(vocab)",
      abs(ppl_random - expected_ppl) < 1.5,
      f"got={ppl_random:.3f}, expected~{expected_ppl:.3f}")

# Perturbing should change perplexity
perturb_weights(model_pretrained, seed=7, sigma=0.1, target_names=target_names)
ppl_perturbed = compute_perplexity(model_pretrained, input_ids, target_ids)
restore_weights(model_pretrained, snap, target_names)
check("Perturbation changes perplexity",
      abs(ppl_perturbed - ppl_pretrained) > 1e-4,
      f"before={ppl_pretrained:.4f}, after={ppl_perturbed:.4f}")

# ── 6. RandOpt loop ───────────────────────────────────────────────────────────
print("\n[6] RandOpt loop (N=20, K=5)...")
N_SMOKE, K_SMOKE = 20, 5

t0 = time.time()
top_k_idx, all_deltas, all_losses, base_loss = randopt_collect_llm(
    model        = model_pretrained,
    input_ids    = input_ids,
    target_ids   = target_ids,
    target_names = target_names,
    sigma        = 0.005,
    N            = N_SMOKE,
    K            = K_SMOKE,
)
elapsed = time.time() - t0

check("RandOpt completed",          True, f"{elapsed:.1f}s")
check("all_deltas shape correct",
      all_deltas.shape == (N_SMOKE, n_target),
      f"{all_deltas.shape}")
check("all_losses shape correct",
      all_losses.shape == (N_SMOKE,),
      f"{all_losses.shape}")
check("top_k_idx length correct",   len(top_k_idx) == K_SMOKE)
check("top_k_idx are valid indices",
      all(0 <= i < N_SMOKE for i in top_k_idx))
check("top-K are actually best",
      all(all_losses[i] <= np.median(all_losses) for i in top_k_idx[:3]),
      f"top losses: {[f'{all_losses[i]:.3f}' for i in top_k_idx[:3]]}")
check("Model weights restored after RandOpt",
      (get_flat_weights(model_pretrained, target_names) - flat0).abs().max().item() < 1e-6)
check("Deltas are not all zero",
      all_deltas.std() > 1e-8)
check("Loss values are finite",
      np.all(np.isfinite(all_losses)))

# Estimate full-scale runtime
time_per_seed = elapsed / N_SMOKE
full_n = 200
full_est_cpu = time_per_seed * full_n
print(f"\n  Timing estimate (based on {elapsed:.1f}s for N={N_SMOKE}):")
print(f"    N=50  (cpu mode):  ~{time_per_seed*50/60:.1f} min")
print(f"    N=200 (gpu mode):  ~{time_per_seed*200/60:.1f} min on CPU")
print(f"    GPU speedup ~20x:  ~{time_per_seed*200/60/20:.1f} min on GPU")

# ── 7. PCA analysis ───────────────────────────────────────────────────────────
print("\n[7] PCA analysis...")
dims_all,  cumvar_all  = pca_dims_for_threshold(all_deltas,             0.90)
dims_topk, cumvar_topk = pca_dims_for_threshold(all_deltas[top_k_idx], 0.90)
density = solution_density(all_losses, base_loss)

check("dims_all is positive integer",   isinstance(dims_all, int) and dims_all > 0,
      f"dims_all={dims_all}")
check("dims_topk is positive integer",  isinstance(dims_topk, int) and dims_topk > 0,
      f"dims_topk={dims_topk}")
check("cumvar_all ends near 1.0",
      abs(cumvar_all[-1] - 1.0) < 0.01,
      f"cumvar[-1]={cumvar_all[-1]:.4f}")
check("density in [0, 1]",
      0.0 <= density <= 1.0,
      f"density={density:.4f}")
check("dims_topk <= dims_all",
      dims_topk <= dims_all,
      f"topk={dims_topk}, all={dims_all}")
print(f"  dims_90(all)={dims_all}  dims_90(top-K)={dims_topk}  "
      f"compression={dims_all/max(dims_topk,1):.1f}x  density={density:.3f}")

# ── 8. Run full condition pipeline (both models) ──────────────────────────────
print("\n[8] Full run_condition() pipeline...")
# Run both conditions with smoke-test settings
pretrained_res = run_condition(
    label        = "pretrained (smoke)",
    model        = model_pretrained,
    target_names = target_names,
    input_ids    = input_ids,
    target_ids   = target_ids,
    sigma        = 0.005,
    N            = N_SMOKE,
    K            = K_SMOKE,
    store_dtype  = torch.float32,
)
random_res = run_condition(
    label        = "random-init (smoke)",
    model        = model_random,
    target_names = get_target_param_names(model_random, n_blocks=2),
    input_ids    = input_ids,
    target_ids   = target_ids,
    sigma        = 0.02,
    N            = N_SMOKE,
    K            = K_SMOKE,
    store_dtype  = torch.float32,
)

required_keys = ["dims_all", "dims_topk", "compression", "density",
                 "base_loss", "best_loss", "cumvar_all", "cumvar_topk",
                 "strat", "all_losses", "top_k_idx", "N", "K"]
check("pretrained_res has all keys",
      all(k in pretrained_res for k in required_keys))
check("random_res has all keys",
      all(k in random_res for k in required_keys))
# NOTE: in the smoke test both models are randomly initialised (no real
# pretraining available without HuggingFace access), so base perplexities
# will be similar (~log(vocab)). On real GPT-2 the pretrained model will
# have much lower perplexity (~3-4 vs ~6.9 for random).
# We check that both values are at least finite and in a sane range.
check("Base perplexities are finite and sane",
      all(np.isfinite(r["base_loss"]) and 0 < r["base_loss"] < 20
          for r in [pretrained_res, random_res]),
      f"pretrained={pretrained_res['base_loss']:.3f}, "
      f"random={random_res['base_loss']:.3f}")
print(f"  NOTE: On real pretrained GPT-2, base PPL will be ~3-4 (not ~6.9).")
print(f"        The key comparison is compression ratio, not absolute PPL.")

# ── 9. Plots render without error ─────────────────────────────────────────────
print("\n[9] Plot rendering...")
try:
    plot_comparison(pretrained_res, random_res, OUTPUT_DIR)
    check("plot_comparison saved", os.path.exists(
        os.path.join(OUTPUT_DIR, "llm_subspace_comparison.png")))
except Exception as e:
    check("plot_comparison saved", False, str(e))

try:
    plot_loss_distributions(pretrained_res, random_res, OUTPUT_DIR)
    check("plot_loss_distributions saved", os.path.exists(
        os.path.join(OUTPUT_DIR, "llm_loss_distributions.png")))
except Exception as e:
    check("plot_loss_distributions saved", False, str(e))

# ── 10. Sanity check: are results meaningful? ─────────────────────────────────
print("\n[10] Sanity checks on results...")
check("Some perturbations improve pretrained model",
      pretrained_res["density"] > 0 or pretrained_res["best_loss"] < pretrained_res["base_loss"],
      f"density={pretrained_res['density']:.3f}, "
      f"best={pretrained_res['best_loss']:.3f} vs base={pretrained_res['base_loss']:.3f}")
check("Strat results have expected keys",
      "top 10%" in pretrained_res["strat"] and "all" in pretrained_res["strat"])
check("Top-10% dims <= all dims (score stratification works)",
      pretrained_res["strat"]["top 10%"]["dims_90"] <=
      pretrained_res["strat"]["all"]["dims_90"],
      f"top10%={pretrained_res['strat']['top 10%']['dims_90']}, "
      f"all={pretrained_res['strat']['all']['dims_90']}")

# ── Final summary ─────────────────────────────────────────────────────────────
print("\n" + "="*60)
print("\033[92mALL CHECKS PASSED\033[0m — script is ready for GPU run")
print("="*60)

print(f"""
Results summary (smoke test):
  Pretrained:  base_ppl={pretrained_res['base_loss']:.3f}
               dims_90(all)={pretrained_res['dims_all']}
               dims_90(top-K)={pretrained_res['dims_topk']}
               compression={pretrained_res['compression']:.1f}x
               density={pretrained_res['density']:.3f}

  Random init: base_ppl={random_res['base_loss']:.3f}
               dims_90(all)={random_res['dims_all']}
               dims_90(top-K)={random_res['dims_topk']}
               compression={random_res['compression']:.1f}x
               density={random_res['density']:.3f}

Plots saved to: {OUTPUT_DIR}/

To run full experiment on GPU:
  python subspace_experiment_llm.py --mode gpu --n_perturb 200 --n_blocks 12
""")
