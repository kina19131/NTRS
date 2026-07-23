#!/bin/bash
# run_basin_widening.sh
# =====================
# Full experimental sweep for the basin-widening claim.
#
# What this runs:
#   Phase 1: GPT-2 (124M) — rank ablation × 2 tasks × 1 step count  (~3 hrs A40)
#   Phase 2: GPT-2 (124M) — steps ablation × 1 task (SST-2)          (~2 hrs A40)
#   Phase 3: GPT-2-medium (354M) — rank ablation × SST-2             (~3 hrs A40)
#
# Total: ~8 hrs on A40. Run phases independently if time is limited.
# Priority order: Phase 1 SST-2 → Phase 1 WikiText-2 → Phase 2 → Phase 3
#
# Usage:
#   bash run_basin_widening.sh            # all phases
#   bash run_basin_widening.sh phase1     # GPT-2 rank ablation only
#   bash run_basin_widening.sh phase2     # steps ablation only
#   bash run_basin_widening.sh phase3     # GPT-2-medium only

set -e
PHASE=${1:-all}

RANKS="1 2 4 8 16 32 64"
SIGMAS="0.0001 0.0002 0.0005 0.001 0.002 0.005 0.01 0.02 0.05 0.1 0.2"
OUT="./results/basin_widening"

echo "============================================================"
echo "Basin Widening Experiment  |  $(date)"
echo "============================================================"

# ── Phase 1: GPT-2 rank ablation ─────────────────────────────────────────────
if [[ "$PHASE" == "all" || "$PHASE" == "phase1" ]]; then

  echo ""
  echo "--- Phase 1A: GPT-2 × SST-2 rank ablation ---"
  python basin_widening_experiment.py \
    --model gpt2 \
    --task  sst2 \
    --ranks $RANKS \
    --lr    1e-4 \
    --train_steps 500 \
    --train_batch 16 \
    --n_perturb   200 \
    --sigmas $SIGMAS \
    --output_dir "$OUT"

  echo ""
  echo "--- Phase 1B: GPT-2 × WikiText-2 rank ablation ---"
  python basin_widening_experiment.py \
    --model gpt2 \
    --task  wikitext2 \
    --ranks $RANKS \
    --lr    1e-4 \
    --train_steps 500 \
    --train_batch 16 \
    --n_perturb   200 \
    --sigmas $SIGMAS \
    --output_dir "$OUT"

fi

# ── Phase 2: GPT-2 steps ablation ────────────────────────────────────────────
# Fixes rank=8 and sweeps training duration.
# Tests: does widening persist as training goes on, or does it peak and reverse?
if [[ "$PHASE" == "all" || "$PHASE" == "phase2" ]]; then

  echo ""
  echo "--- Phase 2: GPT-2 × SST-2 steps ablation (rank=8) ---"
  # Steps go up to 5000 to test whether widening reverses once norm/σ½ > 1.
  # At lr=1e-4, norm ≈ 0.18×σ½ at 500 steps → ~1.8×σ½ at 5000 steps (collapse zone).
  # If σ½ peaks around 1000-2000 steps and then reverses, that closes the loop
  # with the phase transition: widening only holds while norm < σ½.
  python basin_widening_experiment.py \
    --model gpt2 \
    --task  sst2 \
    --ranks 8 \
    --lr    1e-4 \
    --train_steps 100 500 1000 2000 5000 \
    --train_batch 16 \
    --n_perturb   200 \
    --sigmas $SIGMAS \
    --output_dir "$OUT"

fi

# ── Phase 3: GPT-2-medium rank ablation ──────────────────────────────────────
# Confirms the rank→widening relationship holds at 354M parameters.
# If it does, the claim is architecture-size-robust.
if [[ "$PHASE" == "all" || "$PHASE" == "phase3" ]]; then

  echo ""
  echo "--- Phase 3: GPT-2-medium × SST-2 rank ablation ---"
  python basin_widening_experiment.py \
    --model gpt2-medium \
    --task  sst2 \
    --ranks $RANKS \
    --lr    1e-4 \
    --train_steps 500 \
    --train_batch 8 \
    --n_perturb   200 \
    --sigmas $SIGMAS \
    --output_dir "$OUT"

  echo ""
  echo "--- Phase 3B: GPT-2-medium × WikiText-2 rank ablation ---"
  python basin_widening_experiment.py \
    --model gpt2-medium \
    --task  wikitext2 \
    --ranks $RANKS \
    --lr    1e-4 \
    --train_steps 500 \
    --train_batch 8 \
    --n_perturb   200 \
    --sigmas $SIGMAS \
    --output_dir "$OUT"

fi

# ── Phase 4: Llama-3.2-1B rank ablation ──────────────────────────────────────
# Modern architecture check. r=64 excluded (hidden_dim=2048, r=64 is 3% — fine,
# but adds time; include it). Uses --finetune_fp32 for stability at lr=1e-4.
if [[ "$PHASE" == "all" || "$PHASE" == "phase4" ]]; then

  echo ""
  echo "--- Phase 4A: Llama-3.2-1B × SST-2 rank ablation ---"
  python basin_widening_experiment.py \
    --model meta-llama/Llama-3.2-1B \
    --task  sst2 \
    --ranks 1 2 4 8 16 32 64 \
    --lr    1e-4 \
    --train_steps 500 \
    --train_batch 4 \
    --n_perturb   200 \
    --sigmas $SIGMAS \
    --finetune_fp32 \
    --output_dir "$OUT"

  echo ""
  echo "--- Phase 4B: Llama-3.2-1B × WikiText-2 rank ablation ---"
  python basin_widening_experiment.py \
    --model meta-llama/Llama-3.2-1B \
    --task  wikitext2 \
    --ranks 1 2 4 8 16 32 64 \
    --lr    1e-4 \
    --train_steps 500 \
    --train_batch 4 \
    --n_perturb   200 \
    --sigmas $SIGMAS \
    --finetune_fp32 \
    --output_dir "$OUT"

fi

echo ""
echo "============================================================"
echo "All phases complete. Results in: $OUT"
echo "============================================================"
