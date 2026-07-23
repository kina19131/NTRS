# Run Commands — 2× A40 GPU
**Written:** 2026-07-01  
**Working directory for all commands:** `/Users/kinakim/Github/NTRS`

---

## Pre-flight check

All Phase 1 models are already trained and saved. Use `--skip_phase1` everywhere.  
Existing r8+r32 × 5 LR conditions are auto-cached — including them in `--phase2_ranks` is safe and required to get a complete `summary.json`.

```bash
cd /Users/kinakim/Github/NTRS
```

---

## BATCH 1 — Run immediately (both GPUs in parallel)

Open **two terminals** and run one command per terminal.

### GPU 0 — SST-2→MNLI: extend LR sweep to [1e-3, 2e-3, 5e-3]
**Addresses:** V5 (SST-2 one-sided bound). First actual SST-2→MNLI forgetting data.  
**New conditions:** 6 (lr=[1e-3,2e-3,5e-3] × rank=[8,32]). Existing 10 cached, skip instantly.  
**Est. wall clock:** ~2–3h (6 new × ~25 min each)

```bash
CUDA_VISIBLE_DEVICES=0 python h2_sequential_llama.py \
    --task_a sst2 --task_b mnli \
    --skip_phase1 \
    --phase2_lrs 1e-5 5e-5 1e-4 2e-4 5e-4 1e-3 2e-3 5e-3 \
    --phase2_ranks 8 32 \
    --output_dir ./h2_llama \
    --phase2_steps 2000 \
    2>&1 | tee logs/sst2_to_mnli_highlr.log
```

### GPU 1 — MNLI→AGNews: add rank=16
**Addresses:** V4 (rank gap). AGNews is only task where r32 is also rank-invariant — need r16 to characterize.  
**New conditions:** 5 (rank=16 × 5 LRs). Existing 10 cached.  
**Est. wall clock:** ~2h (5 new × ~25 min each)

```bash
CUDA_VISIBLE_DEVICES=1 python h2_sequential_llama.py \
    --task_a mnli --task_b agnews \
    --skip_phase1 \
    --phase2_lrs 1e-5 5e-5 1e-4 2e-4 5e-4 \
    --phase2_ranks 8 16 32 \
    --output_dir ./h2_llama \
    --phase2_steps 2000 \
    2>&1 | tee logs/mnli_to_agnews_r16.log
```

---

## BATCH 2 — Start after Batch 1 finishes (both GPUs in parallel)

### GPU 0 — AGNews→MNLI: add rank=4 and rank=16 (combined)
**Addresses:** V4 for AGNews Phase 1 model in both directions.  
**New conditions:** 10 (rank=[4,16] × 5 LRs). Existing 10 cached.  
**Est. wall clock:** ~4h (10 new × ~25 min each)

```bash
CUDA_VISIBLE_DEVICES=0 python h2_sequential_llama.py \
    --task_a agnews --task_b mnli \
    --skip_phase1 \
    --phase2_lrs 1e-5 5e-5 1e-4 2e-4 5e-4 \
    --phase2_ranks 4 8 16 32 \
    --output_dir ./h2_llama \
    --phase2_steps 2000 \
    2>&1 | tee logs/agnews_to_mnli_r4_r16.log
```

### GPU 1 — MNLI→SST-2: add rank=4
**Addresses:** V4 for MNLI Phase 1 model at low rank.  
**New conditions:** 5 (rank=4 × 5 LRs). Existing 10 cached.  
**Est. wall clock:** ~2h (5 new × ~25 min each)

```bash
CUDA_VISIBLE_DEVICES=1 python h2_sequential_llama.py \
    --task_a mnli --task_b sst2 \
    --skip_phase1 \
    --phase2_lrs 1e-5 5e-5 1e-4 2e-4 5e-4 \
    --phase2_ranks 4 8 32 \
    --output_dir ./h2_llama \
    --phase2_steps 2000 \
    2>&1 | tee logs/mnli_to_sst2_r4.log
```

---

## BATCH 3 — ✅ Complete (2026-07-03)

### GPU 0 — AGNews→SST-2 (r4/r8/r16/r32) THEN H1-MNLI (chained)
**Two experiments chained with `&&` — H1-MNLI starts automatically after AGNews→SST-2 finishes.**

---

#### Part A: AGNews→SST-2 full rank sweep
**Addresses:** V4 (rank coverage) — completes the rank picture for AGNews Phase 1 with SST-2 as Phase 2.  
**Question:** Is AGNews Phase 1 rank-invariant for SST-2 Phase 2, the same way it is for MNLI Phase 2? (AGNews→MNLI showed T*(r8)=T*(r32)=0.064.)  
**New conditions:** 10 (rank=[4,16] × 5 LRs). r8+r32 cached from earlier.  
**Est. wall clock:** ~4h (10 new × ~25 min each)  
**Output dir:** `h2_llama/agnews_to_sst2/`

#### Part B: H1 with MNLI as fine-tuning task (starts after Part A)
**Addresses:** V6 (SST-2-specificity of KL failure) — currently KL(diverse) is shown to be non-monotone only when fine-tuning on SST-2. If KL also fails to track WikiText-2 NLL for MNLI fine-tuning, the failure is general, not SST-2-specific.  
**What it measures:** Same 5-metric suite as H1-SST-2: R, WikiText-2 NLL, KL(wiki), KL(diverse), KL(SuperNI), MMLU accuracy.  
**Key prediction:** R should be monotone with WikiText-2 NLL degradation (like SST-2). KL(diverse) and KL(SuperNI) should be non-monotone.  
**Output dir:** `h1_kl/meta-llama/Llama-3.2-3B_mnli/`  
**Note:** `h1_kl_comparison.py` was updated with `--task mnli` support, MMLU eval, and SuperNI KL before this run. Script changes: added `load_mnli()`, `eval_mnli_accuracy()`, 3-class classifier head.

```bash
CUDA_VISIBLE_DEVICES=0 python h2_sequential_llama.py \
    --task_a agnews --task_b sst2 \
    --skip_phase1 \
    --phase2_lrs 1e-5 5e-5 1e-4 2e-4 5e-4 \
    --phase2_ranks 4 8 16 32 \
    --output_dir ./h2_llama \
    --phase2_steps 2000 \
    2>&1 | tee logs/agnews_to_sst2_r4_r16.log && \
CUDA_VISIBLE_DEVICES=0 python h1_kl_comparison.py \
    --model meta-llama/Llama-3.2-3B \
    --task mnli \
    --lrs 1e-5 5e-5 1e-4 2e-4 5e-4 \
    --rank 8 \
    --steps 2000 \
    --output_dir ./h1_kl \
    2>&1 | tee logs/h1_llama_mnli.log
```

---

### GPU 1 — H1-SST-2 Extended (MMLU + SuperNI rerun)
**Addresses:** Two new H1 metrics that strengthen the core claim.  
**Why rerun:** The existing H1-SST-2 JSONs were computed before MMLU and SuperNI were added. They contain R, WikiText-2 NLL, and KL(diverse/wiki) but are missing `mmlu_acc` and `kl_superni`. The script auto-detects missing keys and reruns those conditions.

**New metric 1 — KL(SuperNI):** Replaces the 50 hand-written custom prompts with samples from `allenai/natural-instructions` (1,600+ diverse NLP tasks). Addresses reviewer concern that custom prompts were cherry-picked to make KL look non-discriminative. If KL is flat across LRs on SuperNI prompts too, the non-monotonicity result is reproducible and prompt-independent.

**New metric 2 — MMLU accuracy:** Zero-shot factual knowledge evaluated on `cais/mmlu` (all 57 subjects, 200 sampled test questions). WikiText-2 NLL measures *language fluency*; MMLU measures *world knowledge*. If R also predicts MMLU degradation at the same threshold where WikiText-2 NLL spikes, the H1 claim generalizes from "R tracks fluency" to "R tracks knowledge."  
**Method:** Score each A/B/C/D option by log-prob of the answer letter token, pick the highest. No fine-tuned classifier needed.

**Conditions rerun:** All 5 LR conditions (lr=[1e-5, 5e-5, 1e-4, 2e-4, 5e-4]) at rank=8.  
**Output dir:** `h1_kl/meta-llama/Llama-3.2-3B/` (same as original SST-2 results, JSONs overwritten with extended metrics)  
**Est. wall clock:** ~3h (σ½_pre is already cached, skips perturbation step)

```bash
CUDA_VISIBLE_DEVICES=1 python h1_kl_comparison.py \
    --model meta-llama/Llama-3.2-3B \
    --task sst2 \
    --lrs 1e-5 5e-5 1e-4 2e-4 5e-4 \
    --rank 8 \
    --steps 2000 \
    --output_dir ./h1_kl \
    2>&1 | tee logs/h1_llama_sst2_extended.log
```

**What to look for:**
```bash
# After completion — full table including new metrics
cat h1_kl/meta-llama/Llama-3.2-3B/summary.json | python -c "
import json, sys
data = json.load(sys.stdin)
print(f'{'LR':>8}  {'R':>6}  {'ΔNLL':>7}  {'KL(div)':>8}  {'KL(SNI)':>8}  {'MMLU':>6}')
for r in data:
    print(f'{r[\"lr\"]:>8.0e}  {r[\"R\"]:>6.3f}  {r.get(\"delta_nll_wiki\",0):>+7.3f}  {r[\"kl_diverse\"]:>8.4f}  {r.get(\"kl_superni\",float(\"nan\")):>8.4f}  {r.get(\"mmlu_acc\",float(\"nan\")):>6.4f}')
"
```

**Interpret as:**
- KL(SuperNI) flat across LRs → non-monotonicity is prompt-independent, not cherry-picked  
- MMLU drops at same LR threshold as WikiText-2 NLL → R tracks world knowledge, not just fluency  
- MMLU flat even when NLL spikes → still interesting, means fine-tuning destroys text generation before world knowledge

---

## CPU (no GPU needed) — Full-model per-param recompute

**Addresses:** V7 (H1-H2 T* comparison unverified). H1 uses full params, H2 uses LoRA only.  
**Script to write:** `compute_fullmodel_norm.py`  
**What it does:** Load base Llama-3.2-3B + Phase 2 LoRA weights. Compute ‖Δθ‖ over ALL 3B params (base + LoRA). Recompute R_A_fullmodel = ‖Δθ_full‖_per-param / σ½_A. Compare T*_fullmodel to H1 T*=[0.22,0.58].

```bash
# After writing the script:
python compute_fullmodel_norm.py \
    --output_dir ./h2_llama \
    --tasks mnli agnews cola \
    --phase2_tasks mnli agnews sst2
```

---

## What to look for in logs

### SST-2→MNLI high-LR result (most important)
After `sst2_to_mnli_highlr.log` completes:

```bash
cat h2_llama/sst2_to_mnli/lr1e-03_rank8_result.json
cat h2_llama/sst2_to_mnli/lr2e-03_rank8_result.json
cat h2_llama/sst2_to_mnli/lr5e-03_rank8_result.json
```

Look for `"acc_phase1_after"`. If it drops below 0.8 (80% of original ≈0.934), forgetting occurred.

**Interpret as:**
- Still no forgetting at lr=1e-3 and lr=2e-3: anomaly is extreme (T* > ~0.6 est.), report as ">8× gap"
- Forgetting first at lr=1e-3: T*(r8) ∈ [0.212, ~0.4], anomaly well-characterized
- Forgetting first at lr=5e-3: T*(r8) ∈ [~0.4, ~0.6], still a clear anomaly

### r16 rank picture
After `mnli_to_agnews_r16.log` and `agnews_to_mnli_r4_r16.log`:

```bash
# Check if AGNews r16 T* ≈ r8 T* (rank-invariant) or shifts
grep -A5 "rank16" h2_llama/mnli_to_agnews/summary.json
grep -A5 "rank16" h2_llama/agnews_to_mnli/summary.json
```

---

## Quick status check (run anytime)

```bash
# Count completed conditions per pair
for dir in h2_llama/*/; do
    count=$(ls ${dir}*_result.json 2>/dev/null | wc -l)
    echo "$count  $dir"
done
```

---

## BATCH 4 — 🔄 Running (2026-07-03)

### GPU 0 — MNLI stochastic boundary (5 seeds)
**Status:** 🔄 Running. Seed 1 started; seeds 2-5 queue sequentially.  
**Problem:** MNLI→SST-2 at lr=2e-4, r8 gave acc_a=0.788 (FORGET) in one run and acc_a=0.840 (SAFE) in another. This makes T*(r8) MNLI ambiguous.  
**What it resolves:** If ≥4/5 seeds FORGET at lr=2e-4 → T*(MNLI r8)≈0.062, matching AGNews (H2 holds cleanly). If ~50/50 → boundary is stochastic, report the distribution.  
**Output:** `h2_llama/mnli_to_sst2/lr1e-04_rank8_seed{1-5}_result.json` and `lr2e-04_rank8_seed{1-5}_result.json`  
**Est. wall clock:** ~4h total (5 seeds × 2 LRs × ~25 min)

```bash
for seed in 1 2 3 4 5; do
  CUDA_VISIBLE_DEVICES=0 python h2_sequential_llama.py \
      --task_a mnli --task_b sst2 \
      --skip_phase1 \
      --phase2_lrs 1e-4 2e-4 \
      --phase2_ranks 8 \
      --output_dir ./h2_llama \
      --phase2_steps 2000 \
      --seed ${seed} \
      2>&1 | tee logs/mnli_sst2_boundary_seed${seed}.log
done
```

---

### GPU 1 — MNLI→AGNews r4, THEN SST-2→AGNews r4+r16 (chained)
**Status:** 🔄 Ready to run.

#### Part A: MNLI→AGNews r4
**What:** Closes the last rank gap for MNLI Phase 1. Currently have r8/r16/r32 for MNLI→AGNews; r4 is missing.  
**New conditions:** 5 (r4 × 5 LRs). r8/r16/r32 cached.  
**Est. wall clock:** ~2h

#### Part B: SST-2→AGNews r4+r16
**What:** Completes the SST-2 Phase 1 rank sweep for →AGNews (currently r8/r32 only). Helps characterize whether SST-2's anomalous high T*→MNLI is rank-dependent or consistent with a wider pattern.  
**New conditions:** 10 (rank=[4,16] × 5 LRs). r8/r32 cached.  
**Est. wall clock:** ~4h

```bash
CUDA_VISIBLE_DEVICES=1 python h2_sequential_llama.py \
    --task_a mnli --task_b agnews \
    --skip_phase1 \
    --phase2_lrs 1e-5 5e-5 1e-4 2e-4 5e-4 \
    --phase2_ranks 4 8 16 32 \
    --output_dir ./h2_llama \
    --phase2_steps 2000 \
    2>&1 | tee logs/mnli_to_agnews_r4.log && \
CUDA_VISIBLE_DEVICES=1 python h2_sequential_llama.py \
    --task_a sst2 --task_b agnews \
    --skip_phase1 \
    --phase2_lrs 1e-5 5e-5 1e-4 2e-4 5e-4 \
    --phase2_ranks 4 8 16 32 \
    --output_dir ./h2_llama \
    --phase2_steps 2000 \
    2>&1 | tee logs/sst2_to_agnews_r4_r16.log
```

---

## Vulnerability status (as of 2026-07-03)

| V# | Status |
|---|---|
| V1 (σ½ redundant within H2) | Open — T_raw ≈ const across tasks; defended via H1 cross-model argument |
| V4 (rank coverage) | ✅ r4/r8/r16/r32 now complete for AGNews→MNLI, MNLI→SST-2, AGNews→SST-2; MNLI→AGNews missing r4 (running) |
| V5 (SST-2 one-sided) | ✅ Closed — T*(r8)=[0.212,0.851] mid=0.532 |
| V6 (KL SST-2-specificity) | ✅ Closed — H1-MNLI complete: KL(diverse) non-monotone for MNLI fine-tuning too |
| V7 (H1-H2 T* unverified) | ✅ Closed — both use merge_and_unload(), T*_raw=1.75e-4 cluster |
| V8 (SST-2 anomaly unexplained) | Open — 3 hypotheses ruled out (direction overlap, effective rank, binary/multi-class) |
| KL cherry-pick concern | ✅ Addressed — KL(SuperNI) is monotone but fails to discriminate at threshold; KL(diverse) fails for both SST-2 and MNLI fine-tuning |
| MMLU knowledge claim | ✅ Confirmed — MMLU drops at same R threshold as NLL for both fine-tuning tasks |
| MNLI stochastic boundary (r8) | 🔄 In progress — 5 seeds running on GPU 0 |

---

## SCP commands (after Batch 4 completes)

```bash
# MNLI stochastic seeds
scp -P 22175 -i ~/.ssh/id_ed25519 \
    "root@69.30.85.51:/workspace/h2_llama/mnli_to_sst2/lr*_rank8_seed*_result.json" \
    /Users/kinakim/Github/NTRS/h2_llama/mnli_to_sst2/

# MNLI→AGNews r4
scp -P 22175 -i ~/.ssh/id_ed25519 \
    "root@69.30.85.51:/workspace/h2_llama/mnli_to_agnews/lr*_rank4_result.json" \
    /Users/kinakim/Github/NTRS/h2_llama/mnli_to_agnews/

# SST-2→AGNews r4 and r16
scp -P 22175 -i ~/.ssh/id_ed25519 \
    "root@69.30.85.51:/workspace/h2_llama/sst2_to_agnews/lr*_rank4_result.json" \
    /Users/kinakim/Github/NTRS/h2_llama/sst2_to_agnews/
scp -P 22175 -i ~/.ssh/id_ed25519 \
    "root@69.30.85.51:/workspace/h2_llama/sst2_to_agnews/lr*_rank16_result.json" \
    /Users/kinakim/Github/NTRS/h2_llama/sst2_to_agnews/
```
