# Experiment Plan — 2× A40 GPU
**Written:** 2026-07-01  
**Hardware:** 2× NVIDIA A40 (48GB each)  
**Model:** Llama-3.2-3B, LoRA fine-tuning (fits on 1× A40 easily)  
**Status key:** 🔴 Blocking | 🟡 Strong | 🟢 Nice-to-have

---

## What σ½_A values are already computed (do NOT recompute)

| Phase 1 task | σ½_A_acc | Status |
|---|---|---|
| MNLI | 0.002857 | ✅ Done |
| AGNews | 0.002756 | ✅ Done |
| SST-2 | 0.003337 | ✅ Done |
| CoLA | 0.001944 | ✅ Done |

**All Phase 1 models are trained and saved. No Phase 1 retraining needed.**

---

## Vulnerability being addressed by each experiment

| Exp | Addresses | Why it matters |
|---|---|---|
| SST-2→MNLI higher LR | V5 (one-sided bound) | ">8× anomaly" claim has unbounded arm — indefensible as-is |
| r16 MNLI→AGNews | V4 (rank gap) | AGNews ONLY task where r32 is also invariant — must explain why |
| r16 AGNews→MNLI | V4 (rank gap) | Same — need both directions to characterize the anomaly |
| r4 MNLI→SST-2 | V4 (rank coverage) | Cannot claim low-rank LoRA safety without r4 data |
| r4 AGNews→MNLI | V4 (rank coverage) | Same |
| Full-model per-param recompute | V7 (H1-H2 T* comparison) | Unverified assumption: H1 uses full params, H2 uses LoRA params |
| H1 with MNLI as Phase 2 | V6 partial (generalizability) | KL failure claim currently only shown for SST-2 fine-tuning |

---

## Batch schedule (2× A40, parallel)

### Batch 1 — Run immediately (~4–5h wall clock)

| GPU | Experiment | Config | Output dir |
|---|---|---|---|
| GPU 0 | SST-2→MNLI higher LR | Phase 1: SST-2 (existing checkpoint), Phase 2: MNLI, rank=8, lr=[1e-3, 2e-3, 5e-3], 2000 steps | h2_llama/sst2_to_mnli_highlr/ |
| GPU 1 | r16 MNLI→AGNews | Phase 1: MNLI (existing checkpoint), Phase 2: AGNews, rank=16, lr=[1e-5, 5e-5, 1e-4, 2e-4, 5e-4], 2000 steps | h2_llama/mnli_to_agnews_r16/ |

**Expected from Batch 1:**
- GPU 0: First actual SST-2→MNLI forgetting data point. Pins T*(r8) upper bound. If SST-2 still doesn't forget at lr=5e-3, the anomaly is extreme and that's a finding itself.
- GPU 1: r16 T* for MNLI→AGNews to compare against r8≈0.062 and r32≈0.131.

---

### Batch 2 — Start after Batch 1 completes (~4–5h wall clock)

| GPU | Experiment | Config | Output dir |
|---|---|---|---|
| GPU 0 | r16 AGNews→MNLI | Phase 1: AGNews (existing checkpoint), Phase 2: MNLI, rank=16, lr=[1e-5, 5e-5, 1e-4, 2e-4, 5e-4], 2000 steps | h2_llama/agnews_to_mnli_r16/ |
| GPU 1 | r4 MNLI→SST-2 | Phase 1: MNLI (existing checkpoint), Phase 2: SST-2, rank=4, lr=[1e-5, 5e-5, 1e-4, 2e-4, 5e-4], 2000 steps | h2_llama/mnli_to_sst2_r4/ |

**Expected from Batch 2:**
- GPU 0: r16 T* for AGNews→MNLI. AGNews has r8≈0.062 and r32≈0.062 (rank-invariant). r16 will either: (a) also ≈0.062 (confirming AGNews is robustly rank-invariant), or (b) start to deviate (showing the invariance breaks between r8 and r32 for AGNews too).
- GPU 1: r4 T* for MNLI. With r8≈0.062, r4 should be even more conservative. Confirms safe zone for low-rank LoRA.

---

### Batch 3 — Start after Batch 2 completes (~4–5h wall clock)

| GPU | Experiment | Config | Output dir |
|---|---|---|---|
| GPU 0 | r4 AGNews→MNLI | Phase 1: AGNews (existing checkpoint), Phase 2: MNLI, rank=4, lr=[1e-5, 5e-5, 1e-4, 2e-4, 5e-4], 2000 steps | h2_llama/agnews_to_mnli_r4/ |
| GPU 1 | H1 MNLI Phase 2 (Llama) | Pretrained Llama-3.2-3B, fine-tune on MNLI, lr=[1e-5, 5e-5, 1e-4, 2e-4, 5e-4], rank=8, 2000 steps, measure WikiText-2 NLL | h1_kl/meta-llama/Llama-3.2-3B-mnli/ |

**Expected from Batch 3:**
- GPU 0: r4 AGNews data. Combined with r4 MNLI from Batch 2, this establishes the r4 picture for the two best-characterized Phase 1 tasks.
- GPU 1: KL comparison for MNLI fine-tuning. If KL(diverse) is also non-monotone here, the V6 concern about SST-2-specificity is partially addressed.

---

### After Batch 3 — CPU/local (~2h, no GPU needed)

**Full-model per-param recompute (V7):**
- Take existing H2 checkpoints (h2_llama/*/phase2_model.pt or lora weights)
- For each checkpoint: compute ‖Δθ‖ over ALL 3B parameters (base + LoRA), not just LoRA
- Recompute R_A_fullmodel = full_per_param / σ½_A_acc
- Compare T*(r8) in full-model R_A space to H1 T* [0.22–0.58]
- If T* shifts toward H1 values → H1-H2 comparison is valid. If not → must retract the comparison.
- Script: write `compute_fullmodel_norm.py`

---

## What each result tells you

### After Batch 1 — SST-2→MNLI higher LR result

**Scenario A: SST-2 still doesn't forget at lr=1e-3 (R_A ≈ 0.4–0.6 based on extrapolation)**
- T*(r8) SST-2→MNLI is genuinely very high (>0.4)
- Ratio vs SST-2→AGNews (T*≈0.027) is >15×
- Anomaly is extreme and unambiguous. Report with both bounds.

**Scenario B: SST-2 starts forgetting at lr=1e-3 or 2e-3**
- Pins T*(r8) in [0.212, ~0.4]
- Anomaly ratio becomes 8–15×, well-characterized
- This is the ideal outcome for the paper

**Scenario C: SST-2 already collapses at lr=1e-3 (unlikely given trend)**
- Immediate forgetting onset — T*(r8) just above current sweep maximum
- Still a clear anomaly vs AGNews

### After Batches 2+3 — rank picture

With r4, r8, r16, r32 for MNLI and AGNews, you will know:
- Where rank-invariance breaks for MNLI (is r16 T* between 0.062 and 0.131, or does it jump?)
- Whether AGNews is truly rank-invariant at all ranks (r4, r8, r16, r32 all ≈0.062 would be striking)
- Whether r4 is in the "safe zone" (T* ≈ same as r8) for both tasks

This turns the rank discussion from "r8 works, r32 sometimes doesn't" into a quantitative picture of rank-dependence.

---

## Total wall-clock time

| Phase | Wall clock | GPUs in use |
|---|---|---|
| Batch 1 | ~4–5h | 2× A40 |
| Batch 2 | ~4–5h | 2× A40 |
| Batch 3 | ~4–5h | 2× A40 |
| CPU recompute | ~2h | None (local) |
| **Total** | **~14–17h** | |

Can run Batches 1–3 overnight across one day. CPU recompute runs locally anytime.

---

## What remains after all batches

| Vulnerability | Status after batches |
|---|---|
| V1 (σ½ redundant in H2) | Still open — needs task with very different σ½_A |
| V2 (monotonicity definitional) | Writing fix only — already corrected |
| V3 (T* precision overstated) | Partially addressed by additional LR points for SST-2→MNLI |
| V4 (rank coverage) | ✅ Closed by r4+r16 experiments |
| V5 (SST-2 one-sided bound) | ✅ Closed by Batch 1 |
| V6 (single architecture) | Partially addressed by H1-MNLI; full fix requires 2nd model (~16h) |
| V7 (H1-H2 T* unverified) | ✅ Closed — norms identical (both use merge_and_unload); T*_raw converges to ~1.75e-4 across H1+H2 |
| V8 (SST-2 anomaly unexplained) | Still open — 3 hypotheses ruled out, no new ones yet |

V1 and V8 are the remaining open questions after all planned experiments.
