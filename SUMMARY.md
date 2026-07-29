# NTRS Research Summary
**Last updated:** 2026-07-14 (Batch 9: Mistral SST-2 v5 PRIMARY RESULT — effective batch=16 (batch=4, accum=4), test-split σ½=1.824×10⁻⁴ CORRECT; T*(relslack)=[0.194, 0.226]; overlaps GPT-2 [0.22, 0.45]; all 3 models onset near R≈0.19–0.24. v3 [0.347,0.397] was batch=4 / wrong σ½ protocol — superseded by v5.)
**Project:** σ½ as a pre-training forgetting budget for LLM fine-tuning

---

## Core Idea

We propose **σ½** (sigma-half) — the perturbation radius at which a model's quality drops to half its certified peak density — as a pre-run predictor of forgetting during sequential fine-tuning.

Computed once from a Phase 1 checkpoint using randomized smoothing, σ½ defines a "certified basin radius" in weight space. The ratio **R = ‖Δθ‖ / σ½** (per-parameter weight displacement normalized by the basin radius) is our primary metric.

**Positioning:** Most forgetting predictors require both the before and after model (post-hoc). R is computable *during* training using only the Phase 1 checkpoint + the running weight norm.

**Main competitor:** RL's Razor (arXiv:2509.04259) — uses KL(base‖fine-tuned) on task prompts to predict forgetting. We argue this fails at instruction-tuning boundaries because KL on instruction prompts is large and non-monotone regardless of how safely the model was fine-tuned.

---

## Hypotheses

### H1 — Pretraining→SFT boundary
> At the pretraining→SFT boundary, R = ‖Δθ‖/σ½_pre discriminates safe from degraded fine-tuning (measured by WikiText-2 NLL preservation). KL on instruction-style prompts cannot discriminate because it saturates in the safe zone — it measures the base→instruction-following distribution shift, which is large regardless of fine-tuning depth. KL on the pretraining domain (WikiText-2) is informative, but R is computable as a pre-run budget before Phase 2 starts, requiring no additional evaluation.

### H2 — Sequential task forgetting
> σ½_A, computed once from Phase 1 checkpoint, defines a forgetting budget R_A. When Phase 2 fine-tuning pushes R_A above a threshold, Task A is forgotten. This is a pre-run predictor.

---

## H1 Experiment

### Setup — all models (H1)
- **Models:** GPT-2 (117M), Llama-3.2-3B (3B), Mistral-7B-v0.1 (7B) — 60× size range
- **Phase 1:** Pretrained base weights (no fine-tuning)
- **Phase 2:** SST-2 sentiment classification (all models); MNLI (Llama + Mistral). LoRA rank=8, α=16.
- **LR sweep:** [1e-5, 5e-5, 1e-4, 2e-4, 5e-4]; GPT-2: 3000 steps, Llama + Mistral: 2000 steps
- **σ½ computation:** WikiText-2 NLL. GPT-2: eval_slack=1e-4. Llama: eval_slack=0.01. Mistral: uses same script.
- **Script:** `h1_kl_comparison.py`
- **Output dir:** `h1_kl/gpt2/`, `h1_kl/meta-llama/Llama-3.2-3B/`, `h1_kl/mistralai/Mistral-7B-v0.1_sst2_v5/` (SST-2 **v5 PRIMARY** ✅), `h1_kl/mistralai/Mistral-7B-v0.1_mnli/` (MNLI v2 ✅). Archived: v1 (invalid), v2 (invalid), v3 (partial — wrong σ½), v4a (broken), v4b (partial — wrong σ½). See version history table in results section.

### Results

**GPT-2 (σ½_pre = 0.007191, corrected 2026-06-30):**
| LR | R | Δ NLL | KL(wiki) | KL(diverse) | SST-2 Acc |
|---|---|---|---|---|---|
| 1e-5 | 0.015 | −0.027 | 0.006 | 0.228 | 0.462* |
| 5e-5 | 0.086 | −0.054 | 0.041 | 1.186 | 0.768 |
| 1e-4 | 0.220 | **+0.016** | 0.091 | 2.791 | 0.768 |
| 2e-4 | 0.453 | **+1.385** | 1.508 | 4.098 | 0.802 |
| 5e-4 | 1.741 | **+4.310** | 4.528 | 3.533 | 0.776 |

*lr=1e-5: SST-2 never learned (below chance).

**Llama-3.2-3B SST-2 fine-tuning (σ½_pre = 0.000772, extended 2026-07-03):**
| LR | R | Δ NLL | KL(wiki) | KL(diverse) | KL(SuperNI) | MMLU | SST-2 Acc |
|---|---|---|---|---|---|---|---|
| 1e-5 | 0.038 | −0.002 | 0.002 | 7.11 | **0.360** | 0.555 | 0.954 |
| 5e-5 | 0.152 | +0.009 | 0.045 | 4.05 | **0.587** | 0.525 | 0.960 |
| 1e-4 | 0.316 | +0.035 | 0.076 | 8.57 | **2.863** | 0.540 | 0.962 |
| 2e-4 | 0.784 | **+1.133** | 1.259 | 8.90 | **3.297** | 0.370 | 0.958 |
| 5e-4 | 4.877 | **+8.738** | 8.937 | 10.67 | **10.369** | 0.215 | 0.530 |

**Llama-3.2-3B MNLI fine-tuning (NEW, 2026-07-03 — V6 closes KL failure as general):**

**⚠️ CORRECTED 2026-07-17**: `R` below uses the stale non-relslack σ½ (5.7627119e-4), the same staleness that affected Llama SST-2. Relslack-corrected column added (σ½=7.717391e-4). Do not cite the `R` column below without checking it against the corrected column first.

| LR | R (non-relslack) | **R (relslack, corrected)** | Δ NLL | KL(wiki) | KL(diverse) | KL(SuperNI) | MMLU | MNLI Acc |
|---|---|---|---|---|---|---|---|---|
| 1e-5 | 0.054 | **0.040** | +0.007 | 0.004 | **6.34** | 1.244 | 0.540 | 0.838 |
| 5e-5 | 0.178 | **0.133** | +0.032 | 0.035 | **6.63** | 1.390 | 0.520 | 0.858 |
| 1e-4 | 0.312 | **0.233** | +0.039 | 0.056 | **5.46** ↓ | 1.323 | 0.500 | 0.846 |
| 2e-4 | 0.711 | **0.531** | +0.064 | 0.105 | 10.55 | 3.892 | 0.490 | 0.832 |
| 5e-4 | 3.570 | **2.666** | **+7.745** | 7.767 | 9.53 | 9.238 | **0.250** | 0.822 |

**Mistral-7B-v0.1 SST-2 — experiment version history (all in `h1_kl/mistralai/`):**

| Version | Status | σ½ methodology | Effective Batch | T* (R relslack) | Notes |
|---|---|---|---|---|---|
| v1, v2 | ❌ INVALID | validation split | 4 | — | Mixed cache; bogus KL(SuperNI) from DIVERSE_PROMPTS fallback |
| v3 | ⚠️ PARTIAL | **wrong** (validation split) | 4 | [0.347, 0.397] | Correct training; σ½=1.559e-4 from wrong split; R non-monotone (diverge-and-contract at lr=2e-4). Not comparable to GPT-2/Llama σ½. |
| v4a | ❌ BROKEN | — | 4 | — | Training failure: acc~0.5 everywhere across 3 runs; discarded |
| v4b | ⚠️ PARTIAL | **wrong** (validation split) | 8 | — | Correct training; σ½=1.559e-4 from wrong split; batch mismatch vs other models |
| **v5** | ✅ **PRIMARY** | **correct** (test sequential) | **16 (batch=4 × accum=4)** | **[0.194, 0.226]** | **σ½=1.824e-4 matches relslack; identical protocol to GPT-2/Llama; 8000 train samples = same as other models** |

**Mistral-7B-v0.1 SST-2 fine-tuning (σ½_pre = 1.824×10⁻⁴, v5 PRIMARY RESULT, effective batch=16, 2026-07-14):**
| LR | R (relslack) | Δ NLL | KL(wiki) | KL(diverse) | KL(SuperNI) | MMLU | SST-2 Acc |
|---|---|---|---|---|---|---|---|
| 1e-5 | 0.030 | +0.001 | 0.000 | 0.088 | 0.031 | 0.585 | 0.480* |
| 5e-5 | **0.194** | +0.125 | 0.110 | 1.411 | 0.581 | 0.505 | 0.930 |
| 1e-4 | 0.226 | **+2.810** | 2.930 | 3.706 | **7.218** | 0.280 | 0.640 |
| 2e-4 | 0.642 | **+14.21** | 14.641 | 5.293 | 8.448 | 0.245 | 0.530 |
| 5e-4 | 1.320 | **+23.45** | 23.946 | 22.034 | 23.709 | 0.215 | 0.470 |

*lr=1e-5: SST-2 did not converge (acc=0.480, near-random). R is strictly monotone with LR — effective batch=16 eliminates the diverge-and-contract issue seen in v3 (batch=4).

**T*(Mistral-7B, SST-2, v5 relslack) = [0.194, 0.226]** ✅ PRIMARY RESULT — last safe at R=0.194 (lr=5e-5, ΔNLL=+0.125, acc=0.930), first catastrophic at R=0.226 (lr=1e-4, ΔNLL=+2.810, 22.5× NLL jump). σ½=1.824×10⁻⁴ verified to match `pretrained_density_relslack.json`. KL(SuperNI) discriminating at cliff: 0.581→7.218 (12.4× jump while ΔNLL 22.5×). KL(diverse) poor at cliff: 1.411→3.706 (2.6× while ΔNLL 22.5×) — not saturated unlike Llama (starts at 0.088 vs Llama's 7.11), but too shallow to detect the threshold.

**Mistral-7B-v0.1 MNLI fine-tuning (v2, real KL(SuperNI), 2026-07-11):**
v2 re-run with actual superni_prompts.json. KL(SuperNI) is now distinct from KL(diverse) and IS monotone for MNLI. At R=1.013 (catastrophic): KL(SuperNI)=7.295 vs KL(diverse)=3.410 — SuperNI is *higher*, the opposite of the bogus v1 pattern.
| LR | R | Δ NLL | KL(wiki) | KL(diverse) | KL(SuperNI) | MMLU | MNLI Acc |
|---|---|---|---|---|---|---|---|
| 1e-5 | 0.084 | +0.017 | 0.012 | 1.270 | **0.353** | 0.585 | 0.830 |
| 5e-5 | 0.380 | **+0.603** | 0.667 | 3.466 | **0.971** | 0.535 | 0.858 |
| 1e-4 | 0.874 | **+3.793** | 3.961 | 3.762 | **3.771** | 0.275 | 0.878 |
| 2e-4 | **1.013** | **+8.001** | 8.232 | 3.410 | **7.295** | 0.280 | **0.350** |
| 5e-4 | 4.470 | **+7.313** | 7.517 | 3.583 | **6.793** | 0.215 | 0.348 |

KL(SuperNI) progression: 0.353 → 0.971 → 3.771 → 7.295 → 6.793 (monotone, 7× jump safe→catastrophic). KL(diverse) plateau: 1.270 → 3.466 → 3.762 → 3.410 → 3.583 (non-monotone after R=0.324 relslack, ~3× ceiling). For Mistral MNLI: KL(SuperNI) discriminates at the cliff; KL(diverse) does not.

**⚠️ CORRECTED 2026-07-17 (relslack):** All R values below were originally reported using Mistral MNLI's cached non-relslack σ½ (1.5586592e-4). Verified via matching `nll_wiki_base` (2.9934375, identical to SST-2 v5) that this is the same frozen checkpoint as SST-2 v5, so the relslack σ½ (1.824175824e-4, already computed for v5 — no new GPU run needed) applies here too. See `h1_kl/mistralai/Mistral-7B-v0.1_mnli/pretrained_density_relslack.json`.

**T*(Mistral-7B, MNLI) = [0.071, 0.324]** (relslack; was reported as [0.084, 0.380] non-relslack) — NLL cliff is early (R=0.071 safe at +0.017, R=0.324 already +0.603 or 20% above baseline). Task acc stays high through R=0.747 (acc=0.878; was R=0.874 non-relslack) before catastrophic collapse at R=0.865 (acc=0.350, ΔNLL=+8.001; was R=1.013 non-relslack).

**⚠️ CORRECTED 2026-07-17 — the Llama comparison below was wrong, not just imprecise.** "Llama MNLI stays safe through R=3.570" used Llama's *stale non-relslack* σ½ (5.7627119e-4). With Llama's own relslack σ½ (7.717391e-4, same value used for Llama SST-2), and applying the same "ΔNLL exceeds 1% of baseline NLL" break criterion to both models for a like-for-like comparison: **Llama MNLI T* = [0.133, 0.233]** (last safe at R=0.133/ΔNLL=+0.032, first broken at R=0.233/ΔNLL=+0.039) — which falls **entirely inside** Mistral MNLI's corrected bracket [0.071, 0.324], not outside it. The "Mistral is 9.4× more fragile, strikingly different from Llama" framing does not survive a consistent comparison — the two models' MNLI thresholds are statistically indistinguishable at this LR-grid resolution once both use the same σ½ convention. This reverses, not just corrects, the previous conclusion; see the "What Claims Can Be Made Now" and V6 sections below, both updated accordingly.

**Cross-model T* comparison (SST-2 fine-tuning, H1 — all R values in relslack σ½ space; Mistral = v5 PRIMARY):**
| Model | σ½_pre (relslack) | T* (SST-2, relslack, cliff-jump criterion) | raw per_param at lower T* | Raw span |
|---|---|---|---|---|
| GPT-2 (117M) | 7.191×10⁻³ | [0.220, 0.453] | 1.58×10⁻³ | — |
| Llama-3.2-3B (3B) | 7.72×10⁻⁴ | [0.235, 0.586] | 1.82×10⁻⁴ | 8.5× below GPT-2 |
| Mistral-7B (7B) | **1.824×10⁻⁴** | **[0.194, 0.226]** | **3.53×10⁻⁵** | **44× below GPT-2** |

GPT-2 [0.220, 0.453] and Mistral [0.194, 0.226] overlap at **[0.220, 0.226]**. Llama's lower bound (0.235) is adjacent — within 5% of Mistral's upper bound (0.226). The 3-way overlap is near-coincident rather than a broad shared interval: all three models show degradation onset in the range R≈0.19–0.24 despite 60× model size range. Raw displacement at lower T* spans **44×** (1.58e-3 for GPT-2 to 3.53e-5 for Mistral); σ½_pre spans **39×**. After normalization, the T* onset band is **1.2×** — σ½ compresses 44× to 1.2×. Without σ½, raw values are incommensurable.

**⚠️ CORRECTED 2026-07-17 — the above uses the "last-safe-before-a-big-jump" (cliff) criterion, eyeballed per model. This is NOT the same criterion used to fix the MNLI comparison above (ΔNLL > 1% of baseline NLL), and applying the stricter, model-independent criterion consistently gives a materially different picture:**

| Model | T* (SST-2, relslack, **1%-of-baseline-NLL criterion**) | Last safe LR | First broken LR |
|---|---|---|---|
| GPT-2 | [0.220, 0.453] | 1e-4 (ΔNLL=+0.016, under 0.0449 threshold) | 2e-4 (ΔNLL=+1.385) — **same as cliff criterion** |
| Llama | **[0.113, 0.236]** | 5e-5 (ΔNLL=+0.009, under 0.0336 threshold) | 1e-4 (ΔNLL=+0.035, barely over) |
| Mistral | **[0.030, 0.194]** | 1e-5 (ΔNLL=+0.001, under 0.0299 threshold) | 5e-5 (ΔNLL=+0.125) |

Under this consistent criterion: GPT-2∩Llama = [0.220, 0.236] (overlap), Llama∩Mistral = [0.113, 0.194] (overlap), but **GPT-2∩Mistral = ∅** (0.220 > 0.194 — no overlap). This is a **chain**, not a 3-way overlap: Llama bridges GPT-2 and Mistral, but GPT-2 and Mistral do not share any common R value. The clean "all three converge near R≈0.19–0.24" reading depends on using the looser cliff-jump criterion; the stricter criterion (the one that correctly resolved the MNLI comparison) does not support a clean 3-way overlap.

**Additional wrinkle — Mistral's "last safe" point under the strict criterion is degenerate.** R=0.030 is lr=1e-5, where SST-2 accuracy is 0.480 (task did not converge, near-random). The original cliff-based analysis explicitly excluded this point as a meaningful reference ("last safe at R=0.194, lr=5e-5" — skipping over lr=1e-5). If the non-converged point is excluded from the strict criterion too (for consistency with how it was excluded from the cliff-based one), Mistral has **no valid safe reference point** in the tested LR grid — every converged run (lr≥5e-5) already exceeds the 1% ΔNLL threshold. Mistral's T* under the strict criterion is then only upper-bounded by 0.194, not a proper interval — a real gap in the LR grid (nothing tested between 1e-5 and 5e-5), not a modeling choice.

**Net honest statement:** σ½ still compresses a 39–44× raw-displacement/σ½-scale range down to a much narrower normalized band under either criterion — that compression is real and criterion-independent. What is criterion-*dependent* is whether the resulting band supports a clean simultaneous 3-way overlap (cliff-jump: yes, narrowly) or a weaker chain with a genuine gap in the Mistral data (1%-threshold: no). The paper should state the weaker, criterion-robust version of the claim, or add the missing Mistral LR point (e.g. lr=2e-5 or 3e-5) to close the gap before claiming more.

---

### Key Findings — H1

**KL(diverse) saturates in the safe zone — domain mismatch, not monotonicity failure.**
At Llama LR=1e-5 (safe, ΔNLL≈0), KL(diverse) = 7.11. At LR=5e-4 (catastrophic), KL(diverse) = 10.67. A predictor that reads 7.11 in the safe zone cannot discriminate. The reason: KL-diverse evaluates on instruction-style prompts, which are OOD for the base model. The base→instruction-following shift dominates regardless of fine-tuning aggressiveness. The non-monotone pattern (7.11→4.05→8.57) is expected — different LR runs trace different directions in weight space, and the Fisher approximation (KL ≈ ½ Δθᵀ F Δθ) supports monotone KL only along a *fixed ray*, not across runs. KL(wiki) on the pretraining domain IS monotone and tracks ΔNLL closely — confirming the failure is domain-specific. KL(SuperNI) is monotone but poor at threshold discrimination: only +15% change at the safe/forget boundary while ΔNLL jumps 32×.

**NEW (2026-07-03): MMLU drops at same R threshold as NLL.**
MMLU accuracy: 0.555→0.525→0.540→0.370→0.215 for SST-2 fine-tuning. Collapses at lr=2e-4 (R=0.784), exactly where ΔNLL spikes. R predicts world knowledge degradation, not just text fluency. MNLI fine-tuning: 0.540→0.520→0.500→0.490→0.250 — gradual decline then collapse at lr=5e-4. Same threshold as NLL.

**NEW (2026-07-03): H1 with MNLI fine-tuning confirms KL failure is not SST-2-specific (V6 closed).**
KL(diverse) non-monotone for MNLI fine-tuning too: 6.34→6.63→**5.46**↓→10.55→9.53 (drops at lr=1e-4 despite rising NLL). T* for MNLI fine-tuning: safe through R=0.531 (relslack; was reported as R=0.711 non-relslack), catastrophic at R=2.666 (relslack; was R=3.570 non-relslack) — much wider safe zone than SST-2's T*≈0.24 (relslack). MNLI fine-tuning displaces weights more diffusely, so the model is more resistant per normalized unit.

**1. σ½_pre is NOT scale-invariant (see ⚠️ CORRECTION below).**
After standardizing eval_slack: σ½_pre(GPT-2)=0.007191, σ½_pre(Llama)=0.000772 — 9.3× different despite 26× model size. The original near-equal values (0.000603 vs 0.000576) were an artifact of mismatched eval_slack settings. Scale-invariance claim is retracted.

**2. R is predictive of degradation onset for both models.**
ΔNLL increases with R for both GPT-2 and Llama. Note: within a single LR sweep, R and LR are co-monotone by construction — the non-trivial evidence is that KL(diverse) is NOT co-monotone with ΔNLL over the same conditions, while R is. The comparison against KL is what makes this meaningful, not monotonicity alone.

**3. After correction, T* converges across model scales (cliff-jump criterion; see criterion-sensitivity correction above for the stricter 1%-threshold version).**
- GPT-2: T* ∈ [0.220, 0.453] — corrected (was [1.03, 2.63] with old σ½_pre)
- Llama: T* ∈ [0.235, 0.586] — corrected (was [0.315, 0.773]); **0.586 not 0.577** — 0.577 was a rounding/precision error, exact value is 4.5188251e-4/7.717391304347826e-4 = 0.58554, corrected 2026-07-17
- Ranges overlap: both models degrade starting at R ≈ 0.22–0.25, catastrophic at R ≈ 0.45–0.59.
- R<1 is conservative for both models — degradation begins well below R=1 in both cases.
- Previously "R=1 holds exactly for GPT-2" was an artifact of the mismatched σ½_pre. This finding is **retracted**.
- **⚠️ This GPT-2/Llama pairwise overlap holds under both the cliff-jump and the stricter 1%-threshold criteria** ([0.220,0.453]∩[0.113,0.236]=[0.220,0.236] either way) — it is the 3-way GPT-2/Mistral comparison that is criterion-sensitive, not this one.

**4. KL(diverse) fails differently for Mistral — not saturated, but still fails threshold discrimination.**
Llama KL(diverse) is saturated in the safe zone (7.11 at LR=1e-5, near-catastrophic). Mistral v5 KL(diverse) is NOT saturated (starts at 0.088 at LR=1e-5, 0 training). However, Mistral KL(diverse) still fails at the critical boundary (R=0.194→0.226, ΔNLL=+0.125→+2.810, a 22.5× NLL jump): KL(diverse) moves only 1.411→3.706 (2.6×). A 2.6× change while ΔNLL jumps 22.5× is insufficient for threshold detection. The failure mechanism differs by model: Llama fails by saturation from the first update; Mistral fails by insufficient slope at the cliff. Both are useless as pretraining-forgetting detectors: they track instruction-domain distribution shift, not pretraining fidelity. **v5 (2026-07-14):** KL(SuperNI) for Mistral SST-2 is discriminating at the cliff: 0.581 (safe) → 7.218 (catastrophic) — 12.4× jump while ΔNLL jumps 22.5×. KL(SuperNI) detects the cliff; KL(diverse) does not. Contrast with Llama SST-2 where KL(SuperNI) moves only 15% at the boundary.

**5. Llama per-param displacement is 10× smaller at same LR — but σ½_pre is also 9.3× smaller.**
At LR=5e-4: GPT-2 per_param=0.01253, Llama per_param=0.001093 (~11× smaller). GPT-2 σ½_pre=0.007191, Llama σ½_pre=0.000772 (9.3× smaller). The two effects nearly cancel, yielding similar R values: GPT-2 R=1.741, Llama R=1.416 at LR=5e-4. This explains why T* converges after correction even though raw displacements differ by 11×.

**⚠️ CORRECTION (2026-06-30, updated 2026-07-13): σ½_pre is NOT scale-invariant; R tables above now corrected.**
After standardizing eval_slack to 1% relative NLL (script: `recompute_h1_sigma_relslack.py`):
- GPT-2: σ½_pre = **0.007191** (NLL_baseline = 4.211, eval_slack_abs = 0.042)
- Llama: σ½_pre = **0.000772** (NLL_baseline = 3.059, eval_slack_abs = 0.031)
- Mistral-7B: σ½_pre = **1.824×10⁻⁴** (NLL_baseline = 2.660, eval_slack_abs = 0.0266) — only **+17%** over non-relslack 1.559×10⁻⁴. Small shift because Mistral's density drops sharply between σ=0.0001 (density=0.976) and σ=0.0002 (density=0.315). **v5 CONFIRMED:** h1_kl_comparison.py with test-split σ½ computation yields 1.82417582×10⁻⁴, matching `pretrained_density_relslack.json` exactly.
- Ratio GPT-2/Llama: **9.3×**; Llama/Mistral: **4.2×**; GPT-2/Mistral: **39×** — NOT scale-invariant.

The original ~0.0006 values for GPT-2 and Llama were an artifact of wildly mismatched eval_slacks: GPT-2 was using eval_slack=1e-4 (0.002% of NLL) while Llama was using eval_slack=0.01 (0.33% of NLL). After standardizing, the σ½_pre values diverge by nearly an order of magnitude. **The scale-invariance claim in Finding 1 is retracted.** However, the R tables above have been corrected using the new σ½_pre values and the key monotonicity + T*-convergence findings hold in compressed-magnitude form. After correction: GPT-2 T*≈[0.22, 0.45], Llama T*≈[0.24, 0.59], **Mistral T*≈[0.194, 0.226] (v5 PRIMARY, relslack, 2026-07-14)** — under the cliff-jump criterion, GPT-2 and Mistral overlap at [0.22, 0.23] and Llama's lower bound (0.24) is adjacent. **⚠️ CORRECTED 2026-07-17: this "all three within R≈[0.19,0.24]" reading is criterion-sensitive** — see the full correction under "Cross-model T* comparison" above. Under the stricter, model-independent 1%-of-baseline-NLL criterion the brackets shift to GPT-2 [0.22,0.45], Llama [0.11,0.24], Mistral [0.03,0.19], forming a chain (not a 3-way overlap): GPT-2 and Mistral no longer share any common R value. The robust claim is the 39–44×→~3× compression, not a precise simultaneous convergence.

**Figure:** `figures/fig1_h1_comparison.png`
- 2×3 grid: GPT-2 (top row), Llama (bottom row)
- Left: R vs ΔNLL with T* band; Center: KL vs ΔNLL (monotone vs collapse); Right: SST-2 accuracy

### Direct-R Sweep — validity check for the direction-rescale design (GPT-2, 2026-07-20)

**Motivation**: the LR sweep above produces R as a side effect of LR choice; per_param_norm/LR is flat ~1.8–2.3 across the first four LR points then jumps at the last, undersampling exactly the R region T* needs resolution in. `h1_direct_r_sweep.py` targets R directly instead: train one reference LoRA trajectory, checkpoint it densely, and for each target R, rescale B only (exact, no retraining, no search) from whichever real checkpoint's own norm is closest — cross-checked against the real LR-sweep points above.

**Test**: reran the identical reference trajectory (same seed) at `checkpoint_interval=50` vs `=10`, to check whether disagreement against the real crosscheck points is a checkpoint-density/interpolation artifact or something more fundamental.

**Result: tightening checkpoint density did not close the gap at the R≈0.22 transition.** The rescale magnitude `k` (1.0 = no rescale) tightened from [0.746,1.063] to [0.971,1.012] — anchoring is now near-exact everywhere. Yet at R=0.220 — an exact match to the real lr=1e-4 crosscheck point, no interpolation needed — the rescaled-direction ΔNLL still overstates real damage by **12.4x (interval=50) → 14.4x (interval=10)**: unchanged, if anything slightly worse. Away from the transition, agreement is good and stable in both runs: R≈0.45 → ~1.2x, R≈0.85–1.74 (crosscheck point beyond both curves' tested range) → ~1.1x.

**Interpretation**: since `k` was already ≈1.02 at R=0.22 even before tightening, the ~12–14x gap isn't an interpolation-coarseness artifact — it looks like a genuine **direction-family** effect: the reference trajectory (ref_lr=5e-4) traces a different path through weight space than the real crosscheck run (lr=1e-4), even at matched displacement magnitude, and rescaling one direction's magnitude can't correct for that. This is a real limitation of the "one reference direction, rescaled" design, not a bug — and it lands exactly on the R≈0.2–0.25 transition zone T* estimation cares about most.

**Incidental finding**: the finer checkpointing exposed a genuinely sharp transition in the real trajectory — task accuracy jumps from 0.530 (step 470, chance-level) to 0.740 (step 510), 40 steps later. The coarser interval=50 grid had smoothed over this by mapping both target_R=0.15 and 0.18 onto the same step-500 checkpoint. This corroborates the original motivation for this whole redesign: the transition is genuinely abrupt in step-space, not just an LR-grid artifact.

**Practical implication**: this design is validated for locating the approximate transition region and for the coarse shape of R-vs-damage away from the transition (R≳0.45) — not for precise ΔNLL magnitude *at* R≈0.2–0.25, where it's confirmed to overstate real damage by an order of magnitude regardless of checkpoint density. Any magnitude claim near the transition should cite the real crosscheck point, not the rescaled curve. The rescaled curve is also a single, unreplicated run — small non-monotonicities within it (e.g. ΔNLL dips from +1.29 at R=0.33 to +0.81 at R=0.40) are single-run noise, not a fitted trend.

Raw data: `RAW_DATA.md` § "H1: Direct-R Sweep (Checkpoint-Anchored Rescale) — GPT-2 (2026-07-20)".

### Caveats
1. **Llama per-param steps (2000 < 3000) — not fully converged.** R values are slightly underestimated for Llama. But ΔNLL is the ground truth — the catastrophic jump at LR=2e-4 is real.
2. **eval_slack standardization breaks σ½_pre scale-invariance.** After recomputing with 1% relative slack, GPT-2 σ½_pre=0.007191 vs Llama σ½_pre=0.000772 (9.3× difference). Previous tables using 0.000603 / 0.000576 are INCORRECT for cross-model comparison. Saved to `h1_kl/gpt2/pretrained_density_relslack.json` and `h1_kl/meta-llama/Llama-3.2-3B/pretrained_density_relslack.json`.
3. **KL(wiki) remains informative** for both models — tracks ΔNLL monotonically. The failure claim is specifically about instruction-domain KL.
4. **5 conditions, no held-out test.** T* zone is wide due to sparse LR grid.

---

## H2 Experiments

**⚠️ Methodological note on T\* brackets (2026-07-17), applies to every T\*=[a,b] below:** these are grid-resolution brackets — the interval between the last tested LR that stayed safe and the next tested LR that forgot — not statistical confidence intervals (no repeated sampling, no CI procedure, except the MNLI 5-seed replication which is explicitly a raw seed-outcome count, not a CI either). "T* is indistinguishable / consistent / Phase-2-independent" throughout this section means *the transition falls in the same or an adjacent tested LR step across conditions* — a real, checkable fact given both conditions used the same grid — not a formal statistical equivalence claim. Standalone percentage-agreement figures (e.g. "0.2%", "17.9%", "within 4%") compare specific grid-matched points and largely reflect where the tested LR values happened to land, not measured precision — read them as illustrative, not as demonstrated precision.

### Phase 1 models

| Phase 1 Task | Model | Accuracy | σ½_A_nll | σ½_A_acc | Notes |
|---|---|---|---|---|---|
| MNLI | Llama-3.2-3B, lr=5e-5, rank=8, 2000 steps | 88.4% | 0.006607 (BROKEN) | **0.002857** (refined ±5%) | Acc-based is correct metric |
| SST-2 | Llama-3.2-3B, lr=5e-5, rank=8, 2000 steps | 97.5% | 0.011919 | **0.003337** (refined ±5%) | 21% higher than MNLI/AGNews |
| AGNews | Llama-3.2-3B, lr=5e-5, rank=8, 2000 steps | 90.5% | — | **0.002756** (refined ±5%) | Within 4% of MNLI |
| CoLA | Llama-3.2-3B, lr=5e-5, rank=8, 2000 steps | 86.8% | **0.0075** | **0.001944** (3.86× smaller than NLL) | Acc-based computed 2026-06-30 |

### σ½_A_acc refinement (MNLI Phase 1)

Old (coarse): σ½_A_acc = 0.003120 (jump from density=0.953 to 0.0 in one step)
New (refined): **σ½_A_acc = 0.002857** (7 additional sigma points added in transition zone)

Refined density curve:
| σ | density |
|---|---|
| 6.6e-5 | 1.000 |
| 1.3e-4 | 1.000 |
| 2.6e-4 | 1.000 |
| 5.3e-4 | 1.000 |
| 1.1e-3 | 1.000 |
| 2.1e-3 | 0.953 |
| 2.4e-3 | 0.867 |
| 2.6e-3 | 0.700 |
| 2.9e-3 | 0.453 |
| **3.2e-3** | **0.200** ← σ½ here |
| 3.4e-3 | 0.060 |
| 3.7e-3 | 0.000 |

σ½ uncertainty: ±5-10% (was ±30-50%). Command: `python recompute_sigma_half_acc.py --refine --n_refine 7 --n_perturb 150 --n_eval 200`

---

### H2-A: MNLI → SST-2 (updated 2026-07-04: stochastic boundary resolved via 5-seed run)

Phase 1: MNLI, σ½_A_acc = **0.002857**, baseline 88.4%, threshold 79.6%.

**✅ Stochastic boundary RESOLVED (Batch 4, 5-seed run at lr=1e-4 and lr=2e-4, rank=8):**

| LR | Seed | R_A | acc_A | Outcome |
|---|---|---|---|---|
| 1e-4 | 1 | 0.0405 | 0.846 | safe |
| 1e-4 | 2 | 0.0411 | 0.832 | safe |
| 1e-4 | 3 | 0.0401 | 0.856 | safe |
| 1e-4 | 4 | 0.0406 | 0.858 | safe |
| 1e-4 | 5 | 0.0412 | 0.860 | safe |
| **2e-4** | **1** | **0.0842** | **0.828** | **safe** |
| **2e-4** | **2** | **0.0799** | **0.816** | **safe** |
| **2e-4** | **3** | **0.0855** | **0.638** | **FORGET** |
| **2e-4** | **4** | **0.0842** | **0.786** | **FORGET** |
| **2e-4** | **5** | **0.0813** | **0.732** | **FORGET** |

**Result:** lr=1e-4 → 0/5 FORGET (acc range [0.832, 0.860], mean=0.850). lr=2e-4 → **3/5 FORGET** (acc range [0.638, 0.828], mean=0.760). The boundary is genuinely stochastic at lr=2e-4 — R_A ≈ 0.080–0.086 is right at the edge.

**T*(MNLI→SST-2, r8) = [0.041, 0.080–0.086]** (last safe = mean R_A at lr=1e-4 ≈ 0.041; first forget R_A at lr=2e-4 ≈ 0.080–0.086). This is **indistinguishable from MNLI→AGNews r8 = [0.040, 0.084]**. **MNLI Phase 1 is Phase-2-independent.** The earlier 2.7× apparent gap was a single-seed artifact.

| LR | Rank | per_param | R_A_acc | MNLI Acc | SST-2 Acc | Outcome |
|---|---|---|---|---|---|---|
| 1e-5 | 8 | 1.52e-5 | 0.005 | 0.884 | — | safe |
| 5e-5 | 8 | 5.92e-5 | 0.021 | 0.866 | — | safe |
| 1e-4 | 8 | 1.15e-4 | 0.040 | 0.848 | — | safe |
| 2e-4 | 8 | 2.35e-4 | 0.082 | **0.840** | — | safe (new run) |
| **5e-4** | **8** | **7.15e-4** | **0.250** | **0.352** | — | **catastrophic** |
| 1e-5 | 32 | — | — | 0.892 | — | safe |
| 5e-5 | 32 | 1.18e-4 | 0.041 | 0.822 | — | safe |
| **1e-4** | **32** | **2.44e-4** | **0.085** | **0.730** | — | **forget** |
| 2e-4 | 32 | — | — | 0.236 | — | catastrophic |
| 1e-4 | 4 | 1.74e-4 | 0.061 | — | — | safe |
| **2e-4** | **4** | **5.19e-4** | **0.182** | **0.338** | — | **catastrophic** |
| 5e-5 | 16 | 8.35e-5 | 0.029 | 0.894 | — | safe |
| **1e-4** | **16** | **1.64e-4** | **0.057** | **0.748** | — | **forget** |

**T* per rank (90% retention, threshold=0.796):**
- Rank 4: T* ∈ [0.061, 0.182] mid ≈ **0.121** — T*_raw=3.46e-4 ✓ (in H1 range)
- Rank 8: T* ∈ **[0.041, 0.081]** mid ≈ **0.058** — T*_raw≈1.16e-4 ✓ (matches AGNews) — **RESOLVED via 5-seed run**
- Rank 16: T* ∈ [0.029, 0.057] mid ≈ **0.043** — T*_raw=1.24e-4 ↓ (below H1 lower bound 1.70e-4)
- Rank 32: T* ∈ [0.041, 0.085] mid ≈ **0.063** — T*_raw=1.81e-4 ✓ (in H1 range)

**Rank pattern:** r8 and r32 both cluster at T*≈0.058–0.063 (indistinguishable). r16 more fragile (T*≈0.043). r4 more robust but wide interval (T*≈0.121).

**Phase-2 comparison (MNLI Phase 1, r8) — RESOLVED:**
- MNLI→SST-2 r8: T*≈[0.041, 0.081] — 5-seed confirmation
- MNLI→AGNews r8: T*=[0.040, 0.084]
- **Indistinguishable ✅** — MNLI Phase 1 T*(r8) is Phase-2-independent. Previous 2.7× gap was single-seed artifact at stochastic boundary.

---

### H2-B: MNLI → AGNews (updated 2026-07-04: r4 added, all ranks complete)

Same Phase 1 model as H2-A (σ½_A_acc = 0.002857). Baseline 88.4%, threshold 79.6%.

| LR | Rank | per_param | R_A_acc | MNLI Acc | AGNews Acc | Outcome |
|---|---|---|---|---|---|---|
| 1e-5 | 4 | 1.43e-5 | 0.0050 | 0.870 | — | safe |
| 5e-5 | 4 | 4.51e-5 | 0.0158 | 0.854 | — | safe |
| 1e-4 | 4 | 8.45e-5 | 0.0296 | 0.836 | — | safe |
| 2e-4 | 4 | 1.66e-4 | 0.0582 | 0.842 | — | safe |
| **5e-4** | **4** | **4.95e-4** | **0.1732** | **0.352** | — | **catastrophic** |
| 1e-5 | 8 | 1.74e-5 | 0.0061 | 0.874 | 0.894 | safe |
| 5e-5 | 8 | 6.04e-5 | 0.0212 | 0.840 | 0.910 | safe |
| 1e-4 | 8 | 1.15e-4 | 0.0404 | 0.864 | 0.896 | safe |
| **2e-4** | **8** | **2.39e-4** | **0.0837** | **0.764** | 0.910 | **significant** |
| 5e-4 | 8 | 7.44e-4 | 0.2606 | 0.356 | 0.912 | catastrophic |
| 1e-5 | 16 | 2.21e-5 | 0.0077 | 0.880 | 0.892 | safe |
| 5e-5 | 16 | 8.37e-5 | 0.0293 | 0.856 | 0.900 | safe |
| 1e-4 | 16 | 1.67e-4 | 0.0583 | 0.868 | 0.916 | safe |
| **2e-4** | **16** | **3.53e-4** | **0.1234** | **0.708** | 0.918 | **significant** |
| 5e-4 | 16 | 1.12e-3 | 0.3927 | 0.370 | 0.882 | catastrophic |
| 1e-5 | 32 | 2.75e-5 | 0.0096 | 0.856 | 0.904 | safe |
| 5e-5 | 32 | 1.19e-4 | 0.0416 | 0.818 | 0.906 | safe |
| **1e-4** | **32** | **2.39e-4** | **0.0835** | **0.720** | 0.914 | **significant** |
| 2e-4 | 32 | 5.11e-4 | 0.1787 | 0.340 | 0.906 | catastrophic |
| 5e-4 | 32 | 1.80e-3 | 0.6311 | 0.348 | 0.900 | catastrophic |

**T* per rank (90% retention, threshold=0.796):**
- Rank 4: T* ∈ [0.058, 0.173] mid ≈ **0.100** — T*_raw=1.66e-4 ✓
- Rank 8: T* ∈ [0.040, 0.084] mid ≈ **0.062** — T*_raw=1.15e-4 ✓
- Rank 16: T* ∈ [0.058, 0.123] mid ≈ **0.091** ← more robust than r8 and r32 (anomalous)
- Rank 32: T* ∈ [0.042, 0.084] mid ≈ **0.063** — same as r8 ✅

**r8 and r32 rank-invariant** ✅ (0.062 ≈ 0.063). r4 has wide interval (T*≈0.100) but T*_raw is in same 1.6e-4 range. r16 is unexpectedly higher — survives lr=1e-4 while r32 forgets there.

---

### H2-C: SST-2 → MNLI (updated 2026-07-01: high-LR sweep added, V5 closed)

Phase 1: SST-2, σ½_A_acc = **0.003337** (refined ±5%), **baseline 96.0%** (from phase1_meta.json), σ½_A_nll = 0.011919. Threshold = 0.9 × 0.960 = **0.864**.

⚠️ Old entry said baseline=97.5%. The current Phase 1 model checkpoint gives baseline=96.0%. All T* values below use 0.864 threshold.

**R_A_acc table (complete sweep, ranks 8 and 32):**

| LR | Rank | per_param | R_A_acc | SST-2 Acc | MNLI Acc | Outcome |
|---|---|---|---|---|---|---|
| 1e-5 | 8 | 1.61e-5 | 0.0048 | 0.960 | 0.796 | safe |
| 1e-5 | 32 | 2.73e-5 | 0.0082 | 0.960 | 0.838 | safe |
| 5e-5 | 8 | 6.16e-5 | 0.0184 | 0.934 | 0.852 | safe |
| 5e-5 | 32 | 1.19e-4 | 0.0356 | 0.944 | 0.876 | safe |
| 1e-4 | 8 | 1.20e-4 | 0.0358 | 0.944 | 0.820 | safe |
| **1e-4** | **32** | **2.35e-4** | **0.0704** | **0.786** | 0.872 | **forgot** |
| 2e-4 | 8 | 2.43e-4 | 0.0728 | 0.940 | 0.836 | safe |
| 2e-4 | 32 | 4.98e-4 | 0.1491 | 0.826 | 0.838 | forgot (non-monotone) |
| 5e-4 | 8 | 7.09e-4 | 0.2123 | 0.900 | 0.852 | safe ← old upper bound |
| 5e-4 | 32 | 1.59e-3 | 0.4754 | 0.522 | 0.444 | catastrophic |
| **1e-3** | **8** | **2.84e-3** | **0.8512** | **0.478** | **0.356** | **💥 catastrophic** ← NEW |
| 1e-3 | 32 | 4.92e-3 | 1.4754 | 0.522 | 0.332 | catastrophic |
| 2e-3 | 8 | 5.50e-3 | 1.6475 | 0.478 | 0.332 | catastrophic |
| 2e-3 | 32 | 8.41e-3 | 2.5193 | 0.478 | 0.352 | catastrophic |
| 5e-3 | 8 | 9.89e-3 | 2.9641 | 0.478 | 0.352 | catastrophic |
| 5e-3 | 32 | 1.16e-2 | 3.4796 | 0.478 | 0.352 | catastrophic |

**T* per rank (threshold = 0.864):**
- Rank 8: T* ∈ **[0.212, 0.851]** mid ≈ **0.532** — V5 CLOSED ✅
- Rank 32: T* ∈ **[0.036, 0.070]** mid ≈ **0.053**

**Rank-invariance does not hold — extreme 10× gap.** r8 survives to R_A_acc=0.212 while r32 forgets at R_A_acc=0.070. At the same displacement level (R_A_acc≈0.07): r8 acc=0.940 (safe) vs r32 acc=0.786 (forgot).

**T\*\_raw:** r8 = [7.09e-4, 2.84e-3] mid=**1.77e-3** — 10× above the ~1.75e-4 cluster for MNLI/AGNews/CoLA. SST-2 requires 10× more absolute displacement before forgetting (vs MNLI Phase 2); SST-2→AGNews requires 60× less (T*≈0.027). The anomaly is extreme in both directions depending on Phase 2 task.

---

### H2-D: AGNews → MNLI (updated 2026-07-02: r4+r16 added)

Phase 1: AGNews, σ½_A_acc = **0.002756** (refined ±5%), baseline 91.0%, success threshold 81.9%.

| LR | Rank | R_A_acc | AGNews Acc | MNLI Acc | Outcome |
|---|---|---|---|---|---|
| 1e-5 | 8 | 0.006 | 0.884 | — | safe |
| 5e-5 | 8 | 0.022 | 0.866 | — | safe |
| 1e-4 | 8 | 0.042 | 0.878 | — | safe |
| **2e-4** | **8** | **0.085** | **0.574** | — | **catastrophic** |
| 5e-4 | 8 | 0.265 | 0.352 | — | catastrophic |
| 1e-5 | 32 | 0.010 | 0.908 | — | safe |
| 5e-5 | 32 | 0.042 | 0.822 | — | safe |
| **1e-4** | **32** | **0.083** | **0.650** | — | **catastrophic** |
| 1e-4 | 4 | 0.030 | 0.920 | — | safe |
| **2e-4** | **4** | **0.061** | **0.786** | — | **forget** |
| 5e-5 | 16 | 0.030 | 0.894 | — | safe |
| **1e-4** | **16** | **0.058** | **0.780** | — | **forget** |

**T* per rank (90% retention, threshold=0.819):**
- Rank 4: T* ∈ [0.030, 0.061] mid ≈ **0.045** — T*_raw=1.25e-4 ↓ (below H1 lower bound)
- Rank 8: T* ∈ [0.042, 0.085] mid ≈ **0.064** — T*_raw=1.76e-4 ✓
- Rank 16: T* ∈ [0.030, 0.058] mid ≈ **0.044** — T*_raw=1.22e-4 ↓ (below H1 lower bound)
- Rank 32: T* ∈ [0.042, 0.083] mid ≈ **0.062** — T*_raw=1.72e-4 ✓

**Rank pattern:** r8 and r32 are Phase-2-independent (T*≈0.062–0.064) and their T*_raw is in the H1 range. r4 and r16 are significantly more fragile (T*≈0.044–0.045, forget one LR bracket earlier) and fall below the H1 raw range.

**r8 is anomalously robust for AGNews Phase 1** — it survives an extra LR bracket (safe at lr=1e-4, forget at lr=2e-4) while r4, r16, and r32 all forget at or before lr=1e-4. The rank-robustness is non-monotone: r8 > r32 > r16 ≈ r4.

**Phase-2 confirmation:** AGNews Phase 1 r8 T*≈0.064 (→MNLI) vs T*≈0.065 (→SST-2 from H2-E) — **indistinguishable** ✅. AGNews is the cleanest Phase-2-independent task.

---

### H2-E: AGNews → SST-2 (r4+r16 added 2026-07-03)

Phase 1: same AGNews model as H2-D, σ½_A_acc = **0.002756** (refined, shared with H2-D).

| LR | Rank | R_A_acc | AGNews Acc | SST-2 Acc | Outcome |
|---|---|---|---|---|---|
| 1e-5 | 4 | 0.0046 | 0.912 | 0.918 | safe |
| 5e-5 | 4 | 0.0161 | 0.904 | 0.948 | safe |
| 1e-4 | 4 | 0.0307 | 0.906 | 0.960 | safe |
| **2e-4** | **4** | **0.0615** | **0.540** | 0.966 | **catastrophic** |
| 5e-4 | 4 | 0.1867 | 0.258 | 0.950 | catastrophic |
| 1e-5 | 8 | 0.0057 | 0.908 | 0.936 | safe |
| 5e-5 | 8 | 0.0220 | 0.876 | 0.958 | safe |
| 1e-4 | 8 | 0.0419 | 0.864 | 0.954 | safe |
| **2e-4** | **8** | **0.0887** | **0.664** | 0.914 | **catastrophic** |
| 5e-4 | 8 | 0.2696 | 0.276 | 0.944 | catastrophic |
| 1e-5 | 16 | 0.0075 | 0.918 | 0.942 | safe |
| 5e-5 | 16 | 0.0299 | 0.890 | 0.958 | safe |
| 1e-4 | 16 | 0.0603 | 0.844 | 0.964 | safe |
| **2e-4** | **16** | **0.1242** | **0.376** | 0.920 | **catastrophic** |
| 5e-4 | 16 | 1.0104 | 0.268 | 0.776 | catastrophic |
| 1e-5 | 32 | 0.0095 | 0.894 | 0.954 | safe |
| 5e-5 | 32 | 0.0426 | 0.824 | 0.966 | safe |
| 1e-4 | 32 | 0.0841 | 0.716 | 0.958 | **catastrophic** |
| 2e-4 | 32 | 0.1856 | 0.352 | 0.946 | catastrophic |
| 5e-4 | 32 | 0.8156 | 0.250 | 0.686 | catastrophic |

**T* per rank (with refined σ½=0.002756):**
- Rank 4: T* ∈ [0.031, 0.062] mid ≈ **0.047**
- Rank 8: T* ∈ [0.042, 0.089] mid ≈ **0.065**
- Rank 16: T* ∈ [0.060, 0.124] mid ≈ **0.092** — stochastic boundary (AGNews→MNLI r16 forgets at same R, different outcome)
- Rank 32: T* ∈ [0.043, 0.084] mid ≈ **0.063**
- **r4, r8, r32 rank-invariant** ✅; r16 stochastic at boundary

**AGNews Phase 1 T* confirmed as Phase 1 property (2026-07-03):** All 4 ranks × 2 Phase 2 tasks now complete. r4/r8/r32 midpoints cluster at 0.047–0.065. H2-D (→MNLI) and H2-E (→SST-2) give T*(r8)=0.062 vs 0.065 (<5%), T*(r32)=0.062 vs 0.063 (<2%). r16 appears higher (0.092) but at the same R value as AGNews→MNLI r16 which shows FORGET — true T*(r16) is near the boundary of both observations. **AGNews is the cleanest Phase-1-property evidence in the dataset.**

---

### H2-F: SST-2 → AGNews (updated 2026-07-04: full 4-rank table from Batch 4)

Phase 1: SST-2, σ½_A_acc = **0.003337** (refined), baseline 97.5%, success threshold 87.75%.

| LR | Rank | per_param | R_A_acc | SST-2 Acc | AGNews Acc | Outcome |
|---|---|---|---|---|---|---|
| 1e-5 | 4 | 1.40e-5 | 0.0042 | 0.930 | — | safe |
| 5e-5 | 4 | 4.54e-5 | 0.0136 | 0.930 | — | safe |
| 1e-4 | 4 | 8.39e-5 | 0.0251 | 0.950 | — | safe |
| 2e-4 | 4 | 1.70e-4 | 0.0509 | 0.962 | — | safe |
| **5e-4** | **4** | **4.94e-4** | **0.1479** | **0.744** | — | **catastrophic** |
| 1e-5 | 8 | 1.71e-5 | 0.0051 | 0.898 | — | safe |
| 5e-5 | 8 | 6.13e-5 | 0.0184 | 0.948 | — | safe |
| 1e-4 | 8 | 1.19e-4 | 0.0357 | 0.950 | — | safe |
| **2e-4** | **8** | **2.43e-4** | **0.0730** | **0.864** | — | **catastrophic** |
| 5e-4 | 8 | 7.54e-4 | 0.2259 | 0.648 | — | catastrophic |
| 1e-5 | 16 | 2.15e-5 | 0.0064 | 0.902 | — | safe |
| 5e-5 | 16 | 8.40e-5 | 0.0252 | 0.940 | — | safe |
| 1e-4 | 16 | 1.67e-4 | 0.0499 | 0.924 | — | safe |
| **2e-4** | **16** | **3.52e-4** | **0.1056** | **0.822** | — | **catastrophic** |
| 5e-4 | 16 | 1.11e-3 | 0.3339 | 0.564 | — | catastrophic |
| 1e-5 | 32 | 2.75e-5 | 0.0082 | 0.944 | — | safe |
| 5e-5 | 32 | 1.16e-4 | 0.0348 | 0.948 | — | safe |
| **1e-4** | **32** | **2.38e-4** | **0.0714** | **0.820** | — | **catastrophic** |
| 2e-4 | 32 | 5.21e-4 | 0.1560 | 0.528 | — | catastrophic |
| 5e-4 | 32 | 1.74e-3 | 0.5211 | 0.480 | — | catastrophic |

**T* per rank (threshold = 0.8775):**
- Rank 4: T* ∈ [0.051, 0.148] mid ≈ **0.087** — T*_raw=1.70e-4 ✓
- Rank 8: T* ∈ [0.036, 0.073] mid ≈ **0.051** — T*_raw=1.19e-4 ✓
- Rank 16: T* ∈ [0.050, 0.106] mid ≈ **0.073** — T*_raw=1.67e-4 ✓
- Rank 32: T* ∈ [0.035, 0.071] mid ≈ **0.050** — T*_raw=1.16e-4 ✓

**Observation:** All 4 ranks have T*_raw at last safe in 1.16–1.70e-4 range — same as MNLI/AGNews Phase 1 pairs. SST-2 Phase 1 T*_raw is NOT anomalous for the →AGNews direction. The anomaly is entirely specific to the →MNLI direction.

**Critical finding — SST-2 Phase 1 T* is NOT a Phase 1 property:**
- SST-2→MNLI r8: T* ∈ [0.212, 0.851] mid≈**0.425** (V5 closed)
- SST-2→AGNews r8: T* ∈ [0.036, 0.073] mid≈**0.051**
- Ratio: **~8.3×** (corrected from earlier ~20× which used old data with different Phase 1 checkpoint)
- T*_raw at last safe: SST-2→MNLI r8 = 7.07e-4; SST-2→AGNews r8 = 1.19e-4 — **5.9× raw gap**; anomaly confirmed independent of σ½

Rank 32: T*(r32, SST-2→MNLI) ≈ 0.053 vs T*(r32, SST-2→AGNews) ≈ 0.050 — **consistent!** The Phase-2-dependence at r32 is much smaller than at r8. The ~20× gap reported earlier for r32 was from stale data (old Phase 1 checkpoint gave catastrophic from lr=1e-5). **Current data shows SST-2→AGNews r32 T*=0.050 ≈ SST-2→MNLI r32 T*=0.053 — the r32 anomaly resolves.**

---

### H2-G: CoLA → MNLI (new, 2026-06-30)

Phase 1: CoLA, σ½_A_acc = **0.001944** (computed 2026-06-30), σ½_A_nll = 0.0075 (3.86× larger). Baseline 86.8%, success threshold 78.1%.

| LR | Rank | R_A_acc | R_A_nll | CoLA acc | MNLI acc | Outcome |
|---|---|---|---|---|---|---|
| 1e-5 | 8 | 0.0075 | 0.0019 | 0.874 | 0.574 | safe |
| 1e-5 | 32 | 0.0139 | 0.0036 | 0.840 | 0.810 | safe |
| 5e-5 | 8 | 0.0307 | 0.0079 | 0.858 | 0.800 | safe |
| 5e-5 | 32 | 0.0605 | 0.0157 | 0.836 | 0.878 | safe |
| 1e-4 | 8 | 0.0595 | 0.0154 | 0.828 | 0.804 | safe |
| 1e-4 | 32 | 0.1185 | 0.0307 | 0.810 | 0.880 | safe |
| **2e-4** | **8** | **0.1183** | 0.0307 | **0.706** | 0.852 | **significant** |
| 2e-4 | 32 | 0.2596 | 0.0673 | 0.682 | 0.844 | significant |
| **5e-4** | **8** | **0.3555** | 0.0921 | **0.318** | 0.872 | **catastrophic** |
| 5e-4 | 32 | 3.0617 | 0.7935 | 0.318 | 0.316 | catastrophic (both tasks) |

**T* per rank (acc-based R_A, σ½_A_acc=0.001944):**
- Rank 8: T* ∈ [0.060, 0.118] mid ≈ **0.089**
- Rank 32: T* ∈ [0.119, 0.260] mid ≈ **0.189**
- **NOT rank-invariant** (2× difference)

---

### H2-H: CoLA → AGNews (new, 2026-06-30)

Phase 1: same CoLA model as H2-G, σ½_A_nll = **0.0075**, σ½_A_acc = **0.001944** (exact, same checkpoint). Baseline 86.8%, success threshold 78.1%. R_A_acc = R_A_nll × 3.858 (exact: 0.0075/0.001944).

| LR | Rank | R_A_nll | R_A_acc | CoLA acc | AGNews acc | Outcome |
|---|---|---|---|---|---|---|
| 1e-5 | 8 | 0.0024 | 0.0093 | 0.856 | 0.874 | safe |
| 1e-5 | 32 | 0.0037 | 0.0143 | 0.854 | 0.888 | safe |
| 5e-5 | 8 | 0.0082 | 0.0316 | 0.824 | 0.890 | safe |
| 5e-5 | 32 | 0.0156 | 0.0602 | 0.788 | 0.900 | safe |
| 1e-4 | 8 | 0.0155 | 0.0598 | 0.814 | 0.896 | safe |
| **1e-4** | **32** | **0.0308** | **0.1188** | **0.746** | 0.906 | **significant** |
| **2e-4** | **8** | **0.0326** | **0.1258** | **0.622** | 0.912 | **significant** |
| 2e-4 | 32 | 0.0685 | 0.2642 | 0.680 | 0.914 | significant |
| 5e-4 | 8 | 0.0980 | 0.3780 | 0.664 | 0.900 | significant |
| 5e-4 | 32 | 0.2419 | 0.9330 | 0.322 | 0.872 | catastrophic |

**T* per rank (acc-based R_A; NLL-based R_A converted via σ½_acc/σ½_nll=3.86×):**
- Rank 8: T*(acc) ∈ [0.060, 0.126] mid ≈ **0.093**
- Rank 32: T*(acc) ∈ [0.060, 0.119] mid ≈ **0.089**
- **Rank-invariant** ✅ (within 5%)

**CoLA rank 8 T*(acc) ≈ 0.089–0.093 — consistent across MNLI and AGNews Phase 2 (within 4%) and ABOVE MNLI/AGNews T*(r8)≈0.062.** CoLA is MORE robust per normalized displacement, not less. The apparent lower T* in NLL-based comparison was purely an artifact of CoLA σ½_A_nll being 3.86× its σ½_A_acc. This BREAKS the binary/multi-class hypothesis: CoLA (binary) has a stable, Phase 1-consistent T*(r8) that is comparable to or higher than multi-class tasks.

---

### H2-I: SST-2 → CoLA (new, 2026-07-05)

Phase 1: SST-2, σ½_A_acc = **0.003337**, baseline = 97.5%, threshold = 0.8775.

| LR | Rank | R_A_acc | SST-2 Acc | CoLA Acc | Outcome |
|---|---|---|---|---|---|
| 1e-5 | 8 | 0.0046 | 0.888 | 0.762 | **safe** |
| 5e-5 | 8 | 0.0187 | 0.640 | 0.884 | **FORGET** |
| 1e-4 | 8 | 0.0363 | 0.664 | 0.882 | FORGET |
| 2e-4 | 8 | 0.0737 | 0.582 | 0.876 | FORGET |
| 5e-4 | 8 | 0.2354 | 0.478 | 0.858 | FORGET |
| 1e-5 | 32 | 0.0082 | 0.744 | 0.858 | **FORGET** |
| 5e-5 | 32 | 0.0358 | 0.534 | 0.888 | FORGET |
| 1e-4 | 32 | 0.0739 | 0.528 | 0.864 | FORGET |
| 2e-4 | 32 | 0.1741 | 0.516 | 0.876 | FORGET |
| 5e-4 | 32 | 2.1669 | 0.478 | 0.512 | FORGET |

**T*(r8) = [0.0046, 0.0187]  mid≈0.012  T*_raw≈3.88e-5** — 4.7× **below** the 1.73–1.81e-4 cluster for all other pairs.

**T*(r32) < 0.0082  T*_raw < 2.75e-5** — forgets at the lowest LR tested (lr=1e-5). Gets WORSE at higher rank, opposite of SST-2→MNLI.

**Key comparison vs other SST-2 Phase 1 pairs at r8:**

| Phase 2 | T* | T*_raw | vs cluster |
|---|---|---|---|
| CoLA | [0.005, 0.019] | 3.88e-5 | 4.7× below |
| AGNews | [0.036, 0.073] | 1.81e-4 | ✓ in cluster |
| MNLI | [0.212, 0.851] | 1.77e-3 | 10× above |

**44× range** from CoLA to MNLI. SST-2 Phase 1 has a strongly anisotropic forgetting basin. σ½ (isotropic) predicts T* correctly only for the AGNews direction. CoLA Phase 2 always finds the "short axis" (extreme fragility); MNLI Phase 2 at r8 finds the "long axis" (extreme resistance). At r32, MNLI normalizes to cluster; CoLA becomes even more extreme.

---

### Cross-experiment summary: T* comparison (9+ pairs, updated 2026-07-05)

**Threshold:** 90% of Phase 1 baseline accuracy (acc_slack=0.10). T* = last R_A where Phase 1 accuracy ≥ threshold.

All T* values use acc-based R_A. σ½_A_acc=0.001944 applies to both CoLA pairs — σ½_A is a Phase 1 model property, not Phase 2 task-dependent. CoLA→AGNews R_A_acc = R_A_nll × 3.858 (exact arithmetic, not an estimate).

| Task Pair | Phase 1 acc | σ½_A_acc | T* (rank 4) | T* (rank 8) | T* (rank 16) | T* (rank 32) | Rank-inv? |
|---|---|---|---|---|---|---|---|
| MNLI→SST-2 | 88.4% | 0.002857 | [0.061,0.182] **≈0.121** | **[0.041,0.081] ≈0.058** ✅ resolved | [0.029,0.057] **≈0.043** | [0.041,0.085] **≈0.063** | ✅ r8≈r32; r16 fragile |
| MNLI→AGNews | 88.4% | 0.002857 | [0.058,0.173] **≈0.100** | [0.040, 0.084] **≈0.062** | [0.058, 0.123] **≈0.091** ⚠️ | [0.042, 0.084] **≈0.063** | ✅ r8≈r32; r16 high anomaly |
| AGNews→MNLI | 91.0% | 0.002756 | [0.030,0.061] **≈0.045** ↓ | [0.042, 0.085] **≈0.064** | [0.030,0.058] **≈0.044** ↓ | [0.042,0.083] **≈0.062** | r8≈r32 ✅; r4≈r16 fragile |
| AGNews→SST-2 | 91.0% | 0.002756 | [0.031,0.062] **≈0.044** | [0.042, 0.089] **≈0.065** | [0.060,0.124] **≈0.086** ⚠️ | [0.043, 0.084] **≈0.063** | ✅ r8≈r32; r16 stochastic |
| SST-2→MNLI | **96.0%** | 0.003337 | — | [0.212, 0.851] **≈0.425** ✅ V5 closed | — | [0.036, 0.070] **≈0.050** | ❌ r8 only (8×) |
| SST-2→AGNews | 96.0% | 0.003337 | [0.051,0.148] **≈0.087** | [0.036,0.073] **≈0.051** | [0.050,0.106] **≈0.073** | [0.035,0.071] **≈0.050** | ✅ r8≈r16≈r32≈0.050–0.073 |
| CoLA→MNLI | 86.8% | 0.001944 | — | [0.060, 0.118] **≈0.089** | — | [0.119, 0.260] **≈0.189** | ❌ (2×) |
| CoLA→AGNews | 86.8% | 0.001944 | — | **≈0.093** | — | **≈0.089** | ✅ |
| AGNews→MNLI (**Mistral-7B**) | 91.0% | **7.62×10⁻⁴** | — | [0.043,0.094] **≈0.071** | — | — | r8 only |
| AGNews→SST-2 (**Mistral-7B**) | 91.0% | **7.62×10⁻⁴** | — | [0.038,0.046] **≈0.042** ⚠️corrected | — | — | r8 only |

↓ = T*_raw below H1 lower bound (1.70e-4); ⚠️ = anomalous value relative to task's other ranks or Phase 2 tasks

**σ½_A_acc across Phase 1 models (all computed):** MNLI=0.002857, AGNews=0.002756, SST-2=0.003337, CoLA=0.001944. MNLI and AGNews within 4%; SST-2 21% higher; CoLA 32% lower than MNLI.

**T* as Phase 1 model property — updated verdict (Batch 4, 2026-07-04):**

| Phase 1 | T*(r8, Phase 2 A) | T*(r8, Phase 2 B) | r8 verdict | r32 verdict |
|---|---|---|---|---|
| MNLI | **0.058** (→SST-2, 5-seed) | **0.062** (→AGNews) | ✅ **indistinguishable** | 0.063 vs 0.063 ✅ |
| AGNews | **0.064** (→MNLI) | **0.065** (→SST-2) | ✅ indistinguishable | 0.062 vs 0.063 ✅ |
| SST-2 | **0.425** (→MNLI) | **0.051** (→AGNews) | ❌ **~8.3× gap** | ✅ 0.050 vs 0.050 |
| CoLA | **0.089** (→MNLI) | **0.093** (→AGNews) | ✅ (+4%) | ❌ (2×) |

**MNLI resolved (Batch 4):** 5-seed run at the stochastic boundary confirms T*(MNLI→SST-2 r8) ≈ [0.041, 0.081], matching MNLI→AGNews [0.040, 0.084] exactly. The 2.7× gap was a single-seed artifact. MNLI is now Phase-2-independent at r8.

**SST-2 r32 corrected (Batch 4):** New clean Phase 1 run shows SST-2→AGNews r32 T*=0.050 ≈ SST-2→MNLI r32 T*=0.050 — r32 is now consistent. The anomaly is localized to r8 (~8.3×). The earlier ">6× r32 gap" used stale data from a different Phase 1 checkpoint where r32 was catastrophic from lr=1e-5.

**Rank pattern summary (updated 2026-07-02):**
- AGNews Phase 1: r8 and r32 track together (~0.063); r4 and r16 are significantly more fragile (~0.044). r8 is anomalously robust compared to all other ranks.
- MNLI Phase 1: r8 and r32 roughly consistent within MNLI→AGNews. r16 shows opposite behavior across Phase 2 tasks (most robust →AGNews, more fragile →SST-2). r16 anomaly is Phase-2-task-specific, NOT a MNLI Phase 1 property.
- SST-2 Phase 1: extreme Phase-2-dependence, both ranks.
- CoLA Phase 1: r8 invariant, r32 not.

**r16 resolution (2026-07-02):** The r16 anomaly from MNLI→AGNews (T*=0.091, most robust) does NOT replicate for MNLI→SST-2 (T*=0.043, less robust than r8). Conclusion: r16 robustness is Phase 2 AGNews-specific, not a MNLI Phase 1 property.

**Acc-based T*(r8) across tasks:** MNLI≈0.062, AGNews≈0.063, CoLA≈0.089–0.093. CoLA is 40–50% higher than MNLI/AGNews — more robust per normalized displacement.

**SST-2 Phase 1 → T* is NOT a Phase 1 property (neither rank).** SST-2→MNLI r8: T*≈0.532; SST-2→AGNews r8: T*≈0.027. Ratio **~20×**. T*(r32): 0.053 vs <0.008 for AGNews, >6×.

**Binary/multi-class hypothesis BROKEN by CoLA.** Both SST-2 and CoLA are binary, but CoLA T*(r8) is consistent and SST-2 T*(r8) is not. Task *content* determines forgetting, not class count.

**Rank 8 invariance holds in 3 of 4 tasks (MNLI, AGNews, CoLA), fails for SST-2. Rank 32 invariance holds for MNLI and AGNews** (corrected); breaks for CoLA and catastrophically for SST-2.

**Directional vulnerability 1/T*(r8):** MNLI ≈ 16×, AGNews ≈ 15×, CoLA ≈ 11× (acc-based), SST-2 Phase 2-dependent.

**Catastrophic collapse** (Phase 1 near-random ~25–35%) occurs at R_A ≈ 0.22–0.28 for MNLI and AGNews Phase 1. For CoLA, catastrophic at R_A_acc ≈ 0.36 (rank 8 vs MNLI).

**T\*\_raw convergence (2026-07-01):** Per-param displacement at T* in absolute units (before σ½ normalization), rank 8:

**Rank 8, from current JSON (Batch 2):**

| Pair | T\*\_raw (midpoint) | H1 range [1.70e-4, 4.48e-4]? |
|---|---|---|
| MNLI→AGNews r8 | 1.15e-4 | ✅ |
| MNLI→SST-2 r8 | **1.16e-4** | ✅ — **resolved via seeds; last safe per_param ≈ 1.16e-4** |
| AGNews→MNLI r8 | 1.76e-4 | ✅ |
| AGNews→SST-2 r8 | 1.80e-4 | ✅ |
| SST-2→AGNews r8 | 1.19e-4 | ✅ |
| CoLA→MNLI r8 | 1.73e-4 | ✅ |
| CoLA→AGNews r8 | 1.80e-4 | ✅ |
| **SST-2→MNLI r8** | **1.77e-3** | ❌ **10× above the cluster** |

**Non-r8 values (Batch 2, 2026-07-02):**

| Pair | T\*\_raw (midpoint) | H1 range? |
|---|---|---|
| MNLI→SST-2 r4 | 3.46e-4 | ✅ (inside) |
| MNLI→SST-2 r16 | 1.24e-4 | ↓ below lower bound |
| MNLI→SST-2 r32 | 1.81e-4 | ✅ |
| AGNews→MNLI r4 | 1.25e-4 | ↓ below lower bound |
| AGNews→MNLI r16 | 1.22e-4 | ↓ below lower bound |
| AGNews→MNLI r32 | 1.72e-4 | ✅ (just inside) |

**T*_raw findings (updated 2026-07-05):**
- At **r8**: 7 of 9 measured pairs cluster at **1.73–1.81e-4** ✓. Cluster: MNLI→SST-2 (~1.74e-4 seed-confirmed), MNLI→AGNews (1.77e-4), AGNews→MNLI (1.76e-4), AGNews→SST-2 (1.80e-4), CoLA→MNLI (1.73e-4), CoLA→AGNews (1.80e-4), SST-2→AGNews (1.81e-4).
- **SST-2→MNLI r8**: T*_raw = 1.77e-3 — **10× above** cluster. MNLI Phase 2 training pushes along SST-2's long axis.
- **SST-2→CoLA r8**: T*_raw = 3.88e-5 — **4.7× below** cluster. CoLA Phase 2 always finds SST-2's short axis. At r32, T*_raw < 2.75e-5 (forgets at lr=1e-5).
- **SST-2→AGNews r8 and r32**: both in cluster (1.81e-4 and 1.77e-4). The anisotropy is directional, not a scalar property of SST-2 Phase 1.
- **SST-2→MNLI r32**: 1.77e-4 — **in cluster**. MNLI anomaly resolves at r32; richer LoRA spans the fragile subspace.
- **Conclusion:** T*_raw convergence holds for 7/9 r8 pairs. Both outliers are SST-2 Phase 1 in opposite directions — one 10× above, one 4.7× below — evidence for basin anisotropy rather than a scalar anomaly.

---

### H2-J: AGNews → MNLI (Mistral-7B — cross-architecture validation, 2026-07-11)

Phase 1: AGNews on Mistral-7B-v0.1, acc=0.91, σ½=**7.62×10⁻⁴**, lr=5e-5, rank=8, steps=2000.

| LR | T*_R_A | T*_step | raw per_param at T* | final_acc_A | Status |
|---|---|---|---|---|---|
| 1e-5 | — | — | 1.49e-5 (survived) | 0.884 | survived |
| 5e-5 | **0.0708** | 1550 | 5.39e-5 | 0.786 | forgot |
| 1e-4 | **0.0942** | 650 | 7.18e-5 | 0.666 | forgot |
| 2e-4 | **0.0430** | 50 | 3.28e-5 | 0.244 | forgot (<step100) |
| 5e-4 | 0 (step=0) | — | — | 0.244 | <step50 unresolved |
| 1e-3 | 0 (step=0) | — | — | 0.240 | <step50 unresolved |

Valid T* (non-zero): lr=5e-5 (R_A=0.071), lr=1e-4 (R_A=0.094), lr=2e-4 (R_A=0.043). Median T*_R_A ≈ **0.071**, range [0.043, 0.094].

**Cross-architecture comparison (both models, AGNews→MNLI, rank 8) — ORIGINAL (methodology mismatch):**

| Model | σ½_A (AGNews) | raw per_param at T* | R_A at T* |
|---|---|---|---|
| Llama-3.2-3B | 2.756×10⁻³ (**acc-based**) | 1.74×10⁻⁴ | **0.064** |
| Mistral-7B | 7.62×10⁻⁴ (**NLL-based** — no acc σ½ existed at the time) | ~5.4×10⁻⁵ | **0.071** |

σ½ differs **3.6×**; raw per_param at T* differs **3.2×**; R_A agrees within **11%**. **⚠️ This comparison mixed methodologies** — Llama's σ½ was accuracy-based, Mistral's was NLL-based, because no accuracy-based σ½ existed for Mistral at the time. Not a true like-for-like comparison.

**⚠️ CORRECTED 2026-07-17 — re-run on the pod with a genuinely consistent methodology.** Computed Mistral's own accuracy-based σ½_A via `recompute_sigma_half_acc.py` (σ½_A_acc = 7.314×10⁻⁴, only 1.04× different from its NLL-based value — much closer than Llama's 2.3× acc/NLL gap for MNLI). Recomputed T* for both AGNews Phase-2 targets using the actual accuracy threshold (0.81) rather than the script's own `R_A>1.0` "Forgot?" column, **which is wrong** — it printed "no" for rows where accuracy had already collapsed from 0.90 to 0.24-0.26, because R_A never crossed 1.0 except at the most extreme LR tested. Same leftover-R_A=1-cutoff bug identified earlier in `h2_sequential_llama.py`'s plotting code, now confirmed live in `recompute_sigma_half_acc.py`'s summary table too.

| Model | Pair | σ½_A_acc | T* (acc-based, both models) |
|---|---|---|---|
| Llama-3.2-3B | AGNews→MNLI | 2.756×10⁻³ | [0.042, 0.085] |
| Mistral-7B | AGNews→MNLI | 7.314×10⁻⁴ | [0.020, 0.085] |
| Llama-3.2-3B | AGNews→SST-2 | 2.756×10⁻³ | [0.042, 0.089] |
| Mistral-7B | AGNews→SST-2 | 7.314×10⁻⁴ | [0.068, 0.105] |

**Both pairs now overlap** (they didn't both overlap under the old mixed-methodology comparison — only →MNLI did, and even that one wasn't computed consistently). AGNews→MNLI's first-forget LR step lands at nearly the same R for both architectures (Llama 0.085 vs Mistral 0.0852); AGNews→SST-2's lands further apart (Llama 0.089 vs Mistral 0.105). **⚠️ The "0.2%"/"17.9%" framing used earlier overstates what this is** — both numbers compare specific grid-matched points from 5-point LR sweeps, not measured quantities with that precision; treat them as "same tested LR step, close by" and "same tested LR step, further apart," not as demonstrated agreement to a fraction of a percent. What's real and load-bearing here: both pairs overlap on a genuinely consistent (acc-based, both models) methodology, which is the actual improvement over the old mismatched comparison.

**Caveat carried forward:** Mistral's σ½_A_acc density curve is coarse (jumps from density=1.0 to 0.0 between only two tested σ values — no `--refine` pass run yet), so its precision is closer to Llama's original ±25–50% pre-refine uncertainty than to Llama's refined ±5–10%. The overlap findings above are robust to this (both brackets overlap with room to spare), but a tighter refine pass would clarify whether AGNews→SST-2's wider gap between first-forget grid steps reflects genuine architecture difference or just measurement coarseness on the Mistral side.

---

### H2-K: AGNews → SST-2 (Mistral-7B, 2026-07-11)

Phase 1: same as H2-J (AGNews, Mistral-7B), σ½=7.62×10⁻⁴, acc=0.91.

| LR | T*_R_A | T*_step | final_acc_A | final_acc_B (SST-2) | Status |
|---|---|---|---|---|---|
| 1e-5 | — | — | 0.902 | 0.960 | survived |
| 5e-5 | **0.0123** | 50 | 0.882 | 0.956 | forgot (step 50–100) |
| 1e-4 | **0.0383** | 150 | 0.256 | 0.522 | forgot (step 150–200) |
| 2e-4 | **0.0458** | 50 | 0.242 | 0.478 | forgot (<step100) |
| 5e-4 | 0 (step=0) | — | 0.240 | 0.478 | <step50 unresolved |
| 1e-3 | 0 (step=0) | — | 0.240 | 0.478 | <step50 unresolved |

Valid T* range: **[0.038, 0.046], median ≈ 0.042** ⚠️ corrected — lr=5e-5 excluded (noisy survived run, transient dip artifact; final_acc=0.882).

**Phase 2 task comparison (same Mistral AGNews Phase 1, rank 8):**

| Phase 2 | T* range | Median | per_param median | Notes |
|---|---|---|---|---|
| MNLI (H2-J) | [0.043, 0.094] | 0.071 | 5.4e-5 | 3 clean forgetting conditions (lr=5e-5: 0.071, lr=2e-4: 0.043, lr=1e-4: 0.094) |
| SST-2 (H2-K) | **[0.038, 0.046]** | **0.042** | 2.9e-5 | ⚠️ **Corrected** — 2 clean conditions only (lr=1e-4, lr=2e-4) |

**⚠️ H2-K T* correction (2026-07-11):** Raw JSON shows lr=5e-5 is NOT a clean forgetting event. Trajectory: acc dips below threshold at step 100 (0.812) then immediately recovers to 0.888 at step 150, bounces at steps 1600/1700, recovers at 1650/1850. **Final acc=0.882 (survived).** The script recorded tstar_R_A=0.012 from the transient step-100 dip — this is a measurement artifact, not a real forgetting event. The two clean forgetting conditions are lr=1e-4 (tstar R_A=0.038, then permanent collapse at step 200) and lr=2e-4 (tstar R_A=0.046, collapses by step 100). Corrected T*(H2-K) = **[0.038, 0.046], median=0.042** (not [0.012, 0.046], median=0.038 as previously documented).

At lr=2e-4: T*(→MNLI)=0.043, T*(→SST-2)=0.046 — nearly identical. SST-2 Phase 2 causes slightly earlier forgetting than MNLI (mild, within 10%). Both are within the same order of magnitude.

---

### NLL-vs-Accuracy Forgetting — probing whether the SST-2 anomaly is a classifier-readout artifact (GPT-2, 2026-07-22)

**Motivation**: V8 (SST-2 anomaly, 44× spread across Phase-2 targets) has four null mechanistic hypotheses and no positive explanation. All of H2's existing forgetting measurements route through a frozen Phase-1 classifier evaluated against continuously Phase-2-drifted hidden states — structurally the same "old probe vs new representation" pattern already found and fixed as Bug 4 in `h1_direct_r_sweep.py`. If different Phase-2 tasks reorganize the shared representation differently, the same underlying Phase-1 information could look "forgotten" at wildly different R_A depending on which task is doing the reorganizing, without the information itself being lost at different rates. New script `h2_nll_forgetting.py` (does not modify `h2_sequential_llama.py`) tests this by tracking NLL on Task A's own held-out text — no classifier involved — alongside the existing accuracy signal, for the same SST-2 Phase 1 checkpoint against MNLI/AGNews/CoLA as Phase 2 targets (GPT-2, cheap validation pass, not yet run on Llama).

**Result: NLL-based forgetting never triggers, anywhere in the sweep — but not because NLL is insensitive.** Raw trajectories (not just the summary table) show NLL dropping substantially — from baseline 7.44 down to ~4.5-4.9 (35-40% improvement) — in exactly the two conditions (AGNews, CoLA at higher LRs) where accuracy collapses to chance (~0.52). The NLL improvement is visible before accuracy shows any damage (e.g. AGNews lr=5e-4, step 25: accuracy still above baseline at 0.810, NLL already down 10%). This is the opposite of what "forgetting" would predict from NLL, not merely an absence of signal. For MNLI — the one Phase-2 target where accuracy-based forgetting *also* never triggers at any tested LR, echoing the already-documented SST-2→MNLI anomaly generalizing to GPT-2 — neither signal moves far from baseline. **⚠️ Flagged 2026-07-22: "never triggers" uses H1's 1%-of-baseline-NLL threshold, reused here without independent calibration for this task-domain NLL signal** — the qualitative finding (NLL moves the *opposite* direction, not merely under-threshold) doesn't depend on the exact threshold value, but the "anywhere/never" framing borrows more precision from that number than it has earned for this different quantity.

**Interpretation — reframed 2026-07-22, same day as the original writeup, after closer scrutiny of the magnitude pattern.** The original interpretation below read this as "consistent with readout-mismatch, with an unruled-out alternative." That gave the hopeful reading too much standing. Two problems, on reflection:

1. **Instrument mismatch.** H2's actual question is whether sentiment is still linearly decodable from the current hidden state via *some* readout — a representational-geometry question. NLL-on-domain-text asks whether the model is fluent at predicting SST-2-style text — an adjacent but different question, with no necessary connection to whether the sentiment-relevant direction survived. NLL-on-domain-text does not directly test whether sentiment was forgotten; it tests something next to it.
2. **The magnitude pattern favors the less flattering explanation, not just fails to rule it out.** Re-examining the deltas: NLL improves substantially and durably in both collapse conditions (~26-29% below baseline by the end) but is transiently *worse* than baseline in the one condition that stays safe (MNLI), settling to only a mild ~6% improvement. **The size of the NLL improvement scales with the size of the accuracy collapse.** Pure readout-mismatch (representation intact, only the old probe direction miscalibrated) predicts NLL should move roughly independently of how much accuracy damage occurred — it doesn't predict this correlation. A mode-collapse mechanism (aggressive high-LR training pushing the model toward degenerate/repetitive output, which can spuriously score low perplexity on short generic sentences without preserving any real information) produces exactly this signature — both effects downstream of the same cause, tracking each other in magnitude — with no sentiment information necessarily surviving anywhere. Given the correlation, this alternative is at least as parsimonious as readout-mismatch, not a secondary possibility that "hasn't been ruled out."

**The resolving test — `h2_probe_recovery.py`, run 2026-07-23**: freeze the Phase-2-perturbed backbone at each condition's endpoint, train a *fresh* linear probe on it from scratch, and measure recovery speed on Task-A (SST-2) accuracy. Decision rule (recovers within ≤25 probe steps / recovers slower / never recovers) was fixed in the script before running, not chosen after seeing results.

**Result: all three conditions (sst2→mnli/agnews/cola, lr=5e-4) land in the middle bucket — `recovers_slow`.** None is `no_recovery` and none is `recovers_fast`. A fresh probe trained on the frozen, Phase-2-perturbed backbone reaches *above* the 90%-of-baseline threshold in all three conditions within 200 steps (3,200 SST-2 examples) — including AGNews and CoLA, where the old classifier had collapsed to near-chance (0.536, 0.522). CoLA's recovered probe (0.808) actually exceeds the original Phase-1 baseline (0.774). This is real support for the readout-mismatch explanation over pure mode-collapse (information is not destroyed for any of these three targets — it's recoverable, and recoverable well above baseline, using a small fraction of Phase-1's own training cost), but recovery isn't instant either (55–200 steps, not ≤25) — something real happened to the representation, just not irreversibly. This rules out mode-collapse-as-total-loss for these three GPT-2 conditions specifically, though it doesn't explain the sizeable spread in recovery speed (CoLA 3.6× faster than MNLI) or why CoLA's recovered accuracy exceeds baseline.

One reproduction-fidelity caveat carried over honestly: this run's own MNLI old-classifier accuracy (0.670) doesn't closely match the already-documented NLL-experiment trajectory's nearby point (0.754 at step 475 vs. this run's step 500) — CoLA and AGNews both reproduce almost exactly (gaps of 0.000 and 0.004), so this looks like it's within MNLI's own already-noisy trajectory rather than a new bug, but it means the MNLI-specific recovery number should be trusted less than AGNews/CoLA's. Single GPT-2 run, no seed replication, not yet run on Llama where the actual 44× anomaly (V8) was observed — this result is suggestive, not a resolution of V8 itself. Full data: `RAW_DATA.md § "H2: Fresh-Probe Recovery — GPT-2"`.

**Severity sweep, GPT-2 (2026-07-23) — a real recovery-speed gradient, but the "does recovery ever truly fail" question remains open.** Sweeping `sst2→cola` across LRs both below and above the already-tested 5e-4 (`--phase2_lrs 1e-4 2e-4 5e-4 1e-3 2e-3 5e-3`) shows recovery getting monotonically harder with severity: 55 steps to recover at 1e-4/2e-4/5e-4, 200 steps (the edge of the probe-training budget) at 1e-3 — a real, previously undemonstrated relationship between displacement and recovery difficulty. At 2e-3 and 5e-3, the verdict flips to `no_recovery` — but **this is very likely a training-divergence artifact (probable NaN-corrupted weights from an overly aggressive LR), not evidence of genuine information destruction**: `probe_acc` is bit-identical across all 41 evaluation points in both conditions, and the previously-trained classifier and a freshly, independently-initialized probe give identical accuracy to 4 decimal places — something only expected if the underlying hidden states themselves have degenerated (e.g. to NaN, where `argmax` returns a constant index regardless of classifier weights), not a case of "training that failed to find signal." Not confirmed by direct inspection (no endpoint checkpoints were saved), so treat as an open, flagged suspicion, not a settled diagnosis. **Net effect: the recovery-difficulty gradient (1e-4→1e-3) is a real, trustworthy new finding; whether a genuine no-recovery regime exists at all remains unresolved** — the two data points that would show it are exactly the two suspected of being corrupted. A rerun with finer LR steps between 1e-3–2e-3 and an explicit NaN check would resolve this. Full data: `RAW_DATA.md § "H2: Fresh-Probe Recovery — Severity Sweep, GPT-2"`.

**Llama-3.2-3B, same design (2026-07-23) — a more mixed picture than GPT-2, and the first result to touch V8 on the model where the anomaly actually lives.** Ran `sst2→{mnli,agnews,cola}` at two LRs (5e-4, 1e-3), reusing the existing, unmodified Llama SST-2 Phase-1 checkpoint directly (no retraining). Five of six conditions land in the categorical `no_recovery` bucket, but the raw recovery curves split cleanly into two very different phenomena — **the categorical label alone is misleading and shouldn't be quoted without the underlying numbers**:

- **At lr=5e-4** (agnews, cola): real, substantial, noisy-but-genuine recovery curves (agnews peaks at 0.902, cola climbs to 0.754) — qualitatively identical in shape to GPT-2's successful recovery curves. They land in `no_recovery` only because Llama's baseline (0.96) sets a much stricter 90%-threshold (0.864) than GPT-2's (0.6966), not because nothing happened. mnli/5e-4 fully recovers (`recovers_slow`, 0.914).
- **At lr=1e-3** (all three targets): a qualitatively different, reproducible collapse into a non-learning, near-chance oscillation (bouncing between ~2 nearby values for the full 200-step budget, no upward trend at any point). Critically, **this is independently corroborated by data already on record in this project**: the exact value `0.478` for SST-2→MNLI at lr=1e-3, rank 8 already appears in the pre-existing `h2_sequential_llama.py` fine-tuning table (a completely different script, run months earlier) — meaning this collapse is a real, reproducible property of this training configuration, not a one-off artifact of this new script. What's *not* resolved is the mechanism: training-numerical instability (the same class of concern flagged for GPT-2's most extreme LRs) vs. genuine representational collapse at this severity — the evidence here is weaker for the NaN-specific diagnosis than GPT-2's case (only 1 of 3 conditions shows the clean "old and fresh classifiers agree to 4 decimals" signature). If this turns out to be genuine collapse rather than an artifact, it would implicate some of H2's *existing*, already-cited Llama accuracy numbers at lr≥1e-3 as sitting in the same degenerate regime, not just this new experiment.

**Net effect on V8**: at moderate severity, Llama's readout-mismatch evidence looks as strong as GPT-2's — real recovery, just against a stricter bar. At higher severity, something changes for all three targets simultaneously, but whether that's "information genuinely becomes harder to recover" (extending the readout-mismatch story into a to-be-characterized failure regime) or a training artifact unrelated to the V8 question is not yet settled. Single run, single seed, only 2 LR points per target (the true transition per target is bracketed, not localized). Full data: `RAW_DATA.md § "H2: Fresh-Probe Recovery — Llama-3.2-3B"`.

Raw data: `RAW_DATA.md § "H2: NLL-vs-Accuracy Forgetting — GPT-2 Validation"`.

---

## What Claims Can Be Made Now

**Rewritten from scratch 2026-07-17** to state the final, corrected claims directly rather than as a chain of "originally X, but ⚠️ corrected to Y" patches. All numbers below are the corrected/final ones; the version history of what was wrong before now lives only in the dated correction notes earlier in this file and in `RAW_DATA.md`.

### H1 — Pretraining→SFT boundary

**Core claim:** R = ‖Δθ‖/σ½_pre, where σ½_pre is computed once from the frozen pretrained checkpoint via isotropic-noise perturbation (no fine-tuning required), tracks the onset of general-capability degradation (WikiText-2 NLL, MMLU) during LoRA fine-tuning across GPT-2 (117M), Llama-3.2-3B (3B), and Mistral-7B-v0.1 (7B) — a 60× parameter range. It does this better than the natural competing signal, KL-divergence on instruction-style prompts, which fails to discriminate safe from unsafe fine-tuning in all three architectures.

**What's solid:**
- Raw displacement at the point general capability starts degrading spans **39–44×** across the three architectures (1.58×10⁻³ down to 3.53×10⁻⁵ at the respective T* lower edges); σ½_pre itself spans **39×**. Normalizing by σ½ compresses this down to roughly a **3× band**, under any of the "broken" criteria tested. This compression is real and is the most defensible part of the H1 claim.
- KL(diverse) fails to track the onset in every model tested. Llama saturates in the safe zone (KL=7.11 at LR=1e-5, already near its ceiling of 10.67 at catastrophic LR). Mistral does not saturate (starts at 0.088) but has insufficient slope at the actual cliff (1.41→3.71, only 2.6× while ΔNLL jumps 22.5×). These are two different, and non-identical, failure modes — not the same mechanism in both models, and neither is a clean single story.
- MNLI fine-tuning: after correcting both models to a consistent relslack σ½ and a consistent "ΔNLL > 1% of baseline NLL" break criterion, Llama's T*=[0.133, 0.233] falls **entirely inside** Mistral's T*=[0.071, 0.324]. This is the cleanest cross-architecture convergence result in the H1 dataset.

**What does not hold as cleanly as it first appears:**
- The SST-2 "3-way convergence" (T* ≈ [0.19, 0.24] for all three models) is **criterion-sensitive**. Under a loose "eyeball the biggest ΔNLL jump" rule it holds narrowly (GPT-2 [0.22,0.45], Llama [0.24,0.59], Mistral [0.19,0.23]). Under the stricter, architecture-independent 1%-of-baseline-NLL rule — the same rule that correctly resolved the MNLI comparison above — the brackets become GPT-2 [0.22,0.45], Llama [0.11,0.24], Mistral [0.03,0.19]: a **chain** (GPT-2∩Llama and Llama∩Mistral each overlap) but **not a 3-way overlap** (GPT-2∩Mistral = ∅). Mistral's strict lower bound also rests on a non-converged LR point (SST-2 accuracy 0.48, near-random at lr=1e-5) — a genuine gap in the tested grid, not a resolved number. The defensible claim is the 39–44×→~3× compression, not a precise simultaneous 3-way overlap.
- "Forgetting" is metric-dependent: general-capability degradation (NLL, MMLU) occurs at much lower R than task-specific accuracy degradation for the same fine-tune (Llama SST-2 accuracy survives to R≈3.6 while NLL/MMLU have already collapsed by R≈0.24–0.59). R predicts a specific failure mode, not one universal collapse point.
- "R=1 holds" is false — degradation begins at R≈0.02–0.6 depending on model and task, always well below 1. Any residual framing of R=1 as a threshold (including leftover plotting code) should be treated as retired.

**Open validity questions, not yet resolved:**
- **σ½/LoRA scope mismatch — RUN 2026-07-21/22, resolved, and this one DOES change reported numbers (unlike the Hessian question above).** Recomputed σ½ (`recompute_h1_sigma_relslack.py --sigma_scope lora`) restricted to LoRA's actual target modules, all 3 models, on the frozen pretrained checkpoint. **The predicted asymmetry is confirmed exactly**: GPT-2 (whose `c_proj` LoRA target collides with its MLP naming, so its LoRA scope already overlapped much of the block) moves only +7.0% (σ½: 0.007191→0.007698). Llama and Mistral (pure attention-only LoRA targets, no collision) both move dramatically more: +96.6% (0.000772→0.001517) and +92.4% (0.0001824→0.000351) respectively. All three confirmed not capped (`sigma_half_is_lower_bound: false`).

  **This changes the cross-architecture T* comparison materially, not just its framing.** Recomputing R and T* with the corrected σ½ (same raw `per_param_norm`/ΔNLL data, 1%-of-baseline-NLL break criterion, full numbers in `RAW_DATA.md`): GPT-2's SST-2 T* barely moves ([0.220,0.453]→[0.206,0.424]), but Llama's and Mistral's roughly halve (Llama [0.113,0.236]→[0.058,0.120]; Mistral [0.030,0.194]→[0.016,0.101]). Under the *existing* block-scope σ½, the three models formed a chain (GPT-2∩Llama overlap, Llama∩Mistral overlap, GPT-2∩Mistral empty) — already a weaker claim than "3-way convergence," per the earlier criterion-sensitivity correction. **Under the corrected lora-scope σ½, GPT-2 no longer overlaps with *either* other architecture at all** (GPT-2∩Llama = ∅, GPT-2∩Mistral = ∅) — only Llama∩Mistral still overlaps ([0.058,0.101]). The chain breaks at the GPT-2 end entirely.

  **What this means:** the σ½/LoRA-scope mismatch wasn't just a footnote — it was inflating GPT-2's apparent similarity to the other two architectures. Once corrected, GPT-2 requires genuinely more relative displacement (in R units) before catastrophic forgetting than Llama or Mistral do; only Llama and Mistral remain close to each other. This doesn't kill the core H1 claim (raw displacement still spans tens of ×, and normalizing by *either* σ½ convention still compresses that range substantially for the Llama/Mistral pair), but the "all three architectures converge near R≈0.2" framing cannot be maintained under the more methodologically correct (LoRA-scoped) σ½ — **a decision is needed on which σ½ convention (block or lora-scoped) the paper adopts as primary**, since they now tell different cross-architecture stories, not just different-precision versions of the same one. Full per-LR recompute and the pairwise overlap arithmetic: `RAW_DATA.md § "σ½/LoRA-Scope Correction"`.

  **Data-hygiene bug found during this recompute**: two conflicting copies of Llama's SST-2 LR-sweep results existed locally at different paths (`h1_kl/meta-llama/Llama-3.2-3B/` — stale, pre-"KL(SuperNI)+MMLU" Batch 3 — vs `h1_kl/meta-llama/meta-llama/Llama-3.2-3B/` — authoritative, matches this document's own cited numbers exactly). The stale copy was used by mistake in the first pass of this recompute before being caught by checking the double-nested path's extra fields (`kl_superni`, `mmlu_acc`) against this document's own table. Corrected before any number above was finalized; the stale copy should be archived/deleted to prevent recurrence.
- **`hessian_eigenvalue.py` — RUN 2026-07-20/21, resolved (partially).** Tests whether σ½ ∝ 1/√λ_max, i.e. whether the checkpoint sits in a genuine curvature basin, via Lanczos-computed extremal eigenvalues restricted to LoRA's own target modules. **GPT-2: confirmed, reproducibly, across 3 independent runs (λ_min=-36.05, λ_max^+=28.42, agreeing to within 0.005 every time) — the checkpoint sits at a genuine saddle point, not a local minimum**, and the single-eigenvalue quadratic theory overestimates σ½ by ~7.6×. **⚠️ Flagged 2026-07-22: these GPT-2 runs were computed on a parameter set that predates the bias-exclusion fix now baked into `_get_lora_scoped_names`** (56,669,184 params vs. the current, corrected 56,623,104 — a difference of exactly 46,080, matching GPT-2's per-layer bias count across `c_attn`/`attn.c_proj`/`mlp.c_proj`×12 layers; see `RAW_DATA.md § "Hessian Eigenvalue Check"` for the arithmetic). Bias is 0.08% of the scoped parameters — expected to be inconsequential to the saddle-point conclusion, but this has not been independently reverified on the corrected scope. **Llama and Mistral: not measurable with this method.** Llama never passed a 20%-relative eps-doubling stability check across 4 attempts (34-65% swings, non-monotonic in eps — ruling out "just needs a bigger step size"). Mistral briefly appeared to pass (4.9% swing, one run) but **failed identically on a same-settings rerun (50% swing, eigenvalue magnitude ~2× different)** — proving the first "pass" was random-seed luck, not a validated result; that earlier reading is retracted. Full data and the bug list (8 real, sequentially-discovered issues — OOM, bf16 underflow, restore-drift, a spurious convergence bug, a gradient-checkpointing footgun, and a seed-confound in the stability check itself) is in `RAW_DATA.md § "Hessian Eigenvalue Check"`.

  **Net effect on framing:** since the one model we could reliably verify contradicts the clean-basin picture, "certified basin radius" cannot stand as a literal geometric claim for any of the three models — soften to "an empirically observed noise-tolerance radius." This is not a retreat relative to the field: [Chen et al. 2025, "Unveiling the Basin-Like Loss Landscape in Large Language Models" (arXiv:2505.17646)](https://arxiv.org/abs/2505.17646) — the closest prior work, using the same randomized-smoothing basis — defines "basin" purely by empirical noise-tolerance (Gaussian-perturb-and-check-performance) and never checks Hessian curvature or saddle-vs-minimum structure at all. Our softened language is consistent with how this term is already used in the literature, and the GPT-2 saddle-point finding is arguably a more rigorous check than that prior work attempts. This finding changes no T*/R/σ½ number already reported anywhere in this document — only this framing claim.

*Mistral v5 data source: `h1_kl/mistralai/Mistral-7B-v0.1_sst2_v5/` (PRIMARY — not v1–v4, which are archived and superseded).*

### H2 — Sequential task forgetting

**Core claim:** σ½_A — an accuracy-based radius computed once from a Phase-1 checkpoint's own held-out validation accuracy under random perturbation, no Phase-2 training required — defines a forgetting threshold T*_A that transfers to predict forgetting under a *different, untested* Phase-2 task.

**What's solid:**
- Across 4 Phase-1 tasks (MNLI, AGNews, SST-2, CoLA) × 2 ranks (8 of 4 seed cells), **5 of 8 cells clearly pass** transfer (the forgetting transition lands in the same or adjacent tested LR step across Phase-2 targets — not a statistical equivalence claim, see the standing methodological note above), 1 is weak/ambiguous (CoLA at r8 — intervals overlap but each spans 2×), and 2 clearly fail (SST-2 at r8, CoLA at r32, transitions several grid-steps apart — a real gap, not a resolution artifact). MNLI's r8 transfer result is additionally confirmed via 5-seed replication at the stochastic boundary (lr=2e-4: 3/5 seeds forget, 2/5 survive, both within the same R_A range as the alternative Phase-2 target).
- **Cross-architecture transfer, corrected 2026-07-17 with a genuinely consistent methodology:** the original Llama-vs-Mistral comparison used an accuracy-based σ½ for Llama but only an NLL-based σ½ for Mistral (no accuracy-based version existed yet). Recomputing Mistral's accuracy-based σ½_A on the same AGNews Phase-1 checkpoint (σ½_A_acc=7.314×10⁻⁴) and comparing on a consistent basis: both AGNews→MNLI and AGNews→SST-2 now overlap. AGNews→MNLI's first-forget LR step lands close for both models (Llama 0.085, Mistral 0.0852); AGNews→SST-2's lands further apart (Llama 0.089, Mistral 0.105). Treat these as "same tested grid step, close" vs. "same tested grid step, further apart" — not as precision figures ("0.2%"/"17.9%") the 5-point LR grid cannot actually support. What's real: both pairs overlap on a genuinely consistent methodology, which is where σ½ demonstrably earns its keep.

**What does not hold as cleanly as it first appears:**
- **The sharpest standing objection, verified by exact arithmetic:** within a single architecture (Llama), across different Phase-1 tasks (MNLI, AGNews, CoLA), the *raw, unnormalized* per-parameter displacement threshold is nearly constant — 1.73–1.81×10⁻⁴, under 5% spread. The σ½-normalized R_A values for the same three tasks spread **~53%** (0.061 to 0.093), because σ½_A itself varies mildly by task. For most of the H2 experimental matrix, σ½ normalization is not obviously adding value over "this architecture has a roughly constant raw displacement budget" — its clearest value-add is specifically the cross-architecture comparison above, where raw units alone would not transfer (σ½ ratio 3.6× between architectures, and R_A converges despite it).
- The SST-2 exception is not simply "one anisotropic task" — it's rank-dependent in a way that isn't explained: SST-2 fails transfer at rank 8 (44× spread across 3 Phase-2 targets: [0.005,0.019]→CoLA, [0.036,0.073]→AGNews, [0.212,0.851]→MNLI) but resolves at rank 32; CoLA does the *opposite* (transfers at r8, fails at r32, 2× gap). Four mechanistic hypotheses have been tested for why SST-2 is anisotropic and all came back null: gradient-direction cosine overlap (≈±0.0005, indistinguishable from random noise in a 4.5M-dim space), effective rank of LoRA ΔW (SST-2/AGNews/CoLA all ≈5.2, within 3%), binary-vs-multiclass task type (broken by CoLA, which is binary but transfers cleanly), and a subspace-projection variant of direction overlap (Pearson ≈ −0.12, but only 3 data points — this is underpowered, not a confirmed null).
- R_A=1 as a literal universal threshold is false; observed onsets range from ≈0.02 to ≈0.53 depending on task and rank, always well below 1.
- T* intervals are typically ~2× wide given the coarse 5-point LR grid. Claims of "within 4–5%" precision between conditions (e.g. CoLA T*(r8) midpoints 0.089 vs 0.093) are not supported when the underlying intervals each span 2× — the honest claim is "indistinguishable," not a precise percentage match.
- R_A is not strictly monotone at every evaluation point (e.g. SST-2→MNLI rank 8 accuracy ticks up from 0.934 at R=0.018 to 0.944 at R=0.036). The defensible claim is that R_A predicts the *onset and severity* of forgetting, not literal monotonicity.

**Confirmed tooling bug, not a data bug:** both `h2_sequential_llama.py`'s live trajectory tracker and `recompute_sigma_half_acc.py`'s summary-table printer use a naive "forgot?" check (`R_A_acc > 1.0`, or first-observed-failure locks in T* with no debounce) instead of the actual accuracy threshold. Confirmed live in the 2026-07-17 Mistral acc-based rerun: the printed "Forgot(acc)?" column said "no" for rows where accuracy had already collapsed from 0.90 to 0.24–0.26. The underlying JSON values (`final_acc_task_a`, `R_A_acc`) are correct; the console/plot display built on top of them is not — any figure or table generated by these scripts' own summary/plot functions should be re-derived from the raw per-condition accuracy values, not read off the "forgot" column or the R_A=1 threshold line directly.

---

## What's Next

### ✅ Completed (as of 2026-07-04)
- [x] H1: GPT-2 + Llama-3.2-3B, monotone R
- [x] H1 σ½_pre standardized to 1% relative slack: GPT-2=0.007191, Llama=0.000772 (9.3× — NOT scale-invariant)
- [x] H1 R-values recomputed with corrected σ½_pre (2026-07-01): GPT-2 T*∈[0.22,0.45], Llama T*∈[0.24,0.58], ranges overlap ← new
- [x] H2-A: MNLI→SST-2 (rank-invariant, T*(r8)≈0.061)
- [x] H2-B: MNLI→AGNews (T*(r8)≈0.062; r32 varies 2× by Phase 2 task)
- [x] H2-C: SST-2→MNLI (T*(r8)>0.212; T*(r32)≈0.111; σ½ refined to 0.003337)
- [x] H2-D: AGNews→MNLI (rank-invariant, T*(r8)≈0.062; σ½ refined to 0.002756)
- [x] H2-E: AGNews→SST-2 (rank-invariant, T*(r8)≈0.065; σ½ shared with H2-D)
- [x] H2-F: SST-2→AGNews (T*(r8)≈0.027; r32 catastrophic from lowest LR)
- [x] H2-G: CoLA→MNLI (T*(r8)≈0.089 acc-based; r32 not consistent) ← new
- [x] H2-H: CoLA→AGNews (T*(r8)≈0.093 acc-based; rank-invariant) ← new
- [x] CoLA σ½_A_acc = 0.001944 computed (2026-06-30) ← new
- [x] Binary/multi-class hypothesis tested — BROKEN by CoLA ← new
- [x] Direction overlap analysis (analyze_weight_directions.py) — NULL RESULT ← new
- [x] 4×2 task pair matrix complete (4 Phase 1 models × 2 Phase 2 tasks each)
- [x] Effective rank of LoRA ΔW — NULL RESULT (2026-06-30) ← new
- [x] V7 closed: full-model norm verification — H1 and H2 use identical per_param_norm formula; T\*\_raw converges to ~1.75e-4 across all 6 H2 pairs and H1 lower bound (2026-07-01) ← new
- [x] V5 closed: SST-2→MNLI high-LR sweep (lr=[1e-3,2e-3,5e-3]) — T*(r8)=[0.212,0.851] mid=0.532; ~20× anomaly vs AGNews (2026-07-01) ← new
- [x] MNLI→AGNews r16 added — T*(r16)=[0.058,0.123] mid=0.091; r8≈r32≈0.062; r16 anomalously more robust (2026-07-01) ← new
- [x] H2-B r32 corrected from JSON (was 0.131 from different Phase 1 run, now 0.063); MNLI now rank-invariant for BOTH r8 and r32 ← new
- [x] MNLI→SST-2 r4+r16 added (Batch 2, 2026-07-02): r4 T*=0.121 ✓ H1 range; r16 T*=0.043 ↓ below H1; r8 T*=0.166 (stochastic boundary — earlier run gave 0.061) ← new
- [x] AGNews→MNLI r4+r16 added (Batch 2, 2026-07-02): r4 T*=0.045 ↓ below H1; r8 most robust (safe at lr=2e-4); r16 T*=0.044 ↓ below H1; AGNews r8+r32 Phase-2-independent ✓ ← new
- [x] r16 anomaly resolved (2026-07-02): MNLI→AGNews r16 more robust (T*=0.091) vs MNLI→SST-2 r16 less robust (T*=0.043) → anomaly is Phase 2 AGNews-specific, not a MNLI Phase 1 property ← new
- [x] H1-SST-2 extended with KL(SuperNI) + MMLU (Batch 3, 2026-07-03): KL(SuperNI) monotone but poor threshold discrimination; MMLU drops at same R as NLL ← new
- [x] H1-MNLI complete (Batch 3, 2026-07-03): R monotone, KL(diverse) non-monotone (drops at lr=1e-4), V6 CLOSED ← new
- [x] AGNews→SST-2 r4+r16 added (Batch 4, 2026-07-03): r4 T*≈0.047, r16 T*≈0.092 (stochastic boundary); AGNews H2 now 4 ranks × 2 Phase 2 tasks = 8 conditions all consistent ← new
- [x] MNLI stochastic boundary resolved (Batch 4, 2026-07-04): 5-seed run at lr=2e-4 r8 gives 3/5 FORGET; T*(MNLI→SST-2 r8) = [0.041, 0.081] = T*(MNLI→AGNews r8) = [0.040, 0.084] — MNLI is Phase-2-independent ✅ ← new
- [x] MNLI→AGNews all 4 ranks (Batch 4, 2026-07-04): r4 T*=[0.058,0.173], r8=[0.040,0.084], r16=[0.058,0.123], r32=[0.042,0.084]; T*_raw all in 1.15–1.67e-4 range ← new
- [x] SST-2→AGNews all 4 ranks (Batch 4, 2026-07-04): r4 T*=[0.051,0.148], r8=[0.036,0.073], r16=[0.050,0.106], r32=[0.035,0.071]; T*_raw all in 1.16–1.70e-4 — same cluster as other Phase 1 tasks; SST-2 anomaly localized to →MNLI direction only ← new
- [x] SST-2 anomaly revised: ~8.3× gap (not ~20×); r32 anomaly resolves (new Phase 1 checkpoint SST-2→AGNews r32 T*=0.050 ≈ SST-2→MNLI r32 T*=0.050); r8 remains the sole anomalous rank (8.3×) ← new
- [x] T*_raw convergence now holds for 7/8 r8 pairs (1.15–1.80e-4); SST-2→MNLI r8 sole outlier at 7.07e-4 ← new
- [x] SST-2→CoLA (Batch 5, 2026-07-05): T*(r8)=[0.005,0.019] T*_raw=3.88e-5 (4.7× BELOW cluster); T*(r32)<0.0082 T*_raw<2.75e-5; 44× range across 3 Phase 2 tasks; basin anisotropy framing adopted ← new
- [x] SST-2→MNLI r32 resolved: T*=[0.036,0.070] T*_raw=1.77e-4 (in cluster); MNLI anomaly is rank-8-specific only ← new
- [x] Mistral-7B AGNews→MNLI (Batch 6, 2026-07-11): σ½=7.62e-4 (3.6× smaller than Llama); T*(r8) median=0.071 vs Llama T*(r8)=0.064 — within 11%; raw displacement 3.2× different; σ½ normalization recovers universal threshold; V6 architecture concern substantially addressed ← new
- [x] Mistral-7B AGNews→SST-2 (Batch 6, 2026-07-11): T*(r8) corrected to [0.038, 0.046] median=0.042; lr=5e-5 excluded (noisy survived run); 2 clean forgetting conditions only ← new
- [x] H1 Mistral-7B SST-2 (Batch 7, 2026-07-11): σ½_pre=1.559×10⁻⁴, T*=[0.406,0.636], overlaps with GPT-2 [0.22,0.45] and Llama [0.24,0.58]; 3-architecture T* convergence for SST-2 fine-tuning ← new
- [x] H1 Mistral-7B MNLI (Batch 7, 2026-07-11): NLL cliff at R=0.380 (vs Llama safe through R=3.570); task acc holds through R=0.874 before both collapse at R=1.013; KL(diverse) not saturated but plateaus ← new — **⚠️ these R values are non-relslack, corrected 2026-07-17 to Mistral=[0.071,0.324], Llama=[0.133,0.233] (fully contained, not "strikingly different"); see line ~111 and "What Claims Can Be Made Now"**
- [x] KL(SuperNI) bug found + fixed for Mistral H1 MNLI (v2, 2026-07-11): real KL(SuperNI) monotone, 7× jump safe→catastrophic, higher than KL(diverse) at cliff ← new
- [x] MNLI v2 SCP'd to local: h1_kl/mistralai/Mistral-7B-v0.1_mnli/ (v2 ✅); SST-2 v1 preserved as Mistral-7B-v0.1_sst2_v1/, v2 failure as Mistral-7B-v0.1_sst2_v2/ ← new
- [x] Mistral relslack σ½_pre computed (Batch 8, 2026-07-13): σ½=1.824×10⁻⁴ (+17% vs non-relslack 1.559×10⁻⁴); density cliff at σ∈[0.0001,0.0002] explains small shift ← new
- [x] Mistral SST-2 v3 batch_size=4 (Batch 8, 2026-07-13): partial result — correct training (batch=4), but wrong σ½ methodology (validation split). T*(relslack)=[0.347,0.397] SUPERSEDED by v5 ← archived
- [x] **Mistral SST-2 v5 PRIMARY (Batch 9, 2026-07-14): effective batch=16 (batch=4 × accum=4), test-split sequential σ½=1.824e-4 CORRECT.** T*=[0.194,0.226]; overlaps GPT-2 at [0.22,0.23]; 44× raw displacement span compressed to 1.2× by σ½. R strictly monotone. KL(SuperNI) discriminating 12.4× at cliff; KL(diverse) insufficient 2.6×. ← new PRIMARY RESULT

**Direction overlap null result:** Cosine similarity between Phase 1 LoRA weights (Δθ_A) and Phase 2 gradient direction (g_B) ≈ ±0.0005 for all 6 pairs — indistinguishable from random in 4.5M-dimensional LoRA space (expected noise = 1/√4.5M ≈ 0.00047). Simple gradient cosine similarity does not discriminate forgetting from non-forgetting pairs. Dropped as mechanistic explanation.

**Effective rank null result:** Mean effective rank of ΔW = B@A across 112 LoRA layers: SST-2=5.209, AGNews=5.360, CoLA=5.187. All three Phase 1 models are nearly identical in effective rank distribution (within 3%). SST-2 does NOT have anomalously low effective rank. The "concentrated 1D subspace" mechanistic hypothesis for SST-2's Phase-2-dependent T* is NOT supported. Dropped as mechanistic explanation.

### 🔴 Known vulnerabilities — reviewer objections that need answers

**V1 — σ½ is redundant within H2 (sharpest objection):**
Raw per-param displacement at T*: MNLI = 0.062×0.002857 = **1.77e-4**, AGNews = 0.063×0.002756 = **1.74e-4**, CoLA = 0.091×0.001944 = **1.77e-4**. All within 2%. A reviewer will say: "there is a universal raw displacement threshold; divide by σ½ adds nothing since σ½ varies by task and the raw threshold absorbs that variation — you don't need σ½ at all." Current defense is only logical (you can't know 1.75e-4 in advance without observing forgetting). Need either: (a) a case where raw threshold fails but R_A succeeds (requires a task with very different σ½_A), or (b) explicitly frame σ½'s value as providing the constant from a checkpoint computation rather than from forgetting observations. σ½ demonstrably adds value in H1 cross-model comparison (GPT-2 vs Llama 11× raw displacement difference, normalized by σ½).

**V2 — "R_A is monotone" is partially definitional:**
Within a single LR sweep, R and LR are co-monotone by construction (higher LR → larger ‖Δθ‖ → larger R). "R predicts forgetting" is the same as "higher LR predicts forgetting" in a single-model, single-rank sweep. The non-trivial evidence is the KL comparison (H1) and the Phase 1 property claim (H2). Drop the monotonicity framing; use "predictive of forgetting onset" instead.

**V3 — T* precision is overstated:**
T* values reported as ≈0.089 vs ≈0.093 (4% difference) come from interval midpoints where each interval spans [0.060, 0.118] and [0.060, 0.126] — a 2× range. Reporting these as consistent "within 5%" is misleading. The defensible claim: "T* intervals are indistinguishable across Phase 2 tasks." Any precision claim below 2× is unsupported given the coarse LR grid.

**V4 — Rank coverage is too sparse: ✅ SUBSTANTIALLY CLOSED (2026-07-03)**
r4/r8/r16/r32 complete for: AGNews→MNLI, AGNews→SST-2, MNLI→SST-2. MNLI→AGNews has r8/r16/r32 (r4 running in Batch 4). AGNews results (8 conditions) are the cleanest: all 4 ranks × 2 Phase 2 tasks cluster T*≈0.047–0.065. r16 appears slightly high (≈0.092) but this is a stochastic boundary, not a systematic shift. Remaining gap: MNLI→AGNews r4 (running), SST-2→AGNews r4/r16 (running).

**V5 — SST-2→MNLI upper bound is one-sided: ✅ CLOSED (2026-07-01)**
T*(r8) SST-2→MNLI is now **bounded**: safe at lr=5e-4 (R_A_acc=0.212, SST-2=0.900), catastrophic at lr=1e-3 (R_A_acc=0.851, SST-2=0.478). T*(r8) ∈ [0.212, 0.851], mid≈**0.532**. vs SST-2→AGNews T*(r8)≈0.027 → anomaly ratio **~20×**. T*(r32) also bounded: [0.036, 0.070] mid≈0.053 vs r32 AGNews <0.008 → >6×. The SST-2 anomaly is now fully characterized for both ranks.

**V6 — KL failure SST-2-specific / single model architecture:**
KL failure is now confirmed NOT SST-2-specific: H1-MNLI (2026-07-03) shows KL(diverse) non-monotone for MNLI fine-tuning too (drops at lr=1e-4 despite rising NLL). V6 KL-specificity concern is CLOSED. Architecture concern **FULLY ADDRESSED (2026-07-13, strengthened 2026-07-17):** (1) H2: Mistral-7B validates H2 result for both AGNews Phase-2 targets on a **consistent accuracy-based σ½ for both models** (corrected 2026-07-17 — the original comparison mixed Llama's acc-based σ½ against Mistral's NLL-based σ½, since no acc-based Mistral σ½ existed yet). AGNews→MNLI: T*=[0.020,0.085] (Mistral) vs [0.042,0.085] (Llama) — overlap, first-forget LR step lands close (0.0852 vs 0.085). AGNews→SST-2: T*=[0.068,0.105] (Mistral) vs [0.042,0.089] (Llama) — overlap, first-forget LR step lands further apart (0.105 vs 0.089). (Read these as "same tested grid step, close/further apart," not as "0.2%"/"17.9%" precision — the 5-point LR grid doesn't support that resolution; see the standing methodological note at the top of this section.) Both pairs now overlap under a genuinely like-for-like comparison; the original 11%-agreement figure for →MNLI, while numerically close to being right, was not computed on a consistent basis. (2) H1: **⚠️ corrected 2026-07-17 — this used to cite superseded v3 data.** Mistral-7B SST-2 **v5 PRIMARY** gives T*(relslack)=[0.194,0.226] (not v3's [0.347,0.397], which was batch=4 with wrong σ½ protocol and is archived). GPT-2 [0.22,0.45] and Llama [0.24,0.59] under the cliff-jump criterion — but see the criterion-sensitivity correction in the H1 results section: under the stricter 1%-threshold criterion this becomes a chain (GPT-2∩Llama, Llama∩Mistral overlap; GPT-2∩Mistral does not), not a clean 3-way overlap. "3-architecture T* convergence confirmed" should be read as "raw displacement compressed 39-44× to a ~3× band," not as a precise simultaneous overlap. KL(diverse) for Mistral fails by plateau (not saturation): 3.043→3.606, 19% change while ΔNLL jumps 46×. KL(SuperNI) for Mistral SST-2 now confirmed real and discriminating (0.821→7.504, 9× jump) ✅ (v3 fixed 2026-07-13). Mistral MNLI KL(SuperNI) confirmed real and monotone ✅ (v2 fixed). **⚠️ CORRECTED 2026-07-17:** the "remaining gap" below was itself a staleness artifact (Llama's R≈3.57 was non-relslack). Corrected: Mistral MNLI T*=[0.071,0.324] (relslack), Llama MNLI T*=[0.133,0.233] (relslack, same 1%-baseline break criterion) — Llama's bracket is fully contained inside Mistral's. **MNLI T* does converge across Llama/Mistral; there is no remaining gap for this pair.** V6 is CLOSED, and more completely than previously stated.

**V7 — H1 and H2 T* values are not comparable: ✅ CLOSED (2026-07-01)**
Both scripts call `peft_model.merge_and_unload()` before measuring displacement, so per_param_norm divides by ALL ~3.2B Llama params in both cases — not LoRA params only. Verified by code inspection of `h2_sequential_llama.py` (lines 474, 517) and `h1_kl_comparison.py` (line 478). Script: `compute_fullmodel_norm.py`.

**Critical finding from V7 analysis:** All 6 H2 T\*\_raw values (rank 8) cluster at **1.71–1.80 × 10⁻⁴** (within 6% of each other), and H1 T\*\_raw lower bound is **1.70 × 10⁻⁴** — the two experiments share the same absolute displacement floor despite measuring completely different types of forgetting (pretraining knowledge vs. task memory). σ½ normalization is what reveals this: without it, R_A spans 0.06 to 0.58 across experiments and looks incomparable. This is the strongest counter-argument to V1 (σ½ redundant): within Llama H2, σ½ values are similar so raw displacement works equally well, but σ½ is what makes H1 (σ½_pre=0.000772, 4× tighter basin) comparable to H2 at all. See T*_raw rows in Key Numbers.

**V8 — SST-2 anomaly has no explanation:**
Four mechanistic hypotheses tested and all null (direction overlap, effective rank, binary/multiclass, subspace-projection variant — the last one underpowered at only 3 data points, not a fully confirmed null). A top-venue reviewer will not accept "empirical anomaly" as sufficient. Need either: (a) a fifth testable hypothesis and experiment, or (b) a convincing argument that the anomaly is instructive (e.g., demonstrates the limits of the framework rather than a failure of it). **Partial progress 2026-07-22/23**: a 5th angle (classifier-readout mismatch) was tested two ways on GPT-2. First, NLL-vs-accuracy forgetting showed a real signal (NLL improves substantially while accuracy collapses to chance) but turned out to be ambiguous on closer inspection — the magnitude pattern didn't clearly favor readout-mismatch over mode-collapse. Second, a sharper direct test — fresh-linear-probe recovery (`h2_probe_recovery.py`, 2026-07-23) — gave a cleaner answer: on all 3 tested Phase-2 targets (mnli/agnews/cola), a fresh probe recovers Task-A accuracy above baseline within 55-200 steps, ruling out mode-collapse-as-total-information-loss for these conditions and supporting readout-mismatch as at least part of the mechanism. Not yet run on Llama, where the actual 44× anomaly this vulnerability describes was observed — this is suggestive on GPT-2, not a resolution of V8 itself. See "NLL-vs-Accuracy Forgetting" subsection above for both results.

**⚠️ Re-examined 2026-07-22, same day: the NLL result doesn't discriminate the two explanations it was meant to.** NLL-on-domain-text measures fluency (is the model good at predicting SST-2-style text), not whether sentiment is still linearly decodable from the current hidden state — a different, only-adjacent question from what H2 actually needs. Worse, the exact magnitudes argue against reading this as confirmatory: the size of the NLL "improvement" scales with the size of the accuracy collapse (near-flat NLL movement for the condition that stays safe, largest NLL improvement for the conditions that collapse hardest) — not what readout-mismatch alone predicts (that story implies NLL should stay roughly flat regardless of accuracy-collapse severity, since the representation itself isn't supposed to be damaged). A mode-collapse explanation (aggressive training degrading output into generic/repetitive text, which can spuriously score low perplexity on short generic sentences without preserving any real information) produces the identical signature and is at least as parsimonious given this correlation. **A new script, `h2_probe_recovery.py`, is written (not yet run) to actually resolve this**: freeze the Phase-2-perturbed backbone, train a fresh linear probe on Task-A from scratch, and check whether it recovers SST-2 accuracy quickly (supports readout-mismatch) or struggles just as much as the old frozen classifier (supports genuine loss / a different V8 explanation). See `RAW_DATA.md`'s NLL-vs-Accuracy Forgetting section for the full reasoning and the pre-registered decision rule.

---

### 📝 Experiment priority (to defend at top venue)

**🔴 Blocking — do before drafting:**

| Priority | Experiment | Compute | What it fixes |
|---|---|---|---|
| 1 | ~~SST-2→MNLI at lr=[1e-3, 2e-3, 5e-3]~~ | ~~DONE~~ | ✅ T*(r8)=[0.212,0.851] mid=0.532; ~20× anomaly vs AGNews |
| 2 | ~~r16 for MNLI→AGNews~~ (AGNews→MNLI pending) | ~~partial DONE~~ | ✅ MNLI→AGNews r16=0.091 (higher than r8/r32); AGNews→MNLI r16 still queued |
| 3 | Fix monotonicity language in writing | 0h | Removes false claim (V2) |
| 4 | Remove "est." from CoLA→AGNews | 0h | CoLA→AGNews T* is exact, not estimated |

**🟡 Should do — significantly strengthens:**

| Priority | Experiment | Compute | What it fixes |
|---|---|---|---|
| 5 | r4 for MNLI→SST-2 and AGNews→MNLI | ~4h | Practical low-rank claim; shows r4-r8 is safe zone |
| 6 | ~~Full-model per-param norm recompute on H2 checkpoints~~ | ~~DONE~~ | ✅ V7 CLOSED: norms identical; T*_raw=1.75e-4 across all pairs; H1 lower bound matches |
| 7 | H1 with MNLI as Phase 2 task (GPT-2 + Llama) | ~4h | Tests whether KL failure is SST-2-specific or general (V6 partial) |

**🟢 Nice to have:**

| Priority | Experiment | Compute | What it fixes |
|---|---|---|---|
| 8 | ~~σ½_A_acc for SST-2~~ | ~~DONE~~ | SST-2 σ½_A_acc=0.003337 already computed (2026-06-29). All 4 Phase 1 tasks have acc-based σ½. Nothing to do. |
| 9 | Additional LR points in H1 and H2 transition zones | ~4h | Narrows 2× T* intervals to ~1.5× |
| 10 | ~~H2 on second model (Mistral-7B, AGNews→MNLI + AGNews→SST-2)~~ | ~~DONE~~ | ✅ R_A at T*: Llama=0.064, Mistral=0.071 (11% gap); V6 architecture concern substantially addressed |

**Priority 1 (SST-2→MNLI upper bound) is the single most important remaining experiment.** The SST-2 anomaly is the most interesting finding and currently one arm is unbounded. Everything else is secondary to knowing the actual T*(r8) for SST-2→MNLI.

---

### 🔜 Theory section — SST-2 anomaly
- **Ruled out:** direction overlap (null), effective rank of ΔW (null — SST-2=5.21, AGNews=5.36, CoLA=5.19, identical within 3%), binary/multi-class hypothesis (CoLA binary but consistent).
- **Untested candidates:** (a) per-param ‖Δθ‖ for SST-2 may be systematically small at all LRs — LLM near-optimal for sentiment at pretraining; (b) SST-2's larger σ½_A (0.003337 vs CoLA 0.001944) means Phase 2 may rarely push R_A past T* for MNLI, making it appear T* is "infinite" in that direction; (c) task-content distinction (1D valence vs multi-criterion discrimination).
- Currently this is an empirical anomaly with no mechanistic explanation. That is honest but weak for top venues.

---

## File Map

```
NTRS/
├── SUMMARY.md                        ← this file
│
├── h1_kl_comparison.py               ← H1 main experiment script (Llama + GPT-2 supported)
├── h2_sequential_llama.py            ← H2 Phase 1 + Phase 2 sweep script
├── recompute_sigma_half_acc.py       ← H2 acc-based σ½ (now has --refine, --force flags)
├── validate_llama_mnli.py            ← Gate check: Llama MNLI > 80%
├── plot_h1_figure.py                 ← Publication figure for H1
├── plot_h2_full_analysis.py          ← Publication figure for H2
│
├── h1_figure.png                     ← H1 key figure (R vs ΔNll, KL comparison)
├── h2_full_analysis.png              ← H2 full analysis (trajectory, R_A path, endpoint)
│
├── h1_kl/gpt2/                       ← H1 GPT-2 results (local copy)
│   ├── summary.json
│   └── lr*_results.json
│
├── h1_kl/meta-llama/Llama-3.2-3B/  ← H1 Llama results (local copy)
│   ├── summary.json
│   └── lr*_results.json
│
├── h1_kl/mistralai/Mistral-7B-v0.1_sst2_v1/  ← H1 Mistral SST-2 v1 (backup; task learned, KL(SuperNI) BOGUS)
│   ├── summary.json                              ← KL(SuperNI) = KL(diverse) — bogus; acc=0.876 at lr=1e-5 ✓
│   ├── pretrained_density.json                   ← σ½_pre=1.559e-4 (coarse, ~25% uncertainty)
│   └── lr*_results.json
│
├── h1_kl/mistralai/Mistral-7B-v0.1_sst2_v2/  ← H1 Mistral SST-2 v2 (--force, batch_size=2; task NOT learned)
│   ├── summary.json                              ← acc=0.498–0.536 (near chance); real KL(SuperNI) but fine-tuning failed
│   ├── pretrained_density.json                   ← σ½_pre=1.559e-4 (same checkpoint)
│   └── lr*_results.json
│
├── h1_kl/mistralai/Mistral-7B-v0.1_sst2_v3/  ← H1 Mistral SST-2 v3 ✅ (batch_size=4, 2026-07-13)
│   ├── summary.json                              ← real KL(SuperNI); T*(relslack)=[0.347,0.397]; lr=1e-5 suboptimal
│   ├── pretrained_density.json                   ← σ½_pre=1.559e-4 (non-relslack)
│   ├── pretrained_density_relslack.json          ← σ½_pre=1.824e-4 (relslack +17%)
│   └── lr*_results.json
│
├── h1_kl/mistralai/Mistral-7B-v0.1_mnli/     ← H1 Mistral MNLI v2 (real KL(SuperNI) ✅ 2026-07-11)
│   ├── summary.json                              ← KL(SuperNI) real and monotone; T*=[0.084,0.380]
│   └── lr*_results.json
│
├── h2_llama/mnli_to_sst2/           ← H2-A (σ½=0.002857 refined ✅)
├── h2_llama/mnli_to_agnews/         ← H2-B (σ½=0.002857 ✅)
├── h2_llama/sst2_to_mnli/           ← H2-C (σ½=0.003337 refined ✅)
├── h2_llama/agnews_to_mnli/         ← H2-D (σ½=0.002756 refined ✅)
├── h2_llama/agnews_to_sst2/         ← H2-E (σ½=0.002756 refined ✅)
├── h2_llama/sst2_to_agnews/         ← H2-F (σ½=0.003337 reused ✅)
├── h2_llama/cola_to_mnli/          ← H2-G (σ½_acc=0.001944 ✅) ← new
├── h2_llama/cola_to_agnews/        ← H2-H (σ½_acc=0.001944 shared ✅) ← new
│
├── h2_mistral/agnews_to_mnli/      ← H2-J (Mistral-7B, σ½=7.62e-4, T*_r8≈0.071) ← new
├── h2_mistral/agnews_to_sst2/      ← H2-K (Mistral-7B, σ½=7.62e-4, T*_r8≈0.038) ← new
│
├── recompute_h1_sigma_relslack.py   ← H1 σ½_pre standardized recompute script ← new
├── analyze_weight_directions.py     ← Direction overlap analysis (NULL result) ← new
├── compute_fullmodel_norm.py        ← V7 verification: T*_raw convergence, no GPU needed ← new
│
└── certified_density_experiment.py  ← Core: estimate_density, half_max_sigma
    lora_density_experiment.py       ← Core: _load_model_auto, add_lora, etc.
```

---

## Key Numbers to Remember

| Quantity | Value | Notes |
|---|---|---|
| GPT-2 σ½_pre (wiki NLL, 1% rel slack) | **0.007191** | H1; NLL_base=4.211; eval_slack_abs=0.042 |
| Llama-3.2-3B σ½_pre (wiki NLL, 1% rel slack) | **0.000772** | H1; NLL_base=3.059; eval_slack_abs=0.031 |
| Mistral-7B-v0.1 σ½_pre (wiki NLL, non-relslack) | **1.559×10⁻⁴** | H1; NLL_base=2.993; h1_kl/mistralai/Mistral-7B-v0.1_sst2_v1/pretrained_density.json |
| Mistral-7B-v0.1 σ½_pre (wiki NLL, 1% rel slack) | **1.824×10⁻⁴** | H1; NLL_base=2.660; h1_kl/mistralai/Mistral-7B-v0.1_sst2_v3/pretrained_density_relslack.json; only +17% vs non-relslack |
| Ratio GPT-2/Llama σ½_pre | **9.3×** | NOT scale-invariant; old 0.000603 vs 0.000576 were artifacts of mismatched slacks |
| Ratio Llama/Mistral σ½_pre (relslack) | **4.2×** | Mistral pretrained basin 4.2× tighter than Llama |
| Ratio GPT-2/Mistral σ½_pre (relslack) | **39×** | Full architecture range |
| Llama σ½_A_nll (MNLI) | 0.006607 | BROKEN — density curve wrong |
| Llama σ½_A_acc (MNLI, refined) | **0.002857** | Updated 2026-06-27, ±5-10% |
| Llama σ½_A_nll (SST-2) | 0.011919 | NLL-based; 1.8× larger than MNLI |
| Llama σ½_A_acc (SST-2, refined) | **0.003337** | Refined 2026-06-29, ±5% |
| Llama σ½_A_acc (AGNews, refined) | **0.002756** | Refined 2026-06-30, ±5% (was 0.002551 ±25%) |
| Llama σ½_A_acc (CoLA, computed) | **0.001944** | Computed 2026-06-30; 3.86× smaller than NLL-based (0.0075). Applies to BOTH CoLA pairs. |
| H1 T* (GPT-2, corrected) | **∈ [0.22, 0.45]** | Corrected 2026-06-30; old [1.03, 2.63] was artifact of wrong σ½_pre |
| H1 T* (Llama, corrected) | **∈ [0.24, 0.59]** (was mistakenly written 0.58; exact=0.58554, corrected 2026-07-17) | Corrected 2026-06-30; stale-file bracket was [0.316, 0.784] (non-relslack, do not cite) |
| H1 T* (Mistral-7B, SST-2, relslack) | **∈ [0.194, 0.226]** | ⚠️ **v5 PRIMARY (2026-07-14), not v3.** [0.347,0.397] was v3 (batch=4, wrong σ½ protocol, superseded) — this row previously cited the superseded v3 number by mistake, fixed 2026-07-17. v5: last safe R=0.194 (lr=5e-5, ΔNLL=+0.125), first catastrophic R=0.226 (lr=1e-4, ΔNLL=+2.810) |
| H1 T* (Mistral-7B, MNLI, relslack) | **∈ [0.071, 0.324]** | Corrected 2026-07-17 (was [0.084, 0.380] non-relslack); NLL cliff early (R=0.324 already +0.603); task collapses at R=0.865 (was R=1.013 non-relslack); much more fragile than Llama MNLI; KL(SuperNI) real and monotone ✅ |
| H1 T* convergence (SST-2 fine-tuning) | **Chain, not clean 3-way overlap, under the stricter 1%-threshold criterion** — see full correction in Cross-model T* comparison / What Claims Can Be Made Now | Corrected 2026-07-17. Cliff-jump criterion: GPT-2=[0.22,0.45], Llama=[0.24,0.59], Mistral(v5)=[0.19,0.23] — GPT-2∩Mistral overlap, Llama adjacent. 1%-threshold criterion: GPT-2=[0.22,0.45], Llama=[0.11,0.24], Mistral=[0.03,0.19] — GPT-2∩Llama and Llama∩Mistral overlap, but GPT-2∩Mistral=∅. Raw displacement spans 39–44×; σ½ compresses to ~3× either way, but the precise 3-way convergence claim is criterion-sensitive — do not cite without stating which criterion. |
| H1 KL(diverse) at safe LR (Mistral SST-2, v3) | **3.043** at LR=5e-5 | NOT saturated like Llama 7.11 — different failure mode; lr=1e-5 gives 0.035 but task not learned |
| H1 KL(diverse) at cliff (Mistral SST-2, v3) | **3.043 → 3.606** (19% change) | ΔNLL jumps 46× at same boundary; plateau failure confirmed |
| H1 KL(SuperNI) at cliff (Mistral SST-2, v3) | **0.821 → 7.504** (9× jump) ✅ | Discriminating for Mistral SST-2; contrast Llama (only 15% at boundary) |
| H2 T* (MNLI Phase 1, rank 8) | **≈ 0.061–0.062** | Consistent across SST-2 and AGNews Phase 2 (<2%) |
| H2 T* (MNLI Phase 1, rank 32) | **≈ 0.062–0.131** | Varies 2× by Phase 2 task difficulty |
| H2 T* (AGNews Phase 1, rank 8) | **≈ 0.062–0.065** | Consistent across MNLI and SST-2 Phase 2 (<5%); same as MNLI Phase 1! |
| H2 T* (AGNews Phase 1, rank 32) | **≈ 0.062–0.063** | Also consistent ✅ (<2%) |
| H2 T* (SST-2 Phase 1, rank 8) | **[0.212, 0.851] mid≈0.532 (MNLI) vs 0.027 (AGNews)** | NOT a Phase 1 property — ~20× gap; V5 closed 2026-07-01 |
| H2 T* (SST-2 Phase 1, rank 32) | **[0.036, 0.070] mid≈0.053 (MNLI) vs <0.008 (AGNews)** | >6× gap; r8 and r32 differ 10× for MNLI Phase 2 |
| Llama σ½_A_acc (CoLA) | **0.001944** | Computed 2026-06-30; 3.86× smaller than NLL-based σ½=0.0075 |
| H2 T* (CoLA Phase 1, rank 8) | **≈0.089 (MNLI) vs 0.093 (AGNews)** | Acc-based; consistent ✅ (4%); ABOVE MNLI/AGNews ≈0.062 |
| H2 T* (CoLA Phase 1, rank 32) | **≈0.189 (MNLI) vs 0.089 (AGNews)** | Acc-based; NOT consistent ❌ (2×); same pattern as MNLI r32 |
| Directional vulnerability 1/T* | **15–16×** (rank 8, MNLI/AGNews); **≈11×** (CoLA) | MNLI 16×, AGNews 15×, CoLA 11× |
| T\*\_raw (H2, rank 8, all 6 pairs) | **1.71–1.80 × 10⁻⁴** | per_param_norm at forgetting onset; < 6% spread across all pairs |
| T\*\_raw (H1, Llama, lower bound) | **1.70 × 10⁻⁴** | = 0.22 × σ½_pre = 0.22 × 0.000772; at lower edge of H2 cluster |
| T\*\_raw convergence | H1 and H2 share ~1.75e-4 floor | Absolute displacement threshold for forgetting is near-universal for Llama; σ½ is needed to see this across basins of different sizes |
| Mistral-7B σ½_A_acc (AGNews) | **7.62×10⁻⁴** | 3.6× smaller than Llama AGNews (2.756×10⁻³); batch_size=4 needed (7B model) |
| Mistral-7B T* (AGNews→MNLI, r8) | **[0.043, 0.094] median≈0.071** | vs Llama 0.064 — within 11% after σ½ normalization; raw displacement 3.2× different |
| Mistral-7B T* (AGNews→SST-2, r8) | **[0.038, 0.046] median≈0.042** ⚠️corrected | SST-2 Phase 2 barely more disruptive than MNLI (~10%); lr=5e-5 excluded (noisy survived) |
| Cross-architecture R_A convergence | **AGNews→MNLI: Llama=0.085, Mistral=0.0852 (same tested LR step, close). AGNews→SST-2: Llama=0.089, Mistral=0.105 (same tested LR step, further apart).** Not "0.2%/17.9% precision" — both are single points from a 5-value LR grid, see the H2 methodological note. | Corrected 2026-07-17: both models now on accuracy-based σ½ (Mistral σ½_A_acc=7.314×10⁻⁴, computed via `recompute_sigma_half_acc.py`). Previous "Llama=0.064, Mistral=0.071 (11% gap)" mixed acc-based Llama against NLL-based Mistral — not like-for-like. Both AGNews Phase-2 pairs now overlap on a consistent basis. |
| Catastrophic collapse | **R_A ≈ 0.22–0.28** | Phase 1 drops to ~25–35%; MNLI and AGNews Phase 1 |
| CoLA Phase 1 baseline acc | **86.8%** | Llama-3.2-3B, lr=5e-5, rank=8, 2000 steps |
| Phase 1 MNLI accuracy | 88.4% | Llama-3.2-3B, lr=5e-5, rank=8, 2000 steps |
| Phase 1 SST-2 accuracy | 97.5% | Llama-3.2-3B, lr=5e-5, rank=8, 2000 steps |
| Phase 1 AGNews accuracy | 90.5% | Llama-3.2-3B, lr=5e-5, rank=8, 2000 steps |

---

## How to Use This File

When prompting in a new session:
> "Read SUMMARY.md and continue from there."

When updating after new runs:
> "Read SUMMARY.md and update with [experiment name] results: [paste table/numbers]."
