# Research Clarity — NTRS Project
**Written:** 2026-07-02 | **Updated:** 2026-07-14 (Batch 9: Mistral SST-2 v5 PRIMARY ✅ — effective batch=16, correct σ½ protocol; T*(relslack)=[0.194,0.226]; supersedes v3 [0.347,0.397])  
**Purpose:** Plain-language summary of what we have, what's trivial, what's honest, and what's left. Companion to SUMMARY.md (which is the technical running log).

---

## What the project is (one paragraph)

You fine-tune a model on Task A, then fine-tune on Task B. Task A gets forgotten. We ask: can you predict this *before* running Task B, using only the Task A checkpoint?

We propose **σ½** — a number computed from the Task A checkpoint via randomized smoothing that measures how fragile the model's weights are to perturbation. We normalize the weight displacement during Task B training by σ½ to get **R_A = ‖Δθ‖ / σ½_A**. The claim: when R_A crosses a threshold T*, Task A is forgotten. σ½_A is computable before Task B starts. That's the operational value.

We also test the same idea at the pretraining→fine-tuning boundary (H1): does R = ‖Δθ‖/σ½_pre predict pretraining knowledge degradation (WikiText-2 perplexity) during SST-2 fine-tuning?

---

## Experiments run and what they showed

### H1 — Pretraining boundary (GPT-2 + Llama, SST-2 and MNLI fine-tuning)

**What was run:** Fine-tuned GPT-2 (117M), Llama-3.2-3B, and Mistral-7B-v0.1 at 5 learning rates on SST-2 (all three models) and MNLI (Llama + Mistral). Measured WikiText-2 perplexity change. Computed σ½_pre from pretrained checkpoint (standardized to 1% relative eval_slack). Extended metrics (KL-SuperNI, MMLU) added in Batch 3. Mistral-7B H1 added in Batch 7 (2026-07-11). Mistral MNLI v2 with real KL(SuperNI) SCP'd locally (2026-07-11). Mistral relslack σ½ computed + SST-2 v3 (batch_size=4) completed 2026-07-13 (Batch 8).

**What the results show (SST-2 fine-tuning):**

1. As LR increases, R increases and perplexity degrades — there is a cliff where degradation goes from negligible to catastrophic.

2. After correcting σ½_pre (there was a computation bug from mismatched eval_slack settings):
   - GPT-2: σ½_pre = 0.007191 (relslack), T* ∈ [0.22, 0.45]
   - Llama: σ½_pre = 0.000772 (relslack), T* ∈ [0.24, 0.58]
   - Mistral-7B: σ½_pre = 1.824×10⁻⁴ (relslack), T* ∈ **[0.194, 0.226] (v5 PRIMARY ✅, 2026-07-14)**
   - GPT-2 [0.22,0.45] and Mistral [0.194,0.226] overlap at **[0.22, 0.23]**. Llama lower bound (0.235) is adjacent, within 5% of Mistral upper bound (0.226). All degradation onsets fall within R≈0.19–0.24 despite 60× model size range.
   - The reason they converge for SST-2: each model's raw ‖Δθ‖ at T* roughly tracks its σ½_pre, so the ratio stays comparable. Raw displacement spans **44×** (1.58e-3 for GPT-2 to 3.53e-5 for Mistral v5); σ½ compresses this to a **1.2×** band; without σ½ normalization there is no comparison.
   - For MNLI fine-tuning: convergence does NOT hold — Llama safe through R≈3.57; Mistral NLL cliff at R≈0.38. The task-dependence of T* carries over to cross-architecture comparison.

3. KL divergence on instruction-domain prompts (KL-diverse) fails as a predictor — the failure mode varies by model but the outcome is identical:
   - **Llama (saturation):** At LR=1e-5 (safe, ΔNLL≈0), KL-diverse = 7.11. This alone disqualifies it: a predictor that reads 7.11 in the safe zone cannot discriminate safe from degraded. At LR=5e-4 (catastrophic, ΔNLL=+8.74): KL-diverse = 10.67 — barely larger. Saturation is from domain mismatch: the base→instruction-following shift dominates from the first update, regardless of fine-tuning aggressiveness.
   - **Mistral-7B (insufficient slope, v5 2026-07-14):** At LR=1e-5 (no SST-2 convergence, acc=0.48), KL-diverse = 0.088 — NOT saturated. At the forgetting cliff (LR=5e-5→1e-4, ΔNLL=+0.125→+2.810, a 22.5× NLL jump): KL-diverse moves only 1.411→3.706 (2.6×). A 2.6× metric change when the target metric jumps 22.5× is not useful for detection. The failure mode: KL-diverse grows, but too slowly at the threshold. Outcome is the same as saturation: KL-diverse cannot detect the forgetting cliff.
   - Non-monotone behavior (Llama LR=5e-5 gives KL=4.05 < 7.11 at LR=1e-5) is corroborating evidence: KL-diverse varies with the *direction* of weight displacement, which changes with LR. Different LR runs trace different trajectories in weight space. The Fisher approximation (KL ≈ ½ Δθᵀ F Δθ) supports monotone KL only along a fixed ray — not across runs. Non-monotonicity is expected, not a measurement artifact.
   - KL on WikiText-2 (KL-wiki) IS monotone and tracks ΔNLL closely for all three models — confirming the failure is domain-specific. KL as a concept is not broken; KL evaluated on instruction prompts measures the wrong distribution.

4. KL on SuperNI-style prompts (KL-SuperNI) is mostly monotone but has poor threshold discrimination:
   - At LR=1e-4 (last safe, ΔNLL=+0.035): KL-SuperNI = 2.86
   - At LR=2e-4 (first forget, ΔNLL=+1.133): KL-SuperNI = 3.30 (only 15% increase)
   - ΔNLL jumps 32× at the safe/forget boundary; KL-SuperNI moves 15%. Not usable as a threshold detector.

5. MMLU zero-shot accuracy drops at the same R threshold as NLL:
   - At LR=1e-4 (last safe): MMLU = 0.540
   - At LR=2e-4 (first forget): MMLU = 0.370 (32% drop)
   - Confirms R predicts factual knowledge degradation, not just perplexity.

**What the results show (MNLI fine-tuning — V6 closure):**

Run to test whether KL-diverse failure is SST-2-specific or general.

| LR | R | ΔNLL | KL-diverse | KL-SuperNI | MMLU |
|---|---|---|---|---|---|
| 1e-5 | 0.054 | +0.007 | 6.34 | 1.244 | 0.540 |
| 5e-5 | 0.178 | +0.032 | 6.63 | 1.390 | 0.520 |
| 1e-4 | 0.312 | +0.039 | **5.46↓** | 1.323 | 0.500 |
| 2e-4 | 0.711 | +0.064 | 10.55 | 3.892 | 0.490 |
| 5e-4 | 3.570 | **+7.745** | 9.53 | 9.238 | **0.250** |

- KL-diverse drops at LR=1e-4 before spiking — non-monotone for MNLI too. **V6 CLOSED.** KL failure is not SST-2-specific.
- KL-SuperNI: larger jump at safe/forget boundary (1.32→3.89, 3×), but ΔNLL jumps 198× there. Still poor discrimination.
- MMLU tracks NLL: drops sharply at LR=5e-4, same R region as NLL cliff.

**Limitations:** 2 models (SST-2 only), 2 fine-tuning tasks (SST-2 + MNLI for Llama), sparse LR grid (5 points). KL-diverse failure and KL-SuperNI poor discrimination confirmed for both fine-tuning tasks.

---

### H2 — Sequential task forgetting (Llama-3.2-3B, 8 task pairs)

**What was run:** For each of 4 Phase 1 tasks (MNLI, AGNews, SST-2, CoLA), fine-tuned Llama, saved checkpoint, computed σ½_A, then fine-tuned on 2–3 different Phase 2 tasks at 5 LRs × 2 ranks. Measured whether Phase 1 accuracy dropped below 90% of original.

**Results per Phase 1 task:**

| Phase 1 | Phase 2 tasks | T*(rank 8) — what the data shows |
|---|---|---|
| MNLI | SST-2, AGNews | **[0.041, 0.081] vs [0.040, 0.084] — indistinguishable ✓** (5-seed confirmation, 2026-07-04) |
| AGNews | MNLI, SST-2 | [0.042, 0.085] and [0.042, 0.089] — indistinguishable ✓ |
| CoLA | MNLI, AGNews | [0.060, 0.118] and [0.060, 0.126] — indistinguishable ✓ |
| SST-2 | CoLA, AGNews, MNLI | **[0.005, 0.019] / [0.036, 0.073] / [0.212, 0.851] — 44× range** (2026-07-05) |

**What it means (revised 2026-07-05):** Two distinct behaviors emerge.

*Isotropic (3/4 Phase 1 tasks):* MNLI, AGNews, CoLA Phase 1 — T*(r8) is indistinguishable across Phase 2 tasks (~1× spread). σ½_A computable from the Phase 1 checkpoint predicts the forgetting threshold regardless of which task comes next. MNLI confirmed via 5-seed replication.

*Anisotropic (SST-2 Phase 1):* T*(r8) varies **44×** across 3 Phase 2 tasks — from [0.005, 0.019] when Phase 2 = CoLA, to [0.036, 0.073] for AGNews, to [0.212, 0.851] for MNLI. SST-2 is simultaneously the most fragile Phase 1 model (CoLA Phase 2 destroys it at the lowest T* of any pair measured — lower than all other Phase 1 tasks) and the most resistant (MNLI Phase 2 requires far more displacement than any other pair).

The 44× spread strongly suggests SST-2 Phase 1 creates an **anisotropic forgetting basin** in weight space — σ½ measures average radius, but the basin has very different radii in different directions. Phase 2 tasks that push in "short-axis" directions (CoLA) easily escape the basin; tasks pushing in "long-axis" directions (MNLI) almost never escape it. This directionality is not captured by isotropic σ½.

**SST-2 Phase 1 rank breakdown (2026-07-05, 3 Phase 2 tasks):**

| Phase 2 | r8 T* | r8 T*_raw | r32 T* | r32 T*_raw |
|---|---|---|---|---|
| CoLA | [0.005, 0.019] | 3.88e-5 (**4.7× below** cluster) | **<0.0082** (forgets at lr=1e-5) | <2.75e-5 |
| AGNews | [0.036, 0.073] | 1.81e-4 (**✓ cluster**) | [0.035, 0.071] | 1.77e-4 (**✓ cluster**) |
| MNLI | [0.212, 0.851] | 1.77e-3 (**10× above** cluster) | [0.036, 0.070] | 1.77e-4 (**✓ cluster**) |

- **CoLA Phase 2:** always below cluster, gets worse at r32. Hits SST-2's fragile direction at any rank.
- **AGNews Phase 2:** always in cluster at both ranks. σ½ works correctly for this direction.
- **MNLI Phase 2:** rank-dependent. r8 = 10× above cluster (long-axis resistance). r32 = perfectly in cluster (richer LoRA spans the fragile subspace). The MNLI anomaly is entirely rank-8-specific.

**Other Phase 1 tasks (rank findings):**
- r4 for MNLI→SST-2: T*=0.121, T*_raw=3.46e-4. r16 MNLI→AGNews is more robust (T*=0.091), r16 MNLI→SST-2 less robust (T*=0.043) — Phase 2-dependent, not a MNLI Phase 1 property.
- AGNews Phase 1 r8 and r32 Phase-2-independent; r4 identical [0.031, 0.062] for both Phase 2 tasks.
- "Catastrophic collapse" (25–35% accuracy) occurs at R_A ≈ 0.22–0.28 for MNLI and AGNews Phase 1.
- σ½_A values (Llama): MNLI=0.002857, AGNews=0.002756, SST-2=0.003337, CoLA=0.001944. Mistral-7B AGNews=0.000762 (3.6× smaller than Llama AGNews). Mistral-7B σ½_pre=1.559×10⁻⁴ (5× smaller than Llama σ½_pre=7.72×10⁻⁴).

---

## Mechanistic hypotheses tested — all null

Three scalar/structural explanations for why SST-2's T* is Phase-2-dependent were tested and ruled out:

**1. Binary/multi-class hypothesis** — SST-2 and CoLA are both binary tasks. If class count caused the anomaly, CoLA should also be inconsistent. CoLA T*(r8) is consistent across Phase 2 tasks. **Hypothesis broken.**

**2. Direction overlap (global cosine similarity)** — Tested cosine similarity between Phase 1 ΔW and Phase 2 gradient directions. All 4 pairs had cosine similarity ≈ ±0.0005 — indistinguishable from noise floor. **Null result.** *Note: this test was for a single point in parameter space; it does not test whether Phase 2 displacement accumulates along a particular subspace over training.*

**3. Effective rank of LoRA ΔW** — Measured effective rank of ΔW = B@A across 112 layers for SST-2, AGNews, CoLA. Results: SST-2=5.21, AGNews=5.36, CoLA=5.19 — identical within 3%. **Null result.**

**New characterization (2026-07-05):** Adding SST-2→CoLA (third Phase 2 task) reveals a 44× spread in T*(r8): CoLA [0.005,0.019] < AGNews [0.036,0.073] << MNLI [0.212,0.851]. SST-2 is both the *most fragile* (CoLA Phase 2 destroys it at lower R_A than any other pair) and *most resistant* (MNLI Phase 2 r8 can't destroy it until R_A ≈ 0.5). This bidirectional extremity needs a mechanism.

**Two competing explanations (2026-07-07):**

*Basin anisotropy (geometric):* SST-2's forgetting basin is cigar-shaped in weight space. CoLA Phase 2 pushes along the short axis (high curvature → forgets immediately); MNLI Phase 2 at rank-8 pushes along the long axis (low curvature → resists). σ½ measures average radius, missing the directional structure.

*Capacity story (rank-limited span):* Rank-8 MNLI LoRA simply cannot span the weight directions SST-2 uses. The LoRA subspace for MNLI r8 is orthogonal to SST-2's critical subspace by chance (task mismatch + rank constraint). Rank-32 MNLI LoRA spans a larger subspace, and CAN reach SST-2's critical directions — explaining why SST-2→MNLI T*_raw = 1.77e-4 at r32 (in cluster) vs. 1.77e-3 at r8 (10× above cluster). Under this framing, the "44× range" in T*(r8) is a rank-8 artifact for MNLI; the anisotropy is only real for CoLA (whose T* is anomalous at ALL ranks).

**Proxy test attempted (2026-07-07) — inconclusive:**
Projected Phase 1 LoRA ΔW for each task (CoLA, AGNews, MNLI) onto SST-2 Phase 1 LoRA subspace. B-projection (output features): CoLA=0.207, MNLI=0.197, AGNews=0.177 — no ordering, Pearson r=-0.12. A-projection appeared to support CoLA (0.935 vs 0.058 for MNLI) but this was a seeding confound: SST-2 and CoLA are both binary tasks (2-class), consuming identical RNG before LoRA A initialization → identical A matrices regardless of task. **Seeding bug fixed** in h2_sequential_llama.py (2026-07-07): now uses task-specific seed offset before `add_lora` in both Phase 1 and Phase 2.

**Definitive test (gradient_interference.py — to run on GPU):**
At theta_A*, compute grad(L_SST2) and grad(L_Phase2_task) for each (Phase 2 task, rank). Compute cosine similarity between effective weight-space gradients per layer. Predictions:
- *Interference story:* cos(g_SST2, g_CoLA) << 0; cos(g_SST2, g_MNLI_r8) ~ 0; cos(g_SST2, g_MNLI_r32) << 0
- *Capacity story:* cos(g_SST2, g_MNLI_r8) ~ 0 AND cos(g_SST2, g_MNLI_r32) << 0 (r32 reaches SST-2's subspace, r8 doesn't)
Script: `gradient_interference.py`. Run on GPU machine. V8 closes if either prediction holds cleanly.

---

## Triviality assessment — honest version

### What's trivially true (not a contribution)

**"R is monotone with forgetting"** — within a single LR sweep, R and LR are co-monotone by construction (higher LR → larger ‖Δθ‖ → larger R). "R is monotone" reduces to "higher LR → more forgetting," which is decades-old knowledge. This cannot be claimed as a finding.

### What's non-trivial

**1. KL-diverse saturates in the safe zone — it measures the wrong thing (H1):**  
At Llama LR=1e-5 (completely safe, ΔNLL≈0), KL-diverse = 7.11. At LR=5e-4 (catastrophic, ΔNLL=+8.74), KL-diverse = 10.67. A metric that already reads 7.11 in the safe zone cannot function as a forgetting detector. The non-monotonicity (drops to 4.05 at LR=5e-5) is corroborating evidence — it reflects that KL-diverse tracks the direction of weight displacement (which varies with LR) rather than its magnitude relative to pretraining forgetting. The correct explanation: KL-diverse measures the base→instruction-following distribution shift, which is large at even the smallest LR and is dominated by task-style change, not pretraining knowledge loss. Confirmed for both SST-2 and MNLI fine-tuning (V6 closed). KL-wiki (pretraining domain) is monotone and informative — the failure is domain-specific, not a flaw in KL as a concept. Note: "R is monotone while KL-diverse is not" is a weaker claim — R is co-monotone with LR by construction within a single sweep (larger LR → larger ‖Δθ‖ → larger R). The real contrast is KL-diverse saturating in the safe zone while R correctly separates safe from degraded.

**2. T* intervals are indistinguishable across Phase 2 tasks (H2, for 3/4 Phase 1 tasks, confirmed with replication):**  
If T* depended on which Phase 2 task you chose, you'd need to re-run experiments for every new task combination. The finding that it doesn't (for MNLI, AGNews, CoLA) means σ½_A computable before Phase 2 gives you the threshold for any future Phase 2 task. This is operationally useful and not obvious. MNLI specifically was confirmed via 5-seed replication at the stochastic boundary — the apparent 2.7× gap was training randomness, not a systematic effect.

**3. There is a sharp threshold at all (H2):**  
In all 8 task pairs, Phase 1 accuracy goes from healthy to near-random (25–35%) in one LR step. A smooth, gradual degradation curve would make σ½ less useful. The existence of a cliff is what makes it a *budget* — you're either inside or outside the safe zone. This is empirically striking.

**4. T* converges across model scales and architectures after σ½ normalization (H1 + H2):**  
H1 (2026-07-14, Mistral v5 PRIMARY): GPT-2 (117M), Llama-3.2-3B (3B), and Mistral-7B (7B) — 60× model size range — have σ½_pre spanning 39× (7.191e-3 to 1.824e-4). Raw ‖Δθ‖ at T* lower bound spans **44×** (1.58e-3 for GPT-2 to 3.53e-5 for Mistral). But R = ‖Δθ‖/σ½ gives T* intervals with onset near R≈0.19–0.24 for all three: GPT-2 [0.22,0.45], Llama [0.235,0.577], Mistral v5 [0.194,0.226]. GPT-2 and Mistral overlap at [0.22, 0.23]; Llama's lower bound (0.235) is adjacent (within 5%). σ½ compresses 44× raw spread to a **1.2× normalized onset band**. For MNLI fine-tuning, convergence breaks: Llama safe through R≈3.57; Mistral cliff at R≈0.38 — so the claim is restricted to the SST-2 fine-tuning task.

H2 (2026-07-11): Mistral-7B and Llama-3.2-3B have 3.6× different σ½_A for the same AGNews Phase 1 task. Raw displacement at T* differs 3.2×. R_A at T* (AGNews→MNLI): Llama=0.064, Mistral=0.071 — within 11%. σ½ absorbs architecture-specific displacement scales. The same R threshold (≈0.065–0.071) predicts forgetting onset regardless of model architecture for this task pair.

Together: T* convergence after σ½ normalization is demonstrated across two different experiment types (H1 pretraining boundary + H2 sequential forgetting) and two different architecture comparisons (GPT-2/Llama/Mistral for H1; Llama/Mistral for H2). The MNLI H1 exception is a genuine limitation.

**5. T\*\_raw (absolute displacement at T*) clusters at ~1.75e-4 for most r8 pairs, with two SST-2 outliers (updated 2026-07-05):**  

At r8, the absolute per-parameter displacement at the forgetting threshold is:

| Pair | T\*\_raw | Status |
|---|---|---|
| AGNews→MNLI, AGNews→SST-2 | 1.76–1.80e-4 | ✓ cluster |
| CoLA→MNLI, CoLA→AGNews | 1.73–1.80e-4 | ✓ cluster |
| MNLI→AGNews | 1.77e-4 | ✓ cluster |
| MNLI→SST-2 (seed-confirmed) | ~1.74e-4 | ✓ cluster |
| SST-2→AGNews | 1.81e-4 | ✓ cluster |
| **SST-2→CoLA** | **3.88e-5** | ❌ 4.7× **below** cluster |
| **SST-2→MNLI** | **1.77e-3** | ❌ 10× **above** cluster |
| H1 T\*\_raw lower bound | 1.70e-4 | reference |

8 of 9 r8 pairs measured. 7/9 in cluster (1.73–1.81e-4). Both outliers are SST-2 Phase 1, in opposite directions. SST-2→AGNews is perfectly in cluster — the anomaly is directional, not a general SST-2 property.

The finding is non-trivial: σ½ differs 4× between pretrained (σ½_pre=0.000772) and task-finetuned models (σ½_A≈0.002–0.003). Without σ½ normalization, R_A values span 0.006–0.851 and are incomparable. σ½ normalization reveals the underlying 1.75e-4 floor and exposes the SST-2 outliers as directional, not scalar.

### What's overclaimed (needs to be dialed back)

**"Within 5% consistency"** — T* values like 0.061 vs 0.062 look precise, but they are midpoints of intervals that are each ~2× wide (e.g., [0.040, 0.082] vs [0.040, 0.084]). The honest statement is: "the intervals are indistinguishable" — not that we've measured T* to 2% precision.

**"σ½ is better than raw ‖Δθ‖"** — within any single experiment, σ½ is a constant so R and ‖Δθ‖ are perfectly correlated. Across H2 pairs, T\*\_raw ≈ 1.75e-4 for all 6 pairs — meaning raw displacement also works within Llama H2. A reviewer will say: "σ½ is redundant; the raw threshold is universal." The two-part defense:
1. σ½ is what *explains why* the raw threshold is universal: all Llama task-finetuned models have similar basin size (σ½_A ≈ 0.002–0.003), so a constant raw threshold is unsurprising once you know σ½ is approximately constant. A different architecture or model scale would have a different raw threshold, making raw displacement task-and-model-specific.
2. σ½ is what makes H1 comparable to H2 at all: the pretrained model has σ½_pre=0.000772 (4× smaller basin). H2 T\*\_R ≈ 0.06–0.09 vs H1 T\*\_R ≈ 0.22–0.58 — same absolute displacement (1.75e-4), different R_A because the pretrained basin is tighter. Without σ½, you can't see this connection at all. Raw displacement fails here: the H1 safe threshold (1.70e-4 per param) is not "universal" across different model states without knowing σ½ explains why.

---

## What the honest claims are

### H1
> On GPT-2 (117M), Llama-3.2-3B (3B), and Mistral-7B-v0.1 (7B) fine-tuned on SST-2, KL divergence on instruction-domain prompts (KL-diverse) fails as a pretraining-forgetting detector — but via two different mechanisms. For Llama, KL-diverse saturates in the safe zone (7.11 at LR=1e-5, barely changes to 10.67 at catastrophic). For Mistral v5, KL-diverse does not saturate (0.088 at LR=1e-5) but has insufficient slope at the cliff: moves only 2.6× (1.411→3.706) while ΔNLL jumps 22.5×. Both are useless. KL on WikiText-2 (pretraining domain) is monotone for all three models — the failure is domain-specific. KL(SuperNI) is monotone but poor at threshold discrimination for Llama SST-2 (15% change while NLL jumps 32×). For Mistral SST-2 (v5 PRIMARY ✅, 2026-07-14), KL(SuperNI) is discriminating (12.4× jump at cliff: 0.581→7.218). For Mistral MNLI, KL(SuperNI) is monotone AND discriminating (7× jump safe→catastrophic). R = ‖Δθ‖/σ½_pre provides a **pre-run budget** for all three architectures. T* converges for SST-2 fine-tuning: GPT-2 [0.22,0.45], Llama [0.235,0.577], Mistral v5 [0.194,0.226] — GPT-2 and Mistral overlap at [0.22, 0.23]; Llama adjacent (within 5%); all onset R values in [0.19, 0.24] despite 44× raw displacement difference and 39× σ½_pre difference. MMLU collapses at the same R threshold as NLL for both Llama and Mistral. For MNLI fine-tuning, convergence breaks: Mistral's NLL cliff is at R≈0.38 (T*=[0.084,0.380]) vs Llama safe through R≈3.57.

*Scope: 3 models, 2 fine-tuning tasks (SST-2: all 3 models; MNLI: Llama + Mistral), 5-point LR grid. KL-diverse failure confirmed for all 3 models on SST-2; MNLI confirms for Llama but MNLI Mistral behavior differs significantly from Llama. T* convergence holds for SST-2 fine-tuning across all 3 architectures; does not hold for MNLI fine-tuning.*

### H2
> For MNLI, AGNews, and CoLA Phase 1 tasks, the R_A interval at which Phase 1 task forgetting begins is indistinguishable across two different Phase 2 tasks at rank 8, suggesting T* is determined by the Phase 1 model rather than the Phase 2 task. MNLI confirmed via 5-seed replication. For SST-2 Phase 1, T*(r8) spans a **44× range** across three Phase 2 tasks: [0.005, 0.019] (→CoLA) / [0.036, 0.073] (→AGNews) / [0.212, 0.851] (→MNLI). SST-2 Phase 1 is simultaneously the most fragile model in one direction and the most resistant in another — consistent with an anisotropic forgetting basin that isotropic σ½ cannot capture. σ½_A parameterizes T* for isotropic (MNLI, AGNews, CoLA) Phase 1 models at rank 8; it does not for SST-2.

*Scope: Llama-3.2-3B, rank 8 LoRA (primary), ranks 4/16/32 also tested for select pairs. 4 Phase 1 tasks × 2 Phase 2 tasks each. Cross-architecture validation: Mistral-7B at rank 8, AGNews Phase 1 → MNLI and SST-2 (2026-07-11).*

**Cross-architecture validation (Mistral-7B, 2026-07-11):** AGNews Phase 1 on Mistral-7B gives σ½=7.62×10⁻⁴ — 3.6× smaller than Llama's 2.756×10⁻³ for the same task. The raw per-parameter displacement at T* is also 3.2× smaller (5.4×10⁻⁵ vs 1.74×10⁻⁴). After σ½ normalization: R_A at T* = 0.071 (Mistral) vs 0.064 (Llama), within 11%. σ½ absorbs the architecture-specific displacement scale — the normalized threshold transfers. For AGNews→SST-2 (Mistral), T* = **[0.038, 0.046]** (corrected from [0.012, 0.046] — the lr=5e-5 lower bound was a transient dip artifact; final_acc=0.882, that condition survived). Only lr=1e-4 and lr=2e-4 are clean forgetting events. The corrected range is barely lower than MNLI, within 10%; SST-2 Phase 2 is only mildly more disruptive than MNLI for Mistral AGNews Phase 1.

**Key limitation discovered (Batch 2):** At rank 16, T* can be either the highest (MNLI→AGNews r16: most robust) or among the lowest (MNLI→SST-2 r16: less robust than r8 and r32) for the same Phase 1 task. The rank × Phase 2 task interaction at r16 is large and not predictable from the Phase 1 checkpoint alone. This limits the T*-as-Phase-1-property claim to rank 8 specifically.

---

## What still needs to be done

### Blocking (before writing the paper)

**0. ~~Cross-architecture validation (H2)~~ — DONE (2026-07-11).** Mistral-7B AGNews→MNLI: R_A at T* = 0.071 (Mistral) vs 0.064 (Llama), within 11%. σ½ normalization transfers across architectures. Paper can claim cross-architecture transferability at rank 8 for the AGNews→MNLI pair.

**0b. ~~H1 cross-architecture (3 models)~~ — DONE (2026-07-14 v5 PRIMARY).** Mistral-7B SST-2 v5: T*=[0.194,0.226] with correct methodology. GPT-2 [0.22,0.45] and Mistral [0.194,0.226] overlap at [0.22,0.23]; Llama [0.235,0.577] adjacent within 5%. KL(diverse) fails for Mistral too (insufficient slope, 2.6× at cliff vs 22.5× NLL jump). KL(SuperNI) discriminating for Mistral v5 (12.4× at cliff). 3-architecture H1 validation complete for SST-2. MNLI does not converge — noted as limitation.

**0c. ~~Fix Mistral H1 KL(SuperNI) for MNLI~~ — DONE (2026-07-11).** v2 re-run with real superni_prompts.json: KL(SuperNI) for MNLI is now real and monotone (0.353 → 0.971 → 3.771 → 7.295 → 6.793). Key finding: KL(SuperNI) > KL(diverse) at the catastrophic threshold (7.295 vs 3.410). **~~SST-2 KL(SuperNI) with correct σ½ protocol~~ — DONE (2026-07-14 v5 PRIMARY).** v5 effective batch=16: KL(SuperNI) real and discriminating (0.031 → 0.581 → 7.218 → 8.448 → 23.709). 12.4× jump at cliff — KL(SuperNI) works for Mistral SST-2, unlike KL(diverse) (2.6×).

**0d. ~~Refine Mistral σ½_pre + use correct methodology~~ — DONE (2026-07-14 v5).** v5 uses test-split sequential σ½ protocol (identical to GPT-2/Llama recompute script): σ½=1.824×10⁻⁴, verified matching `pretrained_density_relslack.json`. T*=[0.194,0.226] is the authoritative value from the PRIMARY result. Prior v3 T*=[0.347,0.397] used wrong σ½ methodology (validation split) — superseded.

**1. ~~Decide how to frame SST-2~~** — **REFRAMED (2026-07-05).** Adding SST-2→CoLA reveals 44× spread across 3 Phase 2 tasks. SST-2 Phase 1 is now characterized as an **anisotropic basin**: CoLA [0.005,0.019] < AGNews [0.036,0.073] << MNLI [0.212,0.851]. The framing is: SST-2 Phase 1 creates a cigar-shaped rather than spherical forgetting basin; isotropic σ½ cannot predict direction-dependent T*. This is a clean, testable narrative.

**2. ~~Resolve MNLI stochastic boundary~~** — **RESOLVED (2026-07-04).** 3/5 seeds FORGET at lr=2e-4 r8. T*(MNLI→SST-2 r8) = [0.041, 0.081] ≈ T*(MNLI→AGNews r8) = [0.040, 0.084]. MNLI is Phase-2-independent. H2 claim holds for 3/4 Phase 1 tasks at r8.

### Non-blocking (nice to have)

- ~~Compute σ½_A_acc for SST-2~~ — **Done.** phase1_sigma_half_acc.json exists in both sst2_to_mnli and sst2_to_agnews dirs: σ½_acc = 0.003337.
- ~~H1 with a second downstream task beyond SST-2~~ — **Done (H1-MNLI, Batch 3).**
- ~~H1 with Mistral-7B~~ — **Done (Batch 9)** — SST-2 **v5 PRIMARY** ✅ T*(relslack)=[0.194,0.226] (2026-07-14); MNLI v2 ✅ T*=[0.084,0.380]; σ½=1.824×10⁻⁴ correct; all Mistral H1 metrics complete with correct methodology.
- Narrow T* intervals by adding intermediate LR points (currently 5-point grid, intervals are 2× wide).
- ~~Refine Mistral σ½_pre density curve~~ — resolved; v5 confirms σ½=1.824e-4 correct.
- SST-2→MNLI rank=4 and rank=16 characterization (V8 anomaly): pending — running on pod GPU 1. SCP when done: `scp -P 22079 -i ~/.ssh/id_ed25519 "root@194.68.245.29:/workspace/h2_llama/sst2_to_mnli/lr*_rank4_result.json" "root@194.68.245.29:/workspace/h2_llama/sst2_to_mnli/lr*_rank16_result.json" /Users/kinakim/Github/NTRS/h2_llama/sst2_to_mnli/`

---

## Key numbers (no fabrication)

| Quantity | Value | Source |
|---|---|---|
| GPT-2 σ½_pre (1% rel slack) | 0.007191 | h1_kl/gpt2/pretrained_density_relslack.json |
| Llama σ½_pre (1% rel slack) | 0.000772 | h1_kl/meta-llama/.../pretrained_density_relslack.json |
| Mistral-7B σ½_pre (non-relslack) | **1.559×10⁻⁴** | h1_kl/mistralai/Mistral-7B-v0.1_sst2_v3/pretrained_density.json (archived) |
| Mistral-7B σ½_pre (relslack 1%) | **1.824×10⁻⁴** | h1_kl/mistralai/Mistral-7B-v0.1_sst2_v5/pretrained_density.json; verified matching relslack script; +17% vs non-relslack |
| GPT-2 T* (H1, corrected) | [0.22, 0.45] | per_param at cliff / σ½_pre=0.007191 |
| Llama T* (H1, corrected) | [0.24, 0.58] | per_param at cliff / σ½_pre=0.000772 |
| Mistral-7B T* (H1, SST-2, relslack) | **[0.194, 0.226]** | 2026-07-14 **v5 PRIMARY** ✅; ΔNLL=+0.125 (safe, acc=0.930) → +2.810 (first catastrophic); raw per_param=3.53e-5 at lower T*; R strictly monotone |
| Mistral-7B T* (H1, MNLI) | **[0.084, 0.380]** | 2026-07-11 v2; NLL cliff earlier than SST-2; task collapses at R=1.013; KL(SuperNI) real and monotone (7× safe→catastrophic) ✅ |
| H1 T* convergence (SST-2 fine-tuning) | All 3 onset near R≈[0.19,0.24] (relslack); GPT-2 ∩ Mistral at [0.22, 0.23]; Llama adjacent | Raw displacement spans **44×** (1.58e-3 to 3.53e-5); σ½_pre spans 39×; σ½ compresses to 1.2× normalized onset band |
| Llama KL-diverse at LR=1e-5 (safe, SST-2) | 7.11 | h1_kl/meta-llama/.../summary.json |
| Llama KL-diverse at LR=5e-4 (catastrophic, SST-2) | 10.67 | h1_kl/meta-llama/.../summary.json |
| Mistral KL-diverse at LR=1e-5 (SST-2, no convergence) | **0.088** | v5 — NOT saturated; much lower than Llama (7.11) |
| Mistral KL-diverse at cliff (SST-2 v5, LR=5e-5→1e-4) | **1.411 → 3.706 (2.6×)** | ΔNLL jumps 22.5×; insufficient slope failure mode |
| Mistral KL(SuperNI) at cliff (SST-2 v5, LR=5e-5→1e-4) | **0.581 → 7.218 (12.4×)** | discriminating — KL(SuperNI) works for Mistral v5 SST-2 |
| Llama KL-SuperNI at LR=1e-4 (last safe) / LR=2e-4 (first forget) — SST-2 | 2.86 / 3.30 | h1_kl/meta-llama/.../summary.json |
| Llama MMLU at LR=1e-4 (last safe) / LR=2e-4 (first forget) — SST-2 | 0.540 / 0.370 | h1_kl/meta-llama/.../summary.json |
| Llama KL-SuperNI at LR=1e-4 (last safe) / LR=2e-4 (first forget) — MNLI | 1.32 / 3.89 | h1_kl/meta-llama/.../Llama-3.2-3B_mnli/summary.json |
| Llama MMLU at LR=1e-4 / LR=5e-4 (NLL cliff) — MNLI | 0.500 / 0.250 | h1_kl/meta-llama/.../Llama-3.2-3B_mnli/summary.json |
| σ½_A_acc: MNLI / AGNews / SST-2 / CoLA | 0.002857 / 0.002756 / 0.003337 / 0.001944 | h2_llama/*/phase1_sigma_half_acc.json |
| T*(r8) MNLI Phase 1 | **[0.041, 0.081] (→SST-2) vs [0.040, 0.084] (→AGNews) — indistinguishable ✓** (5-seed, 2026-07-04) | H2-A, H2-B |
| T*(r8) AGNews Phase 1 | [0.042, 0.085] (→MNLI) vs [0.042, 0.089] (→SST-2) — indistinguishable ✓ | H2-D, H2-E |
| T*(r4) AGNews Phase 1 | [0.031, 0.062] (→MNLI) = [0.031, 0.062] (→SST-2) — identical ✓ | H2-D, H2-E r4 data |
| T*(r8) CoLA Phase 1 | [0.060, 0.118] (→MNLI) vs [0.060, 0.126] (→AGNews) — indistinguishable ✓ | H2-G, H2-H tables |
| T*(r8) SST-2 Phase 1 | [0.005,0.019] (→CoLA) / [0.036,0.073] (→AGNews) / [0.212,0.851] (→MNLI) — **44× range** | H2-C, H2-F, sst2_to_cola (2026-07-05) |
| Mistral-7B σ½_A (AGNews) | **7.62×10⁻⁴** | h2_mistral/agnews_to_mnli/phase1_sigma_half.json |
| Mistral-7B T*(r8) AGNews→MNLI | **[0.043, 0.094] median≈0.071** | h2_mistral/agnews_to_mnli/ (2026-07-11) |
| Mistral-7B T*(r8) AGNews→SST-2 | **[0.038, 0.046] median≈0.042** ⚠️corrected | 2 clean conditions: lr=1e-4, lr=2e-4; lr=5e-5 excluded (transient dip, final_acc=0.882 survived) |
| Cross-arch R_A (AGNews→MNLI) | Llama=0.064, Mistral=0.071 — **11% gap** | σ½ ratio 3.6×; raw displacement ratio 3.2×; normalized R_A ratio 0.90 |
| T*(r16) MNLI Phase 1 | [0.058, 0.123] mid=0.091 (→AGNews) vs [0.029, 0.057] mid=0.043 (→SST-2) | H2-A, H2-B Batch 2 |
| T*_raw AGNews Phase 1 r8 | 1.76–1.80 × 10⁻⁴ (all Phase 2 tasks) | compute_fullmodel_norm.py |
| T*_raw r16 (below H1 floor) | ~1.22–1.24 × 10⁻⁴ for MNLI→SST-2 r16 and AGNews→MNLI r16 | compute_fullmodel_norm.py |
| Effective rank SST-2 / AGNews / CoLA | 5.21 / 5.36 / 5.19 (mean, 112 layers) | local computation 2026-07-01 |
| Direction overlap (cosine sim) | all within ±5.2e-4 (noise floor ≈4.7e-4) | direction_overlap_results.json |
| T\*\_raw (H2, 7/8 rank-8 pairs) | **1.15–1.80 × 10⁻⁴** (cluster); SST-2→MNLI outlier at 7.07×10⁻⁴ | compute_fullmodel_norm.py output |
| T\*\_raw (H1, Llama, lower bound) | **1.70 × 10⁻⁴** | 0.22 × σ½_pre(0.000772) |
