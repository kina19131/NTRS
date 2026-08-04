# Research Clarity — NTRS Project
**Written:** 2026-07-02 | **Updated:** 2026-08-02 (full rewrite — folds in the σ½/LoRA-scope correction, the Hessian eigenvalue check, the entire NLL-vs-accuracy/fresh-probe-recovery investigation, the σ½_A_acc LoRA-scope correction, and the H1 MNLI containment error+retraction)
**Purpose:** Plain-language summary of what we have, what's trivial, what's honest, and what's left. Companion to SUMMARY.md (technical running log) and RAW_DATA.md (raw numbers, one section per experiment). Every number below is cited to its RAW_DATA.md/SUMMARY.md section so it can be re-traced — see the standing rule at the end of this file.

---

## What the project is (one paragraph)

You fine-tune a model on Task A, then fine-tune on Task B. Task A gets forgotten. We ask: can you predict this *before* running Task B, using only the Task A checkpoint?

We propose **σ½** — a number computed from the Task A checkpoint via randomized smoothing that measures how fragile the model's weights are to perturbation. We normalize the weight displacement during Task B training by σ½ to get **R_A = ‖Δθ‖ / σ½_A**. The claim: when R_A crosses a threshold T*, Task A is forgotten. σ½_A is computable before Task B starts. That's the operational value.

We also test the same idea at the pretraining→fine-tuning boundary (H1): does R = ‖Δθ‖/σ½_pre predict pretraining knowledge degradation (WikiText-2 perplexity) during downstream fine-tuning?

---

## Timeline at a glance

| Date | What happened |
|---|---|
| through 2026-07-14 | Core H1 (GPT-2/Llama/Mistral SST-2+MNLI) and H2 (8 Llama task pairs + Mistral cross-arch) results established. This is everything the pre-08-02 version of this file described. |
| 2026-07-17 | H1/H2 corrections: Mistral v5 PRIMARY SST-2 result adopted; H1 MNLI Llama/Mistral compared on a consistent relslack basis (contained); H2 AGNews cross-arch compared on a consistent accuracy-based σ½ for both models (both pairs overlap, later revised — see 07-30). |
| 2026-07-20/21 | **Hessian eigenvalue check**: does σ½ actually track real curvature (σ½ ∝ 1/√λ_max)? GPT-2 confirmed measurable (saddle point, λ_min<0); Llama/Mistral's finite-difference estimate is not stable enough to trust. |
| 2026-07-21/22 | **σ½/LoRA-scope correction**: σ½_pre had been calibrated over the whole decoder block, not LoRA's actual target modules. Recomputed correctly — GPT-2 barely moves (+7%, naming collision), Llama/Mistral nearly double (+92–97%, no collision). Under the corrected scope, GPT-2 drops out of the H1 3-way T* convergence. |
| 2026-07-22/23 | **NLL-vs-accuracy investigation** (GPT-2): does a frozen classifier's accuracy collapse reflect genuine information loss, or just readout misalignment? Built `h2_nll_forgetting.py`, then `h2_probe_recovery.py` (fresh-probe test). GPT-2 SST-2→{mnli,agnews,cola}: all three recover under a fresh probe, none instantly — "real but recoverable." |
| 2026-07-23 | GPT-2 severity sweep (fresh probe) and first Llama fresh-probe run (SST-2 Phase 1) — Llama shows a moderate-severity "real recovery" regime and a high-severity (lr=1e-3) reproducible near-chance collapse. |
| 2026-07-30 | **σ½_A_acc LoRA-scope correction** (H2's own version of the 07-21 H1 bug, found by code inspection — the recompute script had never been updated). Llama H2 σ½_A_acc values all rise 65–82%; Mistral AGNews +33%. One of two previously-overlapping H2 cross-architecture pairs (AGNews→SST-2) breaks under the corrected scope; the other (AGNews→MNLI) survives. |
| 2026-07-30 | Llama fresh-probe severity sweep + NaN diagnostic: NaN/training-divergence ruled out definitively (all 12 conditions finite). New class-count diagnostic discovers a constant-class-collapse artifact (0.478/0.522 recurring implausibly often) — flags several prior "no_recovery" verdicts as unconfirmed. |
| 2026-07-30/31 | **MNLI generality test**: does the readout-mismatch finding generalize beyond SST-2 as Phase 1? GPT-2 MNLI Phase-1 training fails outright (genuine capacity limit, not a bug); pivots to Llama. 4-round escalation, corrected 08-02: 0/3 targets actually cross the recovery threshold. |
| 2026-08-02 | **H1 MNLI containment error and retraction, same day**: a review using the wrong baseline NLL (a σ½-calibration file's own internal figure, not the fine-tuning table's baseline) incorrectly "corrected" the original containment finding to a non-containment finding. Traced and retracted — the original claim was correct all along. This file rewritten to reflect the true current state. |

---

## H1 — Pretraining boundary (GPT-2, Llama-3.2-3B, Mistral-7B-v0.1)

### Core result (SST-2 fine-tuning, block-scope σ½ — the number this project has always led with)

Fine-tuned all three models at 5 LRs on SST-2, measured WikiText-2 perplexity change (ΔNLL). σ½_pre computed from the frozen pretrained checkpoint (1% relative eval-slack convention).

- GPT-2: σ½_pre = 0.007191, T* ∈ [0.22, 0.45]
- Llama-3.2-3B: σ½_pre = 0.000772, T* ∈ [0.24, 0.58] (more precisely [0.1133, 0.2357] under the stricter 1%-threshold criterion — see below, criterion matters)
- Mistral-7B-v0.1 (v5 PRIMARY, the correct-methodology run — v3 used the wrong σ½ protocol and is archived, do not cite): σ½_pre = 1.824×10⁻⁴, T* ∈ [0.194, 0.226]

Raw displacement at T* spans **44×** across the three models (1.58e-3 for GPT-2 to 3.53e-5 for Mistral); σ½_pre itself spans 39×. Normalizing by σ½ compresses this to a **~1.2–3× band** depending on exactly which criterion you use to define "T*" (see next point) — this compression is the headline empirical result of H1.

**Criterion sensitivity, important**: under the loose "cliff-jump" criterion (T* = the LR step where ΔNLL visibly jumps), GPT-2∩Mistral overlap and Llama sits adjacent — reads as a clean 3-way story. Under the stricter, consistently-applied "1%-of-baseline-NLL" criterion, the true picture is a **chain, not a 3-way overlap**: GPT-2=[0.22,0.45], Llama=[0.1133,0.2357], Mistral=[0.03,0.19] (block-scope). GPT-2∩Llama=[0.22,0.2357] (overlap), Llama∩Mistral=[0.1133,0.1937] (overlap), but GPT-2∩Mistral=∅. Do not cite "3-architecture convergence" without stating which criterion.

### The σ½/LoRA-scope correction (2026-07-21/22) — this changes the picture above

**The bug**: σ½_pre was calibrated by perturbing the *entire decoder block*, but LoRA fine-tuning only ever displaces its own target modules (`q/k/v/o_proj` for Llama/Mistral, `c_attn`/`c_proj` for GPT-2). Recomputed with `recompute_h1_sigma_relslack.py --sigma_scope lora`, restricting the perturbation to the actual LoRA-targeted parameters, for all three models on the frozen pretrained checkpoint (no fine-tuning involved — a calibration-only recompute, no hindsight risk).

| Model | σ½ (block, old) | σ½ (lora, corrected) | Change |
|---|---|---|---|
| GPT-2 | 0.007191 | 0.007698 | **+7.0%** |
| Llama-3.2-3B | 0.000772 | 0.001517 | **+96.6%** |
| Mistral-7B-v0.1 | 1.824×10⁻⁴ | 3.510×10⁻⁴ | **+92.4%** |

This confirms the exact prediction made when the question was first raised: GPT-2's LoRA targets (`c_attn`, `c_proj`) happen to collide with its MLP module naming, so GPT-2's LoRA scope already overlapped most of what block-scope certified — recomputing barely moves it. Llama/Mistral's LoRA targets are attention-only with no such collision, so narrowing to the true scope reveals a much tighter basin — σ½ nearly doubles for both.

**T* under the corrected (lora) scope, same 1%-threshold criterion**:

| Model | T* (block, old) | T* (lora, corrected) |
|---|---|---|
| GPT-2 | [0.2200, 0.4534] | [0.2056, 0.4235] |
| Llama-3.2-3B | [0.1133, 0.2357] | [0.0576, 0.1199] |
| Mistral-7B (v5) | [0.0300, 0.1937] | [0.0156, 0.1007] |

Pairwise overlap, lora-scope: **GPT-2∩Llama = ∅** (0.2056 > 0.1199 — no overlap at all, unlike block-scope where they overlapped). Llama∩Mistral = [0.0576,0.1007] (still overlaps, narrower). GPT-2∩Mistral = ∅ (as before). **Under the corrected, more methodologically defensible scope, GPT-2 drops out of the cross-architecture convergence story entirely — only the Llama/Mistral pair survives.** This is a real finding, not a nuisance: it parallels exactly what happened to H2's cross-architecture comparison under the same class of correction (see H2 section below) — one of two things drops out under the fix in both cases.

*(A data-hygiene bug was also found and fixed during this recompute: a stale, superseded copy of the Llama SST-2 results lived at a different path with different numbers. The stale copy has been archived to `archive/old_results/`; all Llama SST-2 numbers in this document use the authoritative path.)*

### MNLI fine-tuning (Llama vs. Mistral) — containment holds, in both scope conventions

Convergence does not hold between SST-2 and MNLI (different tasks have different T*), but within MNLI, Llama and Mistral **do** converge:

- Llama MNLI T* = [0.1331, 0.2329] (block-scope), [0.0677, 0.1184] (lora-scope)
- Mistral MNLI T* = [0.0715, 0.3243] (block-scope), [0.0372, 0.1686] (lora-scope)
- **Llama's bracket is fully contained inside Mistral's, in both scope conventions.**

**A process note worth keeping on record**: on 2026-08-02, a review flagged this containment claim as wrong, using a threshold computed from `nll_baseline` = 3.05859375 — a figure from `pretrained_density_relslack.json`, the file that calibrates σ½_pre itself. That number is *not* what the fine-tuning table's own `ΔNLL` values are computed against; the correct baseline is that table's own `nll_wiki_base` = 3.36375, identical for every row. Traced precisely (2026-08-03): the two numbers differ because they come from three stacked eval-config differences, not one subtle mismatch — `nll_baseline` (`recompute_h1_sigma_relslack.py`'s own `load_wikitext_nll_data`/`compute_nll`) uses WikiText-2's **test** split, **sequential** chunks from the start of the text, seq_len=128, n=50; `nll_wiki_base` (`h1_kl_comparison.py`'s `load_wikitext_for_kl` + `compute_nll_loss`) uses the **validation** split, **random**-start chunks, seq_len=64, n=200. Different split, different sequence length, different sampling scheme, different sample count — any one alone would move the number. Redone with the correct baseline, the original "fully contained" classification is exactly right — the "correction" was itself the error, and was retracted the same day. **Standing rule going forward, stated generally (not just for this pair of files): always pair a baseline NLL with the same eval pipeline that produced the post-fine-tuning NLL it's being subtracted from.** For every fine-tuning table, that's `nll_wiki_base` — never `nll_baseline`, which belongs only to the σ½-calibration pipeline and should never be compared against a fine-tuning table's post-FT NLL. This generalizes to H2's own Task-A pre/post-Phase-2 NLL pairing and to any future third eval pipeline added to this codebase. Full trace in RAW_DATA.md's H1 MNLI section. *(Note: `h1_kl_comparison.py` also computes a third, differently-configured WikiText-2 NLL internally, line ~683, used only to auto-derive `eval_slack` and never written to any output file — it shares the name `nll_baseline` with the calibration script's saved value but is a third, separate number. Not the cause of this retraction, but a latent source of the same confusion if anyone goes looking for "nll_baseline" in this codebase later.)*

### The Hessian eigenvalue check (2026-07-20/21) — does σ½ track real curvature?

Motivating question: σ½ is derived from randomized smoothing under a quadratic-basin assumption (σ½ ∝ 1/√λ_max). Does the basin the model is actually sitting in match that assumption? Tested via Lanczos-based Hessian eigenvalue estimation, restricted to LoRA's target-module subspace.

- **GPT-2**: exact double-backward HVP (fp32, no perturbation noise), 3 independent runs, all agreeing to within 0.005 on both extremal eigenvalues. **λ_min is confirmed negative in all 3 runs (−36.05) — the checkpoint sits in a genuine saddle point, not a local minimum**, within the LoRA-scoped subspace. Measured σ½ is ~7.6× smaller than the single-eigenvalue quadratic theory predicts (measured/theory ≈ 0.132) — consistent, expected direction for a saddle vs. a clean minimum.
- **Mistral-7B**: finite-difference HVP (bf16, necessarily, at this scale). Two "identical" runs (same eps, same batch) gave a 2× different magnitude for the same quantity (2×eps λ_max: ~957 vs. ~1909) — **not stable across even a fresh random seed**. One run's own internal eps-stability check passed; that "stable: True" is retracted as evidence once the cross-run instability was found.
- **Llama-3.2-3B**: 4 attempts, never once passed a 20% eps-doubling stability threshold (best result: 34% swing). No monotonic relationship between eps and stability, consistent with genuine measurement noise, not a fixable step-size problem.

**What this means**: GPT-2's curvature is directly, reliably measurable — and it's a saddle point, not the clean quadratic minimum the theory assumes, yet σ½ still empirically works for GPT-2 (see T* results above). For Llama/Mistral (>2B params), curvature is **not reliably measurable at all** with the tools available — this is a genuine limitation, not a null result; the question "does σ½ track real curvature for the large models" is open, not answered.

### Other H1 findings that still stand (from the pre-08-02 investigation, unchanged by anything above)

- **KL-diverse (instruction-domain KL) fails as a forgetting detector, for two different reasons across models.** Llama: saturates in the safe zone (KL=7.11 at LR=1e-5, safe; only 10.67 at LR=5e-4, catastrophic — the metric is already large before any real damage is done). Mistral: doesn't saturate (0.088 at LR=1e-5) but has insufficient slope at the actual cliff (1.411→3.706, 2.6×, while ΔNLL jumps 22.5×). Both are non-monotone across LR (varies with the *direction* of weight displacement, not just magnitude) — expected under a Fisher-approximation view of KL, not a measurement artifact. Confirmed for both SST-2 and MNLI fine-tuning (this generalization closed the "is this SST-2-specific" question, V6).
- **KL-wiki (pretraining-domain KL) is monotone and tracks ΔNLL for all three models** — the KL-diverse failure is domain-specific, not a flaw in the KL concept.
- **KL-SuperNI** is mostly monotone but a poor threshold detector for Llama (15% change while ΔNLL jumps 32× at the safe/forget boundary); it works well for Mistral SST-2 v5 (12.4× jump) and for Mistral MNLI (monotone, ~7× jump).
- **MMLU zero-shot accuracy drops at the same R threshold as NLL**, for both Llama and Mistral — R predicts factual-knowledge degradation, not just perplexity.

---

## H2 — Sequential task forgetting (Llama-3.2-3B primary; Mistral-7B cross-architecture)

### Core result (unaffected by the LoRA-scope correction below — this is a within-task, ratio-invariant claim)

For 4 Phase-1 tasks (MNLI, AGNews, SST-2, CoLA), fine-tuned Llama, computed σ½_A from the checkpoint, then fine-tuned on 2-3 different Phase-2 tasks. For **3 of 4 Phase-1 tasks (MNLI, AGNews, CoLA)**, the R_A interval at which Phase-1 accuracy starts to break is indistinguishable across different Phase-2 targets — the threshold behaves like a property of the Phase-1 checkpoint, not of which task comes next. MNLI confirmed via 5-seed replication at its stochastic boundary.

For **SST-2 as Phase 1**, T*(r8) spans a **44× range** across three Phase-2 targets: [0.005,0.019] (→CoLA, most fragile) vs. [0.036,0.073] (→AGNews) vs. [0.212,0.851] (→MNLI, most resistant). SST-2 is simultaneously the most fragile and most resistant Phase-1 model depending on direction — consistent with an anisotropic forgetting basin that isotropic σ½ cannot capture. This is the project's standing open anomaly, referred to throughout as **V8**.

*(Why this claim survives the σ½_A_acc correction below unchanged: it's a within-task, same-Phase-1-checkpoint comparison — both sides of every comparison divide by the identical corrected constant, so ratios/overlaps within one Phase-1 task are unaffected. Only absolute R_A values and cross-Phase-1-task/cross-architecture comparisons change.)*

### The σ½_A_acc LoRA-scope correction (2026-07-30) — H2's own version of the H1 bug

**The bug**: H2's Phase-1 σ½_A_acc calibration script (`recompute_sigma_half_acc.py`) was never updated after the H1 LoRA-scope fix was found (2026-07-21) — confirmed by code inspection: it used whole-block scoping exclusively, and its last modification date (2026-06-30) predates the fix's existence by three weeks. Fixed the same way as H1 (`--sigma_scope {lora,block}` flag, default lora), run on all 5 H2 Phase-1 checkpoints (4 Llama tasks + 1 Mistral task; calibration-only, no fine-tuning, no hindsight risk).

| Phase-1 task | σ½_A_acc (block, old) | σ½_A_acc (lora, corrected) | Change |
|---|---|---|---|
| Llama MNLI | 2.857×10⁻³ | 4.726×10⁻³ | **+65.4%** |
| Llama SST-2 | 3.337×10⁻³ | 5.540×10⁻³ | **+66.0%** |
| Llama AGNews | 2.756×10⁻³ | 5.022×10⁻³ | **+82.2%** |
| Llama CoLA | 1.944×10⁻³ | 3.476×10⁻³ | **+78.8%** |
| Mistral AGNews | 7.314×10⁻⁴ | 9.752×10⁻⁴ | **+33.3%** |

Since σ½_A_acc is a per-Phase-1-checkpoint constant, this correction leaves all **within-Phase-1-task** comparisons unaffected (see note above) — the "3/4 tasks are isotropic" and "44× SST-2 spread" findings both stand exactly as before. What changes is any comparison **across** Phase-1 tasks or **across architectures**.

**Cross-architecture comparison, recomputed — the one place this correction actually bites**:

| Pair | T* (block, old) | T* (lora, corrected) |
|---|---|---|
| Llama AGNews→MNLI | [0.0421, 0.0855] | [0.0231, 0.0469] |
| Llama AGNews→SST-2 | [0.0419, 0.0887] | [0.0230, 0.0487] |
| Mistral AGNews→MNLI | [0.0203, 0.0852] | [0.0152, 0.0639] |
| Mistral AGNews→SST-2 | [0.0683, 0.1046] | [0.0512, 0.0785] |

**AGNews→MNLI still overlaps** (Llama's bracket sits entirely inside Mistral's). **AGNews→SST-2 no longer overlaps** (Llama's upper bound, 0.0487, now falls just short of Mistral's lower bound, 0.0512 — a ~5% gap). This directly parallels what happened to H1's cross-architecture chain under the same correction (GPT-2 dropping out) — **one of two previously-overlapping cross-architecture pairs survives the fix, the other doesn't.** This means the prior "H2 architecture concern fully addressed" conclusion needs to be read as "partially addressed — one of two tested pairs overlaps under the corrected, consistent methodology," not as a closed question.

**Not yet done**: the core within-architecture H2 tables (the MNLI/AGNews/CoLA/SST-2 T* values quoted above) have **not** been recomputed under lora-scope — only the calibration constants and the one cross-architecture comparison were. This is fine for the isotropic/anisotropic claims (ratio-invariant, as explained above) but means absolute R_A values in the main H2 tables are still block-scope numbers; don't mix them with the lora-scope cross-architecture numbers above without noting the scope.

---

## The readout-mismatch investigation — NLL-vs-accuracy and fresh-probe recovery

This whole line of work asks one question, from several angles: when a frozen linear classifier's accuracy on Task A collapses during Phase-2 training, has the model genuinely lost the capability, or has the classifier's readout direction simply become misaligned with where the information now sits in the (perturbed) representation? This matters directly for interpreting every accuracy-based T* in H2 — if it's mostly readout mismatch, "forgetting" is a weaker, more optimistic claim than it sounds.

### Step 1 — NLL vs. accuracy on GPT-2 (2026-07-22)

Compared Task-A forgetting measured two ways on the same checkpoints: frozen classifier accuracy (existing H2 signal) vs. NLL on Task A's own held-out text (no classifier involved at all). GPT-2, SST-2 Phase 1 → {mnli, agnews, cola} Phase 2, 5 LRs each.

**Finding**: accuracy-based forgetting triggers in agnews (3/5 LRs) and cola (4/5 LRs), but **NLL-based forgetting never triggers anywhere in the entire 15-condition sweep** — not even in the conditions where accuracy clearly collapsed. This is the first direct evidence that classifier-accuracy collapse and genuine loss of task-relevant information (as measured by a classifier-independent signal) can come apart — motivating the fresh-probe-recovery approach below as a more direct test.

### Step 2 — Fresh-probe recovery, GPT-2 (2026-07-23)

Design: freeze the Phase-2-perturbed backbone, train a brand-new linear probe from scratch on Task A. If the old classifier's collapse were genuine information loss, a fresh probe shouldn't be able to recover either. If it's readout mismatch, a fresh probe should recover quickly.

**Result**: all three GPT-2 targets (mnli, agnews, cola) land in a "recovers, but not instantly" bucket — a fresh probe reaches above 90%-of-baseline accuracy within 55–200 steps (out of a 3,000-step Phase-1 training budget), including the two targets (agnews, cola) where the old classifier had collapsed to near-chance. **Interpretation**: the information is not gone (rules out pure information destruction), but recovery isn't free either (rules out "the old classifier's failure was a trivial, costless misalignment"). Something real happens to how the information is encoded — real enough to require actual training to re-extract — but it's recoverable, cheaply, for all three GPT-2 conditions tested.

### Step 3 — Severity sweep and the NaN question (GPT-2 2026-07-23, Llama 2026-07-23/30)

Pushed the same test across a range of LRs (more severe displacement) to see whether recovery ever genuinely fails.

- **GPT-2** (sst2→cola only, 6 LRs): a clean monotonic progression from 55-step recovery up through 200-step recovery, then flips to `no_recovery` at the two highest LRs (2e-3, 5e-3). Those two `no_recovery` results showed strong circumstantial evidence of being a training-numerics artifact (exactly constant accuracy across 41 evaluation points; old and fresh-untrained classifiers agreeing to 4 decimal places — the classic signature of NaN hidden states making the classifier's weights irrelevant) but this was **never directly confirmed** (no NaN diagnostic existed yet, and the script doesn't save endpoint checkpoints for GPT-2 to check after the fact).
- **Llama** (sst2 Phase 1 → all 3 targets, first pass 2026-07-23, two LRs): moderate severity (5e-4) shows real, substantial recovery just short of Llama's much stricter threshold (0.864, vs. GPT-2's 0.6966) — agnews peaks at 0.902 (misses only on the sustained-crossing check), cola climbs to 0.754. High severity (1e-3) collapses all three targets to a reproducible near-chance state that **independently reproduces a value already on record from an unrelated script months earlier** (SST-2→MNLI lr=1e-3 rank=8 gives final_acc=0.478 in both `h2_sequential_llama.py`'s original run and this probe-recovery run) — real corroboration this specific (model, task pair, LR) combination reliably collapses, whatever the mechanism.
- **Llama severity sweep + NaN diagnostic (2026-07-30)**: extended `h2_probe_recovery.py` with a direct NaN/Inf check on hidden states, reran with intermediate LR points. **Result: `phase2_hidden_nan`/`probe_recovery_hidden_nan` are `False` in all 12 conditions, including every plateaued one — the NaN/training-divergence hypothesis is definitively ruled out for Llama.** Also found: **AGNews's collapse is non-monotonic in LR** — 5e-4 recovers, 6e-4 collapses, 8e-4 recovers again, 1e-3 collapses again (verified via full 400-point curve inspection each time, not endpoints). MNLI, by contrast, shows a clean sharp boundary between (6e-4, 8e-4]. This complicates the idea that R_A (a magnitude) alone should predict recovery difficulty — direction of displacement appears to matter, consistent with other direction-sensitive findings in this project (H1's direct-R sweep, SST-2's own anisotropy).

### The constant-class-collapse artifact (found 2026-07-30, an important methodological correction)

Tallying every accuracy value across all 27 probe-recovery result files collected up to that point: **0.478 and 0.522 (which sum to exactly 1.000, and are exactly 239/500 and 261/500 on the fixed 500-example validation split) occur 372 and 295 times respectively** — an order of magnitude above any other value, and **11 of 27 conditions'** *final* recorded value is exactly one of these two, including every single Llama `no_recovery` condition. This is the exact signature of a classifier converging to always predict one fixed class, which mechanically produces one of these two accuracies from the split's class balance alone, independent of what's actually happening in the hidden states.

This is consistent with two different explanations, not yet distinguished at the time: (a) genuine collapse, correctly detected (majority-class prediction is the loss-minimizing behavior if there truly is no signal), or (b) the probe's own optimization getting stuck in a degenerate local minimum despite real, recoverable signal being present. **Fixed**: `h2_probe_recovery.py` now includes `_prediction_class_counts`, reporting the literal predicted-class histogram, saved in every result JSON going forward. **This directly downgrades several earlier "no_recovery" verdicts (GPT-2's 2e-3/5e-3, and the Llama non-monotonicity finding above) from "confirmed genuine collapse" to "NaN specifically ruled out, but constant-class-collapse not yet distinguished from a probe-training artifact"** — an open item, not yet resolved.

### Step 4 — Does this generalize beyond SST-2? The MNLI generality test (2026-07-30/31, corrected 2026-08-02)

Every fresh-probe-recovery run so far used SST-2 as Phase 1 (the identified V8 anomaly). Open question: does readout-mismatch also show up for Phase-1 tasks that *aren't* anomalous — MNLI, AGNews, CoLA — whose H2 transfer results already looked clean?

**GPT-2 MNLI Phase-1 training failed outright**, 3 attempts (varying LR 2.5× and LoRA rank 4×), all stuck at ≈33% accuracy (3-way chance) with loss flat at ln(3) for the entire run. Tokenizer truncation was directly checked and ruled out (mean tokenized length 41.2/128 max, only 1/200 sampled examples truncated at all). Most likely explanation: a genuine base-model capacity limitation for GPT-2 at this task, not a bug — pivoted to Llama's existing, already-validated MNLI checkpoint (88.4% accuracy).

**Llama MNLI → {sst2, agnews, cola}, progressive budget escalation (200→2000 probe-training steps across 4 rounds), with the new class-count diagnostic deployed live:**

The diagnostic caught a real artifact on its **first actual use**: at round v2 (400 steps, default probe_lr), `mnli→cola`'s fresh probe predicted one class 99.2% of the time — a genuine collapse, fixed by lowering probe_lr for the v3 re-run (confirmed resolved: predictions spread across all 3 classes afterward).

**Corrected result (2026-08-02, after closer scrutiny of round-by-round curve shapes — this session's own retraction of an earlier overclaim, described immediately below):**

- **`sst2`** (strongest case): per-round gains of +0.054, +0.084, +0.070 — real, if now-decelerating, improvement. Peak 0.798, just **0.0024 short** of the 0.7956 threshold, oscillating without a sustained crossing.
- **`agnews`** (genuine but weaker): gains of +0.022, +0.062, +0.034, decelerating faster, ending **0.198 short** of threshold at 0.598. Checked and ruled out majority-class bias directly by arithmetic: the classifier's most-predicted class alone caps at ~33% accuracy on MNLI's balanced 3-way validation set, so 59.8% requires genuine discriminative signal.
- **`cola`**: **not actually a trend at all.** Re-examining the pre-fix `v1` measurement (200 steps, same default probe_lr that caused v2's confirmed collapse) shows the identical shape of degeneracy (`{'2':192,'0':306,'1':2}`, class 1 barely predicted) — just less extreme. This means `v1` was already compromised, leaving exactly **one** trustworthy cola measurement in the whole investigation (`v3=0.362`, the only round run with the corrected probe_lr) and no valid before/after comparison at all.

**The honest bottom line: zero of three tested MNLI Phase-2 targets crossed the recovery threshold, at any budget tested** — a real, qualitative difference from SST-2-as-Phase-1, where all three GPT-2 conditions crossed cleanly within 55–200 steps. The earlier framing of this result as "generalizes, at least partially" **overstated it** and was corrected the same day it was first written. The accurate claim: MNLI shows the *same qualitative signature* as SST-2 (old-classifier collapse + non-degenerate, class-count-verified recovery direction, not chance-level noise) without yet reaching the SST-2-established recovery bar for any condition — a partial, unresolved generality test, not a confirmed extension. Since GPT-2 never produced a usable MNLI checkpoint, there is also no cross-architecture check for this specific question, unlike the original SST-2 investigation.

**Decision (explicit, strategic, made 2026-07-31/08-02): stop chasing V8's mechanism experimentally.** The graded, decelerating pattern across sst2/agnews/cola is itself being treated as the reportable finding, rather than forcing further budget escalation for diminishing returns. Write it up as an honest open question. One narrow, optional, cheap follow-up remains on the table: running the NaN diagnostic on GPT-2's original 2e-3/5e-3 "no_recovery" conditions (never checked, since the diagnostic postdates that run) — a single script rerun, not a new experimental design, not yet done as of this writing.

---

## Mechanistic hypotheses tested for the SST-2 anomaly (V8) — all null, plus one still-open

Four scalar/structural explanations for why SST-2's T* is Phase-2-dependent (the 44× spread) have been tested:

1. **Binary/multi-class hypothesis** — if class count caused it, CoLA (also binary) should show the same inconsistency. It doesn't (CoLA's T* is consistent across Phase-2 targets, subject to the "only 1 trustworthy cola data point" caveat discovered later — see above). **Broken.**
2. **Direction overlap (global cosine similarity)** — cosine similarity between Phase-1 ΔW and Phase-2 gradient directions is ≈±0.0005 for all 4 pairs tested, indistinguishable from noise floor. **Null.**
3. **Effective rank of LoRA ΔW** — SST-2=5.21, AGNews=5.36, CoLA=5.19 (112 layers) — identical within 3%. **Null.**
4. **Rank-limited capacity** (rank-8 MNLI LoRA can't span SST-2's critical subspace) vs. **basin anisotropy** (SST-2's basin is genuinely cigar-shaped) — a projection-based proxy test was inconclusive (a seeding confound was found and fixed; the corrected result showed no clear ordering, Pearson r=-0.12). A more definitive gradient-interference test (`gradient_interference.py`) was designed but its actual run/result predates this session's readout-mismatch work and is not otherwise verified here.
5. **Readout mismatch (fresh-probe recovery)** — the newest and most substantive hypothesis, described in full above. **Not null, but not conclusively confirmed as *the* explanation either**: it clearly shows real, recoverable signal survives Phase-2 training even where accuracy collapses, for both SST-2 (fully) and MNLI (partially, weaker evidence) as Phase 1. It does not, by itself, explain the specific 44× *magnitude* of SST-2's Phase-2-dependence, nor does it explain AGNews's found LR-non-monotonicity. **This is the live, honestly-open thread**, not a closed null result like 1-3.

---

## Triviality assessment — honest version

### What's trivially true (not a contribution)
**"R is monotone with forgetting"** — within a single LR sweep, R and LR are co-monotone by construction (higher LR → larger ‖Δθ‖ → larger R). This reduces to "higher LR → more forgetting," decades-old knowledge, and cannot be claimed as a finding on its own.

### What's non-trivial
1. **KL-diverse fails as a forgetting detector, and fails for structurally different reasons on different models** (H1) — saturation for Llama, insufficient slope for Mistral, both useless, both confirmed for two different fine-tuning tasks.
2. **T* is indistinguishable across Phase-2 tasks for 3/4 Phase-1 tasks** (H2) — this is the core operational claim: σ½_A computed before Phase 2 gives you the threshold for any future Phase-2 task, for MNLI/AGNews/CoLA as Phase 1.
3. **A sharp threshold exists at all** — accuracy goes from healthy to near-chance in one LR step, not a gradual curve, across essentially every task pair tested.
4. **T* converges across model scale/architecture after σ½ normalization** — real, but now with real caveats attached from the two LoRA-scope corrections: H1's convergence is a Llama/Mistral pair, not a clean 3-way GPT-2/Llama/Mistral story, once the scope is corrected. H2's cross-architecture convergence is one-of-two pairs (AGNews→MNLI survives, AGNews→SST-2 doesn't), also post-correction. Both corrections point the same direction: the *strong form* of the convergence claim ("works for everything, everywhere") does not survive the more careful methodology; a *real but narrower form* does.
5. **T*_raw clusters near ~1.75×10⁻⁴ for most H2 rank-8 pairs**, with SST-2's two outliers being directional, not scalar. (This finding predates and is unaffected by the σ½_A_acc scope correction, since it's about raw displacement, not R_A.)
6. **Genuine, non-costless information survives many "forgetting" events** (the fresh-probe-recovery work) — the single biggest new addition to the honest-claims list since 07-14. Frozen-classifier accuracy collapse routinely overstates true information loss; a fresh probe recovers real signal in essentially every GPT-2 condition and most moderate-severity Llama conditions tested. This does not overturn H2's core T* claim (which is about when the *old, frozen* classifier breaks, and that's still real, sharp, and useful) — but it means "Task A is forgotten" needs a more careful definition than "the original classifier's accuracy dropped."

### What's overclaimed (needs to be dialed back, or already was and got corrected)
- **"Within 5% consistency"** for T* midpoints — these are midpoints of intervals that are each ~2× wide. The honest statement is "the intervals are indistinguishable," not "measured to 2% precision."
- **"σ½ is better than raw ‖Δθ‖"** needs the two-part defense already on record: σ½ explains *why* a constant raw threshold appears within one architecture (basin size is roughly constant across Llama H2 tasks), and σ½ is what makes H1 and H2 comparable to each other at all (different basin sizes, same raw-displacement floor).
- **"3-architecture H1 T* convergence"** and **"H2 architecture concern fully addressed"** — both were true under the original (block-scope) methodology and both partially break under the corrected (lora-scope) methodology. Neither should be cited without specifying which scope.
- **"Generalizes, at least partially" (MNLI generality test)** — written, then corrected the same day (2026-08-02) to "same qualitative signature, does not yet cross the established bar." Keep citing the corrected version.
- **"Fully contained" (H1 MNLI Llama/Mistral)** — briefly, incorrectly flagged as broken (2026-08-02), then correctly retracted the same day. The original claim stands. Documented here as a caution about baseline-NLL bookkeeping, not as an active concern.

---

## What the honest claims are now

### H1
> On GPT-2 (117M), Llama-3.2-3B (3B), and Mistral-7B-v0.1 (7B) fine-tuned on SST-2, KL divergence on instruction-domain prompts fails as a pretraining-forgetting detector via two different mechanisms depending on the model (saturation vs. insufficient slope), while KL on WikiText-2 (the pretraining domain) remains monotone for all three — the failure is domain-specific. R = ‖Δθ‖/σ½_pre, computed with σ½_pre correctly restricted to LoRA's target-module subspace, gives a pre-run forgetting budget. Under this corrected scope, **Llama and Mistral's SST-2 T* brackets still overlap ([0.0576,0.1199] vs. [0.0156,0.1007])**, but **GPT-2 no longer overlaps with either** — the strongest form of "3-architecture convergence" does not survive the corrected methodology; a real 2-architecture (Llama/Mistral) convergence does. For MNLI fine-tuning specifically, Llama and Mistral do converge — Llama's bracket is fully contained inside Mistral's, in both scope conventions. A direct Hessian-curvature check confirms GPT-2 sits in a genuine (measurable) saddle point; Llama/Mistral's curvature could not be reliably measured with finite-difference methods at this parameter scale — whether σ½ tracks real curvature for the large models remains open, not resolved either way.

*Scope: 3 models, 2 fine-tuning tasks (SST-2 all 3 models; MNLI Llama+Mistral), 5-point LR grid, two σ½ scoping conventions now on record (cite whichever is being used explicitly).*

### H2
> For MNLI, AGNews, and CoLA Phase-1 tasks, the R_A interval at which Phase-1 forgetting begins is indistinguishable across different Phase-2 tasks at rank 8 — T* behaves as a property of the Phase-1 checkpoint. For SST-2 Phase 1, T*(r8) spans a 44× range across three Phase-2 targets, the project's standing open anomaly (V8). Cross-architecture transfer (Llama vs. Mistral, AGNews Phase 1) holds for one of two tested Phase-2 targets (MNLI) under the corrected, LoRA-scoped σ½_A_acc calibration; the other (SST-2) no longer overlaps once the scope bug is fixed — a real, if partial, walk-back from the original "fully addressed" claim. Separately, whether the frozen-classifier accuracy signal underlying every T* in this section reflects genuine information loss or just readout misalignment was tested directly via fresh-probe recovery: for GPT-2, a fresh probe recovers real signal in every SST-2-as-Phase-1 condition tested, and for Llama, in most moderate-severity conditions — meaning the T* thresholds above should be read as "where the original readout breaks," not as "where the capability is destroyed." Whether this readout-mismatch pattern also holds when a non-anomalous task (MNLI) is Phase 1 is not yet confirmed: none of three tested Phase-2 targets reached the recovery bar, though all three show the same qualitative signature (collapse + non-degenerate recovery direction) rather than a clean chance-level plateau.

*Scope: Llama-3.2-3B primary (rank 8, some rank 4/16/32 data), Mistral-7B cross-architecture check on AGNews Phase 1 only. Fresh-probe-recovery work: GPT-2 (SST-2 and MNLI as Phase 1) + Llama (SST-2 and MNLI as Phase 1), single seed throughout — no seed replication on any probe-recovery result.*

---

## Two verification episodes worth keeping on record (process notes, not results)

These are documented because they establish the correct way to challenge an existing number in this project, and because the user has explicitly and emphatically asked that the specific error not recur.

**1. H1 MNLI containment (2026-08-02).** An attempt to "correct" the established Llama/Mistral MNLI containment claim used `nll_baseline` from the σ½-calibration file (a different script's own internal WikiText-2 measurement, used only to calibrate σ½_pre) instead of `nll_wiki_base` from the fine-tuning results table (the actual baseline `ΔNLL` is computed against). This produced an incorrect reclassification. It was caught because the cited threshold value didn't match a previously-established number, traced to its exact source file/field, and retracted the same day — the original claim was correct all along. **Standing rule**: before using any number to challenge or "correct" an existing claim, identify its exact source file and field, and never assume two same-sounding values (e.g. two different "baseline NLL" measurements for the same checkpoint) are interchangeable. The general form: always pair a baseline with the eval pipeline that produced the number it's being compared against, not with whichever same-named value is closest to hand.

**2. MNLI generality test overclaim (2026-07-31/08-02).** The first write-up of the MNLI fresh-probe-recovery escalation said the readout-mismatch finding "generalizes, at least partially" to MNLI as Phase 1. Closer scrutiny (per-round increment analysis, re-examining cola's pre-fix v1 data for the same collapse signature later confirmed at v2, and an explicit majority-class-bias check for agnews) showed 0/3 targets actually crossed the recovery threshold — corrected the same day to the more precise "same qualitative signature, not yet at the established bar."

---

## What still needs to be done

### Open, real questions (not blocking, but should be in Limitations)
- **σ½ scope convention**: block-scope and lora-scope are both on record for H1 and H2, with different conclusions about cross-architecture convergence. A decision on which is the primary reported convention (lora-scope is more methodologically defensible — it perturbs exactly what fine-tuning touches) has not been formally made for the write-up.
- **V8 (SST-2 anomaly) mechanism**: 4 of 5 tested mechanistic hypotheses are null; readout-mismatch is the one live, substantive lead but does not fully explain the 44× magnitude or AGNews's LR-non-monotonicity. Explicit decision made 2026-07-31/08-02 to stop chasing this experimentally and write it up as an honest open question, rather than continue escalating probe budgets for diminishing returns.
- **Constant-class-collapse artifact**: several `no_recovery` verdicts (GPT-2's 2e-3/5e-3, Llama's AGNews non-monotonicity at 6e-4/1e-3) are not yet distinguished from genuine collapse using the class-count diagnostic that would settle it — the diagnostic exists but hasn't been run retroactively on those specific conditions.
- **AGNews's LR-non-monotonicity** (recovers→collapses→recovers→collapses across 5e-4→6e-4→8e-4→1e-3) is a single-seed finding, not yet replicated with a different seed.
- **Hessian curvature for Llama/Mistral** remains unmeasurable with current finite-difference tooling — whether σ½ actually tracks curvature at this scale is open, not resolved.
- **Optional, cheap, not yet done**: run the NaN diagnostic on GPT-2's original 2e-3/5e-3 fresh-probe severity-sweep conditions (a single script rerun using existing tooling — flagged as worth doing "only if genuinely cheap," decision not finalized as of this writing).
- **Core H2 within-architecture tables** have not been recomputed under lora-scope σ½_A_acc — only the calibration constants and the one cross-architecture comparison were. Ratio-based/isotropy claims are unaffected (see explanation above), but don't cite absolute lora-scope R_A values for the main MNLI/AGNews/CoLA/SST-2 tables — those are still block-scope.

---

## Key numbers (verified against source before writing — see the two verification episodes above for why this matters)

| Quantity | Value | Source |
|---|---|---|
| GPT-2 σ½_pre, block-scope (relslack) | 0.007191 | h1_kl/gpt2/pretrained_density_relslack.json |
| GPT-2 σ½_pre, lora-scope (relslack) | 0.007698 | RAW_DATA.md § σ½/LoRA-Scope Correction |
| Llama σ½_pre, block-scope (relslack) | 0.000772 | h1_kl/meta-llama/.../pretrained_density_relslack.json |
| Llama σ½_pre, lora-scope (relslack) | 0.001517 | RAW_DATA.md § σ½/LoRA-Scope Correction |
| Mistral σ½_pre, block-scope (relslack, v5) | 1.824×10⁻⁴ | h1_kl/mistralai/Mistral-7B-v0.1_sst2_v5/pretrained_density.json |
| Mistral σ½_pre, lora-scope (relslack) | 3.510×10⁻⁴ | RAW_DATA.md § σ½/LoRA-Scope Correction |
| H1 T* GPT-2, block / lora | [0.2200,0.4534] / [0.2056,0.4235] | RAW_DATA.md § σ½/LoRA-Scope Correction |
| H1 T* Llama, block / lora | [0.1133,0.2357] / [0.0576,0.1199] | RAW_DATA.md § σ½/LoRA-Scope Correction |
| H1 T* Mistral v5, block / lora | [0.0300,0.1937] / [0.0156,0.1007] | RAW_DATA.md § σ½/LoRA-Scope Correction |
| Llama MNLI T*, block / lora | [0.1331,0.2329] / [0.0677,0.1184] | RAW_DATA.md H1 MNLI table; retraction note |
| Mistral MNLI T*, block / lora | [0.0715,0.3243] / [0.0372,0.1686] | RAW_DATA.md H1 MNLI table; retraction note |
| Llama MNLI fine-tuning table baseline NLL (`nll_wiki_base`) | 3.36375 | H1 MNLI fine-tuning table — use this, NOT the calibration file's `nll_baseline` |
| GPT-2 λ_min (Hessian, 3 runs) | −36.05 (all 3 agree to 0.005) | RAW_DATA.md § Hessian Eigenvalue Check |
| GPT-2 measured/theory σ½ ratio | 0.132 | RAW_DATA.md § Hessian Eigenvalue Check |
| Llama Hessian stability (best of 4 attempts) | 34% swing, never <20% | RAW_DATA.md § Hessian Eigenvalue Check |
| σ½_A_acc (Llama), block / lora — MNLI, SST-2, AGNews, CoLA | 2.857e-3/4.726e-3, 3.337e-3/5.540e-3, 2.756e-3/5.022e-3, 1.944e-3/3.476e-3 | RAW_DATA.md § H2 σ½_A_acc LoRA-Scope Correction |
| σ½_A_acc (Mistral AGNews), block / lora | 7.314e-4 / 9.752e-4 | RAW_DATA.md § H2 σ½_A_acc LoRA-Scope Correction |
| H2 cross-arch AGNews→MNLI T*, block / lora | Llama [0.0421,0.0855]/[0.0231,0.0469]; Mistral [0.0203,0.0852]/[0.0152,0.0639] — overlaps both scopes | RAW_DATA.md § H2 σ½_A_acc LoRA-Scope Correction |
| H2 cross-arch AGNews→SST-2 T*, block / lora | Llama [0.0419,0.0887]/[0.0230,0.0487]; Mistral [0.0683,0.1046]/[0.0512,0.0785] — overlaps block, NOT lora | RAW_DATA.md § H2 σ½_A_acc LoRA-Scope Correction |
| T*(r8) MNLI Phase 1 (block-scope, unaffected by ratio) | [0.041,0.081] (→SST-2) vs [0.040,0.084] (→AGNews) | 5-seed confirmed, H2-A/H2-B |
| T*(r8) SST-2 Phase 1 (block-scope) | [0.005,0.019]→CoLA / [0.036,0.073]→AGNews / [0.212,0.851]→MNLI — 44× range | H2-C/H2-F/sst2_to_cola |
| GPT-2 SST-2 Phase-1 baseline acc | 0.774 | h2_nll_forgetting.py Phase-1 output |
| GPT-2 fresh-probe recovery (sst2 Phase1 → mnli/agnews/cola, lr=5e-4) | recovers_slow, all 3; crossing steps 200/165/55 | RAW_DATA.md § Fresh-Probe Recovery — GPT-2 |
| GPT-2 severity sweep no_recovery LRs | 2e-3, 5e-3 — flagged likely-NaN, never confirmed | RAW_DATA.md § Severity Sweep, GPT-2 |
| Llama NaN diagnostic result (12 conditions) | all `phase2_hidden_nan`=False, all `probe_recovery_hidden_nan`=False | RAW_DATA.md § Llama Severity Sweep + NaN Diagnostic |
| Constant-class-collapse values | 0.478 (239/500), 0.522 (261/500) — 372 and 295 occurrences across 27 files | RAW_DATA.md § Llama Severity Sweep + NaN Diagnostic |
| MNLI generality test, Llama, final result | 0/3 targets crossed threshold; sst2 missed by 0.0024, agnews by 0.198 (bias ruled out), cola has 1 valid data point only | RAW_DATA.md § Fresh-Probe Recovery Generality Test |
| GPT-2 MNLI Phase-1 training | 3/3 attempts failed, stuck at ~33.3–35.4% (chance), loss flat at ln(3) | RAW_DATA.md § Fresh-Probe Recovery Generality Test |
| Llama MNLI Phase-1 baseline acc | 0.884 | h2_llama/mnli_to_sst2/, reused for generality test |
| Effective rank SST-2/AGNews/CoLA | 5.21 / 5.36 / 5.19 (112 layers) | local computation 2026-07-01 |
| Direction overlap (cosine sim, 4 pairs) | all within ±5.2e-4 (noise floor ≈4.7e-4) | direction_overlap_results.json |

---

**Standing rule for anyone updating this file**: every number here must be traceable to a specific file/field in RAW_DATA.md or SUMMARY.md. Before using any number to challenge, correct, or extend an existing claim — especially two similarly-named values for the same underlying quantity (e.g. two different "baseline NLL" figures, block-scope vs. lora-scope σ½, block-scope vs. lora-scope σ½_A_acc) — identify its exact source file and field and confirm it's the one actually feeding the comparison being made. General form of the rule, for baselines specifically: always pair a baseline NLL with the same eval pipeline that produced the post-fine-tuning NLL it's being subtracted from — a calibration pipeline's own baseline (e.g. `nll_baseline`) is never comparable to a fine-tuning table's baseline (e.g. `nll_wiki_base`), even for the same checkpoint, unless independently confirmed identical. This file was rewritten on 2026-08-02 specifically because that discipline was skipped once, produced an incorrect correction, and had to be traced and retracted.
