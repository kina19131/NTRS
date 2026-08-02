# NTRS Raw Data Extract

**Source**: Values extracted directly from individual JSON files listed under each section.  
`summary.json` used only to identify which files exist — no values taken from it.  
**Date extracted**: 2026-07-14

> **⚠️ STANDING WARNING (added 2026-07-17): every `R (in file)` / `sigma_half_pre` column in the H1
> fine-tuning tables below is potentially stale relative to that model's own relslack density file.**
> This has now been confirmed for GPT-2 (12× stale), Llama SST-2 (1.34× stale), and Llama MNLI
> (same 1.34×, same underlying checkpoint as SST-2) — three separate instances of the same bug.
> Mistral SST-2 v5 and Mistral MNLI (once corrected, see below) are the only two tables where the
> stored value is already correct. **Before citing any `R` or `T*` number from this document, recompute
> it as `per_param_norm / sigma_half` using that model's own `pretrained_density_relslack.json`
> (or `pretrained_density.json` if no separate relslack file exists) — do not trust the file's own
> cached `R`/`sigma_half_pre` column without checking.** Tables below now carry a corrected column
> where this has been verified; absence of a correction marker does not mean a table is clean, it
> may mean it hasn't been re-checked yet.

---

## H1: Pre-trained Model Certified Density

### GPT-2
**File**: `h1_kl/gpt2/pretrained_density_relslack.json`

```
sigma_half      = 0.007191011235955057
relative_slack  = 0.01
eval_slack_abs  = 0.042109375
nll_baseline    = 4.2109375

densities:
  0.0001 → 0.9817246596448638
  0.0002 → 0.9817246596448638
  0.0005 → 0.9817246596448638
  0.001  → 0.9817246596448638
  0.002  → 0.9817246596448638
  0.005  → 0.8382037844304998
  0.01   → 0.0
  0.02   → 0.0
  0.05   → 0.0
```

### Llama-3.2-3B
**File**: `h1_kl/meta-llama/meta-llama/Llama-3.2-3B/pretrained_density_relslack.json`

```
sigma_half      = 0.0007717391304347826
relative_slack  = 0.01
eval_slack_abs  = 0.0305859375
nll_baseline    = 3.05859375

densities:
  0.0001 → 0.9817246596448638
  0.0002 → 0.9817246596448638
  0.0005 → 0.9817246596448638
  0.001  → 0.046417608399822556
  0.002  → 0.0
  0.005  → 0.0
  0.01   → 0.0
  0.02   → 0.0
  0.05   → 0.0
```

### Mistral-7B-v0.1
**File**: `h1_kl/mistralai/Mistral-7B-v0.1_sst2_v5/pretrained_density.json`  
(This is the test-split relslack σ½. Verified 2026-07-17: `sigma_half_pre` inside each `lr*_results.json`
in this v5 directory already equals this same relslack value — no staleness here, unlike GPT-2/Llama's
lr*_results.json files, which do cache a stale non-relslack value. Do not confuse the two: check each
model's own lr*_results.json against its own pretrained_density*.json before trusting `R` from the file.)

```
sigma_half = 0.0001824175824175824

densities:
  0.0001 → 0.9757074028338915
  0.0002 → 0.314652630010597
  0.0005 → 0.0
  0.001  → 0.0
  0.002  → 0.0
  0.005  → 0.0
  0.01   → 0.0
  0.02   → 0.0
  0.05   → 0.0
```

---

## H1: Fine-tuning Run Data

Fields in each `lr*_results.json`: `lr`, `rank`, `steps`, `per_param_norm`, `sigma_half_pre` (non-relslack, stored by script), `R` (= per_param_norm / sigma_half_pre, **not** relslack-corrected), `acc`, `nll_wiki_base`, `nll_wiki_ft`, `delta_nll_wiki`, `kl_wiki`, `kl_diverse`, and (Llama/Mistral only) `kl_superni`, `mmlu_acc`.

### GPT-2 SST-2
**Files**: `h1_kl/gpt2/lr{1e-05,5e-05,1e-04,2e-04,5e-04}_results.json`

**⚠️ CORRECTED 2026-07-17**: `sigma_half_pre`/`R (in file)` below are stale (12× off vs. this model's own relslack σ½=7.191011235955057e-3). Corrected column added.

| lr | per_param_norm | sigma_half_pre (stale) | R (in file, stale) | **R (relslack, corrected)** | acc | nll_wiki_base | nll_wiki_ft | delta_nll_wiki | kl_diverse |
|----|---------------|---------------|------------|------------|-----|--------------|------------|---------------|-----------|
| 1e-5 | 1.0807497e-4 | 6.0256410e-4 | 0.17935846 | **0.01503** | 0.462 | 4.49453125 | 4.46734375 | −0.02718750 | 0.22794062 |
| 5e-5 | 6.1771790e-4 | 6.0256410e-4 | 1.02514885 | **0.08590** | 0.768 | 4.49453125 | 4.44093750 | −0.05359375 | 1.18603924 |
| 1e-4 | 1.5823082e-3 | 6.0256410e-4 | 2.62595830 | **0.22004** | 0.768 | 4.49453125 | 4.51015625 | 0.01562500 | 2.79117902 |
| 2e-4 | 3.2603993e-3 | 6.0256410e-4 | 5.41087538 | **0.45340** | 0.802 | 4.49453125 | 5.87984375 | 1.38531250 | 4.09821784 |
| 5e-4 | 1.2518968e-2 | 6.0256410e-4 | 20.7761592 | **1.74092** | 0.776 | 4.49453125 | 8.80437500 | 4.30984375 | 3.53294747 |

`rank=8, steps=3000` (same across all rows). No `kl_superni` or `mmlu_acc` fields in these files.

### Llama-3.2-3B SST-2
**Files**: `h1_kl/meta-llama/meta-llama/Llama-3.2-3B/lr{1e-05,5e-05,1e-04,2e-04,5e-04}_results.json`

**⚠️ CORRECTED 2026-07-17**: `sigma_half_pre`/`R (in file)` below are stale (1.34× off vs. this model's own relslack σ½=7.717391304347826e-4). Corrected column added.

| lr | per_param_norm | sigma_half_pre (stale) | R (in file, stale) | **R (relslack, corrected)** | acc | nll_wiki_base | nll_wiki_ft | delta_nll_wiki | kl_diverse | kl_superni | mmlu_acc |
|----|---------------|---------------|------------|------------|-----|--------------|------------|---------------|-----------|-----------|---------|
| 1e-5 | 2.1917352e-5 | 5.7627119e-4 | 0.03803305 | **0.02840** | 0.954 | 3.36375 | 3.36156250 | −0.00218750 | 7.11121662 | 0.36014132 | 0.555 |
| 5e-5 | 8.7465360e-5 | 5.7627119e-4 | 0.15177813 | **0.11334** | 0.960 | 3.36375 | 3.37296875 | 0.00921875 | 4.04567418 | 0.58677388 | 0.525 |
| 1e-4 | 1.8189143e-4 | 5.7627119e-4 | 0.31563513 | **0.23569** | 0.962 | 3.36375 | 3.39890625 | 0.03515625 | 8.56666873 | 2.86319734 | 0.540 |
| 2e-4 | 4.5188251e-4 | 5.7627119e-4 | 0.78414907 | **0.58554** | 0.958 | 3.36375 | 4.49710938 | 1.13335938 | 8.90206616 | 3.29702532 | 0.370 |
| 5e-4 | 2.8102279e-3 | 5.7627119e-4 | 4.87657187 | **3.64142** | 0.530 | 3.36375 | 12.1012500 | 8.73750000 | 10.6723661 | 10.3688395 | 0.215 |

`rank=8, steps=2000, task=sst2`

### Llama-3.2-3B MNLI
**Files**: `h1_kl/meta-llama/meta-llama/Llama-3.2-3B_mnli/lr{1e-05,5e-05,1e-04,2e-04,5e-04}_results.json`  
(No `pretrained_density_relslack.json` in this dir; σ½ comes from the SST-2 dir above — same pre-trained model.)

**⚠️ CORRECTED 2026-07-17**: same staleness as the SST-2 table above (same checkpoint, same σ½). Corrected column added. This is the table behind the "Mistral vs. Llama MNLI fragility" comparison — see the correction note in the Mistral MNLI section below; the corrected numbers here overturn that comparison, not just adjust it.

| lr | per_param_norm | sigma_half_pre (stale) | R (in file, stale) | **R (relslack, corrected)** | acc | nll_wiki_base | nll_wiki_ft | delta_nll_wiki | kl_diverse | kl_superni | mmlu_acc |
|----|---------------|---------------|------------|------------|-----|--------------|------------|---------------|-----------|-----------|---------|
| 1e-5 | 3.1179853e-5 | 5.7627119e-4 | 0.05410622 | **0.04040** | 0.838 | 3.36375 | 3.37031250 | 0.00656250 | 6.34102139 | 1.24413757 | 0.540 |
| 5e-5 | 1.0273622e-4 | 5.7627119e-4 | 0.17827756 | **0.13312** | 0.858 | 3.36375 | 3.39593750 | 0.03218750 | 6.62989231 | 1.38960345 | 0.520 |
| 1e-4 | 1.7971414e-4 | 5.7627119e-4 | 0.31185690 | **0.23287** | 0.846 | 3.36375 | 3.40265625 | 0.03890625 | 5.46283410 | 1.32267698 | 0.500 |
| 2e-4 | 4.0981081e-4 | 5.7627119e-4 | 0.71114228 | **0.53102** | 0.832 | 3.36375 | 3.42726563 | 0.06351563 | 10.5531291 | 3.89243674 | 0.490 |
| 5e-4 | 2.0574139e-3 | 5.7627119e-4 | 3.57021819 | **2.66594** | 0.822 | 3.36375 | 11.1084375 | 7.74468750 | 9.52615070 | 9.23812880 | 0.250 |

`rank=8, steps=2000, task=mnli`. **Corrected T\* (relslack, ΔNLL > 1% of baseline NLL rule): [0.133, 0.233]** (last safe at lr=5e-5, first broken at lr=1e-4) — falls **entirely inside** Mistral MNLI's corrected bracket [0.071, 0.324]. See the Mistral MNLI section below for the full correction.

### Mistral-7B-v0.1 SST-2 v5 (PRIMARY)
**Files**: `h1_kl/mistralai/Mistral-7B-v0.1_sst2_v5/lr{1e-05,5e-05,1e-04,2e-04,5e-04}_results.json`  
`sigma_half_pre` in these files = 1.8241758e-4 (test-split relslack; `R` in file is therefore correct for cross-model comparison).

| lr | per_param_norm | sigma_half_pre | R (in file) | acc | nll_wiki_base | nll_wiki_ft | delta_nll_wiki | kl_diverse | kl_superni | mmlu_acc |
|----|---------------|---------------|------------|-----|--------------|------------|---------------|-----------|-----------|---------|
| 1e-5 | 5.4687758e-6 | 1.8241758e-4 | 0.02997943 | 0.480 | 2.9934375 | 2.99453125 | 0.00109375 | 0.08820626 | 0.03093477 | 0.585 |
| 5e-5 | 3.5335289e-5 | 1.8241758e-4 | 0.19370550 | 0.930 | 2.9934375 | 3.11804688 | 0.12460938 | 1.41128229 | 0.58051736 | 0.505 |
| 1e-4 | 4.1152450e-5 | 1.8241758e-4 | 0.22559476 | 0.640 | 2.9934375 | 5.80296875 | 2.80953125 | 3.70603389 | 7.21805244 | 0.280 |
| 2e-4 | 1.1713497e-4 | 1.8241758e-4 | 0.64212543 | 0.530 | 2.9934375 | 17.2028125 | 14.2093750 | 5.29263269 | 8.44821822 | 0.245 |
| 5e-4 | 2.4076601e-4 | 1.8241758e-4 | 1.31986188 | 0.470 | 2.9934375 | 26.4400000 | 23.4465625 | 22.0338370 | 23.7093119 | 0.215 |

`rank=8, steps=500, task=sst2, batch_size=4, grad_accum=4 (eff_batch=16)`

### Mistral-7B-v0.1 MNLI v2
**Files**: `h1_kl/mistralai/Mistral-7B-v0.1_mnli/lr{1e-05,5e-05,1e-04,2e-04,5e-04}_results.json`  
`sigma_half_pre` here = 1.5586592e-4 (non-relslack — this is the value cached in these result files, and unlike GPT-2/Llama it is *not* stale relative to its own source, but it is the wrong convention for comparing against the primary SST-2 v5 numbers).

**⚠️ CORRECTED 2026-07-17**: `nll_wiki_base` in these files (2.9934375) is identical to the SST-2 v5 files' `nll_wiki_base` — confirming this is the same frozen pretrained checkpoint, so σ½_pre is not task-specific and the SST-2 v5 relslack value applies here too. Recomputed with the correct relslack σ½ = 1.824175824e-4 (from `Mistral-7B-v0.1_sst2_v5/pretrained_density.json`, reused — no new GPU computation needed). See `Mistral-7B-v0.1_mnli/pretrained_density_relslack.json` and `summary_relslack_corrected.json`.

| lr | per_param_norm | sigma_half_pre (non-relslack) | R (non-relslack, in file) | **R (relslack, corrected)** | acc | nll_wiki_base | nll_wiki_ft | delta_nll_wiki | kl_diverse | kl_superni | mmlu_acc |
|----|---------------|---------------|------------|------------|-----|--------------|------------|---------------|-----------|-----------|---------|
| 1e-5 | 1.3042044e-5 | 1.5586592e-4 | 0.08367476 | **0.07150** | 0.830 | 2.9934375 | 3.01023438 | 0.01679688 | 1.27023364 | 0.35294925 | 0.585 |
| 5e-5 | 5.9162402e-5 | 1.5586592e-4 | 0.37957240 | **0.32432** | 0.858 | 2.9934375 | 3.59648438 | 0.60304688 | 3.46560548 | 0.97129548 | 0.535 |
| 1e-4 | 1.3623933e-4 | 1.5586592e-4 | 0.87408027 | **0.74685** | 0.878 | 2.9934375 | 6.78656250 | 3.79312500 | 3.76152706 | 3.77121998 | 0.275 |
| 2e-4 | 1.5786692e-4 | 1.5586592e-4 | 1.01283796 | **0.86542** | 0.350 | 2.9934375 | 10.9943750 | 8.00093750 | 3.40995972 | 7.29530439 | 0.280 |
| 5e-4 | 6.9667102e-4 | 1.5586592e-4 | 4.46968146 | **3.81910** | 0.348 | 2.9934375 | 10.3062500 | 7.31281250 | 3.58289986 | 6.79305266 | 0.215 |

`rank=8, steps=2000, task=mnli`. **Corrected T\* bracket (relslack) = [0.071, 0.324]**, down from the previously reported [0.084, 0.380] — a uniform ~15% shift (≈1/1.17, the relslack/non-relslack σ½ ratio).

**⚠️ CORRECTION 2026-07-17 — this changes the finding, not just the number.** The original claim ("Mistral collapses far earlier than Llama MNLI, which stays safe through R=3.570") compared Mistral's corrected R against Llama's *stale, non-relslack* R (5.7627119e-4 instead of 7.717391e-4 — the same staleness documented for Llama's SST-2 table above). Recomputed on Llama's own relslack σ½, using the same "ΔNLL exceeds 1% of baseline NLL" break criterion for both models: **Llama MNLI T* = [0.133, 0.233]**, which sits **entirely inside** Mistral MNLI's [0.071, 0.324] — not outside it. The two models' MNLI thresholds are not distinguishable at this LR-grid resolution once both use the same σ½ convention. Do not cite "Mistral is ~9× more fragile at MNLI than Llama" — that comparison does not survive a consistent recompute.

**⚠️ FURTHER CORRECTION 2026-08-02 — "Llama MNLI T* = [0.133, 0.233]" above is itself a misclassification, independent of anything about σ½ scope.** Checking the stated criterion directly against the data used to produce it: at lr=5e-5, ΔNLL=0.03218750 against a threshold of 1%×3.05859375=0.03058594 — **ΔNLL exceeds the threshold by 5.2%**, meaning lr=5e-5 does not qualify as "safe" under the document's own stated rule. The correct classification (same criterion, same data) is last-safe=lr=1e-5, first-broken=lr=5e-5, giving **Llama MNLI T* = [0.0404, 0.1331]** (block-scope) — not [0.133, 0.233]. Recomputed against Mistral's T*=[0.0715,0.3243] (block-scope): **[0.0715, 0.1331] is a real overlap, but Llama's corrected bracket is *not* fully contained inside Mistral's** (its lower bound, 0.0404, sits below Mistral's lower bound, 0.0715) — the "fully contained" framing used everywhere this comparison is cited overstates what the corrected numbers show. The weaker, defensible claim — **the two models' MNLI T* brackets overlap and are not distinguishable at this LR-grid resolution** — still holds. This also holds under the separately-corrected LoRA-scope σ½_pre (see the σ½/LoRA-Scope Correction section): Llama=[0.0206,0.0677] vs. Mistral=[0.0372,0.1686], again overlapping ([0.0372,0.0677]) but not contained. Every downstream citation of "fully contained" for this pair (SUMMARY.md's V6 closure note, the H1 results section, "What Claims Can Be Made Now") should be read as "overlap confirmed, containment not established" going forward.

---

## H1: Direct-R Sweep (Checkpoint-Anchored Rescale) — GPT-2 (2026-07-20)

**Method**: `h1_direct_r_sweep.py`. One reference LoRA trajectory (`ref_lr=5e-4`, `ref_steps=3000`, `rank=8`, task=sst2, same as the GPT-2 H1 fine-tuning convention), snapshotting full (A, B, classifier) state at every `checkpoint_interval` steps. Each `target_R` anchors at whichever real checkpoint's own `per_param_norm` is closest, applying only a small local rescale `k = target_ppn / anchor_ppn` to B (A and the classifier restored exactly from that checkpoint, unscaled). `sigma_half_pre = 0.0071910112359550565` (relslack, reused from `h1_kl/gpt2/pretrained_density_relslack.json`).

Two runs, identical seed/config, differing only in `checkpoint_interval` (50 vs 10), to test whether the checkpoint-anchoring design's remaining disagreement with real LR-sweep training is a checkpoint-density/interpolation artifact.

### Run @ checkpoint_interval=50 (60 checkpoints over 3000 steps)
**Files**: `h1_direct_r/gpt2_v3/summary.json`, `reference_ppn_trajectory.json`

| target_R | R | anchor_step | k | acc | delta_nll_wiki | kl_wiki | mmlu_acc |
|---|---|---|---|---|---|---|---|
| 0.020 | 0.0200 | 20 | 0.746* | — | -0.0036 | — | — |
| 0.040 | 0.0400 | 150 | 0.978 | 0.5300 | -0.0205 | 0.0074 | 0.2450 |
| 0.060 | 0.0600 | 250 | 1.045 | 0.5300 | -0.0112 | 0.0124 | 0.2450 |
| 0.080 | 0.0800 | 300 | 0.981 | 0.5300 | +0.0286 | 0.1103 | 0.2450 |
| 0.100 | 0.1000 | 350 | 1.015 | 0.5300 | +0.0667 | 0.1416 | 0.2450 |
| 0.120 | 0.1200 | 450 | 0.971 | 0.5380 | +0.0820 | 0.1604 | 0.2450 |
| 0.150 | 0.1500 | 500 | 0.885 | 0.6700 | +0.0520 | 0.1180 | 0.2450 |
| 0.180 | 0.1800 | 500 | 1.063 | 0.6940 | +0.0958 | 0.1742 | 0.2450 |
| 0.220 | 0.2200 | 550 | 1.019 | 0.6740 | +0.1933 | 0.2430 | 0.2450 |
| 0.270 | 0.2700 | 650 | 0.971 | 0.7280 | +0.5622 | 0.6440 | 0.2450 |
| 0.330 | 0.3300 | 750 | 0.992 | 0.7360 | +0.9528 | 1.0333 | 0.2450 |
| 0.400 | 0.4000 | 900 | 1.014 | 0.7580 | +0.9588 | 1.0524 | 0.2450 |
| 0.500 | 0.5000 | 1150 | 0.989 | 0.7740 | +2.3411 | 2.4823 | 0.2400 |
| 0.650 | 0.6500 | 1350 | 1.004 | 0.7840 | +3.7250 | 3.8821 | 0.2750 |
| 0.850 | 0.8500 | 1600 | 0.974 | 0.7840 | +3.9586 | 4.1632 | 0.2950 |
| 1.100 | 1.1000 | 1900 | 1.015 | 0.7820 | +4.8658 | 5.0891 | 0.2300 |

`k` range across all 16 (non-extrapolated) points: **min=0.746, max=1.063**. *The 0.746 minimum is the target_R=0.020 row above — not independently re-verified against a fresh per-row printout (the original console capture for this run started at target_R=0.040), but corroborated two ways: it matches this script's own printed summary line exactly, and independently matches a hand-calculation from `reference_ppn_trajectory.json`'s real checkpoint data done separately during this investigation. Flagged 2026-07-21 after a reader traced the table and found 0.746 didn't appear in the visible rows — a real completeness gap in this table, now fixed, not an error in the underlying number.*

### Run @ checkpoint_interval=10 (300 checkpoints over 3000 steps)
**Files**: `h1_direct_r/gpt2/summary.json`, `reference_ppn_trajectory.json`

| target_R | R | anchor_step | k | acc | delta_nll_wiki | kl_wiki | mmlu_acc |
|---|---|---|---|---|---|---|---|
| 0.020 | 0.0200 | 20 | 0.998 | 0.5300 | -0.0016 | — | — |
| 0.040 | 0.0400 | 150 | 0.978 | 0.5300 | -0.0205 | — | — |
| 0.060 | 0.0600 | 260 | 0.995 | 0.5300 | -0.0142 | — | — |
| 0.080 | 0.0800 | 300 | 0.981 | 0.5300 | +0.0286 | — | — |
| 0.100 | 0.1000 | 360 | 1.002 | 0.5300 | +0.0791 | — | — |
| 0.120 | 0.1200 | 450 | 0.971 | 0.5380 | +0.0820 | — | — |
| 0.150 | 0.1500 | 470 | 1.012 | 0.5300 | +0.2758 | 0.3940 | 0.2450 |
| 0.180 | 0.1800 | 510 | 0.994 | 0.7400 | +0.1802 | 0.2743 | 0.2450 |
| 0.220 | 0.2200 | 560 | 0.995 | 0.6940 | +0.2255 | 0.2900 | 0.2450 |
| 0.270 | 0.2700 | 630 | 1.009 | 0.6400 | +0.3120 | 0.3953 | 0.2450 |
| 0.330 | 0.3300 | 740 | 1.003 | 0.7160 | +1.2919 | 1.3824 | 0.2450 |
| 0.400 | 0.4000 | 910 | 1.002 | 0.7380 | +0.8089 | 0.8921 | 0.2450 |
| 0.500 | 0.5000 | 1140 | 1.004 | 0.7760 | +2.5636 | 2.7149 | 0.2400 |
| 0.650 | 0.6500 | 1350 | 1.004 | 0.7840 | +3.7250 | 3.8821 | 0.2750 |
| 0.850 | 0.8500 | 1580 | 0.994 | 0.7700 | +3.9764 | 4.1638 | 0.2400 |
| 1.100 | 1.1000 | 1920 | 1.000 | 0.7600 | +4.8495 | 5.0796 | 0.2300 |

`k` range across all 16 (non-extrapolated) points: **min=0.971, max=1.012**.

### Crosscheck: real independently-trained LR-sweep points (`h1_kl/gpt2/lr*_results.json`)
R recomputed on the same relslack `sigma_half_pre` (never trusting each file's own cached `R` field):

| file | R | delta_nll_wiki |
|---|---|---|
| lr1e-05_results.json | 0.0150 | -0.0272 |
| lr5e-05_results.json | 0.0859 | -0.0536 |
| lr1e-04_results.json | 0.2200 | +0.0156 |
| lr2e-04_results.json | 0.4534 | +1.3853 |
| lr5e-04_results.json | 1.7409 | +4.3098 |

### Comparison at matched/interpolated R (rescaled-direction ΔNLL vs. real ΔNLL at the same R)

| crosscheck R | real delta_nll | interval=50 rescaled (interp) | ratio | interval=10 rescaled (interp) | ratio |
|---|---|---|---|---|---|
| 0.0150 | -0.0272 | -0.0036 | 0.13x | -0.0016 | 0.06x |
| 0.0859 | -0.0536 | +0.0398 | -0.74x | +0.0435 | -0.81x |
| **0.2200** | **+0.0156** | **+0.1936** | **12.39x** | **+0.2255** | **14.43x** |
| 0.4534 | +1.3853 | +1.6969 | 1.22x | +1.7459 | 1.26x |
| 1.7409 (beyond both curves' max R=1.10; extrapolated as flat) | +4.3098 | +4.8658 | 1.13x | +4.8495 | 1.13x |

At R=0.0150 and R=0.4534, "interval=50/10" values are read directly off the table above (no interpolation needed — R=0.2200 in both runs' own target grid is an exact match to the crosscheck point). R=0.0859 and R=1.7409 are linearly interpolated/extrapolated between adjacent rescaled-direction points.

---

## H2: Phase 1 Measurements

### Llama-3.2-3B Phase 1

Each Phase 1 model (Task A) has two σ½ files. `final_R_A` in result files is computed as `final_per_param / sigma_half_nll`.

#### phase1_meta.json (same values shared across all Task B directions for the same Task A)

| Task A | acc_phase1 | lr | rank | steps | source dir (any pair with this Task A) |
|--------|-----------|-----|------|-------|---------------------------------------|
| mnli | 0.884 | 5e-5 | 8 | 2000 | h2_llama/mnli_to_sst2/phase1_meta.json |
| sst2 | 0.960 | 5e-5 | 8 | 2000 | h2_llama/sst2_to_mnli/phase1_meta.json |
| agnews | 0.910 | 5e-5 | 8 | 2000 | h2_llama/agnews_to_mnli/phase1_meta.json |
| cola | 0.868 | 5e-5 | 8 | 2000 | h2_llama/cola_to_mnli/phase1_meta.json |

#### phase1_sigma_half.json (NLL-based)

| Task A | sigma_half_nll | source file |
|--------|---------------|------------|
| mnli | 6.607142857e-3 | h2_llama/mnli_to_sst2/phase1_sigma_half.json |
| sst2 | 1.191919192e-2 | h2_llama/sst2_to_mnli/phase1_sigma_half.json |
| agnews | 5.291666667e-3 | h2_llama/agnews_to_mnli/phase1_sigma_half.json |
| cola | 7.500000000e-3 | h2_llama/cola_to_mnli/phase1_sigma_half.json |

#### phase1_sigma_half_acc.json (acc-based, preferred)

Not present for: mnli_to_agnews, cola_to_agnews.

| Task A | sigma_half_acc | baseline_acc | success_threshold | source file |
|--------|---------------|-------------|------------------|------------|
| mnli | 2.857142857e-3 | 0.870 | 0.783 | h2_llama/mnli_to_sst2/phase1_sigma_half_acc.json |
| sst2 | 3.337373737e-3 | 0.975 | 0.8775 | h2_llama/sst2_to_mnli/phase1_sigma_half_acc.json |
| agnews | 2.755586420e-3 | 0.905 | 0.8145 | h2_llama/agnews_to_mnli/phase1_sigma_half_acc.json |
| cola | 1.943801653e-3 | 0.875 | 0.7875 | h2_llama/cola_to_mnli/phase1_sigma_half_acc.json |

### Mistral-7B-v0.1 Phase 1

**File**: `h2_mistral/agnews_to_mnli/phase1_sigma_half.json` (NLL-based only; no acc sigma file)

```
task_a        = agnews   (from phase1_meta.json)
acc_phase1    = 0.91     (from phase1_meta.json)
sigma_half    = 7.619047619e-4
```

acc_threshold = 0.819 (from `tstar_checkpoint.acc_threshold` field in result files).

---

## H2: Llama-3.2-3B Sequential Fine-tuning Results

Fields from each `lr*_rank*_result.json`: `lr`, `rank`, `steps`, `sigma_half_a` (NLL-based, stored by script), `final_R_A` (= final_per_param / sigma_half_a), `final_per_param`, `final_acc_task_a`, `final_acc_task_b`.

### mnli_to_sst2
**Dir**: `h2_llama/mnli_to_sst2/`  
sigma_half_a stored in files = 6.607143e-3 (NLL). sigma_half_acc = 2.857143e-3 (acc). success_threshold = 0.783.  
Note: seed variant files exist for lr=1e-4,2e-4,5e-4 at rank=8 (`lr*_rank8_seed{1..5}_result.json`) — not listed here.

| file | lr | rank | final_per_param | final_R_A (nll) | final_acc_task_a | final_acc_task_b |
|------|----|------|----------------|----------------|-----------------|-----------------|
| lr1e-05_rank4_result.json | 1e-5 | 4 | 1.251629e-5 | 0.001894 | 0.886 | 0.930 |
| lr1e-05_rank8_result.json | 1e-5 | 8 | 1.517863e-5 | 0.002297 | 0.884 | 0.932 |
| lr1e-05_rank16_result.json | 1e-5 | 16 | 2.020234e-5 | 0.003058 | 0.876 | 0.940 |
| lr1e-05_rank32_result.json | 1e-5 | 32 | 2.581687e-5 | 0.003907 | 0.880 | 0.944 |
| lr5e-05_rank4_result.json | 5e-5 | 4 | 4.427701e-5 | 0.006701 | 0.878 | 0.946 |
| lr5e-05_rank8_result.json | 5e-5 | 8 | 5.917179e-5 | 0.008956 | 0.866 | 0.956 |
| lr5e-05_rank16_result.json | 5e-5 | 16 | 8.347011e-5 | 0.012633 | 0.850 | 0.954 |
| lr5e-05_rank32_result.json | 5e-5 | 32 | 1.174683e-4 | 0.017779 | 0.822 | 0.966 |
| lr1e-04_rank4_result.json | 1e-4 | 4 | 8.520237e-5 | 0.012895 | 0.866 | 0.956 |
| lr1e-04_rank8_result.json | 1e-4 | 8 | 1.154389e-4 | 0.017472 | 0.848 | 0.964 |
| lr1e-04_rank16_result.json | 1e-4 | 16 | 1.636304e-4 | 0.024766 | 0.748 | 0.964 |
| lr1e-04_rank32_result.json | 1e-4 | 32 | 2.438888e-4 | 0.036913 | 0.730 | 0.962 |
| lr2e-04_rank4_result.json | 2e-4 | 4 | 1.734881e-4 | 0.026258 | 0.796 | 0.974 |
| lr2e-04_rank8_result.json | 2e-4 | 8 | 2.352103e-4 | 0.035599 | 0.840 | 0.932 |
| lr2e-04_rank16_result.json | 2e-4 | 16 | 3.361534e-4 | 0.050877 | 0.544 | 0.960 |
| lr2e-04_rank32_result.json | 2e-4 | 32 | 5.188643e-4 | 0.078531 | 0.486 | 0.964 |
| lr5e-04_rank4_result.json | 5e-4 | 4 | 5.187050e-4 | 0.078507 | 0.338 | 0.942 |
| lr5e-04_rank8_result.json | 5e-4 | 8 | 7.145670e-4 | 0.108151 | 0.352 | 0.962 |
| lr5e-04_rank16_result.json | 5e-4 | 16 | 1.393065e-3 | 0.210842 | 0.332 | 0.892 |
| lr5e-04_rank32_result.json | 5e-4 | 32 | 4.084875e-3 | 0.618251 | 0.332 | 0.780 |

### mnli_to_agnews
**Dir**: `h2_llama/mnli_to_agnews/`  
sigma_half_a stored in files = 6.607143e-3 (NLL). No acc sigma file.

| file | lr | rank | final_per_param | final_R_A (nll) | final_acc_task_a | final_acc_task_b |
|------|----|------|----------------|----------------|-----------------|-----------------|
| lr1e-05_rank4_result.json | 1e-5 | 4 | 1.429293e-5 | 0.002163 | 0.870 | 0.876 |
| lr1e-05_rank8_result.json | 1e-5 | 8 | 1.741379e-5 | 0.002636 | 0.874 | 0.894 |
| lr1e-05_rank16_result.json | 1e-5 | 16 | 2.205744e-5 | 0.003338 | 0.880 | 0.892 |
| lr1e-05_rank32_result.json | 1e-5 | 32 | 2.748010e-5 | 0.004159 | 0.856 | 0.904 |
| lr5e-05_rank4_result.json | 5e-5 | 4 | 4.509767e-5 | 0.006826 | 0.854 | 0.892 |
| lr5e-05_rank8_result.json | 5e-5 | 8 | 6.043575e-5 | 0.009147 | 0.840 | 0.910 |
| lr5e-05_rank16_result.json | 5e-5 | 16 | 8.372758e-5 | 0.012672 | 0.856 | 0.900 |
| lr5e-05_rank32_result.json | 5e-5 | 32 | 1.188883e-4 | 0.017994 | 0.818 | 0.906 |
| lr1e-04_rank4_result.json | 1e-4 | 4 | 8.448702e-5 | 0.012787 | 0.836 | 0.900 |
| lr1e-04_rank8_result.json | 1e-4 | 8 | 1.154597e-4 | 0.017475 | 0.864 | 0.896 |
| lr1e-04_rank16_result.json | 1e-4 | 16 | 1.665193e-4 | 0.025203 | 0.868 | 0.916 |
| lr1e-04_rank32_result.json | 1e-4 | 32 | 2.386288e-4 | 0.036117 | 0.720 | 0.914 |
| lr2e-04_rank4_result.json | 2e-4 | 4 | 1.662022e-4 | 0.025155 | 0.842 | 0.918 |
| lr2e-04_rank8_result.json | 2e-4 | 8 | 2.392588e-4 | 0.036212 | 0.764 | 0.910 |
| lr2e-04_rank16_result.json | 2e-4 | 16 | 3.526297e-4 | 0.053371 | 0.708 | 0.918 |
| lr2e-04_rank32_result.json | 2e-4 | 32 | 5.106657e-4 | 0.077290 | 0.340 | 0.906 |
| lr5e-04_rank4_result.json | 5e-4 | 4 | 4.948411e-4 | 0.074895 | 0.352 | 0.914 |
| lr5e-04_rank8_result.json | 5e-4 | 8 | 7.444942e-4 | 0.112680 | 0.356 | 0.912 |
| lr5e-04_rank16_result.json | 5e-4 | 16 | 1.122039e-3 | 0.169822 | 0.370 | 0.882 |
| lr5e-04_rank32_result.json | 5e-4 | 32 | 1.802953e-3 | 0.272879 | 0.348 | 0.900 |

### sst2_to_mnli
**Dir**: `h2_llama/sst2_to_mnli/`  
sigma_half_a stored in files = 1.191919e-2 (NLL). sigma_half_acc = 3.337374e-3 (acc). success_threshold = 0.8775.

| file | lr | rank | final_per_param | final_R_A (nll) | final_acc_task_a | final_acc_task_b |
|------|----|------|----------------|----------------|-----------------|-----------------|
| lr1e-05_rank8_result.json | 1e-5 | 8 | 1.609434e-5 | 0.001350 | 0.960 | 0.796 |
| lr5e-05_rank8_result.json | 5e-5 | 8 | 6.155287e-5 | 0.005164 | 0.934 | 0.852 |
| lr1e-04_rank8_result.json | 1e-4 | 8 | 1.196109e-4 | 0.010035 | 0.944 | 0.820 |
| lr2e-04_rank8_result.json | 2e-4 | 8 | 2.430508e-4 | 0.020392 | 0.940 | 0.836 |
| lr5e-04_rank8_result.json | 5e-4 | 8 | 7.086152e-4 | 0.059452 | 0.900 | 0.852 |
| lr1e-03_rank8_result.json | 1e-3 | 8 | 2.840747e-3 | 0.238334 | 0.478 | 0.356 |
| lr2e-03_rank8_result.json | 2e-3 | 8 | 5.498288e-3 | 0.461297 | 0.478 | 0.332 |
| lr5e-03_rank8_result.json | 5e-3 | 8 | 9.892476e-3 | 0.829962 | 0.478 | 0.352 |
| lr1e-05_rank32_result.json | 1e-5 | 32 | 2.728629e-5 | 0.002289 | 0.960 | 0.838 |
| lr5e-05_rank32_result.json | 5e-5 | 32 | 1.188157e-4 | 0.009968 | 0.944 | 0.876 |
| lr1e-04_rank32_result.json | 1e-4 | 32 | 2.348921e-4 | 0.019707 | 0.786 | 0.872 |
| lr2e-04_rank32_result.json | 2e-4 | 32 | 4.975165e-4 | 0.041741 | 0.826 | 0.838 |
| lr5e-04_rank32_result.json | 5e-4 | 32 | 1.586424e-3 | 0.133098 | 0.522 | 0.444 |
| lr1e-03_rank32_result.json | 1e-3 | 32 | 4.923815e-3 | 0.413100 | 0.522 | 0.332 |
| lr2e-03_rank32_result.json | 2e-3 | 32 | 8.407773e-3 | 0.705398 | 0.478 | 0.352 |
| lr5e-03_rank32_result.json | 5e-3 | 32 | 1.161276e-2 | 0.974291 | 0.478 | 0.352 |

### sst2_to_agnews
**Dir**: `h2_llama/sst2_to_agnews/`  
sigma_half_a stored in files = 1.191919e-2 (NLL). sigma_half_acc = 3.337374e-3 (acc). success_threshold = 0.8775.

| file | lr | rank | final_per_param | final_R_A (nll) | final_acc_task_a | final_acc_task_b |
|------|----|------|----------------|----------------|-----------------|-----------------|
| lr1e-05_rank4_result.json | 1e-5 | 4 | 1.404572e-5 | 0.001178 | 0.930 | 0.876 |
| lr1e-05_rank8_result.json | 1e-5 | 8 | 1.714183e-5 | 0.001438 | 0.898 | 0.878 |
| lr1e-05_rank16_result.json | 1e-5 | 16 | 2.147556e-5 | 0.001802 | 0.902 | 0.878 |
| lr1e-05_rank32_result.json | 1e-5 | 32 | 2.747649e-5 | 0.002305 | 0.944 | 0.884 |
| lr5e-05_rank4_result.json | 5e-5 | 4 | 4.542812e-5 | 0.003811 | 0.930 | 0.884 |
| lr5e-05_rank8_result.json | 5e-5 | 8 | 6.125312e-5 | 0.005139 | 0.948 | 0.894 |
| lr5e-05_rank16_result.json | 5e-5 | 16 | 8.397591e-5 | 0.007045 | 0.940 | 0.886 |
| lr5e-05_rank32_result.json | 5e-5 | 32 | 1.159871e-4 | 0.009731 | 0.948 | 0.916 |
| lr1e-04_rank4_result.json | 1e-4 | 4 | 8.385082e-5 | 0.007035 | 0.950 | 0.888 |
| lr1e-04_rank8_result.json | 1e-4 | 8 | 1.192743e-4 | 0.010007 | 0.950 | 0.890 |
| lr1e-04_rank16_result.json | 1e-4 | 16 | 1.665786e-4 | 0.013976 | 0.924 | 0.900 |
| lr1e-04_rank32_result.json | 1e-4 | 32 | 2.383912e-4 | 0.020001 | 0.820 | 0.916 |
| lr2e-04_rank4_result.json | 2e-4 | 4 | 1.699167e-4 | 0.014256 | 0.962 | 0.906 |
| lr2e-04_rank8_result.json | 2e-4 | 8 | 2.434695e-4 | 0.020427 | 0.864 | 0.906 |
| lr2e-04_rank16_result.json | 2e-4 | 16 | 3.523455e-4 | 0.029561 | 0.822 | 0.920 |
| lr2e-04_rank32_result.json | 2e-4 | 32 | 5.205483e-4 | 0.043673 | 0.528 | 0.912 |
| lr5e-04_rank4_result.json | 5e-4 | 4 | 4.935836e-4 | 0.041411 | 0.744 | 0.918 |
| lr5e-04_rank8_result.json | 5e-4 | 8 | 7.536906e-4 | 0.063233 | 0.648 | 0.890 |
| lr5e-04_rank16_result.json | 5e-4 | 16 | 1.114165e-3 | 0.093477 | 0.564 | 0.898 |
| lr5e-04_rank32_result.json | 5e-4 | 32 | 1.738828e-3 | 0.145885 | 0.480 | 0.902 |

### agnews_to_mnli
**Dir**: `h2_llama/agnews_to_mnli/`  
sigma_half_a stored in files = 5.291667e-3 (NLL). sigma_half_acc = 2.755586e-3 (acc). success_threshold = 0.8145.

| file | lr | rank | final_per_param | final_R_A (nll) | final_acc_task_a | final_acc_task_b |
|------|----|------|----------------|----------------|-----------------|-----------------|
| lr1e-05_rank4_result.json | 1e-5 | 4 | 9.632249e-6 | 0.001820 | 0.908 | 0.508 |
| lr1e-05_rank8_result.json | 1e-5 | 8 | 1.547259e-5 | 0.002924 | 0.912 | 0.728 |
| lr1e-05_rank16_result.json | 1e-5 | 16 | 2.079930e-5 | 0.003931 | 0.908 | 0.814 |
| lr1e-05_rank32_result.json | 1e-5 | 32 | 2.677182e-5 | 0.005059 | 0.892 | 0.830 |
| lr5e-05_rank4_result.json | 5e-5 | 4 | 4.501178e-5 | 0.008506 | 0.866 | 0.814 |
| lr5e-05_rank8_result.json | 5e-5 | 8 | 6.045060e-5 | 0.011424 | 0.906 | 0.828 |
| lr5e-05_rank16_result.json | 5e-5 | 16 | 8.330393e-5 | 0.015742 | 0.894 | 0.862 |
| lr5e-05_rank32_result.json | 5e-5 | 32 | 1.147950e-4 | 0.021694 | 0.908 | 0.898 |
| lr1e-04_rank4_result.json | 1e-4 | 4 | 8.245345e-5 | 0.015582 | 0.920 | 0.846 |
| lr1e-04_rank8_result.json | 1e-4 | 8 | 1.161159e-4 | 0.021943 | 0.878 | 0.842 |
| lr1e-04_rank16_result.json | 1e-4 | 16 | 1.601835e-4 | 0.030271 | 0.780 | 0.870 |
| lr1e-04_rank32_result.json | 1e-4 | 32 | 2.284308e-4 | 0.043168 | 0.650 | 0.868 |
| lr2e-04_rank4_result.json | 2e-4 | 4 | 1.670768e-4 | 0.031574 | 0.786 | 0.830 |
| lr2e-04_rank8_result.json | 2e-4 | 8 | 2.355676e-4 | 0.044517 | 0.574 | 0.890 |
| lr2e-04_rank16_result.json | 2e-4 | 16 | 3.411930e-4 | 0.064477 | 0.268 | 0.856 |
| lr2e-04_rank32_result.json | 2e-4 | 32 | 4.897836e-4 | 0.092558 | 0.236 | 0.828 |
| lr5e-04_rank4_result.json | 5e-4 | 4 | 4.773497e-4 | 0.090208 | 0.254 | 0.866 |
| lr5e-04_rank8_result.json | 5e-4 | 8 | 6.939753e-4 | 0.131145 | 0.248 | 0.878 |
| lr5e-04_rank16_result.json | 5e-4 | 16 | 1.086098e-3 | 0.205247 | 0.280 | 0.350 |
| lr5e-04_rank32_result.json | 5e-4 | 32 | 7.163223e-3 | 1.353680 | 0.250 | 0.332 |

### agnews_to_sst2
**Dir**: `h2_llama/agnews_to_sst2/`  
sigma_half_a stored in files = 5.291667e-3 (NLL). sigma_half_acc = 2.755586e-3 (acc). success_threshold = 0.8145.

| file | lr | rank | final_per_param | final_R_A (nll) | final_acc_task_a | final_acc_task_b |
|------|----|------|----------------|----------------|-----------------|-----------------|
| lr1e-05_rank4_result.json | 1e-5 | 4 | 1.257929e-5 | 0.002377 | 0.912 | 0.918 |
| lr1e-05_rank8_result.json | 1e-5 | 8 | 1.568715e-5 | 0.002965 | 0.908 | 0.936 |
| lr1e-05_rank16_result.json | 1e-5 | 16 | 2.046277e-5 | 0.003867 | 0.918 | 0.942 |
| lr1e-05_rank32_result.json | 1e-5 | 32 | 2.626056e-5 | 0.004963 | 0.894 | 0.954 |
| lr5e-05_rank4_result.json | 5e-5 | 4 | 4.427110e-5 | 0.008366 | 0.904 | 0.948 |
| lr5e-05_rank8_result.json | 5e-5 | 8 | 6.074954e-5 | 0.011480 | 0.876 | 0.958 |
| lr5e-05_rank16_result.json | 5e-5 | 16 | 8.252939e-5 | 0.015596 | 0.890 | 0.958 |
| lr5e-05_rank32_result.json | 5e-5 | 32 | 1.173411e-4 | 0.022175 | 0.824 | 0.966 |
| lr1e-04_rank4_result.json | 1e-4 | 4 | 8.442598e-5 | 0.015955 | 0.906 | 0.960 |
| lr1e-04_rank8_result.json | 1e-4 | 8 | 1.155486e-4 | 0.021836 | 0.864 | 0.954 |
| lr1e-04_rank16_result.json | 1e-4 | 16 | 1.660774e-4 | 0.031385 | 0.844 | 0.964 |
| lr1e-04_rank32_result.json | 1e-4 | 32 | 2.316234e-4 | 0.043771 | 0.716 | 0.958 |
| lr2e-04_rank4_result.json | 2e-4 | 4 | 1.695374e-4 | 0.032039 | 0.540 | 0.966 |
| lr2e-04_rank8_result.json | 2e-4 | 8 | 2.443642e-4 | 0.046179 | 0.664 | 0.914 |
| lr2e-04_rank16_result.json | 2e-4 | 16 | 3.422473e-4 | 0.064677 | 0.376 | 0.920 |
| lr2e-04_rank32_result.json | 2e-4 | 32 | 5.114494e-4 | 0.096652 | 0.352 | 0.946 |
| lr5e-04_rank4_result.json | 5e-4 | 4 | 5.141740e-4 | 0.097167 | 0.258 | 0.950 |
| lr5e-04_rank8_result.json | 5e-4 | 8 | 7.428211e-4 | 0.140376 | 0.276 | 0.944 |
| lr5e-04_rank16_result.json | 5e-4 | 16 | 2.785207e-3 | 0.526338 | 0.268 | 0.776 |
| lr5e-04_rank32_result.json | 5e-4 | 32 | 2.247485e-3 | 0.424722 | 0.250 | 0.686 |

### cola_to_mnli
**Dir**: `h2_llama/cola_to_mnli/`  
sigma_half_a stored in files = 7.500000e-3 (NLL). sigma_half_acc = 1.943802e-3 (acc). success_threshold = 0.7875.

| file | lr | rank | final_per_param | final_R_A (nll) | final_acc_task_a | final_acc_task_b |
|------|----|------|----------------|----------------|-----------------|-----------------|
| lr1e-05_rank8_result.json | 1e-5 | 8 | 1.450331e-5 | 0.001934 | 0.874 | 0.574 |
| lr5e-05_rank8_result.json | 5e-5 | 8 | 5.961991e-5 | 0.007949 | 0.858 | 0.800 |
| lr1e-04_rank8_result.json | 1e-4 | 8 | 1.156047e-4 | 0.015414 | 0.828 | 0.804 |
| lr2e-04_rank8_result.json | 2e-4 | 8 | 2.299614e-4 | 0.030662 | 0.706 | 0.852 |
| lr5e-04_rank8_result.json | 5e-4 | 8 | 6.910228e-4 | 0.092136 | 0.318 | 0.872 |
| lr1e-05_rank32_result.json | 1e-5 | 32 | 2.692708e-5 | 0.003590 | 0.840 | 0.810 |
| lr5e-05_rank32_result.json | 5e-5 | 32 | 1.175968e-4 | 0.015680 | 0.836 | 0.878 |
| lr1e-04_rank32_result.json | 1e-4 | 32 | 2.303349e-4 | 0.030711 | 0.810 | 0.880 |
| lr2e-04_rank32_result.json | 2e-4 | 32 | 5.045352e-4 | 0.067271 | 0.682 | 0.844 |
| lr5e-04_rank32_result.json | 5e-4 | 32 | 5.951316e-3 | 0.793509 | 0.318 | 0.316 |

### cola_to_agnews
**Dir**: `h2_llama/cola_to_agnews/`  
sigma_half_a stored in files = 7.500000e-3 (NLL). No acc sigma file.

| file | lr | rank | final_per_param | final_R_A (nll) | final_acc_task_a | final_acc_task_b |
|------|----|------|----------------|----------------|-----------------|-----------------|
| lr1e-05_rank8_result.json | 1e-5 | 8 | 1.787115e-5 | 0.002383 | 0.856 | 0.874 |
| lr5e-05_rank8_result.json | 5e-5 | 8 | 6.183086e-5 | 0.008244 | 0.824 | 0.890 |
| lr1e-04_rank8_result.json | 1e-4 | 8 | 1.165038e-4 | 0.015534 | 0.814 | 0.896 |
| lr2e-04_rank8_result.json | 2e-4 | 8 | 2.443927e-4 | 0.032586 | 0.622 | 0.912 |
| lr5e-04_rank8_result.json | 5e-4 | 8 | 7.352499e-4 | 0.098033 | 0.664 | 0.900 |
| lr1e-05_rank32_result.json | 1e-5 | 32 | 2.749525e-5 | 0.003666 | 0.854 | 0.888 |
| lr5e-05_rank32_result.json | 5e-5 | 32 | 1.169788e-4 | 0.015597 | 0.788 | 0.900 |
| lr1e-04_rank32_result.json | 1e-4 | 32 | 2.311404e-4 | 0.030819 | 0.746 | 0.906 |
| lr2e-04_rank32_result.json | 2e-4 | 32 | 5.138031e-4 | 0.068507 | 0.680 | 0.914 |
| lr5e-04_rank32_result.json | 5e-4 | 32 | 1.814503e-3 | 0.241934 | 0.322 | 0.872 |

### sst2_to_cola
**Dir**: `h2_llama/sst2_to_cola/`  
sigma_half_a stored in files = 1.191919e-2 (NLL). sigma_half_acc = 3.337374e-3 (acc). success_threshold = 0.8775.

| file | lr | rank | final_per_param | final_R_A (nll) | final_acc_task_a | final_acc_task_b |
|------|----|------|----------------|----------------|-----------------|-----------------|
| lr1e-05_rank8_result.json | 1e-5 | 8 | 1.519922e-5 | 1.2751885e-3 | 0.888 | 0.762 |
| lr5e-05_rank8_result.json | 5e-5 | 8 | 6.247271e-5 | 5.2413548e-3 | 0.640 | 0.884 |
| lr1e-04_rank8_result.json | 1e-4 | 8 | 1.209808e-4 | 1.0150086e-2 | 0.664 | 0.882 |
| lr2e-04_rank8_result.json | 2e-4 | 8 | 2.460894e-4 | 2.0646484e-2 | 0.582 | 0.876 |
| lr5e-04_rank8_result.json | 5e-4 | 8 | 7.856766e-4 | 6.5916937e-2 | 0.478 | 0.858 |
| lr1e-05_rank32_result.json | 1e-5 | 32 | 2.748507e-5 | 2.3059510e-3 | 0.744 | 0.858 |
| lr5e-05_rank32_result.json | 5e-5 | 32 | 1.195319e-4 | 1.0028521e-2 | 0.534 | 0.888 |
| lr1e-04_rank32_result.json | 1e-4 | 32 | 2.466938e-4 | 2.0697188e-2 | 0.528 | 0.864 |
| lr2e-04_rank32_result.json | 2e-4 | 32 | 5.808182e-4 | 4.8729660e-2 | 0.516 | 0.876 |
| lr5e-04_rank32_result.json | 5e-4 | 32 | 7.230939e-3 | 6.0666351e-1 | 0.478 | 0.512 |

---

## H2: Llama-3.2-3B 5-Seed Replication (MNLI→SST-2, rank=8)

**Dir**: `h2_llama/mnli_to_sst2/`  
**Files**: `lr{1e-04,2e-04}_rank8_seed{1..5}_result.json`  
sigma_half_a = 6.607143e-3 (NLL). sigma_half_acc = 2.857143e-3 (acc). success_threshold = 0.783.  
`seed` field = data-loading seed (1–5). All other hyperparams identical to main sweep.

| file | lr | seed | final_per_param | final_R_A (nll) | final_acc_task_a | final_acc_task_b |
|------|----|------|----------------|----------------|-----------------|-----------------|
| lr1e-04_rank8_seed1_result.json | 1e-4 | 1 | 1.157692e-4 | 1.7521830e-2 | 0.846 | 0.968 |
| lr1e-04_rank8_seed2_result.json | 1e-4 | 2 | 1.173225e-4 | 1.7756914e-2 | 0.832 | 0.956 |
| lr1e-04_rank8_seed3_result.json | 1e-4 | 3 | 1.146419e-4 | 1.7351212e-2 | 0.856 | 0.952 |
| lr1e-04_rank8_seed4_result.json | 1e-4 | 4 | 1.160763e-4 | 1.7568299e-2 | 0.858 | 0.962 |
| lr1e-04_rank8_seed5_result.json | 1e-4 | 5 | 1.175700e-4 | 1.7794381e-2 | 0.860 | 0.954 |
| lr2e-04_rank8_seed1_result.json | 2e-4 | 1 | 2.404659e-4 | 3.6394833e-2 | 0.828 | 0.974 |
| lr2e-04_rank8_seed2_result.json | 2e-4 | 2 | 2.283362e-4 | 3.4558986e-2 | 0.816 | 0.972 |
| lr2e-04_rank8_seed3_result.json | 2e-4 | 3 | 2.441565e-4 | 3.6953418e-2 | 0.638 | 0.954 |
| lr2e-04_rank8_seed4_result.json | 2e-4 | 4 | 2.404502e-4 | 3.6392468e-2 | 0.786 | 0.966 |
| lr2e-04_rank8_seed5_result.json | 2e-4 | 5 | 2.321468e-4 | 3.5135732e-2 | 0.732 | 0.952 |

Outcome at threshold 0.783: lr=1e-4 → 0/5 forget (all acc_a ≥ 0.832). lr=2e-4 → 3/5 forget (seeds 3,4,5: acc_a = 0.638, 0.786, 0.732 < 0.783). Seeds 1,2 safe (0.828, 0.816 > 0.783).

---

## H2: Mistral-7B-v0.1 Sequential Fine-tuning Results

**Dir**: `h2_mistral/`  
`final_R_A` in files = `final_per_param / sigma_half_a` where sigma_half_a = 7.619048e-4 (NLL).  
`tstar_checkpoint` is embedded in result files where forgetting occurred; absent when model stayed safe throughout.  
`tstar_R_A` = R_A at last safe checkpoint. `forget_R_A` = R_A at first forget step.  
NaN for per_param/final_R_A = training diverged.

### agnews_to_mnli (Mistral)

| file | lr | rank | final_per_param | final_R_A (nll) | final_acc_task_a | final_acc_task_b | tstar_R_A | forget_R_A |
|------|----|------|----------------|----------------|-----------------|-----------------|---------|---------|
| lr1e-05_rank8_result.json | 1e-5 | 8 | 1.485568e-5 | 0.019498 | 0.884 | 0.894 | — | — |
| lr5e-05_rank8_result.json | 5e-5 | 8 | 6.231956e-5 | 0.081794 | 0.786 | 0.880 | 0.070847 | 0.072115 |
| lr1e-04_rank8_result.json | 1e-4 | 8 | 1.365726e-4 | 0.179252 | 0.666 | 0.846 | 0.094215 | 0.097983 |
| lr2e-04_rank8_result.json | 2e-4 | 8 | 2.158409e-4 | 0.283291 | 0.244 | 0.316 | 0.042976 | 0.056405 |
| lr5e-04_rank8_result.json | 5e-4 | 8 | 4.478925e-4 | 0.587859 | 0.244 | 0.332 | 0.000000 | 0.123397 |
| lr1e-03_rank8_result.json | 1e-3 | 8 | NaN | NaN | 0.240 | 0.352 | 0.000000 | 0.265358 |

### agnews_to_sst2 (Mistral)

| file | lr | rank | final_per_param | final_R_A (nll) | final_acc_task_a | final_acc_task_b | tstar_R_A | forget_R_A |
|------|----|------|----------------|----------------|-----------------|-----------------|---------|---------|
| lr1e-05_rank8_result.json | 1e-5 | 8 | 1.377197e-5 | 0.018076 | 0.902 | 0.960 | — | — |
| lr5e-05_rank8_result.json | 5e-5 | 8 | 4.992105e-5 | 0.065521 | 0.882 | 0.956 | 0.012308 | 0.018654 |
| lr1e-04_rank8_result.json | 1e-4 | 8 | 7.651350e-5 | 0.100424 | 0.256 | 0.522 | 0.038330 | 0.043102 |
| lr2e-04_rank8_result.json | 2e-4 | 8 | 1.909311e-4 | 0.250597 | 0.242 | 0.478 | 0.045800 | 0.068763 |
| lr5e-04_rank8_result.json | 5e-4 | 8 | NaN | NaN | 0.240 | 0.478 | 0.000000 | 0.112571 |
| lr1e-03_rank8_result.json | 1e-3 | 8 | 9.881502e-4 | 1.296947 | 0.240 | 0.478 | 0.000000 | 0.225476 |

---

## H2: Mistral-7B-v0.1 Accuracy-Based σ½ Recompute (2026-07-17)

Computed via `recompute_sigma_half_acc.py` on a rented pod (root@194.68.245.61:22037, NVIDIA A40), pointed at `--h2_dir h2_mistral` instead of the script's Llama-oriented default. Purpose: the original H2-J/H2-K cross-architecture comparison (above) used an accuracy-based σ½ for Llama but only an NLL-based σ½ for Mistral (no accuracy-based version existed) — not a like-for-like comparison. This run computes the missing accuracy-based σ½_A for Mistral's AGNews Phase-1 checkpoint.

**Files**: `h2_mistral/agnews_to_mnli/phase1_sigma_half_acc.json`, `h2_mistral/agnews_to_sst2/phase1_sigma_half_acc.json` (identical — same Phase-1 checkpoint; the second was produced by copying the first before the second pod run, then confirmed by the second run loading it as cached rather than recomputing).

```
sigma_half        = 0.0007314285714285715
sigma_half_nll    = 0.0007619047619047619   (ratio nll/acc = 1.04×; ±25-50% uncertainty — density
                                              curve below jumps straight from 1.0 to 0.0 between only
                                              two tested σ values, no --refine pass run)
baseline_acc      = 0.9
success_threshold = 0.81
acc_slack         = 0.1

densities:
  7.619048e-6 → 1.0
  1.523810e-5 → 1.0
  3.047619e-5 → 1.0
  6.095238e-5 → 1.0
  1.219048e-4 → 1.0
  2.438095e-4 → 1.0
  4.876190e-4 → 1.0
  9.752381e-4 → 0.0
```

**`summary_acc_sigma.json`** — adds `R_A_acc` (= `final_per_param / 0.0007314285714285715`) and `R_A_nll` (= `final_per_param / 0.0007619047619047619`) to the existing per-condition rows above. `tstar_checkpoint` fields (`tstar_R_A`, `forget_R_A`) are unchanged from the original run — those come from the live training trajectory's NLL-based σ½ and were not recomputed on the acc basis.

### agnews_to_mnli (Mistral, R_A_acc added)

| lr | rank | final_per_param | R_A_acc | R_A_nll | final_acc_task_a | final_acc_task_b | success_threshold | actual outcome |
|----|------|-----------------|---------|---------|-------------------|-------------------|---------------------|----------------|
| 1e-5 | 8 | 1.485568e-5 | 0.020310 | 0.019498 | 0.884 | 0.894 | 0.81 | safe |
| 5e-5 | 8 | 6.231956e-5 | 0.085203 | 0.081794 | 0.786 | 0.880 | 0.81 | **forgot** |
| 1e-4 | 8 | 1.365726e-4 | 0.186720 | 0.179252 | 0.666 | 0.846 | 0.81 | forgot |
| 2e-4 | 8 | 2.158409e-4 | 0.295095 | 0.283291 | 0.244 | 0.316 | 0.81 | forgot |
| 5e-4 | 8 | NaN | NaN | NaN | 0.244 | 0.332 | 0.81 | forgot |
| 1e-3 | 8 | NaN | NaN | NaN | 0.240 | 0.352 | 0.81 | forgot |

### agnews_to_sst2 (Mistral, R_A_acc added)

| lr | rank | final_per_param | R_A_acc | R_A_nll | final_acc_task_a | final_acc_task_b | success_threshold | actual outcome |
|----|------|-----------------|---------|---------|-------------------|-------------------|---------------------|----------------|
| 1e-5 | 8 | 1.377197e-5 | 0.018829 | 0.018076 | 0.902 | 0.960 | 0.81 | safe |
| 5e-5 | 8 | 4.992105e-5 | 0.068251 | 0.065521 | 0.882 | 0.956 | 0.81 | **safe** |
| 1e-4 | 8 | 7.651350e-5 | 0.104608 | 0.100424 | 0.256 | 0.522 | 0.81 | **forgot** |
| 2e-4 | 8 | 1.909311e-4 | 0.261039 | 0.250597 | 0.242 | 0.478 | 0.81 | forgot |
| 5e-4 | 8 | NaN | NaN | NaN | 0.240 | 0.478 | 0.81 | forgot |
| 1e-3 | 8 | 9.881502e-4 | 1.350987 | 1.296947 | 0.240 | 0.478 | 0.81 | forgot |

**"actual outcome" above is computed directly from `final_acc_task_a < success_threshold`, extracted from the same JSON rows — it is not the script's own printed "Forgot(acc)?" column.** That column checks `R_A_acc > 1.0` instead, and is wrong for nearly every row in both tables: e.g. agnews_to_mnli lr=1e-4 has `final_acc_task_a=0.666` (well below the 0.81 threshold, i.e. forgotten) but the script prints "no" because R_A_acc=0.187 hasn't crossed 1.0. Confirmed live in this run's own console output, matching the same class of bug already found by code inspection in `h2_sequential_llama.py`'s trajectory-tracking logic earlier in this project.

**T\* (last safe → first forgot, computed from `final_acc_task_a` vs. `success_threshold`, both columns above):**
- AGNews→MNLI: [0.0203, 0.0852] (last safe lr=1e-5, first forgot lr=5e-5)
- AGNews→SST-2: [0.0683, 0.1046] (last safe lr=5e-5, first forgot lr=1e-4)

**Comparison against Llama's acc-based T\* for the same AGNews Phase-1 checkpoint (from the H2-D/H2-E tables above, σ½_A_acc=2.755586e-3):**
- AGNews→MNLI: Llama [0.042, 0.085] vs. Mistral [0.020, 0.085] — overlap; first-forgot R lands at the same tested LR step for both (0.085 vs 0.0852)
- AGNews→SST-2: Llama [0.042, 0.089] vs. Mistral [0.068, 0.105] — overlap; first-forgot R lands at the same tested LR step but further apart (0.089 vs 0.1046)

**⚠️ Note (2026-07-17): "0.085 vs 0.0852" is not a demonstrated 0.2% precision match** — both are single points from a 5-value LR grid, not a measured quantity with sub-percent precision. Read the two comparisons above as "same tested grid step, close" vs. "same tested grid step, further apart." Both AGNews Phase-2 pairs overlap on a consistent, accuracy-based σ½ for both architectures — this supersedes the earlier mixed-methodology comparison (Llama acc-based vs. Mistral NLL-based) documented in the H2-J section above, which only had a computed comparison for one of the two pairs.

**⚠️ SUPERSEDED 2026-07-30 — the block-scope σ½_A_acc used for every number above (Llama and Mistral alike) has the same scope mismatch already found and fixed for H1's σ½_pre. See the new section immediately below.**

---

## H2: σ½_A_acc LoRA-Scope Correction (2026-07-30)

**Motivation**: H1's σ½_pre was found (2026-07-21/22) to be miscalibrated — perturbed over the whole decoder block instead of LoRA's actual target modules — and corrected via `recompute_h1_sigma_relslack.py --sigma_scope lora`. H2's own σ½_A_acc calibration (`recompute_sigma_half_acc.py`) was never checked for the same issue. Code inspection (2026-07-29) confirmed it: the script used `_get_target_names_auto` (whole-block scope) exclusively, with zero references to `_get_lora_scoped_names`, and its last modification date (2026-06-30) predates the LoRA-scope fix's existence (2026-07-21) by three weeks — this script had never been touched since the bug was found. Fixed by adding the same `--sigma_scope {lora,block}` flag already used for H1 (default `lora`), writing to new filenames (`phase1_sigma_half_acc_lorascope.json`, `summary_acc_sigma_lorascope.json`) so the legacy block-scope files — read directly by `plot_h2_full_analysis.py`, `compute_fullmodel_norm.py`, `plot_comprehensive_analysis.py`, `make_ntrs_figures.py`, `make_h2_figures.py` — are never overwritten.

Run on all 5 H2 Phase-1 checkpoints (frozen-checkpoint calibration only, no fine-tuning, no risk of hindsight selection): 4 Llama-3.2-3B Phase-1 tasks + 1 Mistral-7B-v0.1 Phase-1 task.

### New σ½_A_acc values (lora-scope) vs. existing (block-scope)

| Phase-1 task | σ½_A_acc (block, existing) | σ½_A_acc (lora, new) | Ratio | n_target_params/n_total |
|---|---|---|---|---|
| Llama MNLI | 2.857143e-03 | 4.726050e-03 | **1.654× (+65.4%)** | 704,643,072/3,212,749,824 |
| Llama SST-2 | 3.337374e-03 | 5.540249e-03 | **1.660× (+66.0%)** | 704,643,072/3,212,749,824 |
| Llama AGNews | 2.755586e-03 | 5.021609e-03 | **1.822× (+82.2%)** | 704,643,072/3,212,749,824 |
| Llama CoLA | 1.943802e-03 | 3.476471e-03 | **1.788× (+78.8%)** | 704,643,072/3,212,749,824 |
| Mistral AGNews | 7.314286e-04 | 9.752381e-04 | **1.333× (+33.3%)** | 1,342,177,280/7,241,732,096 |

All four Llama Phase-1 tasks' `n_target_params`/`n_total_params` match `hessian_eigenvalue.py`'s and H1's independently-computed LoRA-scoped counts exactly (704,643,072/3,212,749,824) — a cross-script consistency check, same as H1's correction. Same for Mistral (1,342,177,280/7,241,732,096).

Since σ½_A_acc is a per-Phase-1-task constant, this rescaling applies uniformly to every Phase-2 target and rank tested against a given Phase-1 checkpoint — **within-task comparisons (rank-invariance, Phase-2-independence, e.g. "MNLI T*(r8) consistent across SST-2/AGNews targets") are unaffected**, since both sides of any such comparison divide by the same corrected constant. What changes is (a) every absolute R_A/T* value in every H2 table, and (b) any comparison across Phase-1 tasks or across architectures.

### Cross-architecture comparison, recomputed (the one place this correction actually matters)

Recomputed directly from each condition's unchanged `final_per_param`/`final_acc_task_a` (never trusting a rescaled bracket estimate):

| Pair | T* (block, existing) | T* (lora, corrected) |
|---|---|---|
| Llama AGNews→MNLI | [0.0421, 0.0855] | [0.0231, 0.0469] |
| Llama AGNews→SST-2 | [0.0419, 0.0887] | [0.0230, 0.0487] |
| Mistral AGNews→MNLI | [0.0203, 0.0852] | [0.0152, 0.0639] |
| Mistral AGNews→SST-2 | [0.0683, 0.1046] | [0.0512, 0.0785] |

**AGNews→MNLI still overlaps** ([0.0231,0.0469] ∩ [0.0152,0.0639] = [0.0231,0.0469] — Llama's bracket sits entirely inside Mistral's). **AGNews→SST-2 no longer overlaps** ([0.0230,0.0487] vs. [0.0512,0.0785] — Llama's upper bound now falls just short of Mistral's lower bound, a ~5% gap). This directly parallels what happened to H1's GPT-2/Llama/Mistral chain under the same correction: one of two previously-overlapping cross-architecture comparisons survives, the other breaks under the more methodologically correct scope — not a dramatic reversal, but a real one, and it means the 2026-07-17 "V6 architecture concern FULLY ADDRESSED" conclusion (based on both pairs overlapping) needs updating to "partially addressed — one of two pairs overlaps under the corrected, consistent methodology."

Raw files: `sigma_lorascope_h2/h2_llama/{mnli_to_sst2,sst2_to_mnli,agnews_to_mnli,cola_to_mnli}/phase1_sigma_half_acc_lorascope.json`, `sigma_lorascope_h2/h2_mistral/agnews_to_mnli/phase1_sigma_half_acc_lorascope.json`, plus per-config `summary_acc_sigma_lorascope.json` and `sigma_lora_*.log`.

---

## Seed Variance Experiment: GPT-2 SST-2

**Model**: GPT-2 | **Task**: SST-2 | **rank**: 8 | **init_seed**: 0 | **data_seeds**: [42, 123, 456, 789, 1337]  
**sigma_half_pre**: 7.191e-3 (from `--sigma_half` arg, matching `h1_kl/gpt2/pretrained_density_relslack.json`)  
**Script**: `seed_variance_experiment.py`  
`R` = per_param_norm / sigma_half_pre. `pairwise_dist` = RMS of (θ_a − θ_b) over all params. `dist_over_sigma_half` = pairwise_dist / sigma_half_pre.

---

### Run A: 500 steps, lrs=[5e-5, 1e-4, 2e-4]
**Dir**: `seed_variance/seed_variance/gpt2/sst2/`  
**Source**: `lr*_seed*_result.json`, `lr*_pairwise.json`, `summary.json`

#### Per-LR summary (from `summary.json`)

| lr | sigma_half_pre | n_seeds | acc_mean | acc_std | R_mean | R_std | per_param_norm_mean | per_param_norm_std | pairwise_dist_mean | pairwise_dist_min | pairwise_dist_max | pairwise_ratio_mean | pairwise_ratio_min | pairwise_ratio_max |
|----|---------------|---------|----------|---------|--------|-------|--------------------|--------------------|-------------------|------------------|------------------|--------------------|--------------------|-------------------|
| 5e-5 | 7.1910112e-3 | 5 | 0.53 | 0.0 | 1.9949715e-2 | 7.5221766e-4 | 1.4345863e-4 | 5.4092056e-6 | 1.0619449e-4 | 9.9883990e-5 | 1.1273949e-4 | 1.4767672e-2 | 1.3890117e-2 | 1.5677835e-2 |
| 1e-4 | 7.1910112e-3 | 5 | 0.53 | 0.0 | 3.0479730e-2 | 1.8018697e-3 | 2.1918008e-4 | 1.2957265e-5 | 1.9881625e-4 | 1.7766273e-4 | 2.1424379e-4 | 2.7647884e-2 | 2.4706224e-2 | 2.9793277e-2 |
| 2e-4 | 7.1910112e-3 | 5 | 0.5592 | 5.7405226e-2 | 5.6432032e-2 | 1.2263268e-2 | 4.0580337e-4 | 8.8185297e-5 | 4.8624616e-4 | 3.8841753e-4 | 5.9363930e-4 | 6.7618607e-2 | 5.4014312e-2 | 8.2552965e-2 |

#### Pairwise distances — lr=5e-5 (from `lr5e-05_pairwise.json`)

| seed_a | seed_b | pairwise_dist | dist_over_sigma_half |
|--------|--------|--------------|---------------------|
| 42 | 123 | 1.0083034e-4 | 1.4021928e-2 |
| 42 | 456 | 1.0549283e-4 | 1.4670116e-2 |
| 42 | 789 | 1.0600800e-4 | 1.4741652e-2 |
| 42 | 1337 | 1.0349350e-4 | 1.4392175e-2 |
| 123 | 456 | 1.1273949e-4 | 1.5677835e-2 |
| 123 | 789 | 1.0867413e-4 | 1.5113645e-2 |
| 123 | 1337 | 1.0436721e-4 | 1.4514517e-2 |
| 456 | 789 | 1.0777247e-4 | 1.4988220e-2 |
| 456 | 1337 | 9.9883990e-5 | 1.3890117e-2 |
| 789 | 1337 | 1.0152278e-4 | 1.4118042e-2 |

#### Pairwise distances — lr=1e-4 (from `lr1e-04_pairwise.json`)

| seed_a | seed_b | pairwise_dist | dist_over_sigma_half |
|--------|--------|--------------|---------------------|
| 42 | 123 | 1.9843785e-4 | 2.7594006e-2 |
| 42 | 456 | 2.1424379e-4 | 2.9793277e-2 |
| 42 | 789 | 2.0594282e-4 | 2.8637239e-2 |
| 42 | 1337 | 1.9038064e-4 | 2.6474228e-2 |
| 123 | 456 | 2.0787685e-4 | 2.8906374e-2 |
| 123 | 789 | 1.9432481e-4 | 2.7023337e-2 |
| 123 | 1337 | 1.7766273e-4 | 2.4706224e-2 |
| 456 | 789 | 2.0677990e-4 | 2.8753593e-2 |
| 456 | 1337 | 1.9676070e-4 | 2.7361929e-2 |
| 789 | 1337 | 1.8626753e-4 | 2.5902527e-2 |

#### Pairwise distances — lr=2e-4 (from `lr2e-04_pairwise.json`)

| seed_a | seed_b | pairwise_dist | dist_over_sigma_half |
|--------|--------|--------------|---------------------|
| 42 | 123 | 5.9363930e-4 | 8.2552965e-2 |
| 42 | 456 | 5.3724424e-4 | 7.4709337e-2 |
| 42 | 789 | 4.4694028e-4 | 6.2149127e-2 |
| 42 | 1337 | 4.3291408e-4 | 6.0198157e-2 |
| 123 | 456 | 5.1945626e-4 | 7.2233158e-2 |
| 123 | 789 | 4.7070697e-4 | 6.5456986e-2 |
| 123 | 1337 | 4.5817527e-4 | 6.3716617e-2 |
| 456 | 789 | 4.4128167e-4 | 6.1362235e-2 |
| 456 | 1337 | 4.2832497e-4 | 5.9561806e-2 |
| 789 | 1337 | 3.8841753e-4 | 5.4014312e-2 |

---

### Run B: 3000 steps, lr=5e-4 (fresh); lr=1e-4 and lr=2e-4 loaded from Run A cache
**Dir**: `seed_variance_3k/gpt2/sst2/`  
**Source**: `lr*_seed*_result.json`, `lr*_pairwise.json`, `summary.json`  
Note: lr=1e-4 and lr=2e-4 entries in summary.json are identical to Run A (script hit cache). Only lr=5e-4 ran fresh at 3000 steps.

#### Per-LR summary — lr=5e-4 only (from `summary.json`)

| lr | sigma_half_pre | n_seeds | acc_mean | acc_std | R_mean | R_std | per_param_norm_mean | per_param_norm_std | pairwise_dist_mean | pairwise_dist_min | pairwise_dist_max | pairwise_ratio_mean | pairwise_ratio_min | pairwise_ratio_max |
|----|---------------|---------|----------|---------|--------|-------|--------------------|--------------------|-------------------|------------------|------------------|--------------------|--------------------|-------------------|
| 5e-4 | 7.191e-3 | 5 | 0.778 | 1.6061880e-1 | NaN | NaN | NaN | NaN | NaN | NaN | NaN | NaN | NaN | NaN |

#### Per-seed results — lr=5e-4, 3000 steps (from `lr5e-04_seed*_result.json`)

| data_seed | steps | per_param_norm | R | acc | delta_nll_wiki |
|-----------|-------|---------------|---|-----|---------------|
| 42 | 3000 | NaN | NaN | 0.47 | NaN |
| 123 | 3000 | 6.6014560e-3 | 9.1801641e-1 | 0.906 | 3.9421875e-1 |
| 456 | 3000 | 7.0243116e-3 | 9.7681985e-1 | 0.906 | 1.0037500e0 |
| 789 | 3000 | 1.2172032e-2 | 1.6926759e0 | 0.802 | 3.9382813e0 |
| 1337 | 3000 | 1.2389836e-2 | 1.7229642e0 | 0.806 | 3.9585938e0 |

#### Pairwise distances — lr=5e-4, 3000 steps (from `lr5e-04_pairwise.json`)

| seed_a | seed_b | pairwise_dist | dist_over_sigma_half |
|--------|--------|--------------|---------------------|
| 42 | 123 | NaN | NaN |
| 42 | 456 | NaN | NaN |
| 42 | 789 | NaN | NaN |
| 42 | 1337 | NaN | NaN |
| 123 | 456 | 9.2715707e-3 | 1.2893298e0 |
| 123 | 789 | 1.3832080e-2 | 1.9235267e0 |
| 123 | 1337 | 1.4006985e-2 | 1.9478494e0 |
| 456 | 789 | 1.4029911e-2 | 1.9510376e0 |
| 456 | 1337 | 1.4194800e-2 | 1.9739674e0 |
| 789 | 1337 | 1.6987829e-2 | 2.3623736e0 |

---

### Run C: 500 steps, lr=5e-4 (fresh, separate output dir)
**Dir**: `seed_variance_5e4_500steps/gpt2/sst2/`  
**Source**: `lr5e-04_seed*_result.json`, `lr5e-04_pairwise.json`, `summary.json`

#### Per-LR summary (from `summary.json`)

| lr | sigma_half_pre | n_seeds | acc_mean | acc_std | R_mean | R_std | per_param_norm_mean | per_param_norm_std | pairwise_dist_mean | pairwise_dist_min | pairwise_dist_max | pairwise_ratio_mean | pairwise_ratio_min | pairwise_ratio_max |
|----|---------------|---------|----------|---------|--------|-------|--------------------|--------------------|-------------------|------------------|------------------|--------------------|--------------------|-------------------|
| 5e-4 | 7.191e-3 | 5 | 7.196e-1 | 1.5591998e-1 | 2.6816940e-1 | 9.2603571e-2 | 1.9284062e-3 | 6.6591228e-4 | 2.7318872e-3 | 1.7434887e-3 | 3.6267660e-3 | 3.7990366e-1 | 2.4245428e-1 | 5.0434793e-1 |

#### Per-seed results — lr=5e-4, 500 steps (from `lr5e-04_seed*_result.json`)

| data_seed | steps | per_param_norm | R | acc | delta_nll_wiki |
|-----------|-------|---------------|---|-----|---------------|
| 42 | 500 | 1.6123942e-3 | 2.2422392e-1 | 0.686 | 1.2156250e-1 |
| 123 | 500 | 2.6234342e-3 | 3.6482189e-1 | 0.898 | 1.1984375e-1 |
| 456 | 500 | 2.7869317e-3 | 3.8755829e-1 | 0.874 | 5.0859375e-1 |
| 789 | 500 | 1.5660019e-3 | 2.1777248e-1 | 0.670 | 2.2203125e-1 |
| 1337 | 500 | 1.0532688e-3 | 1.4647042e-1 | 0.470 | 1.5140625e-1 |

#### Pairwise distances — lr=5e-4, 500 steps (from `lr5e-04_pairwise.json`)

| seed_a | seed_b | pairwise_dist | dist_over_sigma_half |
|--------|--------|--------------|---------------------|
| 42 | 123 | 3.0543185e-3 | 4.2474182e-1 |
| 42 | 456 | 3.1933445e-3 | 4.4407516e-1 |
| 42 | 789 | 2.0114387e-3 | 2.7971613e-1 |
| 42 | 1337 | 1.7540172e-3 | 2.4391840e-1 |
| 123 | 456 | 3.6267660e-3 | 5.0434793e-1 |
| 123 | 789 | 3.0359404e-3 | 4.2218612e-1 |
| 123 | 1337 | 2.7856152e-3 | 3.8737522e-1 |
| 456 | 789 | 3.1717296e-3 | 4.4106934e-1 |
| 456 | 1337 | 2.9422133e-3 | 4.0915218e-1 |

---

## Hessian Eigenvalue Check (Q1: does σ½ track real curvature?) — 2026-07-20/21

**Script**: `hessian_eigenvalue.py`. Tests whether σ½ ∝ 1/√λ_max (the checkpoint sitting in a quadratic basin) by computing the Hessian's extremal eigenvalues, restricted to LoRA's actual target modules (not the whole decoder block — see script docstring), via Lanczos tridiagonalization (replaced an earlier power-iteration design that had two confirmed bugs: a spurious Phase-2 restart convergence, and bf16 finite-difference underflow — see script history/comments for both). GPT-2 uses exact double-backward HVP (fp32, no perturbation, no eps); Llama/Mistral (>2B params) use finite-difference HVP in bf16, with a built-in stability check — Lanczos is run at `eps` and `2×eps` from an *identical* starting Krylov vector (shared seed — see below for why this matters), and the result is only trusted if the two agree within 20% relative.

LoRA-scoped target params (of total): GPT-2 56,669,184/124,439,808 (45.5%); Llama-3.2-3B 704,643,072/3,212,749,824 (21.9%); Mistral-7B-v0.1 1,342,177,280/7,241,732,096 (18.5%).

**⚠️ Flagged 2026-07-22 — GPT-2 param count predates the bias-exclusion fix, unlike Llama/Mistral's.** The σ½/LoRA-Scope Correction section below reports GPT-2's LoRA-scoped count as 56,623,104 (from `_get_lora_scoped_names`'s current, bias-excluding definition — see that function's docstring in `h1_kl_comparison.py`, which explains GPT-2's `Conv1D` layers are the only ones of the three architectures with bias params in scope) — 46,080 fewer than the 56,669,184 above. That difference is exactly explained by bias: GPT-2 has 12 layers, and the three matched modules per layer (`c_attn`, `attn.c_proj`, `mlp.c_proj`, via the same name-collision this document already documents elsewhere) sum to 3×768+768+768=3,840 bias params/layer × 12 = 46,080 — confirmed by direct computation, not coincidence. `hessian_eigenvalue.py` now imports `_get_lora_scoped_names` from `h1_kl_comparison.py` (moved there 2026-07-21, same window as these GPT-2 runs), but the runs below almost certainly predate that move and ran on the script's own earlier, bias-included inline version — Llama's and Mistral's counts match the current function exactly (see the correction section below), only GPT-2's doesn't, consistent with GPT-2 being the one architecture where bias inclusion actually changes the scoped count. **Practical impact: bias is 0.08% of GPT-2's scoped parameters — expected to be inconsequential to the λ_min<0 saddle-point conclusion, but this has not been independently reverified on the corrected parameter set.**

### GPT-2 — exact double-backward, 3 independent runs (fresh random Lanczos seed each time)

| Run | λ_min | λ_max^+ | σ½ (measured) | σ½ (theory) | measured/theory |
|---|---|---|---|---|---|
| 1 | -36.0472 | 28.4231 | 0.007191 | 0.054434 | 0.132 |
| 2 | -36.0453 | 28.4231 | 0.007191 | 0.054434 | 0.132 |
| 3 | -36.0487 | 28.4234 | 0.007191 | 0.054433 | 0.132 |

All 3 agree to within 0.005 on both eigenvalues. `λ_min` confirmed negative in all 3 — genuine saddle point in the LoRA-scoped subspace, not a local minimum. `measured/theory=0.132` consistently (single-eigenvalue quadratic theory overestimates σ½ by ~7.6×).

### Mistral-7B-v0.1 — finite-difference, 2 independent runs, identical settings (eps=1e-4, hessian_batch=4)

| Run | base λ_min | base λ_max^+ | 2×eps λ_min | 2×eps λ_max^+ | swing | stable |
|---|---|---|---|---|---|---|
| 1 | -908.6729 | 910.2041 | -958.5185 | 957.06 | ~4.9% | **True** |
| 2 | (not printed) | (not printed) | -1908.9922 | 1909.32 | 50% | **False** |

Run 1's full result: σ½=0.0001824175824175824, theory_σ½=0.0076453818325957715, ratio_meas_theory=0.023859839366014726 (measured/theory ≈ 0.024, i.e. theory overestimates σ½ by ~42×). Run 2 used the identical CLI settings (confirmed via the run's own printed startup line: `hessian_batch=4 ... eps=0.0001`) — the only thing that differs between the two runs is the random seed for Lanczos's initial Krylov vector (freshly generated each process invocation; the "shared seed" fix only pins the seed *within* one run's eps-vs-2eps comparison, not *across* separate script invocations). **Run 1's "stable: True" is retracted as evidence** — it was one seed's eps-check passing, not proof the measurement is seed-robust, which Run 2 disproves directly.

### Llama-3.2-3B — finite-difference, 4 attempts, never passed the 20% stability threshold

| Attempt | eps | hessian_batch | seed-fix applied? | base λ_max^+ | 2×eps λ_max^+ | swing | stable |
|---|---|---|---|---|---|---|---|
| 1 | 1e-4 (default) | 4 | No (confounded) | 126.206 | 56.4433 | 55% | False |
| 2 | 1e-3 | 4 | No (confounded) | (not printed) | 5.21466 | 65% | False |
| 3 | 1e-4 (default) | 4 | **Yes** (clean) | (not printed) | 256.619 | **34%** (best result) | False |
| 4 | 3e-4 | 8 | **Yes** (clean) | (not printed) | 23.3727 | 58% | False |

Attempts 1-2 used independently-random Krylov starts for the base and 2×eps runs (a confound fixed after attempt 2 — see script's `lanczos_spectrum` `v0` parameter). Attempts 3-4, with that confound removed, still show no monotonic relationship between eps and stability (34%→58% when eps increased 3×) — consistent with genuine, non-eps-tunable measurement noise rather than a fixable step-size problem.

**⚠️ Verifiability caveat, flagged 2026-07-21 by independent re-derivation of every swing % from this section:** only 3 of the 7 swing percentages above can actually be recomputed by a reader from the numbers this document shows.
- **Independently checkable, confirmed exact or near-exact:** GPT-2's 3 runs (self-consistent by construction). Llama attempt 1: |126.206−56.4433|/126.206 ≈ 55.3%, matches "55%". Mistral run 1: (958.52−908.67)/908.67 ≈ 5.5% and (957.06−910.20)/910.20 ≈ 5.2%, close to but not exactly "4.9%" (minor, doesn't change the conclusion).
- **Not independently checkable from this document:** Mistral run 2 and Llama attempts 2-4 all have a base λ_max^+ marked "(not printed)" — the swing % shown was computed and printed by the script itself (`top_eigenvalue`'s own WARNING line), not asserted by hand, but the base value needed to redo that arithmetic was never captured in what got pasted into this conversation (the per-step trajectory and the base-run's own summary line print *before* the stability-check line, and were evidently scrolled past in the terminal before copying). If bulletproof verification is ever needed, the original values would still be in that pod's terminal scrollback/session if it's still alive — this is a documentation gap, not a claim that the underlying numbers are wrong.
- **What still holds given only the checkable subset:** Llama attempt 1 alone already fails the 20% threshold at 55% (fully verified) — sufficient by itself to support "not reliably measurable," independent of attempts 2-4. Mistral's retraction doesn't depend on run 2's exact swing % either: run 2's *2×eps* magnitude (~1909) is directly comparable to run 1's *2×eps* magnitude (~957) — same setting, same quantity, both fully shown — and that alone (nearly 2× different across two "identical" runs) is a real, checkable red flag independent of whether "50%" is exactly right for run 2's internal swing.

### Bugs found and fixed in `hessian_eigenvalue.py` during this investigation (chronological)
1. **SIGMA_SOURCES pointed at a nonexistent pre-relslack directory layout** — every model silently hit "no σ½ data found," the script had never produced output before this session.
2. **Whole-decoder-block Hessian scope OOM'd on Mistral** (~7B params, fp32 grad buffers) — fixed by scoping to LoRA's own target modules (`_get_lora_scoped_names`), which also directly serves the σ½/LoRA-scope validity question (see SUMMARY.md).
3. **bf16 finite-difference underflow** — a globally-unit-normalized perturbation vector's per-coordinate magnitude shrinks as 1/√d; for multi-billion-parameter models this fell below bf16's representable increment, making `Hv=0` a numerical artifact, not a real zero-curvature finding. Fixed by scaling the FD step by √d.
4. **Inexact perturb/restore cycle causing cross-call drift** — three sequential in-place ±shifts each rounded to bf16 independently, not cancelling exactly; compounded across ~24 Lanczos steps per run. Fixed via snapshot-once-and-exact-restore instead of sequential shifts.
5. **Spurious Phase-2 power-iteration convergence** (GPT-2, pre-Lanczos) — a fresh random restart's Rayleigh quotient on a shifted matrix was dominated by the shift term, converging in 2 iterations to just the shift value. Fixed by replacing the whole power-iteration+shift scheme with Lanczos tridiagonalization.
6. **Gradient checkpointing silently disabled by `model.eval()`** — HF gates checkpointing on `model.training`; fixed by using `model.train()` + explicitly zeroing all Dropout modules for the FD path instead.
7. **Reentrant checkpointing's known gradient-flow gotcha** with sparse (LoRA-only) `requires_grad` — switched to `use_reentrant=False`.
8. **Eps-doubling stability check confounded by independent random restarts** — fixed by sharing one Krylov seed vector (`v0`) between a single run's eps and 2×eps comparison. This fix alone dropped Llama's swing from 55% to 34% at the same eps, but did not make it pass — and did not protect against the cross-run (Mistral) seed variance found afterward, which remains unresolved.

---

## σ½/LoRA-Scope Correction — Cross-Architecture Recompute (2026-07-21/22)

**Script**: `recompute_h1_sigma_relslack.py --sigma_scope lora` (new `--sigma_scope` flag, default changed to `lora`; `--sigma_scope block` reproduces the legacy numbers already cited throughout this document). Addresses the open validity question already on record: σ½ was being calibrated by perturbing the *entire* decoder block, while LoRA fine-tuning only ever displaces its own target modules — this recomputes σ½ restricted to that same LoRA-scoped parameter set, for all three H1 models, on the frozen pretrained checkpoint (no fine-tuning involved, no risk of hindsight/outcome-based selection).

### New σ½ values (relslack, LoRA-scoped) vs. existing (relslack, block-scoped)

| Model | σ½ (block, existing) | σ½ (lora, new) | Ratio (lora/block) | `sigma_half_is_lower_bound` |
|---|---|---|---|---|
| GPT-2 | 0.0071910112359550565 | 0.007697841726618705 | **1.070× (+7.0%)** | false |
| Llama-3.2-3B | 0.0007717391304347826 | 0.0015172413793103448 | **1.966× (+96.6%)** | false |
| Mistral-7B-v0.1 | 0.0001824175824175824 | 0.00035100671140939597 | **1.924× (+92.4%)** | false |

All three confirmed **not** capped (`sigma_half_is_lower_bound: false` in each saved JSON) — genuine resolved half-max measurements, not lower bounds. LoRA-scoped target param counts (from each JSON's `n_target_params`/`n_total_params`): GPT-2 56,623,104/124,439,808; Llama 704,643,072/3,212,749,824; Mistral 1,342,177,280/7,241,732,096 — the Llama and Mistral counts match `hessian_eigenvalue.py`'s independently-computed LoRA-scoped counts exactly (704,643,072 and 1,342,177,280 respectively), a cross-script consistency check.

**This confirms the exact directional prediction made when this validity question was first raised**: GPT-2's LoRA targets (`c_attn`, `c_proj`) collide with its MLP naming (`c_proj` names both `attn.c_proj` and `mlp.c_proj`), so GPT-2's LoRA scope already overlapped much of what block-scope certified — recomputing moves it only slightly (+7.0%). Llama/Mistral's LoRA targets (`q/k/v/o_proj`) have no such collision — attention-only — so narrowing to the true scope reveals a much tighter basin, nearly doubling σ½ for both (+96.6%, +92.4%).

### ⚠️ Data-hygiene bug found during this recompute: stale duplicate Llama SST-2 results

Two different copies of the Llama SST-2 LR-sweep results exist locally, at different paths, with **different numbers**, not just duplicated files:

| Path | Date | Fields present | lr=1e-4: per_param_norm | delta_nll_wiki | acc |
|---|---|---|---|---|---|
| `h1_kl/meta-llama/Llama-3.2-3B/` | Jun 28 | no `task`, no `kl_superni`, no `mmlu_acc` | 1.812863e-04 | +0.02531 | 0.966 |
| `h1_kl/meta-llama/meta-llama/Llama-3.2-3B/` | Jul 3 | has `task`, `kl_superni`, `mmlu_acc` | 1.818914e-04 | +0.03516 | 0.962 |

The double-nested path's numbers (ΔNLL=+0.035, KL(wiki)≈0.076) match SUMMARY.md's cited H1 Llama SST-2 table exactly; the single-nested path does not. Combined with the date and the presence of `kl_superni`/`mmlu_acc` (added per SUMMARY.md's own "2026-07-03: H1-SST-2 extended with KL(SuperNI) + MMLU (Batch 3)" entry), **`h1_kl/meta-llama/meta-llama/Llama-3.2-3B/` is the authoritative, final run; `h1_kl/meta-llama/Llama-3.2-3B/` is a stale pre-Batch-3 run that was superseded and never cleaned up.** All recomputation below uses the double-nested (authoritative) path. Both directions' own `pretrained_density.json`/`pretrained_density_relslack.json` (calibrated on the frozen pretrained checkpoint, independent of which fine-tuning results happen to sit alongside them) were verified byte-identical between the two paths — the staleness is confined to the `lr*_results.json` fine-tuning outputs, not the σ½ calibration.

**Resolved 2026-07-22**: the stale single-nested directory (all files dated 2026-06-28, confirmed via `stat`) has been moved to `archive/old_results/h1_kl_llama-3.2-3b_stale_20260628/` to prevent this ambiguity from recurring. `h1_kl/meta-llama/` now contains only the authoritative double-nested path.

### Full R recompute, all 3 models, both σ½ scopes (SST-2 fine-tuning, rank 8)

Raw `per_param_norm`/`delta_nll_wiki` extracted directly from each model's own `lr*_results.json` (unchanged by this correction — only the σ½ denominator changes).

**GPT-2** (`h1_kl/gpt2/`):
| lr | per_param_norm | R (block) | R (lora) | ΔNLL(wiki) |
|---|---|---|---|---|
| 1e-5 | 1.080750e-04 | 0.0150 | 0.0140 | −0.0272 |
| 5e-5 | 6.177179e-04 | 0.0859 | 0.0802 | −0.0536 |
| 1e-4 | 1.582308e-03 | 0.2200 | 0.2056 | +0.0156 |
| 2e-4 | 3.260399e-03 | 0.4534 | 0.4235 | +1.3853 |
| 5e-4 | 1.251897e-02 | 1.7409 | 1.6263 | +4.3098 |

**Llama-3.2-3B** (`h1_kl/meta-llama/meta-llama/Llama-3.2-3B/`, authoritative path):
| lr | per_param_norm | R (block) | R (lora) | ΔNLL(wiki) |
|---|---|---|---|---|
| 1e-5 | 2.191735e-05 | 0.0284 | 0.0144 | −0.0022 |
| 5e-5 | 8.746536e-05 | 0.1133 | 0.0576 | +0.0092 |
| 1e-4 | 1.818914e-04 | 0.2357 | 0.1199 | +0.0352 |
| 2e-4 | 4.518825e-04 | 0.5855 | 0.2978 | +1.1334 |
| 5e-4 | 2.810228e-03 | 3.6414 | 1.8522 | +8.7375 |

**Mistral-7B-v0.1 v5 PRIMARY** (`h1_kl/mistralai/Mistral-7B-v0.1_sst2_v5/`):
| lr | per_param_norm | R (block) | R (lora) | ΔNLL(wiki) |
|---|---|---|---|---|
| 1e-5 | 5.468776e-06 | 0.0300 | 0.0156 | +0.0011 |
| 5e-5 | 3.533529e-05 | 0.1937 | 0.1007 | +0.1246 |
| 1e-4 | 4.115245e-05 | 0.2256 | 0.1172 | +2.8095 |
| 2e-4 | 1.171350e-04 | 0.6421 | 0.3337 | +14.2094 |
| 5e-4 | 2.407660e-04 | 1.3199 | 0.6859 | +23.4466 |

### T* brackets, both scopes (1%-of-baseline-NLL break criterion — same consistent criterion used elsewhere in this document, not the looser cliff-jump one)

| Model | T* (block-scope) | T* (lora-scope) |
|---|---|---|
| GPT-2 | [0.2200, 0.4534] | [0.2056, 0.4235] |
| Llama-3.2-3B | [0.1133, 0.2357] | [0.0576, 0.1199] |
| Mistral-7B (v5) | [0.0300, 0.1937] | [0.0156, 0.1007] |

Block-scope brackets match SUMMARY.md's already-documented values exactly (Llama [0.113,0.236], confirming the authoritative-path correction above didn't silently change anything already trusted).

**Pairwise overlap, block-scope (already documented — the "chain, not 3-way" finding):**
GPT-2∩Llama = [0.2200,0.2357] (overlap). Llama∩Mistral = [0.1133,0.1937] (overlap). GPT-2∩Mistral = ∅.

**Pairwise overlap, lora-scope (NEW):**
GPT-2∩Llama = ∅ (0.2056 > 0.1199 — no overlap at all). Llama∩Mistral = [0.0576,0.1007] (overlap, narrower than block-scope but still real). GPT-2∩Mistral = ∅.

Under the corrected scope, **GPT-2 no longer overlaps with either other architecture** — its bracket sits entirely above both. Only the Llama-Mistral pair still overlaps. See SUMMARY.md's "Open validity questions" for the interpretation.

---

## H2: NLL-vs-Accuracy Forgetting — GPT-2 Validation (2026-07-22)

**Script**: `h2_nll_forgetting.py` (new file, does not modify `h2_sequential_llama.py`). Tests whether Task A "forgetting" measured via a frozen classifier's accuracy — the existing H2 signal — agrees with forgetting measured via NLL on Task A's own held-out text (no classifier involved). Motivated by the unresolved SST-2 anomaly (V8): SST-2 Phase 1's forgetting threshold varies 44× depending on the Phase 2 target, and the classifier-accuracy signal is structurally the same "old probe vs new representation" pattern already found and fixed as Bug 4 in `h1_direct_r_sweep.py`.

**Config**: `--model gpt2 --task_a sst2 --task_b mnli agnews cola --phase1_lr 2e-4 --phase1_steps 3000 --force`. Phase 1: SST-2 accuracy=0.7740, NLL_A baseline=7.4352 (on held-out SST-2 text, `--nll_holdout 256` sequences, separate draw from the σ½_A calibration holdout). Phase 2: rank=8, 5 LRs × 3 task_b targets = 15 conditions, 500 steps each, `track_interval=25`. Thresholds: accuracy forgetting = acc_A < 0.9×0.7740; NLL forgetting = ΔNLL_A > 1%×7.4352 = 0.0744 (same relslack convention as H1).

### Summary table (all 15 conditions)

| task_b | lr | T*_acc (last_safe, first_forgot) | T*_nll (last_safe, first_forgot) |
|---|---|---|---|
| mnli | 1e-5 – 5e-4 (all 5) | never forgot | never forgot |
| agnews | 1e-5, 5e-5 | never forgot | never forgot |
| agnews | 1e-4 | (0.01610, 0.01868) | never forgot |
| agnews | 2e-4 | (0.01513, 0.02049) | never forgot |
| agnews | 5e-4 | (0.02046, 0.02722) | never forgot |
| cola | 1e-5 | never forgot | never forgot |
| cola | 5e-5 | (0.01020, 0.01072) | never forgot |
| cola | 1e-4 | (0.01121, 0.01205) | never forgot |
| cola | 2e-4 | (0.01350, 0.01640) | never forgot |
| cola | 5e-4 | (0.01639, 0.02261) | never forgot |

Two things on record already in the table alone: (1) **MNLI never triggers accuracy-based forgetting at any tested LR** — echoes the already-documented SST-2→MNLI anomaly (extreme robustness in that specific direction) generalizing to GPT-2, not just Llama. (2) **NLL-based forgetting never triggers anywhere in this sweep**, including the agnews/cola conditions where accuracy clearly collapsed.

### The NLL trajectories, not just the summary table — this is the actual finding

Raw per-step data from `sst2_to_cola/lr5e-04_rank8_trajectory.json` (accuracy-forgetting condition):
| step | R_A | acc_A | NLL_A | ΔNLL_A |
|---|---|---|---|---|
| 0 | 0.0000 | 0.774 | 7.4352 | +0.0000 |
| 50 | 0.0164 | 0.756 | 7.3500 | −0.0852 |
| 75 | 0.0226 | 0.662 | 5.3546 | −2.0806 |
| 100 | 0.0263 | 0.634 | 5.0546 | −2.3806 |
| 125 | 0.0307 | 0.542 | 4.5348 | −2.9004 |
| 150 | 0.0348 | 0.536 | 4.4925 | −2.9427 |
| 300 | 0.0577 | 0.522 | 4.8448 | −2.5903 |
| 475 | 0.0903 | 0.522 | 5.5161 | −1.9191 |

`sst2_to_agnews/lr5e-04_rank8_trajectory.json` (same pattern):
| step | R_A | acc_A | NLL_A | ΔNLL_A |
|---|---|---|---|---|
| 0 | 0.0000 | 0.774 | 7.4352 | +0.0000 |
| 25 | 0.0114 | 0.810 | 6.7025 | −0.7327 |
| 100 | 0.0272 | 0.562 | 4.7935 | −2.6417 |
| 200 | 0.0671 | 0.522 | 4.5808 | −2.8544 |
| 475 | 0.1280 | 0.540 | 5.2474 | −2.1878 |

`sst2_to_mnli/lr5e-04_rank8_trajectory.json` (never-forgot condition, for contrast):
| step | R_A | acc_A | NLL_A | ΔNLL_A |
|---|---|---|---|---|
| 0 | 0.0000 | 0.774 | 7.4352 | +0.0000 |
| 200 | 0.0263 | 0.630 | 7.9847 | +0.5495 |
| 375 | 0.0479 | 0.744 | 6.8321 | −0.6031 |
| 475 | 0.0641 | 0.754 | 6.9990 | −0.4362 |

**In both accuracy-forgetting conditions (cola, agnews at lr=5e-4), NLL does not stay flat while accuracy collapses — it drops by ~35-40% from baseline (7.44→~4.5-4.9), then partially recovers to ~5.2-5.5, while accuracy craters to chance (~0.52) and stays there.** The NLL improvement is visible from the very first measured steps, before accuracy shows any damage (e.g. agnews step 25: accuracy still 0.810, above baseline; NLL already down 0.73). For the never-forgot MNLI condition, both signals stay much closer to baseline throughout, with no comparable dramatic movement in either direction.

### Interpretation

**⚠️ Reframed 2026-07-22 — the original interpretation below gave the readout-mismatch story more standing than a fluency-proxy signal earns; superseded by the analysis in this box.**

**What NLL-on-SST2-text actually measures vs. what H2 needs.** H2's forgetting question is specifically whether the frozen Phase-1 classifier's linear readout direction still correctly decodes sentiment from the current hidden state — a question about a specific direction in representation space. NLL-on-domain-text measures something adjacent but different: whether the model is fluent at predicting the next token of movie-review-style text. A model can become more fluent at generating SST-2-style text while the sentiment-relevant direction is rotated, reorganized, or overwritten — these are close to independent computations. **NLL-on-domain-text does not directly test whether SST-2 sentiment knowledge was forgotten; it tests something adjacent to it.**

**A second explanation the magnitude pattern doesn't rule out — and may favor.** Re-examining the deltas precisely (not just their sign): in both accuracy-collapse conditions (CoLA, AGNews), NLL improves substantially and durably (~26-29% below baseline by the end of training). In the one condition where accuracy stays safe (MNLI), NLL is transiently *worse* than baseline (step 200: +0.55) before creeping to only a mild ~6% improvement by the end. **The magnitude of NLL "improvement" scales with the magnitude of accuracy collapse** — not merely opposite in sign. If the representation were intact and only the old classifier's readout direction were miscalibrated (the hopeful "readout-mismatch" story), there's no reason NLL should improve *more* in the conditions where accuracy damage is *worse* — that story predicts NLL roughly independent of accuracy-collapse severity, since the representation itself isn't supposed to be damaged. The observed correlation is at least as consistent with a different, less flattering mechanism: aggressive Phase-2 training at high LR could push the model toward degenerate or repetitive output, and collapsed/homogenized text often scores artificially low perplexity on short, generic-looking benchmark sentences — not because sentiment information was preserved anywhere, but because outputs became generic. This would produce exactly this signature (NLL improves, accuracy is catastrophic, magnitudes track each other) without any sentiment information surviving. **Given the magnitude-correlation, this alternative should be treated as at least as parsimonious as the readout-mismatch reading, not a secondary alternative that merely "hasn't been ruled out."**

**The design that would actually resolve this**: freeze the Phase-2-perturbed backbone at each condition's endpoint, train a *fresh* linear probe on its current hidden states, and measure how quickly/well it recovers Task-A (SST-2) accuracy. Fast, low-data recovery → real evidence sentiment information is still linearly present, just inaccessible to the old probe direction (supports readout-mismatch). No better recovery than the old probe → the information itself is genuinely gone, and H2's original accuracy-based forgetting measurement was right all along — a different explanation for the SST-2 anomaly (V8) is needed. This is the standard linear-probing design used elsewhere in the continual-learning literature to distinguish representation drift from readout drift, and it directly targets what H2 cares about rather than a fluency proxy. **Run 2026-07-23 — see full section below.** Script written and executed (`h2_probe_recovery.py`, new file, imports `h2_nll_forgetting.py`'s Phase-1 checkpoint + `h2_sequential_llama.py`'s data/eval utilities unchanged; does not modify either). Reproduces the exact Phase-2 training trajectory (same seeding scheme, LoRA target modules/rank/alpha as `run_phase2_condition_dual`) up to a condition's endpoint, then freezes the entire backbone (base weights + Phase-2 LoRA) and trains a fresh `nn.Linear` probe on Task-A (SST-2) from scratch, recording an accuracy-vs-probe-step recovery curve against the same validation split used for `old_frozen_acc`. Decision rule fixed in the script before any run: crosses 0.9×baseline within ≤25 probe steps → `recovers_fast` (supports readout-mismatch); crosses later → `recovers_slow`; never crosses within the probe-training budget → `no_recovery` (supports the mode-collapse/genuine-loss alternative).

**Caveats**: GPT-2 only (cheap validation per this project's established practice — not yet run on Llama/Mistral). The NLL signal here is domain-specific (does the model still model SST-2-*style text* well), not the same general-capability WikiText-2 NLL H1 uses — a different kind of signal from classification accuracy, not a strictly-better version of the same one. Single run, no seed replication; `nll_A`/`acc_A` are each evaluated on 500/256-example held-out sets respectively, so step-to-step noise is expected and visible in the trajectories (e.g. mnli step 125 shows a one-step NLL spike to +0.234 that doesn't persist).

**⚠️ Flagged 2026-07-22 — the 1%-of-baseline-NLL threshold (0.0744) is reused from H1 without independent calibration for this quantity.** H1 validated that specific relative-slack convention against general-capability WikiText-2 NLL; this experiment applies the identical convention to a different signal (NLL on SST-2-style task text). Nothing here suggests the "right" threshold for task-domain NLL should be numerically the same. This doesn't undermine the actual finding — the observed pattern is qualitative (NLL moves in the *opposite* direction entirely, not merely "stays under a threshold that was set slightly wrong") — but the "T*_nll never triggers, anywhere" framing borrows more precision from the threshold than the threshold has actually earned for this signal. Read "never forgot" as "never crossed this particular convention," not as a validated null.

Raw files: `h2_nll_forgetting_gpt2/gpt2/{sst2_to_mnli,sst2_to_agnews,sst2_to_cola}/lr*_rank8_{result,trajectory}.json`, `sst2_summary.json`, `nll_vs_acc_forgetting.png`.

---

## H2: Fresh-Probe Recovery — GPT-2 (2026-07-23)

**Script**: `h2_probe_recovery.py`, run on a fresh A40 pod. Config: `--model gpt2 --task_a sst2 --task_b mnli agnews cola` (all other args at script defaults: `--phase2_lrs 5e-4 --phase2_steps 500 --phase2_rank 8 --probe_lr 1e-3 --probe_steps 200 --probe_eval_interval 5 --probe_fast_recovery_steps 25`), reusing the exact Phase-1 SST-2 checkpoint from the NLL-vs-accuracy run above (`acc_phase1_a=0.7740`).

### Result table (verified directly from each condition's `result.json`, not the console summary)

| task_b | acc_phase1_a | old_frozen_acc | 0.9×baseline threshold | final_probe_acc | crossing_step | verdict |
|---|---|---|---|---|---|---|
| mnli | 0.774 | 0.6700 | 0.6966 | 0.7080 | 200 | recovers_slow |
| agnews | 0.774 | 0.5360 | 0.6966 | 0.7440 | 165 | recovers_slow |
| cola | 0.774 | 0.5220 | 0.6966 | 0.8080 | 55 | recovers_slow |

**All three conditions land in the middle bucket** — none is `no_recovery` (which would rule in favor of information being genuinely destroyed) and none is `recovers_fast` (which would mean the old classifier's failure was a trivial, near-instant misalignment). A fresh linear probe, trained on nothing but the frozen Phase-2-perturbed backbone, reaches *above* the 90%-of-baseline threshold in all three conditions using at most 200 steps (3,200 SST-2 examples) — including the two conditions (agnews, cola) where the old classifier had collapsed to nearly chance-level accuracy (0.536, 0.522). CoLA's recovered probe (0.808) actually **exceeds** the original Phase-1 baseline (0.774).

### Reproduction-fidelity check (comparing this run's `old_frozen_acc` against the already-documented h2_nll_forgetting.py trajectory's last tracked point, step=475 — the closest available comparison, since this run trains 25 further steps to step=500)

| task_b | this run (step 500) | documented trajectory (step 475) | gap |
|---|---|---|---|
| cola | 0.5220 | 0.522 | **0.0000 — exact match** |
| agnews | 0.5360 | 0.540 | 0.0040 |
| mnli | 0.6700 | 0.754 | **0.0840 — notable gap** |

Two of three conditions reproduce almost exactly, despite this script's `train_phase2` being an independent reimplementation (not a call into the original `run_phase2_condition_dual`) — real evidence the seeding/training reproduction is faithful, not a coincidence. The MNLI gap is larger than the other two, but MNLI's own already-documented trajectory was already the noisiest of the three at this LR (step 200: 0.630 → step 375: 0.744 → step 475: 0.754, a 12-point swing over 100 steps) — an additional ~8-point move over 25 further steps is within that same volatile regime, not obviously a new bug, but is flagged here rather than silently accepted. This weakens confidence specifically in MNLI's `crossing_step=200` value; it does not change the qualitative bucket (`recovers_slow`), since even the documented 0.754 endpoint would still fall well outside the 25-step fast-recovery window.

### Confirmation that the pre-run `_verdict` bug fix (sustained-crossing vs. naive first-crossing) was not just theoretical

CoLA's raw recovery curve spikes to 0.790 at probe_step=15 (well inside the 25-step fast cutoff), then dips to 0.672 (below the 0.6966 threshold) at step=50, before recovering for good from step=55 onward (minimum value from there to step=200 is 0.706, still above threshold). **Naive first-crossing logic — the exact anti-pattern already found and fixed once in H2-K and again in `h2_nll_forgetting.py`'s `_bracket` — would have reported `crossing_step=15` here, giving an incorrect `recovers_fast` verdict.** The sustained-crossing check applied in `_verdict` (fixed during pre-run code review, before this data existed) correctly identifies `crossing_step=55` instead. This is a concrete, empirical case where the review caught a bug that would have changed a real result, not just a theoretical concern.

### Interpretation

The information is not gone (rules against pure mode-collapse/genuine-destruction for all three Phase-2 targets tested), but recovery is not instant either (rules against "the old classifier's failure was a trivial, near-costless misalignment"). Something real happened to how sentiment is linearly encoded — real enough that recovering it takes actual training (55–200 steps depending on target) — but it is recoverable, and recoverable to at or above the original baseline using a tiny fraction of the compute that produced that baseline (200 steps of probe-only training vs. 3,000 steps of full Phase-1 fine-tuning). This is a stronger form of support for the readout-mismatch explanation of the SST-2 anomaly (V8) than the NLL experiment provided, and directly rules out mode-collapse-as-total-information-loss for these three GPT-2 conditions specifically — though it does not by itself explain *why* CoLA recovers over 3x faster than MNLI, or why the recovered representation for CoLA ends up strictly better than the original.

**Caveats**: GPT-2 only, single run, single seed (no seed replication on either the Phase-2 training or the probe initialization/data order). Recovery-speed differences across task_b (55 vs. 165 vs. 200 steps) are single-run point estimates, not confirmed to be stable across seeds. The MNLI reproduction-fidelity gap (above) means that condition's specific number should be read with more caution than agnews/cola's. Not yet run on Llama, where the actual 44× SST-2 anomaly (V8) this was designed to investigate was originally observed — this GPT-2 result is suggestive, not a resolution of V8 itself.

Raw files: `h2_probe_recovery_gpt2/h2_probe_recovery/gpt2/{sst2_to_mnli,sst2_to_agnews,sst2_to_cola}/lr5e-04_rank8_result.json`, `sst2_probe_summary.json`, `probe_recovery.png`, `probe_recovery.log`.

---

## H2: Fresh-Probe Recovery — Severity Sweep, GPT-2 (2026-07-23)

**Motivation**: the run above tested only one, already-catastrophic LR (5e-4). It could not answer the sharper question — does recovery ever genuinely fail, at high enough displacement, or does a fresh probe always eventually recover? Same script (`h2_probe_recovery.py`), same Phase-1 checkpoint, `sst2→cola` only (the target that showed the fastest recovery at 5e-4), swept across `--phase2_lrs 1e-4 2e-4 5e-4 1e-3 2e-3 5e-3` — spanning both below and above the previously-tested point.

### Result table (verified from each condition's `result.json`)

| lr | old_frozen_acc | threshold (0.9×0.774) | final_probe_acc | crossing_step | verdict |
|---|---|---|---|---|---|
| 1e-4 | 0.5300 | 0.6966 | 0.8100 | 65 | recovers_slow |
| 2e-4 | 0.5240 | 0.6966 | 0.8000 | 55 | recovers_slow |
| 5e-4 | 0.5220 | 0.6966 | 0.8080 | 55 | recovers_slow |
| 1e-3 | 0.4780 | 0.6966 | 0.7600 | 200 | recovers_slow |
| 2e-3 | 0.4780 | 0.6966 | 0.4780 | None | no_recovery |
| 5e-3 | 0.4780 | 0.6966 | 0.4780 | None | no_recovery |

A clean, monotonic progression: recovery speed degrades steadily from 55 steps (1e-4/2e-4/5e-4) to 200 steps (right at the edge of the probe-training budget, 1e-3) before flipping to `no_recovery` at 2e-3 and 5e-3. Read in isolation, this looks like exactly the severity-dependent boundary the severity sweep set out to find.

### ⚠️ The two `no_recovery` results are very likely a training-divergence artifact, not evidence of genuine information loss — flagged before, not after, drawing conclusions

Two independent checks on the raw recovery curves (not just the summary numbers) point the same way:

1. **Zero variation across 41 evaluation points.** At lr=2e-3 and 5e-3, `probe_acc` is *exactly* 0.4780 at every single measured step from 0 to 200 — no fluctuation whatsoever. Real SGD training on real minibatches essentially never produces perfectly constant accuracy across 41 independent evaluations; this is the signature of a degenerate, unchanging computation, not "training that failed to find signal."
2. **The old (previously-trained) classifier and a freshly, independently-initialized probe give *identical* accuracy.** In every other condition (1e-4 through 1e-3), the old classifier and the untrained fresh probe — two linear layers with unrelated random/trained weights — give *different* accuracies at step 0, as expected. At lr=2e-3 and 5e-3, both give exactly 0.4780. Two independently-parameterized classifiers agreeing to 4 decimal places is essentially impossible by chance unless the hidden states feeding both of them have degenerated into something that makes classifier weights irrelevant to the output — the standard signature of this is NaN hidden states, since `argmax` on an all-NaN row returns a constant index in PyTorch regardless of the classifier's weights.

This is consistent with the same LoRA training-divergence failure mode already documented elsewhere in this project at aggressive learning rates (e.g. Mistral's `final_per_param: NaN` entries in the H2-J/H2-K tables) — GPT-2 LoRA at lr=2e-3/5e-3 for 500 steps is a very aggressive setting relative to the 1e-5–5e-4 range validated elsewhere in this project's H1/H2 sweeps, and likely diverged partway through Phase-2 training.

**This has not been confirmed by direct inspection** (the script doesn't save Phase-2 endpoint checkpoints, so the actual weights/hidden-state tensors from this run can't be inspected after the fact) — it is a strong, well-reasoned suspicion from the data available, not a proven diagnosis. Treat the 2e-3/5e-3 `no_recovery` verdicts as **an open, unconfirmed instrumentation artifact**, not as evidence that genuine, unrecoverable information loss exists at high severity. A follow-up with an explicit `torch.isnan(hidden).any()` check added at each Phase-2 tracking step would resolve this definitively; not yet run.

### What this run does and doesn't establish

**Trustworthy**: the 1e-4→1e-3 progression (recovers in 55, 55, 55, then 200 steps) is real data, unaffected by the NaN concern (each of those conditions shows normal, varying, sensible recovery curves). Recovery gets *harder* as displacement increases — a real, monotonic relationship, not previously demonstrated.

**Not yet established**: whether a genuine (non-artifactual) `no_recovery` regime exists at all. The two data points that would show this are exactly the two suspected of being corrupted. The severity sweep needs to be re-run with smaller LR increments between 1e-3 and 2e-3 (where recovery is still just barely working) and with a NaN diagnostic added, before any claim about a true information-loss regime can be made.

Raw files: `h2_probe_recovery_severity_gpt2/h2_probe_recovery_severity/gpt2/sst2_to_cola/lr*_rank8_result.json`, `sst2_probe_summary.json`, `probe_recovery.png`, `severity_sweep_gpt2.log`.

---

## H2: Fresh-Probe Recovery — Llama-3.2-3B (2026-07-23)

**Config**: `h2_probe_recovery.py --model meta-llama/Llama-3.2-3B --task_a sst2 --task_b mnli agnews cola --phase1_dir h2_llama/sst2_to_mnli --phase2_lrs 5e-4 1e-3 --phase2_rank 8`. Reuses the existing, authoritative Llama SST-2 Phase-1 checkpoint from `h2_sequential_llama.py` (`acc_phase1_a=0.96`) directly — no retraining, confirmed compatible without any code changes (same `rank`/`alpha`/target-module conventions in both scripts' shared `add_lora` calls). Two LRs per target because Llama's per-target forgetting thresholds differ too widely for one shared LR to be meaningful: MNLI stays safe under the old classifier even at 5e-4 (only forgets by 1e-3, per the existing H2-C table), while AGNews/CoLA already forget by 5e-4.

### Result table (verified directly from each condition's `result.json`)

| task_b | lr | old_frozen_acc | threshold (0.9×0.96=0.864) | final_probe_acc | verdict |
|---|---|---|---|---|---|
| mnli | 5e-4 | 0.5220 | 0.8640 | 0.9140 | recovers_slow |
| mnli | 1e-3 | 0.4780 | 0.8640 | 0.5220 | no_recovery |
| agnews | 5e-4 | 0.6460 | 0.8640 | 0.8260 | no_recovery |
| agnews | 1e-3 | 0.4780 | 0.8640 | 0.5220 | no_recovery |
| cola | 5e-4 | 0.4740 | 0.8640 | 0.7540 | no_recovery |
| cola | 1e-3 | 0.4780 | 0.8640 | 0.4780 | no_recovery |

**The categorical verdict alone is misleading here and should not be read without the raw curves — five of six conditions land in the same `no_recovery` bucket, but they are two qualitatively different phenomena, not one.**

### Group 1 (lr=5e-4): real, substantial recovery — just short of a very high bar

Llama's baseline (0.96) sets the 90%-threshold at 0.864 — a much stricter bar than GPT-2's (0.774 baseline → 0.6966 threshold). Examining the full recovery curves (not just the endpoint):

- **agnews, lr=5e-4**: climbs from 0.480 to a peak of 0.902 by late training (range 0.42), with several points exceeding 0.864 (0.886, 0.888, 0.902, 0.898...) but not *sustained* through the final step (last point: 0.826). This is genuine, substantial, noisy-but-real learning — a near-miss on the sustained-crossing criterion, not a failure to learn.
- **cola, lr=5e-4**: climbs from 0.478 to 0.754 (range 0.276) with a clear, if noisy, upward trend across the full 200 steps — real partial recovery, same qualitative shape as GPT-2's successful recovery curves, just not reaching Llama's much higher absolute bar within budget.

Both curves look qualitatively identical in shape to the GPT-2 conditions that were classified `recovers_slow` — the only reason these two are labeled `no_recovery` is the much higher threshold Llama's baseline imposes, not any qualitative difference in what's happening to the representation.

### Group 2 (lr=1e-3, all three targets): a reproducible collapse to a near-chance, non-learning state

At the higher LR, mnli, agnews, and cola all show the same signature: the recovery curve oscillates between only two nearby values (typically ~0.46–0.54, near SST-2's 2-class chance) for the entire 200-step budget, with **no upward trend at any point** — qualitatively distinct from Group 1's curves.

```
mnli/1e-3:   ['0.478','0.522','0.478','0.478','0.522', ... ] (41 points, range=0.044, never trends up)
agnews/1e-3: ['0.522','0.522','0.478','0.522','0.522', ... ] (41 points, range=0.066, never trends up)
cola/1e-3:   ['0.464','0.464','0.522','0.464','0.478', ... ] (41 points, range=0.078, never trends up)
```

**This is not a fresh, isolated finding — the exact `0.478` value independently reproduces a number already on record in this project from a completely different script and time.** The existing SST-2→MNLI (r8) fine-tuning table (`h2_sequential_llama.py`, months earlier) reports `final_acc_task_a=0.478` at lr=1e-3 exactly (line 371 of this document: `lr1e-03_rank8_result.json | 1e-3 | 8 | 2.840747e-3 | 0.238334 | 0.478 | 0.356`). Two unrelated scripts, run at different times, land on the identical value for the identical (model, Phase-1 task, Phase-2 target, LR, rank) configuration — this is real corroboration that "SST-2 Phase 1 → MNLI Phase 2 at lr=1e-3 on Llama-3.2-3B, rank 8" reliably collapses to this specific near-chance state, not a one-off artifact of this particular run.

**What remains genuinely unresolved is the mechanism, not the fact of the collapse.** Two explanations are both consistent with the data, and this experiment cannot cleanly distinguish them:
- **Training-numerical instability** (the same class of concern already flagged for GPT-2's 2e-3/5e-3 conditions) — an overly aggressive LR corrupting weights/hidden-states partway through Phase-2 training.
- **Genuine representational collapse** — the model's hidden states have actually degenerated toward a near-constant or very-low-rank state at this severity, such that no linear probe (old or fresh) can extract more than chance-level signal, regardless of training budget.

The evidence here is weaker for the NaN-specific diagnosis than GPT-2's case: unlike GPT-2's 2e-3/5e-3 conditions (where the old classifier and an untrained fresh probe agreed to 4 decimal places in *every* corrupted condition — the clean NaN-argmax signature), only `mnli/1e-3` shows that exact agreement (old_frozen_acc=0.478=fresh-probe-step-0); `agnews/1e-3` and `cola/1e-3`'s untrained-probe values (0.480, 0.464) are close to but not identical to their own old_frozen_acc (0.478 both) — consistent with either explanation, not decisive for either.

### Interpretation

Llama gives a genuinely more mixed picture than GPT-2's cleaner "always eventually recovers" result. At moderate severity (lr=5e-4), the pattern matches GPT-2 closely — real, substantial, readout-mismatch-consistent recovery, just measured against a much stricter bar. At higher severity (lr=1e-3), all three targets collapse into the same reproducible near-chance state — independently corroborated by pre-existing project data, but with an unresolved mechanism (instability artifact vs. genuine collapse). Unlike GPT-2's severity sweep (where the ambiguous conditions were the two *most* extreme, novel LRs never tested before), here the ambiguous regime is a LR (1e-3) already used elsewhere in this project's core H2 tables — meaning if this is a genuine representational collapse rather than an artifact, it would also implicate some of the *existing*, already-cited Llama H2 accuracy numbers at this LR as sitting in the same degenerate regime, not just this new probe-recovery experiment.

**Caveats**: single run, single seed, no seed replication. Only 2 LR points per target (not a fine-grained sweep) — the true transition between "real partial recovery" and "collapse to near-chance" for each target is not localized, just bracketed between 5e-4 and 1e-3. `h2_probe_recovery.py` does not save Phase-2 endpoint checkpoints, so direct tensor inspection (to settle the instability-vs-collapse question) is not possible after the fact without a rerun.

Raw files: `h2_probe_recovery_llama/h2_probe_recovery_llama/meta-llama_Llama-3.2-3B/{sst2_to_mnli,sst2_to_agnews,sst2_to_cola}/lr*_rank8_result.json`, `sst2_probe_summary.json`, `probe_recovery.png`, `probe_recovery_llama.log`.

---

## H2: Fresh-Probe Recovery — Llama-3.2-3B Severity Sweep + NaN Diagnostic (2026-07-30)

**Motivation**: the run above tested only 2 LR points (5e-4, 1e-3) per target and left the mechanism of the lr=1e-3 collapse unresolved (training instability vs. genuine representational collapse). `h2_probe_recovery.py` extended with a NaN/Inf diagnostic (`_hidden_has_nan`, checks the frozen backbone's actual hidden states on real batches after Phase-2 training and again after probe recovery) and rerun with two intermediate LR points (6e-4, 8e-4) and a doubled probe-training budget (400 steps, up from 200) to also check whether the earlier lr=5e-4 "no_recovery" verdicts for agnews/cola were genuinely final or just budget-limited. Config: `--phase2_lrs 5e-4 6e-4 8e-4 1e-3 --probe_steps 400 --force`, same Phase-1 checkpoint reused (no retraining).

### Result table (verified from each condition's `result.json`)

| task_b | lr | old_frozen_acc | final_probe_acc (400 steps) | phase2_hidden_nan | probe_recovery_hidden_nan | verdict |
|---|---|---|---|---|---|---|
| mnli | 5e-4 | 0.522 | 0.918 | False | False | recovers_slow |
| mnli | 6e-4 | 0.596 | 0.916 | False | False | recovers_slow |
| mnli | 8e-4 | 0.478 | 0.478 | False | False | no_recovery |
| mnli | 1e-3 | 0.478 | 0.478 | False | False | no_recovery |
| agnews | 5e-4 | 0.646 | 0.814 | False | False | no_recovery |
| agnews | 6e-4 | 0.478 | 0.522 | False | False | no_recovery |
| agnews | 8e-4 | 0.470 | 0.840 | False | False | no_recovery |
| agnews | 1e-3 | 0.478 | 0.522 | False | False | no_recovery |
| cola | 5e-4 | 0.474 | 0.738 | False | False | no_recovery |
| cola | 6e-4 | 0.522 | 0.522 | False | False | no_recovery |
| cola | 8e-4 | 0.478 | 0.522 | False | False | no_recovery |
| cola | 1e-3 | 0.478 | 0.462 | False | False | no_recovery |

### Finding 1 — the NaN/training-instability hypothesis is now definitively ruled out

**`phase2_hidden_nan` and `probe_recovery_hidden_nan` are both `False` in all 12 conditions, including every single plateaued one.** The frozen backbone's hidden states are finite (no NaN/Inf) in every case checked — including mnli/agnews/cola at lr=1e-3, the exact conditions whose 0.478 collapse independently reproduces a value already on record in this project from `h2_sequential_llama.py` months earlier (see the previous Llama fresh-probe-recovery section above). This rules out the specific "training diverged into NaN weights" mechanism proposed as an open alternative there. It does not, by itself, prove the strongest form of "genuine information loss" (a finite-but-degenerate, e.g. near-constant or very-low-variance, representation is also consistent with these results and hasn't been separately ruled out) — but the crude numerical-artifact explanation is now closed off.

### Finding 2 — AGNews's collapse is genuinely non-monotonic in LR, not just harder-to-recover with severity

The clean "recovery gets monotonically harder as LR increases" story (established for GPT-2's cola severity sweep) does **not** hold for Llama AGNews: 5e-4 shows real partial recovery (peaks near 0.90, ends at 0.814), 6e-4 collapses to a flat, non-learning chance-level oscillation (full curve range 0.44–0.53, no trend across 400 steps), **8e-4 recovers again** — a real, sustained climb to a stable ~0.78–0.84 plateau by step 30 that holds for the remaining 370 steps (min 0.484, max 0.842, structurally identical in shape to the genuine partial-recovery curves, not a fluke single point) — then 1e-3 collapses again to the same flat chance-level signature as 6e-4. Both "recovers" and "collapses" states were verified by inspecting the full 81-point recovery curve for each condition, not just the endpoint. MNLI, by contrast, shows a clean, sharp transition (recovers fully at both 5e-4 and 6e-4, collapses completely at both 8e-4 and 1e-3 — the boundary sits somewhere in (6e-4, 8e-4], not localized further since no point was tested in between).

### Interpretation

Two real, previously-unavailable findings: (1) Llama's high-severity collapse is not a training-numerics artifact — the representation genuinely stops being linearly decodable by any tested probe, a real (if not yet fully characterized) phenomenon, not a bug to fix and rerun past. (2) Recovery difficulty is not a simple monotonic function of displacement/LR for every Phase-2 target — AGNews's non-monotonic pattern suggests different LRs trace genuinely different paths through weight space (not just different distances along one path), consistent with the same "direction matters, not just magnitude" finding already established elsewhere in this project (H1's direct-R sweep, the SST-2 anisotropy findings in H2-I). This complicates any claim that R_A alone (a magnitude, not a direction) should be expected to predict recovery difficulty — R_A predicts *whether the old classifier breaks* reasonably well, but not, on this evidence, *whether a fresh probe can fix it*.

**Caveats**: single run, single seed, no seed replication — AGNews's non-monotonicity in particular should be treated as a single-seed finding until replicated with a different data/probe seed, since it's exactly the kind of pattern that could in principle be seed-sensitive rather than a stable property of the LR itself. The NaN check only confirms finiteness on 2 real batches per condition, not exhaustively over the full validation set. Not yet extended beyond lr=1e-3 on Llama (GPT-2's severity sweep went to 5e-3; Llama has not been pushed that far).

**⚠️ MAJOR CAVEAT, found 2026-07-30 (same day, before the above was acted on further) — "genuine, finite-valued representational phenomenon" above is premature.** Tallying every accuracy value across all 27 locally-held probe-recovery result files (GPT-2 + Llama, both runs): **0.5220 and 0.4780 — which sum to exactly 1.000, and exactly equal 261/500 and 239/500 on the fixed 500-example SST-2 validation split (`seed=42`, identical across every run)** — are by far the two most common values recorded (372 and 295 occurrences respectively, an order of magnitude above any other value). Restricting to post-step-0 evaluations (step=0 landing here is mundane — an untrained random probe defaulting near-chance is unremarkable): 631 of the ~1560 post-step-0 evaluations land exactly on one of these two values, and **11 of the 27 conditions'** *final* recorded value is exactly 0.478 or 0.522 — including every single Llama `no_recovery` condition in both runs, and GPT-2's lr=2e-3/5e-3 (previously flagged as likely NaN, never actually checked for NaN since that diagnostic postdates the GPT-2 run).

This is the exact signature of a classifier — old or fresh — converging to **always predicting the same fixed class**, which mechanically produces one of these two accuracies regardless of what's actually happening in the hidden states, since the accuracy is then determined purely by the fixed validation split's class balance. This is consistent with two different explanations not yet distinguished by anything collected so far: (a) genuine collapse, *correctly detected* — if there truly is no recoverable signal, converging to majority-class prediction is the loss-minimizing, expected behavior of a working optimizer, not a bug; or (b) the probe's own optimization (fixed `probe_lr`, fixed step budget) getting stuck in a degenerate local minimum despite real, recoverable signal being present in the hidden states — a genuine artifact unrelated to the representation itself. One point favors (a) over a *blanket* version of (b): the identical hyperparameters demonstrably learn well past 0.478/0.522 in every condition that shows real recovery (mnli 5e-4/6e-4 → 0.916–0.918; agnews 8e-4 → 0.840; cola 5e-4 → 0.738) — so the optimizer is not broadly incapable of learning. But this doesn't rule out (b) specifically for the stuck conditions, where a different hidden-state scale/geometry could still defeat the same nominal hyperparameters.

**This was not verifiable from data already collected** — the script only recorded aggregate accuracy at each step, never the actual predicted-class distribution. Fixed 2026-07-30: `h2_probe_recovery.py` now includes `_prediction_class_counts` (reports the literal predicted-class histogram for both the old classifier and the final-step fresh probe), wired into `train_phase2`/`run_probe_recovery` and saved as `old_classifier_class_counts`/`final_probe_class_counts` in every future result JSON. **Not yet run with this diagnostic** — none of the numbers/findings above have been re-verified against actual prediction distributions. Until that rerun happens, "confirmed real, finite-valued phenomenon, not a bug" should be read as **"NaN specifically is ruled out; constant-class-collapse (in either sense (a) or (b) above) is not yet distinguished from a probe-training artifact."** This also means the AGNews non-monotonicity finding (Finding 2) needs the same check *and* seed replication before being trusted as a genuine representational effect — agnews/6e-4's collapse value (0.522) is exactly the suspicious constant, and if that specific run's probe fell into this trap by chance rather than by correctly detecting no signal, the whole "recovers→collapses→recovers→collapses" story could be partly or wholly probe-training noise rather than a real effect of direction-not-magnitude.

Raw files: `h2_probe_recovery_llama_severity/h2_probe_recovery_llama/meta-llama_Llama-3.2-3B/{sst2_to_mnli,sst2_to_agnews,sst2_to_cola}/lr{5e-04,6e-04,8e-04,1e-03}_rank8_result.json`, `sst2_probe_summary.json`, `probe_recovery.png`, `llama_severity_sweep.log`.

---

## H2: Fresh-Probe Recovery Generality Test — MNLI as Phase-1 Task (2026-07-30/31)

**Motivation**: every fresh-probe-recovery experiment so far (GPT-2 single-point, GPT-2 severity sweep, Llama severity sweep) used SST-2 as Phase 1 — because SST-2 was the identified anomaly (V8: 44× T* spread across Phase-2 targets). This left a genuinely bigger, previously-untested question open: does the same readout-mismatch pattern (frozen classifier breaks, fresh probe recovers) hold for Phase-1 tasks that *weren't* anomalous — MNLI, AGNews, CoLA — whose H2 transfer results already looked clean? If it's general rather than SST-2-specific, it reframes what every T* in H2's core tables actually measures, not just the SST-2 footnote.

### GPT-2 MNLI Phase-1 training failure (3 attempts, before pivoting to Llama)

No GPT-2 MNLI Phase-1 checkpoint existed (GPT-2 has only ever been used with SST-2 as Task A). Attempted to train one via `h2_nll_forgetting.py`'s `run_phase1_generic`:

| Attempt | lr | rank | steps | acc_phase1 |
|---|---|---|---|---|
| 1 | 2e-4 | 8 | 3000 | 0.352 |
| 2 | 5e-4 | 8 | 3000 | 0.354 |
| 3 | 2e-4 | 32 | 3000 | 0.340 |

All three near MNLI's 3-way chance level (0.333), with training loss flat at 1.05–1.18 (≈ `ln(3)=1.0986`, exactly the loss of random 3-class guessing) for the entire run in every attempt — not slowly converging, never moving at all. A 2.5× LR change and a 4× rank change both had no effect, ruling out both as the cause. Directly checked and ruled out a tokenizer-truncation hypothesis (GPT-2's tokenizer cutting the `hypothesis:` portion off the formatted `"premise: X hypothesis: Y"` string before the model ever sees it): on 200 sampled MNLI train examples, mean tokenized length is 41.2, only 1/200 exceeds the 128-token limit, and even that one example retains the `hypothesis:` marker after truncation. With LR, rank, and truncation all ruled out, and the identical shared `load_task`/training code already validated on Llama (88.4% MNLI accuracy), the most likely explanation is a genuine base-model capacity limitation — GPT-2 (117M, 12 layers, 768-dim) may lack the representational structure to support 3-way natural language inference via last-token pooling + rank-8/32 LoRA, regardless of tuning. Abandoned further GPT-2 MNLI attempts; pivoted to Llama's existing, already-validated MNLI Phase-1 checkpoint (`h2_llama/mnli_to_sst2/`, `acc_phase1=0.884`, no retraining needed).

### Progressive probe-budget escalation, Llama MNLI → {sst2, agnews, cola}, lr=5e-4

Same script (`h2_probe_recovery.py`, now with the `_prediction_class_counts` diagnostic added — see previous section), same reused Phase-1 checkpoint throughout. Budget escalated across 4 rounds because each successive round's curves were still climbing, not plateaued — the stopping criterion (this project's standing convention) is a genuinely flat/plateaued curve, not just reaching a step-count target.

| Round | probe_steps | mnli→sst2 final_probe_acc | mnli→agnews final_probe_acc | mnli→cola final_probe_acc |
|---|---|---|---|---|
| v1 | 200 | 0.590 | 0.480 | 0.354 *(also compromised — see below)* |
| v2 | 400 | 0.644 | 0.502 | 0.312 *(probe collapsed — see below)* |
| v3 | 800 | 0.728 | 0.564 | 0.362 *(re-run, probe_lr lowered to 3e-4 — the only clean cola measurement)* |
| v4 | 2000 | **0.780** (peaked at 0.798) | 0.598 | *(not re-escalated — see below)* |

Threshold for all three: 0.9×0.884 = **0.7956**. `old_frozen_acc` (unchanged across rounds, same Phase-2 checkpoint reused): sst2=0.298, agnews=0.384, cola=0.318 — all three catastrophically collapsed under the old classifier, similar in severity to SST-2's own worst collapses.

### The class-count diagnostic caught a real artifact on its first use — cola's v2 result was a probe-training collapse, not genuine non-recovery

At v2 (400 steps, default `probe_lr=1e-3`), `mnli→cola`'s `final_probe_class_counts = {'1': 496, '2': 4}` — the fresh probe predicted class 1 on 99.2% of examples, the exact signature of degenerate constant-class collapse (not inferred from a suspicious accuracy value this time, as with the earlier SST-2 case — directly observed in the prediction distribution). Its curve was also genuinely flat (mean of last 10 points ≈ mean of middle 10, no climb: 0.336 vs. 0.340). Re-run at v3 with `probe_lr=3e-4` (same 800-step budget): `final_probe_class_counts = {'2': 256, '1': 232, '0': 12}` — collapse resolved, predictions now spread across classes. This confirms the earlier v2 result was a probe-optimization artifact tied to the learning rate, not evidence about the representation — exactly the failure mode the diagnostic was built to catch, on its very first real use.

### Where the curves stand, round by round (verified via full-curve inspection each time, not just endpoints)

**`mnli→sst2`**: v2 (400 steps) clearly still climbing (last-10 mean 0.710 > mid-10 mean 0.657). v3 (800 steps) still climbing (0.722 > 0.704). v4 (2000 steps): trend has visibly slowed — quarter-by-quarter means 0.702 → 0.734 → 0.746, much smaller increments than earlier rounds — consistent with approaching an asymptote near the threshold rather than still climbing steadily. **Peaked at 0.798, just 0.002 short of the 0.7956 threshold**, oscillating in a 0.70–0.80 band without ever sustaining above it for the required consecutive-points check (hence `verdict=no_recovery` despite the near-miss). `final_probe_class_counts={'1': 208, '0': 150, '2': 142}` — no collapse, genuine spread.

**`mnli→agnews`**: v2 still climbing (0.505 > 0.465). v3 still climbing (0.561 > 0.521). v4 (2000 steps): quarter-by-quarter means 0.534 → 0.595 → 0.632 — still climbing but increments shrinking (+0.061, then +0.037), suggesting an asymptote somewhat below the threshold, not a clean crossing. `final_probe_class_counts={'0': 89, '1': 276, '2': 135}` — no collapse.

**`mnli→cola`**: v3 (800 steps, post-fix) shows real but very weak, very slow improvement (0.337 → 0.350 over the last-10-vs-mid-10 comparison) — the slowest of the three by a wide margin. **Not re-escalated beyond 800 steps** — given the pace, further budget was judged to have poor cost/information payoff relative to sst2/agnews, which were both closer to a real resolution.

### Decision to stop here

Both sst2 and agnews show the signature of diminishing returns (shrinking per-round increments) rather than imminent resolution at v4. Continuing to escalate risks spending substantially more compute for a shrinking chance of a clean crossing/non-crossing answer. Stopped at this point and treated the graded pattern itself as the reportable finding, rather than forcing a binary recovered/not-recovered verdict.

### Interpretation

**⚠️ Revised 2026-08-02 — "generalizes, at least partially" (below, struck through in spirit if not in text) overstated this result.** The precise, load-bearing fact: **zero of three tested MNLI Phase-2 targets crossed the recovery threshold, at any budget tested** — a real, qualitative difference from SST-2-as-Phase-1, where all three GPT-2 conditions crossed cleanly within 55–200 steps. "Generalizes" risks implying the MNLI case basically worked too, just slower; the more honest claim is that MNLI shows the *same qualitative signature* (old-classifier collapse + non-degenerate, class-count-verified recovery direction) without yet reaching the SST-2-established bar for any condition.

The three targets are not equally solid evidence for even that weaker claim, and treating them as a uniform "graded recovery" trio (as the original version of this section did) obscured real differences in data quality:

- **`sst2`**: the strongest case. Per-round gains (0.054, 0.084, 0.070) show real, if now-decelerating, improvement, and the peak (0.798) missed the threshold (0.7956) by only 0.0024. "Would likely cross with substantially more budget" is a reasonable inference here specifically.
- **`agnews`**: genuine but markedly weaker. Per-round gains (0.022, 0.062, 0.034) are decelerating faster, ending 0.198 below threshold — a much less confident case that more budget would close the gap. **Checked and ruled out a specific alternative explanation**: could the 0.598 accuracy be inflated by the classifier defaulting to its most-predicted class (`1`, predicted 276/500 times) rather than genuine signal? Arithmetic against MNLI's ~balanced validation split (~167 true examples/class) rules this out — even crediting *every* class-1 prediction as correct caps out at 167 of the 299 total correct; the remaining 132 correct predictions must come from the 224 non-majority (class 0/2) predictions, a 58.9% hit rate among those — far above the ~33% ceiling pure majority-class bias could produce on a balanced 3-way task. The recovery here is real signal, just incomplete.
- **`cola`**: **not actually a trend at all.** Re-examining `v1`'s (200-step, default `probe_lr=1e-3`) class counts — `{'2': 192, '0': 306, '1': 2}` — shows the *same shape* of degeneracy later confirmed at `v2` (`{1: 496, 2: 4}`), just less extreme (class 1 predicted on only 0.4% of examples, same failure mode, same default `probe_lr`). This means `v1`'s 0.354 is *also* compromised, not a clean baseline — there is exactly **one** trustworthy cola measurement in this entire investigation (`v3=0.362`, the only round run with the corrected `probe_lr=3e-4`), and no valid before/after comparison exists for it at all. The earlier framing ("real but very weak, very slow improvement, 0.337→0.350") implied a measured trend that does not actually exist in the data.

**Revised bottom line**: MNLI as a Phase-1 task shows the same *qualitative* signature as SST-2 (catastrophic old-classifier collapse, non-degenerate recovery direction rather than a chance-level or NaN plateau) — genuinely useful evidence against "MNLI's forgetting is simply, cleanly genuine, unlike SST-2's." But it is a **partial, unresolved generalization test, not a confirmed extension of readout-mismatch**: no condition reached the established recovery bar, sst2 is the only one close enough to guess it might with more budget, and cola has no real trend data behind it at all.

**A separate, not-yet-closeable limitation**: because GPT-2 could never produce a usable MNLI Phase-1 checkpoint, this entire generality test exists only on Llama — there is no GPT-2/Llama cross-architecture check for this specific question, unlike the original SST-2 investigation which had both.

**Caveats**: single run, single seed throughout — no seed replication on any of the 4 rounds. Stopping point for sst2/agnews (2000 steps) is a judgment call based on shrinking increments, not a proven plateau — a much larger budget could in principle still resolve either one, just at increasing cost for decreasing expected information gain. Cola's true asymptote at any budget is unknown — the one clean measurement available (v3) is a single point, not a curve. GPT-2's Phase-1 training failure (loss genuinely never moving) is itself an unplanned but real finding about base-model capacity limits, not something this investigation set out to find.

Raw files: `h2_probe_recovery_mnli_gpt2/` (failed GPT-2 attempts), `h2_probe_recovery_mnli_llama/`, `h2_probe_recovery_mnli_llama_v2/`, `h2_probe_recovery_mnli_llama_v3/`, `h2_probe_recovery_mnli_llama_v4/` (progressive Llama rounds), corresponding `.log` files for each round.
