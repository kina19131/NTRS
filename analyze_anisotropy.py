"""
analyze_anisotropy.py
=====================
Test the SST-2 basin anisotropy hypothesis.

Key question: Does Phase 2 fine-tuning direction overlap more with SST-2's Phase 1
LoRA subspace for tasks that cause early forgetting (CoLA) vs. resistant ones (MNLI r8)?

Method (efficient — works in rank-r space, never materializes m x n matrices):
  For each task T in {CoLA, AGNews, MNLI}:
    1. Load SST-2 Phase 1 LoRA: B_SST2 in R^{m x r}, A_SST2 in R^{r x n}
    2. Load task T Phase 1 LoRA:  B_T    in R^{m x r}, A_T    in R^{r x n}
       (used as proxy for "direction Phase 2 training would push in")
    3. SVD of B_SST2 -> U_SST2 in R^{m x r} (left singular vecs, orthonormal cols)
    4. B-projection:  ||U_SST2^T @ B_T||_F / ||B_T||_F
       Measures: what fraction of task T's output feature directions overlap with SST-2?
    5. SVD of A_SST2 -> Vh_SST2 in R^{r x n}, V_SST2 = Vh_SST2^T in R^{n x r}
       A-projection:  ||A_T @ V_SST2||_F / ||A_T||_F
       Measures: what fraction of task T's input readout directions overlap with SST-2?

Prediction if anisotropy holds (CoLA most fragile, MNLI r8 most resistant):
  proj(CoLA -> SST-2) > proj(AGNews -> SST-2) > proj(MNLI -> SST-2)

T* reference (SST-2 Phase 1, Phase 2 task):
  CoLA:   T* ~ 0.006   (most fragile)
  AGNews: T* ~ 0.016   (intermediate)
  MNLI:   T* ~ 0.122   (most resistant at r32; r8 is ~0.53)
"""

import torch
import numpy as np
import json

CHECKPOINTS = {
    "SST-2": "h2_llama/sst2_to_mnli/phase1_model.pt",
    "CoLA":  "h2_llama/cola_to_mnli/phase1_model.pt",
    "AGNews":"h2_llama/agnews_to_mnli/phase1_model.pt",
    "MNLI":  "h2_llama/mnli_to_sst2/phase1_model.pt",
}

T_STAR_SST2_PHASE2 = {
    "CoLA":   0.0061,
    "AGNews": 0.0162,
    "MNLI":   0.1223,
}


def load_lora_BA(ckpt_path: str) -> dict:
    """Load LoRA state dict. Returns dict: layer_key -> (B, A) tensors."""
    sd = torch.load(ckpt_path, map_location="cpu")
    layers = {}
    for k, v in sd.items():
        if "lora_A" in k:
            base = k.replace(".lora_A.default.weight", "")
            layers.setdefault(base, {})["A"] = v.float()  # (r, n)
        elif "lora_B" in k:
            base = k.replace(".lora_B.default.weight", "")
            layers.setdefault(base, {})["B"] = v.float()  # (m, r)
    return {k: (v["B"], v["A"]) for k, v in layers.items() if "B" in v and "A" in v}


def subspace_projections(B_ref: torch.Tensor, A_ref: torch.Tensor,
                          B_q: torch.Tensor,   A_q: torch.Tensor):
    """
    Compute two projection fractions, all in rank-r space.

    B-projection (output feature dirs):
      U_ref = left sing vecs of B_ref (m x r orthonormal)
      frac_B = ||U_ref^T @ B_q||_F / ||B_q||_F

    A-projection (input readout dirs):
      Vh_ref from SVD of A_ref -> V_ref = Vh_ref^T in R^{n x r}
      frac_A = ||A_q @ V_ref||_F / ||A_q||_F

    Both numerators are r x r matrices -> cheap.
    """
    # B-projection
    U_ref, _, _ = torch.linalg.svd(B_ref, full_matrices=False)  # U_ref: (m, r)
    proj_B = U_ref.T @ B_q                                        # (r, r)
    frac_B = torch.norm(proj_B, p="fro").item() / (torch.norm(B_q, p="fro").item() + 1e-12)

    # A-projection
    _, _, Vh_ref = torch.linalg.svd(A_ref, full_matrices=False)  # Vh_ref: (r, n)
    V_ref = Vh_ref.T                                               # (n, r)
    proj_A = A_q @ V_ref                                           # (r, r)
    frac_A = torch.norm(proj_A, p="fro").item() / (torch.norm(A_q, p="fro").item() + 1e-12)

    return frac_B, frac_A


def mean_proj(data_ref: dict, data_q: dict, ref_keys: list):
    """Average B- and A-projections over all layers."""
    fb_list, fa_list = [], []
    for k in ref_keys:
        B_ref, A_ref = data_ref[k]
        B_q,   A_q   = data_q[k]
        fb, fa = subspace_projections(B_ref, A_ref, B_q, A_q)
        fb_list.append(fb)
        fa_list.append(fa)
    return float(np.mean(fb_list)), float(np.mean(fa_list)), fb_list, fa_list


def main():
    print("Loading Phase 1 LoRA checkpoints...")
    all_data = {}
    for task, path in CHECKPOINTS.items():
        all_data[task] = load_lora_BA(path)
        print(f"  {task}: {len(all_data[task])} layers")

    ref_keys = sorted(all_data["SST-2"].keys())
    for task in all_data:
        assert sorted(all_data[task].keys()) == ref_keys, f"Layer mismatch: {task}"
    print(f"All tasks: {len(ref_keys)} LoRA layers\n")

    tasks_q = ["CoLA", "AGNews", "MNLI"]

    # ── Main test: project each task onto SST-2 subspace ──────────────────────
    print("=" * 62)
    print("Projection onto SST-2 Phase 1 LoRA subspace")
    print("Prediction: CoLA > AGNews > MNLI if anisotropy holds")
    print("=" * 62)
    print(f"  {'Task':8s}  {'frac_B (output)':16s}  {'frac_A (input)':16s}  {'T*':8s}")

    results = {}
    for task in tasks_q:
        fb, fa, fb_list, fa_list = mean_proj(all_data["SST-2"], all_data[task], ref_keys)
        results[task] = {"frac_B": fb, "frac_A": fa,
                          "frac_B_per_layer": fb_list, "frac_A_per_layer": fa_list}
        t = T_STAR_SST2_PHASE2[task]
        print(f"  {task:8s}  {fb:.4f}             {fa:.4f}             {t:.4f}")

    # Sanity: SST-2 onto itself = 1.0
    fb_self, fa_self, _, _ = mean_proj(all_data["SST-2"], all_data["SST-2"], ref_keys)
    print(f"  {'SST-2':8s}  {fb_self:.4f}             {fa_self:.4f}             (self, should be 1.0)")

    # ── Pairwise matrix (rows=reference, cols=query) ───────────────────────────
    tasks_all = ["SST-2", "CoLA", "AGNews", "MNLI"]
    print("\nPairwise B-projection matrix (rows=reference subspace, cols=query task)")
    print(f"  {'':8s}", end="")
    for t in tasks_all:
        print(f"  {t:8s}", end="")
    print()

    pairwise_B = {}
    for row in tasks_all:
        pairwise_B[row] = {}
        print(f"  {row:8s}", end="")
        for col in tasks_all:
            fb, _, _, _ = mean_proj(all_data[row], all_data[col], ref_keys)
            pairwise_B[row][col] = fb
            print(f"  {fb:.4f}  ", end="")
        print()

    # ── Layer-level: where is the CoLA-MNLI gap largest? ─────────────────────
    print("\n" + "=" * 62)
    print("Top 10 layers by CoLA-MNLI B-projection gap (onto SST-2 subspace)")
    print("=" * 62)
    gaps = []
    fb_cola = results["CoLA"]["frac_B_per_layer"]
    fb_mnli = results["MNLI"]["frac_B_per_layer"]
    for i, key in enumerate(ref_keys):
        gap = fb_cola[i] - fb_mnli[i]
        short = key.split("layers.")[-1] if "layers." in key else key
        gaps.append((short, gap, fb_cola[i], fb_mnli[i]))

    gaps.sort(key=lambda x: x[1], reverse=True)
    for name, gap, c, m in gaps[:10]:
        print(f"  {name:35s}  CoLA={c:.4f}  MNLI={m:.4f}  gap={gap:+.4f}")

    # ── Correlation summary ────────────────────────────────────────────────────
    print("\n" + "=" * 62)
    print("Summary: projection onto SST-2 subspace vs. T* (Phase 2 task fragility)")
    print("Hypothesis: higher proj -> lower T* (more fragile) if anisotropy")
    print("=" * 62)
    print(f"  {'Task':8s}  {'frac_B':8s}  {'frac_A':8s}  {'T*':8s}  {'1/T*':8s}")
    for task in tasks_q:
        t = T_STAR_SST2_PHASE2[task]
        fb = results[task]["frac_B"]
        fa = results[task]["frac_A"]
        print(f"  {task:8s}  {fb:.4f}    {fa:.4f}    {t:.4f}    {1/t:.1f}x")

    projs_B = [results[t]["frac_B"] for t in tasks_q]
    t_stars  = [T_STAR_SST2_PHASE2[t] for t in tasks_q]
    corr = np.corrcoef(projs_B, [-t for t in t_stars])[0, 1]
    print(f"\n  Pearson r(frac_B, -T*) = {corr:.4f}  (positive = hypothesis supported)")

    # ── Save ──────────────────────────────────────────────────────────────────
    out = {
        "description": "SST-2 basin anisotropy: projection of Phase 2 task LoRA onto SST-2 Phase 1 subspace",
        "projection_onto_sst2_subspace": {
            t: {"frac_B": results[t]["frac_B"], "frac_A": results[t]["frac_A"]}
            for t in tasks_q
        },
        "pairwise_B_projection": pairwise_B,
        "t_star_sst2_phase2": T_STAR_SST2_PHASE2,
        "pearson_frac_B_vs_neg_tstar": float(corr),
    }
    with open("direction_overlap_results_subspace.json", "w") as f:
        json.dump(out, f, indent=2)
    print("\nSaved: direction_overlap_results_subspace.json")


if __name__ == "__main__":
    main()
