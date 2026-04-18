"""
Experiment v2: Does more pretraining create a more structured (lower-dimensional) thicket?

Builds on v1 (thicket vs needle) by sweeping pretraining iterations.
The key new question: does dims_90 for top-K decrease monotonically as pretraining increases?

If yes -> pretraining is the causal mechanism behind subspace concentration
       -> this is the toy-scale proof that motivates the LLM experiment
       -> directly parallels the Neural Thickets density scaling law

Structure:
  Experiment 1 (from v1): thicket vs needle, single snapshot
  Experiment 2 (new):     sweep pretraining iters [0, 50, 100, 250, 500, 1000]
                          measure dims_90(top-K), dims_90(all), best_loss at each
  Experiment 3 (new):     summary figure — the "scaling law" plot for subspace dim
"""

import copy
import random
import time
import numpy as np
import torch
import torch.nn as nn
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ── reproducibility ───────────────────────────────────────────────────────────
GLOBAL_SEED = 42
random.seed(GLOBAL_SEED)
np.random.seed(GLOBAL_SEED)
torch.manual_seed(GLOBAL_SEED)
device = torch.device("cpu")

# ── constants ─────────────────────────────────────────────────────────────────
FREQ, SCALE = 4.0, 1.0
CTX_SZ, FUT_SZ = 10, 60
N_PERTURBATIONS = 200
K_SELECT = 20
DATASET_SZ = 64
N_COMPONENTS = 40  # PCA components to compute

# ── model ─────────────────────────────────────────────────────────────────────
class Net(nn.Module):
    def __init__(self, width=64, depth=3, dim_in=CTX_SZ, dim_out=1, device=device):
        super().__init__()
        layers = [nn.Linear(dim_in, width, device=device)]
        for _ in range(depth - 2):
            layers.extend([nn.ReLU(), nn.Linear(width, width, device=device)])
        layers.extend([nn.ReLU(), nn.Linear(width, dim_out, device=device)])
        self.layers = nn.ModuleList(layers)

    def forward(self, ctx):
        was_1d = ctx.dim() == 1
        if was_1d:
            ctx = ctx.unsqueeze(0)
        h = ctx
        for layer in self.layers:
            h = layer(h)
        if was_1d:
            h = h.squeeze(0)
        return h.squeeze(-1)

    def compute_loss(self, ctx, y):
        return nn.MSELoss()(self.forward(ctx), y.squeeze(-1))

    def perturb_weights(self, seed, sigma):
        torch.manual_seed(seed)
        for p in self.parameters():
            p.data.add_(torch.randn_like(p.data) * sigma)

    def AR_rollout(self, ctx, T):
        y_preds = []
        for t in range(T):
            y_pred = self.forward(ctx)
            ctx = torch.cat([ctx, y_pred.unsqueeze(-1)], dim=1)[:, 1:]
            y_preds.append(y_pred)
        return torch.stack(y_preds, dim=1)

    def get_flat_weights(self):
        return torch.cat([p.data.flatten() for p in self.parameters()])

    def xavier_init(self):
        for layer in self.layers:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                nn.init.zeros_(layer.bias)

# ── data generators ───────────────────────────────────────────────────────────
def _make_sinusoid():
    phase  = np.random.uniform(0, 2*np.pi)
    amp    = np.random.uniform(0.8, 1.2)
    offset = np.random.uniform(-0.5, 0.5)
    return lambda x: amp * np.sin(FREQ * np.asarray(x) + phase) + offset

def _make_squarewave():
    phase  = np.random.uniform(0, 2*np.pi)
    amp    = np.random.uniform(0.2, 0.4)
    sharp  = np.random.uniform(4, 6)
    offset = np.random.uniform(-0.5, 0.5)
    return lambda x: amp * np.tanh(sharp * np.sin(FREQ * np.asarray(x) + phase)) + offset

def _make_line():
    slope     = np.random.uniform(-0.5, 0.5)
    intercept = np.random.uniform(-1, 1)
    return lambda x: slope * np.asarray(x) + intercept

def generate_mixed():
    return np.random.choice([_make_sinusoid, _make_squarewave, _make_line])()

def generate_one_sinusoid():
    return lambda x: 0.5 * np.sin(FREQ * np.asarray(x))

def load_data(bsz, generator_fn, ctx_sz=CTX_SZ, fut_sz=FUT_SZ):
    ctx_y_list, fut_y_list = [], []
    for _ in range(bsz):
        gt_fn  = generator_fn()
        x_vals = -2.5 + np.arange(ctx_sz + fut_sz) * 0.1
        y_vals = [float(gt_fn(x)) for x in x_vals]
        ctx_y_list.append(y_vals[:ctx_sz])
        fut_y_list.append(y_vals[ctx_sz:])
    ctx_y = torch.tensor(ctx_y_list, dtype=torch.float32, device=device)
    fut_y = torch.tensor(fut_y_list, dtype=torch.float32, device=device)
    return ctx_y, fut_y

# ── pretraining ───────────────────────────────────────────────────────────────
def pretrain(model, iters=500, bsz=128, lr=0.001, verbose=True):
    """Train on mixed signals. Returns model trained for exactly `iters` steps."""
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    for i in range(iters):
        model.train()
        opt.zero_grad()
        ctx_y, fut_y = load_data(bsz, generate_mixed)
        loss = model.compute_loss(ctx_y, fut_y[:, [0]])
        loss.backward()
        opt.step()
    if verbose:
        print(f"    pretrained {iters} iters, final loss={loss.item():.4f}")
    return model

# ── RandOpt with delta collection ─────────────────────────────────────────────
def randopt_collect(base_model, sigma, N=N_PERTURBATIONS, K=K_SELECT):
    """
    Returns:
      top_k_indices : indices into all_deltas of top-K models
      all_deltas    : (N, D) float32 — perturbation vectors
      all_losses    : (N,)   float32 — loss per perturbation
      base_loss     : scalar — loss of unperturbed model
    """
    ctx_y, fut_y = load_data(DATASET_SZ, generate_one_sinusoid)

    base_weights = base_model.get_flat_weights().numpy().copy()
    D = len(base_weights)
    all_deltas = np.zeros((N, D), dtype=np.float32)
    all_losses = np.zeros(N, dtype=np.float32)

    # base model loss
    base_model.eval()
    with torch.no_grad():
        base_pred = base_model.AR_rollout(ctx_y, FUT_SZ)
        base_loss = ((base_pred - fut_y)**2).sum(dim=1).mean().item()

    for seed in range(N):
        m = copy.deepcopy(base_model)
        m.perturb_weights(seed, sigma)
        m.eval()
        delta = m.get_flat_weights().numpy() - base_weights
        all_deltas[seed] = delta
        with torch.no_grad():
            pred = m.AR_rollout(ctx_y, FUT_SZ)
            loss = ((pred - fut_y)**2).sum(dim=1).mean().item()
        all_losses[seed] = loss
        del m

    ranked = np.argsort(all_losses)
    top_k_indices = ranked[:K].tolist()
    return top_k_indices, all_deltas, all_losses, base_loss

# ── PCA metrics ───────────────────────────────────────────────────────────────
def pca_dims_for_threshold(vecs, threshold=0.90, n_components=None):
    """
    Return the number of PCA dims needed to explain `threshold` fraction
    of variance in `vecs` (shape: N x D).
    Also returns the full cumulative variance curve.
    """
    if len(vecs) < 2:
        return 1, np.array([1.0])
    centered = vecs - vecs.mean(axis=0)
    n_comp = min(len(vecs), vecs.shape[1], n_components or N_COMPONENTS)
    _, S, _ = np.linalg.svd(centered, full_matrices=False)
    S = S[:n_comp]
    explained = (S**2) / ((S**2).sum() + 1e-10)
    cumvar = np.cumsum(explained)
    dims = int(np.searchsorted(cumvar, threshold)) + 1
    return dims, cumvar

def solution_density(all_losses, base_loss, margin=0.0):
    """Fraction of perturbations that improve on base loss by at least margin."""
    return float((all_losses <= base_loss - margin).mean())

# ── Experiment 1: thicket vs needle (from v1, cleaner) ───────────────────────
def experiment_1_thicket_vs_needle():
    print("\n" + "="*60)
    print("EXPERIMENT 1: Thicket vs Needle subspace structure")
    print("="*60)

    results = {}
    for condition, pretrain_iters, sigma in [
        ("needle (no pretrain)", 0,   0.05),
        ("thicket (pretrained)", 500, 0.002),
    ]:
        print(f"\n  Condition: {condition}")
        torch.manual_seed(GLOBAL_SEED)
        model = Net()
        model.xavier_init()

        if pretrain_iters > 0:
            model = pretrain(model, iters=pretrain_iters)

        top_k_idx, all_deltas, all_losses, base_loss = randopt_collect(model, sigma)

        dims_all,   cumvar_all   = pca_dims_for_threshold(all_deltas,           0.90)
        dims_topk,  cumvar_topk  = pca_dims_for_threshold(all_deltas[top_k_idx], 0.90)
        density = solution_density(all_losses, base_loss)

        print(f"    dims_90 (all):   {dims_all}")
        print(f"    dims_90 (top-K): {dims_topk}")
        print(f"    compression:     {dims_all/max(dims_topk,1):.1f}x")
        print(f"    solution density:{density:.3f}")

        results[condition] = {
            "dims_all": dims_all, "dims_topk": dims_topk,
            "cumvar_all": cumvar_all, "cumvar_topk": cumvar_topk,
            "density": density, "sigma": sigma,
            "all_losses": all_losses, "base_loss": base_loss,
            "top_k_idx": top_k_idx, "all_deltas": all_deltas,
        }

    return results

# ── Experiment 2: pretraining scaling sweep (THE NEW EXPERIMENT) ──────────────
def experiment_2_pretraining_sweep():
    """
    Sweep pretraining iterations. For each checkpoint:
      - dims_90(top-K):  dimensionality of the good perturbation subspace
      - dims_90(all):    dimensionality of all perturbations (should be ~constant)
      - solution_density: fraction of perturbations that improve on base
      - best_loss:       best loss achieved by any perturbation

    The hypothesis: dims_90(top-K) decreases as pretraining increases,
    while dims_90(all) stays roughly constant.
    This would mean pretraining makes the GOOD region more structured,
    not just more populated.
    """
    print("\n" + "="*60)
    print("EXPERIMENT 2: Pretraining scaling sweep")
    print("  Question: does more pretraining -> more concentrated good subspace?")
    print("="*60)

    # Checkpoints to evaluate. 0 = xavier init (needle), rest = progressive pretraining.
    PRETRAIN_CHECKPOINTS = [0, 25, 50, 100, 200, 350, 500]
    SIGMA = 0.002  # fixed sigma throughout (same as thicket condition)

    # We build the model progressively — train from checkpoint to checkpoint
    # so we don't retrain from scratch each time (saves compute)
    torch.manual_seed(GLOBAL_SEED)
    model = Net()
    model.xavier_init()

    sweep_results = []
    prev_iters = 0

    for target_iters in PRETRAIN_CHECKPOINTS:
        additional_iters = target_iters - prev_iters

        if additional_iters > 0:
            print(f"\n  Training {additional_iters} more iters (total: {target_iters})...")
            model = pretrain(model, iters=additional_iters, verbose=True)
        else:
            print(f"\n  Checkpoint: {target_iters} iters (xavier init, no training)")

        prev_iters = target_iters

        # Run RandOpt and collect metrics
        top_k_idx, all_deltas, all_losses, base_loss = randopt_collect(
            model, sigma=SIGMA
        )

        dims_all,  cumvar_all  = pca_dims_for_threshold(all_deltas,            0.90)
        dims_topk, cumvar_topk = pca_dims_for_threshold(all_deltas[top_k_idx], 0.90)
        density  = solution_density(all_losses, base_loss)
        best_loss = all_losses.min()
        compression = dims_all / max(dims_topk, 1)

        print(f"    dims_90(all)={dims_all:3d}  dims_90(top-K)={dims_topk:3d}  "
              f"compression={compression:.1f}x  density={density:.3f}  "
              f"best_loss={best_loss:.4f}  base_loss={base_loss:.4f}")

        sweep_results.append({
            "iters":       target_iters,
            "dims_all":    dims_all,
            "dims_topk":   dims_topk,
            "compression": compression,
            "density":     density,
            "best_loss":   best_loss,
            "base_loss":   base_loss,
            "cumvar_all":  cumvar_all,
            "cumvar_topk": cumvar_topk,
        })

    return sweep_results

# ── Plotting ───────────────────────────────────────────────────────────────────
def plot_experiment_1(results, save_path):
    """Side-by-side cumulative variance curves for thicket vs needle."""
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle("Exp 1: Subspace Structure — Thicket vs Needle", fontsize=13, fontweight="bold")

    colors = {"needle (no pretrain)": "#4878CF", "thicket (pretrained)": "#D65F5F"}

    for ax_idx, (condition, res) in enumerate(results.items()):
        ax = axes[ax_idx]
        color = colors[condition]
        n_all  = len(res["cumvar_all"])
        n_topk = len(res["cumvar_topk"])
        ax.plot(range(1, n_all+1),  res["cumvar_all"]*100,
                color=color, lw=2.5, linestyle="--", label=f"All {N_PERTURBATIONS} perturbations")
        ax.plot(range(1, n_topk+1), res["cumvar_topk"]*100,
                color=color, lw=2.5, linestyle="-",  label=f"Top-{K_SELECT} only")
        ax.axhline(90, color="gray", lw=1, linestyle=":", alpha=0.7)
        ax.fill_between(range(1, n_topk+1), res["cumvar_topk"]*100, 100,
                        alpha=0.08, color=color)
        ax.set_title(f"{condition}\n"
                     f"dims_90: all={res['dims_all']}, top-K={res['dims_topk']} "
                     f"({res['dims_all']/max(res['dims_topk'],1):.1f}x compression)\n"
                     f"solution density={res['density']:.3f}",
                     fontsize=9)
        ax.set_xlabel("Number of principal components")
        ax.set_ylabel("Cumulative variance explained (%)")
        ax.set_xlim([1, min(N_COMPONENTS, 40)])
        ax.set_ylim([0, 102])
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.25)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"  Saved: {save_path}")
    plt.close()

def plot_experiment_2(sweep_results, save_path):
    """The scaling law plot — the central new figure."""
    iters      = [r["iters"]       for r in sweep_results]
    dims_all   = [r["dims_all"]    for r in sweep_results]
    dims_topk  = [r["dims_topk"]   for r in sweep_results]
    compress   = [r["compression"] for r in sweep_results]
    density    = [r["density"]     for r in sweep_results]
    best_loss  = [r["best_loss"]   for r in sweep_results]
    base_loss  = [r["base_loss"]   for r in sweep_results]

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle("Exp 2: Pretraining Scaling — Does More Pretraining Create a More Structured Thicket?",
                 fontsize=12, fontweight="bold")

    # ── Plot A: dims_90 vs pretraining iters (THE KEY PLOT) ──────────────────
    ax = axes[0]
    ax.plot(iters, dims_all,  "b--o", lw=2, ms=7, label="dims_90 (all perturbations)")
    ax.plot(iters, dims_topk, "r-o",  lw=2.5, ms=7, label="dims_90 (top-K only)")
    ax.fill_between(iters, dims_topk, dims_all, alpha=0.1, color="purple",
                    label="Compression gap")
    ax.set_xlabel("Pretraining iterations")
    ax.set_ylabel("Dims for 90% variance")
    ax.set_title("Subspace Dimensionality vs Pretraining\n"
                 "(key result: does red line fall as training increases?)")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # ── Plot B: compression ratio + solution density ──────────────────────────
    ax = axes[1]
    ax2 = ax.twinx()
    l1, = ax.plot(iters, compress, "purple", lw=2.5, marker="s", ms=7,
                  label="Compression ratio (dims_all / dims_topK)")
    l2, = ax2.plot(iters, density, "g--", lw=2, marker="^", ms=7,
                   label="Solution density")
    ax.set_xlabel("Pretraining iterations")
    ax.set_ylabel("Compression ratio", color="purple")
    ax2.set_ylabel("Solution density (frac improving base)", color="green")
    ax.set_title("Compression & Density vs Pretraining\n"
                 "(both should rise with more pretraining)")
    lines = [l1, l2]
    ax.legend(lines, [l.get_label() for l in lines], fontsize=8)
    ax.grid(True, alpha=0.3)

    # ── Plot C: cumulative variance curves at each checkpoint ─────────────────
    ax = axes[2]
    cmap = plt.cm.viridis
    norm_iters = [(it / max(iters) if max(iters) > 0 else 0) for it in iters]
    for i, res in enumerate(sweep_results):
        cv   = res["cumvar_topk"]
        x    = range(1, len(cv)+1)
        col  = cmap(norm_iters[i])
        lbl  = f"{res['iters']} iters (d={res['dims_topk']})"
        ax.plot(x, cv*100, color=col, lw=1.8, label=lbl)
    ax.axhline(90, color="gray", lw=1, linestyle=":", alpha=0.7, label="90% threshold")
    ax.set_xlabel("Number of principal components")
    ax.set_ylabel("Cumulative variance of top-K (%)")
    ax.set_title("Top-K Variance Curves at Each Checkpoint\n"
                 "(curves should shift LEFT with more pretraining)")
    ax.legend(fontsize=7, ncol=1)
    ax.set_xlim([1, 25])
    sm = plt.cm.ScalarMappable(cmap=cmap,
                               norm=plt.Normalize(vmin=0, vmax=max(iters)))
    sm.set_array([])
    plt.colorbar(sm, ax=ax, label="Pretraining iters")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"  Saved: {save_path}")
    plt.close()

def plot_summary_table(exp1_results, sweep_results, save_path):
    """Clean summary figure suitable for a paper draft."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle("Summary: Pretraining Creates a Geometrically Structured Thicket",
                 fontsize=13, fontweight="bold")

    # Left: compression ratio as a function of pretraining (the scaling law)
    ax = axes[0]
    iters   = [r["iters"]       for r in sweep_results]
    compress= [r["compression"] for r in sweep_results]
    density = [r["density"]     for r in sweep_results]

    ax.plot(iters, compress, "o-", color="#D65F5F", lw=2.5, ms=8,
            label="Compression ratio\n(dims_all / dims_topK)")
    ax.set_xlabel("Pretraining iterations", fontsize=11)
    ax.set_ylabel("Subspace compression ratio", fontsize=11, color="#D65F5F")
    ax.tick_params(axis="y", labelcolor="#D65F5F")

    ax2 = ax.twinx()
    ax2.plot(iters, density, "s--", color="#4878CF", lw=2, ms=7,
             label="Solution density")
    ax2.set_ylabel("Solution density", fontsize=11, color="#4878CF")
    ax2.tick_params(axis="y", labelcolor="#4878CF")
    ax.set_title("Pretraining Scaling Law\n"
                 "More pretraining → higher compression & density", fontsize=10)
    ax.grid(True, alpha=0.25)

    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1+lines2, labels1+labels2, fontsize=8, loc="upper left")

    # Right: bar chart comparing thicket vs needle at matched budget
    ax = axes[1]
    conditions = list(exp1_results.keys())
    dims_all_vals  = [exp1_results[c]["dims_all"]  for c in conditions]
    dims_topk_vals = [exp1_results[c]["dims_topk"] for c in conditions]
    density_vals   = [exp1_results[c]["density"]   for c in conditions]

    x = np.arange(len(conditions))
    w = 0.35
    b1 = ax.bar(x - w/2, dims_all_vals,  w, label="dims_90 (all)",   color="#4878CF", alpha=0.8)
    b2 = ax.bar(x + w/2, dims_topk_vals, w, label="dims_90 (top-K)", color="#D65F5F", alpha=0.8)

    # Annotate with density
    for i, (c, d) in enumerate(zip(conditions, density_vals)):
        ax.text(i, max(dims_all_vals[i], dims_topk_vals[i]) + 0.5,
                f"density={d:.3f}", ha="center", fontsize=8, color="gray")

    ax.set_xticks(x)
    ax.set_xticklabels([c.split("(")[0].strip() for c in conditions], fontsize=10)
    ax.set_ylabel("Dims for 90% variance")
    ax.set_title("Thicket vs Needle: Subspace Structure\n"
                 "(gap between bars = compression)", fontsize=10)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.25, axis="y")

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"  Saved: {save_path}")
    plt.close()

# ── main ──────────────────────────────────────────────────────────────────────
def main():
    print("="*60)
    print("SUBSPACE EXPERIMENT v2")
    print("Toy-scale proof that pretraining creates structured thickets")
    print("="*60)

    # ── Experiment 1: thicket vs needle ──────────────────────────────────────
    exp1_results = experiment_1_thicket_vs_needle()
    plot_experiment_1(exp1_results, "/mnt/user-data/outputs/exp1_thicket_vs_needle.png")

    # ── Experiment 2: pretraining scaling sweep ───────────────────────────────
    sweep_results = experiment_2_pretraining_sweep()
    plot_experiment_2(sweep_results, "/mnt/user-data/outputs/exp2_pretraining_sweep.png")

    # ── Summary figure ────────────────────────────────────────────────────────
    plot_summary_table(exp1_results, sweep_results,
                       "/mnt/user-data/outputs/exp_summary.png")

    # ── Print the key numbers table ───────────────────────────────────────────
    print("\n" + "="*60)
    print("KEY RESULTS TABLE (copy this into your notes)")
    print("="*60)
    print(f"\n{'Pretrain iters':>16} | {'dims_90(all)':>12} | {'dims_90(top-K)':>14} | "
          f"{'compression':>11} | {'density':>9} | {'best_loss':>10}")
    print("-"*85)
    for r in sweep_results:
        print(f"{r['iters']:>16} | {r['dims_all']:>12} | {r['dims_topk']:>14} | "
              f"{r['compression']:>11.2f} | {r['density']:>9.4f} | {r['best_loss']:>10.4f}")

    print("\n" + "="*60)
    print("INTERPRETATION")
    print("="*60)
    print("""
What to look for:

1. dims_90(top-K) should DECREASE with more pretraining
   -> the good perturbation region becomes more geometrically concentrated
   -> this is the toy-scale analog of the Neural Thickets density scaling law

2. dims_90(all) should stay ROUGHLY CONSTANT
   -> random perturbations don't become more structured
   -> pretraining specifically organizes the GOOD region, not all of space

3. Compression ratio should INCREASE with pretraining
   -> the gap between all-perturbation and top-K dimensionality widens
   -> more pretraining = more exploitable geometric structure

4. Solution density should INCREASE with pretraining
   -> replicates the Neural Thickets main finding at toy scale

If 1+2+3+4 all hold -> you have a clean toy-scale story that motivates
the LLM experiment and directly supports the certified subspace idea.

NEXT STEP: Run the same analysis on GPT-2 vs random init.
The LLM version of this script is in subspace_experiment_llm.py (TBD).
""")

if __name__ == "__main__":
    main()
