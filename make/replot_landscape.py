"""
replot_landscape.py
Reload a saved *_landscape_grid.json and regenerate the landscape PNG.

Usage:
  python3 replot_landscape.py outputs_0604/basin_widening/gpt2/sst2/rank1_steps500_landscape_grid.json
  python3 replot_landscape.py outputs_0604/basin_widening/gpt2/sst2/  # all grids in a dir
"""

import json, sys, os, glob
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import mpl_toolkits.mplot3d  # registers projection='3d'
try:
    from scipy.ndimage import gaussian_filter as _gf
    _HAVE_SCIPY = True
except ImportError:
    _HAVE_SCIPY = False


def plot_from_grid(grid_path, out_png=None):
    with open(grid_path) as f:
        d = json.load(f)

    nll_grid    = np.array(d["nll_grid"])
    alphas      = np.array(d["alphas"])
    betas       = np.array(d["betas"])
    sweep       = d["sweep"]
    sigma_mult  = d["sigma_mult"]
    sh_pre      = d["sh_pre"]
    sh_ft       = d.get("sh_ft")
    eval_slack  = d.get("eval_slack", 1e-4)
    nd          = d["norm_delta"]
    traj_alphas = np.array(d["traj_alphas"])
    nll_traj    = np.array(d["nll_traj"])
    # fine trajectory added in latest format; fall back to coarse if absent
    fine_alphas = np.array(d.get("fine_alphas", d["traj_alphas"]))
    nll_fine    = np.array(d.get("nll_fine",    d["nll_traj"]))

    nll_smooth = _gf(nll_grid, sigma=0.9) if _HAVE_SCIPY else nll_grid
    nll_finite = np.where(np.isfinite(nll_smooth), nll_smooth,
                          float(np.nanmax(nll_smooth)))
    A, B = np.meshgrid(alphas, betas)
    mid  = len(alphas) // 2

    ft_i_c    = int(np.argmin(np.abs(traj_alphas)))
    ft_i_f    = int(np.argmin(np.abs(fine_alphas)))
    nll_at_ft = (float(nll_traj[ft_i_c]) if np.isfinite(nll_traj[ft_i_c])
                 else float(nll_fine[ft_i_f]))
    dnll_traj = nll_traj - nll_at_ft
    dnll_fine = nll_fine - nll_at_ft

    fig = plt.figure(figsize=(17, 5.8))
    gs  = gridspec.GridSpec(1, 3, figure=fig, wspace=0.42,
                            left=0.05, right=0.97, top=0.88, bottom=0.12)

    # ── Panel 1: 2D contour with σ½ circles ──────────────────────────────────
    ax2d = fig.add_subplot(gs[0])
    vmin, vmax = float(np.nanmin(nll_smooth)), float(np.nanmax(nll_smooth))
    levels = np.linspace(vmin, vmax, 25)
    cf = ax2d.contourf(A, B, nll_smooth, levels=levels, cmap="RdYlGn_r")
    ax2d.contour(A, B, nll_smooth, levels=levels,
                 colors="k", linewidths=0.35, alpha=0.4)
    fig.colorbar(cf, ax=ax2d, label="NLL", format="%.4f")
    ax2d.scatter([0], [0], s=200, color="#2563EB", marker="*",
                 zorder=6, label="θ_ft")
    circle_t = np.linspace(0, 2 * np.pi, 200)
    ax2d.plot(sh_pre * np.cos(circle_t), sh_pre * np.sin(circle_t),
              color="#3B82F6", lw=2, ls="--", label=f"σ½_pre={sh_pre:.4f}")
    if sh_ft is not None:
        ax2d.plot(sh_ft * np.cos(circle_t), sh_ft * np.sin(circle_t),
                  color="#F97316", lw=2, ls="--", label=f"σ½_ft={sh_ft:.4f}")
    ax2d.axvline(0, color="white", lw=0.6, ls="--", alpha=0.3)
    ax2d.axhline(0, color="white", lw=0.6, ls="--", alpha=0.3)
    ax2d.set_xlabel("α  (LoRA update direction d1)", fontsize=9)
    ax2d.set_ylabel("β  (random orthogonal d2)", fontsize=9)
    ax2d.set_title("2D Contour  (top-down)\nCircles = certified basin width σ½", fontsize=9)
    ax2d.legend(fontsize=7.5, loc="upper right")

    # Scale-context inset showing θ_pre→θ_ft vs the visible window
    axsc = ax2d.inset_axes([0.01, 0.01, 0.40, 0.26])
    axsc.set_xlim(0, 1)
    axsc.set_ylim(0, 1)
    axsc.set_facecolor("#F8FAFC")
    axsc.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
    for spine in axsc.spines.values():
        spine.set_linewidth(0.5)
    axsc.annotate("", xy=(0.88, 0.52), xytext=(0.12, 0.52),
                  xycoords="axes fraction", textcoords="axes fraction",
                  arrowprops=dict(arrowstyle="-|>", color="#111827",
                                  lw=1.2, mutation_scale=8))
    axsc.scatter([0.12], [0.52], s=28, color="#111827", marker="o",
                 transform=axsc.transAxes, zorder=4)
    axsc.scatter([0.88], [0.52], s=40, color="#2563EB", marker="*",
                 transform=axsc.transAxes, zorder=4)
    axsc.text(0.12, 0.75, "θ_pre", ha="center", fontsize=5.5,
              transform=axsc.transAxes, color="#111827")
    axsc.text(0.88, 0.75, "θ_ft", ha="center", fontsize=5.5,
              transform=axsc.transAxes, color="#2563EB")
    axsc.text(0.50, 0.88, f"‖ΔW‖ = {nd/sh_pre:.0f}×σ½",
              ha="center", fontsize=5.5, transform=axsc.transAxes, color="#111827")
    vis_frac = (sweep / nd) * 0.76
    rect_x = max(0.88 - vis_frac / 2, 0.80)
    from matplotlib.patches import FancyBboxPatch
    highlight = FancyBboxPatch((rect_x, 0.25), vis_frac, 0.54,
                               boxstyle="square,pad=0", linewidth=1.0,
                               edgecolor="#F97316", facecolor="#FEF3C7",
                               alpha=0.7, transform=axsc.transAxes, zorder=3)
    axsc.add_patch(highlight)
    axsc.text(rect_x + vis_frac / 2, 0.12, "← map", ha="center",
              fontsize=5, transform=axsc.transAxes, color="#B45309")

    # ── Panel 2: 3D surface ───────────────────────────────────────────────────
    ax3d = fig.add_subplot(gs[1], projection="3d")
    surf = ax3d.plot_surface(A, B, nll_finite, cmap="RdYlGn_r",
                             alpha=0.75, linewidth=0, antialiased=True,
                             rstride=1, cstride=1)
    # Project the fine 1D trajectory onto the surface (β=0 slice)
    traj_in_view = np.isfinite(nll_fine) & (np.abs(fine_alphas) <= sweep)
    if traj_in_view.any():
        _fa = fine_alphas[traj_in_view]
        _nf = nll_fine[traj_in_view]
        ax3d.plot(_fa, np.zeros_like(_fa), _nf + 0.00005,
                  color="#7C3AED", lw=2.5, zorder=10)
    ax3d.scatter([0], [0], [nll_finite[mid, mid]],
                 s=100, color="#2563EB", marker="*", zorder=11, depthshade=False)
    ax3d.set_xlabel("α (LoRA dir)", fontsize=7, labelpad=1)
    ax3d.set_ylabel("β (⊥)", fontsize=7, labelpad=1)
    ax3d.set_zlabel("NLL", fontsize=7, labelpad=1)
    ax3d.tick_params(labelsize=6)
    ax3d.set_title(f"3D Basin  (purple = trajectory slice)\nsweep ±{sigma_mult:.0f}×σ½",
                   fontsize=9, pad=4)
    fig.colorbar(surf, ax=ax3d, shrink=0.45, pad=0.08, label="NLL", format="%.4f")

    # ── Panel 3: θ_pre → θ_ft → beyond + zoom inset ──────────────────────────
    ax1d = fig.add_subplot(gs[2])
    valid_c = np.isfinite(dnll_traj)
    ax1d.plot(traj_alphas[valid_c], dnll_traj[valid_c],
              color="#7C3AED", lw=2.5, label="ΔNLL along d1")
    pre_i = int(np.argmin(np.abs(traj_alphas - (-nd))))
    ax1d.scatter([traj_alphas[pre_i]], [dnll_traj[pre_i]],
                 s=120, color="#111827", marker="o", zorder=5, label="θ_pre")
    ax1d.scatter([traj_alphas[ft_i_c]], [dnll_traj[ft_i_c]],
                 s=150, color="#2563EB", marker="*", zorder=5, label="θ_ft")
    ax1d.axhline(eval_slack, color="#16A34A", lw=1.3, ls="--",
                 label=f"cert. threshold (+{eval_slack:.0e})")
    ax1d.axhline(0, color="#2563EB", lw=0.7, ls=":", alpha=0.4)
    ax1d.axvline(0,   color="#2563EB", lw=0.8, ls=":", alpha=0.5)
    ax1d.axvline(-nd, color="#111827", lw=0.8, ls=":", alpha=0.5)
    ax1d.set_xlabel("α  (LoRA update direction, raw units)", fontsize=9)
    ax1d.set_ylabel("ΔNLL  =  NLL − NLL_ft", fontsize=9)
    ax1d.set_title("θ_pre → θ_ft → beyond\nΔNLL = 0 at θ_ft by definition", fontsize=9)
    ax1d.legend(fontsize=7.5, loc="upper left")
    ax1d.grid(True, alpha=0.2)

    # Inset: ±5×σ½ zoom showing certified region and collapse
    axins = ax1d.inset_axes([0.52, 0.38, 0.46, 0.55])
    valid_f = np.isfinite(dnll_fine)
    axins.plot(fine_alphas[valid_f], dnll_fine[valid_f], color="#7C3AED", lw=2)
    axins.fill_between(fine_alphas[valid_f], 0, eval_slack,
                       where=((dnll_fine[valid_f] >= -eval_slack * 0.5) &
                              (dnll_fine[valid_f] <= eval_slack)),
                       alpha=0.30, color="#16A34A")
    axins.axhline(eval_slack, color="#16A34A", lw=1.2, ls="--")
    axins.axhline(0, color="#2563EB", lw=0.7, ls=":", alpha=0.5)
    axins.scatter([0], [0], s=80, color="#2563EB", marker="*", zorder=5)
    axins.axvline(-sh_pre, color="#3B82F6", lw=1.1, ls=":")
    axins.axvline( sh_pre, color="#3B82F6", lw=1.1, ls=":", label="±σ½_pre")
    if sh_ft is not None:
        axins.axvline(-sh_ft, color="#F97316", lw=1.1, ls=":")
        axins.axvline( sh_ft, color="#F97316", lw=1.1, ls=":", label="±σ½_ft")
    axins.set_xlim(-5 * sh_pre, 5 * sh_pre)
    y_lo = min(-eval_slack * 0.5, float(np.nanmin(dnll_fine)))
    y_hi = max(eval_slack * 4,    float(np.nanmax(dnll_fine)) * 0.5)
    axins.set_ylim(y_lo, y_hi)
    axins.set_title("zoom  ±5×σ½", fontsize=7)
    axins.tick_params(labelsize=6)
    axins.legend(fontsize=6, loc="upper right")
    axins.text(0, eval_slack * 1.15, "certified\nregion ↓",
               ha="center", fontsize=6, color="#16A34A")
    ax1d.indicate_inset_zoom(axins, edgecolor="#888888", alpha=0.6)

    fig.suptitle(
        f"Loss Landscape  |  ★ θ_ft = origin  "
        f"|  sweep ±{sigma_mult:.1f}×σ½={sweep:.2e}  "
        f"|  ‖ΔW‖={nd:.2e}  ({nd/sh_pre:.0f}×σ½)",
        fontsize=10, y=0.97,
    )

    if out_png is None:
        out_png = grid_path.replace("_grid.json", ".png")
    fig.savefig(out_png, dpi=150, bbox_inches="tight")
    print(f"Saved {out_png}")
    plt.close()


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)
    target = sys.argv[1]
    if os.path.isdir(target):
        paths = sorted(glob.glob(os.path.join(target, "*_landscape_grid.json")))
        if not paths:
            print(f"No *_landscape_grid.json files found in {target}")
            sys.exit(1)
    else:
        paths = [target]
    for p in paths:
        plot_from_grid(p)
