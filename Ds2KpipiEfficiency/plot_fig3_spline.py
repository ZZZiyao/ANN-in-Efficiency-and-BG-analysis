import numpy as np
import uproot
import matplotlib.pyplot as plt
from scipy.interpolate import RectBivariateSpline

def hist2d(m, t, bins):
    H, xe, ye = np.histogram2d(m, t, bins=[bins, bins], range=[[0,1],[0,1]])
    xc = 0.5 * (xe[:-1] + xe[1:])
    yc = 0.5 * (ye[:-1] + ye[1:])
    return H, xc, yc

def proj_1d(x, bins=100):
    H, edges = np.histogram(x, bins=bins, range=(0,1))
    centers = 0.5 * (edges[:-1] + edges[1:])
    width = edges[1] - edges[0]
    y = H / width
    yerr = np.sqrt(H) / width
    return centers, y, yerr, width, H

def plot_fig3_spline(
    root_path="eff_toy_4e6.root",
    tree_name=None,
    m_branch="mprime",
    t_branch="thetaprime",
    out_png="fig3_spline.png",
    bins_spline=10,   # paper: 10×10
    bins_proj=100
):

    # ---- load ----
    f = uproot.open(root_path)
    if tree_name is None:
        tree = f[f.keys()[0]]
    else:
        tree = f[tree_name]

    m = tree[m_branch].array(library="np").astype(np.float64)
    t = tree[t_branch].array(library="np").astype(np.float64)

    mask = (m>=0)&(m<=1)&(t>=0)&(t<=1)
    m, t = m[mask], t[mask]

    # ---- 10x10 histogram (as in paper) ----
    H, xc, yc = hist2d(m, t, bins_spline)

    # relative density (mean=1)
    meanH = H.mean() if H.mean() > 0 else 1.0
    eps = H / meanH

    # spline interpolation
    spline = RectBivariateSpline(xc, yc, eps, kx=3, ky=3, s=0.0)

    # evaluate on fine grid
    fine = 200
    mf = np.linspace(0,1,fine)
    tf = np.linspace(0,1,fine)
    eps_fit = spline(mf, tf).T  # transpose for plotting

    # ---- projections ----
    m_cent, m_y, m_yerr, m_w, Hm = proj_1d(m, bins=bins_proj)
    t_cent, t_y, t_yerr, t_w, Ht = proj_1d(t, bins=bins_proj)

    # build fit projection lines
    eps_m_shape = eps_fit.mean(axis=0)
    eps_t_shape = eps_fit.mean(axis=1)

    def scale_shape(shape, centers, y_vals, width):
        shape = np.clip(shape, 0, None)
        if shape.sum() <= 0:
            shape = np.ones_like(shape)

        shape_cent = np.interp(centers, mf, shape)
        shape_cent = np.clip(shape_cent, 0, None)

        total_counts = np.sum(y_vals * width)
        scale = total_counts / shape_cent.sum()
        return (shape_cent * scale) / width

    m_line = scale_shape(eps_m_shape, m_cent, m_y, m_w)
    t_line = scale_shape(eps_t_shape, t_cent, t_y, t_w)

    # ---- plot ----
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.8), constrained_layout=True)

    im = axes[0].imshow(
        eps_fit,
        origin="lower",
        extent=[0,1,0,1],
        aspect="auto"
    )
    axes[0].set_xlabel("m'")
    axes[0].set_ylabel("θ'")
    axes[0].text(0.92, 0.08, "(d)", transform=axes[0].transAxes,
                 ha="right", va="bottom", fontsize=12,
                 bbox=dict(facecolor="white", edgecolor="none", alpha=0.9))
    plt.colorbar(im, ax=axes[0], label="ε(m',θ') (relative)")

    axes[1].errorbar(m_cent, m_y, yerr=m_yerr, fmt="o", ms=3, label="Simulation")
    axes[1].plot(m_cent, m_line, "-", color="green", lw=2, label="Fit result")
    axes[1].set_xlabel("m'")
    axes[1].set_ylabel(f"Entries / ({m_w:.2f})")
    axes[1].text(0.92, 0.08, "(e)", transform=axes[1].transAxes,
                 ha="right", va="bottom", fontsize=12)
    axes[1].legend(frameon=False)

    axes[2].errorbar(t_cent, t_y, yerr=t_yerr, fmt="o", ms=3, label="Simulation")
    axes[2].plot(t_cent, t_line, "-", color="green", lw=2, label="Fit result")
    axes[2].set_xlabel("θ'")
    axes[2].set_ylabel(f"Entries / ({t_w:.2f})")
    axes[2].text(0.92, 0.08, "(f)", transform=axes[2].transAxes,
                 ha="right", va="bottom", fontsize=12)
    axes[2].legend(frameon=False)

    for ax in axes:
        ax.tick_params(direction="in", top=True, right=True)

    fig.suptitle("Fig.3 (bottom): cubic spline 10×10", fontsize=12)
    fig.savefig(out_png, dpi=200)
    print(f"[OK] saved {out_png}")

if __name__ == "__main__":
    plot_fig3_spline(
        root_path="eff_toy_4e6.root",
        tree_name=None,
        m_branch="mprime",
        t_branch="thetaprime",
        out_png="fig3_spline.png"
    )