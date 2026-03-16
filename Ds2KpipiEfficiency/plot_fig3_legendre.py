import numpy as np
import uproot
import matplotlib.pyplot as plt

from scipy.special import eval_legendre
from sklearn.linear_model import ElasticNet


# ===============================
# Legendre basis
# ===============================
def legendre_design(u, v, n):
    Pu = np.stack([eval_legendre(i, u) for i in range(n + 1)], axis=1)
    Pv = np.stack([eval_legendre(j, v) for j in range(n + 1)], axis=1)
    return (Pu[:, :, None] * Pv[:, None, :]).reshape(len(u), (n + 1) ** 2)


def predict_legendre(coeff, n, u, v):
    X = legendre_design(u, v, n)
    return X @ coeff


# ===============================
# Histogram helpers
# ===============================
def hist2d(m, t, bins):
    H, xe, ye = np.histogram2d(
        m, t,
        bins=[bins, bins],
        range=[[0,1],[0,1]]
    )
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


# ===============================
# MAIN
# ===============================
def plot_fig3_legendre(
    root_path="acceptance.root",
    tree_name=None,
    out_png="fig3_legendre.png",
    n_leg=8,
    lambda1=0.01,
    lambda2=0.1,
    bins_fit=50,
    bins_proj=100,
):

    # ---- load ROOT ----
    f = uproot.open(root_path)
    tree = f[f.keys()[0]] if tree_name is None else f[tree_name]

    m = tree["mprime"].array(library="np")
    t = tree["thetaprime"].array(library="np")
    p = tree["pass"].array(library="np")

    mask = (m>=0)&(m<=1)&(t>=0)&(t<=1)
    m, t, p = m[mask], t[mask], p[mask]

    # ---- 2D histograms ----
    H_before, xc, yc = hist2d(m, t, bins_fit)
    H_after, _, _ = hist2d(m[p==1], t[p==1], bins_fit)

    # ---- efficiency map ----
    with np.errstate(divide="ignore", invalid="ignore"):
        eps = np.divide(H_after, H_before,
                        out=np.zeros_like(H_after),
                        where=H_before>0)

    # bin centers
    Xc, Yc = np.meshgrid(xc, yc, indexing="xy")
    u = 2.0 * Xc.ravel() - 1.0
    v = 2.0 * Yc.ravel() - 1.0
    y = eps.T.ravel()

    # Poisson error
    sigma = np.sqrt(H_after.T.ravel()+1) / (H_before.T.ravel()+1)
    w = 1.0 / (sigma**2 + 1e-12)

    # ---- Legendre fit ----
    X = legendre_design(u, v, n_leg)

    alpha = lambda1 + lambda2
    l1_ratio = lambda1 / alpha if alpha>0 else 0.0

    reg = ElasticNet(
        alpha=alpha,
        l1_ratio=l1_ratio,
        fit_intercept=False,
        max_iter=200000,
        tol=1e-8
    )
    reg.fit(X, y, sample_weight=w)
    coeff = reg.coef_.copy()

    # ---- evaluate smooth map ----
    fine = 200
    mf = np.linspace(0,1,fine)
    tf = np.linspace(0,1,fine)
    Mg, Tg = np.meshgrid(mf, tf, indexing="xy")
    uf = 2.0*Mg.ravel()-1.0
    vf = 2.0*Tg.ravel()-1.0
    eps_fit = predict_legendre(coeff, n_leg, uf, vf).reshape(Mg.shape)

    # ===============================
    # 1D projections (selected events)
    # ===============================
    m_sel = m[p==1]
    t_sel = t[p==1]

    m_cent, m_y, m_yerr, m_w, _ = proj_1d(m_sel, bins_proj)
    t_cent, t_y, t_yerr, t_w, _ = proj_1d(t_sel, bins_proj)

    # model projection
    eps_m_shape = eps_fit.mean(axis=0)
    eps_t_shape = eps_fit.mean(axis=1)

    eps_m_interp = np.interp(m_cent, mf, eps_m_shape)
    eps_t_interp = np.interp(t_cent, tf, eps_t_shape)

    # scale to match total counts
    scale_m = (m_y*m_w).sum() / eps_m_interp.sum()
    scale_t = (t_y*t_w).sum() / eps_t_interp.sum()

    m_line = (eps_m_interp*scale_m)/m_w
    t_line = (eps_t_interp*scale_t)/t_w

    # ===============================
    # Plot
    # ===============================
    fig, axes = plt.subplots(1,3,figsize=(12,3.8), constrained_layout=True)

    im = axes[0].imshow(
        eps_fit,
        origin="lower",
        extent=[0,1,0,1],
        aspect="auto",
        cmap="afmhot_r",
        vmin=0.0,
        vmax=1.2
    )
    axes[0].set_xlabel("m'")
    axes[0].set_ylabel("θ'")
    plt.colorbar(im, ax=axes[0], label="ε(m',θ')")

    axes[1].errorbar(m_cent, m_y, yerr=m_yerr, fmt="o", ms=3)
    axes[1].plot(m_cent, m_line, "-")
    axes[1].set_xlabel("m'")
    axes[1].set_ylabel(f"Entries / ({m_w:.2f})")

    axes[2].errorbar(t_cent, t_y, yerr=t_yerr, fmt="o", ms=3)
    axes[2].plot(t_cent, t_line, "-")
    axes[2].set_xlabel("θ'")
    axes[2].set_ylabel(f"Entries / ({t_w:.2f})")

    for ax in axes:
        ax.tick_params(direction="in", top=True, right=True)

    fig.suptitle("Legendre acceptance fit", fontsize=12)
    fig.savefig(out_png, dpi=200)
    print("Saved:", out_png)


if __name__ == "__main__":
    plot_fig3_legendre(root_path="acceptance.root")