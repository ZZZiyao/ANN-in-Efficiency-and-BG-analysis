"""
Diagnostic script: generate background with varying kstarfrac and rhofrac,
and plot distributions for each combination.

For each (kstarfrac, rhofrac) two figures are produced:
  1) 3x3 diagnostic figure:
       Row 1 (a-c): pure-component 2D m' vs theta' (combinatorial, K*, rho)
       Row 2 (d-f): fig2abc-style normalised-density 2D (m' vs theta', m' vs mD, theta' vs mD)
       Row 3 (g-i): fig2def-style 1D (mD, m' projections, theta' projections) [10^5 subsample]
  2) 3x6 component figure:
       Each row = one pure component (combinatorial, K*, rho)
       Cols 0-4: 1D errorbar histograms (mD, m', theta', m2Kpi, m2pipi) [10^5 subsample]
       Col 5: 2D m' vs theta' [full statistics]

Uses generate_selection from DistributionModel directly.
"""
import sys, os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

import amplitf.interface as atfi

from DistributionModel import (
    true_cuts, generate_selection, generated_variables,
    observables_phase_space, random_array_size,
)

plt.rcParams.update({
    "font.size": 12,
    "axes.labelsize": 14,
    "xtick.direction": "in",
    "ytick.direction": "in",
})

# -------------------- region definitions (fig2def) --------------------
MD_LO, MD_S1, MD_S2, MD_HI = 1.77, 1.92, 2.02, 2.17

def define_regions_hard(md, lo=MD_LO, s1=MD_S1, s2=MD_S2, hi=MD_HI):
    full   = (md > lo) & (md < hi)
    lowSB  = full & (md >= lo) & (md <  s1)
    signal = full & (md >= s1) & (md <= s2)
    upSB   = full & (md >  s2) & (md <= hi)
    return signal, lowSB, upSB, full

# -------------------- normalised 2D histogram (fig2abc) --------------------
def hist2d_avg_density_1(x, y, bins, ranges):
    H, xedges, yedges = np.histogram2d(x, y, bins=bins, range=ranges)
    N = H.sum()
    if N <= 0:
        return H, xedges, yedges
    dx = xedges[1] - xedges[0]
    dy = yedges[1] - yedges[0]
    rho = H / (N * dx * dy)
    m = rho.mean()
    if m > 0:
        rho = rho / m
    return rho, xedges, yedges

# -------------------- 1D histogram with error bars (fig2def) --------------------
def hist_errorbar(ax, x, bins, mask=None, label=None,
                  fmt="k.", ls="none", lw=1.2, markersize=4,
                  scale_to=None):
    xx = x if mask is None else x[mask]
    counts, edges = np.histogram(xx, bins=bins)
    centers = 0.5 * (edges[1:] + edges[:-1])
    y = counts.astype(float)
    yerr = np.sqrt(counts)
    if scale_to is not None:
        total = y.sum()
        if total > 0:
            s = scale_to / total
            y *= s
            yerr *= s
    ax.errorbar(centers, y, yerr=yerr,
                fmt=fmt, linestyle=ls, linewidth=lw,
                markersize=markersize, capsize=0, label=label)
    return centers, y, yerr


def generate_mix(kstarfrac, rhofrac, nev, chunk_size=500000):
    """
    Generate background events with given kstarfrac/rhofrac.
    Returns dict of arrays keyed by generated_variables names.
    """
    cuts = list(true_cuts[:6]) + [atfi.const(kstarfrac), atfi.const(rhofrac)]

    arrays = []
    n = 0
    while n < nev:
        rnd = tf.random.uniform([chunk_size, random_array_size], dtype=atfi.fptype())
        obs = generate_selection(cuts, rnd, constant_cuts=True)
        batch = atfi.stack(obs, axis=1)
        arrays.append(batch)
        n += batch.shape[0]
        print(f"  kstar={kstarfrac:.1f} rho={rhofrac:.1f}: collected {n}/{nev}")

    data = np.concatenate(arrays, axis=0)[:nev]
    return {var: data[:, i] for i, var in enumerate(generated_variables)}


def generate_pure_components(nev, chunk_size=500000):
    """
    Generate pure combinatorial, K*, and rho samples (once).
    Returns (data_comb, data_kstar, data_rho).
    """
    print("\n=== Generating pure components ===")
    data_comb  = generate_mix(0.0, 0.0, nev, chunk_size)  # pure combinatorial
    data_kstar = generate_mix(1.0, 0.0, nev, chunk_size)  # pure K*
    data_rho   = generate_mix(0.0, 1.0, nev, chunk_size)  # pure rho
    return data_comb, data_kstar, data_rho


def _plot_2d_density(ax, x, y, bins, ranges, vmin=0.0, vmax=5.0, cmap="afmhot_r"):
    """Plot normalised-density 2D histogram on *ax* (fig2abc style)."""
    rho, xedges, yedges = hist2d_avg_density_1(x, y, bins=bins, ranges=ranges)
    pcm = ax.pcolormesh(xedges, yedges, rho.T, shading="auto",
                        cmap=cmap, vmin=vmin, vmax=vmax)
    ax.minorticks_on()
    ax.tick_params(which="both", top=True, right=True)
    return pcm


def plot_diagnostics(data, kstarfrac, rhofrac, outname,
                     data_comb=None, data_kstar=None, data_rho=None):
    """
    2x3 diagnostic figure for one (kstarfrac, rhofrac) combination.

    Row 1 (a-c): fig2abc normalised density for the *mixed* data
        (a) m' vs theta'   (b) m' vs mD   (c) theta' vs mD
    Row 2 (d-f): fig2def 1D distributions for the *mixed* data (10^5 subsample)
        (d) mD   (e) m' projections   (f) theta' projections
    """
    fig, axes = plt.subplots(2, 3, figsize=(18, 11))
    fig.suptitle(f"kstarfrac = {kstarfrac:.1f},  rhofrac = {rhofrac:.1f}",
                 fontsize=16, y=0.98)

    # ---- Row 1: fig2abc normalised-density 2D for mixed data ----
    bins2d = [60, 60]

    ax = axes[0, 0]
    pcm = _plot_2d_density(ax, data["mprime"], data["thetaprime"],
                           bins2d, [[0, 1], [0, 1]], vmin=0, vmax=5)
    ax.set_xlabel(r"$m'$"); ax.set_ylabel(r"$\theta'$")
    ax.set_title("(a) $m'$ vs $\\theta'$")
    fig.colorbar(pcm, ax=ax, fraction=0.046, pad=0.04)

    ax = axes[0, 1]
    pcm = _plot_2d_density(ax, data["mprime"], data["md"],
                           bins2d, [[0, 1], [MD_LO, MD_HI]], vmin=0, vmax=3)
    ax.set_xlabel(r"$m'$"); ax.set_ylabel(r"$m_D$ (GeV)")
    ax.set_title("(b) $m'$ vs $m_D$")
    fig.colorbar(pcm, ax=ax, fraction=0.046, pad=0.04)

    ax = axes[0, 2]
    pcm = _plot_2d_density(ax, data["thetaprime"], data["md"],
                           bins2d, [[0, 1], [MD_LO, MD_HI]], vmin=0, vmax=3)
    ax.set_xlabel(r"$\theta'$"); ax.set_ylabel(r"$m_D$ (GeV)")
    ax.set_title("(c) $\\theta'$ vs $m_D$")
    fig.colorbar(pcm, ax=ax, fraction=0.046, pad=0.04)

    # ---- Row 2: fig2def 1D distributions (10^5 subsample) ----
    nrow2 = min(100000, len(data["mprime"]))
    idx2 = np.random.choice(len(data["mprime"]), nrow2, replace=False)
    m  = data["mprime"][idx2]
    t  = data["thetaprime"][idx2]
    md = data["md"][idx2]
    signal, lowSB, upSB, full = define_regions_hard(md)
    Ns = int(signal.sum())

    ax = axes[1, 0]
    bins_md = np.arange(MD_LO, MD_HI + 1e-12, 0.004)
    hist_errorbar(ax, md, bins_md, mask=full, fmt="k.", ls="none")
    ax.axvline(MD_S1, color="red", linestyle="--", linewidth=1.5)
    ax.axvline(MD_S2, color="red", linestyle="--", linewidth=1.5)
    ax.set_xlabel(r"$m_D$ (GeV)")
    ax.set_ylabel("Entries / (0.004 GeV)")
    ax.set_title("(d) $m_D$")
    ax.set_ylim(bottom=0)
    ax.minorticks_on()
    ax.tick_params(which="both", top=True, right=True)

    ax = axes[1, 1]
    bins_m = np.arange(0, 1.0 + 1e-12, 0.01)
    hist_errorbar(ax, m, bins_m, mask=signal, label="Signal", fmt="k.", ls="none")
    hist_errorbar(ax, m, bins_m, mask=lowSB,  label="Lower SB",
                  fmt="b", ls="--", lw=1.5, scale_to=Ns)
    hist_errorbar(ax, m, bins_m, mask=upSB,   label="Upper SB",
                  fmt="r", ls="--", lw=1.5, scale_to=Ns)
    ax.set_xlabel(r"$m'$")
    ax.set_ylabel("Entries / (0.01)")
    ax.set_title("(e) $m'$ projections")
    ax.set_ylim(bottom=0)
    ax.legend(frameon=False, fontsize=10)
    ax.minorticks_on()
    ax.tick_params(which="both", top=True, right=True)

    ax = axes[1, 2]
    bins_t = np.arange(0, 1.0 + 1e-12, 0.01)
    hist_errorbar(ax, t, bins_t, mask=signal, label="Signal", fmt="k.", ls="none")
    hist_errorbar(ax, t, bins_t, mask=lowSB,  label="Lower SB",
                  fmt="b", ls="--", lw=1.5, scale_to=Ns)
    hist_errorbar(ax, t, bins_t, mask=upSB,   label="Upper SB",
                  fmt="r", ls="--", lw=1.5, scale_to=Ns)
    ax.set_xlabel(r"$\theta'$")
    ax.set_ylabel("Entries / (0.01)")
    ax.set_title("(f) $\\theta'$ projections")
    ax.set_ylim(bottom=0)
    ax.legend(frameon=False, fontsize=10)
    ax.minorticks_on()
    ax.tick_params(which="both", top=True, right=True)

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(outname, dpi=200)
    print(f"Saved {outname}")
    plt.close()


def plot_1d_components(kstarfrac, rhofrac, outname,
                       data_comb=None, data_kstar=None, data_rho=None):
    """
    3x6 figure showing 1D distributions of each pure component (errorbar style).

    Row 1: combinatorial  --  mD, m', theta', m2(Kpi), m2(pipi), 2D m' vs theta'
    Row 2: K*             --  same
    Row 3: rho            --  same

    Columns 0-4 use a 10^5 subsample; column 5 (2D) uses full statistics.
    """
    comp_data  = [data_comb, data_kstar, data_rho]
    comp_label = ["Combinatorial", r"$K^*$", r"$\rho$"]

    hist_specs = [
        ("md",         r"$m_D$ (GeV)",              (1.77, 2.17), 80),
        ("mprime",     r"$m'$",                      (0, 1),       80),
        ("thetaprime", r"$\theta'$",                 (0, 1),       80),
        ("m2kpi",      r"$m^2(K\pi)$ (GeV$^2$)",    None,         80),
        ("m2pipi",     r"$m^2(\pi\pi)$ (GeV$^2$)",  None,         80),
    ]

    fig, axes = plt.subplots(3, 6, figsize=(30, 14))
    fig.suptitle(f"Pure components  (kstarfrac = {kstarfrac:.1f},  rhofrac = {rhofrac:.1f})",
                 fontsize=16, y=0.98)

    for row, (cd, cl) in enumerate(zip(comp_data, comp_label)):
        if cd is None:
            for col in range(6):
                axes[row, col].axis("off")
            continue

        # Subsample to 10^5 for 1D histograms (columns 0-4)
        n_1d = min(100000, len(cd["mprime"]))
        idx_1d = np.random.choice(len(cd["mprime"]), n_1d, replace=False)

        # Columns 0-4: 1D errorbar histograms (subsampled)
        for col, (key, xlabel, xlim, nbins) in enumerate(hist_specs):
            ax = axes[row, col]
            d = cd[key][idx_1d]
            if xlim is None:
                xlim = (np.percentile(cd[key], 0.5), np.percentile(cd[key], 99.5))
            bins = np.linspace(xlim[0], xlim[1], nbins + 1)
            hist_errorbar(ax, d, bins, fmt="k.", ls="none")
            ax.set_xlabel(xlabel)
            ax.set_ylim(bottom=0)
            ax.minorticks_on()
            ax.tick_params(which="both", top=True, right=True)
            if col == 0:
                ax.set_ylabel(cl)

        # Column 5: 2D m' vs theta' (full statistics)
        ax = axes[row, 5]
        ax.hist2d(cd["mprime"], cd["thetaprime"], bins=60,
                  range=[[0, 1], [0, 1]], cmap="afmhot_r")
        ax.set_xlabel(r"$m'$")
        ax.set_ylabel(r"$\theta'$")
        ax.minorticks_on()
        ax.tick_params(which="both", top=True, right=True)

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(outname, dpi=200)
    print(f"Saved {outname}")
    plt.close()


def main():
    nev = int(float(sys.argv[1])) if len(sys.argv) > 1 else 4000000

    atfi.set_seed(42)

    os.makedirs("plots", exist_ok=True)

    # Generate pure components once (reused for every combination)
    data_comb, data_kstar, data_rho = generate_pure_components(nev)

    # Scan rhofrac for fixed kstarfrac values
    scan = {
        0.1: np.arange(0.0, 0.6, 0.1),    # kstarfrac=0.1, rhofrac 0~0.5
        0.2: np.arange(0.0, 0.6, 0.1),    # kstarfrac=0.2, rhofrac 0~0.5
        0.3: np.arange(0.0, 0.6, 0.1),    # kstarfrac=0.3, rhofrac 0~0.5
    }

    for kf, rho_values in scan.items():
        for rf in rho_values:
            print(f"\n=== kstarfrac={kf:.1f}, rhofrac={rf:.1f} ===")
            data = generate_mix(kf, rf, nev)
            plot_diagnostics(
                data, kf, rf,
                os.path.join("plots", f"diagnose_kf{kf:.1f}_rf{rf:.1f}.png"),
            )
            plot_1d_components(
                kf, rf,
                os.path.join("plots", f"components_kf{kf:.1f}_rf{rf:.1f}.png"),
                data_comb=data_comb, data_kstar=data_kstar, data_rho=data_rho,
            )


if __name__ == "__main__":
    main()
