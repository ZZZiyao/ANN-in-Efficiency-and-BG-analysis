"""
Diagnostic script: generate background with varying kstarfrac and rhofrac,
and plot distributions for each combination.

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


def plot_diagnostics(data, kstarfrac, rhofrac, outname):
    """
    Plot 6 panels for one (kstarfrac, rhofrac) combination:
      m_D, m', theta', m2(Kpi), m2(pipi), m' vs theta'
    """
    hist_specs = [
        ("md",         r"$m_D$ (GeV)",            (1.77, 2.17), 80),
        ("mprime",     r"$m'$",                    (0, 1),       80),
        ("thetaprime", r"$\theta'$",               (0, 1),       80),
        ("m2kpi",      r"$m^2(K\pi)$ (GeV$^2$)",  None,         80),
        ("m2pipi",     r"$m^2(\pi\pi)$ (GeV$^2$)", None,        80),
    ]

    fig, axes = plt.subplots(1, 6, figsize=(30, 5))
    fig.suptitle(f"kstarfrac={kstarfrac:.1f}, rhofrac={rhofrac:.1f}", fontsize=16)

    for j, (key, xlabel, xlim, nbins) in enumerate(hist_specs):
        ax = axes[j]
        d = data[key]
        if xlim is None:
            xlim = (np.percentile(d, 0.5), np.percentile(d, 99.5))
        ax.hist(d, bins=nbins, range=xlim, histtype="stepfilled",
                alpha=0.7, color="C0", density=True)
        ax.set_xlabel(xlabel)
        ax.minorticks_on()
        ax.tick_params(which="both", top=True, right=True)

    # 2D: m' vs theta'
    ax = axes[5]
    ax.hist2d(data["mprime"], data["thetaprime"], bins=60,
              range=[[0, 1], [0, 1]], cmap="afmhot_r")
    ax.set_xlabel(r"$m'$")
    ax.set_ylabel(r"$\theta'$")

    plt.tight_layout()
    plt.savefig(outname, dpi=200)
    print(f"Saved {outname}")
    plt.close()


def plot_summary_grid(all_results, labels, plotdir):
    """
    Plot 6 summary figures across all (kstarfrac, rhofrac) combinations:
      1. 2D m' vs theta'
      2. m' vs m_D
      3. theta' vs m_D
      4. m_D entries (1D)
      5. theta' entries (1D)
      6. m' entries (1D)
    """
    n = len(all_results)
    ncols = 6
    nrows = (n + ncols - 1) // ncols

    # 1) 2D m' vs theta'
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 5 * nrows), squeeze=False)
    for i, (data, label) in enumerate(zip(all_results, labels)):
        ax = axes[i // ncols, i % ncols]
        ax.hist2d(data["mprime"], data["thetaprime"], bins=60,
                  range=[[0, 1], [0, 1]], cmap="afmhot_r")
        ax.set_xlabel(r"$m'$")
        ax.set_ylabel(r"$\theta'$")
        ax.set_title(label)
    for i in range(n, nrows * ncols):
        axes[i // ncols, i % ncols].axis("off")
    plt.tight_layout()
    plt.savefig(os.path.join(plotdir, "summary_2d_mprime_thetaprime.png"), dpi=200)
    print(f"Saved summary_2d_mprime_thetaprime.png")
    plt.close()

    # 2) 2D m' vs m_D
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 5 * nrows), squeeze=False)
    for i, (data, label) in enumerate(zip(all_results, labels)):
        ax = axes[i // ncols, i % ncols]
        ax.hist2d(data["mprime"], data["md"], bins=60,
                  range=[[0, 1], [1.77, 2.17]], cmap="afmhot_r")
        ax.set_xlabel(r"$m'$")
        ax.set_ylabel(r"$m_D$ (GeV)")
        ax.set_title(label)
    for i in range(n, nrows * ncols):
        axes[i // ncols, i % ncols].axis("off")
    plt.tight_layout()
    plt.savefig(os.path.join(plotdir, "summary_2d_mprime_md.png"), dpi=200)
    print(f"Saved summary_2d_mprime_md.png")
    plt.close()

    # 3) 2D theta' vs m_D
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 5 * nrows), squeeze=False)
    for i, (data, label) in enumerate(zip(all_results, labels)):
        ax = axes[i // ncols, i % ncols]
        ax.hist2d(data["thetaprime"], data["md"], bins=60,
                  range=[[0, 1], [1.77, 2.17]], cmap="afmhot_r")
        ax.set_xlabel(r"$\theta'$")
        ax.set_ylabel(r"$m_D$ (GeV)")
        ax.set_title(label)
    for i in range(n, nrows * ncols):
        axes[i // ncols, i % ncols].axis("off")
    plt.tight_layout()
    plt.savefig(os.path.join(plotdir, "summary_2d_thetaprime_md.png"), dpi=200)
    print(f"Saved summary_2d_thetaprime_md.png")
    plt.close()

    # 4) 1D m_D entries
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4 * nrows), squeeze=False)
    for i, (data, label) in enumerate(zip(all_results, labels)):
        ax = axes[i // ncols, i % ncols]
        ax.hist(data["md"], bins=80, range=(1.77, 2.17), histtype="stepfilled",
                alpha=0.7, color="C0", density=True)
        ax.set_xlabel(r"$m_D$ (GeV)")
        ax.set_title(label)
        ax.minorticks_on()
        ax.tick_params(which="both", top=True, right=True)
    for i in range(n, nrows * ncols):
        axes[i // ncols, i % ncols].axis("off")
    plt.tight_layout()
    plt.savefig(os.path.join(plotdir, "summary_1d_md.png"), dpi=200)
    print(f"Saved summary_1d_md.png")
    plt.close()

    # 5) 1D theta' entries
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4 * nrows), squeeze=False)
    for i, (data, label) in enumerate(zip(all_results, labels)):
        ax = axes[i // ncols, i % ncols]
        ax.hist(data["thetaprime"], bins=80, range=(0, 1), histtype="stepfilled",
                alpha=0.7, color="C0", density=True)
        ax.set_xlabel(r"$\theta'$")
        ax.set_title(label)
        ax.minorticks_on()
        ax.tick_params(which="both", top=True, right=True)
    for i in range(n, nrows * ncols):
        axes[i // ncols, i % ncols].axis("off")
    plt.tight_layout()
    plt.savefig(os.path.join(plotdir, "summary_1d_thetaprime.png"), dpi=200)
    print(f"Saved summary_1d_thetaprime.png")
    plt.close()

    # 6) 1D m' entries
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4 * nrows), squeeze=False)
    for i, (data, label) in enumerate(zip(all_results, labels)):
        ax = axes[i // ncols, i % ncols]
        ax.hist(data["mprime"], bins=80, range=(0, 1), histtype="stepfilled",
                alpha=0.7, color="C0", density=True)
        ax.set_xlabel(r"$m'$")
        ax.set_title(label)
        ax.minorticks_on()
        ax.tick_params(which="both", top=True, right=True)
    for i in range(n, nrows * ncols):
        axes[i // ncols, i % ncols].axis("off")
    plt.tight_layout()
    plt.savefig(os.path.join(plotdir, "summary_1d_mprime.png"), dpi=200)
    print(f"Saved summary_1d_mprime.png")
    plt.close()


def main():
    nev = int(float(sys.argv[1])) if len(sys.argv) > 1 else 4000000

    atfi.set_seed(42)

    os.makedirs("plots", exist_ok=True)

    # Scan rhofrac for fixed kstarfrac values
    scan = {
        0.1: np.arange(0.0, 0.6, 0.1),    # kstarfrac=0.1, rhofrac 0~0.5
        0.2: np.arange(0.0, 0.6, 0.1),    # kstarfrac=0.2, rhofrac 0~0.5
        0.3: np.arange(0.0, 0.6, 0.1),    # kstarfrac=0.3, rhofrac 0~0.5
    }

    all_results = []
    labels = []

    for kf, rho_values in scan.items():
        for rf in rho_values:
            print(f"\n=== kstarfrac={kf:.1f}, rhofrac={rf:.1f} ===")
            data = generate_mix(kf, rf, nev)
            plot_diagnostics(data, kf, rf, os.path.join("plots", f"diagnose_kf{kf:.1f}_rf{rf:.1f}.png"))
            all_results.append(data)
            labels.append(f"K*={kf:.1f}, rho={rf:.1f}")

    plot_summary_grid(all_results, labels, "plots")


if __name__ == "__main__":
    main()
