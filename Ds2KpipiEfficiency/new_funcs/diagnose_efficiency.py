"""
Diagnostic script for the efficiency model.
Plot m' and theta' distributions before and after selection,
plus 2D efficiency map.
Uses selection() from DistributionModel with true_cuts.
"""
import sys
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

import amplitf.interface as atfi

from DistributionModel import (
    true_cuts, selection, observables_phase_space, random_array_size,
)

plt.rcParams.update({
    "font.size": 12,
    "axes.labelsize": 14,
    "xtick.direction": "in",
    "ytick.direction": "in",
})


def generate_before_after(nev, chunk_size=1000000):
    """
    Generate Dalitz plot samples before and after selection.
    Returns (before, after) where each is dict with mprime, thetaprime.
    """
    all_before_mp, all_before_tp = [], []
    all_after_mp, all_after_tp = [], []
    n_after = 0

    while n_after < nev:
        # Uniform sample in square Dalitz plot
        unfiltered = observables_phase_space.unfiltered_sample(chunk_size)
        sample = observables_phase_space.filter(unfiltered)
        size = sample.shape[0]

        all_before_mp.append(sample[:, 0].numpy())
        all_before_tp.append(sample[:, 1].numpy())

        # Apply selection
        rnd = tf.random.uniform([size, random_array_size], dtype=atfi.fptype())
        obs = selection(sample, true_cuts, rnd, constant_cuts=True)
        # obs = [mprime, thetaprime]
        all_after_mp.append(obs[0].numpy())
        all_after_tp.append(obs[1].numpy())

        n_after += obs[0].shape[0]
        print(f"  after selection: {n_after}/{nev}")

    before = {
        "mprime": np.concatenate(all_before_mp),
        "thetaprime": np.concatenate(all_before_tp),
    }
    after = {
        "mprime": np.concatenate(all_after_mp)[:nev],
        "thetaprime": np.concatenate(all_after_tp)[:nev],
    }
    return before, after


def main():
    nev = int(float(sys.argv[1])) if len(sys.argv) > 1 else 200000
    atfi.set_seed(42)

    print("Generating samples ...")
    before, after = generate_before_after(nev)

    # --- 1D histograms: before vs after ---
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    for j, (key, xlabel) in enumerate([("mprime", r"$m'$"), ("thetaprime", r"$\theta'$")]):
        ax = axes[j]
        ax.hist(before[key], bins=80, range=(0, 1), histtype="step",
                label="before selection", color="C0", density=True, linewidth=1.5)
        ax.hist(after[key], bins=80, range=(0, 1), histtype="step",
                label="after selection", color="C1", density=True, linewidth=1.5)
        ax.set_xlabel(xlabel)
        ax.set_ylabel("density")
        ax.legend(fontsize=10)
        ax.minorticks_on()
        ax.tick_params(which="both", top=True, right=True)

    plt.tight_layout()
    plt.savefig("diagnose_eff_1d.png", dpi=200)
    print("Saved diagnose_eff_1d.png")
    plt.close()

    # --- 2D plots: before, after, ratio (efficiency) ---
    nbins = 50
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    for i, (data, title) in enumerate([
        (before, "Before selection"),
        (after, "After selection"),
    ]):
        ax = axes[i]
        ax.hist2d(data["mprime"], data["thetaprime"], bins=nbins,
                  range=[[0, 1], [0, 1]], cmap="afmhot_r")
        ax.set_xlabel(r"$m'$")
        ax.set_ylabel(r"$\theta'$")
        ax.set_title(title)

    # Efficiency = after / before (bin-by-bin ratio)
    H_before, xedges, yedges = np.histogram2d(
        before["mprime"], before["thetaprime"], bins=nbins, range=[[0, 1], [0, 1]])
    H_after, _, _ = np.histogram2d(
        after["mprime"], after["thetaprime"], bins=nbins, range=[[0, 1], [0, 1]])

    with np.errstate(divide="ignore", invalid="ignore"):
        eff = np.where(H_before > 0, H_after / H_before, 0.0)

    ax = axes[2]
    im = ax.imshow(eff.T, origin="lower", extent=[0, 1, 0, 1],
                   aspect="auto", cmap="afmhot_r")
    ax.set_xlabel(r"$m'$")
    ax.set_ylabel(r"$\theta'$")
    ax.set_title(r"Efficiency $\varepsilon(m', \theta')$")
    plt.colorbar(im, ax=ax)

    plt.tight_layout()
    plt.savefig("diagnose_eff_2d.png", dpi=200)
    print("Saved diagnose_eff_2d.png")
    plt.close()


if __name__ == "__main__":
    main()
