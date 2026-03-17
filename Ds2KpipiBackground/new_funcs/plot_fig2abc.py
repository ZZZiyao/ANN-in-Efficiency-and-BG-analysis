import uproot
import numpy as np
import matplotlib.pyplot as plt

# ------------------ style (ROOT-ish) ------------------
plt.rcParams.update({
    "font.size": 14,
    "axes.labelsize": 16,
    "xtick.direction": "in",
    "ytick.direction": "in",
})

def hist2d_avg_density_1(x, y, bins, ranges):
    """
    Build 2D histogram and normalise such that the *average density* equals 1.

    Steps:
      1) H_ij = counts
      2) rho_ij = H_ij / (N * dx * dy)  -> true density, integral = 1
      3) rho'_ij = rho_ij / mean(rho_ij) -> average density = 1 (paper convention)

    Returns:
      rho_norm (2D array), xedges, yedges
    """
    H, xedges, yedges = np.histogram2d(x, y, bins=bins, range=ranges)

    # Protect against empty histogram
    N = H.sum()
    if N <= 0:
        return H, xedges, yedges

    # bin areas (assumes uniform bins; histogram2d produces uniform edges for int bins)
    dx = xedges[1] - xedges[0]
    dy = yedges[1] - yedges[0]

    rho = H / (N * dx * dy)  # density
    m = rho.mean()
    if m > 0:
        rho = rho / m        # average density = 1

    return rho, xedges, yedges

def plot_2d(rho, xedges, yedges, xlabel, ylabel, title, outname,
            vmin=0.0, vmax=3.0, cmap="afmhot_r"):
    plt.figure(figsize=(6, 5))

    # ROOT-like blocky histogram
    pcm = plt.pcolormesh(
        xedges, yedges, rho.T,
        shading="auto",
        cmap=cmap,
        vmin=vmin, vmax=vmax
    )

    plt.minorticks_on()
    plt.tick_params(which="both", top=True, right=True)

    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)

    cbar = plt.colorbar(pcm)
    cbar.set_label("Normalised density (⟨ρ⟩ = 1)")

    plt.tight_layout()
    plt.savefig(outname, dpi=300)
    plt.show()

# ------------------ main ------------------
infile = "test_ratio.root"   # <-- change if needed
f = uproot.open(infile)
tree = f[f.keys()[0]]

# (a) m' vs theta'
arr = tree.arrays(["mprime", "thetaprime"], library="np")
m = arr["mprime"]
t = arr["thetaprime"]

rho, xedges, yedges = hist2d_avg_density_1(
    m, t,
    bins=[60, 60],
    ranges=[[0.0, 1.0], [0.0, 1.0]]
)
plot_2d(
    rho, xedges, yedges,
    xlabel="m'",
    ylabel="θ'",
    title="(a)",
    outname="fig2a.png",
    vmin=0.0, vmax=5.0
)

# (b) m' vs m_D
arr = tree.arrays(["mprime", "md"], library="np")
m = arr["mprime"]
mD = arr["md"]

# Use paper-like window (you used 1.77-2.17 elsewhere); change if your paper uses different
md_lo, md_hi = 1.77, 2.17

rho, xedges, yedges = hist2d_avg_density_1(
    m, mD,
    bins=[60, 60],
    ranges=[[0.0, 1.0], [md_lo, md_hi]]
)
plot_2d(
    rho, xedges, yedges,
    xlabel="m'",
    ylabel="m_D (GeV)",
    title="(b)",
    outname="fig2b.png",
    vmin=0.0, vmax=3.0
)

# (c) theta' vs m_D
arr = tree.arrays(["thetaprime", "md"], library="np")
t = arr["thetaprime"]
mD = arr["md"]

rho, xedges, yedges = hist2d_avg_density_1(
    t, mD,
    bins=[60, 60],
    ranges=[[0.0, 1.0], [md_lo, md_hi]]
)
plot_2d(
    rho, xedges, yedges,
    xlabel="θ'",
    ylabel="m_D (GeV)",
    title="(c)",
    outname="fig2c.png",
    vmin=0.0, vmax=3.0
)