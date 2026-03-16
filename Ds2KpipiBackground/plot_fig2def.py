import uproot
import numpy as np
import matplotlib.pyplot as plt

# -------- style (ROOT-like) ----------
plt.rcParams.update({
    "font.size": 14,
    "axes.labelsize": 16,
    "xtick.direction": "in",
    "ytick.direction": "in",
})

# ---------------- hard-boundary regions ----------------
def define_regions_hard(md, lo=1.77, s1=1.92, s2=2.02, hi=2.17):
    """
    Hard boundaries (as you requested):
      lower  : [lo,  s1)
      signal : [s1, s2]
      upper  : (s2, hi]
    plus full window (lo, hi).

    Returns: signal_mask, lower_mask, upper_mask, full_mask
    """
    full   = (md > lo) & (md < hi)
    lowSB  = full & (md >= lo) & (md <  s1)
    signal = full & (md >= s1) & (md <= s2)
    upSB   = full & (md >  s2) & (md <= hi)
    return signal, lowSB, upSB, full

def hist_errorbar(x, bins, mask=None, label=None,
                  fmt="k.", ls="none", lw=1.2, markersize=4,
                  density=False, scale_to=None):
    """
    density=False: plot raw counts per bin with sqrt(N) error
    density=True : plot probability density (area=1), errors scaled accordingly
    scale_to: None or target total yield (scale histogram to this total count)
    """
    xx = x if mask is None else x[mask]

    counts, edges = np.histogram(xx, bins=bins)
    centers = 0.5 * (edges[1:] + edges[:-1])
    widths  = edges[1:] - edges[:-1]

    y = counts.astype(float)
    yerr = np.sqrt(counts)

    # optional scaling to match a target total yield
    if scale_to is not None:
        total = y.sum()
        if total > 0:
            s = scale_to / total
            y *= s
            yerr *= s

    # optional density normalisation (area=1)
    if density:
        area = (y * widths).sum()
        if area > 0:
            y /= area
            yerr /= area

    plt.errorbar(
        centers, y, yerr=yerr,
        fmt=fmt, linestyle=ls, linewidth=lw,
        markersize=markersize, capsize=0,
        label=label
    )
    return centers, y, yerr

# ---------------- read data ----------------
infile = "test_ratio.root"   # change if needed
f = uproot.open(infile)
tree = f[f.keys()[0]]

arr = tree.arrays(["mprime", "thetaprime", "md"], library="np")
m  = arr["mprime"]
t  = arr["thetaprime"]
md = arr["md"]

# ---------------- region parameters ----------------
lo, s1, s2, hi = 1.77, 1.92, 2.02, 2.17
signal, lowSB, upSB, full = define_regions_hard(md, lo=lo, s1=s1, s2=s2, hi=hi)

Ns   = int(signal.sum())
Nlow = int(lowSB.sum())
Nup  = int(upSB.sum())
Nall = int(full.sum())

print("Counts (hard boundaries):")
print("  full  :", Nall)
print("  signal:", Ns)
print("  lowSB :", Nlow)
print("  upSB  :", Nup)

# =========================================================
# (d) m_D with signal window lines (error bars)
# =========================================================
plt.figure(figsize=(6, 5))

# Paper label says /0.004 GeV. Use 0.04 bins to match exactly.
bins_md = np.arange(lo, hi + 1e-12, 0.004)
hist_errorbar(md, bins_md, mask=full, fmt="k.", ls="none")

plt.axvline(s1, color="red", linestyle="--", linewidth=1.5)
plt.axvline(s2, color="red", linestyle="--", linewidth=1.5)

plt.minorticks_on()
plt.tick_params(which="both", top=True, right=True)
plt.ylim(bottom=0)

plt.xlabel("m_D (GeV)")
plt.ylabel("Entries / (0.004 GeV)")
plt.title("(d)")
plt.tight_layout()
plt.savefig("fig2d.png", dpi=300)
plt.show()

# =========================================================
# (e) m' projections in signal / sidebands (error bars)
# =========================================================
plt.figure(figsize=(6, 5))

bins_m = np.arange(0, 1.0 + 1e-12, 0.01)  # paper: Entries/(0.01)

# If you want raw yields (signal likely lower), leave scale_to=None.
# If you want curves comparable like the paper, scale sidebands to Ns:
scale_sidebands_to_signal = True

hist_errorbar(m, bins_m, mask=signal, label="Signal region", fmt="k.", ls="none")

if scale_sidebands_to_signal:
    hist_errorbar(m, bins_m, mask=lowSB, label="Lower sideband",
                  fmt="b", ls="--", lw=1.5, scale_to=Ns)
    hist_errorbar(m, bins_m, mask=upSB, label="Upper sideband",
                  fmt="r", ls="--", lw=1.5, scale_to=Ns)
else:
    hist_errorbar(m, bins_m, mask=lowSB, label="Lower sideband",
                  fmt="b", ls="--", lw=1.5)
    hist_errorbar(m, bins_m, mask=upSB, label="Upper sideband",
                  fmt="r", ls="--", lw=1.5)

plt.minorticks_on()
plt.tick_params(which="both", top=True, right=True)
plt.ylim(bottom=0)

plt.xlabel("m'")
plt.ylabel("Entries / (0.01)")
plt.title("(e)")
plt.legend(frameon=False)
plt.tight_layout()
plt.savefig("fig2e.png", dpi=300)
plt.show()

# =========================================================
# (f) theta' projections in signal / sidebands (error bars)
# =========================================================
plt.figure(figsize=(6, 5))

bins_t = np.arange(0, 1.0 + 1e-12, 0.01)  # paper: Entries/(0.01)

hist_errorbar(t, bins_t, mask=signal, label="Signal region", fmt="k.", ls="none")

if scale_sidebands_to_signal:
    hist_errorbar(t, bins_t, mask=lowSB, label="Lower sideband",
                  fmt="b", ls="--", lw=1.5, scale_to=Ns)
    hist_errorbar(t, bins_t, mask=upSB, label="Upper sideband",
                  fmt="r", ls="--", lw=1.5, scale_to=Ns)
else:
    hist_errorbar(t, bins_t, mask=lowSB, label="Lower sideband",
                  fmt="b", ls="--", lw=1.5)
    hist_errorbar(t, bins_t, mask=upSB, label="Upper sideband",
                  fmt="r", ls="--", lw=1.5)

plt.minorticks_on()
plt.tick_params(which="both", top=True, right=True)
plt.ylim(bottom=0)

plt.xlabel("θ'")
plt.ylabel("Entries / (0.01)")
plt.title("(f)")
plt.legend(frameon=False)
plt.tight_layout()
plt.savefig("fig2f.png", dpi=300)
plt.show()