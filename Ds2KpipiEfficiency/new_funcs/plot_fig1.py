import uproot
import numpy as np
import matplotlib.pyplot as plt

# -------- ROOT style tweaks ----------
plt.rcParams.update({
    "font.size": 14,
    "axes.labelsize": 16,
    "xtick.direction": "in",
    "ytick.direction": "in",
})

# -------- readdata ----------
f = uproot.open("highstat.root")
tree = f[f.keys()[0]]
arr = tree.arrays(["mprime", "thetaprime"], library="np")

m = arr["mprime"]
t = arr["thetaprime"]

# -------- 2D histogram ----------
H, xedges, yedges = np.histogram2d(
    m, t,
    bins=50,
    range=[[0,1],[0,1]]
)

# -------- normalize ----------
H = H / np.mean(H[H>0])

# -------- plot ----------
plt.figure(figsize=(6,5))

plt.imshow(
    H.T,
    origin="lower",
    extent=[0,1,0,1],
    aspect="auto",
    cmap="afmhot_r",
    vmin=0.0,
    vmax=1.3
)

plt.xlabel("m'")
plt.ylabel("θ'")
cbar = plt.colorbar()
cbar.set_label("ε(m', θ')")

plt.tight_layout()
plt.savefig("fig1_reproduce.png", dpi=300)
plt.show()
