"""File to make a temperature density plot, weighted by HI fraction"""

import os.path
import numpy as np
from fake_spectra import abstractsnapshot as absn
from ratenetworkspectra import RateNetworkGas
import matplotlib
matplotlib.use("PDF")
import matplotlib.pyplot as plt

def maketdplot(num, base, nhi=True, nbins=500):
    """Make a temperature density plot of neutral hydrogen."""
    snap = absn.AbstractSnapshotFactory(num, base)

    redshift = 1./snap.get_header_attr("Time") - 1
    hubble = 1./snap.get_header_attr("HubbleParam") - 1
    rates = RateNetworkGas(redshift, snap, hubble)

    dens = snap.get_data(0, "Density", -1)

    temp = rates.get_temp(0, -1)

    dens = rates.get_code_rhoH(0, -1)

    if nhi:
        nhi = rates.get_reproc_HI(0, -1)
    else:
        nhi = dens

    hist, dedges, tedges = np.histogram2d(np.log10(dens), np.log10(temp), bins=nbins, weights=nhi, density=True)

    plt.imshow(hist.T, interpolation='nearest', origin='low', extent=[dedges[0], dedges[-1], tedges[0], tedges[-1]], cmap=plt.cm.cubehelix_r, vmax=0.75, vmin=0.01)
    dd = np.array([-6,-5,-4,-3])
    plt.xticks(dd, [r"$10^{%d}$" % d for d in dd])
    tt = np.array([2000, 3000, 5000, 10000, 20000, 30000, 50000, 100000])
    plt.yticks(np.log10(tt), tt//1000)
    plt.ylabel(r"T ($10^3$ K)")
    plt.xlabel(r"$\rho$ (cm$^{-3}$)")

    plt.xlim(-6.3,-3)
    plt.ylim(3.4,5)
    #plt.colorbar()
    plt.tight_layout()
    return hist.T, dedges, tedges

if __name__ == "__main__":
    maketdplot(10, os.path.expanduser("~/data/Lya_Boss/hires_s8_test/ns0.97As2.2e-09heat_slope0.083heat_amp0.92hub0.69/output/"))
    plt.savefig("plots/tempdens.pdf")
