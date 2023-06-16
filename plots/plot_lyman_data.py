"""Quick script to plot available Lyman alpha forest flux power spectrum data"""

import numpy as np
import matplotlib.pyplot as plt
from lyaemu import lyman_data as lyd

# set up the ticks and axes
plt.rc('xtick',labelsize=20)
plt.rc('ytick',labelsize=20)
plt.rcParams['xtick.major.size'] = 8
plt.rcParams['xtick.major.width'] = 1.75
plt.rcParams['xtick.minor.size'] = 5
plt.rcParams['xtick.minor.width'] = 1.25
plt.rcParams['ytick.major.size'] = 8
plt.rcParams['ytick.major.width'] = 1.75
plt.rcParams['ytick.minor.size'] = 5
plt.rcParams['ytick.minor.width'] = 1.25
plt.rcParams['axes.linewidth'] = 2
plt.rcParams['axes.labelsize'] = 20
plt.rcParams['legend.fontsize'] = 17
#     gdplot.settings.axes_fontsize = 20
#     gdplot.settings.axes_labelsize = 28
#     gdplot.settings.legend_fontsize = 34

#Kodiaq
kodiaq = lyd.KSData(conservative=True)
#DR14 data
boss = lyd.BOSSData()
#DR9 data
bossdr9 = lyd.BOSSData(datafile="dr9")
#Old SDSS data
sdss = lyd.SDSSData()
#XQ100
xq100 = lyd.XQ100Data()

koqk = kodiaq.get_kf()
bok = boss.get_kf()
bok9 = bossdr9.get_kf()
sdk = sdss.get_kf()
xqk = xq100.get_kf()

#zz = np.array([2.2,2.4,2.6])
zz = np.arange(2.2, 4.8, 0.2)

sigma = 1
for z in zz:
#     plt.figure()
    kpf = koqk * kodiaq.get_pf(zbin=z) / np.pi
    plt.plot(koqk, kpf, label="KODIAQ z=%.1f" % z, ls="--", color="blue")
    koqstd = koqk * np.sqrt(kodiaq.get_covar_diag(zbin=z))
    plt.fill_between(koqk, kpf-sigma*koqstd , kpf+sigma*koqstd , alpha=0.25, color="blue")

    bpf = bok * boss.get_pf(zbin=z) / np.pi
    plt.plot(bok, bpf, label="DR14 z=%.1f" % z, color="black")
    #Standard errors
    bstd = bok * np.sqrt(np.diag(boss.get_covar(zbin=z)))
    plt.fill_between(bok, bpf-sigma*bstd, bpf+sigma*bstd, alpha=0.5, color="grey")
    xqpf = xq100.get_pf(zbin=z)
    if np.size(xqpf) > 0:
        xqpf *=xqk / np.pi
        plt.plot(xqk, xqpf, label="XQ100 z=%.1f" % z, ls="-.", color="orange")
    bpf9 = bossdr9.get_pf(zbin=z)
    if np.size(bpf9) > 0:
        bpf9*=bok9 / np.pi
        plt.plot(bok9, bpf9, label="DR9 z=%.1f" % z, ls=":", color="green")
    # sdf = sdss.get_pf(zbin=z)
    # if np.size(sdf) > 0:
        # sdf*=sdk
        # plt.plot(sdk, sdf, label="SDSS z=%.1f" % z, ls="-.")
#     plt.yscale('linear')
    plt.xlim(1e-3, 0.02)
#     plt.ylim(0.8*np.min(bpf), 1.2*np.max([np.max(kpf), np.max(bpf)]))
    plt.ylim(0., 1.1*np.max([np.max(kpf), np.max(bpf)]))
    plt.xlabel(r"$k_F$ (s/km)")
    plt.ylabel(r"$k P_F(k) / \pi$")
    plt.legend()
    plt.tight_layout()
    plt.savefig("lymandata-z%.1f.pdf" % z)
    plt.clf()

zz2 = np.arange(2.0, 2.2)
for z in zz2:
#     plt.figure()
    kpf = kodiaq.get_pf(zbin=z)
    plt.plot(koqk, kpf, label="KODIAQ z=%.1f" % z, ls="--", color="blue")
    koqstd = np.sqrt(kodiaq.get_covar_diag(zbin=z))
    plt.fill_between(koqk, kpf-sigma*koqstd , kpf+sigma*koqstd , alpha=0.25, color="blue")
    plt.xlim(1e-3, 0.02)
    plt.ylim(0., 20)
    plt.xlabel(r"$k_F$ (s/km)")
    plt.ylabel(r"$k P_F(k)$")
    plt.legend()
    plt.tight_layout()
    plt.savefig("lymandata-z%.1f.pdf" % z)
    plt.clf()
