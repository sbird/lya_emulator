"""Plot leave one out errors"""
import h5py
import numpy as np
from matplotlib import pyplot as plt

basedir = 'dtau-48-48/'
figbase = '../figures/'

c_midnight = "grey"
c_sunshine = "gold"
# leave-one-out for LF FPS
ff = h5py.File(basedir+'loo_fps.hdf5', 'r')
fpp, fpt, pp, std = ff['flux_predict'][:], ff['flux_true'][:], ff['params'][:], ff['std_predict'][:]
ff.close()
err = np.abs(fpp/fpt-1).flatten()
errSTD = ((fpp-fpt)/std).flatten()

# leave-one-out for HF FPS (MF)
ff = h5py.File(basedir+'hires/loo_fps.hdf5', 'r')
fpp, fpt, pp, std = ff['flux_predict'][:], ff['flux_true'][:], ff['params'][:], ff['std_predict'][:]
ff.close()
MFerr = np.abs(fpp/fpt-1).flatten()
MFerrSTD = ((fpp-fpt)/std).flatten()

# FPS LOO plot
fig, ax = plt.subplots(figsize=(10.625*2, 8), nrows=1, ncols=2, sharey=True)
bins = np.linspace(-5, 5, 300)
ax[0].plot(bins, 4200*np.exp(-bins**2*0.5), color="black")
ax[0].hist(errSTD, bins=bins, color=c_midnight, histtype='stepfilled', alpha=0.9)
bins = np.linspace(-3, 3, 56)
ax[0].hist(MFerrSTD, bins=bins, color=c_sunshine, histtype='stepfilled', alpha=0.55, weights=np.ones(MFerr.size)*48/3*56/200)
#ax[0].set_yticks([])
ax[0].set_xticks([-4,-3,-2,-1,0, 1,2,3,4], labels=[-4,-3,-2,-1,0, 1,2,3,4], fontsize=20)
ax[0].tick_params(which='both', direction='inout', right=True, labelright=True, labelleft=False, length=12)
ax[0].tick_params(which='minor', length=8, labelright=False, labelleft=False)
ax[0].set_xlabel(r'$\left(P_F^{{pred}}-P_F^{{true}}\right)/\sigma^{{pred}}$', fontsize=26)
ax[0].set_xlim([-4,4])
# ax[0].set_yscale('log')
# ax[0].set_ylim([1,1e4])

logbins = np.logspace(np.log10(err.min()), np.log10(err.max()), 280)
ax[1].hist(err, bins=logbins, color=c_midnight, histtype='stepfilled', alpha=0.9, label='Single-Fidelity, LF')
logbins = np.logspace(np.log10(MFerr.min()), np.log10(MFerr.max()), 56)
ax[1].hist(MFerr, bins=logbins, color=c_sunshine, histtype='stepfilled', alpha=0.55, label='Multi-Fidelity, HF', weights=np.ones(MFerr.size)*48/3*56/280)
ax[1].tick_params(which='both', direction='inout', right=True, labelright=True, labelleft=False, length=12)
ax[1].tick_params(which='minor', length=8, labelright=False, labelleft=False)
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
ax[1].set_xlabel(r'$\|P_F^{{pred}}/P_F^{{true}}-1\|$', fontsize=26)
ax[1].set_xscale('log')
ax[1].set_xlim([2e-5,0.1])
ax[1].legend(loc='upper left', fontsize=26)

fig.patch.set_facecolor('none')
fig.subplots_adjust(hspace=0, wspace=0)
fig.savefig(figbase+"fpsemu_errors.pdf", bbox_inches='tight', pad_inches=0.075)
plt.show()

# leave-one-out for T0
ff = h5py.File(basedir+'loo_t0.hdf5', 'r')
fpp, fpt, pp, std = ff['meanT_predict'][:], ff['meanT_true'][:], ff['params'][:], ff['std_predict'][:]
ff.close()
err = np.abs(fpp/fpt-1).flatten()
errSTD = ((fpp-fpt)/std).flatten()

# leave-one-out MF T0
ff = h5py.File(basedir+'hires/loo_t0.hdf5', 'r')
fpp, fpt, pp, std = ff['meanT_predict'][:], ff['meanT_true'][:], ff['params'][:], ff['std_predict'][:]
ff.close()
MFerr = np.abs(fpp/fpt-1).flatten()
MFerrSTD = ((fpp-fpt)/std).flatten()

# T0 LOO plot
fig, ax = plt.subplots(figsize=(10.625*2, 8), nrows=1, ncols=2, sharey=True)
bins = np.linspace(-5, 5, 25)
ax[0].hist(errSTD, bins=bins, color=c_midnight, histtype='stepfilled', alpha=0.9)
ax[0].hist(MFerrSTD, bins=bins, color=c_sunshine, histtype='stepfilled', alpha=0.55, weights=np.ones(MFerr.size)*46/3)
ax[0].plot(bins, 60*np.exp(-bins**2*0.5), color="black")
ax[0].set_xticks([-4,-2,0, 2,4], labels=[-4,-2,0, 2,4], fontsize=20)
ax[0].set_xlim([-5,5])
ax[0].tick_params(which='both', direction='inout', right=True, labelright=True, labelleft=False, length=12)
ax[0].tick_params(which='minor', length=8, labelright=False, labelleft=False)
ax[0].set_xlabel(r'$\left(T_0^{{pred}}-T_0^{{true}}\right)/\sigma^{{pred}}$', fontsize=26)

logbins = np.logspace(np.log10(err.min()), np.log10(err.max()), 25)
ax[1].hist(err, bins=logbins, color=c_midnight, histtype='stepfilled', alpha=0.9, label='Single-Fidelity, LF')
logbins = np.logspace(np.log10(MFerr.min()), np.log10(MFerr.max()), 15)
ax[1].hist(MFerr, bins=logbins, color=c_sunshine, histtype='stepfilled', alpha=0.5, weights=np.ones(MFerr.size)*46/3, label='Multi-Fidelity, HF')
ax[1].tick_params(which='both', direction='inout', right=True, labelright=True, labelleft=False, length=12)
ax[1].tick_params(which='minor', length=8, labelright=False, labelleft=False)
ax[1].set_xlabel(r'$\|T_0^{{pred}}/T_0^{{true}}-1\|$', fontsize=26)
ax[1].set_xscale('log')
ax[1].set_xlim([5e-5,2e-1])
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
ax[1].legend(loc='upper left', fontsize=26)

fig.patch.set_facecolor('none')
fig.subplots_adjust(hspace=0, wspace=0)
fig.savefig(figbase+"t0emu_errors.pdf", bbox_inches='tight', pad_inches=0.075)
plt.show()
