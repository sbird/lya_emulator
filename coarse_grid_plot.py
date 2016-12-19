"""Generate a test plot for an emulator"""
from __future__ import print_function
import os.path
import numpy as np
import matplotlib
import gpemulator
import coarse_grid
import flux_power
import matter_power
matplotlib.use('PDF')
import matplotlib.pyplot as plt

def plot_test_interpolate(emulatordir,testdir, mean_flux=True, max_z=4.2):
    """Make a plot showing the interpolation error."""
    data = gpemulator.SDSSData()
    params = coarse_grid.Emulator(emulatordir)
    params.load()
    gp = params.get_emulator(mean_flux=mean_flux, max_z=max_z)
    params_test = coarse_grid.Emulator(testdir)
    params_test.load()
    myspec = flux_power.MySpectra(max_z=max_z)
    #Constant mean flux.
    if mean_flux:
        t0 = [1.,]
    else:
        t0 = None
    for pp in params_test.get_parameters():
        dd = params_test.get_outdir(pp)
        if mean_flux:
            pp = np.append(pp, np.exp(-1.*flux_power.obs_mean_tau(3.)))
        predicted,std = gp.predict(pp.reshape(1,-1))
        exact = myspec.get_flux_power(dd,data.get_kf(),tau0_factors=t0)
        ratio = predicted[0]/exact[0]
        upper = (predicted[0] + std[0])/exact[0]
        lower = (predicted[0] - std[0])/exact[0]
        nred = len(myspec.zout)
        nk = len(data.get_kf())
        assert np.shape(ratio) == (nred*nk,)
        for i in range(nred):
            plt.semilogx(data.get_kf(),ratio[i*nk:(i+1)*nk],label=myspec.zout[i])
            plt.fill_between(data.get_kf(),lower[i*nk:(i+1)*nk], upper[i*nk:(i+1)*nk],alpha=0.3, color="grey")
        plt.xlabel(r"$k_F$ (s/km)")
        plt.ylabel(r"Predicted/Exact")
        name = params_test.build_dirname(pp)
        plt.title(name)
        plt.legend(loc=0)
        plt.show()
        if mean_flux:
            fname = name+"mf"+str(t0[0])+".pdf"
            plt.ylim(0.95, 1.05)
        else:
            fname = name+".pdf"
            plt.ylim(0.9, 1.1)
        plt.savefig(os.path.join(testdir, fname))
        print(name+".pdf")
        plt.clf()
    return gp

def plot_test_matter_interpolate(emulatordir,testdir, redshift=3.):
    """Make a plot showing the interpolation error for the matter power spectrum."""
    params = coarse_grid.MatterPowerEmulator(emulatordir)
    params.load()
    gp = params.get_emulator()
    params_test = coarse_grid.MatterPowerEmulator(testdir)
    params_test.load()
    for pp in params_test.get_parameters():
        dd = params_test.get_outdir(pp)
        predicted = gp.predict(pp)
        exact = matter_power.get_matter_power(dd,params.kf, redshift=redshift)
        ratio = predicted[0]/exact
        name = params_test.build_dirname(pp)
        plt.semilogx(params.kf,ratio,label=name)
    plt.xlabel(r"$k$ (h/kpc)")
    plt.ylabel(r"Predicted/Exact")
    plt.title("Matter power")
    plt.legend(loc=0)
    plt.show()
    plt.savefig(testdir+"matter_power.pdf")
    print(testdir+"matter_power.pdf")
    plt.clf()
    return gp
