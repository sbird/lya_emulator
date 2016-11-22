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
    params = coarse_grid.Emulator(emulatordir)
    params.load()
    data = gpemulator.SDSSData()
    gp = params.get_emulator(mean_flux=mean_flux, max_z=max_z)
    params_test = coarse_grid.Emulator(testdir)
    params_test.load()
    myspec = flux_power.MySpectra(max_z=max_z)
    #Constant mean flux.
    if mean_flux:
        mf = 0.3
    else:
        mf = None
    for pp in params_test.get_parameters():
        dd = params_test.get_outdir(pp)
        if mean_flux:
            pp = np.append(pp, mf)
        predicted,_ = gp.predict(pp)
        exact = myspec.get_flux_power(dd,data.get_kf(),mean_flux=mf,flat=True)
        ratio = predicted[0]/exact
        nred = len(myspec.zout)
        nk = len(data.get_kf())
        assert np.shape(ratio) == (nred*nk,)
        for i in range(nred):
            plt.loglog(data.get_kf(),ratio[i*nk:(i+1)*nk],label=myspec.zout[i])
        plt.xlabel(r"$k_F$ (s/km)")
        plt.ylabel(r"Predicted/Exact")
        name = params_test.build_dirname(pp)
        plt.title(name)
        plt.legend(loc=0)
        plt.show()
        if mean_flux:
            fname = name+"mf"+str(mf)+".pdf"
        else:
            fname = name+".pdf"
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
        predicted,_ = gp.predict(pp)
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
