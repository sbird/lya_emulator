"""Generate a test plot for an emulator"""
from __future__ import print_function
import numpy as np
import matplotlib
import gpemulator
import coarse_grid
import flux_power
matplotlib.use('PDF')
import matplotlib.pyplot as plt


def plot_test_interpolate(emulatordir,testdir, mean_flux=True):
    """Make a plot showing the interpolation error."""
    params = coarse_grid.Params(emulatordir)
    params.load()
    data = gpemulator.SDSSData()
    gp = params.get_emulator(data.get_kf(), mean_flux=mean_flux)
    params_test = coarse_grid.Params(testdir)
    params_test.load()
    myspec = flux_power.MySpectra()
    #Constant mean flux.
    if mean_flux:
        mf = 0.3
    else:
        mf = None
    for pp in params_test.get_parameters():
        dd = params_test.get_outdir(pp)
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
            plt.savefig(name+"mf"+str(mf)+".pdf")
        else:
            plt.savefig(name+".pdf")
        print(name+".pdf")
        plt.clf()
    return gp
