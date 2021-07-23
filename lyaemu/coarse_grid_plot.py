"""Generate a test plot for an emulator"""
from __future__ import print_function
import os
import os.path
import re
import math
from datetime import datetime
import numpy as np
from . import coarse_grid
from . import matter_power
from . import flux_power
from . import mean_flux as mflux
import matplotlib
matplotlib.use('PDF')
import matplotlib.pyplot as plt
from . import distinct_colours_py3 as dc

def _plot_by_redshift_bins(savedir, plotname, z_labs, all_power_array_all_kf):
    """Plot the different redshift bins on different plots"""
    ncols = 3
    nrows = math.ceil(len(z_labs) / ncols)
    figure, axes = plt.subplots(nrows=nrows, ncols=ncols)
    for z in range(all_power_array_all_kf.shape[3]): #Loop over redshift bins
        power_difference = all_power_array_all_kf[:, :, 1, z] - all_power_array_all_kf[:, :, 3, z]
        err_norm = power_difference / all_power_array_all_kf[:, :, 2, z]
        _plot_error_histogram(savedir, 'z =' + z_labs[z], err_norm.flatten(), axis=axes.flatten()[z])
    plt.tight_layout()
    figure.subplots_adjust(hspace=0.)
    plt.savefig(os.path.join(savedir, "errhist_z_bins" + plotname + ".pdf"))
    plt.clf()

def _plot_error_histogram(savedir, plotname, err_norm, axis=None, xlim=6., nbins=100, xlabel=r"(Predicted - Exact) / $1 \sigma$"):
    """Plot a histogram of the errors from the emulator with the expected errors.
       The axis keyword controls which figure we plot on."""
    if axis is None:
        plt.hist(err_norm, bins=nbins, density=True)
        xx = np.arange(-6, 6, 0.01)
#         np.savetxt(os.path.join(savedir, "table_errhist" + plotname + ".txt"), err_norm)
        _plot_unit_Gaussians(xx)
        plt.xlabel(xlabel)
        plt.ylabel("PDF")
        plt.xlim(-1. * xlim, xlim)
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(savedir, "errhist_" + plotname))
        plt.clf()
    else:
        axis.hist(err_norm, bins=nbins, density=True, label=plotname)
        xx = np.arange(-6, 6, 0.01)
        _plot_unit_Gaussians(xx, axis=axis)
        axis.set_xlabel(xlabel)
        axis.set_xlim(-1. * xlim, xlim)
        axis.legend(frameon=False, fontsize=5.)

def _plot_unit_Gaussians(xx, axis=None):
    """Plot a unit gaussian and a 2-unit gaussian"""
    if axis is None:
        axis = plt
    axis.plot(xx, np.exp(-xx ** 2 / 2) / np.sqrt(2 * np.pi), ls="-", color="black", label=r"Unit Gaussian")

def plot_test_interpolate(emulatordir,testdir, savedir=None, plotname="", mean_flux=1, max_z=4.2, emuclass=None, showerr=True):
    """Make a plot showing the interpolation error."""
    if savedir is None:
        savedir = emulatordir
    if not os.path.exists(savedir):
        os.makedirs(savedir)
    t0 = None
    if mean_flux:
        t0 = 0.9
    mftest = mflux.ConstMeanFlux(value=t0) #In 'ConstMeanFlux' case: multiply tau_0_i[z] by t0 = 0.95
    if mean_flux == 2:
        mf = mflux.MeanFluxFactor() #In 'MeanFluxFactor' case: DON'T multiply tau_0_i[z] by t0 - because *emulate* t0[z]
    else:
        mf = mftest
    params_test = coarse_grid.Emulator(testdir,mf=mftest)
    params_test.load()
    if emuclass is None:
        params = coarse_grid.Emulator(emulatordir, mf=mf)
    else:
        params = emuclass(emulatordir, mf=mf)
    params.load()
    print('Beginning to generate emulator at', str(datetime.now()))
    gp = params.get_emulator(max_z=max_z)
    print('Finished generating emulator at', str(datetime.now()))
    myspec = flux_power.MySpectra(max_z=max_z, max_k=params.maxk)
    errlist = np.array([])

    nred = len(myspec.zout)
    dist_col = dc.get_distinct(nred)
    #print("Number of validation points =", params_test.get_parameters().shape[0])
    test_par, test_kf, test_flux = params_test.get_flux_vectors()

    for pp, okf, exact in zip(test_par, test_kf, test_flux):
        if mean_flux == 2:
            pp = np.concatenate([[t0,], pp]) #In 'MeanFluxFactor' case: choose t0 point for fair comparison
        predicted,std = gp.predict(pp.reshape(1,-1)) #.predict takes [{list of parameters: t0; cosmo.; thermal},]

        #assert np.all(np.abs(gp.kf/okf - 1) < 1e-5)
        ratio =  predicted[0]/exact
        errrr =  (predicted[0]-exact)/std[0]
        errlist = np.concatenate([errlist, errrr])
        for i in range(nred):
            nk = np.size(okf[i])
            plt.semilogx(okf[i],ratio[i*nk:(i+1)*nk],label=round(myspec.zout[i],1), color=dist_col[i])

        upper =  ((predicted[0] + std[0])/exact).reshape(-1, nk)
        lower =  ((predicted[0]-std[0])/exact).reshape(-1, nk)
        low = np.min(lower, axis=0)
        low = np.concatenate([[low[0],], low])
        upp = np.max(upper, axis=0)
        upp = np.concatenate([[upp[0],], upp])
        if showerr:
            plt.fill_between(np.concatenate([[okf[0][0],], okf[-1]]),low, upp,alpha=0.3, color="grey")
        #plt.yscale('log')
        plt.xlabel(r"$k_F$ (s/km)")
        plt.ylabel(r"Predicted/Exact")
        plt.ylim(0.95,1.05)
        plt.xticks([1e-3, 1e-2, 0.05],[r"$10^{-3}$",r"$10^{-2}$","0.05"])
        name = params.build_dirname(pp, include_dense=True)
#         plt.title(name)
        plt.xlim(1e-3, 0.052)
        if np.max(ratio) > 1.035:
            plt.legend(loc='lower left', ncol=4)
        else:
            plt.legend(loc='upper left', ncol=4)
        plt.tight_layout()
        name_ending = ".pdf"
        name = re.sub(r"\.","_",str(name))+plotname+name_ending
        #So we can use it in a latex document
        plt.savefig(os.path.join(savedir, name))
        plt.clf()
        #Make plot of errors
        _plot_error_histogram(savedir, name, errrr, xlim=5., nbins=50)
        print(name)

        #Save output
        array_savename = os.path.join(savedir, name[:-4] + '.npy')
        np.save(array_savename, [okf, predicted[0], std[0], exact])

    #Plot the distribution of errors, compared to a Gaussian
    if np.all(np.isfinite(errlist)):
        _plot_error_histogram(savedir, plotname, errlist, xlim=6., nbins=100)

    return gp, myspec.zout

def plot_test_loo_interpolate(emulatordir, savedir=None, plotname="", max_z=4.2, emuclass=None, subsample=None):
    """Make a plot showing the interpolation error using a leave-one-out emulator."""
    if savedir is None:
        savedir = emulatordir
    if not os.path.exists(savedir):
        os.makedirs(savedir)
    mf = mflux.ConstMeanFlux(value=1)
    if emuclass is None:
        params = coarse_grid.Emulator(emulatordir, mf=mf)
    else:
        params = emuclass(emulatordir, mf=mf)
    params.load()
    zout = flux_power.MySpectra(max_z=max_z, max_k=params.maxk).zout
    errlist = np.array([])
    nred = len(zout)
    dist_col = dc.get_distinct(nred)
    parameters = params.get_parameters()
    nsims = np.shape(parameters)[0]
    for ii in range(nsims):
        (kf, pkdiff, errrr) = params.do_loo_cross_validation(remove=ii, max_z=max_z, subsample=subsample)
        errlist = np.concatenate([errlist, errrr])
        for i in range(nred):
            nk = np.size(kf)
            plt.semilogx(kf,pkdiff[i*nk:(i+1)*nk],label=round(zout[i],1), color=dist_col[i])
        #plt.yscale('log')
        plt.xlabel(r"$k_F$ (h/mpc)")
        plt.ylabel(r"Predicted/Exact-1")
        plt.ylim(-0.08,0.08)
        #plt.xticks([1e-3, 1e-2, 0.05],[r"$10^{-3}$",r"$10^{-2}$","0.05"])
        name = params.build_dirname(parameters[ii,:], include_dense=True)
#         plt.title(name)
        #plt.xlim(1e-3, 0.052)
        plt.legend(loc='upper left', ncol=4)
        plt.tight_layout()
        name_ending = ".pdf"
        name = re.sub(r"\.","_",str(name))+plotname+name_ending
        #So we can use it in a latex document
        plt.savefig(os.path.join(savedir, name))
        plt.clf()
        print(name)
    #Plot the distribution of errors, compared to a Gaussian
    if np.all(np.isfinite(errlist)):
        _plot_error_histogram(savedir, plotname, errlist, xlim=6., nbins=100)

def plot_test_matter_interpolate(emulatordir,testdir, savedir=None, redshift=3.):
    """Make a plot showing the interpolation error for the matter power spectrum."""
    if savedir is None:
        savedir = testdir
    savename = testdir+"/matter_power.pdf"

    params = matter_power.MatterPowerEmulator(emulatordir)
    params.load()
    gp = params.get_emulator()
    params_test = matter_power.MatterPowerEmulator(testdir)
    params_test.load()
    for pp in params_test.get_parameters():
        dd = params_test.get_outdir(pp)
        if not os.path.exists(dd):
            dd = params_test.get_outdir(pp, strsz=2)
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
    plt.savefig(savename)
    print(savename)
    plt.clf()
    return gp
