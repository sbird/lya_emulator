"""Generate a test plot for an emulator"""
from __future__ import print_function
import os.path
import re
import math
from datetime import datetime
import scipy.spatial
import numpy as np
import coarse_grid
import flux_power
import matter_power
import lyman_data
import mean_flux as mflux
import matplotlib
matplotlib.use('PDF')
import matplotlib.pyplot as plt
from datetime import datetime

def plot_convexhull(emulatordir):
    """Plot the convex hull of the projection of the emulator parameters"""
    params = coarse_grid.Emulator(emulatordir, mf=None)
    params.load()
    points = params.sample_params
    hull = scipy.spatial.ConvexHull(points)
    K = np.shape(points)[1]
    _, axes = plt.subplots(K, K)
    for i in range(K):
        for j in range(K):
            ax = axes[i,j]
            if j >= i:
                ax.set_frame_on(False)
                ax.set_xticks([])
                ax.set_yticks([])
                continue
            ax.plot(points[:,i], points[:,j], 'o')
            projected = np.vstack([points[:,i], points[:,j]]).T
            hull = scipy.spatial.ConvexHull(projected)
            for simplex in hull.simplices:
                ax.plot(projected[simplex, 0], projected[simplex, 1], 'k-')
    return hull

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
        np.savetxt(os.path.join(savedir, "table_errhist" + plotname + ".txt"), err_norm)
        _plot_unit_Gaussians(xx)
        plt.xlabel(xlabel)
        plt.xlim(-1. * xlim, xlim)
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(savedir, "errhist" + plotname + ".pdf"))
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
        plt.plot(xx, np.exp(-xx ** 2 / 2) / np.sqrt(2 * np.pi), ls="-", color="black", label=r"Unit Gaussian")
        plt.plot(xx, np.exp(-xx ** 2 / 2 / 2 ** 2) / np.sqrt(2 * np.pi * 2 ** 2), ls="--", color="grey")
    else:
        axis.plot(xx, np.exp(-xx ** 2 / 2) / np.sqrt(2 * np.pi), ls="-", color="black", label=r"Unit Gaussian")
        axis.plot(xx, np.exp(-xx ** 2 / 2 / 2 ** 2) / np.sqrt(2 * np.pi * 2 ** 2), ls="--", color="grey")

def plot_test_interpolate(emulatordir,testdir, savedir=None, plotname="", mean_flux=1, max_z=4.2, emuclass=None):
    """Make a plot showing the interpolation error."""
    if savedir is None:
        savedir = emulatordir
    myspec = flux_power.MySpectra(max_z=max_z)
    t0 = None
    if mean_flux:
        t0 = 1. #0.95
    mf = mflux.ConstMeanFlux(value=t0) #In 'ConstMeanFlux' case: multiply tau_0_i[z] by t0 = 0.95
    if mean_flux == 2:
        mf = mflux.MeanFluxFactor() #In 'MeanFluxFactor' case: DON'T multiply tau_0_i[z] by t0 - because *emulate* t0[z]
    params_test = coarse_grid.Emulator(testdir,mf=mf)
    params_test.load()
    if emuclass is None:
        params = coarse_grid.Emulator(emulatordir, mf=mf)
    else:
        params = emuclass(emulatordir, mf=mf)
    params.load()
    print('Beginning to generate emulator at', str(datetime.now()))
    gp = params.get_emulator(max_z=max_z)
    print('Finished generating emulator at', str(datetime.now()))
    kf = params.kf
    del params
    errlist = np.array([])
    #Constant mean flux.

    # Save output
    nred = len(myspec.zout)
    nkf = kf.size
    #print("Number of validation points =", params_test.get_parameters().shape[0])
    all_power_array = np.zeros((params_test.get_parameters().shape[0], 4, nkf*nred)) #kf, predicted, std, exact
    validation_number = 0

    for pp in params_test.get_parameters():
        dd = params_test.get_outdir(pp)
        if not os.path.exists(dd):
            dd = params_test.get_outdir(pp, strsz=2)
        if mean_flux == 2:
            pp = np.concatenate([[t0,], pp]) #In 'MeanFluxFactor' case: choose t0 point for fair comparison
        predicted_nat,std_nat = gp.predict(pp.reshape(1,-1)) #.predict takes [{list of parameters: t0; cosmo.; thermal},]
        #This is binned as for the simulation, in comoving Mpc. Needs rebinning
        omega_m = params_test.omegamh2/pp[len(params_test.mf.dense_param_names)+params_test.param_names['hub']]**2
        okf, predicted = flux_power.rebin_power_to_kms(kfkms=kf, kfmpc=gp.kf, flux_powers = predicted_nat[0], zbins=myspec.zout, omega_m = omega_m)
        _, std = flux_power.rebin_power_to_kms(kfkms=kf, kfmpc=gp.kf, flux_powers = std_nat[0], zbins=myspec.zout, omega_m = omega_m)

        ps = myspec.get_snapshot_list(dd)
        meanfluxes = None
        if t0 is not None:
            meanfluxes = np.exp(-t0*mflux.obs_mean_tau(myspec.zout))
        exact_nat = ps.get_power_native_binning(mean_fluxes = meanfluxes)
        okf_ex, exact = flux_power.rebin_power_to_kms(kfkms=kf, kfmpc=gp.kf, flux_powers = exact_nat, zbins=myspec.zout, omega_m = omega_m)
        assert np.all([np.all(np.abs(okf_ex[ii]/okf[ii]-1) < 1e-5) for ii in range(nred)])
        ratio =  [predicted[ii]/exact[ii] for ii in range(nred)]
        upper =  [(predicted[ii] + std[ii])/exact[ii] for ii in range(nred)]
        lower =  [(predicted[ii]-std[ii])/exact[ii] for ii in range(nred)]
        errrr =  [(predicted[ii]-exact[ii])/std[ii] for ii in range(nred)]
        errlist = np.concatenate([errlist, errrr])
        #REMOVE
        plt.hist(errrr,bins=100 , density=True) #No 'density' property in Matplotlib v1
        xx = np.arange(-6, 6, 0.01)
        plt.plot(xx, np.exp(-xx**2/2)/np.sqrt(2*np.pi), ls="-", color="black")
        plt.plot(xx, np.exp(-xx**2/2/2**2)/np.sqrt(2*np.pi*2**2), ls="--", color="grey")
        plt.xlim(-6,6)
        plt.savefig(os.path.join(savedir, "errhist_"+str(np.size(errlist))+plotname+".pdf"))
        plt.clf()
        #DONE
        for i in range(nred):
            assert np.size(ratio[i]) == np.size(okf[i])
            plt.semilogx(okf[i],ratio[i],label=round(myspec.zout[i],1))
            plt.fill_between(okf[i],lower[i], upper[i],alpha=0.3, color="grey")
        #plt.yscale('log')
        plt.xlabel(r"$k_F$ (s/km)")
        plt.ylabel(r"Predicted/Exact")
        name = params_test.build_dirname(pp, include_dense=True)
#         plt.title(name)
        plt.xlim(xmax=0.05)
        plt.legend(loc='right')
        plt.tight_layout()
        plt.show()
        name_ending = ".pdf"
        name = re.sub(r"\.","_",str(name))+plotname+name_ending
        #So we can use it in a latex document
        plt.savefig(os.path.join(savedir, name))
        print(name)
        plt.clf()

        #Save output
        all_power_array[validation_number] = np.vstack((np.tile(kf, nred), predicted[0], std[0], exact))
        array_savename = os.path.join(savedir, name[:-4] + '.npy')
        np.save(array_savename, all_power_array[validation_number])
        validation_number+=1

    #Plot the distribution of errors, compared to a Gaussian
    if np.all(np.isfinite(errlist)):
        #plt.hist(errlist,bins=100, density=True)
        #xx = np.arange(-6, 6, 0.01)
        #plt.plot(xx, np.exp(-xx**2/2)/np.sqrt(2*np.pi), ls="-", color="black")
        #plt.plot(xx, np.exp(-xx**2/2/2**2)/np.sqrt(2*np.pi*2**2), ls="--", color="grey")
        #plt.xlim(-6,6)
        #plt.savefig(os.path.join(savedir, "errhist"+plotname+".pdf"))
        #plt.clf()
        _plot_error_histogram(savedir, plotname, errlist, xlim=6., nbins=250) #, xlabel=r"(Predicted - Exact) / $1 \sigma$ [BOSS error]")

    return gp, all_power_array, myspec.zout

def plot_test_matter_interpolate(emulatordir,testdir, savedir=None, redshift=3.):
    """Make a plot showing the interpolation error for the matter power spectrum."""
    if savedir is None:
        savedir = testdir
    savename = testdir+"/matter_power.pdf"

    params = coarse_grid.MatterPowerEmulator(emulatordir)
    params.load()
    gp = params.get_emulator()
    params_test = coarse_grid.MatterPowerEmulator(testdir)
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
