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

def plot_test_interpolate_kf_bin_loop(emulatordir, testdir, savedir=None, plotname="", kf_bin_nums=np.arange(1)):
    """Plot the validation set vs the test points from the emulator for a specific k bin,
       looping over all the validation points."""
    if savedir is None:
        savedir = emulatordir

    all_power_array_all_kf = [None] * kf_bin_nums.size
    for i in range(kf_bin_nums.size):
        plotname_single_kf_bin = plotname + '_' + str(kf_bin_nums[i])
        _, all_power_array_all_kf[i], z_labs = plot_test_interpolate(emulatordir, testdir, savedir=savedir, plotname=plotname_single_kf_bin, kf_bin_nums=[kf_bin_nums[i],])

    all_power_array_all_kf = np.array(all_power_array_all_kf)
    for j in range(all_power_array_all_kf.shape[1]): #Loop over validation points in parameter space
        print("Validation point", j+1, "/", all_power_array_all_kf.shape[1])
        #Plot error histogram
        power_difference = all_power_array_all_kf[:, j, 1, :] - all_power_array_all_kf[:, j, 3, :]
        err_norm = power_difference / all_power_array_all_kf[:, j, 2, :]
        _plot_error_histogram(savedir, "_validation_parameters_" + str(j) + plotname, err_norm.flatten())

        #Plot predicted/exact
        power_ratio = all_power_array_all_kf[:, j, 1, :] / all_power_array_all_kf[:, j, 3, :]
        power_lower = (all_power_array_all_kf[:, j, 1, :] - all_power_array_all_kf[:, j, 2, :]) / all_power_array_all_kf[:, j, 3, :]
        power_upper = (all_power_array_all_kf[:, j, 1, :] + all_power_array_all_kf[:, j, 2, :]) / all_power_array_all_kf[:, j, 3, :]
        for k in range(all_power_array_all_kf.shape[3]): #Loop over redshift bins
            kf = all_power_array_all_kf[:, j, 0, k]
            plt.semilogx(kf, power_ratio[:, k], label=z_labs[k])
            plt.fill_between(kf, power_lower[:, k], power_upper[:, k], alpha=0.3, color="grey")
        plt.xlabel(r"$k_F$ (s/km)")
        plt.ylabel(r"Predicted/Exact")
        plt.xlim(xmax=0.05)
        plt.legend(loc=0)
        plt.tight_layout()
        plt.show()
        name = "validation_parameters_" + str(j) + plotname + ".pdf"
        plt.savefig(os.path.join(savedir, name))
        print(name)
        plt.clf()

    _plot_by_redshift_bins(savedir, plotname, z_labs, all_power_array_all_kf)

    #Plot combined error histogram
    power_difference = all_power_array_all_kf[:, :, 1, :] - all_power_array_all_kf[:, :, 3, :]
    err_norm = power_difference / all_power_array_all_kf[:, :, 2, :]
    _plot_error_histogram(savedir, plotname, err_norm.flatten())

    #Save combined output
    array_savename = os.path.join(savedir, "combined_output" + plotname + '.npy')
    np.save(array_savename, all_power_array_all_kf)

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

def plot_test_interpolate(emulatordir,testdir, savedir=None, plotname="", mean_flux=1, max_z=4.2, emuclass=None, kf_bin_nums=None,data_err=False):
    """Make a plot showing the interpolation error."""
    if savedir is None:
        savedir = emulatordir
    myspec = flux_power.MySpectra(max_z=max_z)
    t0 = None
    if mean_flux:
        t0 = 0.95
    mf = mflux.ConstMeanFlux(value=t0)
    if mean_flux == 2:
        mf = mflux.MeanFluxFactor()
    params_test = coarse_grid.Emulator(testdir,mf=mf, kf_bin_nums=kf_bin_nums)
    params_test.load()
    if emuclass is None:
        params = coarse_grid.Emulator(emulatordir, mf=mf, kf_bin_nums=kf_bin_nums)
    else:
        params = emuclass(emulatordir, mf=mf, kf_bin_nums=kf_bin_nums)
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

    #Get BOSS flux power spectra measurement errors
    if data_err is True:
        BOSS_data_instance = lyman_data.BOSSData()
        measurement_errors = np.sqrt(BOSS_data_instance.get_covar_diag()) #sqrt[(stat. 1 sigma)**2 + (sys. 1 sigma)**2]

    for pp in params_test.get_parameters():
        dd = params_test.get_outdir(pp)
        if mean_flux == 2:
            pp = np.concatenate([[t0,], pp])
        predicted,std = gp.predict(pp.reshape(1,-1))
        ps = myspec.get_snapshot_list(dd)
        tfac = t0*mflux.obs_mean_tau(myspec.zout)
        exact = ps.get_power(kf = kf, tau0_factors = tfac)
        ratio = predicted[0]/exact
        upper = (predicted[0] + std[0])/exact
        lower = (predicted[0] - std[0])/exact
        if data_err is False:
            errlist = np.concatenate([errlist, (predicted[0] - exact)/std[0]])
        else:
            measurement_errors_to_max_z = measurement_errors[:nred * nkf].reshape((nred, nkf))[::-1].flatten()
            if kf_bin_nums is not None:
                measurement_errors_to_max_z = measurement_errors_to_max_z.reshape((nred, nkf))[:,kf_bin_nums].flatten()
            errlist = np.concatenate([errlist, (predicted[0] - exact) / measurement_errors_to_max_z])
        print(measurement_errors_to_max_z)
        #REMOVE
        plt.hist((predicted[0]-exact)/std[0],bins=100 , density=True) #No 'density' property in Matplotlib v1
        xx = np.arange(-6, 6, 0.01)
        plt.plot(xx, np.exp(-xx**2/2)/np.sqrt(2*np.pi), ls="-", color="black")
        plt.plot(xx, np.exp(-xx**2/2/2**2)/np.sqrt(2*np.pi*2**2), ls="--", color="grey")
        plt.xlim(-6,6)
        plt.savefig(os.path.join(savedir, "errhist_"+str(np.size(errlist))+plotname+".pdf"))
        plt.clf()
        #DONE
        nk = len(kf)
        assert np.shape(ratio) == (nred*nk,)
        for i in range(nred):
            plt.semilogx(kf,ratio[i*nk:(i+1)*nk],label=myspec.zout[i])
            if data_err is False:
                lower_plot = lower
                upper_plot = upper
            elif data_err is True:
                lower_plot = (predicted[0] - measurement_errors_to_max_z) / exact
                upper_plot = (predicted[0] + measurement_errors_to_max_z) / exact
                plt.ylim([0.9, 1.1])
            plt.fill_between(kf,lower_plot[i*nk:(i+1)*nk], upper_plot[i*nk:(i+1)*nk],alpha=0.3, color="grey")
        plt.xlabel(r"$k_F$ (s/km)")
        plt.ylabel(r"Predicted/Exact")
        name = params_test.build_dirname(pp, include_dense=True)
#         plt.title(name)
        plt.xlim(xmax=0.05)
        plt.legend(loc=0)
        plt.tight_layout()
        plt.show()
        if mean_flux:
            name = name+"mf0.95"
        if data_err is False:
            name_ending = ".pdf"
        elif data_err is True:
            name_ending = '_data_err.pdf'
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
    if data_err is True:
        plotname = plotname + "_data_err"
    if np.all(np.isfinite(errlist)):
        #plt.hist(errlist,bins=100, density=True)
        #xx = np.arange(-6, 6, 0.01)
        #plt.plot(xx, np.exp(-xx**2/2)/np.sqrt(2*np.pi), ls="-", color="black")
        #plt.plot(xx, np.exp(-xx**2/2/2**2)/np.sqrt(2*np.pi*2**2), ls="--", color="grey")
        #plt.xlim(-6,6)
        #plt.savefig(os.path.join(savedir, "errhist"+plotname+".pdf"))
        #plt.clf()
        _plot_error_histogram(savedir, plotname, errlist, xlim=6., nbins=250, xlabel=r"(Predicted - Exact) / $1 \sigma$ [BOSS error]")

    return gp, all_power_array, myspec.zout

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
