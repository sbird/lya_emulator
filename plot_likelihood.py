"""Module for plotting generated likelihood chains"""
import json
import os
import glob
import math as mh
from datetime import datetime
import numpy as np
import distinct_colours_py3 as dc
import likelihood as likeh
from mean_flux import mean_flux_slope_to_factor
import matplotlib
matplotlib.use('PDF')
import matplotlib.pyplot as plt
import corner

def make_plot_emulator_error(emulator_training_directory, savefile, mean_flux_label='c', likelihood_instance=None, max_z=4.2):
    """Make a plot of emulator error as a fraction of the data error"""
    if likelihood_instance is None:
        likelihood_instance = likeh.LikelihoodClass(emulator_training_directory, mean_flux=mean_flux_label, max_z=max_z)
    _, z, n_k_los, n_z = get_k_z(likelihood_instance)

    parameter_value_samples = np.linspace(0.8, 1.2, num=200) #HeliumHeatAmp
    mean_flux_sample = np.array([0., 0.95]) #dtau0, tau0
    emulator_error_plot = np.zeros((parameter_value_samples.shape[0], n_z))

    emulated_flux_power = [None] * parameter_value_samples.shape[0]
    emulator_error = [None] * parameter_value_samples.shape[0]
    fractional_emulator_error = [None] * parameter_value_samples.shape[0]

    for i in range(parameter_value_samples.shape[0]):
        param_val = np.array([parameter_value_samples[i],])
        tau0_factors = None
        if mean_flux_label == 's':
            param_val = np.concatenate(([mean_flux_sample[1],], param_val))
            tau0_factors = mean_flux_slope_to_factor(z, mean_flux_sample[0])
        emulated_flux_power[i], emulator_error[i] = likelihood_instance.gpemu.predict(param_val.reshape(1, -1), tau0_factors=tau0_factors)
        fractional_emulator_error[i] = (emulator_error[i][0] / emulated_flux_power[i][0]).reshape(n_z, n_k_los)
        emulator_error_plot[i] = np.nanmean(fractional_emulator_error[i], axis=-1)
        print('Standard deviation of fractional emulator error with scale =', np.std(fractional_emulator_error[i], axis=-1))

    _, axis = plt.subplots(nrows=1, ncols=1, figsize=(6.4 * 2., 10.))
    distinct_colours = dc.get_distinct(n_z)
    line_width = 0.5
    fontsize = 7.
    for i in range(n_z):
        data_error = np.sqrt(likelihood_instance.sdss.get_covar(z[i]).diagonal()) #n_k_los
        data_error_rescaling_factor = likelihood_instance.data_fluxpower[n_k_los*i:n_k_los*(i+1)] / likelihood_instance.BOSS_flux_power[i]
        data_error *= data_error_rescaling_factor
        fractional_data_error = data_error[np.newaxis, :] / np.array(emulated_flux_power)[:, 0].reshape(-1, n_z, n_k_los)[:, i]
        data_error_plot = np.nanmean(fractional_data_error, axis=-1)
        #n_samples, n_k_los

        axis.plot(parameter_value_samples, emulator_error_plot[:,i], color=distinct_colours[i], lw=line_width, label=r'$z = %.1f$' % z[i])
        axis.plot(parameter_value_samples, data_error_plot, color=distinct_colours[i], lw=line_width, ls='--')
        #axis.scatter(parameter_value_samples, emulator_error_plot[:,i], c=distinct_colours[i], label=r'$z = %.1f$' % z[i])

    axis.plot([], color='gray', ls='-', label=r'emulated sigma')
    axis.plot([], color='gray', ls='--', label=r'BOSS sigma')

    axis.axvline(x=0.9, color='black', ls=':', lw=line_width)
    #axis.axvline(x=0.95, color='black', ls=':', lw=line_width)
    axis.axvline(x=1., color='black', ls=':', lw=line_width)
    #axis.axvline(x=1.05, color='black', ls=':', lw=line_width)
    axis.axvline(x=1.1, color='black', ls=':', lw=line_width)
    axis.legend(frameon=False, fontsize=fontsize)
    axis.set_yscale('log')
    axis.set_xlabel(r'HeliumHeatAmp')
    axis.set_ylabel(r'(sigma / emulated P(k)) [averaged over scale]')

    #np.savez('/home/keir/Data/emulator/emulator_error.npz', parameter_value_samples, emulator_error_plot, np.array(fractional_emulator_error), np.array(emulator_error), np.array(emulated_flux_power), k_los)
    plt.savefig(savefile)

def get_k_z(likelihood_instance):
    """Get k and z bins"""
    k_los = likelihood_instance.gpemu.kf
    n_k_los = k_los.size
    z = likelihood_instance.zout #Highest redshift first
    n_z = z.size
    return k_los, z, n_k_los, n_z

def make_plot_flux_power_spectra(like, params, datadir, savefile):
    """Make a plot of the power spectra, with redshift, the BOSS power and the sigmas. Four plots stacked."""
    k_los = like.gpemu.kf
    n_k_los = k_los.size
    z = like.zout #Highest redshift first
    n_z = z.size

    data_fluxpower = likeh.load_data(datadir, kf=k_los)
    exact_flux_power = data_fluxpower.reshape(n_z, n_k_los)
    emulated_flux_power, emulated_flux_power_std = like.get_predicted(params)
    emulated_flux_power = emulated_flux_power[0].reshape(n_z, n_k_los)
    emulated_flux_power_std = emulated_flux_power_std[0].reshape(n_z, n_k_los)
    data_flux_power = like.sdss.pf.reshape(-1, n_k_los)[:n_z][::-1]

    figure, axes = plt.subplots(nrows=4, ncols=1, figsize=(6.4*2., 10.))
    distinct_colours = dc.get_distinct(n_z)
    scaling_factor = k_los / mh.pi
    for i in range(n_z):
        data_flux_power_std_single_z = np.sqrt(like.sdss.get_covar(z[i]).diagonal())
        exact_flux_power_std_single_z = np.sqrt(np.diag(like.get_rescaled_BOSS_error(i, data_power = data_fluxpower)))
#         print('Diagonal elements of BOSS covariance matrix at single redshift:', data_flux_power_std_single_z)

        line_width = 0.5
        axes[0].plot(k_los, exact_flux_power[i]*scaling_factor, color=distinct_colours[i], ls='-', lw=line_width, label=r'$z = %.1f$'%z[i])
        axes[0].plot(k_los, emulated_flux_power[i]*scaling_factor, color=distinct_colours[i], ls='--', lw=line_width)
        axes[0].errorbar(k_los, emulated_flux_power[i]*scaling_factor, yerr=emulated_flux_power_std[i]*scaling_factor, ecolor=distinct_colours[i], ls='')

        axes[1].plot(k_los, data_flux_power[i]*scaling_factor, color=distinct_colours[i], lw=line_width)
        axes[1].errorbar(k_los, data_flux_power[i]*scaling_factor, yerr=data_flux_power_std_single_z*scaling_factor, ecolor=distinct_colours[i], ls='')

        axes[2].plot(k_los, exact_flux_power_std_single_z / exact_flux_power[i], color=distinct_colours[i], ls='-', lw=line_width)
        axes[2].plot(k_los, emulated_flux_power_std[i] / exact_flux_power[i], color=distinct_colours[i], ls='--',
                     lw=line_width)

        #axes[3].plot(k_los, data_flux_power_std_single_z / data_flux_power[i], color=distinct_colours[i], ls='-', lw=line_width)
        axes[3].plot(k_los, emulated_flux_power[i] / exact_flux_power[i], color=distinct_colours[i], ls='-', lw=line_width)

    fontsize = 7.
    xlim = [1.e-3, 0.022]
    xlabel = r'$k$ ($\mathrm{s}\,\mathrm{km}^{-1}$)'
    ylabel = r'$k P(k) / \pi$'

    axes[0].plot([], color='gray', ls='-', label=r'exact')
    axes[0].plot([], color='gray', ls='--', label=r'emulated')
    axes[0].legend(frameon=False, fontsize=fontsize)
    axes[0].set_xlim(xlim)  # 4.e-2])
    #axes[0].set_xscale('log')
    axes[0].set_yscale('log')
    axes[0].set_xlabel(xlabel)
    axes[0].set_ylabel(ylabel)

    axes[1].plot([], color='gray', label=r'BOSS data')
    axes[1].legend(frameon=False, fontsize=fontsize)
    axes[1].set_xlim(xlim)
    axes[1].set_yscale('log')
    axes[1].set_xlabel(xlabel)
    axes[1].set_ylabel(ylabel)

    axes[2].plot([], color='gray', ls='-', label=r'measurement sigma')
    axes[2].plot([], color='gray', ls='--', label=r'emulated sigma')
    axes[2].legend(frameon=False, fontsize=fontsize)
    axes[2].set_xlim(xlim)
    axes[2].set_yscale('log')
    axes[2].set_xlabel(xlabel)
    axes[2].set_ylabel(r'sigma / exact P(k)')

    axes[3].axhline(y=1., color='black', ls=':', lw=line_width)
    axes[3].set_xlim(xlim)
    #axes[3].set_yscale('log')
    axes[3].set_xlabel(xlabel)
    axes[3].set_ylabel(r'emulated P(k) / exact P(k)') #BOSS sigma / BOSS P(k)')

    figure.subplots_adjust(hspace=0)
    plt.savefig(savefile)
    plt.show()

    print(datadir)
    print('Maximum fractional overestimation of flux power spectrum =', np.max((emulated_flux_power / exact_flux_power) - 1.))
    print('Maximum fractional underestimation of flux power spectrum =', np.min((emulated_flux_power / exact_flux_power) - 1.))

    #make_plot_emulator_error(emudir, '/home/keir/Plots/Emulator/emulator_error_hot_cold.pdf', likelihood_instance=like)

    return like

def make_plot(chainfile, savefile, true_parameter_values=None):
    """Make a plot of parameter posterior values"""
    with open(chainfile+"_names.txt") as ff:
        names = ff.read().split('\n')
    pnames = [i.split(' ')[0] for i in names if len(i) > 0]
    samples = np.loadtxt(chainfile)
    plt.rc('font', family='serif', size=15.)
    corner.corner(samples, labels=pnames, truths=true_parameter_values)
    plt.savefig(savefile)

def get_simulation_parameters_knots(base):
    """Get the parameters of a knot-based simulation from the SimulationICs JSON file."""
    jsin = open(os.path.join(base, "SimulationICs.json"), 'r')
    pp = json.load(jsin)
    knv = pp["knot_val"]
    #This will fail!
    slope, amp = _therm_params(pp)
    parvec = [0., 1., *knv, slope, amp, pp["hubble"]]
    return parvec

def _therm_params(pp):
    """Helper to get thermal parameters from a json dictionary."""
    try:
        #Old-style emulator
        assert pp["code_args"]["rescale_gamma"] is True
        slope = pp["code_args"]["rescale_slope"]
        amp = pp["code_args"]["rescale_amp"]
    except KeyError:
        assert pp["rescale_gamma"] is True
        slope = pp["rescale_slope"]
        amp = pp["rescale_amp"]
    return slope, amp

def get_simulation_parameters_s8(base, pivot=0.05):
    """Get the parameters of a sigma8-ns-based simulation from the SimulationICs JSON file."""
    jsin = open(os.path.join(base, "SimulationICs.json"), 'r')
    pp = json.load(jsin)
    slope, amp = _therm_params(pp)
    #Change the pivot value
    As = pp['scalar_amp'] / (pivot/(2*np.pi/8.))**(pp['ns']-1.)
    parvec = [0., 1., pp['ns'], As, slope, amp, pp["hubble"]]
    return parvec

def run_likelihood_test(testdir, emudir, savedir=None, plot=True, mean_flux_label='s'):
    """Generate some likelihood samples"""

    if savedir is None:
        savedir=emudir
    #Find all subdirectories
    subdirs = glob.glob(testdir + "/*/")
    assert len(subdirs) > 1

    like = likeh.LikelihoodClass(basedir=emudir, mean_flux=mean_flux_label)
    for sdir in subdirs:
        sname = os.path.basename(os.path.abspath(sdir))
        chainfile = os.path.join(savedir, 'chain_' + sname + '.txt')
        datadir = os.path.join(sdir, "output")
        true_parameter_values = get_simulation_parameters_s8(sdir)
        if plot is True:
            fp_savefile = os.path.join(savedir, 'flux_power_'+sname + ".pdf")
            make_plot_flux_power_spectra(like, true_parameter_values, datadir, savefile=fp_savefile)
        print('Beginning to sample likelihood at', str(datetime.now()))
        like.do_sampling(chainfile, datadir=datadir)
        print('Done sampling likelihood at', str(datetime.now()))
        if plot is True:
            savefile = os.path.join(savedir, 'corner_'+sname + ".pdf")
            make_plot(chainfile, savefile, true_parameter_values=true_parameter_values)
    return like

if __name__ == "__main__":
    sim_rootdir = "simulations"
    plotdir = 'plots'
    savedir=os.path.join(plotdir,"hires_s8")
    emud = os.path.join(sim_rootdir,'hires_s8')
    testdirs = os.path.join(sim_rootdir,'hires_s8_test')

    like = run_likelihood_test(testdirs, emud, savedir=savedir, plot=True)
