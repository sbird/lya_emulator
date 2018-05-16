"""Module for plotting generated likelihood chains"""
import math as mh
import numpy as np
import matplotlib
matplotlib.use('PDF')
import matplotlib.pyplot as plt
#import corner
from datetime import datetime
import distinct_colours_py3 as dc

from likelihood import *

def make_plot_emulator_error(emulator_training_directory, savefile, mean_flux_label='c', likelihood_instance=None):
    if likelihood_instance is None:
        likelihood_instance = generate_likelihood_class(emulator_training_directory, emulator_training_directory, mean_flux_label=mean_flux_label)
    k_los, z, n_k_los, n_z = get_k_z(likelihood_instance)

    parameter_value_samples = np.linspace(0.8, 1.2, num=200) #HeliumHeatAmp
    emulator_error_plot = np.zeros((parameter_value_samples.shape[0], n_z))

    emulated_flux_power = [None] * parameter_value_samples.shape[0]
    emulator_error = [None] * parameter_value_samples.shape[0]
    fractional_emulator_error = [None] * parameter_value_samples.shape[0]

    for i in range(parameter_value_samples.shape[0]):
        param_val = parameter_value_samples[i]
        emulated_flux_power[i], emulator_error[i] = likelihood_instance.gpemu.predict(np.array([param_val,]).reshape(1, -1), tau0_factors=None)
        fractional_emulator_error[i] = (emulator_error[i][0] / emulated_flux_power[i][0]).reshape(n_z, n_k_los)
        emulator_error_plot[i] = np.nanmean(fractional_emulator_error[i], axis=-1)
        print('Standard deviation of fractional emulator error with scale =', np.std(fractional_emulator_error[i], axis=-1))

    figure, axis = plt.subplots(nrows=1, ncols=1, figsize=(6.4 * 2., 10.))
    distinct_colours = dc.get_distinct(n_z)
    line_width = 0.5
    fontsize = 7.
    for i in range(n_z):
        axis.plot(parameter_value_samples, emulator_error_plot[:,i], color=distinct_colours[i], lw=line_width, label=r'$z = %.1f$' % z[i])
        #axis.scatter(parameter_value_samples, emulator_error_plot[:,i], c=distinct_colours[i], label=r'$z = %.1f$' % z[i])
    axis.axvline(x=0.9, color='black', ls=':', lw=line_width)
    axis.axvline(x=1., color='black', ls=':', lw=line_width)
    axis.axvline(x=1.1, color='black', ls=':', lw=line_width)
    axis.legend(frameon=False, fontsize=fontsize)
    axis.set_yscale('log')
    axis.set_xlabel(r'HeliumHeatAmp')
    axis.set_ylabel(r'(emulated sigma / emulated P(k)) [averaged over scale]')

    #np.savez('/home/keir/Data/emulator/emulator_error.npz', parameter_value_samples, emulator_error_plot, np.array(fractional_emulator_error), np.array(emulator_error), np.array(emulated_flux_power), k_los)
    plt.savefig(savefile)

def make_plot_compare_two_simulations(simdir1, simdir2, simname1, simname2, savefile, mean_flux_label1='c', mean_flux_label2='c'):
    likelihood_instance1 = generate_likelihood_class(simdir1, simdir1, simulation_sub_directory=simname1, mean_flux_label=mean_flux_label1)
    likelihood_instance2 = generate_likelihood_class(simdir2, simdir2, simulation_sub_directory=simname2,
                                                     mean_flux_label=mean_flux_label2)
    k_los, z, n_k_los, n_z = get_k_z(likelihood_instance1)
    flux_power1 = likelihood_instance1.data_fluxpower.reshape(n_z, n_k_los)
    flux_power2 = likelihood_instance2.data_fluxpower.reshape(n_z, n_k_los)

    figure, axes = plt.subplots(nrows=2, ncols=1, figsize=(6.4 * 2., 10.))
    distinct_colours = dc.get_distinct(n_z)
    line_width = 0.5
    scaling_factor = k_los / mh.pi
    for i in range(n_z):
        axes[0].plot(k_los, flux_power1[i] * scaling_factor, color=distinct_colours[i], ls='-', lw=line_width,
                     label=r'$z = %.1f$' % z[i])
        axes[0].plot(k_los, flux_power2[i] * scaling_factor, color=distinct_colours[i], ls='--', lw=line_width)

        axes[1].plot(k_los, ((flux_power2 - flux_power1) / flux_power1)[i], color=distinct_colours[i], lw=line_width)

    fontsize = 7.
    xlim = [1.e-3, 0.022]
    xlabel = r'$k$ ($\mathrm{s}\,\mathrm{km}^{-1}$)'
    ylabel = r'$k P(k) / \pi$'

    axes[0].plot([], color='gray', ls='-', label=simname1)
    axes[0].plot([], color='gray', ls='--', label=simname2)
    axes[0].legend(frameon=False, fontsize=fontsize)
    axes[0].set_xlim(xlim)
    axes[0].set_yscale('log')
    axes[0].set_ylabel(ylabel)

    axes[1].set_xlim(xlim)
    axes[1].set_xlabel(xlabel)
    axes[1].set_ylabel(simname2 + '-' + simname1 + '/' + simname1)

    figure.subplots_adjust(hspace=0)
    plt.savefig(savefile)

def get_k_z(likelihood_instance):
    k_los = likelihood_instance.gpemu.kf
    n_k_los = k_los.size
    z = likelihood_instance.zout #Highest redshift first
    n_z = z.size
    return k_los, z, n_k_los, n_z

def make_plot_flux_power_spectra(testdir, emudir, savefile, mean_flux_label='c'):
    """Make a plot of the power spectra, with redshift, the BOSS power and the sigmas. Four plots stacked."""
    like, like_true = run_and_plot_likelihood_samples(testdir, emudir, None, '', mean_flux_label=mean_flux_label, return_class_only=True)
    k_los, z, n_k_los, n_z = get_k_z(like)
    exact_flux_power = like.data_fluxpower.reshape(n_z, n_k_los)
    emulated_flux_power = like.emulated_flux_power[0].reshape(n_z, n_k_los)
    emulated_flux_power_std = like.emulated_flux_power_std[0].reshape(n_z, n_k_los)
    data_flux_power = like.sdss.pf.reshape(-1, n_k_los)[:n_z][::-1]

    '''emulated_flux_power_direct = [None] * 200
    emulator_error_direct = [None] * 200
    j=0
    np.savez('/home/keir/Data/emulator/emulator_error_test.npz', np.array(like.gpemu.predict(np.array([np.linspace(0.8, 1.2, num=200)[50],]).reshape(1, -1), tau0_factors=None)), np.array(np.linspace(0.8, 1.2, num=200)[50]))
    for i in np.linspace(0.8, 1.2, num=200):
        emulated_flux_power_direct[j], emulator_error_direct[j] = like.gpemu.predict(np.array([i,]).reshape(1, -1), tau0_factors=None)
        j+=1
    np.savez('/home/keir/Data/emulator/emulator_error_direct.npz', np.array(emulated_flux_power_direct), np.array(emulator_error_direct), k_los)'''

    figure, axes = plt.subplots(nrows=4, ncols=1, figsize=(6.4*2., 10.))
    distinct_colours = dc.get_distinct(n_z)
    scaling_factor = k_los / mh.pi
    for i in range(n_z):
        data_flux_power_std_single_z = np.sqrt(like.sdss.get_covar(z[i]).diagonal())
        print('Diagonal elements of BOSS covariance matrix at single redshift:', data_flux_power_std_single_z)

        line_width = 0.5
        axes[0].plot(k_los, exact_flux_power[i]*scaling_factor, color=distinct_colours[i], ls='-', lw=line_width, label=r'$z = %.1f$'%z[i])
        axes[0].plot(k_los, emulated_flux_power[i]*scaling_factor, color=distinct_colours[i], ls='--', lw=line_width)
        axes[0].errorbar(k_los, emulated_flux_power[i]*scaling_factor, yerr=emulated_flux_power_std[i]*scaling_factor, ecolor=distinct_colours[i], ls='')

        axes[1].plot(k_los, data_flux_power[i]*scaling_factor, color=distinct_colours[i], lw=line_width)
        axes[1].errorbar(k_los, data_flux_power[i]*scaling_factor, yerr=data_flux_power_std_single_z*scaling_factor, ecolor=distinct_colours[i], ls='')

        axes[2].plot(k_los, data_flux_power_std_single_z / exact_flux_power[i], color=distinct_colours[i], ls='-', lw=line_width)
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

    axes[2].plot([], color='gray', ls='-', label=r'BOSS sigma')
    axes[2].plot([], color='gray', ls='--', label=r'emulated sigma')
    axes[2].legend(frameon=False, fontsize=fontsize)
    axes[2].set_xlim(xlim)
    axes[2].set_yscale('log')
    axes[2].set_xlabel(xlabel)
    axes[2].set_ylabel(r'sigma / exact P(k)')

    axes[3].axhline(y=1., color='black', ls=':', lw=line_width)
    axes[3].set_xlim(xlim)
    #axes[3].set_ylim([1.0011, 0.9989])
    #axes[3].set_yscale('log')
    axes[3].set_xlabel(xlabel)
    axes[3].set_ylabel(r'emulated P(k) / exact P(k)') #BOSS sigma / BOSS P(k)')

    figure.subplots_adjust(hspace=0)
    plt.savefig(savefile)
    plt.show()

    print('Maximum fractional overestimation of flux power spectrum =', np.max((emulated_flux_power / exact_flux_power) - 1.))
    print('Maximum fractional underestimation of flux power spectrum =', np.min((emulated_flux_power / exact_flux_power) - 1.))

    #make_plot_emulator_error(emudir, '/home/keir/Plots/Emulator/emulator_error_hot_cold.pdf', likelihood_instance=like)

    return like

def make_plot(chainfile, savefile, teerue_parameter_values=None):
    """Make a plot of parameter posterior values"""
    import corner
    with open(chainfile+"_names.txt") as ff:
        names = ff.read().split('\n')
    pnames = [i.split(' ')[0] for i in names if len(i) > 0]
    samples = np.loadtxt(chainfile)
    plt.rc('font', family='serif', size=15.)
    corner.corner(samples, labels=pnames, truths=true_parameter_values)
    plt.savefig(savefile)

def generate_likelihood_class(testdir, emudir, simulation_sub_directory=None, mean_flux_label='c'):
    if simulation_sub_directory is None:
        #simulation_sub_directory = "/AA0.97BB1.3CC0.67DD1.3heat_slope0.083heat_amp0.92hub0.69/output"
        #simulation_sub_directory = '/AA1.1BB1.1CC1.4DD1.4heat_slope0.43heat_amp1hub0.71/output'
        #simulation_sub_directory = '/ns0.97As2.2e-09heat_slope0.083heat_amp0.92hub0.69/output'
        #simulation_sub_directory = '/ns0.96As2.6e-09heat_slope-0.19heat_amp1hub0.74/output'
        simulation_sub_directory = '/HeliumHeatAmp0.9/output'
    print('Beginning to initialise LikelihoodClass at', str(datetime.now()))
    return LikelihoodClass(basedir=emudir, datadir=testdir+simulation_sub_directory, mean_flux=mean_flux_label)

def run_and_plot_likelihood_samples(testdir, emudir, savefile, plotname, plot=True, chain_savedir=None, n_walkers=100, n_burn_in_steps=100, n_steps=400, while_loop=True, mean_flux_label='s', return_class_only=False, include_emulator_error=True):
    """Generate some likelihood samples"""
    # TODO: Add true values #Read from filenames
    #true_parameter_values = [None, None, 0.97, 1.3, 0.67, 1.3, 0.083, 0.92, 0.69]
    #true_parameter_values = [0., 1., 0.97, 1.3, 0.67, 1.3, 0.083, 0.92, 0.69]
    #true_parameter_values = [0.97, 1.3, 0.67, 1.3, 0.083, 0.92, 0.69]
    #true_parameter_values = [0., 0.95, 1.1, 1.1, 1.4, 1.4, 0.43, 1., 0.71]
    #true_parameter_values = [1.1357142857142857, 1.0928571428571427, 1.35, 1.35, 0.4285714285714285, 1.0476190476190474, 0.7142857142857143]
    #true_parameter_values = [0., 1., 1.1357142857142857, 1.0928571428571427, 1.35, 1.35, 0.4285714285714285, 1.0476190476190474,
    #                         0.7142857142857143]
    #true_parameter_values = [0.975, 2.25e-09, 0.08333333333333326, 0.9166666666666666, 0.6916666666666667]
    #true_parameter_values = [None, None, 0.97, 2.2e-9, 0.083, 0.92, 0.69]
    #true_parameter_values = [0., 1., 0.975, 2.25e-09, 0.08333333333333326, 0.9166666666666666, 0.6916666666666667]
    #true_parameter_values = [0.9642857142857143, 2.614285714285714e-09, -0.19047619047619047, 1.0476190476190474, 0.7428571428571429]
    #true_parameter_values = [0., 1., 0.9642857142857143, 2.614285714285714e-09, -0.19047619047619047, 1.0476190476190474, 0.7428571428571429]
    #true_parameter_values = [0.95, 0., 1.,]
    true_parameter_values = [0.9,]

    if chain_savedir is None:
        chain_savedir = testdir
    chainfile = chain_savedir + '/AA0.97BB1.3_chain_' + plotname + '.txt'

    like = generate_likelihood_class(testdir, emudir, mean_flux_label=mean_flux_label)

    if return_class_only is False:
        print('Beginning to sample likelihood at', str(datetime.now()))
        output = like.do_sampling(chainfile, nwalkers=n_walkers, burnin=n_burn_in_steps, nsamples=n_steps, while_loop=while_loop, include_emulator_error=include_emulator_error)
        if plot is True:
            print('Beginning to make corner plot at', str(datetime.now()))
            make_plot(chainfile, savefile, true_parameter_values=true_parameter_values)
        return like
    else:
        likelihood_at_true_values = like.likelihood(true_parameter_values, include_emu=include_emulator_error)
        return like, likelihood_at_true_values
