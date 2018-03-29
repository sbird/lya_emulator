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

def make_plot_flux_power_spectra(testdir, emudir, savefile, mean_flux_label='s'):
    like, like_true = run_and_plot_likelihood_samples(testdir, emudir, None, '', mean_flux_label=mean_flux_label, return_class_only=True)
    k_los = like.gpemu.kf
    n_k_los = k_los.size
    z = like.sdss.get_redshifts() #Highest redshift first
    n_z = z.size
    exact_flux_power = like.data_fluxpower.reshape(n_z, n_k_los)
    emulated_flux_power = like.emulated_flux_power[0].reshape(n_z, n_k_los)
    emulated_flux_power_std = like.emulated_flux_power_std[0].reshape(n_z, n_k_los)

    figure, axes = plt.subplots(nrows=2, ncols=1)
    distinct_colours = dc.get_distinct(n_z)
    scaling_factor = k_los / mh.pi
    for i in range(n_z):
        axes[0].plot(k_los, exact_flux_power[i]*scaling_factor, color=distinct_colours[i], ls='-', label=r'$z = %d$'%z[i])
        axes[0].plot(k_los, emulated_flux_power[i]*scaling_factor, color=distinct_colours[i], ls='--')
        axes[0].errorbar(k_los, emulated_flux_power[i]*scaling_factor, yerr=emulated_flux_power_std[i]*scaling_factor, ecolor=distinct_colours[i], ls='')
    axes[0].legend(frameon=False)
    plt.savefig(savefile)

def make_plot(chainfile, savefile, true_parameter_values=None):
    """Make a plot of parameter posterior values"""
    import corner
    with open(chainfile+"_names.txt") as ff:
        names = ff.read().split('\n')
    pnames = [i.split(' ')[0] for i in names if len(i) > 0]
    samples = np.loadtxt(chainfile)
    plt.rc('font', family='serif', size=15.)
    corner.corner(samples, labels=pnames, truths=true_parameter_values)
    plt.savefig(savefile)

def generate_likelihood_class(testdir, emudir, mean_flux_label='s'):
    print('Beginning to initialise LikelihoodClass at', str(datetime.now()))
    return LikelihoodClass(basedir=emudir, datadir=testdir+"/AA0.97BB1.3CC0.67DD1.3heat_slope0.083heat_amp0.92hub0.69/output", mean_flux=mean_flux_label)

def run_and_plot_likelihood_samples(testdir, emudir, savefile, plotname, plot=True, chain_savedir=None, n_walkers=100, n_burn_in_steps=100, n_steps=400, while_loop=True, mean_flux_label='s', return_class_only=False):
    # TODO: Add true values #Read from filenames
    #true_parameter_values = [None, None, 0.97, 1.3, 0.67, 1.3, 0.083, 0.92, 0.69]
    true_parameter_values = [0.97, 1.3, 0.67, 1.3, 0.083, 0.92, 0.69]

    if chain_savedir is None:
        chain_savedir = testdir
    chainfile = chain_savedir + '/AA0.97BB1.3_chain_' + plotname + '.txt'

    like = generate_likelihood_class(testdir, emudir, mean_flux_label=mean_flux_label)

    if return_class_only is False:
        print('Beginning to sample likelihood at', str(datetime.now()))
        output = like.do_sampling(chainfile, nwalkers=n_walkers, burnin=n_burn_in_steps, nsamples=n_steps, while_loop=while_loop)
        if plot is True:
            print('Beginning to make corner plot at', str(datetime.now()))
            make_plot(chainfile, savefile, true_parameter_values=true_parameter_values)
        return like
    else:
        likelihood_at_true_values = like.likelihood(true_parameter_values)
        return like, likelihood_at_true_values