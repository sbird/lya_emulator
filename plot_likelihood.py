"""Module for plotting generated likelihood chains"""
import json
import os
import glob
import math as mh
from datetime import datetime
import numpy as np
import likelihood as likeh
import distinct_colours_py3 as dc
import matplotlib
matplotlib.use('PDF')
import matplotlib.pyplot as plt


def make_plot_flux_power_spectra(like, savefile):
    """Make a plot of the power spectra, with redshift, the BOSS power and the sigmas. Four plots stacked."""
    k_los = like.gpemu.kf
    n_k_los = k_los.size
    z = like.zout #Highest redshift first
    n_z = z.size
    exact_flux_power = like.data_fluxpower.reshape(n_z, n_k_los)
    emulated_flux_power = like.emulated_flux_power[0].reshape(n_z, n_k_los)
    emulated_flux_power_std = like.emulated_flux_power_std[0].reshape(n_z, n_k_los)
    data_flux_power = like.sdss.pf.reshape(-1, n_k_los)[:n_z][::-1]

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

        axes[2].plot(k_los, data_flux_power_std_single_z / emulated_flux_power[i], color=distinct_colours[i], ls='-', lw=line_width)
        axes[2].plot(k_los, emulated_flux_power_std[i] / emulated_flux_power[i], color=distinct_colours[i], ls='--',
                     lw=line_width)

        axes[3].plot(k_los, data_flux_power_std_single_z / data_flux_power[i], color=distinct_colours[i], ls='-', lw=line_width)

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
    axes[2].set_ylabel(r'sigma / emulated P(k)')

    axes[3].set_xlim(xlim)
    axes[3].set_yscale('log')
    axes[3].set_xlabel(xlabel)
    axes[3].set_ylabel(r'BOSS sigma / BOSS P(k)')

    figure.subplots_adjust(hspace=0)
    plt.savefig(savefile)
    plt.show()

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

def get_simulation_parameters_knots(base):
    """Get the parameters of a knot-based simulation from the SimulationICs JSON file."""
    jsin = open(os.path.join(base, "../SimulationICs.json"), 'r')
    pp = json.load(jsin)
    knv = pp["knot_val"]
    #This will fail!
    assert pp["code_args"]["rescale_gamma"] is True
    parvec = [0., 1., *knv, pp["code_args"]["rescale_slope"], pp["code_args"]["rescale_amp"], pp["hubble"]]
    return parvec

def get_simulation_parameters_s8(base):
    """Get the parameters of a sigma8-ns-based simulation from the SimulationICs JSON file."""
    jsin = open(os.path.join(base, "../SimulationICs.json"), 'r')
    pp = json.load(jsin)
    assert pp["code_args"]["rescale_gamma"] is True
    parvec = [0., 1., pp['ns'], pp['scalar_amp'], pp["code_args"]["rescale_slope"], pp["code_args"]["rescale_amp"], pp["hubble"]]
    return parvec

def run_likelihood_test(testdir, emudir, plot=True, mean_flux_label='s'):
    """Generate some likelihood samples"""
    like = likeh.LikelihoodClass(basedir=emudir, mean_flux=mean_flux_label)

    #Find all subdirectories
    subdirs = glob.glob(testdir + "/*/")

    for sdir in subdirs:
        sname = os.path.basename(os.path.abspath(sdir))
        chainfile = os.path.join(emudir, 'chain_' + sname + '.txt')
        print('Beginning to sample likelihood at', str(datetime.now()))
        output = like.do_sampling(chainfile, datadir=sdir)
        if plot is True:
            true_parameter_values = get_simulation_parameters_s8(sdir)
            print('Beginning to make corner plot at', str(datetime.now()))
            savefile = os.path.join(emudir, 'corner_'+sname + ".pdf")
            make_plot(chainfile, savefile, true_parameter_values=true_parameter_values)
            fp_savefile = os.path.join(emudir, 'flux_power_'+sname + ".pdf")
            make_plot_flux_power_spectra(like, fp_savefile)
        return like, output

if __name__ == "__main__":
    sim_rootdir = "simulations"
    emud = sim_rootdir + '/hires_s8'
    testdirs = sim_rootdir + '/hires_s8_test'

    run_likelihood_test(testdirs, emud)
