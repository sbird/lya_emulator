"""Module for plotting generated likelihood chains"""
import os
import re
import glob
import math as mh
from datetime import datetime
import numpy as np
import distinct_colours_py3 as dc
import lyman_data
import likelihood as likeh
from coarse_grid import get_simulation_parameters_s8
import matplotlib
matplotlib.use('PDF')
import matplotlib.pyplot as plt
import getdist as gd
import getdist.plots as gdp

def get_k_z(likelihood_instance):
    """Get k and z bins"""
    k_los = likelihood_instance.gpemu.kf
    n_k_los = k_los.size
    z = likelihood_instance.zout #Highest redshift first
    n_z = z.size
    return k_los, z, n_k_los, n_z

def make_plot_flux_power_spectra(like, params, datadir, savefile, t0=1.):
    """Make a plot of the power spectra, with redshift, the BOSS power and the sigmas. Four plots stacked."""
    sdss = lyman_data.BOSSData()
    #'Data' now is a simulation
    k_los = sdss.get_kf()
    n_k_los = k_los.size
    z = like.zout #Highest redshift first
    n_z = z.size

    assert params[1] == t0
    data_fluxpower = likeh.load_data(datadir, kf=k_los, t0=t0)
    exact_flux_power = data_fluxpower.reshape(n_z, n_k_los)

    ekf, emulated_flux_power, emulated_flux_power_std = like.get_predicted(params)

    data_flux_power = like.sdss.pf.reshape(-1, n_k_los)[:n_z][::-1]

    figure, axes = plt.subplots(nrows=4, ncols=1, figsize=(6.4*2., 10.))
    distinct_colours = dc.get_distinct(n_z)
    for i in range(n_z):
        idp = np.where(k_los >= ekf[i][0])

        scaling_factor = ekf[i]/ mh.pi
        data_flux_power_std_single_z = np.sqrt(like.sdss.get_covar(z[i]).diagonal())
        exact_flux_power_std_single_z = np.sqrt(np.diag(like.get_BOSS_error(i)))
#         print('Diagonal elements of BOSS covariance matrix at single redshift:', data_flux_power_std_single_z)

        line_width = 0.5
        axes[0].plot(ekf[i], exact_flux_power[i][idp]*scaling_factor, color=distinct_colours[i], ls='-', lw=line_width, label=r'$z = %.1f$'%z[i])
        axes[0].plot(ekf[i], emulated_flux_power[i]*scaling_factor, color=distinct_colours[i], ls='--', lw=line_width)
        axes[0].errorbar(ekf[i], emulated_flux_power[i]*scaling_factor, yerr=emulated_flux_power_std[i]*scaling_factor, ecolor=distinct_colours[i], ls='')

        axes[1].plot(ekf[i], data_flux_power[i][idp]*scaling_factor, color=distinct_colours[i], lw=line_width)
        axes[1].errorbar(ekf[i], data_flux_power[i][idp]*scaling_factor, yerr=data_flux_power_std_single_z[idp]*scaling_factor, ecolor=distinct_colours[i], ls='')

        axes[2].plot(ekf[i], exact_flux_power_std_single_z[idp] / exact_flux_power[i][idp], color=distinct_colours[i], ls='-', lw=line_width)
        axes[2].plot(ekf[i], emulated_flux_power_std[i] / exact_flux_power[i][idp], color=distinct_colours[i], ls='--',
                     lw=line_width)

        #axes[3].plot(ekf[i], data_flux_power_std_single_z / data_flux_power[i], color=distinct_colours[i], ls='-', lw=line_width)
        axes[3].plot(ekf[i], emulated_flux_power[i] / exact_flux_power[i][idp], color=distinct_colours[i], ls='-', lw=line_width)
        print('z=%.2g Max frac overestimation of P_F =' % z[i], np.max((emulated_flux_power[i] / exact_flux_power[i][idp]) - 1.))
        print('z=%.2g Min frac underestimation of P_F =' % z[i] , np.min((emulated_flux_power[i] / exact_flux_power[i][idp]) - 1.))

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

    #make_plot_emulator_error(emudir, '/home/keir/Plots/Emulator/emulator_error_hot_cold.pdf', likelihood_instance=like)

    return like

def make_plot(chainfile, savefile, true_parameter_values=None):
    """Make a getdist plot"""
    with open(chainfile+"_names.txt") as ff:
        names = ff.read().split('\n')
    pnames = [i.split(' ')[0] for i in names if len(i) > 0]
    samples = np.loadtxt(chainfile)
    posterior_MCsamples = gd.MCSamples(samples=samples, names=pnames, labels=pnames, label='')

    subplot_instance = gdp.getSubplotPlotter()
    subplot_instance.triangle_plot([posterior_MCsamples], filled=True)
#     colour_array = np.array(['black', 'red', 'magenta', 'green', 'green', 'purple', 'turquoise', 'gray', 'red', 'blue'])

    for pi in range(samples.shape[1]):
        for pi2 in range(pi + 1):
            #Place horizontal and vertical lines for the true point
            ax = subplot_instance.subplots[pi, pi2]
            ax.axvline(true_parameter_values[pi2], color='gray', ls='--', lw=0.75)
            if pi2 < pi:
                ax.axhline(true_parameter_values[pi], color='gray', ls='--', lw=0.75)
                #Plot the emulator points
#                 if parameter_index > 1:
#                     ax.scatter(simulation_parameters_latin[:, parameter_index2 - 2], simulation_parameters_latin[:, parameter_index - 2], s=54, color=colour_array[-1], marker='+')

#     legend_labels = ['+ Initial Latin hypercube']
#     subplot_instance.add_legend(legend_labels, legend_loc='upper right', colored_text=True, figure=True)
    plt.savefig(savefile)

def run_likelihood_test(testdir, emudir, savedir=None, plot=True, mean_flux_label='s', t0_training_value=1., emulator_class="standard"):
    """Generate some likelihood samples"""

    #Find all subdirectories
    subdirs = glob.glob(testdir + "/*/")
    assert len(subdirs) > 1

    like = likeh.LikelihoodClass(basedir=emudir, mean_flux=mean_flux_label, t0_training_value = t0_training_value, emulator_class=emulator_class)
    for sdir in subdirs:
        single_likelihood_plot(sdir, like, savedir=savedir, plot=plot, t0=t0_training_value)
    return like

def single_likelihood_plot(sdir, like, savedir, plot=True, t0=1.):
    """Make a likelihood and error plot for a single simulation."""
    sname = os.path.basename(os.path.abspath(sdir))
    if t0 != 1.0:
        sname = re.sub(r"\.","_", "tau0%.3g" % t0) + sname
    chainfile = os.path.join(savedir, 'chain_' + sname + '.txt')
    sname = re.sub(r"\.", "_", sname)
    datadir = os.path.join(sdir, "output")
    true_parameter_values = get_simulation_parameters_s8(sdir, t0=t0)
    if plot is True:
        fp_savefile = os.path.join(savedir, 'flux_power_'+sname + ".pdf")
        make_plot_flux_power_spectra(like, true_parameter_values, datadir, savefile=fp_savefile, t0=t0)
    print('Beginning to sample likelihood at', str(datetime.now()))
    if not os.path.exists(chainfile):
        like.do_sampling(chainfile, datadir=datadir)
    print('Done sampling likelihood at', str(datetime.now()))
    if plot is True:
        savefile = os.path.join(savedir, 'corner_'+sname + ".pdf")
        make_plot(chainfile, savefile, true_parameter_values=true_parameter_values)

if __name__ == "__main__":
    sim_rootdir = "simulations2"
    plotdir = 'plots/simulations2'
    gpsavedir=os.path.join(plotdir,"hires_s8")
    quadsavedir = os.path.join(plotdir, "hires_s8_quad_quad")
    emud = os.path.join(sim_rootdir,'hires_s8')
    quademud = os.path.join(sim_rootdir, "hires_s8_quadratic")
    testdirs = os.path.join(sim_rootdir,'hires_s8_test')

    gplike09 = run_likelihood_test(testdirs, emud, savedir=gpsavedir, plot=True, t0_training_value=0.9)
#     gplike = run_likelihood_test(testdirs, emud, savedir=gpsavedir, plot=True)
    quadlike09 = run_likelihood_test(testdirs, quademud, savedir=quadsavedir, plot=True, t0_training_value=0.9, emulator_class="quadratic")
#     quadlike = run_likelihood_test(testdirs, quademud, savedir=quadsavedir, plot=True, emulator_class="quadratic")
