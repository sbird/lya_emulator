"""Make some plots"""
import sys

from make_paper_plots import *
from coarse_grid_plot import *
from plot_likelihood import *
from plot_latin_hypercube import *

if __name__ == "__main__":
    sim_rootdir = sys.argv[1]
    savedir = sys.argv[2]
    plotname = sys.argv[3]
    chain_savedir = sys.argv[4]

    testdir = sim_rootdir + '/hires_s8_test' #'/hot_cold_test' #/share/hypatia/sbird
    emudir = sim_rootdir + '/hires_s8' #'/hot_cold'

    simulation_sub_directory1 = '/HeliumHeatAmp0.9/output'
    simulation_sub_directory2 = '/HeliumHeatAmp1.1/output'

    likelihood_samples_plot_savefile = savedir + '/likelihood_samples_' + plotname + '.pdf'
    flux_power_plot_savefile = savedir + '/flux_power_' + plotname + '.pdf'
    compare_plot_savefile = savedir + '/flux_power_comparison_' + plotname + '.pdf'
    emulator_error_plot_savefile = savedir + '/emulator_error_' + plotname + '.pdf'
    initial_parameter_samples_plot_savefile = savedir + '/initial_parameter_samples_' + plotname + '.pdf'

    #test_knot_plots(testdir=testdir, emudir=emudir, plotdir=savedir, plotname=plotname, mf=2, kf_bin_nums=None, data_err=False, max_z=4.2)
    #plot_test_interpolate_kf_bin_loop(emudir, testdir, savedir=savedir, plotname="_Two_loop", kf_bin_nums=np.arange(2))

    #output = run_and_plot_likelihood_samples(testdir, emudir, likelihood_samples_plot_savefile, plotname, plot=True, chain_savedir=chain_savedir, n_burn_in_steps=50, n_steps=150, while_loop=False, mean_flux_label='s', return_class_only=False, rescale_data_error=False, include_emulator_error=True) #, max_z=2.6)
    #make_plot(chain_savedir + '/AA0.97BB1.3_chain_20000_MeanFluxFactor.txt', likelihood_samples_plot_savefile)
    output = make_plot_flux_power_spectra(testdir, emudir, flux_power_plot_savefile, mean_flux_label='s', rescale_data_error=True)
    #make_plot_compare_two_simulations(emudir, emudir, simulation_sub_directory1, simulation_sub_directory2, compare_plot_savefile)
    #make_plot_emulator_error(emudir, emulator_error_plot_savefile, mean_flux_label='s') #, max_z=2.6)
    #output = make_plot_initial_parameter_samples(initial_parameter_samples_plot_savefile)