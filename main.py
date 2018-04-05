"""Make some plots"""
import sys

from make_paper_plots import *
from coarse_grid_plot import *
from plot_likelihood import *

if __name__ == "__main__":
    sim_rootdir = "simulations"
    savedir = sys.argv[1]
    plotname = sys.argv[2]
    chain_savedir = sys.argv[3]

    testdir = sim_rootdir + '/hires_knots_test' #/share/hypatia/sbird
    emudir = sim_rootdir + '/hires_knots'

    likelihood_samples_plot_savefile = savedir + '/likelihood_samples_' + plotname + '.pdf'
    flux_power_plot_savefile = savedir + '/flux_power' + plotname + '.pdf'

    #test_knot_plots(testdir=testdir, emudir=emudir, plotname=plotname, kf_bin_nums=None, data_err=True) #"_All_kf2"
    #plot_test_interpolate_kf_bin_loop(emudir, testdir, savedir=savedir, plotname="_Two_loop", kf_bin_nums=np.arange(2))

    output = run_and_plot_likelihood_samples(testdir, emudir, likelihood_samples_plot_savefile, plotname, plot=True, chain_savedir=chain_savedir, n_burn_in_steps=50, n_steps=100, while_loop=False, mean_flux_label='c', return_class_only=False)
    #make_plot(chain_savedir + '/AA0.97BB1.3_chain_20000_MeanFluxFactor.txt', likelihood_samples_plot_savefile)
    make_plot_flux_power_spectra(testdir, emudir, flux_power_plot_savefile, mean_flux_label='s')
