import sys

from make_paper_plots import *
from coarse_grid_plot import *
from plot_likelihood import *

if __name__ == "__main__":
    sim_rootdir = sys.argv[1]
    savedir = sys.argv[2]
    plotname = sys.argv[3]
    chain_savedir = sys.argv[4]

    testdir = sim_rootdir + '/Lya_Boss/hires_knots_test' #/share/hypatia/sbird
    emudir = sim_rootdir + '/Lya_Boss/hires_knots'

    likelihood_samples_plot_savefile = savedir + '/likelihood_samples_' + plotname + '.pdf'

    #test_knot_plots(testdir=testdir, emudir=emudir, plotname=plotname, kf_bin_nums=None, data_err=True) #"_All_kf2"
    #plot_test_interpolate_kf_bin_loop(emudir, testdir, savedir=savedir, plotname="_Two_loop", kf_bin_nums=np.arange(2))

    run_and_plot_likelihood_samples(testdir, emudir, likelihood_samples_plot_savefile, plotname, plot=True, chain_savedir=chain_savedir, n_burn_in_steps=5000, n_steps=5000, while_loop=False, mean_flux_label='c')
