"""Module for plotting generated likelihood chains"""
import numpy as np
import matplotlib
matplotlib.use('PDF')
import matplotlib.pyplot as plt
#import corner
from datetime import datetime

from likelihood import *

def make_plot(chainfile, savefile):
    """Make a plot of parameter posterior values"""
    import corner

    with open(chainfile+"_names.txt") as ff:
        names = ff.read().split('\n')
    pnames = [i.split(' ')[0] for i in names if len(i) > 0]
    #TODO: Add true values
    samples = np.loadtxt(chainfile)
    corner.corner(samples, labels=pnames)
    plt.savefig(savefile)

def run_and_plot_likelihood_samples(testdir, emudir, savefile, plotname, plot=True, chain_savedir=None, n_walkers=100, n_burn_in_steps=100, n_steps=400, while_loop=True, mean_flux_label='s'):
    if chain_savedir is None:
        chain_savedir = testdir
    chainfile = chain_savedir + '/AA0.97BB1.3_chain_' + plotname + '.txt'
    print('Beginning to initialise LikelihoodClass at', str(datetime.now()))
    like = LikelihoodClass(basedir=emudir, datadir=testdir+"/AA0.97BB1.3CC0.67DD1.3heat_slope0.083heat_amp0.92hub0.69/output", mean_flux=mean_flux_label)
    print('Beginning to sample likelihood at', str(datetime.now()))
    output = like.do_sampling(chainfile, nwalkers=n_walkers, burnin=n_burn_in_steps, nsamples=n_steps, while_loop=while_loop)
    if plot is True:
        print('Beginning to make corner plot at', str(datetime.now()))
        make_plot(chainfile, savefile)