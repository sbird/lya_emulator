"""Module for plotting generated likelihood chains"""
import numpy as np
import matplotlib
matplotlib.use('PDF')
import matplotlib.pyplot as plt
import corner

from likelihood import *

def make_plot(chainfile, savefile):
    """Make a plot of parameter posterior values"""
    with open(chainfile+"_names.txt") as ff:
        names = ff.read().split('\n')
    pnames = [i.split(' ')[0] for i in names if len(i) > 0]
    #TODO: Add true values
    samples = np.loadtxt(chainfile)
    corner.corner(samples, labels=pnames)
    plt.savefig(savefile)

def run_and_plot_likelihood_samples(testdir, emudir, savefile, chain_savedir=None, n_walkers=100, n_burn_in_steps=100, n_steps=400):
    if chain_savedir is None:
        chain_savedir = testdir
    chainfile = chain_savedir + '/AA0.97BB1.3_chain.txt'
    like = LikelihoodClass(basedir=emudir, datadir=testdir+"/AA0.97BB1.3CC0.67DD1.3heat_slope0.083heat_amp0.92hub0.69/output")
    output = like.do_sampling(chainfile, nwalkers=n_walkers, burnin=n_burn_in_steps, nsamples=n_steps)
    make_plot(chainfile, savefile)