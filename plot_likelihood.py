"""Module for plotting generated likelihood chains"""
import numpy as np
import matplotlib
matplotlib.use('PDF')
import matplotlib.pyplot as plt
import corner

def make_plot(chainfile, savefile):
    """Make a plot of parameter posterior values"""
    with open(chainfile+"_names.txt") as ff:
        names = ff.read().split('\n')
    pnames = [i.split(' ')[0] for i in names if len(i) > 0]
    #TODO: Add true values
    samples = np.loadtxt(chainfile)
    corner.corner(samples, labels=pnames)
    plt.savefig(savefile)
