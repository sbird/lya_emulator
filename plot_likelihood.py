"""Module for plotting generated likelihood chains"""
import matplotlib
matplotlib.use('PDF')
import matplotlib.pyplot as plt
import corner

def make_plot(chainfile, savefile):
    """Make a plot of parameter posterior values"""
    samples = np.loadtxt(chainfile)
    names = np.loadtxt(chainfile + "_names.txt")
    #TODO: Add true values
    corner.corner(samples, labels=names)
    plt.savefig(savefile)
