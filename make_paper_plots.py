"""Make plots for the first emulator paper"""
import os.path as path
import numpy as np
import matplotlib
matplotlib.use("PDF")
import matplotlib.pyplot as plt
import latin_hypercube
from plot_latin_hypercube import plot_points_hypercube

plotdir = path.expanduser("~/papers/emulator_paper_1/plots")
def hypercube_plot():
    """Make a plot of some hypercubes"""
    limits = np.array([[0,1],[0,1]])
    cut = np.linspace(0, 1, 8 + 1)
    # Fill points uniformly in each interval
    a = cut[:8]
    b = cut[1:8 + 1]
    #Get list of central values
    xval = (a + b)/2
    plot_points_hypercube(xval, xval)
    plt.savefig(path.join(plotdir,"latin_hypercube_bad.pdf"))
    plt.clf()
    samples = latin_hypercube.get_hypercube_samples(limits, 8)
    plot_points_hypercube(samples[:,0], samples[:,1])
    plt.savefig(path.join(plotdir,"latin_hypercube_good.pdf"))
    plt.clf()


if __name__ == "__main__":
    hypercube_plot()
