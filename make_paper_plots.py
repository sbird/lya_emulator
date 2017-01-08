"""Make plots for the first emulator paper"""
import os.path as path
import numpy as np
import matplotlib
matplotlib.use("PDF")
import matplotlib.pyplot as plt
import latin_hypercube
from plot_latin_hypercube import plot_points_hypercube
import coarse_grid
import gpemulator

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

def single_parameter_plot():
    """Plot change in each parameter of an emulator from direct simulations."""
    emulatordir = path.expanduser("~/data/Lya_Boss/emulator_quadratic")
    data = gpemulator.SDSSData()
    kf = data.get_kf()
    emu = coarse_grid.Emulator(emulatordir)
    emu.load()
    gp = emu.get_emulator(mean_flux=True, max_z=4.2)
    params = emu.param_names
    defpar = gp.params[5,:]
    deffv = gp.flux_vectors[5,:]
    for (name, index) in params.items():
        ind = np.where((gp.params[:,index] != defpar[index])*(gp.params[:,-1]==defpar[-1]))
        for i in np.ravel(ind):
            tp = gp.params[i,index]
            fp = (gp.flux_vectors[i,:]/deffv).reshape(-1,len(kf))
            nred = np.shape(fp)[0]
            plt.semilogx(kf, fp[7,:], label=name+"="+str(tp)+" (z=3)")
        plt.legend()
        plt.savefig(path.join(plotdir,"single_param_"+name+".pdf"))
        plt.clf()

if __name__ == "__main__":
    hypercube_plot()
