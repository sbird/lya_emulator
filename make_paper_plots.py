"""Make plots for the first emulator paper"""
import os.path as path
import numpy as np
import latin_hypercube
import matplotlib
matplotlib.use("PDF")
import matplotlib.pyplot as plt
from plot_latin_hypercube import plot_points_hypercube
import coarse_grid
import coarse_grid_plot
import gpemulator
from quadratic_emulator import QuadraticEmulator

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
    xval = (a + b)/2
    xval_quad = np.concatenate([xval, np.repeat(xval[3],8)])
    yval_quad = np.concatenate([np.repeat(xval[3],8),xval])
    ndivision = 8
    xticks = np.linspace(0,1,ndivision+1)
    plt.scatter(xval_quad, yval_quad, marker='o', s=300, color="blue")
    plt.grid(b=True, which='major')
    plt.xticks(xticks)
    plt.yticks(xticks)
    plt.xlim(0,1)
    plt.ylim(0,1)
    plt.savefig(path.join(plotdir,"latin_hypercube_quadratic.pdf"))
    plt.clf()
    samples = latin_hypercube.get_hypercube_samples(limits, 8)
    plot_points_hypercube(samples[:,0], samples[:,1])
    plt.savefig(path.join(plotdir,"latin_hypercube_good.pdf"))
    plt.clf()

def single_parameter_plot():
    """Plot change in each parameter of an emulator from direct simulations."""
    emulatordir = path.expanduser("~/data/Lya_Boss/hires_s8_quadratic")
    data = gpemulator.SDSSData()
    kf = data.get_kf()
    emu = coarse_grid.Emulator(emulatordir)
    emu.load()
    gp = emu.get_emulator(max_z=4.2)
    params = emu.param_names
    defpar = gp.params[0,:]
    deffv = gp.powers[5].get_power(kf=kf, tau0_factor=1.)
    for (name, index) in params.items():
        ind = np.where(gp.params[:,index] != defpar[index])
        for i in np.ravel(ind):
            tp = gp.params[i,index]
            fp = (gp.powers[i].get_power(kf=kf, tau0_factor=1.)/deffv).reshape(-1,len(kf))
            plt.semilogx(kf, fp[7,:], label=name+"="+str(tp)+" (z=3)")
        plt.legend()
        plt.savefig(path.join(plotdir,"single_param_"+name+".pdf"))
        plt.clf()

def test_plots():
    """Plot emulator test-cases"""
    testdir = path.expanduser("~/data/Lya_Boss/hires_s8_test")
    emudir = path.expanduser("~/data/Lya_Boss/hires_s8_quadratic")
    quaddir = path.expanduser("~/data/Lya_Boss/hires_s8")
    gp_emu = coarse_grid_plot.plot_test_interpolate(emudir, testdir,savedir=path.join(plotdir,"hires_s8"))
    gp_quad = coarse_grid_plot.plot_test_interpolate(quaddir, testdir,savedir=path.join(plotdir,"hires_s8_quadratic"))
    quad_quad = coarse_grid_plot.plot_test_interpolate(quaddir, testdir,savedir=path.join(plotdir,"hires_s8_quad_quad"),emuclass=QuadraticEmulator)
    return (gp_emu, gp_quad, quad_quad)

if __name__ == "__main__":
    hypercube_plot()
    single_parameter_plot()
    test_plots()
