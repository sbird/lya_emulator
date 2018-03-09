"""Make plots for the first emulator paper"""
import os.path as path
import numpy as np
import latin_hypercube
import coarse_grid
import flux_power
from quadratic_emulator import QuadraticEmulator
from mean_flux import ConstMeanFlux
import lyman_data
import matplotlib
matplotlib.use("PDF")
import matplotlib.pyplot as plt
from plot_latin_hypercube import plot_points_hypercube
import coarse_grid_plot

#plotdir = path.expanduser("~/papers/emulator_paper_1/plots")
plotdir = '/home/keir/Plots/Emulator'
#plotdir = '/Users/kwame/Documents/emulator_paper_1/plots'
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
    mf = ConstMeanFlux(value=1.)
    emu = coarse_grid.Emulator(emulatordir, mf=mf)
    emu.load()
    par, flux_vectors = emu.get_flux_vectors(max_z=2.4)
    params = emu.param_names
    defpar = par[0,:]
    deffv = flux_vectors[0]
    for (name, index) in params.items():
        ind = np.where(par[:,index] != defpar[index])
        for i in np.ravel(ind):
            tp = par[i,index]
            fp = (flux_vectors[i]/deffv).reshape(-1,len(emu.kf))
            plt.semilogx(emu.kf, fp[0,:], label=name+"="+str(tp)+" (z=2.4)")
        plt.xlim(1e-3,2e-2)
        plt.ylim(ymin=0.6)
        plt.legend(loc=0)
        plt.savefig(path.join(plotdir,"single_param_"+name+".pdf"))
        plt.clf()

def test_s8_plots():
    """Plot emulator test-cases"""
    testdir = path.expanduser("~/data/Lya_Boss/hires_s8_test")
    quaddir = path.expanduser("~/data/Lya_Boss/hires_s8_quadratic")
    emudir = path.expanduser("~/data/Lya_Boss/hires_s8")
    gp_emu = coarse_grid_plot.plot_test_interpolate(emudir, testdir,savedir=path.join(plotdir,"hires_s8"))
    gp_quad = coarse_grid_plot.plot_test_interpolate(quaddir, testdir,savedir=path.join(plotdir,"hires_s8_quadratic"))
    quad_quad = coarse_grid_plot.plot_test_interpolate(quaddir, testdir,savedir=path.join(plotdir,"hires_s8_quad_quad"),emuclass=QuadraticEmulator)
    return (gp_emu, gp_quad, quad_quad)

def test_knot_plots(mf=1, testdir = None, emudir = None, kf_bin_nums=None):
    """Plot emulator test-cases"""
    if testdir is None:
        testdir = path.expanduser("~/data/Lya_Boss/hires_knots_test")
    if emudir is None:
        emudir = path.expanduser("~/data/Lya_Boss/hires_knots")
    gp_emu = coarse_grid_plot.plot_test_interpolate(emudir, testdir,savedir=path.join(plotdir,"hires_knots_mf"+str(mf)),mean_flux=mf,kf_bin_nums=kf_bin_nums)
    return gp_emu

def sample_var_plot():
    """Check the effect of sample variance"""
    mys = flux_power.MySpectra()
    sd = lyman_data.SDSSData()
    kf = sd.get_kf()
    fp0 = mys.get_snapshot_list("/home/spb/data/Lya_Boss/hires_sample/ns1.1As2.1e-09heat_slope0heat_amp1hub0.7/output/")
    fp1 = mys.get_snapshot_list("/home/spb/data/Lya_Boss/hires_sample/ns1.1As2.1e-09heat_slope0heat_amp1hub0.7seed1/output/")
    fp2 = mys.get_snapshot_list("/home/spb/data/Lya_Boss/hires_sample/ns1.1As2.1e-09heat_slope0heat_amp1hub0.7seed2/output/")
    pk0 = fp0.get_power(kf,tau0_factors=None)
    pk1 = fp1.get_power(kf,tau0_factors=None)
    pk2 = fp2.get_power(kf,tau0_factors=None)
    nred = len(mys.zout)
    nk = len(kf)
    assert np.shape(pk0) == (nred*nk,)
    for i in (5,10):
        plt.semilogx(kf,(pk1/pk2)[i*nk:(i+1)*nk],label="Seed 1 z="+str(mys.zout[i]))
        plt.semilogx(kf,(pk0/pk2)[i*nk:(i+1)*nk],label="Seed 2 z="+str(mys.zout[i]))
    plt.xlabel(r"$k_F$ (s/km)")
    plt.ylabel(r"Sample Variance Ratio")
    plt.title("Sample Variance")
    plt.xlim(xmax=0.05)
    plt.legend(loc=0)
    plt.savefig(path.join(plotdir, "sample_var.pdf"))
    plt.clf()

if __name__ == "__main__":
    sample_var_plot()
    hypercube_plot()
    single_parameter_plot()
    test_s8_plots()
    test_knot_plots(mf=1)
    test_knot_plots(mf=2)
