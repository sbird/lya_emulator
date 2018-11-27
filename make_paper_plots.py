"""Make plots for the first emulator paper"""
import os.path as path
import re
import numpy as np
import latin_hypercube
import coarse_grid
import flux_power
from quadratic_emulator import QuadraticEmulator
from mean_flux import ConstMeanFlux,MeanFluxFactor
import lyman_data
import matplotlib
matplotlib.use("PDF")
import matplotlib.pyplot as plt
from plot_latin_hypercube import plot_points_hypercube
import coarse_grid_plot
from plot_likelihood import make_plot

#plotdir = path.expanduser("~/papers/emulator_paper_1/plots")
#plotdir = '/home/keir/Plots/Emulator'
plotdir = 'plots'

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


def dlogPfdt(spec, t1, t2):
    """Computes the change in flux power with optical depth"""
    pf1 = spec.get_flux_power_1D("H",1,1215,mean_flux_desired=np.exp(-t1))
    pf2 = spec.get_flux_power_1D("H",1,1215,mean_flux_desired=np.exp(-t2))
    return (pf1[0], (np.log(pf1[1]) - np.log(pf2[1]))/(t1-t2))

def show_t0_gradient(spec, tmin,tmax,steps=20):
    """Find the mean gradient of the flux power with tau0"""
    tt = np.linspace(tmin,tmax,steps)
    df = [np.mean(dlogPfdt(spec, t,t-0.005)[1]) for t in tt]
    return tt, df

def mean_flux_rescale():
    """Plot the effect of changing the mean flux as a function of cosmology."""
    emulatordir = path.expanduser("simulations/hires_s8_quadratic")
    mf = MeanFluxFactor()
    emu = coarse_grid.Emulator(emulatordir, mf=mf)
    emu.load()
    par, kfs, flux_vectors = emu.get_flux_vectors(max_z=2.4)
    nmflux = mf.dense_samples
    nsims = np.shape(emu.get_parameters())[0]
    lss = ["-", "--", ":", "-."]
    for (name, iparam) in emu.param_names.items():
        ind = np.where(par[:nsims, iparam+1] != par[0, iparam+1])
        js = [0, ind[0][2]]
        for jj in range(2):
            j = js[jj]
            simpar = par[j::nsims]
            simkf = kfs[j::nsims]
            nk = np.shape(simkf)[-1]
            simflux = flux_vectors[j::nsims]
            defpar = simpar[nmflux//2,0]
            deffv = simflux[nmflux//2][:nk]
            ind = np.where(simpar[:,0] != defpar)
            assert np.size(ind) > 0
            for i in np.ravel(ind):
                tp = simpar[i,0]
                fp = simflux[i][:nk]/deffv
                assert np.shape(kfs[i][0]) == np.shape(fp)
                plt.semilogx(kfs[i][0], fp, ls=lss[jj%4], label=r"$\tau_0$=%.3g" % tp)
        plt.xlim(1e-3,2e-2)
        plt.xlabel(r"$k_F$ (s/km)")
        plt.ylabel(r'$P_\mathrm{F}(k)$ ratio')
        plt.ylim(ymin=0.3)
        plt.legend(loc="lower left",ncol=4, fontsize=8)
        plt.title("Mean flux, z=2.4, varying "+name)
        plt.savefig(path.join(plotdir,"sp_"+name+"_mean_flux.pdf"))
        plt.clf()
    return par

def single_parameter_plot():
    """Plot change in each parameter of an emulator from direct simulations."""
    emulatordir = path.expanduser("simulations/hires_s8_quadratic")
    mf = ConstMeanFlux(value=1.)
    emu = coarse_grid.Emulator(emulatordir, mf=mf)
    emu.load()
    par, kfs, flux_vectors = emu.get_flux_vectors(max_z=2.4)
    params = emu.param_names
    defpar = par[0,:]
    deffv = flux_vectors[0]
    for (name, index) in params.items():
        ind = np.where(par[:,index] != defpar[index])
        for i in np.ravel(ind):
            tp = par[i,index]
            fp = (flux_vectors[i]/deffv)
            plt.semilogx(kfs[i][0], fp[0:np.size(kfs[i][0])], label=name+"=%.2g (z=2.4)" % tp)
            plt.semilogx(kfs[i][1], fp[np.size(kfs[i][0]):], label=name+"=%.2g (z=2.2)" % tp, ls="--")
        plt.xlim(1e-3,2e-2)
        plt.ylim(bottom=0.8, top=1.1)
        plt.legend(loc="lower left", ncol=2,fontsize=8)
        plt.savefig(path.join(plotdir,"single_param_"+name+".pdf"))
        plt.clf()

def test_s8_plots():
    """Plot emulator test-cases"""
    testdir = path.expanduser("simulations/hires_s8_test")
    quaddir = path.expanduser("simulations/hires_s8_quadratic")
    emudir = path.expanduser("simulations/hires_s8")
    gp_emu, _ = coarse_grid_plot.plot_test_interpolate(emudir, testdir,savedir=path.join(plotdir,"hires_s8"),mean_flux=2)
    #Also test with the quadratic emulator
    coarse_grid_plot.plot_test_interpolate(emudir, quaddir,savedir=path.join(plotdir,"hires_s8"),mean_flux=2)
    gp_quad, _ = coarse_grid_plot.plot_test_interpolate(quaddir, testdir,savedir=path.join(plotdir,"hires_s8_quadratic"),mean_flux=2)
    quad_quad, _ = coarse_grid_plot.plot_test_interpolate(quaddir, testdir,savedir=path.join(plotdir,"hires_s8_quad_quad"),emuclass=QuadraticEmulator, mean_flux=2)
    return (gp_emu, gp_quad, quad_quad)

def test_knot_plots(mf=1, testdir = None, emudir = None, plotdir = None, plotname="", max_z=4.2):
    """Plot emulator test-cases"""
    if testdir is None:
        testdir = path.expanduser("simulations/hires_knots_test")
    if emudir is None:
        emudir = path.expanduser("simulations/hires_knots")
    if plotdir is None:
        plotdir = path.expanduser('plots/hires_knots_mf')
    gp_emu,_ = coarse_grid_plot.plot_test_interpolate(emudir, testdir,savedir=plotdir+str(mf),plotname=plotname,mean_flux=mf,max_z=max_z)
    return gp_emu

def sample_var_plot():
    """Check the effect of sample variance"""
    mys = flux_power.MySpectra()
    sd = lyman_data.SDSSData()
    kf = sd.get_kf()
    fp0 = mys.get_snapshot_list("simulations/hires_sample/ns1.1As2.1e-09heat_slope0heat_amp1hub0.7/output/")
    fp1 = mys.get_snapshot_list("simulations/hires_sample/ns1.1As2.1e-09heat_slope0heat_amp1hub0.7seed1/output/")
    fp2 = mys.get_snapshot_list("simulations/hires_sample/ns1.1As2.1e-09heat_slope0heat_amp1hub0.7seed2/output/")
    pk0 = fp0.get_power(kf,mean_fluxes=None)
    pk1 = fp1.get_power(kf,mean_fluxes=None)
    pk2 = fp2.get_power(kf,mean_fluxes=None)
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

def plot_likelihood_chains(tau0=1.):
    """Plot the chains we made from quadratic and GP emulators."""
    cdir = "ns0.968As1.5e-09heat_slope-0.367heat_amp0.8hub0.692"
    sdir = path.join("simulations/hires_s8_test", cdir)
    true_parameter_values = coarse_grid.get_simulation_parameters_s8(sdir, t0=tau0)
    if tau0 != 1.0:
        cdir = re.sub(r"\.","_", "tau0%.3g" % tau0) + cdir

    chainfile = path.join("simulations/hires_s8", "chain_"+cdir+".txt")
    savefile = path.join(plotdir, 'hires_s8/corner_'+cdir + ".pdf")
    make_plot(chainfile, savefile, true_parameter_values=true_parameter_values)
    chainfile = path.join("simulations/hires_s8", "chain_"+cdir+".txt-noemuerr")
    savefile = path.join(plotdir, 'hires_s8/corner_'+cdir + "-noemuerr.pdf")
    make_plot(chainfile, savefile, true_parameter_values=true_parameter_values)
    chainfile = path.join("simulations/hires_s8_quadratic", "chain_"+cdir+".txt")
    savefile = path.join(plotdir, 'hires_s8_quad_quad/corner_'+cdir + ".pdf")
    make_plot(chainfile, savefile, true_parameter_values=true_parameter_values)

if __name__ == "__main__":
    plot_likelihood_chains(tau0=0.95)
    plot_likelihood_chains()
    gp_emu, gp_quad, gp_quad_quad = test_s8_plots()
    single_parameter_plot()
#     pars = mean_flux_rescale()
    hypercube_plot()
#     sample_var_plot()
#     test_knot_plots(mf=1)
#     test_knot_plots(mf=2)
