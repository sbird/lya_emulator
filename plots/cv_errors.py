"""Script to attempt to compute cosmic variance errors."""

import os.path
import h5py
import scipy.interpolate
import numpy as np
import matplotlib.pyplot as plt
from fake_spectra import fluxstatistics as fstat
from fake_spectra import spectra
from lyaemu.likelihood import LikelihoodClass
from lyaemu.flux_power import rebin_power_to_kms
from lyaemu import lyman_data as lyd


def compute_cosmic_variance_fps_hub(basedir="../dtau-48-48"):
    """Using that the effect of the hubble parameter on the flux power spectrum is dominated
       by the cosmic variance modes shifting between bins, we can estimate cosmic variance. """
    emulatordir = os.path.join(os.path.dirname(__file__), basedir)
    hremudir = os.path.join(os.path.dirname(__file__), basedir+"/hires")
    like = LikelihoodClass(basedir=emulatordir, HRbasedir=hremudir, data_corr=False, tau_thresh=1e6, loo_errors=False, traindir=emulatordir+"/trained_mf")
    plimits = like.param_limits
    means = np.mean(plimits, axis=1)
    _, defaultfv, _ = like.get_predicted(means)
    pnames = like.get_pnames()
    assert len(pnames) == np.size(means)
    nsamples = 30
    flux_vectors = np.zeros((nsamples,) + np.shape(defaultfv))
    i = 7
    assert pnames[i][0] == 'hub'
    spaced = np.linspace(plimits[i, 0], plimits[i,1], nsamples)
    #Compute means
    for (nn, ss) in enumerate(spaced):
        means[i] = ss
        _, fv2, _ = like.get_predicted(means)
        flux_vectors[nn, :] = fv2
    var = np.var(flux_vectors, axis=0)
    return var

def compute_cosmic_variance(basedir="../dtau-48-48", seed_flux="seed_converge.hdf5"):
    """Estimate the cosmic variance using the ratio between two simulations with different seeds."""
    boss = lyd.BOSSData()
    kfkms = boss.get_kf()
    with h5py.File(os.path.join(basedir,seed_flux),'r') as hh:
        #Low-res is current.
        flux_orig = hh["flux_powers"]["orig"][:]
        flux_seed = hh["flux_powers"]["seed"][:]
        kfmpc = hh["kfmpc"][:]
        zout = hh["zout"][:]
        #Loading an element in the training set
        omega_m = 0.288
        _, data_flux_orig = rebin_power_to_kms(kfkms=kfkms, kfmpc=kfmpc, flux_powers=flux_orig, zbins=zout, omega_m=omega_m)
        _, data_flux_seed = rebin_power_to_kms(kfkms=kfkms, kfmpc=kfmpc, flux_powers=flux_seed, zbins=zout, omega_m=omega_m)
    #Single-element estimate of the variance!
    return np.var([data_flux_seed[:], data_flux_orig[:]], axis=0)

def compute_cosmic_variance_ratio(basedir="../dtau-48-48", seed_flux="seed_converge.hdf5"):
    """Estimate the cosmic variance using the ratio between two simulations with different seeds."""
    boss = lyd.BOSSData()
    kfkms = boss.get_kf()
    with h5py.File(os.path.join(basedir,seed_flux),'r') as hh:
        #Low-res is current.
        flux_orig = hh["flux_powers"]["orig"][:]
        flux_seed = hh["flux_powers"]["seed"][:]
        kfmpc = hh["kfmpc"][:]
        zout = hh["zout"][:]
        #Loading an element in the training set
        omega_m = 0.288
        _, data_flux_ratio = rebin_power_to_kms(kfkms=kfkms, kfmpc=kfmpc, flux_powers=flux_orig/flux_seed, zbins=zout, omega_m=omega_m)
        # _, data_flux_seed = rebin_power_to_kms(kfkms=kfkms, kfmpc=kfmpc, flux_powers=flux_seed, zbins=zout, omega_m=omega_m)
    return np.abs(data_flux_ratio-1)
    #Single-element estimate of the variance!
    # return np.abs(data_flux_seed[:] / data_flux_orig[:]-1)

def get_loo_errors(l1norm=True, basedir="../dtau-48-48", savefile="loo_fps.hdf5", hremu=False):
    """Get the LOO errors"""
    if hremu:
        filepath = os.path.join(os.path.join(basedir, "hires"), savefile)
    else:
        filepath = os.path.join(basedir, savefile)
    ff = h5py.File(filepath, 'r')
    fpp, fpt, looz = ff['flux_predict'][:], ff['flux_true'][:], ff['zout'][:]
    ff.close()
    # after loading the absolute difference, calculate errors including BOSS data
    if l1norm:
        loo_errors = np.mean(np.abs(fpp - fpt), axis=0)
    else:
        loo_errors = np.sqrt(np.mean((fpp - fpt)**2, axis=0))
    return looz, loo_errors

def plot_errors(basedir=None):
    """Plot some different error terms."""
    looz, loo_error2 = get_loo_errors(l1norm=False)
    _, loo_error_hr= get_loo_errors(l1norm=False, hremu=True)
    cv_err = compute_cosmic_variance_ratio()
    #Get the eBOSS errors
    boss = lyd.BOSSData()
    kf = boss.get_kf()
    boss_diag = np.sqrt(np.diag(boss.get_covar()))
    boss_diag = boss_diag.reshape(13,-1)[::-1]
    boss_pf = (boss.get_pf()).reshape(13, -1)[::-1]

    for i in range(0,13):
        if i >=2:
            plt.plot(kf, cv_err[i-2,:], ls="--", label=r"$|P_F(\mathrm{Seed})|$", color="brown")
        plt.plot(kf, boss_diag[i,:] / boss_pf[i,:], ls="-", label=r"$\mathrm{diag}(\sqrt{K_\mathrm{BOSS}})$", color="black")
        plt.plot(kf, loo_error2[i,:]/boss_pf[i,:], ls="-.", color="blue", label=r"$\sigma_{CV}$")
        if basedir is not None:
            plot_axis_error(looz[i], basedir)
        # plt.plot(kf, loo_error_hr[i,:]/boss_pf[i,:], ls=":", label="LOOHR")
        plt.title("z=%.2g" % looz[i], fontsize=16)
        plt.legend(fontsize=13)
        plt.gca().tick_params(axis='both', which='major', labelsize=14)
        plt.ylim(ymin=1e-3)
        plt.yscale('log')
        plt.xlabel(r"$k_F$ (s/km)", fontsize=16)
        plt.ylabel(r"$\sigma / P_F(z,k)$", fontsize=16)
        plt.tight_layout()
        plt.savefig("err-%.2g.pdf" % looz[i])
        plt.clf()

def plot_cv_error():
    """Plot some different error terms."""
    looz, loo_error2 = get_loo_errors(l1norm=False)
    #Get the eBOSS errors
    boss = lyd.BOSSData()

    kf = boss.get_kf()
    boss_diag = np.sqrt(np.diag(boss.get_covar()))
    boss_diag = boss_diag.reshape(13,-1)[::-1]
    boss_pf = (boss.get_pf()).reshape(13, -1)[::-1]

    plt.figure()
    lss = ["-.", ":", "--", "-", "--", ":"]
    colors = ["black", "blue", "grey", "brown", "red", "orange"]
    for j,i in enumerate([9,6]):
        plt.plot(kf, boss_diag[i,:] / boss_pf[i,:], ls=lss[2*j], color=colors[2*j], label=r"$\sqrt{K_\mathrm{BOSS}}\; z=%.2g$" % looz[i])
        plt.plot(kf, loo_error2[i,:] / boss_pf[i,:], ls=lss[2*j+1], color=colors[2*j+1], label=r"$\sigma_{CV}$ z=%.2g" % looz[i])
        # plt.title("z=%.2g" % looz[i])
    plt.legend()
    plt.yscale('log')
    plt.xlabel(r"$k_F$ (s/km)")
    plt.ylabel(r"$\mathrm{diag}(\sqrt{K}) / P_F$")
    plt.savefig("errors_loo.pdf")
    plt.clf()

def plot_covar_errors():
    """Plot some different error terms."""
    looz, _ = get_loo_errors(l1norm=False)
    #Get the eBOSS errors
    boss = lyd.BOSSData()
    kf = boss.get_kf()
    boss_diag = np.sqrt(np.diag(boss.get_covar()))
    boss_diag = boss_diag.reshape(13,-1)[::-1]
    #Get the eBOSS errors
    desi = lyd.DESIEDRData()
    colors=["blue", "red", "black", "brown"]
    ccnt = 0
    for i in [5, 12]:
        plt.plot(kf, boss_diag[i,:], ls="-", label="BOSS z=%.2g" % looz[i], color=colors[ccnt])
        ccnt += 1
        if looz[i] < 3.9:
            desi_diag = np.sqrt(desi.get_covar_diag(zbin=looz[i]))
            dkf = desi.get_kf(zbin=looz[i])
            plt.plot(dkf, desi_diag, ls="--", label="DESI z=%.2g" % looz[i], color=colors[ccnt])
            ccnt += 1
    plt.legend()
    plt.xlabel(r"$k_F$ (s/km)")
    plt.ylabel(r"Diagonal Covariance")
    plt.savefig("covar-bossdesi.pdf")

def plot_axis_error(zz, basedir):
    """Plot the average leave-one-out axis ratio difference."""
    base = os.path.expanduser(basedir)
    for nn in range(4, 23):
        try:
            spec = spectra.Spectra(nn, base, None, None, savefile="lya_forest_spectra_grid_480.hdf5", res=None)
        except IOError:
            continue
        if np.abs(spec.red - zz) > 0.04:
            continue
        tau = spec.get_tau("H", 1, 1215)
        axis = spec.axis
        flux_power = []
        for ax in range(1,4):
            ii = np.where(axis == ax)[0]
            (kf, avg_flux_power) = fstat.flux_power(tau[ii,:], spec.vmax, spec_res=spec.spec_res, mean_flux_desired=None, window=False)
            flux_power.append(avg_flux_power)
        (kf, avg_flux_power) = fstat.flux_power(tau, spec.vmax, spec_res=spec.spec_res, mean_flux_desired=None, window=False)
        flux = np.mean([np.abs(flux_power[i]/avg_flux_power-1) for i in range(3)], axis=0)
        boss = lyd.BOSSData(datafile="dr14")
        kfkms = boss.get_kf()
        rebinned = scipy.interpolate.interp1d(kf[1:], flux[1:])
        newflux = rebinned(kfkms)
        plt.semilogx(kfkms, newflux, label=r"$\bar{P}_F(Axis)/P_F$", ls="--", color="grey")
        break

if __name__=="__main__":
#     based = os.path.expanduser("~/shared/Lya_emu_spectra/emu_full/ns0.972Ap1.69e-09herei3.87heref2.65alphaq2.12hub0.722omegamh20.144hireionz7.53bhfeedback0.0507/output/")
#     plot_errors(based)
    variance = compute_cosmic_variance_fps_hub()
    varrat, orig, seed = compute_cosmic_variance_ratio()
    with h5py.File("variance.hdf5", 'w') as hh:
        hh["var_hub"] = variance
        hh["var_rat"] = varrat
        hh["fps_orig"] = orig
        hh["fps_seed"] = seed
