"""Script to attempt to compute cosmic variance errors."""

import os.path
import h5py
import numpy as np
from lyaemu.likelihood import LikelihoodClass
from lyaemu.flux_power import rebin_power_to_kms
from lyaemu.lyman_data import BOSSData

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
    variance = np.var(flux_vectors, axis=0)
    return variance

def compute_cosmic_variance_ratio(basedir="../dtau-48-48", seed_flux="seed_converge.hdf5"):
    """Estimate the cosmic variance using the ratio between two simulations with different seeds."""
    boss = BOSSData()
    kfkms = boss.get_kf()
    with h5py.File(os.path.join(basedir,seed_flux),'r') as hh:
        #Low-res is current.
        flux_orig = hh["flux_powers"]["orig"][:]
        flux_seed = hh["flux_powers"]["seed"][:]
        kfmpc = hh["kfmpc"][:]
        zout = hh["zout"][:]
        nk = np.shape(kfmpc)[0]
        #Loading an element in the training set
        omega_m = 0.288
        _, data_flux_orig = rebin_power_to_kms(kfkms=kfkms, kfmpc=kfmpc, flux_powers=flux_orig, zbins=zout, omega_m=omega_m)
        _, data_flux_seed = rebin_power_to_kms(kfkms=kfkms, kfmpc=kfmpc, flux_powers=flux_seed, zbins=zout, omega_m=omega_m)
    #Single-element estimate of the variance!
    return data_flux_seed**2 + data_flux_orig**2 - (data_flux_seed + data_flux_orig)**2/4, data_flux_orig, data_flux_seed

if __name__=="__main__":
    variance = compute_cosmic_variance_fps_hub()
    varrat, orig, seed = compute_cosmic_variance_ratio()
    with h5py.File("variance.hdf5", 'w') as hh:
        hh["var_hub"] = variance
        hh["var_rat"] = varrat
        hh["fps_orig"] = orig
        hh["fps_seed"] = seed
