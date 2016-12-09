"""Module for computing the likelihood function for the forest emulator."""
import os.path
import numpy as np
import emcee
import coarse_grid
import flux_power
import gpemulator

def lnlike_linear(params, gp=None, data=None):
    """A simple emcee likelihood function for the Lyman-alpha forest."""
    assert gp is not None
    assert data is not None
    predicted,_ = gp.predict(params.reshape(1,-1))
    diff = predicted[0]-data['pf']
    return -np.dot(diff,np.dot(data['invcovar'],diff))/2.0

def init_lnlike(basedir, datadir, mean_flux=False, max_z=4.2):
    """Initialise the emulator by loading the flux power spectra from the simulations."""
    #Parameter names
    sdss = gpemulator.SDSSData()
    data = {}
    myspec = flux_power.MySpectra(max_z=max_z)
    data['pf'] = myspec.get_flux_power(datadir,sdss.get_kf(),tau0_factors=[1.,])
    #Use the SDSS covariance matrix
    data['invcovar'] = sdss.get_icovar()
    params = coarse_grid.Emulator(basedir)
    params.load()
    gp = params.get_emulator(max_z=max_z, mean_flux=mean_flux)
    return gp, data, params.get_param_limits(include_dense=mean_flux)

def init_emcee(basedir, datadir, mean_flux=False):
    """Initialise and run emcee."""
    (gp, data, param_limits) = init_lnlike(basedir, datadir, mean_flux=mean_flux, max_z = 4.2)
    #Number of knots plus one cosmology plus one for mean flux.
    ndim = np.shape(param_limits)[0]
    nwalkers = 100
    #Limits: we need to hard-prior to the volume of our emulator.
    pr = (param_limits[:,1]-param_limits[:,0])
    pl = param_limits[:,0]
    p0 = [pr*np.random.rand(ndim)+pl for i in range(nwalkers)]
    sampler = emcee.EnsembleSampler(nwalkers, ndim, lnlike_linear, args=[gp,data])
    sampler.run_mcmc(p0, 1000)
    return sampler

if __name__ == "__main__":
    chains = init_emcee(os.path.expanduser("~/data/Lya_Boss/cosmo-only-emulator"), os.path.expanduser("~/data/Lya_Boss/cosmo-only-test/AA0.94BB1.2CC0.71DD1.2hub0.71/output/"),mean_flux=True)
