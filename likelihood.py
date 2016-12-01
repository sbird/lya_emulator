"""Module for computing the likelihood function for the forest emulator."""
import numpy as np
import emcee
import coarse_grid
import flux_power

def lnlike_linear(params, gp=None, data=None):
    """A simple emcee likelihood function for the Lyman-alpha forest."""
    assert gp is not None
    assert data is not None
    predicted,cov = gp.predict(params)
    diff = predicted-data.pf
    return -np.dot(diff,np.dot(data.invcovar + np.identity(np.size(diff))/cov,diff))/2.0

def init_lnlike(basedir, datadir, mean_flux=False, max_z=4.2):
    """Initialise the emulator by loading the flux power spectra from the simulations."""
    #Parameter names
    params = coarse_grid.Emulator(basedir)
    params.load()
    gp = params.get_emulator(max_z=max_z, mean_flux=mean_flux)
    myspec = flux_power.MySpectra(max_z=max_z)
    data = myspec.get_flux_power(datadir,params.kf,tau0_factor=1.,flat=True)
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
