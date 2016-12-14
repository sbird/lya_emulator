"""Test the emulator using a simple linear theory model."""
import numpy as np
# import emcee
import gpemulator

import linear_theory
import latin_hypercube

def lnlike_linear(params, *, gp=None, data=None):
    """A simple emcee likelihood function for the Lyman-alpha forest using the
       simple linear model with only cosmological parameters.
       This neglects many important properties!"""
    assert gp is not None
    assert data is not None
    predicted = gp.predict(params)
    diff = predicted-data.pf
    return -np.dot(diff,np.dot(data.invcovar + np.identity(np.size(diff))/cov,diff))/2.0

def init_lnlike(nsamples, data=None):
    """Initialise the emulator for the likelihood function."""
#     param_names = ['bias_flux', 'ns', 'As']
    param_limits = np.array([[-2., 0], [0.5, 1.5], [1.5e-8, 8.0e-9]])
    params = latin_hypercube.get_hypercube_samples(param_limits, nsamples)
    data = gpemulator.SDSSData()
    #Get unique values
    flux_vectors = np.array([linear_theory.get_flux_power(bias_flux = pp[0], ns=pp[1], As=pp[2], zz=data.get_redshifts(), kf=data.get_kf()) for pp in params])
    gp = gpemulator.SkLearnGP(params=params, kf=data.kf, flux_vectors=flux_vectors)
    return gp, data

def build_fake_fluxes(nsamples):
    """Simple function using linear test case to build an emulator."""
    param_limits = np.array([[-1., 0], [0.9, 1.0], [1.5e-9, 3.0e-9]])
    params = latin_hypercube.get_hypercube_samples(param_limits, nsamples)
    data = gpemulator.SDSSData()
    flux_vectors = np.array([linear_theory.get_flux_power(bias_flux = pp[0], ns=pp[1], As=pp[2], kf=data.get_kf(), zz=data.get_redshifts()) for pp in params])
    gp = gpemulator.SkLearnGP(params=params, kf=data.kf, flux_vectors=flux_vectors)
    random_samples = latin_hypercube.get_random_samples(param_limits, nsamples//2)
    random_test_flux_vectors = np.array([linear_theory.get_flux_power(bias_flux = pp[0], ns=pp[1], As=pp[2], kf=data.get_kf(),zz=data.get_redshifts()) for pp in random_samples])
    diff_sk = gp.get_predict_error(random_samples, random_test_flux_vectors)
    return gp, diff_sk
