"""Building a surrogate using a Gaussian Process."""
import numpy as np
from sklearn import gaussian_process
import george

import linear_theory
import latin_hypercube

param_names = ['bias_flux', 'ns', 'As']

def build_fake_fluxes(nsamples):
    """Simple function using linear test case to build an emulator."""
    param_limits = np.array([[-1., 0], [0.9, 1.0], [1.5e-9, 3.0e-9]])
    params = latin_hypercube.get_hypercube_samples(param_limits, nsamples)
    flux_vectors = np.array([linear_theory.get_flux_power(bias_flux = pp[0], ns=pp[1], As=pp[2]) for pp in params])
    gp = SkLearnGP(params, flux_vectors)
    random_samples = latin_hypercube.get_random_samples(param_limits, nsamples//2)
    random_test_flux_vectors = np.array([linear_theory.get_flux_power(bias_flux = pp[0], ns=pp[1], As=pp[2]) for pp in random_samples])
    diff_sk = gp.get_predict_error(random_samples, random_test_flux_vectors)
    return gp, diff_sk

class SkLearnGP(object):
    """An emulator using the one in Scikit-learn"""
    def __init__(self, params, flux_vectors):
        self.gp = gaussian_process.GaussianProcess()
        self.gp.fit(params, flux_vectors)

    def predict(self, params):
        """Get the predicted flux at a parameter value (or list of parameter values)."""
        flux_predict , cov = self.gp.predict(params, eval_MSE=True)
        return flux_predict, cov

    def get_predict_error(self, test_params, test_exact):
        """Get the difference between the predicted GP interpolation and some exactly computed test parameters."""
        predict, sigma = self.predict(test_params)
        #The transposes are because of numpy broadcasting rules only doing the last axis
        return ((test_exact - predict).T/np.sqrt(sigma)).T

class GeorgeGP(object):
    """An emulator using the george Gaussian Process code: NON-FUNCTIONAL because 1D prediction assumed."""
    def __init__(self, params, flux_vectors):
        kernel = george.kernels.ExpSquaredKernel(1.0, ndim=np.shape(params)[1])
        self.gp = george.GP(kernel)
        self.gp.compute(params)
        self.flux_vectors = flux_vectors

    def predict(self, params):
        """Get the predicted flux at a parameter value (or list of parameter values)."""
        #This does not work as George requires the predicted vector to be 1D!
        flux_predict , cov = self.gp.predict(self.flux_vectors, params)
        return flux_predict, cov
