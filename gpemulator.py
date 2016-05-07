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
    flux_vectors = [linear_theory.get_flux_power(bias_flux = pp[0], ns=pp[1], As=pp[2]) for pp in params]
    gp_sk = SkLearnGP(params, flux_vectors)
    random_samples = latin_hypercube.get_random_samples(param_limits, nsamples/2.)
    random_test_flux_vectors = [linear_theory.get_flux_power(bias_flux = pp[0], ns=pp[1], As=pp[2]) for pp in random_samples]
    predicted_flux_sk = gp_sk.predict(random_samples)
    gp_george = GeorgeGP(params, flux_vectors)
    predicted_flux_george = gp_george.predict(random_samples)
    diff_george = random_test_flux_vectors - predicted_flux_george
    diff_sk = random_test_flux_vectors - predicted_flux_sk
    return diff_sk, diff_george


class SkLearnGP(object):
    """An emulator using the one in Scikit-learn"""
    def __init__(self, params, flux_vectors):
        self.gp = gaussian_process.GaussianProcess()
        self.gp.fit(params, flux_vectors)

    def predict(self, params):
        """Get the predicted flux at a parameter value (or list of parameter values)."""
        flux_predict , cov = self.gp.predict(params, eval_MSE=True)
        return flux_predict, cov

class GeorgeGP(object):
    """An emulator using the george Gaussian Process code."""
    def __init__(self, params, flux_vectors):
        kernel = george.kernels.ExpSquaredKernel(1.0)
        self.gp = george.GP(kernel)
        self.gp.compute(params)
        self.flux_vectors = flux_vectors

    def predict(self, params):
        """Get the predicted flux at a parameter value (or list of parameter values)."""
        flux_predict , cov = self.gp.predict(self.flux_vectors, params)
        return flux_predict, cov
