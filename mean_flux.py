"""Module for holding different mean flux models"""

import numpy as np

class ConstMeanFlux(object):
    """Object which implements different mean flux models. This model fixes the mean flux to a constant value.
    """
    def __init__(self, value = 0.95):
        self.value = value

    def get_t0(self):
        """Get change in mean optical depth from parameter values"""
        return self.value

    def get_params(self):
        """Returns a list of parameters where the mean flux is evaluated."""
        return None

    def get_limits(self):
        """Get limits on the dense parameters"""
        return None

class MeanFluxFactor(ConstMeanFlux):
    """Object which implements different mean flux models. This model parametrises
    uncertainty in the mean flux with a simple scaling factor.
    """
    def __init__(self, dense_samples = 5):
        #Limits on factors to multiply the thermal history by.
        #Mean flux is known to about 10% from SDSS, so we don't need a big range.
        self.dense_param_limits = np.array([[0.7,1.3],])
        self.dense_samples = dense_samples

    def get_t0(self):
        return self.get_params()

    def get_params(self):
        """Returns a list of parameters where the mean flux is evaluated."""
        #Number of dense parameters
        ndense = np.shape(self.dense_param_limits)[0]
        #This grid will hold the expanded grid of parameters: dense parameters are on the end.
        #Initially make it NaN as a poisoning technique.
        pvals = np.nan*np.zeros((self.dense_samples, ndense))
        for dd in range(ndense):
            #Build grid of mean fluxes
            dlim = self.dense_param_limits[dd]
            dense = np.linspace(dlim[0], dlim[1], self.dense_samples)
            pvals[:,dd] = dense
        assert not np.any(np.isnan(pvals))
        return pvals

    def get_limits(self):
        """Get limits on the dense parameters"""
        return self.dense_param_limits
