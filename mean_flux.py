"""Module for holding different mean flux models"""

import numpy as np

def obs_mean_tau(redshift, amp=0, slope=0):
    """The mean flux from 0711.1862: is (0.0023±0.0007) (1+z)^(3.65±0.21)
    Note we constrain this much better from the SDSS data itself:
    this is a weak prior"""
    return (0.0023+amp)*3**3.65*((1.0+redshift)/3.)**(3.65+slope)

class ConstMeanFlux(object):
    """Object which implements different mean flux models. This model fixes the mean flux to a constant value.
    """
    def __init__(self, value = 0.95):
        self.value = value
        self.dense_param_names = {}

    def get_t0(self, zzs):
        """Get change in mean optical depth from parameter values"""
        return self.value * obs_mean_tau(zzs)

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
    def __init__(self, dense_samples = 5, dense_limits = None):
        #Limits on factors to multiply the thermal history by.
        #Mean flux is known to about 10% from SDSS, so we don't need a big range.
        if dense_limits is None:
            self.dense_param_limits = np.array([[0.7,1.3]])
        else:
            self.dense_param_limits = dense_limits
        self.dense_samples = dense_samples
        self.dense_param_names = { 'tau0': 0, }

    def get_t0(self, zzs):
        """Get the mean flux as a function of redshift for all parameters."""
        return np.array([t0 * obs_mean_tau(zzs) for t0 in self.get_params()])

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

class MeanFluxSlope(MeanFluxFactor):
    """Object which implements different mean flux models. This model parametrises
    uncertainty in the mean flux with a scaling factor and a slope.
    """
    def __init__(self, dense_samples = 5):
        #Limits on factors to multiply the thermal history by.
        #Mean flux is known to about 10% from SDSS, so we don't need a big range.
        #Convert Kim constraints to z=2
        maxt = 7 * 3**0.23 + 23 * (3**0.23 - 1)
        mint = -7 * 3**-0.23 + 23 * (3**-0.23 - 1)
        super().__init__(dense_samples = dense_samples**2, dense_limits = np.array([[mint,maxt],[-0.23,0.23]]))
        self.dense_param_names = { 'tau0': 0 , 'dtau0': 1}

    def get_t0(self, zzs):
        """Get the mean flux as a function of redshift for all parameters."""
        t0fac = self.get_params()
        return np.array([obs_mean_tau(zzs, amp = t0[0], slope=t0[0]) for t0 in t0fac])
