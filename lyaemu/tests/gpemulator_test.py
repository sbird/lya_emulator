"""Test the gaussian process emulator classes using simple models
for the data."""

import numpy as np
from lyaemu import gpemulator

class Power(object):
    """Mock power object"""
    def __init__(self,params):
        self.params = params

    def get_power(self, *, kf, mean_fluxes=None):
        """Get the flux power spectrum."""
        flux_vector = kf*100.
        if mean_fluxes is not None:
            flux_vector *= mean_fluxes
        return flux_vector * self.params

def test_emu_multiple():
    """Generate the simplest model possible,
    with an amplitude depending linearly on one parameter."""
    kf = np.array([ 0.00141,  0.00178,  0.00224,  0.00282])
    params = np.reshape(np.linspace(0.25,1.75,10), (10,1))
    powers = np.array([Power(par).get_power(kf=kf) for par in params])
    plimits = np.array((0.25,1.75),ndmin=2)
    gp = gpemulator.MultiBinGP(params = params, kf = kf, powers = powers, param_limits = plimits)
    predict,_ = gp.predict(np.reshape(np.array([0.5]), (1,1)))
    assert np.sum(np.abs(predict - 0.5 * kf*100)/predict) < 1e-4

def test_emu_single():
    """Generate the simplest model possible,
        with an amplitude depending linearly on
        one parameter and a single value."""
    kf = np.array([ 0.00141])
    params = np.reshape(np.linspace(0.25,1.75,10),(10,1))
    plimits = np.array((0.25,1.75),ndmin=2)
    powers = np.array([Power(par).get_power(kf=kf) for par in params])
    gp = gpemulator.MultiBinGP(params=params, kf=kf, powers = powers, param_limits = plimits)
    predict, _ = gp.predict(np.reshape(np.array([0.5]), (1,1)))
    assert np.abs(predict/kf/100 - 0.5) < 1e-4

class MultiPower(object):
    """Mock power object"""
    def __init__(self,params):
        self.params = params

    def get_power(self, *, kf, mean_fluxes=None):
        """Get the flux power spectrum."""
        flux_vector = kf*100*(self.params[0] + self.params[1]**2)
        if mean_fluxes is not None:
            flux_vector *= mean_fluxes
        return flux_vector

def test_emu_multi_param():
    """Simplest model possible with multiple parameters.
    One is linear multiplication, one is a squared term."""
    kf = np.array([ 0.00141,  0.00178,  0.00224,  0.00282])
    p1 = np.linspace(0.25,1.75,10)
    p2 = np.linspace(0.1,1.,10)
    p2 = np.tile(p2,10)
    p1 = np.repeat(p1,10)
    params = np.vstack([p1.T,p2.T]).T
    powers = np.array([MultiPower(par).get_power(kf=kf) for par in params])
    plimits = np.array(((0.25,1.75),(0.1,1)))
    gp = gpemulator.MultiBinGP(params=params, kf=kf, powers = powers, param_limits = plimits)
    predict,_ = gp.predict(np.reshape(np.array([0.5,0.288]),(1,-1)))
    assert np.max(np.abs(predict - (0.5+0.288**2) * 100*kf)/predict) < 1e-4
