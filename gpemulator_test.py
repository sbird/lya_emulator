"""Test the gaussian process emulator classes using simple models
for the data."""

import numpy as np
import gpemulator

def test_emu_multiple():
    """Generate the simplest model possible,
    with an amplitude depending linearly on one parameter."""
    kf = np.array([ 0.00141,  0.00178,  0.00224,  0.00282])
    params = np.linspace(0.25,1.75,10).reshape(10,1)
    flux_vector = np.tile(kf*100.,(np.size(params),1))
    flux_vectors = ((flux_vector * params).T).T
    gp = gpemulator.SkLearnGP(params=params, kf=kf, flux_vectors=flux_vectors)
    predict,_ = gp.predict(np.array([0.5]).reshape(1,1))
    assert np.sum(np.abs(predict - 0.5 * flux_vector)/flux_vector) < 0.01
    return flux_vectors

def test_emu_single():
    """Generate the simplest model possible,
        with an amplitude depending linearly on
        one parameter and a single value."""
    kf = np.array([ 0.00141,])
    params = np.linspace(0.25,1.75,10).reshape(10,1)
    gp = gpemulator.SkLearnGP(params=params, kf=kf, flux_vectors=params)
    predict,_ = gp.predict(np.array([0.5]).reshape(1,1))
    assert np.abs(predict - 0.5) < 0.01
