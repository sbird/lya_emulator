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
    flux_vectors = flux_vector * params
    gp = gpemulator.SkLearnGP(params=params, kf=kf, flux_vectors=flux_vectors)
    predict,_ = gp.predict(np.array([0.5]).reshape(1,1))
    print(predict-0.5*kf*100)
    assert np.sum(np.abs(predict - 0.5 * kf*100)/predict) < 1e-4
    return flux_vectors

def test_emu_single():
    """Generate the simplest model possible,
        with an amplitude depending linearly on
        one parameter and a single value."""
    kf = np.array([ 0.00141,])
    params = np.linspace(0.25,1.75,10).reshape(10,1)
    gp = gpemulator.SkLearnGP(params=params, kf=kf, flux_vectors=params)
    predict,_ = gp.predict(np.array([0.5]).reshape(1,1))
    assert np.abs(predict - 0.5) < 1e-4

def test_emu_multi_param():
    """Simplest model possible with multiple parameters.
    One is linear multiplication, one is a squared term."""
    kf = np.array([ 0.00141,  0.00178,  0.00224,  0.00282])
    p1 = np.linspace(0.25,1.75,10)
    p2 = np.linspace(0.1,1.,10)
    p2 = np.tile(p2,10)
    p1 = np.repeat(p1,10)
    params = np.vstack([p1.T,p2.T]).T
    flux_vectors = np.array([kf*100*(pp[0] + pp[1]**2) for pp in params])
    gp = gpemulator.SkLearnGP(params=params, kf=kf, flux_vectors=flux_vectors)
    predict,_ = gp.predict(np.array([0.5,0.288]).reshape(1,-1))
    assert np.sum(np.abs(predict - (0.5+0.288**2) * 100*kf)/predict) < 1e-4
