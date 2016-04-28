"""Tests for the latin hypercube module."""

import numpy as np
import latin_hypercube

def test_from_and_to_unit_cube():
    """Check we can map to and from the unit cube correctly."""
    param_limits = np.array([-1*np.ones(5), 4*np.ones(5)]).T
    param_vec = np.random.random_sample(5)
    new_params = latin_hypercube.map_to_unit_cube(param_vec,param_limits)
    assert np.all((0 <= new_params)*(new_params <= 1))
    new_new_params = latin_hypercube.map_from_unit_cube(new_params,param_limits)
    assert np.all(param_vec - new_new_params <= 1e-12)

def test_remove_single_parameter():
    """Check we can correctly find those elements of an array not in an already sampled array."""
    prior_points = np.random.permutation(np.linspace(0,1,7))
    center = np.linspace(0,1,18)
    center = (center[:17] + center[1:])/2
    new_center = latin_hypercube.remove_single_parameter(center, prior_points)
    #The right size
    assert np.size(new_center) == np.size(center) - np.size(prior_points)
    #All points in the new set were also in the old set
    assert np.size(np.setdiff1d(new_center,center)) == 0
    #Some of the priors overlap with the centers
    minns = np.array([np.min(np.abs(center - pp)) for pp in prior_points])
    assert np.any(minns < 1/17./2.)
    #None of the new points are in the same bin as the old points
    minns = np.array([np.min(np.abs(new_center - pp)) for pp in prior_points])
    assert np.all(minns > 1/17./2.)

def _gen_hyp_check(hyper):
    """Helper function to test a specific array has hypercube design."""
    nsamples, ndims = np.shape(hyper)
    for i in range(ndims):
        hist = np.histogram(hyper[:,i],bins=nsamples)
        assert np.all(hist[0] == 1)

def test_lhscentered():
    """Check that we can produce a latin hypercube of the desired properties,
    which are even spacing when projected into any direction."""
    #Test a few designs
    x1 = latin_hypercube.lhscentered(2,5)
    _gen_hyp_check(x1)
    #Check that if we refine we are still a hypercube
    x2 = latin_hypercube.lhscentered(2,15,prior_points = x1)
    _gen_hyp_check(x2)
    #Check that we can do higher dimensional hypercubes
    x3 = latin_hypercube.lhscentered(7,15)
    _gen_hyp_check(x2)
        #Check that we can do higher dimensional hypercubes
    x4 = latin_hypercube.lhscentered(7,100)
    _gen_hyp_check(x4)

def test_default_metric():
    """Test the default metric function is returning something reasonable."""
    #First generate a bad hypercube.
    a = np.linspace(0,1,5)
    cc = (a[1:] + a[:-1])/2.
    #This has all entries along the diagonal!
    lhs = np.vstack([cc, cc]).T
    #Shuffle the second parameter, which should make it better.
    lhs2 = np.vstack([cc, np.roll(cc,2)]).T
    #Check that this better hypercube is actually better.
    assert latin_hypercube._default_metric_func(lhs) < latin_hypercube._default_metric_func(lhs2)
    #Check that it doesn't matter which order the parameters are in.
    lhs3 = np.vstack([np.roll(cc,2), cc]).T
    assert latin_hypercube._default_metric_func(lhs3) == latin_hypercube._default_metric_func(lhs2)

def test_maximin():
    """Test that the maximin finder is working."""
    xmax = latin_hypercube.maximinlhs(2,8)
    #Occasionally this may fail purely because we didn't converge.
    #Hopefully this is rare.
    assert xmax[1] > 1.5
