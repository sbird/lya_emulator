"""
This file contains functions which pick a set of samples from a parameter space
which will allow a Gaussian process to best interpolate the samples to new positions in parameter space.
Several schemes for this are possible.

We use rejection-sampled latin hypercubes.
"""

import numpy as np

def convert_to_simulation_parameters(input_parameters, omegamh2=0.1199, omegab=0.0483):
    """Convert latin hypercube parameters to input parameters for MP-Gadget"""
    omegam = omegamh2 / (input_parameters[4] ** 2)
    AsCLASS = input_parameters[1] * ((5.e-2 / (2. * np.pi / 8.)) ** (input_parameters[0] - 1.))
    return {'Omega0': omegam, 'OmegaLambda': 1. - omegam, 'OmegaBaryon': omegab, 'HubbleParam': input_parameters[4], 'PrimordialIndex': input_parameters[0], 'PrimordialAmp': AsCLASS}

def get_hypercube_samples(param_limits, nsamples, prior_points = None):
    """This function is the main wrapper. Given limits on a set of
    parameters (and optionally some prior points), it will generate a hypercube design."""
    ndim,nlims = np.shape(param_limits)
    assert nlims == 2
    if prior_points is not None:
        if len(prior_points) == 0:
            prior_points = None
        else:
            prior_points = np.array([map_to_unit_cube(pp, param_limits) for pp in prior_points])
    (sample_points, _) = maximinlhs(ndim, nsamples, prior_points=prior_points)
    remapped = np.array([map_from_unit_cube(pp, param_limits) for pp in sample_points])
    assert np.shape(remapped) == (nsamples, ndim)
    return remapped

def get_random_samples(param_limits, nsamples):
    """This function randomly samples the parameters within the allowed space.
    Mostly for testing purposes, as well as evaluating how much better our hypercube does."""
    ndim,nlims = np.shape(param_limits)
    assert nlims == 2
    sample_points =  np.random.random_sample(ndim*nsamples).reshape(nsamples, ndim)
    remapped = np.array([map_from_unit_cube(pp, param_limits) for pp in sample_points])
    assert np.shape(remapped) == (nsamples, ndim)
    return remapped

def default_metric_func(lhs):
    """Default metric function for the maximinlhs, below.
    This is the sum of the Euclidean distances between each point and the closest other point."""
    #First find minimum distance between every two points
    nsamples, _ = np.shape(lhs)
    #This is an array of the minimum squared distance between every two points.
    #We only compute minima for the upper triangle, because of symmetry.
    minn = np.array([np.min(np.sum((lhs[j+1:,:] - lhs[j,:])**2,axis=1)) for j in range(nsamples-1)])
    assert np.shape(minn) == (nsamples - 1,)
    return np.sqrt(np.sum(minn))

def maximinlhs(n, samples, prior_points = None, metric_func = None, maxlhs = 10000):
    """Generate multiple latin hypercubes and pick the one that maximises the metric function.
    Arguments:
    n: dimensionality of the cube to sample [0-1]^n
    samples: total number of samples.
    prior_points: List of previously evaluated points. If None, totally repopulate the space.
    metric_func: Function with which to judge the 'goodness' of the generated latin hypercube.
    Should be a scalar function of one hypercube sample set.
    maxlhs: Maximum number of latin hypercube to generate in total.
    Note convergence is pretty slow at the moment."""
    #Use the default metric if none is specified.
    if metric_func is None:
        metric_func = default_metric_func
    #Minimal metric is zero.
    metric = -1
    group = 1000
    for _ in range(maxlhs//group):
        new = [lhscentered(n, samples, prior_points = prior_points) for _ in range(group)]
        new_metric = [metric_func(nn) for nn in new]
        best = np.argmax(new_metric)
        if new_metric[best] > metric:
            metric = new_metric[best]
            current = new[best]
    return current,metric

def remove_single_parameter(center, prior_points):
    """Remove all values within cells covered by prior samples for a particular parameter.
    Arguments:
    center contains the central values of each (evenly spaced) bin.
    prior_points contains the values of each already computed point."""
    #Find which bins the previously computed points are in
    already_taken = np.array([np.argmin(np.abs(center - pp)) for pp in prior_points])
    #Find the indices of points not in already_taken
    not_taken = np.setdiff1d(range(np.size(center)), already_taken)
    new_center = center[not_taken]
    assert np.size(new_center) == np.size(center) - np.size(prior_points)
    return new_center,not_taken

def lhscentered(n, samples, prior_points = None):
    """
    Generate a latin hypercube design where all samples are
    centered on their respective cells. Can specify an already
    existing set of points using prior_points; these must also
    be a latin hypercube on a smaller sample, but need not be centered.
    """
    #Set up empty prior points if needed.
    if prior_points is None:
        prior_points = np.empty([0,n])

    npriors = np.shape(prior_points)[0]
    # Generate the intervals
    cut = np.linspace(0, 1, samples + 1)

    # Fill points uniformly in each interval
    # Number of stratified layers used is samples desired + prior_points.
    a = cut[:samples]
    b = cut[1:samples + 1]
    #Get list of central values
    _center = (a + b)/2
    # Choose a permutation so each sample is in one bin for each factor.
    H = np.zeros((samples, n))
    for j in range(n):
        #Remove all values within cells covered by prior samples for this parameter.
        #The prior samples must also be a latin hypercube!
        if npriors > 0:
            H[:,j] = _center
            new_center, not_taken = remove_single_parameter(_center, prior_points[:,j])
            H[not_taken, j] = np.random.permutation(new_center)
        else:
            H[:, j] = np.random.permutation(_center)
    assert np.shape(H) == (samples, n)
    return H

def map_from_unit_cube(param_vec, param_limits):
    """
    Map a parameter vector from the unit cube to the original dimensions of the space.
    Arguments:
    param_vec - the vector of parameters to map. Should all be [0,1]
    param_limits - the maximal limits of the parameters to choose.
    """
    assert (np.size(param_vec),2) == np.shape(param_limits)
    assert np.all((param_vec >= 0)*(param_vec <= 1))
    assert np.all(param_limits[:,0] <= param_limits[:,1])
    new_params = param_limits[:,0] + param_vec*(param_limits[:,1] - param_limits[:,0])
    assert np.all(new_params <= param_limits[:,1])
    assert np.all(new_params >= param_limits[:,0])
    return new_params

def map_to_unit_cube(param_vec, param_limits):
    """
    Map a parameter vector to the unit cube from the original dimensions of the space.
    Arguments:
    param_vec - the vector of parameters to map.
    param_limits - the limits of the allowed parameters.
    Returns:
    vector of parameters, all in [0,1].
    """
    assert (np.size(param_vec),2) == np.shape(param_limits)
    assert np.all(param_vec-1e-16 <= param_limits[:,1])
    assert np.all(param_vec+1e-16 >= param_limits[:,0])
    ii = np.where(param_vec > param_limits[:,1])
    param_vec[ii] = param_limits[ii,1]
    ii = np.where(param_vec < param_limits[:,0])
    param_vec[ii] = param_limits[ii,0]
    assert np.all(param_limits[:,0] <= param_limits[:,1])
    new_params = (param_vec-param_limits[:,0])/(param_limits[:,1] - param_limits[:,0])
    assert np.all((new_params >= 0)*(new_params <= 1))
    return new_params

def map_to_unit_cube_list(param_vec_list, param_limits):
    """Map multiple parameter vectors to the unit cube"""
    return np.array([map_to_unit_cube(param_vec, param_limits) for param_vec in param_vec_list])

def map_from_unit_cube_list(param_vec_list, param_limits):
    """Map multiple parameter vectors back from the unit cube"""
    return np.array([map_from_unit_cube(param_vec, param_limits) for param_vec in param_vec_list])
