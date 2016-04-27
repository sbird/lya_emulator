"""
This file contains functions pick a set of samples from a parameter space
which, when evaluated, will allow us to best reconstruct the likelihood.
Several schemes for this are possible.
"""

from scipy.stats.distributions import norm
import numpy as np

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
    return new_center

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
    #Enforce that we are subsampling the earlier distribution, not supersampling it.
    assert samples > npriors
    new_samples = samples - npriors
    # Generate the intervals
    cut = np.linspace(0, 1, samples + 1)

    # Fill points uniformly in each interval
    # Number of stratified layers used is samples desired + prior_points.
    a = cut[:samples]
    b = cut[1:samples + 1]
    #Get list of central values
    _center = (a + b)/2
    # Choose a permutation so each sample is in one bin for each factor.
    H = np.zeros((new_samples, n))
    for j in range(n):
        #Remove all values within cells covered by prior samples for this parameter.
        #The prior samples must also be a latin hypercube!
        new_center = remove_single_parameter(_center, prior_points[:,j])
        H[:, j] = np.random.permutation(new_center)
    Hp = np.vstack((prior_points, H))
    assert np.shape(Hp) == (samples, n)
    return Hp

def map_from_unit_cube(param_vec, param_limits):
    """
    Map a parameter vector from the unit cube to the original dimensions of the space.
    Arguments:
    param_vec - the vector of parameters to map. Should all be [0,1]
    param_limits - the maximal limits of the parameters to choose.
    """
    assert (np.size(param_vec),2) == np.shape(param_limits)
    assert np.all((0 <= param_vec)*(param_vec <= 1))
    assert np.all(param_limits[:,0] < param_limits[:,1])
    new_params = param_limits[:,0] + param_vec*(param_limits[:,1] - param_limits[:,0])
    assert np.all(new_params < param_limits[:,1])
    assert np.all(new_params > param_limits[:,0])
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
    assert np.all(param_vec < param_limits[:,1])
    assert np.all(param_vec > param_limits[:,0])
    assert np.all(param_limits[:,0] < param_limits[:,1])
    new_params = (param_vec-param_limits[:,0])/(param_limits[:,1] - param_limits[:,0])
    assert np.all((0 <= new_params)*(new_params <= 1))
    return new_params

def weight_cube(sample, means, sigmas):
    """
    Here we want to weight each dimension in the cube by its cumulative distribution function. So for parameter p in [p1, p2] we sample from p ~ CDF^-1(p1, p2)
    TODO: How to do this when we only approximately know the likelihood?

    """
    #This samples from the inverse CDF
    return norm(loc=means, scale=sigmas).ppf(sample)

def sample_unit_cube(ndims, prior_points):
    """Produce an even sampling of the unit cube.
    Prior points should be included by adding a series of points
    before the LHS generated ones."""

#Wrap the plotting scripts in a try block so it succeeds on X-less clusters
try:
    import matplotlib.pyplot as plt


    def plot_points_hypercube(lhs_xval, lhs_yval, color="blue"):
        """Make a plot of the hypercube output points positioned on a regular grid"""
        ndivision = np.size(lhs_xval)
        assert ndivision == np.size(lhs_yval)
        xticks = np.linspace(0,1,ndivision+1)
        plt.scatter(lhs_xval, lhs_yval, marker='o', s=300, color=color)
        plt.grid(b=True, which='major')
        plt.xticks(xticks)
        plt.yticks(xticks)
        plt.xlim(0,1)
        plt.ylim(0,1)

except ImportError:
    pass
