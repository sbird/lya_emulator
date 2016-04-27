"""
This file contains functions pick a set of samples from a parameter space
which, when evaluated, will allow us to best reconstruct the likelihood.
Several schemes for this are possible.
"""

from scipy.stats.distributions import norm
import numpy as np
from doe_lhs import lhs

def lhscentered(n, samples, prior_points = None):
    """
    Generate a latin hypercube design where all samples are
    centered on their respective cells. Can specify an already
    existing set of points using prior_points; these must also
    be a latin hypercube (on a smaller sample), but need not be centered.

    To make sure the grid points line up, the cardinality of the new LHS
    must be an ODD multiple of the cardinality of the old.
    ie:  samples == a * len(prior_points)  where a is an odd integer.

    So each refinement triples the number of points.
    """
    #Set up empty prior points if needed.
    if prior_points is None:
        prior_points = np.empty([0,n])

    npriors = np.shape(prior_points)[0]
    #Enforce that we are subsampling the earlier distribution, not supersampling it.
    assert samples % npriors == 0 and samples % (2*npriors) != 0
    new_samples = samples - npriors
    # Generate the intervals
    cut = np.linspace(0, 1, samples + 1)

    # Fill points uniformly in each interval
    # Number of stratified layers used is samples desired + prior_points.
    a = cut[:samples]
    b = cut[1:samples + 1]
    #Get list of central values
    _center = (a + b)/2
    #Multiply by this factor so that everything an integer
    #and thus equality is easy to establish below
    scale = _center[0]
    _center /= scale

    # Choose a permutation so each sample is in one bin for each factor.
    H = np.zeros((new_samples, n))
    for j in range(n):
        #Remove all values within cells covered by prior samples for this parameter.
        #The prior samples must also be a latin hypercube!
        already_taken = prior_points[:,j]/scale
        #Use only those points not already in prior_points for this parameter.
        new_center = np.setdiff1d(_center.astype(int), already_taken.astype(int))*scale
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
    assert((np.size(param_vec),2) == np.shape(max_params))
    assert(np.all(0 <= param_vec <= 1))
    assert(np.all(param_limits[:,0] < param_limits[:,1]))
    new_params = param_limits[:,0] + param_vec*(param_limits[:,1] - param_limits[:,0])
    assert(np.all(new_params < param_limits[:,1]))
    assert(np.all(new_params > param_limits[:,0]))
    return new_params

def weight_cube(sample, means, sigmas):
    """
    Here we want to weight each dimension in the cube by its cumulative distribution function. So for parameter p in [p1, p2] we sample from p ~ CDF^-1(p1, p2)
    TODO: How to do this when we only approximately know the likelihood?

    """
    #This samples from the inverse CDF
    return norm(loc=means, scale=sigmas).ppf(lhd)

def sample_unit_cube(ndims, prior_points):
    """Produce an even sampling of the unit cube.
    Prior points should be included by adding a series of points
    before the LHS generated ones."""


def plot_points_hypercube(lhs_xval, lhs_yval):
    """Make a plot of the hypercube output points positioned on a regular grid"""
    ndivision = np.size(lhs_xval)
    assert ndivision == np.size(lhs_yval)
    xticks = np.linspace(0,1,ndivision+1)
    plt.scatter(lhs_xval, lhs_yval, marker='o', s=300)
    plt.grid(b=True, which='major')
    plt.xticks(xticks)
    plt.yticks(xticks)
    plt.xlim(0,1)
    plt.ylim(0,1)
