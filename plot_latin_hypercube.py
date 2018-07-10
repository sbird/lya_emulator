"""Small script to plot latin hypercubes. Separate so it works without X forwarding"""
import matplotlib.pyplot as plt
import numpy as np

from latin_hypercube import *

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

def make_plot_initial_parameter_samples(savefile):
    """Make a plot of the hypercube samples (rescaled to the prior (hyper)volume)"""
    n_parameter_samples = 36
    parameter_prior_limits = np.array([[0.9, 1.1], [-0.1, 0.1]]) #n_parameters x 2
    initial_parameter_samples = get_hypercube_samples(parameter_prior_limits, n_parameter_samples)

    initial_parameter_samples_Sobol_sequence = get_hypercube_samples_Sobol_sequence(parameter_prior_limits, n_parameter_samples)

    figure, axes = plt.subplots(nrows=2, ncols=1, figsize=(6.4 * 2., 10.))
    axes[0].scatter(initial_parameter_samples[:, 0], initial_parameter_samples[:, 1], label=r'Rejection-sampled Latin hypercube')
    axes[0].set_xlabel(r'HeliumHeatAmp')
    axes[0].set_ylabel(r'HeliumHeatExp')
    axes[0].legend(frameon=False)
    axes[0].set_xlim(parameter_prior_limits[0])
    axes[0].set_ylim(parameter_prior_limits[1])

    axes[1].scatter(initial_parameter_samples_Sobol_sequence[:, 0], initial_parameter_samples_Sobol_sequence[:, 1],
                    label=r'Sobol sequence')
    axes[1].set_xlabel(r'HeliumHeatAmp')
    axes[1].set_ylabel(r'HeliumHeatExp')
    axes[1].legend(frameon=False)
    axes[1].set_xlim(parameter_prior_limits[0])
    axes[1].set_ylim(parameter_prior_limits[1])

    plt.savefig(savefile)

    return initial_parameter_samples, initial_parameter_samples_Sobol_sequence