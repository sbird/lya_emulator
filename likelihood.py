"""Module for computing the likelihood function for the forest emulator."""
import os
import os.path
import math
import numpy as np
from latin_hypercube import map_from_unit_cube
#Import PolyChord
import PolyChord.PyPolyChord.PyPolyChord as PolyChord
from PolyChord.PyPolyChord.priors import UniformPrior
from PolyChord.PyPolyChord.settings import PolyChordSettings
import coarse_grid
import flux_power
import getdist.plots
import lyman_data
import matplotlib
matplotlib.use('PDF')
import matplotlib.pyplot as plt


def _siIIIcorr(kf):
    """For precomputing the shape of the SiIII correlation"""
    #Compute bin boundaries in logspace.
    kmids = np.zeros(np.size(kf)+1)
    kmids[1:-1] = np.exp((np.log(kf[1:])+np.log(kf[:-1]))/2.)
    #arbitrary final point
    kmids[-1] = 2*math.pi/2271 + kmids[-2]
    # This is the average of cos(2271k) across the k interval in the bin
    siform = np.zeros_like(kf)
    siform = (np.sin(2271*kmids[1:])-np.sin(2271*kmids[:-1]))/(kmids[1:]-kmids[:-1])/2271.
    #Correction for the zeroth bin, because the integral is oscillatory there.
    siform[0] = np.cos(2271*kf[0])
    return siform

def SiIIIcorr(fSiIII, tau_eff, kf):
    """The correction for SiIII contamination, as per McDonald."""
    assert tau_eff > 0
    aa = fSiIII/(1-np.exp(-tau_eff))
    return 1 + aa**2 + 2 * aa * _siIIIcorr(kf)

class LikelihoodClass(object):
    """Class to contain likelihood computations."""
    def __init__(self, basedir, datadir, file_root="lymanalpha"):
        """Initialise the emulator by loading the flux power spectra from the simulations."""
        #Parameter names
        sdss = lyman_data.SDSSData()
        myspec = flux_power.MySpectra(max_z=4.2)
        pps = myspec.get_snapshot_list(datadir)
        self.data_fluxpower = pps.get_power(kf=sdss.get_kf(),tau0_factor=0.95)[0]
        self.file_root = file_root
        #Use the SDSS covariance matrix
        self.data_covar = sdss.get_covar()
        self.emulator = coarse_grid.KnotEmulator(basedir)
        self.emulator.load()
        self.param_limits = self.emulator.get_param_limits()
        self.ndim = np.shape(self.param_limits)[0]
        self.gpemu = self.emulator.get_emulator(max_z=4.2)
        #Make sure there is a save directory, and we can write to it.
        if not os.access("chains/clusters", os.W_OK):
            os.makedirs("chains/clusters")

    def prior(self, hypercube):
        """ Uniform prior from [-1,1]^D. """
        theta = [0.0] * self.ndim
        for i, x in enumerate(hypercube):
            theta[i] = UniformPrior(0, 1)(x)
        return theta

    def likelihood(self, params):
        """A simple likelihood function for the Lyman-alpha forest.
        Assumes data is quadratic with a covariance matrix."""
        #Set parameter limits as the hull of the original emulator.
        new_params = map_from_unit_cube(np.array(params), self.param_limits)
        predicted, std = self.gpemu.predict(new_params.reshape(1,-1), tau0_factor=None)
        diff = predicted[0]-self.data_fluxpower
        #Ideally I would find a way to avoid this inversion
        icov = np.linalg.inv(self.data_covar + np.diag(std**2))
        #PolyChord requires a second argument for derived parameters
        return (-np.dot(diff,np.dot(icov,diff))/2.0,[])

    def do_sampling(self):
        """Initialise and run PolyChord."""
        #Number of knots plus one cosmology plus one for mean flux.
        settings = PolyChordSettings(self.ndim, 0)
        settings.file_root = self.file_root
        settings.do_clustering = False
        #Make output
        result = PolyChord.run_polychord(self.likelihood, self.ndim, 0, settings, self.prior)
        #Save output
        result.make_paramnames_files(list(self.emulator.param_names.keys()))
        #Check things are reasonable
        self.cur_result = result
        return result

    def new_parameter_limits(self, all_samples, coverage=99):
        """Find a square region which includes coverage of the parameters in each direction, for refinement."""
        assert 50 < coverage < 100
        #Use the marginalised distributions to find the square region.
        #If there are strong degeneracies this will be very inefficient.
        #We could rotate the parameters here,
        #but ideally we would do that before running the coarse grid anyway.
        new_par = np.percentile(all_samples,[100-coverage,coverage],axis=0)
        nsparse = np.shape(self.emulator.get_param_limits(include_dense=False))[0]
        return new_par.T[:nsparse,:]

    def refinement(self,nsamples,coverage=99):
        """Do the refinement step."""
        new_limits = self.new_parameter_limits(self.cur_result.posterior,coverage=coverage)
        new_samples = self.emulator.build_params(nsamples=nsamples,limits=new_limits, use_existing=True)
        self.emulator.gen_simulations(nsamples=nsamples, samples=new_samples)

    def make_plot(self, chain):
        """Make a plot of parameter posterior values"""
        posterior = chain.posterior
        g = getdist.plots.getSubplotPlotter()
        g.triangle_plot(posterior, filled=True)
        plt.show()

if __name__ == "__main__":
    like = LikelihoodClass(os.path.expanduser("~/data/Lya_Boss/hires_knots"), os.path.expanduser("~/data/Lya_Boss/hires_knots_test/AA1.1BB0.82CC0.82DD0.67heat_slope-0.42heat_amp0.58hub0.66/output/"))
    output = like.do_sampling()
