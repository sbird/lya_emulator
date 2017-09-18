"""Module for computing the likelihood function for the forest emulator."""
import os
import os.path
import math
import numpy as np
import PyPolyChord
from PyPolyChord.settings import PolyChordSettings
import coarse_grid
import flux_power
import getdist.plots
import getdist.mcsamples
import lyman_data
import mean_flux as mflux
from latin_hypercube import map_from_unit_cube
import matplotlib
matplotlib.use('PDF')
import matplotlib.pyplot as plt

def load_chain(file_root):
    """Load a chain using getdist"""
    return getdist.mcsamples.loadMCSamples(file_root)

def make_plot(chain):
    """Make a plot of parameter posterior values"""
    g = getdist.plots.getSubplotPlotter()
    g.triangle_plot(chain)
    plt.show()

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
    def __init__(self, basedir, datadir, file_root="lymanalpha", mean_flux='s'):
        """Initialise the emulator by loading the flux power spectra from the simulations."""
        #Use the BOSS covariance matrix
        self.sdss = lyman_data.BOSSData()
        #'Data' now is a simulation
        myspec = flux_power.MySpectra(max_z=4.2)
        pps = myspec.get_snapshot_list(datadir)
        self.data_fluxpower = pps.get_power(kf=self.sdss.get_kf(),tau0_factors=0.95)
        assert np.size(self.data_fluxpower) % np.size(self.sdss.get_kf) == 0
        self.file_root = file_root
        #Get the emulator
        if mean_flux == 'c':
            mf = mflux.ConstMeanFlux(value = 0.95)
        elif mean_flux == 'f':
            mf = mflux.MeanFluxFactor()
        elif mean_flux == 's':
            mf = mflux.MeanFluxSlope()
        self.emulator = coarse_grid.KnotEmulator(basedir, kf=self.sdss.get_kf(), mf=mf)
        self.emulator.load()
        self.param_limits = self.emulator.get_param_limits(include_dense=True)
        self.ndim = np.shape(self.param_limits)[0]
        self.gpemu = self.emulator.get_emulator(max_z=4.2)
        #Make sure there is a save directory
        try:
            os.makedirs("chains/clusters")
        except FileExistsError:
            pass

    def prior(self, hypercube):
        """Maps the unit hypercube [0,1]^D to the units of the emulator."""
        #Sample only from the inner 90% of the hypercube,
        #to avoid edge effects from the emulator.
        return list(map_from_unit_cube(np.array(hypercube), self.param_limits))

    def likelihood(self, params):
        """A simple likelihood function for the Lyman-alpha forest.
        Assumes data is quadratic with a covariance matrix."""
        #Set parameter limits as the hull of the original emulator.
        predicted, std = self.gpemu.predict(np.array(params).reshape(1,-1))
        diff = predicted[0]-self.data_fluxpower
        nkf = len(self.sdss.get_kf())
        nz = int(len(diff)/nkf)
        #Likelihood using full covariance matrix
        chi2 = 0
        #Redshifts
        zout = self.sdss.get_redshifts()
        for bb in range(nz):
            diff_bin = diff[nkf*bb:nkf*(bb+1)]
            covar_bin = self.sdss.get_covar(zout[bb])
            icov_bin = np.linalg.inv(covar_bin + np.diag(std**2))
            chi2 += - np.dot(diff_bin, np.dot(icov_bin, diff_bin),)/2.
        assert 0 > chi2 > -2**31
        assert not np.isnan(chi2)
        #PolyChord requires a second argument for derived parameters
        return (chi2,[])

    def do_sampling(self):
        """Initialise and run PolyChord."""
        #Number of knots plus one cosmology plus one for mean flux.
        settings = PolyChordSettings(self.ndim, 0)
        settings.file_root = self.file_root
        settings.do_clustering = False
        settings.feedback = 3
        settings.read_resume = False
        #Make output
        result = PyPolyChord.run_polychord(self.likelihood, self.ndim, 0, settings, self.prior)
        #Save parameter names
        result.make_paramnames_files(self.emulator.print_pnames())
        #Save output
        #Check things are reasonable
        self.cur_result = result
        return result

    def new_parameter_limits(self, all_samples, coverage=99.9):
        """Find a square region which includes coverage of the parameters in each direction, for refinement."""
        assert 50 < coverage < 100
        #Use the marginalised distributions to find the square region.
        #If there are strong degeneracies this will be very inefficient.
        #We could rotate the parameters here,
        #but ideally we would do that before running the coarse grid anyway.
        new_par = np.percentile(all_samples,[100-coverage,coverage],axis=0)
        ndense = len(self.emulator.mf.dense_param_names)
        return new_par.T[ndense:,:]

    def refinement(self,nsamples,coverage=99):
        """Do the refinement step."""
        new_limits = self.new_parameter_limits(self.cur_result.posterior.samples,coverage=coverage)
        new_samples = self.emulator.build_params(nsamples=nsamples,limits=new_limits, use_existing=True)
        self.emulator.gen_simulations(nsamples=nsamples, samples=new_samples)

if __name__ == "__main__":
    like = LikelihoodClass(basedir=os.path.expanduser("~/data/Lya_Boss/hires_knots"), datadir=os.path.expanduser("~/data/Lya_Boss/hires_knots_test/AA1.1BB0.82CC0.82DD0.67heat_slope-0.42heat_amp0.58hub0.66/output/"))
    output = like.do_sampling()
