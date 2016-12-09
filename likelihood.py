"""Module for computing the likelihood function for the forest emulator."""
import os.path
import numpy as np
import emcee
import coarse_grid
import flux_power
import gpemulator

class LikelihoodClass(object):
    """Class to contain likelihood computations."""
    def __init__(self, basedir, datadir, mean_flux=True):
        """Initialise the emulator by loading the flux power spectra from the simulations."""
        #Parameter names
        sdss = gpemulator.SDSSData()
        myspec = flux_power.MySpectra(max_z=4.2)
        self.data_fluxpower = myspec.get_flux_power(datadir,sdss.get_kf(),tau0_factors=[1.,])[0]
        #Use the SDSS covariance matrix
        self.data_icovar = sdss.get_icovar()
        params = coarse_grid.Emulator(basedir)
        params.load()
        self.gpemu = params.get_emulator(max_z=4.2, mean_flux=mean_flux)
        self.param_limits = params.get_param_limits(include_dense=mean_flux)
        #Initialise sampler and make a few samples.
        self.sampler = self.init_emcee()

    def lnlike_linear(self, params):
        """A simple emcee likelihood function for the Lyman-alpha forest."""
        #Set parameter limits as the hull of the original emulator.
        if np.any(params < self.param_limits[:,0]) or np.any(params > self.param_limits[:,1]):
            return -np.inf
        predicted,_ = self.gpemu.predict(params.reshape(1,-1))
        diff = predicted[0]-self.data_fluxpower
        #TODO: emuerr should contain an estimate of the 'theory error' on the power spectrum.
        emuerr = 1e99
        return -np.dot(diff,np.dot(self.data_icovar+np.identity(np.size(diff))/emuerr,diff))/2.0

    def init_emcee(self,nwalkers=100, burnin=1000, nsamples = 10000):
        """Initialise and run emcee."""
        #Number of knots plus one cosmology plus one for mean flux.
        ndim = np.shape(self.param_limits)[0]
        #Limits: we need to hard-prior to the volume of our emulator.
        pr = (self.param_limits[:,1]-self.param_limits[:,0])
        pl = self.param_limits[:,0]
        p0 = [pr*np.random.rand(ndim)+pl for i in range(nwalkers)]
        emcee_sampler = emcee.EnsembleSampler(nwalkers, ndim, self.lnlike_linear)
        emcee_sampler.run_mcmc(p0, burnin)
        emcee_sampler.reset()
        emcee_sampler.run_mcmc(None, nsamples)
        return emcee_sampler

    def new_parameter_limits(self, all_samples, coverage=0.99):
        """Find a square region which includes coverage=0.99 of the total likelihood, for refinement."""
        #Find total amount of likelihood.
        total_likelihood = np.sum(all_samples)
        #Rank order the samples
        #Go down the list of samples until we have > coverage fraction of the total.
        #Find square region which includes all these samples.


if __name__ == "__main__":
    like = LikelihoodClass(os.path.expanduser("~/data/Lya_Boss/cosmo-only-emulator"), os.path.expanduser("~/data/Lya_Boss/cosmo-only-test/AA0.94BB1.2CC0.71DD1.2hub0.71/output/"),mean_flux=True)
