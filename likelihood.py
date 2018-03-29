"""Module for computing the likelihood function for the forest emulator."""
import os
import os.path
import math
import numpy as np
import numpy.testing as npt
import emcee
import coarse_grid
import flux_power
import lyman_data
import mean_flux as mflux
from datetime import datetime

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

def gelman_rubin(chain):
    """Compute the Gelman-Rubin statistic for a chain"""
    ssq = np.var(chain, axis=1, ddof=1)
    W = np.mean(ssq, axis=0)
    tb = np.mean(chain, axis=1)
    tbb = np.mean(tb, axis=0)
    m = chain.shape[0]
    n = chain.shape[1]
    B = n / (m - 1) * np.sum((tbb - tb)**2, axis=0)
    var_t = (n - 1) / n * W + 1 / n * B
    R = np.sqrt(var_t / W)
    return R

class LikelihoodClass(object):
    """Class to contain likelihood computations."""
    def __init__(self, basedir, datadir, mean_flux='s'):
        """Initialise the emulator by loading the flux power spectra from the simulations."""
        #Use the BOSS covariance matrix
        self.sdss = lyman_data.BOSSData()
        #'Data' now is a simulation
        self.max_z = 4.2
        myspec = flux_power.MySpectra(max_z=self.max_z)
        self.zout = myspec.zout
        pps = myspec.get_snapshot_list(datadir)
        self.data_fluxpower = pps.get_power(kf=self.sdss.get_kf(),tau0_factors=mflux.obs_mean_tau(self.zout, amp = -0.5e-4))
        assert np.size(self.data_fluxpower) % np.size(self.sdss.get_kf) == 0
        #Get the emulator
        if mean_flux == 'c':
            mf = mflux.ConstMeanFlux(value = 0.95)
        else:
            mf = mflux.MeanFluxFactor()
        self.emulator = coarse_grid.KnotEmulator(basedir, kf=self.sdss.get_kf(), mf=mf)
        self.emulator.load()
        self.param_limits = self.emulator.get_param_limits(include_dense=True)
        #As each redshift bin is independent, for redshift-dependent mean flux models
        #we just need to convert the input parameters to a list of mean flux scalings
        #in each redshift bin.
        #This is an example which parametrises the mean flux as an amplitude and slope.
        self.mf_slope = False
        if mean_flux == 's':
            #Add a slope to the parameter limits
            self.param_limits = np.vstack([[-0.25, 0.25], self.param_limits])
            #Shrink param limits t0 so that even with
            #a slope they are within the emulator range
            self.param_limits[1,:] = [0.75,1.25]
            self.mf_slope = True
        self.ndim = np.shape(self.param_limits)[0]
        assert np.shape(self.param_limits)[1] == 2
        print('Beginning to generate emulator at', str(datetime.now()))
        self.gpemu = self.emulator.get_emulator(max_z=4.2)
        print('Finished generating emulator at', str(datetime.now()))

    def likelihood(self, params, include_emu=True):
        """A simple likelihood function for the Lyman-alpha forest.
        Assumes data is quadratic with a covariance matrix."""
        nparams = params
        if self.mf_slope:
            tau0_fac = mflux.mean_flux_slope_to_factor(self.zout, params[0])
            nparams = params[1:]
        else: #Otherwise bug if choose mean_flux = 'c'
            tau0_fac = None
        if np.any(params >= self.param_limits[:,1]) or np.any(params <= self.param_limits[:,0]):
            return -np.inf
        #Set parameter limits as the hull of the original emulator.
        predicted, std = self.gpemu.predict(np.array(nparams).reshape(1,-1), tau0_factors = tau0_fac)

        #Save emulated flux power specra for analysis
        self.emulated_flux_power = predicted
        self.emulated_flux_power_std = std

        diff = predicted[0]-self.data_fluxpower
        nkf = len(self.sdss.get_kf())
        nz = int(len(diff)/nkf)
        #Likelihood using full covariance matrix
        chi2 = 0
        #Redshifts
        sdssz = self.sdss.get_redshifts()

        #Fix maximum redshift bug
        sdssz = sdssz[sdssz <= self.max_z]

        #Important assertion
        assert nz == sdssz.size
        npt.assert_allclose(sdssz, self.zout, atol=1.e-16)
        #print('SDSS redshifts are', sdssz)

        for bb in range(nz):
            diff_bin = diff[nkf*bb:nkf*(bb+1)]
            std_bin = std[0,nkf*bb:nkf*(bb+1)]
            covar_bin = self.sdss.get_covar(sdssz[bb])
            assert np.shape(np.diag(std_bin**2)) == np.shape(covar_bin)
            if include_emu:
                #Assume each k bin is independent
                covar_bin += np.diag(std_bin**2)
                #Assume completely correlated emulator errors within this bin
#                 covar_bin += np.matmul(np.diag(std_bin**2),np.ones_like(covar_bin))
            icov_bin = np.linalg.inv(covar_bin)
            (sign, cdet) = np.linalg.slogdet(covar_bin)
            dcd = - np.dot(diff_bin, np.dot(icov_bin, diff_bin),)/2.
            chi2 += dcd -0.5* cdet
            assert 0 > chi2 > -2**31
            assert not np.isnan(chi2)
        return chi2

    def do_sampling(self, savefile, nwalkers=100, burnin=5000, nsamples=5000, while_loop=True):
        """Initialise and run emcee."""
        pnames = self.emulator.print_pnames()
        if self.mf_slope:
            pnames = [('dtau0',r'd\tau_0'),]+pnames
        with open(savefile+"_names.txt",'w') as ff:
            for pp in pnames:
                ff.write("%s %s\n" % pp)
        #Limits: we need to hard-prior to the volume of our emulator.
        pr = (self.param_limits[:,1]-self.param_limits[:,0])
        #Priors are assumed to be in the middle.
        cent = (self.param_limits[:,1]+self.param_limits[:,0])/2.
        p0 = [cent+2*pr/16.*np.random.rand(self.ndim)-pr/16. for _ in range(nwalkers)]
        assert np.all([np.isfinite(self.likelihood(pp)) for pp in p0])
        emcee_sampler = emcee.EnsembleSampler(nwalkers, self.ndim, self.likelihood)
        pos, _, _ = emcee_sampler.run_mcmc(p0, burnin)
         #Check things are reasonable
        assert np.all(emcee_sampler.acceptance_fraction > 0.01)
        emcee_sampler.reset()
        self.cur_results = emcee_sampler
        gr = 10.
        while np.any(gr > 1.05):
            emcee_sampler.run_mcmc(pos, nsamples)
            gr = gelman_rubin(emcee_sampler.chain)
            print("Total samples:",nsamples," Gelman-Rubin: ",gr)
            np.savetxt(savefile, emcee_sampler.flatchain)
            if while_loop is False:
                break
        return emcee_sampler

    def new_parameter_limits(self, confidence=0.99, include_dense=False):
        """Find a square region which includes coverage of the parameters in each direction, for refinement.
        Confidence must be 0.68, 0.95 or 0.99."""
        #Use the marginalised distributions to find the square region.
        #If there are strong degeneracies this will be very inefficient.
        #We could rotate the parameters here,
        #but ideally we would do that before running the coarse grid anyway.
        #Get marginalised statistics.
        limits = np.percentile(self.cur_results.flatchain, [100-100*confidence, 100*confidence], axis=0)
        #Discard dense params
        ndense = len(self.emulator.mf.dense_param_names)
        if self.mf_slope:
            ndense+=1
        if include_dense:
            ndense = 0
        lower = limits[ndense:,0]
        upper = limits[ndense:, 1]
        assert np.all(lower < upper)
        new_par = limits[ndense:,:]
        return new_par

    def check_for_refinement(self, conf = 0.95, frac = 1.3):
        """Crude check for refinement: check whether the likelihood is dominated by
           emulator error at the 1 sigma contours."""
        limits = self.new_parameter_limits(confidence=conf, include_dense = True)
        while True:
            midpt = np.mean(limits, axis=1)
            limits[:,0] = 1.4*(limits[:,0] - midpt) + midpt
            limits[:,0] = np.max([limits[:,0], self.param_limits[:,0]],axis=0)
            limits[:,1] = 1.4*(limits[:,1] - midpt) + midpt
            limits[:,1] = np.min([limits[:,1], self.param_limits[:,1]],axis=0)
            if np.all(limits == self.param_limits):
                break
            ue = self.likelihood(limits[:,0])[0]
            un = self.likelihood(limits[:,0],include_emu=False)[0]
            le = self.likelihood(limits[:,1])[0]
            ln = self.likelihood(limits[:,1],include_emu=False)[0]
            #This should be close to 1.
            print("up =",un/ue," low=",ln/le)
            if (un/ue < frac) and (ln/le < frac):
                break
        return limits

    def refinement(self,nsamples,confidence=0.99):
        """Do the refinement step."""
        new_limits = self.new_parameter_limits(confidence=confidence)
        new_samples = self.emulator.build_params(nsamples=nsamples,limits=new_limits, use_existing=True)
        assert np.shape(new_samples)[0] == nsamples
        self.emulator.gen_simulations(nsamples=nsamples, samples=new_samples)

if __name__ == "__main__":
#     like = LikelihoodClass(basedir=os.path.expanduser("~/data/Lya_Boss/hires_knots_refine"), datadir=os.path.expanduser("~/data/Lya_Boss/hires_knots_test/AA0.97BB1.3CC0.67DD1.3heat_slope0.083heat_amp0.92hub0.69/output"))
    like = LikelihoodClass(basedir=os.path.expanduser("~/Simulations/Lya_Boss/hires_knots"), datadir=os.path.expanduser("~/Simulations/Lya_Boss/hires_knots_test/AA0.97BB1.3CC0.67DD1.3heat_slope0.083heat_amp0.92hub0.69/output"))
    #Works very well!
    #     like = LikelihoodClass(basedir=os.path.expanduser("~/data/Lya_Boss/hires_knots"), datadir=os.path.expanduser("~/data/Lya_Boss/hires_knots/AA0.96BB1.3CC1DD1.3heat_slope-5.6e-17heat_amp1.2hub0.66/output"))
    output = like.do_sampling(os.path.expanduser("~/Simulations/Lya_Boss/hires_knots_test/AA0.97BB1.3_chain.txt"))
