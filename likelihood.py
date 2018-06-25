"""Module for computing the likelihood function for the forest emulator."""
import os
import os.path
import math
from datetime import datetime
import numpy as np
import numpy.testing as npt
import emcee
import scipy.interpolate
import coarse_grid
import flux_power
import lyman_data
import mean_flux as mflux
from quadratic_emulator import QuadraticEmulator

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

def load_data(datadir, *, kf, max_z=4.2):
    """Load and initialise a "fake data" flux power spectrum"""
    #Load the data directory
    myspec = flux_power.MySpectra(max_z=max_z)
    pps = myspec.get_snapshot_list(datadir)
    #self.data_fluxpower is used in likelihood.
    data_fluxpower = pps.get_power(kf=kf, mean_fluxes=np.exp(-mflux.obs_mean_tau(myspec.zout, amp=0)))
    assert np.size(data_fluxpower) % np.size(kf) == 0
    return data_fluxpower

class LikelihoodClass(object):
    """Class to contain likelihood computations."""
    def __init__(self, basedir, mean_flux='s', max_z = 4.2, emulator_class="standard"):
        """Initialise the emulator by loading the flux power spectra from the simulations."""
        #Use the BOSS covariance matrix
        self.sdss = lyman_data.BOSSData()
        #'Data' now is a simulation
        self.max_z = max_z
        myspec = flux_power.MySpectra(max_z=max_z)
        self.zout = myspec.zout
        self.kf = self.sdss.get_kf()
        self.mf_slope = False
        #Param limits on t0
        t0_factor = np.array([0.75,1.25])
        #Get the emulator
        if mean_flux == 'c':
            mf = mflux.ConstMeanFlux(value = 1)
        #As each redshift bin is independent, for redshift-dependent mean flux models
        #we just need to convert the input parameters to a list of mean flux scalings
        #in each redshift bin.
        #This is an example which parametrises the mean flux as an amplitude and slope.
        elif mean_flux == 's':
            #Add a slope to the parameter limits
            t0_slope =  np.array([-0.25, 0.25])
            self.mf_slope = True
            slopehigh = np.max(mflux.mean_flux_slope_to_factor(np.linspace(2.2, max_z, 11),0.25))
            slopelow = np.min(mflux.mean_flux_slope_to_factor(np.linspace(2.2, max_z, 11),-0.25))
            dense_limits = np.array([np.array(t0_factor) * np.array([slopelow, slopehigh])])
            mf = mflux.MeanFluxFactor(dense_limits = dense_limits)
        else:
            mf = mflux.MeanFluxFactor()
        if emulator_class == "standard":
            self.emulator = coarse_grid.Emulator(basedir, kf=self.kf, mf=mf)
        elif emulator_class == "knot":
            self.emulator = coarse_grid.KnotEmulator(basedir, kf=self.kf, mf=mf)
        elif emulator_class == "quadratic":
            self.emulator = QuadraticEmulator(basedir, kf=self.kf, mf=mf)
        else:
            raise ValueError("Emulator class not recognised")
        self.emulator.load()
        self.param_limits = self.emulator.get_param_limits(include_dense=True)
        if mean_flux == 's':
            #Add a slope to the parameter limits
            self.param_limits = np.vstack([t0_slope, self.param_limits])
            #Shrink param limits t0 so that even with
            #a slope they are within the emulator range
            self.param_limits[1,:] = t0_factor
        self.ndim = np.shape(self.param_limits)[0]
        assert np.shape(self.param_limits)[1] == 2
        print('Beginning to generate emulator at', str(datetime.now()))
        self.gpemu = self.emulator.get_emulator(max_z=max_z)
        print('Finished generating emulator at', str(datetime.now()))

    def likelihood(self, params, include_emu=True, data_power=None):
        """A simple likelihood function for the Lyman-alpha forest.
        Assumes data is quadratic with a covariance matrix.
        The covariance for the emulator points is assumed to be
        completely correlated with each z bin, as the emulator
        parameters are estimated once per z bin."""
        nparams = params
        if data_power is None:
            data_power = self.data_fluxpower
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

        diff = predicted[0]-data_power
        nkf = len(self.kf)
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
#                 covar_emu = np.diag(std_bin**2)
                #Assume completely correlated emulator errors within this bin
                covar_emu = np.outer(std_bin, std_bin)
                covar_bin += covar_emu
            icov_bin = np.linalg.inv(covar_bin)
            (_, cdet) = np.linalg.slogdet(covar_bin)
            dcd = - np.dot(diff_bin, np.dot(icov_bin, diff_bin),)/2.
            chi2 += dcd -0.5* cdet
            assert 0 > chi2 > -2**31
            assert not np.isnan(chi2)
        return chi2

    def load(self, savefile):
        """Load the chain from a savefile"""
        self.flatchain = np.loadtxt(savefile)

    def do_sampling(self, savefile, datadir, nwalkers=100, burnin=1000, nsamples=3000, while_loop=True):
        """Initialise and run emcee."""
        pnames = self.emulator.print_pnames()
        #Load the data directory
        self.data_fluxpower = load_data(datadir, kf=self.kf)
        #Set up mean flux
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
        while np.any(gr > 1.01):
            emcee_sampler.run_mcmc(pos, nsamples)
            gr = gelman_rubin(emcee_sampler.chain)
            print("Total samples:",nsamples," Gelman-Rubin: ",gr)
            np.savetxt(savefile, emcee_sampler.flatchain)
            if while_loop is False:
                break
        self.flatchain = emcee_sampler.flatchain
        return emcee_sampler

    def new_parameter_limits(self, confidence=0.99, include_dense=False):
        """Find a square region which includes coverage of the parameters in each direction, for refinement.
        Confidence must be 0.68, 0.95 or 0.99."""
        #Use the marginalised distributions to find the square region.
        #If there are strong degeneracies this will be very inefficient.
        #We could rotate the parameters here,
        #but ideally we would do that before running the coarse grid anyway.
        #Get marginalised statistics.
        limits = np.percentile(self.flatchain, [100-100*confidence, 100*confidence], axis=0).T
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

    def get_covar_det(self, params, include_emu):
        """Get the determinant of the covariance matrix.for certain parameters"""
        nparams = params
        if self.mf_slope:
            tau0_fac = mflux.mean_flux_slope_to_factor(self.zout, params[0])
            nparams = params[1:]
        else: #Otherwise bug if choose mean_flux = 'c'
            tau0_fac = None
        if np.any(params >= self.param_limits[:,1]) or np.any(params <= self.param_limits[:,0]):
            return -np.inf
        sdssz = self.sdss.get_redshifts()
        #Fix maximum redshift bug
        sdssz = sdssz[sdssz <= self.max_z]
        nz = sdssz.size
        nkf = len(self.kf)
        if include_emu:
            _, std = self.gpemu.predict(np.array(nparams).reshape(1,-1), tau0_factors = tau0_fac)
        detc = 1
        for bb in range(nz):
            covar_bin = self.sdss.get_covar(sdssz[bb])
            if include_emu:
                std_bin = std[0,nkf*bb:nkf*(bb+1)]
                #Assume completely correlated emulator errors within this bin
                covar_emu = np.outer(std_bin, std_bin)
                covar_bin += covar_emu
            _, det_bin = np.linalg.slogdet(covar_bin)
            #We have a block diagonal covariance
            detc *= det_bin
        return detc

    def refine_metric(self, params):
        """This evaluates the 'refinement metric':
           the extent to which the emulator error dominates the covariance.
           The idea is that when it is > 1, refinement is necessary"""
        detnoemu = self.get_covar_det(params, False)
        detemu = self.get_covar_det(params, True)
        return detemu/detnoemu

    def check_for_refinement(self, conf = 0.95, thresh = 1.05):
        """Crude check for refinement: check whether the likelihood is dominated by
           emulator error at the 1 sigma contours."""
        limits = self.new_parameter_limits(confidence=conf, include_dense = True)
        while True:
            #Do the check
            uref = self.refine_metric(limits[:,0])
            lref = self.refine_metric(limits[:,1])
            #This should be close to 1.
            print("up =",uref," low=",lref)
            if (uref < thresh) and (lref < thresh):
                break
            #Iterate by moving each limit 40% outwards.
            midpt = np.mean(limits, axis=1)
            limits[:,0] = 1.4*(limits[:,0] - midpt) + midpt
            limits[:,0] = np.max([limits[:,0], self.param_limits[:,0]],axis=0)
            limits[:,1] = 1.4*(limits[:,1] - midpt) + midpt
            limits[:,1] = np.min([limits[:,1], self.param_limits[:,1]],axis=0)
            if np.all(limits == self.param_limits):
                break
        return limits

    def refinement(self,nsamples,confidence=0.99):
        """Do the refinement step."""
        new_limits = self.new_parameter_limits(confidence=confidence)
        new_samples = self.emulator.build_params(nsamples=nsamples,limits=new_limits, use_existing=True)
        assert np.shape(new_samples)[0] == nsamples
        self.emulator.gen_simulations(nsamples=nsamples, samples=new_samples)

    def make_err_grid(self, i, j, samples = 30000):
        """Make an error grid"""
        ndim = np.size(self.param_limits[:,0])
        rr = lambda x : np.random.rand(ndim)*(self.param_limits[:,1]-self.param_limits[:,0]) + self.param_limits[:,0]
        rsamples = np.array([rr(i) for i in range(samples)])
        randscores = [self.refine_metric(rr) for rr in rsamples]
        grid_x, grid_y = np.mgrid[0:1:200j, 0:1:200j]
        grid_x = grid_x * (self.param_limits[i,1] - self.param_limits[i,0]) + self.param_limits[i,0]
        grid_y = grid_y * (self.param_limits[j,1] - self.param_limits[j,0]) + self.param_limits[j,0]
        grid = scipy.interpolate.griddata(rsamples[:,(i,j)], randscores,(grid_x,grid_y),fill_value = 0)
        return grid

if __name__ == "__main__":
#     like = LikelihoodClass(basedir=os.path.expanduser("~/data/Lya_Boss/hires_knots_refine"), datadir=os.path.expanduser("~/data/Lya_Boss/hires_knots_test/AA0.97BB1.3CC0.67DD1.3heat_slope0.083heat_amp0.92hub0.69/output"))
    like = LikelihoodClass(basedir=os.path.expanduser("simulations/hires_s8"), )
    testdata=os.path.expanduser("simulations/hires_s8_test/AA0.97BB1.3CC0.67DD1.3heat_slope0.083heat_amp0.92hub0.69/output")
    #Works very well!
    #     like = LikelihoodClass(basedir=os.path.expanduser("~/data/Lya_Boss/hires_knots"), datadir=os.path.expanduser("~/data/Lya_Boss/hires_knots/AA0.96BB1.3CC1DD1.3heat_slope-5.6e-17heat_amp1.2hub0.66/output"))
    output = like.do_sampling(os.path.expanduser("simulations/hires_s8_test/AA0.97BB1.3_chain.txt"), datadir=testdata)
