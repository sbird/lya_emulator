"""Building a surrogate using a Gaussian Process."""
import numpy as np
import GPy
from latin_hypercube import map_to_unit_cube
import scipy.optimize
import emcee

def fmin_bounds(obj_func, initial_theta, bounds):
    """Call simplex algorithm for optimisation, ignoring the bounds."""
    _ = bounds
    result = scipy.optimize.minimize(lambda x0 : obj_func(x0)[0], initial_theta, method="Nelder-Mead")
    return result.x, result.fun

def fmin_emcee(obj_func, initial_theta, bounds, nwalkers=30, burnin=100, nsamples = 800):
    """Initialise and run emcee."""
    #Number of knots plus one cosmology plus one for mean flux.
    ndim = np.shape(bounds)[0]
    #Limits: we need to hard-prior to the volume of our emulator.
    pr = (bounds[:,1]-bounds[:,0])
    #Priors are assumed to be in the middle.
    p0 = [initial_theta+2*pr/16.*np.random.rand(ndim)-pr/16. for _ in range(nwalkers)]
    #assert np.all([np.isfinite(self.lnlike_linear(pp)) for pp in p0])
    def bnd_obj_func(x0):
        """Version of obj_func which returns -Nan when outside of bounds"""
        if np.any(x0 < bounds[:,0]) or np.any(x0 > bounds[:,1]):
            return -np.inf
        return -obj_func(x0)[0]
    emcee_sampler = emcee.EnsembleSampler(nwalkers, ndim, bnd_obj_func)
    pos, _, _ = emcee_sampler.run_mcmc(p0, burnin)
    #Check things are reasonable
    assert np.all(emcee_sampler.acceptance_fraction > 0.01)
    emcee_sampler.reset()
    emcee_sampler.run_mcmc(pos, nsamples)
    #Return maximum likelihood
    lnp = emcee_sampler.flatlnprobability
    theta,fmin = emcee_sampler.flatchain[np.argmax(lnp)],-np.max(lnp)
#     print("theta_mc=",theta)
#     print("fmin_mc=", fmin)
    return theta,fmin

class SkLearnGP(object):
    """An emulator using the one in Scikit-learn"""
    def __init__(self, *, params, kf, powers,param_limits):
        self.powers = powers
        self.params = params
        self.param_limits = param_limits
        self.cur_tau_factor = -1
        self.kf = kf
        self.intol = 3e-5

    def _get_interp(self, tau0_factor=None):
        """Build the actual interpolator."""
        self.cur_tau_factor = tau0_factor
        flux_vectors = np.array([ps.get_power(kf = self.kf, tau0_factor = tau0_factor) for ps in self.powers])
        #Map the parameters onto a unit cube so that all the variations are similar in magnitude
        nparams = np.shape(self.params)[1]
        params_cube = np.array([map_to_unit_cube(pp, self.param_limits) for pp in self.params])
        #Normalise the flux vectors by the median power spectrum.
        #This ensures that the GP prior (a zero-mean input) is close to true.
        medind = np.argsort(np.mean(flux_vectors, axis=1))[np.shape(flux_vectors)[0]//2]
        self.scalefactors = flux_vectors[medind,:]
        self.paramzero = params_cube[medind,:]
        #Normalise by the median value
        normspectra = flux_vectors/self.scalefactors -1.
        #Standard squared-exponential kernel with a different length scale for each parameter, as
        #they may have very different physical properties.
        kernel = GPy.kern.Bias(input_dim = nparams)
        kernel += GPy.kern.Matern32(nparams)
        kernel += GPy.kern.Linear(nparams)
        self.gp = GPy.models.GPRegression(params_cube, normspectra,kernel=kernel, noise_var=1e-7)
        self.gp.optimize(messages=True) #, optimizer=fmin_emcee)
        #Check we reproduce the input
        test,_ = self.predict(self.params[0,:].reshape(1,-1), tau0_factor=tau0_factor)
        worst = np.abs(test[0] / flux_vectors[0,:]-1)
        if np.max(worst) > self.intol:
            print("Bad interpolation at:",np.where(worst > np.max(worst)*0.9), np.max(worst))
            assert np.max(worst) < self.intol

    def predict(self, params,tau0_factor):
        """Get the predicted flux at a parameter value (or list of parameter values)."""
        #First get the residuals
        if tau0_factor is not self.cur_tau_factor:
            self._get_interp(tau0_factor = tau0_factor)
        #Map the parameters onto a unit cube so that all the variations are similar in magnitude
        params_cube = np.array([map_to_unit_cube(pp, self.param_limits) for pp in params])
        flux_predict, std = self.gp.predict(params_cube)
        mean = (flux_predict+1)*self.scalefactors
        std = std * self.scalefactors
        return mean, std

    def get_predict_error(self, test_params, test_exact):
        """Get the difference between the predicted GP
        interpolation and some exactly computed test parameters."""
        test_exact = test_exact.reshape(np.shape(test_params)[0],-1)
        predict, sigma = self.predict(test_params,tau0_factor=self.cur_tau_factor)
        #The transposes are because of numpy broadcasting rules only doing the last axis
        return ((test_exact - predict).T/np.sqrt(sigma)).T

class SDSSData(object):
    """A class to store the flux power and corresponding covariance matrix from SDSS. A little tricky because of the redshift binning."""
    def __init__(self, datafile="data/lya.sdss.table.txt", covarfile="data/lya.sdss.covar.txt"):
        # Read SDSS best-fit data.
        # Contains the redshift wavenumber from SDSS
        # See 0405013 section 5.
        # First column is redshift
        # Second is k in (km/s)^-1
        # Third column is P_F(k)
        # Fourth column (ignored): square roots of the diagonal elements
        # of the covariance matrix. We use the full covariance matrix instead.
        # Fifth column (ignored): The amount of foreground noise power subtracted from each bin.
        # Sixth column (ignored): The amound of background power subtracted from each bin.
        # A metal contamination subtraction that McDonald does but we don't.
        data = np.loadtxt(datafile)
        self.redshifts = data[:,0]
        self.kf = data[:,1]
        self.pf = data[:,1]
        #The covariance matrix, correlating each k and z bin with every other.
        #kbins vary first, so that we have 11 bins with z=2.2, then 11 with z=2.4,etc.
        self.covar = np.loadtxt(covarfile)

    def get_kf(self):
        """Get the (unique) flux k values"""
        return np.sort(np.array(list(set(self.kf))))

    def get_redshifts(self):
        """Get the (unique) redshift bins, sorted in decreasing redshift"""
        return np.sort(np.array(list(set(self.redshifts))))[::-1]

    def get_icovar(self):
        """Get the inverse covariance matrix"""
        return np.linalg.inv(self.covar)

    def get_covar(self):
        """Get the inverse covariance matrix"""
        return self.covar
