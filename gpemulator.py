"""Building a surrogate using a Gaussian Process."""
# from datetime import datetime
import numpy as np
from latin_hypercube import map_to_unit_cube
#Make sure that we don't accidentally
#get another backend when we import GPy.
import matplotlib
matplotlib.use('PDF')
import GPy
import scipy.optimize
import george

class MultiBinGP(object):
    """A wrapper around the emulator that constructs a separate emulator for each bin.
    Each one has a separate mean flux parameter.
    The t0 parameter fed to the emulator should be constant factors."""
    def __init__(self, *, params, kf, powers, param_limits, gpclass=None, single_output=True):
        #Build an emulator for each redshift separately. This means that the
        #mean flux for each bin can be separated.
        #If this flag is true, each k bin gets its own Gaussian Process
        self.single_output = single_output
        self.kf = kf
        self.nk = np.size(kf)
        assert np.shape(powers)[1] % self.nk == 0
        self.nz = int(np.shape(powers)[1]/self.nk)
        if gpclass is None:
            gpclass = GeorgeGP
        gp = lambda i: gpclass(params=params, powers=powers[:,i].reshape(-1, 1), param_limits = param_limits)
        ngp = self.nz
        if self.single_output:
            ngp *= self.nk
        self.gps = [gp(i) for i in range(ngp)]

    def predict(self,params, tau0_factors = None):
        """Get the predicted flux at a parameter value (or list of parameter values)."""
        std = np.zeros([1,self.nk*self.nz])
        means = np.zeros([1,self.nk*self.nz])
        for i, gp in enumerate(self.gps):
            #Adjust the slope of the mean flux for this bin
            zparams = np.array(params)
            if tau0_factors is not None:
                #Make sure we use optical depth from the right bin: we want integer division.
                if self.single_output:
                    ii = i//self.nk
                else:
                    ii = i
                zparams[0][0] *= tau0_factors[ii]
            (m, s) = gp.predict(zparams)
            if self.single_output:
                means[0,i] = m
                std[:,i] = s
            else:
                means[0,i*self.nk:(i+1)*self.nk] = m
                std[:,i*self.nk:(i+1)*self.nk] = s
        return means, std

#This tends to estimate errors a little smaller than they really are.
#By comparison to the George results I think this is because it is assuming
#an intrinsic non-zero error in the input data.

class GPyGP(object):
    """An emulator using the one in GPy.
       Parameters: params is a list of parameter vectors.
                   powers is a list of flux power spectra (same shape as params).
                   param_limits is a list of parameter limits (shape 2,params)."""
    def __init__(self, *, params, powers, param_limits):
        self.params = params
        self.param_limits = param_limits
        self.intol = 3e-5
        #Should we test the built emulator?
        self._test_interp = False
        #Normalise the flux vectors by the median power spectrum.
        #This ensures that the GP prior (a zero-mean input) is close to true.
        medind = np.argsort(np.mean(powers, axis=1))[np.shape(powers)[0]//2]
        #Normalize by the median vector.
        self.scalefactors = powers[medind,:]
        self.paramzero = params[medind,:]
        #Get the flux power and build an emulator
        self._get_interp(flux_vectors=powers)
        if self._test_interp:
            self._check_interp(powers)
            self._test_interp = False

    def _get_interp(self, powers):
        """Build the actual interpolator."""
        #Map the parameters onto a unit cube so that all the variations are similar in magnitude
        flux_vectors = powers/self.scalefactors -1.
        nparams = np.shape(self.params)[1]
        params_cube = np.array([map_to_unit_cube(pp, self.param_limits) for pp in self.params])
        #Standard squared-exponential kernel with a different length scale for each parameter, as
        #they may have very different physical properties.
        kernel = GPy.kern.Linear(nparams)
        kernel += GPy.kern.RBF(nparams)
        noutput = np.shape(flux_vectors)[1]
        #Note the value of noise_var is actually optimized if non-zero
        self.gp = GPy.models.GPRegression(params_cube, flux_vectors,kernel=kernel, noise_var=1e-10)
        self.gp.optimize(messages=False)
        print(kernel)

    def _check_interp(self, flux_vectors):
        """Check we reproduce the input"""
        means, std = zip(*[self.predict(pp.reshape(1,-1)) for pp in self.params])
        worst = np.abs(np.array(means)/flux_vectors[:,0] - 1)
        if np.max(worst) > self.intol:
            print("Bad interpolation at:", np.where(worst > np.max(worst)*0.9), np.max(worst))
            assert np.max(worst) < self.intol

    def predict(self, params):
        """Get the predicted flux at a parameter value (or list of parameter values)."""
        #Map the parameters onto a unit cube so that all the variations are similar in magnitude
        params_cube = np.array([map_to_unit_cube(pp, self.param_limits) for pp in params])
        flux_predict, var = self.gp.predict(params_cube)
        mean = (flux_predict + 1) * self.scalefactors
        std = np.sqrt(var) * self.scalefactors
        return mean, std

    def get_predict_error(self, test_params, test_exact):
        """Get the difference between the predicted GP
        interpolation and some exactly computed test parameters."""
        #Note: this is not used anywhere
        test_exact = test_exact.reshape(np.shape(test_params)[0],-1)
        predict, sigma = self.predict(test_params)
        return (test_exact - predict)/sigma

def optimize_hypers(gp, data):
    """Optimize the hyperparameters of the GP kernel using george"""

    def nll(p):
        """Objective function (negative log-likelihood) for optimization."""
        gp.set_parameter_vector(p)
        ll = gp.log_likelihood(data, quiet=True)
        return -ll if np.isfinite(ll) else 1e25

    def grad_nll(p):
        """Gradient of the objective function."""
        gp.set_parameter_vector(p)
        return -gp.grad_log_likelihood(data, quiet=True)

    # Run the optimization routine.
    p0 = gp.get_parameter_vector()
    results = scipy.optimize.minimize(nll, p0, jac=grad_nll, method="L-BFGS-B")
    # Update the kernel hyperparameters and print them
    gp.set_parameter_vector(results.x)
    #print(gp.kernel)

class GeorgeGP(GPyGP):
    """Use george as the basic GP code."""

    def _get_interp(self, flux_vectors):
        """Build the actual interpolator."""
        #Need to store the data separately, and it needs to be 1D
        self.flux_vectors = flux_vectors[:,0]
        nparams = np.shape(self.params)[1]
        #Map the parameters onto a unit cube so that all the variations are similar in magnitude
        params_cube = np.array([map_to_unit_cube(pp, self.param_limits) for pp in self.params])
        #Standard squared-exponential kernel with adjustable white noise.
        #Can also use LinearKernel(order=1)
        kernel = 1**2 * george.kernels.DotProductKernel(ndim=nparams)
        #Each parameter gets a different length scale. This doesn't really improve the test fit,
        #but it makes the errors more reasonable.
        kernel += 1**2 * george.kernels.ExpSquaredKernel(metric=np.ones(nparams), ndim=nparams)
        self.gp = george.GP(kernel=kernel, mean = self.scalefactors, fit_mean=False, white_noise=-23, fit_white_noise=True)
        #Specify the 'data errors' on the input flux power spectrum.
        #This models the part of the variation the GP
        #should not attempt to model, because it comes from numerical noise in the simulation.
        #Use a constant 1% of the median power.
        yerr = 1e-2 * self.scalefactors
        #This pre-computes the covariance matrix
        self.gp.compute(params_cube,yerr=yerr)
        optimize_hypers(self.gp, self.flux_vectors)

    def predict(self, params):
        """Get the predicted flux at a parameter value (or list of parameter values)."""
        #Map the parameters onto a unit cube so that all the variations are similar in magnitude
        params_cube = np.array([map_to_unit_cube(pp, self.param_limits) for pp in params])
        #Should I get the full covariance instead of just the diagonal? I think probably not,
        #because that is the covariance between different points in parameter space, and
        #I'm not sure how to translate that into my likelihood. The covariance used in
        #likelihood.py is between different output dimensions.
        flux_predict, var = self.gp.predict(self.flux_vectors, params_cube, return_var=True)
        std = np.sqrt(var)
        return flux_predict[0], std


# This does not work. I have been unable to figure out exactly why, but it appears to be a fairly deep problem,
# possibly numerical issues with marginally ill-conditioned matrices.

# The main problem is that the optimization routine is extremely finicky. This is strange because it is
# based on fmin_l_bfgs_b and does optimization in log space, just like GPy.

# It appears that the gradient of the log likelihood is very different from the same function evaluated in GPy.
# I suspect there is a scaling factor that is missing from scikit-learn's implementation, but I have been unable
# determine where exactly. There also seems to be some difference in the log likelihood function.

# scikit-learn has a nice function to restart the optimizer at a randomly chosen
# (with uniform distribution: should really be log-uniform) initial value.
# However, this doesn't help much since their optimization is broken!

# GPy has code to do something similar (in paramz.model: optimize_restarts), but it is not run by default.
# Also the new optimizer start value is chosen from a normal distribution (!)

#Other problems: The multi-valued output case performs badly (much worse than GPy). Probably this is just the optimizer being bad, but I'm not sure.

# Finally, the error estimation is often very wrong. With a single-output case it generally produces results that are about twice the real error.
#The return_cov output function returns a different answer to return_std - this is expected when there are multiple output dimensions, but
#not when there is only one. It appears that different Cholesky decompositions of the input matrix are being run, so this makes me think
#that there are numerical problems in the code somewhere.

import sklearn.gaussian_process as SKGP

class SkLearnGP(GPyGP):
    """Optimize the emulator with scikit-learn instead"""
    def _get_interp(self, powers):
        """Build the actual interpolator."""
        #Map the parameters onto a unit cube so that all the variations are similar in magnitude
        flux_vectors = powers/self.scalefactors -1.
        nparams = np.shape(self.params)[1]
        params_cube = np.array([map_to_unit_cube(pp, self.param_limits) for pp in self.params])
        #Standard squared-exponential kernel with adjustable white noise.
        #Why is scikit learn's objective function different from gpy?
        kernel = SKGP.kernels.DotProduct(np.sqrt(3))
        kernel += np.sqrt(3) * SKGP.kernels.RBF(np.sqrt(3))
        kernel += SKGP.kernels.WhiteKernel(1e-10,  noise_level_bounds = (1e-18, 1e-8))
        noutput = np.shape(flux_vectors)[1]
        self.gp = SKGP.GaussianProcessRegressor(kernel=kernel, alpha=0, normalize_y=False, copy_X_train=False, n_restarts_optimizer=0)
        self.gp.fit(params_cube, flux_vectors)
        print(self.gp.kernel_)

    def predict(self, params):
        """Get the predicted flux at a parameter value (or list of parameter values)."""
        #Map the parameters onto a unit cube so that all the variations are similar in magnitude
        params_cube = np.array([map_to_unit_cube(pp, self.param_limits) for pp in params])
        #return_std and return_cov give different answers for 1-D GPs. Why?
        flux_predict, var = self.gp.predict(params_cube, return_cov=True)
        std = np.sqrt(np.diag(var))
        std[np.where(std < 1e-2)] = 1e-2
        return (flux_predict[0] + 1) * self.scalefactors, std * self.scalefactors

import emcee

# Working optimizer for scikit-learn. Unfortunately this is unreasonably slow.
def fmin_emcee(obj_func, initial_theta, bounds, nwalkers=20, nsamples = 200):
    """Initialise and run emcee to do my optimization. Use scipy.optimize at the end to tune."""
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
    pos, _, _ = emcee_sampler.run_mcmc(p0, nsamples)
    #Check things are reasonable
    assert np.all(emcee_sampler.acceptance_fraction > 0.01)
    #Return maximum likelihood
    lnp = emcee_sampler.flatlnprobability
    theta,fmin = emcee_sampler.flatchain[np.argmax(lnp)],-np.max(lnp)
    theta_scipy, fmin_scipy, convergence_dict = scipy.optimize.fmin_l_bfgs_b(obj_func, theta, bounds=None)
    if convergence_dict["warnflag"] != 0:
        print("fmin_l_bfgs_b terminated abnormally with the "
                      " state: %s" % convergence_dict)
    #Probably this means that bfgs went crazy.
    if convergence_dict["warnflag"] != 0 or np.any(theta_scipy > bounds[:,1]) or np.any(theta_scipy < bounds[:,0]):
        return theta, fmin
    return theta_scipy,fmin_scipy
