"""Building a surrogate using a Gaussian Process."""
# from datetime import datetime
import numpy as np
from latin_hypercube import map_to_unit_cube
#Make sure that we don't accidentally
#get another backend when we import GPy.
import matplotlib
matplotlib.use('PDF')
import GPy
import george
import scipy
import emcee

class MultiBinGP(object):
    """A wrapper around the emulator that constructs a separate emulator for each bin.
    Each one has a separate mean flux parameter.
    The t0 parameter fed to the emulator should be constant factors."""
    def __init__(self, *, params, kf, powers, param_limits):
        #Build an emulator for each redshift separately. This means that the
        #mean flux for each bin can be separated.
        self.kf = kf
        self.nk = np.size(kf)
        assert np.shape(powers)[1] % self.nk == 0
        self.nz = int(np.shape(powers)[1]/self.nk)
        gp = lambda i: SkLearnGP(params=params, powers=powers[:,i*self.nk:(i+1)*self.nk], param_limits = param_limits, kf=kf)
        print('Number of redshifts for emulator generation =', self.nz)
        self.gps = [gp(i) for i in range(self.nz)]

    def predict(self,params, tau0_factors = None):
        """Get the predicted flux at a parameter value (or list of parameter values)."""
        std = np.zeros([1,self.nk*self.nz])
        means = np.zeros([1,self.nk*self.nz])
        for i, gp in enumerate(self.gps): #Looping over redshifts
            #Adjust the slope of the mean flux for this bin
            zparams = np.array(params)
            if tau0_factors is not None:
                zparams[0][0] *= tau0_factors[i] #Multiplying t0[z] by "tau0_factors"[z]
            (m, s) = gp.predict(zparams)
            means[0,i*self.nk:(i+1)*self.nk] = m
            std[:,i*self.nk:(i+1)*self.nk] = s
        return means, std

class SkLearnGP(object):
    """An emulator wrapping a GP code.
       Parameters: params is a list of parameter vectors.
                   powers is a list of flux power spectra (same shape as params).
                   param_limits is a list of parameter limits (shape 2,params)."""
    def __init__(self, *, params, powers,param_limits, kf):
        self.params = params
        self.param_limits = param_limits
        self.kf = kf
        self.intol = 1e-4
        #Should we test the built emulator?
        #Turn this off because our emulator is now so large
        #that it always fails because of Gaussianity!
        self._test_interp = False
        #Get the flux power and build an emulator
        self._get_interp(flux_vectors=powers)
        if self._test_interp:
            self._check_interp(powers)
            self._test_interp = False

    def _get_interp(self, flux_vectors):
        """Build the actual interpolator."""
        #Map the parameters onto a unit cube so that all the variations are similar in magnitude
        nparams = np.shape(self.params)[1]
        params_cube = np.array([map_to_unit_cube(pp, self.param_limits) for pp in self.params])
        #Check that we span the parameter space
        for i in range(nparams):
            assert np.max(params_cube[:,i]) > 0.9
            assert np.min(params_cube[:,i]) < 0.1
        #print('Normalised parameter values =', params_cube)
        #Normalise the flux vectors by the median power spectrum.
        #This ensures that the GP prior (a zero-mean input) is close to true.
        medind = np.argsort(np.mean(flux_vectors, axis=1))[np.shape(flux_vectors)[0]//2]
        self.scalefactors = flux_vectors[medind,:]
        self.paramzero = params_cube[medind,:]
        #Normalise by the median value
        normspectra = flux_vectors/self.scalefactors
        #Find the mean value and perform a separate GP fit for it.
        imax = np.max(np.where(self.kf < 0.01))
        self.means = np.mean(normspectra[:,:imax], axis=1)
        #Standard squared-exponential kernel with axis-aligned metric function.
        #Each parameter gets a different length scale.
        mkernel = 1**2 * george.kernels.ExpSquaredKernel(metric=2*np.ones(nparams-1), ndim=nparams,axes=np.arange(1,nparams))
        #Can also use LinearKernel(order=1)
        #Use a different kernel for each axis because we want each parameter to have a different linear fit.
        #Linear kernel is specific to this problem and models parameters that we know are fit by linear regression.
        #This is for the mean flux
        mkernel += george.kernels.LinearKernel(log_gamma2=0.5, order=1, ndim=nparams, axes=0)
        #This is for sigma8
        mkernel += george.kernels.LinearKernel(log_gamma2=0.5, order=1, ndim=nparams, axes=2)
#         for i in range(nparams):
#             mkernel += george.kernels.ConstantKernel(0.1, ndim=nparams, axes=i) * george.kernels.Matern52Kernel(metric = 2, ndim=nparams, axes=i)
#             mkernel += george.kernels.LinearKernel(log_gamma2=0.5, order=1, ndim=nparams, axes=i)
        self.meangp = george.GP(kernel=mkernel, mean = 0, fit_mean=True, white_noise=0)
        self.meangp.compute(params_cube,yerr=0)
        optimize_hypers(self.meangp, self.means)

        #Now fit the multi-dimensional residual shapes.
        normflux = normspectra-np.outer(self.means, np.ones(np.shape(normspectra)[1]))
        kernel = GPy.kern.Linear(nparams)
        kernel += GPy.kern.RBF(nparams)
        self.gp = GPy.models.GPRegression(params_cube, normflux,kernel=kernel, noise_var=1e-10)
        status = self.gp.optimize(messages=True) #True
        #print('Gradients of model hyperparameters [after optimisation] =', self.gp.gradient)
        #Let's check that hyperparameter optimisation is converged
        if status.status != 'Converged':
            self.gp.optimize_restarts(num_restarts=3)
        print(self.gp)
        #print('Gradients of model hyperparameters [after second optimisation (x 10)] =', self.gp.gradient)

    def _check_interp(self, flux_vectors):
        """Check we reproduce the input"""
        for i, pp in enumerate(self.params):
            means, _ = self.predict(pp.reshape(1,-1))
            worst = np.abs(np.array(means) - flux_vectors[i,:])/self.scalefactors
            if np.max(worst) > self.intol:
                print("Bad interpolation at:", np.where(worst > np.max(worst)*0.9), np.max(worst))
                assert np.max(worst) < self.intol

    def predict(self, params):
        """Get the predicted flux at a parameter value (or list of parameter values)."""
        #Map the parameters onto a unit cube so that all the variations are similar in magnitude
        params_cube = np.array([map_to_unit_cube(pp, self.param_limits) for pp in params])
        flux_predict, var = self.gp.predict(params_cube)
        mean_predict, meanvar = self.meangp.predict(self.means, params_cube, return_var=True)
        mean = (mean_predict + flux_predict)*self.scalefactors
        #This works almost as well as the combination!
#         mean = (mean_predict)*self.scalefactors
        std = np.sqrt(meanvar + var) * self.scalefactors
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
    if not results.success:
        print(results.message)
    # Update the kernel hyperparameters and print them
    gp.set_parameter_vector(results.x)
    print(gp.kernel)

# Emcee optimizer for hyperparameters for george, so we can find errors.
def optimize_hypers_emcee(gp, data, nwalkers=20, nsamples = 20000):
    """Initialise and run emcee to do my optimization."""

    def nll(p):
        """Objective function (negative log-likelihood) for optimization."""
        if np.any((-15 > p) + (p > 15)):
            return -np.inf
        gp.set_parameter_vector(p)
        ll = gp.log_likelihood(data, quiet=True)
        return -ll if np.isfinite(ll) else 1e25

    ndim = len(gp)
    #Priors are assumed to be in the middle.
    p0 = gp.get_parameter_vector() + 1e-4 * np.random.randn(nwalkers, ndim)
    emcee_sampler = emcee.EnsembleSampler(nwalkers, ndim, nll)
    pos, _, _ = emcee_sampler.run_mcmc(p0, 200)
    #Check things are reasonable
    assert np.all(emcee_sampler.acceptance_fraction > 0.01)
    emcee_sampler.reset()
    emcee_sampler.run_mcmc(pos, nsamples)
    #Save results for future analysis
    savefile = "hypers_"+str(optimize_hypers_emcee.count)+".txt"
    optimize_hypers_emcee.count+=1
    np.savetxt(savefile, emcee_sampler.flatchain)
    #Return maximum likelihood
    lnp = emcee_sampler.flatlnprobability
    theta = emcee_sampler.flatchain[np.argmax(lnp)]
    gp.set_parameter_vector(theta)
    print(gp.kernel)

optimize_hypers_emcee.count = 0
