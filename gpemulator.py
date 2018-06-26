"""Building a surrogate using a Gaussian Process."""
# from datetime import datetime
import numpy as np
from latin_hypercube import map_to_unit_cube
#Make sure that we don't accidentally
#get another backend when we import GPy.
import matplotlib
matplotlib.use('PDF')
import GPy
import sklearn.gaussian_process as SKGP

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
        gp = lambda i: GPyGP(params=params, powers=powers[:,i].reshape(-1, 1), param_limits = param_limits)
        skgp = lambda i: SkLearnGP(params=params, powers=powers[:,i].reshape(-1, 1), param_limits = param_limits)
        self.gps = [gp(i) for i in range(self.nz * self.nk)]

    def predict(self,params, tau0_factors = None):
        """Get the predicted flux at a parameter value (or list of parameter values)."""
        std = np.zeros([1,self.nk*self.nz])
        means = np.zeros([1,self.nk*self.nz])
        for i, gp in enumerate(self.gps):
            #Adjust the slope of the mean flux for this bin
            zparams = np.array(params)
            if tau0_factors is not None:
                #Make sure we use optical depth from the right bin: we want integer division.
                zparams[0][0] *= tau0_factors[i//self.nk]
            (m, s) = gp.predict(zparams)
            means[0,i] = m
            std[:,i] = s
        return means, std

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
        normspectra = powers/self.scalefactors -1.
        #Get the flux power and build an emulator
        self._get_interp(flux_vectors=normspectra)
        if self._test_interp:
            self._check_interp(powers)
            self._test_interp = False

    def _get_interp(self, flux_vectors):
        """Build the actual interpolator."""
        #Map the parameters onto a unit cube so that all the variations are similar in magnitude
        nparams = np.shape(self.params)[1]
        params_cube = np.array([map_to_unit_cube(pp, self.param_limits) for pp in self.params])
        #Standard squared-exponential kernel with a different length scale for each parameter, as
        #they may have very different physical properties.
        kernel = GPy.kern.Linear(nparams)
        kernel += GPy.kern.RBF(nparams)
        noutput = np.shape(flux_vectors)[1]
        self.gp = GPy.models.GPRegression(params_cube, flux_vectors,kernel=kernel, noise_var=1e-10)
        self.gp.optimize(messages=False)
        #print(kernel)

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

class SkLearnGP(GPyGP):
    """Optimize the emulator with scikit-learn instead"""
    def _get_interp(self, flux_vectors):
        """Build the actual interpolator."""
        #Map the parameters onto a unit cube so that all the variations are similar in magnitude
        nparams = np.shape(self.params)[1]
        params_cube = np.array([map_to_unit_cube(pp, self.param_limits) for pp in self.params])
        #Standard squared-exponential kernel with a different length scale for each parameter, as
        #they may have very different physical properties.
        kernel = (SKGP.kernels.DotProduct(sigma_0_bounds=(0.1,100)) + 1**2 * SKGP.kernels.RBF(length_scale_bounds=(0.1, 1000.0)))
        #kernel *= 1**2
        #kernel += SKGP.kernels.WhiteKernel(1e-8,  noise_level_bounds = (1e-10, 1e-5))
        noutput = np.shape(flux_vectors)[1]
        self.gp = SKGP.GaussianProcessRegressor(kernel=kernel, alpha=0, normalize_y=False, copy_X_train=False, n_restarts_optimizer=100)
        self.gp.fit(params_cube, flux_vectors)
        print(self.gp.kernel_)

    def predict(self, params):
        """Get the predicted flux at a parameter value (or list of parameter values)."""
        #Map the parameters onto a unit cube so that all the variations are similar in magnitude
        params_cube = np.array([map_to_unit_cube(pp, self.param_limits) for pp in params])
        #There is a return_std option, which gives a different answer.
        flux_predict, var = self.gp.predict(params_cube, return_cov=True)
        std = np.sqrt(var)
        if std[0] < 1e-2:
            std[0] = 1e-2
        return (flux_predict[0] + 1) * self.scalefactors, std * self.scalefactors

