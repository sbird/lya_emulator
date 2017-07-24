"""Building a surrogate using a Gaussian Process."""
import numpy as np
from latin_hypercube import map_to_unit_cube
#Make sure that we don't accidentally
#get another backend when we import GPy.
import matplotlib
matplotlib.use('PDF')
import GPy

class SkLearnGP(object):
    """An emulator using the one in Scikit-learn"""
    def __init__(self, *, params, kf, powers,param_limits, coreg=False):
        self.powers = powers
        self.zout = self.powers[0].zout
        self.params = params
        self.param_limits = param_limits
        self.cur_tau_factor = -1
        self.kf = kf
        self.intol = 3e-5
        self.coreg=coreg
        #In case we need it, we can rescale the errors using cross-validation.
        #self.sdscale = np.mean([self._get_cv_one(exclude) for exclude in range(len(self.powers))])

    def _get_cv_one(self, exclude):
        """Get the prediction error for one point when
        excluding that point from the emulator."""
        self._get_interp(tau0_factor = 1., exclude=exclude)
        test_exact = self.powers[exclude].get_power(kf = self.kf, tau0_factor = 1.)
        return self.get_predict_error(self.params[exclude], test_exact)

    def _get_interp(self, tau0_factor=None, exclude=None):
        """Build the actual interpolator."""
        self.cur_tau_factor = tau0_factor
        flux_vectors = np.array([ps.get_power(kf = self.kf, tau0_factor = tau0_factor) for nn,ps in enumerate(self.powers) if nn is not exclude])
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
        kernel = GPy.kern.Linear(nparams)
        kernel += GPy.kern.RBF(nparams)
        noutput = np.shape(normspectra)[1]
        if self.coreg and noutput > 1:
            coreg = GPy.kern.Coregionalize(input_dim=nparams,output_dim=noutput)
            kernel = kernel.prod(coreg,name='coreg.kern')
        self.gp = GPy.models.GPRegression(params_cube, normspectra,kernel=kernel, noise_var=1e-10)
        self.gp.optimize(messages=True)
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
        flux_predict, var = self.gp.predict(params_cube)
        mean = (flux_predict+1)*self.scalefactors
        std = np.sqrt(var) * self.scalefactors
        return mean, std

    def get_predict_error(self, test_params, test_exact):
        """Get the difference between the predicted GP
        interpolation and some exactly computed test parameters."""
        test_exact = test_exact.reshape(np.shape(test_params)[0],-1)
        predict, sigma = self.predict(test_params,tau0_factor=self.cur_tau_factor)
        #The transposes are because of numpy broadcasting rules only doing the last axis
        return ((test_exact - predict).T/np.sqrt(sigma)).T
