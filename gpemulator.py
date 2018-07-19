"""Building a surrogate using a Gaussian Process."""
# from datetime import datetime
import numpy as np
from latin_hypercube import map_to_unit_cube
#Make sure that we don't accidentally
#get another backend when we import GPy.
import matplotlib
matplotlib.use('PDF')
import GPy

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
        gp = lambda i: SkLearnGP(params=params, powers=powers[:,i*self.nk:(i+1)*self.nk], param_limits = param_limits)
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
    """An emulator using the one in Scikit-learn.
       Parameters: params is a list of parameter vectors.
                   powers is a list of flux power spectra (same shape as params).
                   param_limits is a list of parameter limits (shape 2,params)."""
    def __init__(self, *, params, powers,param_limits):
        self.params = params
        self.param_limits = param_limits
        self.intol = 3e-5
        #Should we test the built emulator?
        self._test_interp = True
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
        #print('Normalised parameter values =', params_cube)
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
        print(kernel)
        kernel += GPy.kern.RBF(nparams)

        #Try rational quadratic kernel
        #kernel += GPy.kern.RatQuad(nparams)

        noutput = np.shape(normspectra)[1]
        self.gp = GPy.models.GPRegression(params_cube, normspectra,kernel=kernel, noise_var=1e-10)

        self.gp.optimize(messages=False) #True
        print(self.gp)
        #print('Gradients of model hyperparameters [after optimisation] =', self.gp.gradient)
        #Let's check that hyperparameter optimisation is converged
        #self.gp.optimize_restarts(num_restarts=10)
        #print(self.gp)
        #print('Gradients of model hyperparameters [after second optimisation (x 10)] =', self.gp.gradient)

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
        mean = (flux_predict+1)*self.scalefactors
        std = np.sqrt(var) * self.scalefactors
        return mean, std

    def get_predict_error(self, test_params, test_exact):
        """Get the difference between the predicted GP
        interpolation and some exactly computed test parameters."""
        #Note: this is not used anywhere
        test_exact = test_exact.reshape(np.shape(test_params)[0],-1)
        predict, sigma = self.predict(test_params)
        return (test_exact - predict)/sigma
