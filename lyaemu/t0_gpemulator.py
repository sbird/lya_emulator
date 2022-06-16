"""Building a surrogate using a Gaussian Process."""
# from datetime import datetime
import copy as cp
import numpy as np
from .latin_hypercube import map_to_unit_cube,map_to_unit_cube_list
import GPy

class T0MultiBinGP(MultiBinGP):
    """A wrapper around the emulator that constructs a separate emulator for each bin.
    Each one has a separate mean flux parameter.
    The t0 parameter fed to the emulator should be constant factors."""
    def __init__(self, *, params, kf, powers, param_limits, singleGP=None):
        #Build an emulator for each redshift separately. This means that the
        #mean flux for each bin can be separated.
        if singleGP is None:
            singleGP = SkLearnGP
        self.kf = kf
        self.nk = np.size(kf)
        assert np.shape(powers)[1] % self.nk == 0
        self.nz = int(np.shape(powers)[1]/self.nk)
        gp = lambda i: singleGP(params=params, powers=powers[:,i*self.nk:(i+1)*self.nk], param_limits = param_limits)
        print('Number of redshifts for emulator generation=%d nk= %d' % (self.nz, self.nk))
        self.gps = [gp(i) for i in range(self.nz)]
        self.powers = powers
        self.params = params

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

    def get_training_data(self):
        """Get the originally input training data so we can easily rebuild the GP"""
        return self.params, self.kf, self.powers

    def add_to_training_set(self, new_params):
        """Add to training set and update emulator (without re-training) -- for all redshifts"""
        # not implemented

class T0GP(SkLearnGP):
    """An emulator wrapping a GP code.
       Parameters: params is a list of parameter vectors.
                   powers is a list of flux power spectra (same shape as params).
                   param_limits is a list of parameter limits (shape 2,params)."""
    def __init__(self, *, params, powers,param_limits):
        self.params = params
        self.param_limits = param_limits
        self.intol = 1e-4
        #Get the flux power and build an emulator
        self._get_interp(flux_vectors=powers)


    def _get_interp(self, flux_vectors):
        """Build the actual interpolator."""
        #Map the parameters onto a unit cube so that all the variations are similar in magnitude
        nparams = np.shape(self.params)[1]
        params_cube = map_to_unit_cube_list(self.params, self.param_limits)
        #Check that we span the parameter space
        for i in range(nparams):
            assert np.max(params_cube[:,i]) > 0.8
            assert np.min(params_cube[:,i]) < 0.2
        #Normalise the flux vectors by the median power spectrum.
        #This ensures that the GP prior (a zero-mean input) is close to true.
        medind = np.argsort(np.mean(flux_vectors, axis=1))[np.shape(flux_vectors)[0]//2]
        self.scalefactors = flux_vectors[medind,:]
        self.paramzero = params_cube[medind,:]
        #Normalise by the median value
        normspectra = flux_vectors/self.scalefactors -1.

        # Standard squared-exponential kernel with a different length scale for each
        # parameter, as they may have very different physical properties.
        kernel = GPy.kern.Linear(nparams)
        kernel += GPy.kern.RBF(nparams)

        self.gp = GPy.models.GPRegression(params_cube, normspectra,kernel=kernel, noise_var=1e-10)

        status = self.gp.optimize(messages=False) #True
        #Let's check that hyperparameter optimisation is converged
        if status.status != 'Converged':
            print("Restarting optimization")
            self.gp.optimize_restarts(num_restarts=10)

    def predict(self, params):
        """Get the predicted flux at a parameter value (or list of parameter values)."""
        #Map the parameters onto a unit cube so that all the variations are similar in magnitude
        params_cube = map_to_unit_cube_list(params, self.param_limits)
        flux_predict, var = self.gp.predict(params_cube)
        mean = (flux_predict+1)*self.scalefactors
        std = np.sqrt(var) * self.scalefactors
        return mean, std

    def get_predict_error(self, test_params, test_exact):
        """Get the difference between the predicted GP
        interpolation and some exactly computed test parameters."""
        # not implemented

    def add_to_training_set(self, new_params):
        """Add to training set and update emulator (without re-training). Takes a single set of new parameters"""
        # not implemented

    def _check_interp(self, flux_vectors):
        """Check we reproduce the input"""
        # not implemented
