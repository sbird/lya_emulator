"""Building a surrogate using a Gaussian Process."""
# from datetime import datetime
import copy as cp
import numpy as np
import os
import json
from .latin_hypercube import map_to_unit_cube,map_to_unit_cube_list
#Make sure that we don't accidentally
#get another backend when we import GPy.
import matplotlib
matplotlib.use('PDF')
import GPy

class MultiBinGP:
    """A wrapper around the emulator that constructs a separate emulator for each redshift bin.
    Each one has a separate mean flux parameter.
    The t0 parameter fed to the emulator should be constant factors."""
    def __init__(self, *, params, kf, powers, param_limits, zout, singleGP=None, traindir=None):
        #Build an emulator for each redshift separately. This means that the
        #mean flux for each bin can be separated.
        if singleGP is None:
            singleGP = SkLearnGP
        self.kf, self.nk = kf, np.size(kf)
        assert np.shape(powers)[1] % self.nk == 0
        self.nz = zout.size
        gp = lambda i: singleGP(params=params, powers=powers[:,i*self.nk:(i+1)*self.nk], param_limits=param_limits, traindir=traindir, zbin=zout[i])
        print('Number of redshifts for emulator generation=%d nk= %d' % (self.nz, self.nk))
        self.gps = [gp(i) for i in range(self.nz)]
        self.powers = powers
        self.params = params

    def predict(self, params, tau0_factors=None):
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
        for i in range(self.nz): #Loop over redshifts
            self.gps[i].add_to_training_set(new_params)

class SkLearnGP:
    """An emulator wrapping a GP code.
       Parameters: params is a list of parameter vectors.
                   powers is a list of flux power spectra (same shape as params).
                   param_limits is a list of parameter limits (shape 2,params)."""
    def __init__(self, *, params, powers, param_limits, zbin, traindir=None):
        self.params = params
        self.param_limits = param_limits
        self.intol = 1e-4
        self.traindir = traindir
        self.zbin = np.round(zbin, 1)
        #Should we test the built emulator?
        #Turn this off because our emulator is now so large
        #that it always fails because of Gaussianity!
        self._test_interp = False
        #Get the flux power and build an emulator
        self._get_interp(flux_vectors=powers)
        self.gp_updated = None
        if self._test_interp:
            self._check_interp(powers)
            self._test_interp = False

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

        #Standard squared-exponential kernel with a different length scale for each parameter, as
        #they may have very different physical properties.
        kernel = GPy.kern.Linear(nparams)
        kernel += GPy.kern.RBF(nparams)
        self.gp = GPy.models.GPRegression(params_cube, normspectra,kernel=kernel, noise_var=1e-10)
        try:
            zbin_file = os.path.join(os.path.abspath(self.traindir), 'zbin'+str(self.zbin))
            load_gp = json.load(open(zbin_file+'.json', 'r'))
            self.gp.from_dict(load_gp)
            print('Loading pre-trained GP for z:'+str(self.zbin))
        except:
            print('Optimizing GP for z:'+str(self.zbin))
            status = self.gp.optimize(messages=False) #True
            #print('Gradients of model hyperparameters [after optimisation] =', self.gp.gradient)
            #Let's check that hyperparameter optimisation is converged
            if status.status != 'Converged':
                print("Restarting optimization")
                self.gp.optimize_restarts(num_restarts=10)
            #print('Gradients of model hyperparameters [after second optimisation (x 10)] =', self.gp.gradient)
            if self.traindir != None: # if a traindir was requested, but not populated
                print('Saving GP to', zbin_file)
                if not os.path.exists(self.traindir): os.makedirs(self.traindir)
                with open(zbin_file+'.json', 'w') as jfile:
                    json.dump(self.gp.to_dict(), jfile)

    def _check_interp(self, flux_vectors):
        """Check we reproduce the input"""
        for i, pp in enumerate(self.params):
            means, _ = self.predict(pp.reshape(1,-1))
            worst = np.abs(np.array(means) - flux_vectors[i,:])/self.scalefactors
            if np.max(worst) > self.intol:
                print("Bad interpolation at:", np.where(worst > np.max(worst)*0.9), np.max(worst))
                assert np.max(worst) < self.intol

    def add_to_training_set(self, new_params):
        """Add to training set and update emulator (without re-training). Takes a single set of new parameters"""
        gp_new = cp.deepcopy(self.gp)
        mean_flux_training_samples = np.unique(self.gp.X[:, 0]).reshape(-1, 1)
        #Note mean flux excluded from param_limits
        new_params_unit_cube = map_to_unit_cube(new_params, self.param_limits[-1 * new_params.shape[0]:])
        new_params_unit_cube_expand = np.tile(new_params_unit_cube, (mean_flux_training_samples.shape[0], 1))
        new_params_unit_cube_mean_flux = np.hstack((mean_flux_training_samples, new_params_unit_cube_expand))
        #new_params_mean_flux = map_from_unit_cube_list(new_params_unit_cube_mean_flux, self.param_limits)
        gp_updated_X_new = np.vstack((gp_new.X, new_params_unit_cube_mean_flux))
        gp_updated_Y_new = np.vstack((gp_new.Y, self.gp.predict(new_params_unit_cube_mean_flux)[0]))
        gp_new.set_XY(X=gp_updated_X_new, Y=gp_updated_Y_new)
        self.gp = gp_new

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
        #Note: this is not used anywhere
        test_exact = test_exact.reshape(np.shape(test_params)[0],-1)
        predict, sigma = self.predict(test_params)
        return (test_exact - predict)/sigma
