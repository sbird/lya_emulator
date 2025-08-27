"""Building a surrogate using a Gaussian Process."""
import numpy as np
import os
import h5py
from .latin_hypercube import map_to_unit_cube_list
import torch
import gpytorch
from gpytorch.kernels import RBFKernel, LinearKernel

class MultiBinGP:
    """A wrapper around the emulator that constructs a separate emulator for each redshift bin.
    Each one has a separate mean flux parameter.
        - The t0 parameter fed to the emulator should be constant factors.
        - To use multi-fidelity, pass a list to HRdat where the first entry is the high resolution parameter set, and the second entry is the associated high resolution flux power spectra.
        - Passing a directory for traindir will either load previously trained GP (if one exists),
        or train a GP and save it to that directory."""
    def __init__(self, *, params, kf, powers, param_limits, zout, HRdat=None, traindir=None):
        #Build an emulator for each redshift separately. This means that the
        #mean flux for each bin can be separated.
        self.kf, self.nk = kf, np.size(kf)
        assert np.shape(powers)[1] % self.nk == 0
        self.nz = zout.size
        if HRdat is None:
            gp = lambda i: GaussianProcess(params=params, powers=powers[:,i*self.nk:(i+1)*self.nk], param_limits=param_limits, traindir=traindir, zbin=zout[i])
        else:
            gp = lambda i: SingleBinAR1(LRparams=params, HRparams=HRdat[0], LRfps=powers[:,i*self.nk:(i+1)*self.nk], HRfps=HRdat[1][:,i*self.nk:(i+1)*self.nk], param_limits=param_limits, traindir=traindir, zbin=zout[i])
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

class ExactGPModel(gpytorch.models.ExactGP):
    """Subclass the exact inference GP with the kernel we want."""
    def __init__(self, train_x, train_y, likelihood):
        super(ExactGPModel, self).__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ConstantMean()
        #Standard squared-exponential kernel with a different length scale for each parameter, as
        #they may have very different physical properties.
        self.covar_module = LinearKernel() + RBFKernel()
        #What is a ScaleKernel?
#        self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel())

    def forward(self, x):
        """Takes n x d data (where d is the number of input parameters and n is the number of outputs)
        and returns the prior mean and covariance of the GP."""
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

class GaussianProcess:
    """An emulator wrapping a GP code.
       Parameters: params is a list of parameter vectors (shape nsims,params).
                   powers is a list of flux power spectra (shape nsims,nk).
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

    def _get_interp(self, flux_vectors, training_iter=50):
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
        self.likelihood = gpytorch.likelihoods.GaussianLikelihood()
        self.gp = ExactGPModel(params_cube, normspectra)
        #kernel = GPy.kern.Linear(nparams, ARD=True) + GPy.kern.RBF(nparams, ARD=True)
        #self.gp = GPy.models.GPRegression(params_cube, normspectra,kernel=kernel, noise_var=1e-10)
        #Save file for this model
        zbin_file = os.path.join(os.path.abspath(self.traindir), 'zbin'+str(self.zbin)+".pth")
        # try to load previously saved trained GPs
        if os.path.exists(zbin_file):
            state_dict = torch.load(zbin_file)
            self.gp.load_state_dict(state_dict)
            print('Loading pre-trained GP for z:'+str(self.zbin))
        # need to train from scratch
        else:
            print('Optimizing GP for z:'+str(self.zbin))
            # Find optimal model hyperparameters
            self.gp.train()
            self.likelihood.train()

            # Use the adam optimizer
            optimizer = torch.optim.Adam(self.gp.parameters(), lr=0.1)  # Includes GaussianLikelihood parameters

            # "Loss" for GPs - the marginal log likelihood
            mll = gpytorch.mlls.ExactMarginalLogLikelihood(self.likelihood, self.gp)
            # Training loop
            for i in range(training_iter):
                # Zero gradients from previous iteration
                optimizer.zero_grad()
                # Output from model
                output = self.gp(params_cube)
                # Calc loss and backprop gradients
                loss = -mll(output, normspectra)
                loss.backward()
                print('Iter %d/%d - Loss: %.3f   lengthscale: %.3f   noise: %.3f' % (
                    i + 1, training_iter, loss.item(),
                    self.gp.covar_module.base_kernel.lengthscale.item(),
                    self.gp.likelihood.noise.item()
                ))
                optimizer.step()

            if self.traindir is not None: # if a traindir was requested, but not populated, save it
                print('Saving GP to', zbin_file)
                if not os.path.exists(self.traindir):
                    os.makedirs(self.traindir)
                torch.save(self.gp.state_dict(), zbin_file)

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
        params_cube = map_to_unit_cube_list(params, self.param_limits)
        #This is a distribution over a function f
        f_predicts = self.gp(params_cube)
        #Need the mean and variance
        flux_predict, var = f_predicts.mean, f_predicts.variance
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


class SingleBinAR1:
    """
    A wrapper around GPy that constructs a multi-fidelity emulator for the flux power spectrum (single redshift).
    Parameters: LRparams, HRparams are the input parameter sets (nsims, nparams).
                LRfps, HRfps are the corresponding flux powers (nsims, nk).
                param_limits is a list of parameter limits (nparams, 2).
                zbin is the redshift of the input flux power.
                traindir is the directory to load/save the trained GP.
                n_fidelities is the number of fidelities.
                n_restarts is the number of optimization restarts.
    """

    def __init__(self, LRparams, HRparams, LRfps, HRfps, param_limits, zbin, traindir=None, n_fidelities=2, n_restarts=10):
        self.zbin = np.round(zbin, 1)
        self.traindir = traindir
        self.n_fidelities = n_fidelities
        self.param_limits = param_limits
        # assert that the two sets have the same number of parameters and k-bins
        assert np.shape(LRparams)[1] == np.shape(HRparams)[1]
        assert np.shape(LRfps)[1] == np.shape(HRfps)[1]
        self.nparams = np.shape(LRparams)[1]
        self.nk = np.shape(LRfps)[1]

        # get parameters into correct format
        param_cube = [map_to_unit_cube_list(LRparams, param_limits), map_to_unit_cube_list(HRparams, param_limits)]
        # ensure that the GP prior (a zero-mean input) is close to true.
        medind = np.argsort(np.mean(LRfps, axis=1))[np.shape(LRfps)[0]//2]
        self.scalefactors = LRfps[medind,:]
        LRnormfps = LRfps/self.scalefactors - 1.
        HRnormfps = HRfps/self.scalefactors - 1.
        # this also adds the fidelity flag: 0 for LR, 1 for HR
        params, normfps = convert_xy_lists_to_arrays(param_cube, [LRnormfps, HRnormfps])

        kernel_list = []
        for j in range(n_fidelities):
            # Standard squared-exponential kernel with a different length scale for
            # each parameter, as they may have very different physical properties.
            kernel = GPy.kern.Linear(self.nparams, ARD=True)
            kernel += GPy.kern.RBF(self.nparams, ARD=True)
            # final fidelity not ARD due to lack of training data
            if j == n_fidelities - 1:
                kernel = GPy.kern.Linear(self.nparams, ARD=False)
                kernel += GPy.kern.RBF(self.nparams, ARD=False)
            kernel_list.append(kernel)
        # make multi-fidelity kernels
        kernel = LinearMultiFidelityKernel(kernel_list)

        # Make default likelihood as different noise for each fidelity
        likelihood = GPy.likelihoods.mixed_noise.MixedNoise([GPy.likelihoods.Gaussian(variance=1.0) for _ in range(n_fidelities)])
        y_metadata = {"output_index": params[:, -1].astype(int)}

        self.gpy_models = GPy.core.GP(params, normfps, kernel, likelihood, Y_metadata=y_metadata)
        self.optimize(n_restarts)


    def optimize(self, n_optimization_restarts):
        # fix noise
        getattr(self.gpy_models.mixed_noise, "Gaussian_noise").fix(1e-6)
        for j in range(1, self.n_fidelities):
            getattr(self.gpy_models.mixed_noise, "Gaussian_noise_{}".format(j)).fix(1e-6)
        model = GPyMultiOutputWrapper(self.gpy_models, n_outputs=self.n_fidelities, n_optimization_restarts=n_optimization_restarts)

        # attempt to load previously trained GP
        # this is less than ideal -- the GPy package may have better methods in future updates
        try:
            zbin_file = os.path.join(os.path.abspath(self.traindir), 'zbin'+str(self.zbin))
            model = self.load_GP(model, zbin_file)
            print('Loading pre-trained GP for z:'+str(self.zbin))
            self.models = model
        except:
            print('Optimizing GP for z:'+str(self.zbin))
            # first step optimization with fixed noise
            model.gpy_model.optimize_restarts(n_optimization_restarts, verbose=model.verbose_optimization, robust=True, parallel=False,)

            # unfix noise and re-optimize
            getattr(model.gpy_model.mixed_noise, "Gaussian_noise").unfix()
            for j in range(1, self.n_fidelities):
                getattr(model.gpy_model.mixed_noise, "Gaussian_noise_{}".format(j)).unfix()
            model.gpy_model.optimize_restarts(n_optimization_restarts, verbose=model.verbose_optimization, robust=True, parallel=False,)
            self.models = model

            # save trained GP, if not already
            # this is less than ideal -- the GPy package may have better methods in future updates
            if self.traindir != None: # if a traindir was requested, but not populated
                print('Saving GP to', zbin_file)
                if not os.path.exists(self.traindir): os.makedirs(self.traindir)
                model.gpy_model.save(zbin_file)

    def predict(self, params, res=1):
        """
        Predicts mean and variance for fidelity specified by last column of params.
        params is the point at which to predict.
        """
        assert res == 0 or res == 1
        params_cube = map_to_unit_cube_list(params.reshape(1, -1), self.param_limits)
        params_cube = np.concatenate([params_cube[0], np.ones(1)*res]).reshape(1,-1)
        fps_predict, var = self.models.predict(params_cube)
        mean = (fps_predict+1) * self.scalefactors
        std = np.sqrt(var) * self.scalefactors
        return mean, std

    def load_GP(self, model, zbin_file):
        """
        Attempt to load a previously trained GP, given the model setup and file location.
        """
        trained = h5py.File(zbin_file, 'r')
        # directory structure is odd, so there isn't a clean way to do this -- just
        # have to go through each part and update the values
        model.gpy_model.mixed_noise.Gaussian_noise_1.variance = trained['mixed_noise_Gaussian_noise_1_variance'][:]
        model.gpy_model.mixed_noise.Gaussian_noise.variance = trained['mixed_noise_Gaussian_noise_variance'][:]
        model.gpy_model.multifidelity.scale = trained['multifidelity_scale'][:]
        model.gpy_model.multifidelity.sum_1.linear.variances = trained['multifidelity_sum_1_linear_variances'][:]
        model.gpy_model.multifidelity.sum_1.rbf.lengthscale = trained['multifidelity_sum_1_rbf_lengthscale'][:]
        model.gpy_model.multifidelity.sum_1.rbf.variance = trained['multifidelity_sum_1_rbf_variance'][:]
        model.gpy_model.multifidelity.sum.linear.variances = trained['multifidelity_sum_linear_variances'][:]
        model.gpy_model.multifidelity.sum.rbf.lengthscale = trained['multifidelity_sum_rbf_lengthscale'][:]
        model.gpy_model.multifidelity.sum.rbf.variance = trained['multifidelity_sum_rbf_variance'][:]
        return model
