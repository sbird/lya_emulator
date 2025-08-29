"""Building a surrogate using a Gaussian Process."""
import numpy as np
import os
from .latin_hypercube import map_to_unit_cube_list
import torch
import gpytorch
import gpytorch.kernels as kern

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

class LinearMultiFidelityKernel(gpytorch.kernels.Kernel):
    """
    Linear Multi-Fidelity Kernel (Kennedy & O’Hagan, 2000).

    This kernel models the high-fidelity function as:

        f_H(x) = ρ * f_L(x) + δ(x)

    where:
        - f_L(x) is a Gaussian process modeling the low-fidelity function.
        - δ(x) is an independent GP modeling discrepancies.
        - ρ is a learnable scaling factor.

    The covariance matrix has a block structure:

        K =
        [  K_LL   K_LH  ]
        [  K_HL   K_HH  ]

    where:
        - K_LL = Covariance matrix for low-fidelity points.
        - K_LH = K_HL^T = Scaled cross-covariance.
        - K_HH = Scaled LF + discrepancy covariance.

    Parameters:
        - kernel_L: GP kernel for the low-fidelity function.
        - kernel_delta: GP kernel for the discrepancy.
        - num_output_dims: The number of independent outputs (Y.shape[1]).
    """
    # The multi-fidelity kernels depend only on the difference
    # between two parameters and not their absolute values.
    is_stationary = True

    def __init__(self, kernel_L, kernel_delta, num_output_dims=1, batch_shape=torch.Size([]), active_dims=None, **kwargs):
        super().__init__(batch_shape=batch_shape, active_dims=active_dims, **kwargs)

        self.kernel_L = kernel_L  # Kernel for low-fidelity function
        self.kernel_delta = kernel_delta  # Kernel for discrepancy

        self.num_output_dims = num_output_dims

        #Should have shape ((num_output_dims, 1) so one rho for each output dimension
        self.register_parameter(
            name='raw_rho', parameter=torch.nn.Parameter(torch.ones(*self.batch_shape, num_output_dims, 1))
        )

        # register the constraint
        self.register_constraint("raw_rho", gpytorch.constrains.Positive())

    # now set up the 'actual' parameter
    @property
    def rho(self):
        # when accessing the parameter, apply the constraint transform
        return self.raw_rho_constraint.transform(self.raw_rho)

    @rho.setter
    def rho(self, value):
        return self._set_rho(value)

    def _set_rho(self, value):
        if not torch.is_tensor(value):
            value = torch.as_tensor(value).to(self.raw_rho)
        # when setting the paramater, transform the actual value to a raw one by applying the inverse transform
        self.initialize(raw_rho=self.raw_rho_constraint.inverse_transform(value))

    def forward(self, x1, x2, diag=False, last_dim_is_batch=False, **params):
        """
        Constructs the full covariance matrix for multi-fidelity modeling.

        Args:
            x1: First input tensor (n1 x d+1), where last column is fidelity indicator
            x2: Second input tensor (n2 x d+1), where last column is fidelity indicator
            diag: If True, return only diagonal elements
            last_dim_is_batch: If True, treat the last dimension as batch dimension
            **params: Additional parameters

        Returns:
            Covariance matrix or diagonal elements
        """
        if last_dim_is_batch:
            raise NotImplementedError("Batch mode not implemented")

        if diag:
            return self._diag(x1)

        # Ensure x2 is not None
        if x2 is None:
            x2 = x1

        # Extract fidelity indicators (last column)
        fidelity_1 = x1[..., -1]
        fidelity_2 = x2[..., -1]

        # Create masks for low and high fidelity points
        mask_L1 = fidelity_1 == 0
        mask_H1 = fidelity_1 == 1
        mask_L2 = fidelity_2 == 0
        mask_H2 = fidelity_2 == 1

        # Extract data without fidelity column
        x1_data = x1[..., :-1]
        x2_data = x2[..., :-1]

        # Extract LF and HF data
        x1_L = x1_data[mask_L1]
        x1_H = x1_data[mask_H1]
        x2_L = x2_data[mask_L2]
        x2_H = x2_data[mask_H2]

        # Initialize full covariance matrix
        n1 = x1.shape[0]
        n2 = x2.shape[0]
        K_full = torch.zeros(n1, n2, dtype=x1.dtype, device=x1.device)

        # Compute covariance components
        if x1_L.numel() > 0 and x2_L.numel() > 0:
            K_LL = self.kernel_L(x1_L, x2_L).evaluate()
            # Use advanced indexing to place values
            idx1 = torch.where(mask_L1)[0].unsqueeze(1)
            idx2 = torch.where(mask_L2)[0].unsqueeze(0)
            K_full[idx1, idx2] = K_LL

        if x1_L.numel() > 0 and x2_H.numel() > 0:
            K_LH = self.kernel_L(x1_L, x2_H).evaluate() * self.rho
            idx1 = torch.where(mask_L1)[0].unsqueeze(1)
            idx2 = torch.where(mask_H2)[0].unsqueeze(0)
            K_full[idx1, idx2] = K_LH

        if x1_H.numel() > 0 and x2_L.numel() > 0:
            K_HL = self.kernel_L(x1_H, x2_L).evaluate() * self.rho
            idx1 = torch.where(mask_H1)[0].unsqueeze(1)
            idx2 = torch.where(mask_L2)[0].unsqueeze(0)
            K_full[idx1, idx2] = K_HL

        if x1_H.numel() > 0 and x2_H.numel() > 0:
            K_HH_L = self.kernel_L(x1_H, x2_H).evaluate() * (self.rho ** 2)
            K_HH_delta = self.kernel_delta(x1_H, x2_H).evaluate()
            K_HH = K_HH_L + K_HH_delta
            idx1 = torch.where(mask_H1)[0].unsqueeze(1)
            idx2 = torch.where(mask_H2)[0].unsqueeze(0)
            K_full[idx1, idx2] = K_HH

        return K_full

    def _diag(self, x):
        """
        Computes the diagonal elements of the covariance matrix.

        Args:
            x: Input tensor (n x d+1), where last column is fidelity indicator

        Returns:
            Diagonal elements of the covariance matrix
        """
        # Extract fidelity indicators
        fidelity = x[..., -1]

        # Create masks
        mask_L = fidelity == 0
        mask_H = fidelity == 1

        # Extract data without fidelity column
        x_data = x[..., :-1]
        x_L = x_data[mask_L]
        x_H = x_data[mask_H]

        # Initialize diagonal vector
        n = x.shape[0]
        K_diag_full = torch.zeros(n, dtype=x.dtype, device=x.device)

        # Compute diagonal elements for low-fidelity points
        if x_L.numel() > 0:
            K_diag_L = self.kernel_L(x_L, x_L, diag=True).evaluate()
            K_diag_full[mask_L] = K_diag_L

        # Compute diagonal elements for high-fidelity points
        if x_H.numel() > 0:
            K_diag_H_L = self.kernel_L(x_H, x_H, diag=True).evaluate() * (self.rho ** 2)
            K_diag_H_delta = self.kernel_delta(x_H, x_H, diag=True).evaluate()
            K_diag_H = K_diag_H_L + K_diag_H_delta
            K_diag_full[mask_H] = K_diag_H

        return K_diag_full

class ExactGPAR1(gpytorch.models.ExactGP):
    """Subclass the exact inference GP with the kernel we want."""
    def __init__(self, train_x, train_y, likelihood, use_ar1_kernel=False):
        super().__init__(train_x, train_y, likelihood)
        #Standard squared-exponential kernel with a different length scale for each parameter, as
        #they may have very different physical properties.
        nparam = np.shape(train_x)[1]
        #Each dimension of the output vector is called a task in GPyTorch
        ntask = np.shape(train_y)[0]
        self.mean_module = gpytorch.means.MultitaskMean(gpytorch.means.ConstantMean(), num_tasks=ntask)
        if use_ar1_kernel:
            kernel_l = kern.LinearKernel(ard_num_dims=nparam) + kern.ScaleKernel(kern.RBFKernel(ard_num_dims=nparam))
            kernel_delta = kern.RBFKernel()
            singletaskkernel = LinearMultiFidelityKernel(kernel_l, kernel_delta)
        else:
            singletaskkernel = kern.LinearKernel(ard_num_dims=nparam) + kern.ScaleKernel(kern.RBFKernel(ard_num_dims=nparam))
        self.covar_module = kern.MultitaskKernel(singletaskkernel, num_tasks=ntask)

    def forward(self, x):
        """Takes n x d data (where d is the number of input parameters and n is the number of outputs)
        and returns the prior mean and covariance of the GP."""
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultitaskMultivariateNormal(mean_x, covar_x)

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
        ntasks = np.shape(flux_vectors)[0]
        medind = np.argsort(np.mean(flux_vectors, axis=1))[ntasks//2]
        self.scalefactors = flux_vectors[medind,:]
        self.paramzero = params_cube[medind,:]
        #Normalise by the median value
        normspectra = flux_vectors/self.scalefactors -1.
        self.likelihood = gpytorch.likelihoods.MultitaskGaussianLikelihood(num_tasks = ntasks, noise_constraint=gpytorch.constraints.GreaterThan(1e-10))
        self.gp = ExactGPModel(params_cube, normspectra, self.likelihood)
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
