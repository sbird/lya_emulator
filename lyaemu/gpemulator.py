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

        def gpredbin(i):
            HRparams = None
            HRpowers = None
            if HRdat is not None:
                HRparams=HRdat[0]
                HRpowers=HRdat[1][:,i*self.nk:(i+1)*self.nk]
            return GaussianProcessAR1(params=params, HRparams=HRparams, powers=powers[:,i*self.nk:(i+1)*self.nk], HRpowers=HRpowers, param_limits=param_limits, traindir=traindir, zbin=zout[i])

        print('Number of redshifts for emulator generation=%d nk= %d' % (self.nz, self.nk))
        self.gps = [gpredbin(i) for i in range(self.nz)]
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
    def __init__(self, train_x, train_y, likelihood, num_tasks=1, use_ar1_kernel=False):
        super().__init__(train_x, train_y, likelihood)
        #Standard squared-exponential kernel with a different length scale for each parameter, as
        #they may have very different physical properties.
        nparam = np.shape(train_x)[1]
        #Each dimension of the output vector is called a task in GPyTorch
        assert num_tasks == np.shape(train_y)[1]
        self.mean_module = gpytorch.means.MultitaskMean(gpytorch.means.ConstantMean(), num_tasks=num_tasks)
        if use_ar1_kernel:
            kernel_l = kern.LinearKernel(ard_num_dims=nparam) + kern.ScaleKernel(kern.RBFKernel(ard_num_dims=nparam))
            kernel_delta = kern.RBFKernel()
            singletaskkernel = LinearMultiFidelityKernel(kernel_l, kernel_delta)
        else:
            singletaskkernel = kern.LinearKernel(ard_num_dims=nparam) + kern.ScaleKernel(kern.RBFKernel(ard_num_dims=nparam))
        self.covar_module = kern.MultitaskKernel(singletaskkernel, num_tasks=num_tasks)

    def forward(self, x):
        """Takes n x d data (where d is the number of input parameters and n is the number of outputs)
        and returns the prior mean and covariance of the GP."""
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultitaskMultivariateNormal(mean_x, covar_x)

def convert_parameter_fidelity_list(x_list):
    """
    Take a list of parameters for the training and convert it to an array
    with the zero-based fidelity index appended as the last column. From emukit
    """
    x_array = np.concatenate(x_list, axis=0)
    indices = []
    for i, x in enumerate(x_list):
        indices.append(i * np.ones((len(x), 1)))
    x_with_index = np.concatenate((x_array, np.concatenate(indices)), axis=1)
    return x_with_index

class GaussianProcessAR1:
    """An emulator wrapping a GP code.
       Parameters: params is a list of parameter vectors (shape nsims,params).
                   powers is a list of flux power spectra (shape nsims,nk).
                   param_limits is a list of parameter limits (shape 2,params)."""
    def __init__(self, *, params, powers, param_limits, zbin, HRparams=None, HRpowers=None, traindir=None, training_iter=50):
        self.params = params
        self.param_limits = param_limits
        self.use_ar1_kernel = (HRparams is not None)
        self.traindir = traindir
        self.zbin = np.round(zbin, 1)

        #Map the parameters onto a unit cube so that all the variations are similar in magnitude
        nparams = np.shape(self.params)[1]
        params_cube = map_to_unit_cube_list(self.params, self.param_limits)

        #Check that we span the parameter space
        for i in range(nparams):
            assert np.max(params_cube[:,i]) > 0.8
            assert np.min(params_cube[:,i]) < 0.2
        #Normalise the flux vectors by the median power spectrum.
        #This ensures that the GP prior (a zero-mean input) is close to true.
        ntasks = np.shape(powers)[1]
        ninput = np.shape(powers)[0]
        medind = np.argsort(np.mean(powers, axis=1))[ninput//2]
        self.scalefactors = powers[medind,:]
        self.paramzero = params_cube[medind,:]
        #Normalise by the median value
        normspectra = powers/self.scalefactors -1.

        #Add the HR spectra to the training data
        if self.use_ar1_kernel:
            # assert that the two sets have the same number of parameters and k-bins
            assert np.shape(params)[1] == np.shape(HRparams)[1]
            assert np.shape(powers)[1] == np.shape(HRpowers)[1]
            HRparams_cube = map_to_unit_cube_list(HRparams, self.param_limits)
            HRnormspectra = HRpowers/self.scalefactors - 1.
            # this also adds the fidelity flag: 0 for LR, 1 for HR
            params_cube = convert_parameter_fidelity_list([params_cube, HRparams_cube])
            normspectra = np.concatenate([normspectra, HRnormspectra], axis=0)

        #Convert to tensors
        tense_params_cube = torch.from_numpy(params_cube).float()
        tense_normspectra = torch.from_numpy(normspectra).float()
        # Move to GPU if available
        #device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        #tense_params_cube = tense_params_cube.to(device)
        #tense_normspectra = tense_normspectra.to(device)

        self.likelihood = gpytorch.likelihoods.MultitaskGaussianLikelihood(num_tasks=ntasks, noise_constraint=gpytorch.constraints.GreaterThan(1e-10))
        self.gp = ExactGPAR1(tense_params_cube, tense_normspectra, self.likelihood, num_tasks=ntasks, use_ar1_kernel=self.use_ar1_kernel)
        #Save file for this model
        zbin_file = 'zbin'+str(self.zbin)
        if self.traindir is not None:
            zbin_file = os.path.join(os.path.abspath(self.traindir), zbin_file)
        if self.use_ar1_kernel:
            zbin_file+="_ar1_"
        zbin_file+=".pth"
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
                output = self.gp(tense_params_cube)
                # Calc loss and backprop gradients
                loss = -mll(output, tense_normspectra)
                loss.backward()
                print('Iter %d/%d - Loss: %.3f noise: %.3f' % (
                    i + 1, training_iter, loss.item(),
                    self.gp.likelihood.noise.item()
                ))
                optimizer.step()

            if self.traindir is not None: # if a traindir was requested, but not populated, save it
                print('Saving GP to', zbin_file)
                if not os.path.exists(self.traindir):
                    os.makedirs(self.traindir)
                torch.save(self.gp.state_dict(), zbin_file)

        #Test the built emulator
        self._check_interp(powers)

    def _check_interp(self, flux_vectors, intol=1e-3):
        """Check we reproduce the input"""
        for i, pp in enumerate(self.params):
            means, _ = self.predict(pp.reshape(1,-1))
            worst = np.abs(np.array(means) - flux_vectors[i,:])/self.scalefactors
            if np.max(worst) > intol:
                print("Bad interpolation at:", np.where(worst > np.max(worst)*0.9), np.max(worst))
                assert np.max(worst) < intol

    def predict(self, params, res=1):
        """
        Predicts mean and variance for fidelity specified by last column of params.
        params is the point at which to predict. res=1 is high fidelity, res=0 is low fidelity
        """
        #Map the parameters onto a unit cube so that all the variations are similar in magnitude
        params_cube = map_to_unit_cube_list(params, self.param_limits)
        #Add the resolution parameter to get a fidelity
        if self.use_ar1_kernel:
            assert res == 0 or res == 1
            params_cube = np.concatenate([params_cube[0], np.ones(1)*res]).reshape(1,-1)
        #This is a distribution over a function f
        f_predicts = self.gp(params_cube)
        #Need the mean and variance
        flux_predict, var = f_predicts.mean, f_predicts.variance
        mean = (flux_predict+1)*self.scalefactors
        std = np.sqrt(var) * self.scalefactors
        return mean, std
