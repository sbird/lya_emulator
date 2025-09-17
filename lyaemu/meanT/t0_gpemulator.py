"""Building a surrogate using a Gaussian Process."""
import numpy as np
from ..latin_hypercube import map_to_unit_cube_list
from ..gpemulator import convert_parameter_fidelity_list, LinearMultiFidelityKernel
import torch
import gpytorch
import gpytorch.kern as kern

class ExactGPAR1(gpytorch.models.ExactGP):
    """Subclass the exact inference GP with the kernel we want."""
    def __init__(self, train_x, train_y, likelihood, use_ar1_kernel=False):
        super().__init__(train_x, train_y, likelihood)
        #Standard squared-exponential kernel with a different length scale for each parameter, as
        #they may have very different physical properties.
        nparam = np.shape(train_x)[1]
        self.mean_module = gpytorch.means.ConstantMean()
        if use_ar1_kernel:
            #The final dimension is the flag specifying the fidelity.
            kernel_l = kern.LinearKernel(ard_num_dims=nparam-1) + kern.ScaleKernel(kern.RBFKernel(ard_num_dims=nparam-1))
            kernel_delta = kern.RBFKernel()
            self.covar_module = LinearMultiFidelityKernel(kernel_l, kernel_delta)
        else:
            self.covar_module = kern.LinearKernel(ard_num_dims=nparam) + kern.ScaleKernel(kern.RBFKernel(ard_num_dims=nparam))

    def forward(self, x):
        """Takes n x d data (where d is the number of input parameters and n is the number of outputs)
        and returns the prior mean and covariance of the GP."""
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

class T0MultiBinAR1:
    """
    A wrapper that constructs a multi-fidelity emulator for the mean temperature over all redshifts.
    Parameters: LRparams, HRparams are the input parameter sets (nsims, nparams)
                LRtemps, HRtemps are the corresponding temperatures (nsims, nz)
                param_limits is a list of parameter limits (nparams, 2)
    """

    def __init__(self, LRparams, HRparams, LRtemps, HRtemps, param_limits, training_iter=50):
        self.param_limits = param_limits
        # assert that the two sets have the same number of parameters and redshifts
        assert np.shape(LRparams)[1] == np.shape(HRparams)[1]
        assert np.shape(LRtemps)[1] == np.shape(HRtemps)[1]
        self.nparams = np.shape(LRparams)[1]
        self.use_ar1_kernel = (HRparams is not None)

        # get parameters into correct format
        params_cube = map_to_unit_cube_list(LRparams, param_limits)
        # ensure that the GP prior (a zero-mean input) is close to true.
        self.scalefactors = np.mean(LRtemps, axis=0)
        LRnormtemps = LRtemps/self.scalefactors - 1.
        # this also adds the fidelity flag, 0 for LR, 1 for HR
        #Add the HR spectra to the training data
        if self.use_ar1_kernel:
            # assert that the two sets have the same number of parameters and k-bins
            assert np.shape(params_cube)[1] == np.shape(HRparams)[1]
            assert np.shape(LRtemps)[1] == np.shape(HRtemps)[1]
            HRparams_cube = map_to_unit_cube_list(HRparams, self.param_limits)
            HRnormtemps = HRtemps/self.scalefactors - 1.
            # this also adds the fidelity flag: 0 for LR, 1 for HR
            params_cube = convert_parameter_fidelity_list([params_cube, HRparams_cube])
            normtemps = np.concatenate([LRnormtemps, HRnormtemps], axis=0)

        #Convert to tensors
        tense_params_cube = torch.from_numpy(params_cube).float()
        tense_normtemps = torch.from_numpy(normtemps).float()
        # Move to GPU if available
        #device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        #tense_params_cube = tense_params_cube.to(device)
        #tense_normspectra = tense_normspectra.to(device)

        self.likelihood = gpytorch.likelihoods.GaussianLikelihood(noise_constraint=gpytorch.constraints.GreaterThan(1e-6))
        self.gp = ExactGPAR1(tense_params_cube, tense_normtemps, self.likelihood, use_ar1_kernel=self.use_ar1_kernel)

        print('Optimizing temp GP for z:'+str(self.zbin))
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
            loss = -mll(output, tense_normtemps)
            loss.backward()
            print('Iter %d/%d - Loss: %.3f noise: %.3f' % (
                i + 1, training_iter, loss.item(),
                self.gp.likelihood.noise.item()
            ))
            optimizer.step()

        #Get into evaluation mode so we can make predictions
        self.gp.eval()
        self.likelihood.eval()

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
        params_cube = torch.from_numpy(params_cube).float()
        f_predicts = self.gp(params_cube)
        #Need the mean and variance
        temp_predict, var = f_predicts.mean, f_predicts.variance
        mean = (temp_predict.detach().numpy()+1)*self.scalefactors
        std = np.sqrt(var.detach().numpy()) * self.scalefactors
        return mean, std

class T0MultiBinGP(T0MultiBinAR1):
    """A wrapper that constructs an emulator for the mean temperature over all redshifts.
        Parameters: params is a list of parameter vectors.
                    temps is a list of mean temperatures (shape nsims, nz).
                    param_limits is a list of parameter limits (shape params, 2)."""
    def __init__(self, *, params, temps, param_limits):
        super().__init__(LRparams = params, HRparams=None, LRtemps=temps, HRtemps=None, param_limits=param_limits)
        print('Number of redshifts for emulator generation=%d' % (np.shape(temps)[1]))
