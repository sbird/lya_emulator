"""Building a surrogate using a Gaussian Process."""
import numpy as np
from ..latin_hypercube import map_to_unit_cube, map_to_unit_cube_list
import GPy
from emukit.model_wrappers import GPyMultiOutputWrapper
from emukit.multi_fidelity.kernels import LinearMultiFidelityKernel
from emukit.multi_fidelity.convert_lists_to_array import convert_xy_lists_to_arrays


class T0MultiBinGP:
    """A wrapper around GPy that constructs an emulator for the mean temperature over all redshifts.
        Parameters: params is a list of parameter vectors.
                    temps is a list of mean temperatures (shape nsims, nz).
                    param_limits is a list of parameter limits (shape params, 2)."""
    def __init__(self, *, params, temps, param_limits):
        self.temps = temps
        self.params = params
        self.param_limits = param_limits
        print('Number of redshifts for emulator generation=%d' % (np.shape(temps)[1]))
        self._get_interp(mean_temps=temps)

    def _get_interp(self, mean_temps):
        """Build the GP interpolator."""
        # map the parameters onto a unit cube (so all variations have similar magnitude)
        nparams = np.shape(self.params)[1]
        param_cube = map_to_unit_cube_list(self.params, self.param_limits)
        # ensure that the GP prior (a zero-mean input) is close to true.
        self.scalefactors = np.mean(mean_temps, axis=0)
        normtemps = mean_temps/self.scalefactors - 1.
        # Standard squared-exponential kernel with a different length scale for each
        # parameter, as they may have very different physical properties.
        kernel = GPy.kern.Linear(nparams)
        kernel += GPy.kern.RBF(nparams, ARD=True)
        self.gp = GPy.models.GPRegression(param_cube, normtemps, kernel=kernel, noise_var=1e-10)
        status = self.gp.optimize(messages=False)
        if status.status != 'Converged':
            print("Restarting optimization")
            self.gp.optimize_restarts(num_restarts=10)

    def predict(self, params):
        """Get the predicted temperatures for a parameter set."""
        params_cube = map_to_unit_cube_list(params.reshape(1, -1), self.param_limits)
        temp_predict, var = self.gp.predict(params_cube)
        mean = (temp_predict+1)*self.scalefactors
        std = np.sqrt(var) * self.scalefactors
        return mean, std


class T0MultiBinAR1:
    """
    A wrapper around GPy that constructs a multi-fidelity emulator for the mean temperature over all redshifts.
    Parameters: LRparams, HRparams are the input parameter sets (nsims, nparams)
                LRtemps, HRtemps are the corresponding temperatures (nsims, nz)
                param_limits is a list of parameter limits (nparams, 2)
                n_fidelities is the number of fidelities
                ARD_last_fidelity flags whether to apply ARD for the last (highest) fidelity.
                n_restarts is the number of optimization restarts.
    """

    def __init__(self, LRparams, HRparams, LRtemps, HRtemps, param_limits, n_fidelities=2, ARD_last_fidelity=False, n_restarts=10):
        self.n_fidelities = n_fidelities
        self.param_limits = param_limits
        # assert that the two sets have the same number of parameters and redshifts
        assert np.shape(LRparams)[1] == np.shape(HRparams)[1]
        assert np.shape(LRtemps)[1] == np.shape(HRtemps)[1]
        self.nparams = np.shape(LRparams)[1]

        # get parameters into correct format
        param_cube = [map_to_unit_cube_list(LRparams, param_limits), map_to_unit_cube_list(HRparams, param_limits)]
        # ensure that the GP prior (a zero-mean input) is close to true.
        self.scalefactors = np.mean(LRtemps, axis=0)
        LRnormtemps = LRtemps/self.scalefactors - 1.
        HRnormtemps = HRtemps/self.scalefactors - 1.
        # this also adds the fidelity flag, 0 for LR, 1 for HR
        params, normtemps = convert_xy_lists_to_arrays(param_cube, [LRnormtemps, HRnormtemps])

        kernel_list = []
        for j in range(n_fidelities):
            # Standard squared-exponential kernel with a different length scale for
            # each parameter, as they may have very different physical properties.
            kernel = GPy.kern.Linear(self.nparams, ARD=False)
            kernel += GPy.kern.RBF(self.nparams, ARD=True)
            # final fidelity not ARD due to lack of training data
            if j == n_fidelities - 1:
                kernel = GPy.kern.Linear(self.nparams, ARD=ARD_last_fidelity)
                kernel += GPy.kern.RBF(self.nparams, ARD=ARD_last_fidelity)
            kernel_list.append(kernel)
        # make multi-fidelity kernels
        kernel = LinearMultiFidelityKernel(kernel_list)

        # Make default likelihood as different noise for each fidelity
        likelihood = GPy.likelihoods.mixed_noise.MixedNoise([GPy.likelihoods.Gaussian(variance=1.0) for _ in range(n_fidelities)])
        y_metadata = {"output_index": params[:, -1].astype(int)}

        self.gpy_models = GPy.core.GP(params, normtemps, kernel, likelihood, Y_metadata=y_metadata)
        self.optimize(n_restarts)


    def optimize(self, n_optimization_restarts):
        # no saving mechanism included here (yet) -- this trains so fast, it isn't really needed
        # fix noise
        getattr(self.gpy_models.mixed_noise, "Gaussian_noise").fix(1e-6)
        for j in range(1, self.n_fidelities):
            getattr(self.gpy_models.mixed_noise, "Gaussian_noise_{}".format(j)).fix(1e-6)
        model = GPyMultiOutputWrapper(self.gpy_models, n_outputs=self.n_fidelities, n_optimization_restarts=n_optimization_restarts)

        # first step optimization with fixed noise
        model.gpy_model.optimize_restarts(n_optimization_restarts, verbose=model.verbose_optimization, robust=True, parallel=False,)

        # unfix noise and re-optimize
        getattr(model.gpy_model.mixed_noise, "Gaussian_noise").unfix()
        for j in range(1, self.n_fidelities):
            getattr(model.gpy_model.mixed_noise, "Gaussian_noise_{}".format(j)).unfix()
        model.gpy_model.optimize_restarts(n_optimization_restarts, verbose=model.verbose_optimization, robust=True, parallel=False,)
        self.models = model

    def predict(self, params, res=1):
        """
        Predicts mean and variance for fidelity specified by last column of params.
        params is the point at which to predict.
        """
        assert res == 0 or res == 1
        params_cube = map_to_unit_cube_list(params.reshape(1, -1), self.param_limits)
        params_cube = np.concatenate([params_cube[0], np.ones(1)*res]).reshape(1,-1)
        temp_predict, var = self.models.predict(params_cube)
        mean = (temp_predict+1) * self.scalefactors
        std = np.sqrt(var) * self.scalefactors
        return mean, std
