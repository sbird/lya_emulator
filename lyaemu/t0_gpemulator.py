"""Building a surrogate using a Gaussian Process."""
import numpy as np
from .latin_hypercube import map_to_unit_cube, map_to_unit_cube_list
import GPy

class T0MultiBinGP:
    """A wrapper around the emulator that constructs a separate emulator for each redshift.
        Parameters: params is a list of parameter vectors.
                    temps is a list of mean temperatures (shape nparams, nz).
                    param_limits is a list of parameter limits (shape params, 2)."""
    def __init__(self, *, params, temps, param_limits):
        # Build an emulator for each redshift separately.
        self.nz = np.shape(temps)[1]
        gp = lambda i: T0SingleBinGP(params=params, temps=temps[:,i], param_limits=param_limits)
        print('Number of redshifts for emulator generation=%d' % (self.nz))
        self.gps = [gp(i) for i in range(self.nz)]
        self.temps = temps
        self.params = params

    def predict(self, params):
        """Get the predicted flux at a parameter set."""
        std = np.zeros(self.nz)
        means = np.zeros(self.nz)
        for i, gp in enumerate(self.gps):
            m, s = gp.predict(params)
            means[i] = m
            std[i] = s
        return means, std

class T0SingleBinGP:
    """An emulator wrapping a GP code for a single redshift.
       Parameters: params is a list of parameter vectors.
                   temps is a list of mean temperatures (same length as params).
                   param_limits is a list of parameter limits (shape 2,params)."""
    def __init__(self, *, params, temps, param_limits):
        self.params = params
        self.param_limits = param_limits
        self._get_interp(mean_temps=temps.reshape(-1, 1))

    def _get_interp(self, mean_temps):
        """Build the GP interpolator."""
        # Map the parameters onto a unit cube (so all variations have similar magnitude)
        nparams = np.shape(self.params)[1]
        params_cube = map_to_unit_cube_list(self.params, self.param_limits)
        # Check that we span the parameter space (comment out if using few samples)
        for i in range(nparams):
            assert np.max(params_cube[:,i]) > 0.8
            assert np.min(params_cube[:,i]) < 0.2
        # Normalise the mean temperature by the median value.
        # This ensures that the GP prior (a zero-mean input) is close to true.
        medind = np.argsort(mean_temps)[np.size(mean_temps)//2]
        self.scalefactors = mean_temps[medind]
        normtemps = mean_temps/self.scalefactors - 1.

        # Standard squared-exponential kernel with a different length scale for each
        # parameter, as they may have very different physical properties.
        kernel = GPy.kern.Linear(nparams)
        kernel += GPy.kern.RBF(nparams)
        self.gp = GPy.models.GPRegression(params_cube, normtemps, kernel=kernel, noise_var=1e-10)
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
