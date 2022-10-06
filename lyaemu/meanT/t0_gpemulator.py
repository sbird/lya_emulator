"""Building a surrogate using a Gaussian Process."""
import numpy as np
from ..latin_hypercube import map_to_unit_cube, map_to_unit_cube_list
import GPy

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
        # Map the parameters onto a unit cube (so all variations have similar magnitude)
        nparams = np.shape(self.params)[1]
        param_cube = map_to_unit_cube_list(self.params, self.param_limits)
        # Ensure that the GP prior (a zero-mean input) is close to true.
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
