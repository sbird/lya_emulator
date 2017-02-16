"""Building a surrogate using a Gaussian Process."""
import numpy as np
from sklearn import gaussian_process
from sklearn.gaussian_process import kernels
from latin_hypercube import map_to_unit_cube

class SkLearnGP(object):
    """An emulator using the one in Scikit-learn"""
    def __init__(self, *, params, kf, powers,param_limits):
        self.powers = powers
        self.params = params
        self.param_limits = param_limits
        self.cur_tau_factor = -1
        self.kf = kf
        self.intol = 1e-5

    def _get_interp(self, tau0_factor=None):
        """Build the actual interpolator."""
        self.cur_tau_factor = tau0_factor
        flux_vectors = np.array([ps.get_power(kf = self.kf, tau0_factor = tau0_factor) for ps in self.powers])
        #Standard squared-exponential kernel with a different length scale for each parameter, as
        #they may have very different physical properties.
        kernel = 1.0*kernels.RBF(length_scale=np.ones_like(self.params[0,:]), length_scale_bounds=(1e-3, 2))
        #White noise kernel to account for residual noise in the FFT, etc.
        kernel+= kernels.WhiteKernel(noise_level=1e-5, noise_level_bounds=(1e-7, 1e-4))
        self.gp = gaussian_process.GaussianProcessRegressor(normalize_y=False, n_restarts_optimizer = 20,kernel=kernel)
        #Map the parameters onto a unit cube so that all the variations are similar in magnitude
        params_cube = map_to_unit_cube(self.params, self.param_limits)
        #Normalise the flux vectors by the median power spectrum.
        #This ensures that the GP prior (a zero-mean input) is close to true.
        medind = np.argsort(np.mean(flux_vectors, axis=1))[np.shape(flux_vectors)[0]//2]
        self.scalefactors = flux_vectors[medind,:]
        self.paramzero = params_cube[medind,:]
        #Normalise by the median value
        normspectra = flux_vectors/self.scalefactors-1.
        dparams = params_cube - self.paramzero
        #Do a linear fit first, and fit the GP to the residuals.
        self.linearcoeffs = self._get_linear_fit(dparams, normspectra)
        newspec = normspectra / self._get_linear_pred(dparams) -1
        #Avoid nan from the division
        newspec[medind,:] = 0
        self.gp.fit(params_cube, newspec)
        #Check we reproduce the input
        test,_ = self.predict(params_cube[0,:].reshape(1,-1), tau0_factor=tau0_factor)
        worst = np.abs(test[0] / flux_vectors[0,:]-1)
        if np.max(worst) > self.intol:
            print("Bad interpolation at:",np.where(worst > np.max(worst)*0.9))
            assert np.max(worst) < self.intol

    def _get_linear_fit(self, dparams, normspectra):
        """Fit a multi-variate linear trend line through the points."""
        (derivs, _,_, _)=np.linalg.lstsq(dparams, normspectra)
        return derivs

    def _get_linear_pred(self, dparams):
        """Get the linear trend prediction."""
        return np.dot(self.linearcoeffs.T, dparams.T).T

    def predict(self, params,tau0_factor):
        """Get the predicted flux at a parameter value (or list of parameter values)."""
        #First get the residuals
        if tau0_factor is not self.cur_tau_factor:
            self._get_interp(tau0_factor = tau0_factor)
        #Map the parameters onto a unit cube so that all the variations are similar in magnitude
        params_cube = map_to_unit_cube(params, self.param_limits)
        flux_predict, std = self.gp.predict(params_cube, return_std=True)
        #x = x/q - 1
        #E(y) = E(x) /q - 1
        #Var(y) = Var(x)/q^2
        #Make sure std is reasonable
        std = np.max([np.min([std,1e7]),1e-8])
        #Then multiply by linear fit.
        lincorr = self._get_linear_pred(params - self.paramzero)
        lin_predict = (flux_predict +1) * lincorr
        #Then multiply by mean value to denorm.
        mean = (lin_predict+1)*self.scalefactors
        std = std * self.scalefactors * lincorr
        return mean, std

    def get_predict_error(self, test_params, test_exact):
        """Get the difference between the predicted GP
        interpolation and some exactly computed test parameters."""
        test_exact = test_exact.reshape(np.shape(test_params)[0],-1)
        return self.gp.score(test_params, test_exact)

class SDSSData(object):
    """A class to store the flux power and corresponding covariance matrix from SDSS. A little tricky because of the redshift binning."""
    def __init__(self, datafile="data/lya.sdss.table.txt", covarfile="data/lya.sdss.covar.txt"):
        # Read SDSS best-fit data.
        # Contains the redshift wavenumber from SDSS
        # See 0405013 section 5.
        # First column is redshift
        # Second is k in (km/s)^-1
        # Third column is P_F(k)
        # Fourth column (ignored): square roots of the diagonal elements
        # of the covariance matrix. We use the full covariance matrix instead.
        # Fifth column (ignored): The amount of foreground noise power subtracted from each bin.
        # Sixth column (ignored): The amound of background power subtracted from each bin.
        # A metal contamination subtraction that McDonald does but we don't.
        data = np.loadtxt(datafile)
        self.redshifts = data[:,0]
        self.kf = data[:,1]
        self.pf = data[:,1]
        #The covariance matrix, correlating each k and z bin with every other.
        #kbins vary first, so that we have 11 bins with z=2.2, then 11 with z=2.4,etc.
        self.covar = np.loadtxt(covarfile)

    def get_kf(self):
        """Get the (unique) flux k values"""
        return np.sort(np.array(list(set(self.kf))))

    def get_redshifts(self):
        """Get the (unique) redshift bins, sorted in decreasing redshift"""
        return np.sort(np.array(list(set(self.redshifts))))[::-1]

    def get_icovar(self):
        """Get the inverse covariance matrix"""
        return np.linalg.inv(self.covar)

    def get_covar(self):
        """Get the inverse covariance matrix"""
        return self.covar
