"""Building a surrogate using a Gaussian Process."""
import math
import numpy as np
from sklearn import gaussian_process
import emcee

import linear_theory
import latin_hypercube

def lnlike_linear(params, *, gp=None, data=None):
    """A simple emcee likelihood function for the Lyman-alpha forest using the
       simple linear model with only cosmological parameters.
       This neglects many important properties!"""
    assert gp is not None
    assert data is not None
    predicted = gp.predict(params)
    diff = predicted-data.pf
    return -np.dot(diff,np.dot(data.invcovar,diff))/2.0

def init_lnlike(nsamples, data=None):
    """Initialise the emulator for the likelihood function."""
#     param_names = ['bias_flux', 'ns', 'As']
    param_limits = np.array([[-2., 0], [0.5, 1.5], [1.5e-8, 8.0e-9]])
    params = latin_hypercube.get_hypercube_samples(param_limits, nsamples)
    data = SDSSData()
    #Get unique values
    flux_vectors = np.array([linear_theory.get_flux_power(bias_flux = pp[0], ns=pp[1], As=pp[2], zz=data.get_redshifts(), kf=data.get_kf()) for pp in params])
    gp = SkLearnGP(tau_means = params[:,0], ns = params[:,1], As = params[:,2], kf=data.kf, flux_vectors=flux_vectors)
    return gp, data

def build_fake_fluxes(nsamples):
    """Simple function using linear test case to build an emulator."""
    param_limits = np.array([[-1., 0], [0.9, 1.0], [1.5e-9, 3.0e-9]])
    params = latin_hypercube.get_hypercube_samples(param_limits, nsamples)
    data = SDSSData()
    flux_vectors = np.array([linear_theory.get_flux_power(bias_flux = pp[0], ns=pp[1], As=pp[2], kf=data.get_kf(), zz=data.get_redshifts()) for pp in params])
    gp = SkLearnGP(tau_means = params[:,0], ns=params[:,1], As=params[:,2], kf=data.kf, flux_vectors=flux_vectors)
    random_samples = latin_hypercube.get_random_samples(param_limits, nsamples//2)
    random_test_flux_vectors = np.array([linear_theory.get_flux_power(bias_flux = pp[0], ns=pp[1], As=pp[2], kf=data.get_kf(),zz=data.get_redshifts()) for pp in random_samples])
    diff_sk = gp.get_predict_error(random_samples, random_test_flux_vectors)
    return gp, diff_sk

class SkLearnGP(object):
    """An emulator using the one in Scikit-learn"""
    def __init__(self, *, tau_means, ns, As, kf, flux_vectors):
        params = np.array([tau_means, ns, As]).T
        self._siIIIform = self._siIIIcorr(kf)
        flux_vectors = flux_vectors.reshape(np.size(tau_means),-1)
        assert np.shape(flux_vectors) == (np.size(tau_means), np.size(kf))
        self.gp = gaussian_process.GaussianProcess()
        self.gp.fit(params, flux_vectors)

    def predict(self, *, tau_means, ns, As, fSiIII):
        """Get the predicted flux at a parameter value (or list of parameter values)."""
        params = np.array([tau_means, ns, As]).T
        flux_predict , cov = self.gp.predict(params) * self.SiIIIcorr(fSiIII, tau_means)
        return flux_predict, cov

    def get_predict_error(self, test_params, test_exact):
        """Get the difference between the predicted GP interpolation and some exactly computed test parameters."""
        test_exact = test_exact.reshape(np.shape(test_params)[0],-1)
        return self.gp.score(test_params, test_exact)

    def _siIIIcorr(self, kf):
        """For precomputing the shape of the SiIII correlation"""
        #Compute bin boundaries in logspace.
        kmids = np.zeros(np.size(kf)+1)
        kmids[1:-1] = np.exp((np.log(kf[1:])+np.log(kf[:-1]))/2.)
        #arbitrary final point
        kmids[-1] = 2*math.pi/2271 + kmids[-2]
        # This is the average of cos(2271k) across the k interval in the bin
        siform = np.zeros_like(kf)
        siform = (np.sin(2271*kmids[1:])-np.sin(2271*kmids[:-1]))/(kmids[1:]-kmids[:-1])/2271.
        #Correction for the zeroth bin, because the integral is oscillatory there.
        siform[0] = np.cos(2271*kf[0])
        return siform

    def SiIIIcorr(self, fSiIII, tau_eff):
        """The correction for SiIII contamination, as per McDonald."""
        assert tau_eff > 0
        aa = fSiIII/(1-np.exp(-tau_eff))
        return 1 + aa**2 + 2 * aa * self._siIIIform

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
        covar = np.loadtxt(covarfile)
        self.invcovar = np.linalg.inv(covar)

    def get_kf(self):
        """Get the (unique) flux k values"""
        return np.sort(np.array(list(set(self.kf))))

    def get_redshifts(self):
        """Get the (unique) redshift bins, sorted in decreasing redshift"""
        return np.sort(np.array(list(set(self.redshifts))))[::-1]
