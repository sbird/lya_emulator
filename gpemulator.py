"""Building a surrogate using a Gaussian Process."""
import math
import os.path
import json
import numpy as np
from sklearn import gaussian_process

class SkLearnGP(object):
    """An emulator using the one in Scikit-learn"""
    def __init__(self, *, params, kf, flux_vectors, savedir=None):
        if params is None:
            (params, flux_vectors) = self.load(savedir)
        self._siIIIform = self._siIIIcorr(kf)
        assert np.shape(flux_vectors)[1] % np.size(kf) == 0
        self.gp = gaussian_process.GaussianProcessRegressor(normalize_y=True)
        self.gp.fit(params, flux_vectors)
        self.params = params
        self.flux_vectors = flux_vectors
        if savedir is not None:
            self.dump(savedir)

    def predict(self, params,fSiIII=0.):
        """Get the predicted flux at a parameter value (or list of parameter values)."""
        flux_predict, std = self.gp.predict(params, return_std=True)
#         flux_predict *= self.SiIIIcorr(fSiIII,tau_means)
        return flux_predict, std

    def get_predict_error(self, test_params, test_exact):
        """Get the difference between the predicted GP
        interpolation and some exactly computed test parameters."""
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

    def dump(self, savedir, dumpfile="gp_training.json"):
        """Dump training data to a textfile."""
        #Arrays can't be serialised so convert them back and forth to lists
        ppl = self.params.tolist()
        fvl = self.flux_vectors.tolist()
        with open(os.path.join(savedir, dumpfile), 'w') as jsout:
            json.dump([ppl, fvl], jsout)

    def load(self,savedir, dumpfile="gp_training.json"):
        """Load parameters from a textfile."""
        with open(os.path.join(savedir, dumpfile), 'r') as jsin:
            [ppl, fvl] = json.load(jsin)
        return (np.array(ppl), np.array(fvl))

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

    def get_icovar(self):
        """Get the inverse covariance matrix"""
        return self.invcovar
