"""Module to load the covariance matrix (from BOSS DR9 or SDSS DR5 data) from tables."""
import os.path
import numpy as np

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

    def get_covar(self, zbin=None):
        """Get the inverse covariance matrix"""
        _ = zbin
        return self.covar

class BOSSData(SDSSData):
    """A class to store the flux power and corresponding covariance matrix from BOSS."""
    def __init__(self, datafile=os.path.expanduser("~/Software/lya_emulator/data/boss_dr9_data/table4a.dat"), covardir=os.path.expanduser("~/Software/lya_emulator/data/boss_dr9_data")):
        # Read SDSS best-fit data.
        # Contains the redshift wavenumber from SDSS
        # See Readme file.
        # Fourth column: square roots of covariance diagonal
        data = np.loadtxt(datafile)
        self.redshifts = data[:,2]
        self.kf = data[:,3]
        self.pf = data[:,4]
        self.covar_diag = data[:,5]**2 + data[:,8]**2
        #The covariance matrix, correlating each k and z bin with every other.
        #kbins vary first, so that we have 11 bins with z=2.2, then 11 with z=2.4,etc.
        self.covar = np.zeros((len(self.redshifts),len(self.redshifts)))
        for bb in range(12):
            dfile = os.path.join(covardir,"cct4b"+str(bb+1)+".dat")
            dd = np.loadtxt(dfile)
            self.covar[35*bb:35*(bb+1),35*bb:35*(bb+1)] = dd

    def get_covar(self, zbin=None):
        """Get the covariance matrix"""
        if zbin is None:
            return self.covar * self.covar_diag
        ii = np.where((self.redshifts < zbin + 0.01)*(self.redshifts > zbin - 0.01))
        rr = (np.min(ii), np.max(ii)+1)
        return self.covar[rr[0]:rr[1],rr[0]:rr[1]] * self.covar_diag[rr[0]:rr[1]]

    def get_covar_diag(self):
        """Get the covariance matrix"""
        return self.covar_diag
