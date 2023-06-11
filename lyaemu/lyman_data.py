"""Module to load the covariance matrix (from BOSS DR9 or SDSS DR5 data) from tables."""
import os.path
import pandas
import numpy as np
import numpy.testing as npt

class SDSSData:
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
        cdir = os.path.dirname(__file__)
        datafile = os.path.join(cdir,datafile)
        covarfile = os.path.join(cdir, covarfile)
        data = np.loadtxt(datafile)
        self.redshifts = data[:, 0]
        self.kf = data[:, 1]
        self.pf = data[:, 2]
        self.nz = np.size(self.get_redshifts())
        self.nk = np.size(self.get_kf())
        assert self.nz * self.nk == np.size(self.kf)
        #The covariance matrix, correlating each k and z bin with every other.
        #kbins vary first, so that we have 11 bins with z=2.2, then 11 with z=2.4,etc.
        self.covar = np.loadtxt(covarfile)

    def get_kf(self, kf_bin_nums=None):
        """Get the (unique) flux k values"""
        kf_array = np.sort(np.array(list(set(self.kf))))
        if kf_bin_nums is None:
            return kf_array
        return kf_array[kf_bin_nums]

    def get_redshifts(self):
        """Get the (unique) redshift bins, sorted in decreasing redshift"""
        return np.sort(np.array(list(set(self.redshifts))))[::-1]

    def get_pf(self, zbin=None):
        """Get the power spectrum"""
        if zbin is None:
            return self.pf
        ii = np.where((self.redshifts < zbin + 0.01)*(self.redshifts > zbin - 0.01))
        return self.pf[ii]

    def get_icovar(self):
        """Get the inverse covariance matrix"""
        return np.linalg.inv(self.covar)

    def get_covar(self, zbin=None):
        """Get the covariance matrix"""
        _ = zbin
        return self.covar

class BOSSData(SDSSData):
    """A class to store the flux power and corresponding covariance matrix from BOSS."""
    def __init__(self, datafile=None, covardir=None):
        cdir = os.path.dirname(__file__)
        # by default load the more recent data, from DR14: Chanbanier 2019, arXiv:1812.03554
        if datafile is None or datafile == 'dr14':
            datafile = os.path.join(cdir,"data/boss_dr14_data/Pk1D_data.dat")
            covarfile = os.path.join(cdir, "data/boss_dr14_data/Pk1D_cor.dat")
            systfile = os.path.join(cdir, "data/boss_dr14_data/Pk1D_syst.dat")
            # Read BOSS DR14 flux power data.
            # Fourth column: statistical uncertainty
            # Fifth and Sixth column (unused) are noise and side-band powers
            data = np.loadtxt(datafile)
            self.redshifts = data[:, 0]
            self.kf = data[:, 1]
            self.pf = data[:, 2]
            self.nz = np.size(self.get_redshifts())
            self.nk = np.size(self.get_kf())
            assert self.nz * self.nk == np.size(self.kf)
            # systematic uncertainies (8 contributions):
            # continuum, noise, resolution, SB, linemask, DLAmask, DLAcompleteness, BALcompleteness
            syst = np.loadtxt(systfile)
            self.covar_diag = np.sum(syst**2, axis=1) + data[:, 3]**2
            # The correlation matrix, correlating each k and z bin with every other.
            # file includes a series of 13 (for each z bin) 35x35 matrices
            corr = np.loadtxt(covarfile)
            self.covar = np.zeros((len(self.redshifts), len(self.redshifts))) #Full covariance matrix (35*13 x 35*13) for k, z
            for bb in range(self.nz):
                dd = corr[35*bb:35*(bb+1)] # k-bin covariance matrix (35 x 35) for single redshift
                self.covar[35*bb:35*(bb+1), 35*bb:35*(bb+1)] = dd #Filling in block matrices along diagonal

        # load the older dataset, from DR9: Palanque-Delabrouille 2013, arXiv:1306.5896
        elif datafile == 'dr9':
            datafile = os.path.join(cdir, "data/boss_dr9_data/table4a.dat")
            covardir = os.path.join(cdir, "data/boss_dr9_data")
            # Read BOSS DR9 flux power data. See Readme file.
            # Sixth column: statistical uncertainty
            # Ninth column: systematic uncertainty
            # correlation matrices for each redshift stored in separate files, "cct4b##.dat"
            data = np.loadtxt(datafile)
            self.redshifts = data[:, 2]
            self.kf = data[:, 3]
            self.pf = data[:, 4]
            self.nz = np.size(self.get_redshifts())
            self.nk = np.size(self.get_kf())
            assert self.nz * self.nk == np.size(self.kf)
            self.covar_diag = data[:, 5]**2 + data[:, 8]**2
            # The correlation matrix, correlating each k and z bin with every other.
            # kbins vary first, so that we have 11 bins with z=2.2, then 11 with z=2.4, etc.
            self.covar = np.zeros((len(self.redshifts), len(self.redshifts))) #Full matrix (35*12 x 35*12) for k, z
            for bb in range(self.nz):
                dfile = os.path.join(covardir, "cct4b"+str(bb+1)+".dat")
                dd = np.loadtxt(dfile) #k-bin correlation matrix (35 x 35) for single redshift
                self.covar[35*bb:35*(bb+1), 35*bb:35*(bb+1)] = dd #Filling in block matrices along diagonal
        else:
            raise NotImplementedError("SDSS Data %s not found!" % datafile)

    def get_covar(self, zbin=None):
        """Get the covariance matrix"""
        # Note, DR9 and DR14 datasets report correlation matrices,
        # hence the conversion factor (outer product of covar_diag)
        if zbin is None:
            # return the full covariance matrix (all redshifts) sorted in blocks from low to high redshift
            return self.covar * np.outer(np.sqrt(self.covar_diag), np.sqrt(self.covar_diag))
        # return the covariance matrix for a specified redshift
        ii = np.where((self.redshifts < zbin + 0.01)*(self.redshifts > zbin - 0.01)) #Elements in full matrix for given z
        rr = (np.min(ii), np.max(ii)+1)
        std_diag_single_z = np.sqrt(self.covar_diag[rr[0]:rr[1]])
        covar_matrix = self.covar[rr[0]:rr[1], rr[0]:rr[1]] * np.outer(std_diag_single_z, std_diag_single_z)
        npt.assert_allclose(np.diag(covar_matrix), self.covar_diag[rr[0]:rr[1]], atol=1.e-16)
        return covar_matrix

    def get_covar_diag(self):
        """Get the diagonal of the covariance matrix"""
        return self.covar_diag

class KSData(SDSSData):
    """A class to store the flux power and corresponding covariance matrix from KODIAQ-SQUAD."""
    def __init__(self, datafile=None, conservative=True):
        cdir = os.path.dirname(__file__)
        # data from the supplementary material in Karacayli+21](https://academic.oup.com/mnras/article/509/2/2842/6425772)
        if conservative:
            datafile = os.path.join(cdir,"data/kodiaq_squad/final-conservative-p1d-karacayli_etal2021.txt")
            # Read KODIAQ-SQUAD flux power data.
            # Column #1 : redshift
            # Column #2: k
            # Column #3: power
            # Column #4: estimated error
            a = pandas.read_csv(datafile, skiprows=[0], sep='|', header=None)
            self.redshifts = np.array(a[1], dtype='float')
            self.kf = np.array(a[2], dtype='float')
            self.pf = np.array(a[3], dtype='float')
            self.nz = np.size(self.get_redshifts())
            self.nk = np.size(self.get_kf())
            self.covar_diag = np.array(a[4], dtype='float')
        else:
            datafile = os.path.join(cdir,"data/kodiaq_squad/detailed-p1d-results-karacayli_etal2021.txt")

            # Read KODIAQ-SQUAD flux power data.
            # Column #1 : redshift
            # Column #2: k
            # Column #3: power
            a = pandas.read_csv(datafile, skiprows=[0], sep='|', header=None)
            self.redshifts = np.array(a[1], dtype='float')
            self.kf = np.array(a[2], dtype='float')
            self.pf = np.array(a[3], dtype='float')
            self.nz = np.size(self.get_redshifts())
            self.nk = np.size(self.get_kf())
            # systematic uncertainies (8 contributions):
            # continuum, noise, resolution, DLA, metals
            self.covar_diag = np.zeros_like(self.kf)
            for i in [7,9,10,12,13,14]:
                self.covar_diag += np.array(a[i], dtype='float')**2
            self.covar_diag = np.sqrt(self.covar_diag)

    def get_covar_diag(self, zbin=None):
        """Get the diagonal of the covariance matrix"""
        ii = np.where((self.redshifts < zbin + 0.01)*(self.redshifts > zbin - 0.01))
        return self.covar_diag[ii]


class XQ100Data(SDSSData):
    """A class to store the flux power and corresponding covariance matrix from XQ100."""
    def __init__(self, datafile=None):
        cdir = os.path.dirname(__file__)

        # data from https://github.com/bayu-wilson/lyb_pk/tree/main/output , [Wilson+21](https://arxiv.org/abs/2106.04837)

        datafile = os.path.join(cdir,"data/xq100/pk_obs_corrNR_offset_DLATrue_metalTrue_res0.csv")

        # Read XQ100 flux power data.
        # Columns are labeled as 'z', 'k' and 'paa'

        a = pandas.read_csv(datafile)
        self.redshifts = np.array(a['z'], dtype='float')
        self.kf = np.array(a['k'], dtype='float')
        self.pf = np.array(a['paa'], dtype='float')
        self.nz = np.size(self.get_redshifts())
        self.nk = np.size(self.get_kf())
