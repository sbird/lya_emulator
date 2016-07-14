"""A rate network for neutral hydrogen following
Katz, Weinberg & Hernquist 1996, eq. 28-32."""
import numpy as np
import scipy.interpolate as interp

class RateNetwork(object):
    """A rate network for neutral hydrogen following
    Katz, Weinberg & Hernquist 1996, eq. 28-32."""
    def __init__(self,redshift, helium=0.24, converge = 1e-5):
        self.recomb = RecombRates()
        self.photo = PhotoRates()
        self.helium = helium
        #proton mass in g
        self.protonmass=1.67262178e-24
        self.redshift = redshift
        self.converge = converge

    def get_nH(self, density):
        """The hydrogen atom number density"""
        return density * (1 - self.helium) / self.protonmass

    def _nH0(self, nh, temp, ne):
        """The neutral hydrogen number density. Eq. 33 of KWH."""
        alphaHp = self.recomb.alphaHp(temp)
        GammaeH0 = self.recomb.GammaeH0(temp)
        return nh * alphaHp/ (alphaHp + GammaeH0 + self.photo.gH0(self.redshift)/ne)

    def _nHp(self, nh, temp, ne):
        """The ionised hydrogen number density. Eq. 34 of KWH."""
        return nh - self._nH0(nh, temp, ne)

    def _nHep(self, nh, temp, ne):
        """The ionised helium number density. Eq. 35 of KWH."""
        yy = self.helium / 4 / (1 - self.helium)
        alphaHep = self.recomb.alphaHep(temp) + self.recomb.alphad(temp)
        alphaHepp = self.recomb.alphaHepp(temp)
        GammaHe0 = self.recomb.GammaeHe0(temp) + self.photo.gHe0(self.redshift)/ne
        GammaHep = self.recomb.GammaeHep(temp) + self.photo.gHep(self.redshift)/ne
        return yy * nh / (1 + alphaHep / GammaHe0 + GammaHep/alphaHepp)

    def _nHe0(self, nh, temp, ne):
        """The neutral helium number density. Eq. 36 of KWH."""
        alphaHep = self.recomb.alphaHep(temp) + self.recomb.alphad(temp)
        GammaHe0 = self.recomb.GammaeHe0(temp) + self.photo.gHe0(self.redshift)/ne
        return self._nHep(nh, temp, ne) * alphaHep / GammaHe0

    def _nHepp(self, nh, temp, ne):
        """The doubly ionised helium number density. Eq. 37 of KWH."""
        GammaHep = self.recomb.GammaeHep(temp) + self.photo.gHep(self.redshift)/ne
        alphaHepp = self.recomb.alphaHepp(temp)
        return self._nHep(nh, temp, ne) * GammaHep / alphaHepp

    def _ne(self, nh, temp, ne):
        """The electron number density. Eq. 38 of KWH."""
        return self._nHp(nh, temp, ne) + self._nHep(nh, temp, ne) + 2*self._nHepp(nh, temp, ne)

    def get_equilib_ne(self, density, temp):
        """Solve the system of equations for photo-ionisation equilibrium,
        starting with ne = nH and continuing until convergence."""
        nh = self.get_nH(density)
        ne = nh
        nenew = self._ne(nh, temp, ne)
        while np.abs(nenew - ne)/np.max([ne,1e-30]) > self.converge:
            ne = nenew
            nenew = self._ne(nh, temp, ne)
        return ne

    def get_neutral_fraction(self, density, temp):
        """Get the neutral hydrogen fraction at a given temperature and density."""
        ne = self.get_equilib_ne(density, temp)
        nh = self.get_nH(density)
        return self._nH0(nh, temp, ne) / nh

class RecombRates(object):
    """Recombination rates and collisional ionization rates, as a function of temperature.
    Currently KWH 06, astro-ph/9509107, Table 2."""
    def alphaHp(self,temp):
        """Recombination rate for H+, ionized hydrogen, in cm^3/s.
        Temp in K."""
        return 8.4e-11 / np.sqrt(temp) / np.power(temp/1000, 0.2) / (1+ np.power(temp/1e6, 0.7))

    def alphaHep(self,temp):
        """Recombination rate for He+, ionized helium, in cm^3/s.
        Temp in K."""
        return 1.5e-10 / np.power(temp,0.6353)

    def alphad(self, temp):
        """Recombination rate for dielectronic recombination, in cm^3/s.
        Temp in K."""
        return 1.9e-3 / np.power(temp,1.5) * np.exp(-4.7e5/temp)*(1+0.3*np.exp(-9.4e4/temp))

    def alphaHepp(self, temp):
        """Recombination rate for doubly ionized helium, in cm^3/s.
        Temp in K."""
        return 3.36e-10 / np.sqrt(temp) / np.power(temp/1000, 0.2) / (1+ np.power(temp/1e6, 0.7))

    def GammaeH0(self,temp):
        """Collisional ionization rate for H0 in cm^3/s. Temp in K"""
        return 5.85e-11 * np.sqrt(temp) * np.exp(-157809.1/temp) / (1+ np.sqrt(temp/1e5))

    def GammaeHe0(self,temp):
        """Collisional ionization rate for H0 in cm^3/s. Temp in K"""
        return 2.38e-11 * np.sqrt(temp) * np.exp(-285335.4/temp) / (1+ np.sqrt(temp/1e5))

    def GammaeHep(self,temp):
        """Collisional ionization rate for H0 in cm^3/s. Temp in K"""
        return 5.68e-12 * np.sqrt(temp) * np.exp(-631515.0/temp) / (1+ np.sqrt(temp/1e5))

class PhotoRates(object):
    """The photoionization rates for a given species.
    Eq. 29 of KWH 06. This is loaded from a TREECOOL table."""
    def __init__(self, treecool_file="TREECOOL", factor=1.):
        self.set_factor(factor)
        #Format of the treecool table:
        # log_10(1+z), Gamma_HI, Gamma_HeI, Gamma_HeII,  Qdot_HI, Qdot_HeI, Qdot_HeII,
        # where 'Gamma' is the photoionization rate and 'Qdot' is the photoheating rate.
        # The Gamma's are in units of s^-1, and the Qdot's are in units of erg s^-1.
        data = np.loadtxt(treecool_file)
        redshifts = data[:,0]
        photo_rates = data[:,1:4]
        assert np.shape(redshifts)[0] == np.shape(photo_rates)[0]
        self.Gamma_HI = interp.InterpolatedUnivariateSpline(redshifts, photo_rates[:,0])
        self.Gamma_HeI = interp.InterpolatedUnivariateSpline(redshifts, photo_rates[:,1])
        self.Gamma_HeII = interp.InterpolatedUnivariateSpline(redshifts, photo_rates[:,2])

    def set_factor(self, factor=1.):
        """Set a photoionisation rate scaling factor."""
        self.factor = factor

    def gHe0(self,redshift):
        """Get photo rate for neutral Helium"""
        log1z = np.log10(1+redshift)
        return self.Gamma_HeI(log1z)*self.factor

    def gHep(self,redshift):
        """Get photo rate for singly ionized Helium"""
        log1z = np.log10(1+redshift)
        return self.Gamma_HeII(log1z)*self.factor

    def gH0(self,redshift):
        """Get photo rate for neutral Hydrogen"""
        log1z = np.log10(1+redshift)
        return self.Gamma_HI(log1z)*self.factor
