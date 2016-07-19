"""A rate network for neutral hydrogen following
Katz, Weinberg & Hernquist 1996, eq. 28-32."""
import numpy as np
import scipy.interpolate as interp

class RateNetwork(object):
    """A rate network for neutral hydrogen following
    Katz, Weinberg & Hernquist 1996, eq. 28-32."""
    def __init__(self,redshift, photo_factor = 1., f_bar = 0.17, converge = 1e-5):
        self.recomb = RecombRates()
        self.photo = PhotoRates()
        self.photo_factor = photo_factor
        self.f_bar = f_bar
        #proton mass in g
        self.protonmass = 1.67262178e-24
        self.redshift = redshift
        self.converge = converge
        zz = [0, 1, 2, 3, 4, 5, 6, 7,8]
        #Tables for the self-shielding correction. Note these are not well-measured for z > 5!
        gray_opac = [2.59e-18,2.37e-18,2.27e-18, 2.15e-18, 2.02e-18, 1.94e-18, 1.82e-18, 1.71e-18, 1.60e-18]
        self.Gray_ss = interp.InterpolatedUnivariateSpline(zz, gray_opac)

    def _nH0(self, nh, temp, ne):
        """The neutral hydrogen number density. Eq. 33 of KWH."""
        alphaHp = self.recomb.alphaHp(temp)
        GammaeH0 = self.recomb.GammaeH0(temp)
        photofac = self.photo_factor*self._self_shield_corr(nh, temp)
        return nh * alphaHp/ (alphaHp + GammaeH0 + self.photo.gH0(self.redshift)/ne)*photofac

    def _nHp(self, nh, temp, ne):
        """The ionised hydrogen number density. Eq. 34 of KWH."""
        return nh - self._nH0(nh, temp, ne)

    def _nHep(self, nh, temp, ne):
        """The ionised helium number density, divided by the helium number fraction. Eq. 35 of KWH."""
        alphaHep = self.recomb.alphaHep(temp) + self.recomb.alphad(temp)
        alphaHepp = self.recomb.alphaHepp(temp)
        photofac = self.photo_factor*self._self_shield_corr(nh, temp)
        GammaHe0 = self.recomb.GammaeHe0(temp) + self.photo.gHe0(self.redshift)/ne*photofac
        GammaHep = self.recomb.GammaeHep(temp) + self.photo.gHep(self.redshift)/ne*photofac
        return nh / (1 + alphaHep / GammaHe0 + GammaHep/alphaHepp)

    def _nHe0(self, nh, temp, ne):
        """The neutral helium number density, divided by the helium number fraction. Eq. 36 of KWH."""
        alphaHep = self.recomb.alphaHep(temp) + self.recomb.alphad(temp)
        photofac = self.photo_factor*self._self_shield_corr(nh, temp)
        GammaHe0 = self.recomb.GammaeHe0(temp) + self.photo.gHe0(self.redshift)/ne*photofac
        return self._nHep(nh, temp, ne) * alphaHep / GammaHe0

    def _nHepp(self, nh, temp, ne):
        """The doubly ionised helium number density, divided by the helium number fraction. Eq. 37 of KWH."""
        photofac = self.photo_factor*self._self_shield_corr(nh, temp)
        GammaHep = self.recomb.GammaeHep(temp) + self.photo.gHep(self.redshift)/ne*photofac
        alphaHepp = self.recomb.alphaHepp(temp)
        return self._nHep(nh, temp, ne) * GammaHep / alphaHepp

    def _ne(self, nh, temp, ne, helium=0.24):
        """The electron number density. Eq. 38 of KWH."""
        yy = helium / 4 / (1 - helium)
        return self._nHp(nh, temp, ne) + yy * self._nHep(nh, temp, ne) + 2* yy * self._nHepp(nh, temp, ne)

    def get_equilib_ne(self, nh, temp,helium=0.24):
        """Solve the system of equations for photo-ionisation equilibrium,
        starting with ne = nH and continuing until convergence."""
        ne = nh
        nenew = self._ne(nh, temp, ne, helium=helium)
        while np.abs(nenew - ne)/np.max([ne,1e-30]) > self.converge:
            ne = nenew
            nenew = self._ne(nh, temp, ne, helium=helium)
        return ne

    def get_neutral_fraction(self, nh, temp, helium=0.24):
        """Get the neutral hydrogen fraction at a given temperature and density."""
        ne = self.get_equilib_ne(nh, temp, helium=helium)
        return self._nH0(nh, temp, ne) / nh

    def _self_shield_corr(self, nh, temp):
        """Photoionisation rate as  a function of density from Rahmati 2012, eq. 14.
        Calculates Gamma_{Phot} / Gamma_{UVB}.
        Inputs: hydrogen density, temperature
            n_H
        The coefficients are their best-fit from appendix A."""
        nSSh = 1.003*self._self_shield_dens(self.redshift, temp)
        return 0.98*(1+(nh/nSSh)**1.64)**-2.28+0.02*(1+nh/nSSh)**-0.84

    def _self_shield_dens(self,redshift, temp):
        """Calculate the critical self-shielding density. Rahmati 202 eq. 13.
        gray_opac is a parameter of the UVB used.
        gray_opac is in cm^2 (2.49e-18 is HM01 at z=3)
        temp is particle temperature in K
        f_bar is the baryon fraction. 0.17 is roughly 0.045/0.265
        Returns density in atoms/cm^3"""
        T4 = temp/1e4
        G12 = self.photo.gH0(redshift)/1e-12
        return 6.73e-3 * (self.Gray_ss(redshift) / 2.49e-18)**(-2./3)*(T4)**0.17*(G12)**(2./3)*(self.f_bar/0.17)**(-1./3)

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
    def __init__(self, treecool_file="TREECOOL"):
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

    def gHe0(self,redshift):
        """Get photo rate for neutral Helium"""
        log1z = np.log10(1+redshift)
        return self.Gamma_HeI(log1z)

    def gHep(self,redshift):
        """Get photo rate for singly ionized Helium"""
        log1z = np.log10(1+redshift)
        return self.Gamma_HeII(log1z)

    def gH0(self,redshift):
        """Get photo rate for neutral Hydrogen"""
        log1z = np.log10(1+redshift)
        return self.Gamma_HI(log1z)
