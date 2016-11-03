"""Module for subclassing spectra to generate spectra with a different
photo-ionising background. This works by generating new spectra.
The neutral fraction of each particle is adjusted as it goes through the spectral generator.
This should avoid the need to rescale to the mean flux, and may be substantially more accurate."""

import math
import numpy as np
import spectra
import rate_network
import unitsystem

class RescaledGasProperties(object):
    """Class which does the actual rescaling of the temperature."""
    def __init__(self, redshift, photo_factor=1., temp_factor=1., gamma_factor = 1., hubble = 0.71, omega_matter = 0.27, f_bar=0.17, units = None):
        if units is not None:
            self.units = units
        else:
            self.units = unitsystem.UnitSystem()
        self.rates = rate_network.RateNetwork(redshift=redshift, photo_factor = photo_factor,f_bar = f_bar, cool="KWH", recomb="Cen92")
        #Some constants and unit systems
        #self.hy_mass = 0.76 # Hydrogen massfrac
        self.gamma=5./3
        #Boltzmann constant (cgs)
        self.boltzmann=1.38066e-16
        self.hubble = hubble
        self.redshift = redshift
        self.temp_factor = temp_factor
        self.gamma_factor = gamma_factor
        #Mean density
        self.density0 = self.rhocrit(redshift, omega_matter)

    def rhocrit(self, redshift, omega_matter):
        """Critical density (also the mean density) at given redshift. Units are atoms / cm^3."""
        #gravity in cgs: cm^3 g^-1 s^-2
        gravity = 6.674e-8
        #Hubble factor (~70km/s/Mpc) at z=0 in s^-1
        hubble = self.hubble * 100 / 3.0856776e+19
        hubz2 = (omega_matter*(1+redshift)**3 + (1-omega_matter)) * hubble**2
        #Critical density at redshift in g cm^-3
        rhocrit = 3 * hubz2 / (8*math.pi* gravity)
        #in protons cm^-3
        return rhocrit / self.units.protonmass

    def get_code_rhoH(self,bar):
        """Convert density to protons /cm^3: internal gadget density unit is h^2 (1e10 M_sun) / kpc^3.
        Note this is gas density, NOT hydrogen density!"""
        nH = np.array(bar["Density"])
        conv = np.float32(self.units.UnitDensity_in_cgs*self.hubble**2/(self.units.protonmass)*(1+self.redshift)**3)
        #Convert to physical
        return nH*conv

    def get_reproc_HI(self, bar):
        """Get a neutral hydrogen *fraction* using values given by Arepo
        which are based on Rahmati 2012 if UVB_SELF_SHIELDING is on.
        Above the star formation density use the Rahmati fitting formula directly,
        as Arepo reports values for the eEOS. """
        return self.rates.get_neutral_fraction(self.get_code_rhoH(bar), self._get_ienergy_rescaled(bar), helium=self._get_helium(bar))

    def get_temp(self,bar):
        """Compute temperature (in K) from internal energy."""
        #Internal energy units are 10^-10 erg/g
        return self.rates.get_temp(self.get_code_rhoH(bar), self._get_ienergy_rescaled(bar), helium=self._get_helium(bar))

    def _get_helium(self, bar):
        """Get the helium abundance"""
        try:
            return np.array(bar["GFM_Metals"][:,1], dtype=np.float32)
        except KeyError:
            return 0.24

    def _get_ienergy_rescaled(self, bar):
        """Get the internal energy, rescaled to give the desired equation of state.
        Technically the e. of s. normally used is:
            T = T_0 (rho / rho_0)^(gamma-1)
        However in photoionisation equilibrium the electron density depends very weakly
        on the temperature, and so T/T_0 = U/U_0
        So we can just rescale the internal energy:
        when T_0 -> T_0' U -> U * T_0'/T_0.
        Ditto for gamma, when gamma -> gamma' we have:
        U -> U (rho/rho_0) ^(gamma'-gamma)
        Note this means that if any particle lies off the original equation of state,
        it lies off the new one by a similar amount; the dispersion is preserved!
        """
        ienergy=np.array(bar["InternalEnergy"])
        #Adjust temperature by desired factor, to give desired equation of state.
        ienergy *= self.temp_factor
        #Adjust slope by same factor: note use gamma_factor -1 so gamma_factor = 1 means no change.
        if self.gamma_factor != 1.:
            density = self.get_code_rhoH(bar)
            ienergy *= (density/self.density0)**self.gamma_factor-1.
        return ienergy

class RescaledSpectra(spectra.Spectra):
    """Class which is the same as Spectra but with an adjusted neutral fraction and temperature of the particles before spectra are generated."""
    def __init__(self, photo_factor = 1., temp_factor = 1., gamma_factor = 1., *args, **kwargs):
        self.photo_factor = photo_factor
        self.temp_factor = temp_factor
        super().__init__(self, args, kwargs)
        self.gasprop = RescaledGasProperties(redshift = self.red, photo_factor = photo_factor, temp_factor = temp_factor, gamma_factor=gamma_factor, hubble = self.hubble, omega_matter = self.OmegaM)
