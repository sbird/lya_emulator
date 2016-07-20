"""Module for subclassing spectra to generate spectra with a different
photo-ionising background. This works by generating new spectra.
The neutral fraction of each particle is adjusted aas it goes through the spectral generator.
This should avoid the need to rescale to the mean flux, and may be substantially more accurate."""

import numpy as np
import spectra
import rate_network
import unitsystem

class RescaledGasProperties(object):
    """Class which does the actual rescaling of the temperature."""
    def __init__(self, photo_factor, redshift,hubble = 0.71, f_bar=0.17, units = None):
        if units is not None:
            self.units = units
        else:
            self.units = unitsystem.UnitSystem()
        self.rates = rate_network.RateNetwork(redshift=redshift, photo_factor = photo_factor,f_bar = f_bar)
        #Some constants and unit systems
        #self.hy_mass = 0.76 # Hydrogen massfrac
        self.gamma=5./3
        #Boltzmann constant (cgs)
        self.boltzmann=1.38066e-16
        self.hubble = hubble
        self.redshift = redshift

    def get_code_rhoH(self,bar):
        """Convert density to physical atoms /cm^3: internal gadget density unit is h^2 (1e10 M_sun) / kpc^3"""
        nH = np.array(bar["Density"])
        conv = np.float32(self.units.UnitDensity_in_cgs*self.hubble**2/(self.units.protonmass)*(1+self.redshift)**3)
        #Convert to physical
        return nH*conv

    def get_reproc_HI(self, bar):
        """Get a neutral hydrogen *fraction* using values given by Arepo
        which are based on Rahmati 2012 if UVB_SELF_SHIELDING is on.
        Above the star formation density use the Rahmati fitting formula directly,
        as Arepo reports values for the eEOS. """
        return self.rates.get_neutral_fraction(self.get_code_rhoH(bar), np.array(bar["InternalEnergy"]), helium=self._get_helium(bar))

    def get_temp(self,bar):
        """Compute temperature (in K) from internal energy."""
        #Internal energy units are 10^-10 erg/g
        ienergy=np.array(bar["InternalEnergy"])
        return self.rates.get_temp(self.get_code_rhoH(bar), ienergy, helium=self._get_helium(bar))

    def _get_helium(self, bar):
        """Get the helium abundance"""
        try:
            return np.array(bar["GFM_Metals"][:,1], dtype=np.float32)
        except KeyError:
            return 0.24

class RescaledSpectra(spectra.Spectra):
    """Class which is the same as Spectra but with an adjusted neutral fraction and temperature of the particles before spectra are generated."""
    def __init__(self, photo_factor = 1., *args, **kwargs):
        self.photo_factor = photo_factor
        super().__init__(self, args, kwargs)
        self.gasprop = RescaledGasProperties(redshift = self.red, photo_factor = self.photo_factor)
