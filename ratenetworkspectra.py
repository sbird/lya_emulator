"""Modified versions of gas properties and spectra that use the rate network."""

import numpy as np
from scipy.interpolate import interp2d
from fake_spectra import gas_properties
from fake_spectra import spectra
from rate_network import RateNetwork

def RateNetworkGasTest():
    """Test that the spline is working."""
    gasprop = RateNetworkGas(3, None)
    randd = (np.max(gasprop.densgrid) - np.min(gasprop.densgrid)) * np.random.random(size=2000) + np.min(gasprop.densgrid)
    randi = (np.max(gasprop.ienergygrid) - np.min(gasprop.ienergygrid)) * np.random.random(size=2000) + np.min(gasprop.ienergygrid)
    for dd, ii in zip(randd, randi):
        spl = gasprop.spline(dd, ii)
        rate = np.log(gasprop.rates.get_neutral_fraction(np.exp(dd), ii))
        assert np.abs(spl - rate) < 1e-5

class RateNetworkGas(gas_properties.GasProperties):
    """Replace the get_reproc_HI function with something that solves the rate network. Optionally can also do self-shielding."""
    def __init__(self, redshift, absnap, hubble=0.71, fbar=0.17, units=None, sf_neutral=True, selfshield=False, photo_factor=1):
        super().__init__(redshift, absnap, hubble=hubble, fbar=fbar, units=units, sf_neutral=sf_neutral)
        self.rates = RateNetwork(redshift, photo_factor = photo_factor, f_bar = fbar, cool="KWH", recomb="C92", selfshield=selfshield, treecool_file="TREECOOL_fg_dec11")
        self.temp_factor = 1
        self.gamma_factor = 1
        #Build interpolation
        sz = 10**3
        self.densgrid = np.log(np.logspace(-7, -2, 2*sz))
        self.ienergygrid = np.linspace(20, 1000, sz)
        dgrid, egrid = np.meshgrid(self.densgrid, self.ienergygrid)
        self.lh0grid = np.log(self.rates.get_neutral_fraction(np.exp(dgrid), egrid))
        #We assume primordial helium
        self.spline = interp2d(self.densgrid, self.ienergygrid, self.lh0grid, kind='cubic')

    def get_reproc_HI(self, part_type, segment):
        """Get a neutral hydrogen fraction using a rate network which reads temperature and density of the gas."""
        #expecting units of atoms/cm^3
        density = self.get_code_rhoH(part_type, segment)
        #expecting units of 10^-10 ergs/g
        ienergy = self.absnap.get_data(part_type, "InternalEnergy", segment=segment)*self.units.UnitInternalEnergy_in_cgs/1e10
        outside = density > np.max(self.densgrid) + density < np.min(self.densgrid) + ienergy > np.max(self.ienergygrid) + ienergy < np.min(self.ienergygrid)
        ii = np.where(np.logical_not(outside))
        nh0 = np.zeros_like(density)
        nh0[ii] = np.exp(self.spline(np.log(density[ii]), ienergy[ii]))
        ii = np.where(outside)
        if np.size(ii) > 0.05*np.size(nh0):
            print("Interpolation range misses %d of particles." % np.size(ii)/np.size(nh0))
            nh0[ii] = np.array([self.rates.get_neutral_fraction(dd, ii) for (dd, ii) in zip(density[ii], ienergy[ii])])
        return nh0

    def _get_ienergy_rescaled(self, density, ienergy, density0):
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
        #Adjust temperature by desired factor, to give desired equation of state.
        ienergy *= self.temp_factor
        #Adjust slope by same factor: note use gamma_factor -1 so gamma_factor = 1 means no change.
        if self.gamma_factor != 1.:
            ienergy *= (density/density0)**self.gamma_factor-1.
        return ienergy

class RateNetworkSpectra(spectra.Spectra):
    """Generate spectra with a neutral fraction from a rate network"""
    def __init__(self, *args, photo_factor = 1, selfshield=False, **kwargs):
        super().__init__(*args, **kwargs)
        try:
            self.gasprop = RateNetworkGas(redshift = self.red, absnap = self.snapshot_set, hubble=self.hubble, units = self.units, sf_neutral=False, photo_factor = photo_factor, selfshield=selfshield)
        except AttributeError:
            pass
