"""Modified versions of gas properties and spectra that use the rate network."""

from fake_spectra import gas_properties
from fake_spectra import spectra
from rate_network import RateNetwork

class RateNetworkGas(gas_properties.GasProperties):
    """Replace the get_reproc_HI function with something that solves the rate network. Optionally can also do self-shielding."""
    def __init__(self, redshift, absnap, hubble=0.71, fbar=0.17, units=None, sf_neutral=True, selfshield=False, photo_factor=1):
        super().__init__(redshift, absnap, hubble=hubble, fbar=fbar, units=units, sf_neutral=sf_neutral)
        self.rates = RateNetwork(redshift, photo_factor = photo_factor, f_bar = fbar, cool="KWH", recomb="C92", selfshield=selfshield, treecool_file="TREECOOL_fg_dec11")
        self.temp_factor = 1
        self.gamma_factor = 1

    def get_reproc_HI(self, part_type, segment):
        """Get a neutral hydrogen fraction using a rate network which reads temperature and density of the gas."""
        #expecting units of atoms/cm^3
        density = self.get_code_rhoH(part_type, segment)
        #expecting units of 10^-10 ergs/g
        ienergy = self.absnap.get_data(part_type, "InternalEnergy", segment=segment)*self.units.UnitInternalEnergy_in_cgs/1e10
        #We assume primordial helium
        nh0 = self.rates.get_neutral_fraction(density, ienergy)
        return nh0

    def _get_ienergy_rescaled(self, ienergy):
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
            density = self.get_code_rhoH(bar)
            ienergy *= (density/self.density0)**self.gamma_factor-1.
        return ienergy

class RateNetworkSpectra(spectra.Spectra):
    """Generate spectra with a neutral fraction from a rate network"""
    def __init__(self, *args, photo_factor = 1, selfshield=False, **kwargs):
        super().__init__(*args, **kwargs)
        try:
            self.gasprop = RateNetworkGas(redshift = self.red, absnap = self.snapshot_set, hubble=self.hubble, units = self.units, sf_neutral=False, photo_factor = photo_factor, selfshield=selfshield)
        except AttributeError:
            pass
