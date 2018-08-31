"""Modified versions of gas properties and spectra that """

from fake_spectra import gas_properties
from fake_spectra import spectra
from rate_network import RateNetwork

class RateNetworkGas(gas_properties.GasProperties):
    """Replace the get_reproc_HI function with something that solves the rate network. Optionally can also do self-shielding."""
    def __init__(self, redshift, absnap, hubble=0.71, fbar=0.17, units=None, sf_neutral=True, selfshield=False, photo_factor=1):
        super().__init__(redshift, absnap, hubble=hubble, fbar=fbar, units=units, sf_neutral=sf_neutral)
        self.rates = RateNetwork(redshift, photo_factor = photo_factor, f_bar = fbar, cool="KWH", recomb="C92", selfshield=selfshield)

    def get_reproc_HI(self, part_type, segment):
        """Get a neutral hydrogen fraction using a rate network which reads temperature and density of the gas."""
        #expecting units of atoms/cm^3
        density = self.get_code_rhoH(part_type, segment)
        #expecting units of 10^-10 ergs/g
        ienergy = self.absnap.get_data(part_type, "InternalEnergy", segment=segment)*units.UnitInternalEnergy_in_cgs/1e10
        #We assume primordial helium
        nh0 = self.rates.get_neutral_fraction(density, ienergy)
        return nh0

class RateNetworkSpectra(spectra.Spectra):
    """Generate spectra with a neutral fraction from a rate network"""
    def __init__(self, *args, photo_factor = 1, selfshield=False, **kwargs):
        super().__init__(*args, **kwargs)
        try:
            self.gasprop = RateNetworkGas(redshift = self.redshift, absnap = self.absnap, hubble=self.hubble, units = self.units, sf_neutral=False, photo_factor = photo_factor, selfshield=selfshield)
        except AttributeError:
            pass
