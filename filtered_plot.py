"""Make a plot which shows the effect of dense particles,
for which the mean flux rescaling is not fully accurate, on the flux power spectrum."""
import numpy as np
from fake_spectra import spectra
import matplotlib
matplotlib.use("PDF")
import matplotlib.pyplot as plt

class FilteredSpectra(spectra.Spectra):
    """Class which makes a flux power spectrum with the dense (helium not fully ionized) particles removed."""

    def __init__(self, *args, filter_dens = 4e-4, **kwargs):
        super().__init__(*args, **kwargs)
        #in atoms/cm^3
        self.filter_dens = filter_dens

    def _read_particle_data(self, fn, elem, ion, get_tau):
        """Read the particle data for a single interpolation, filter out particles above some mass"""
        (pos, vel, elem_den, temp, hh, amumass) = super()._read_particle_data(fn, elem, ion, get_tau)

        den=self.gasprop.get_code_rhoH(0, segment=fn).astype(np.float32)
        pos2 = self.snapshot_set.get_data(0,"Position",segment = fn).astype(np.float32)
        hh2 = self.snapshot_set.get_smooth_length(0,segment=fn).astype(np.float32)
        ind = self.particles_near_lines(pos2, hh2,self.axis,self.cofm)
        den = den[ind]
        ii2 = np.where(den < self.filter_dens)
        if get_tau:
            return (pos[ii2], vel[ii2], elem_den[ii2], temp[ii2], hh[ii2], amumass)
        else:
            return (pos[ii2], vel, elem_den[ii2], temp, hh[ii2], amumass)


def make_plot(num, base):
    """Make the plot."""
    spec = spectra.Spectra(num, base, None, None, savefile="lya_forest_spectra.hdf5", sf_neutral=False)

    try:
        filtspec = FilteredSpectra(num, base, spec.cofm, spec.axis, savefile="filtered_lya_forest_spectra.hdf5", sf_neutral=False, reload_file=False, res=10.)
    except IOError:
        filtspec = FilteredSpectra(num, base, spec.cofm, spec.axis, savefile="filtered_lya_forest_spectra.hdf5", sf_neutral=False, reload_file=True, res=10.)
        filtspec.get_tau("H", 1, 1215)
        filtspec.get_col_density("H", 1)
        filtspec.save_file()

    print("Mass fraction filtered: ", np.sum(filtspec.get_col_density("H", 1))/np.sum(spec.get_col_density("H", 1)))

    #Note this is done without mean flux rescaling
    (kf, pkf) = spec.get_flux_power_1D()
    (fikf, fipkf) = filtspec.get_flux_power_1D()

    assert np.all(kf == fikf)

    plt.semilogx(kf, fipkf/pkf, ls="-")
    plt.xlim(1.e-3, 0.03)
    plt.axvspan(1.084e-3, 1.95e-2, facecolor='grey', alpha=0.3)
    plt.ylim(0.9,1.0)
    plt.xlabel(r'$k$ ($\mathrm{s}\,\mathrm{km}^{-1}$)')
    plt.ylabel(r'$P_\mathrm{F}(k)$ ratio')
    plt.tight_layout()
    plt.title(r"Ratio removing particles with $\rho > 3 \times 10^{-4}$")

    plt.savefig("plots/filtered_power.pdf")


if __name__ == "__main__":
    make_plot(10, "simulations/hires_s8_test/ns0.97As2.2e-09heat_slope0.083heat_amp0.92hub0.69/output")
