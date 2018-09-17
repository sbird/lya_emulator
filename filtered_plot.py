"""Make a plot which shows the effect of dense particles,
for which the mean flux rescaling is not fully accurate, on the flux power spectrum."""
import numpy as np
from fake_spectra import spectra
from fake_spectra import fluxstatistics as fstat
from fake_spectra.ratenetworkspectra import RateNetworkSpectra
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


def filtering_effect_plot(num, base):
    """Plot the effect of filtering out the largest density particles."""
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
    plt.clf()

def get_mean_flux_effect(num, base, collisional=True):
    """Get the effect of rescaling the mean flux vs solving a rate network on the flux power"""
    spec = spectra.Spectra(num, base, None, None, savefile="lya_forest_spectra.hdf5", sf_neutral=False)
    #Get the mean flux scaling factor
    mf = np.exp(-fstat.obs_mean_tau(spec.red))
    #mean flux scaling is 1/UVB amp in photo-ion equilib.
    photo_factor = 1./fstat.mean_flux(spec.get_tau("H",1,1215),mean_flux_desired=mf)
    (kf, pkf) = spec.get_flux_power_1D(mean_flux_desired=mf)
    sfile = "lya_forest_spectra_uvb"
    if not collisional:
        sfile += "_nocol"
    sfile += ".hdf5"
    try:
        rnspecph = RateNetworkSpectra(num, base, spec.cofm, spec.axis, savefile=sfile, sf_neutral=False, reload_file=False, res=10.,photo_factor=photo_factor, collisional=collisional)
    except IOError:
        rnspecph = RateNetworkSpectra(num, base, spec.cofm, spec.axis, savefile=sfile, sf_neutral=False, reload_file=True, res=10.,photo_factor=photo_factor, collisional=collisional)
        rnspecph.get_tau("H", 1, 1215)
        rnspecph.save_file()
    #Get without mean flux rescaling
    (kfph, pkfph) = rnspecph.get_flux_power_1D(mean_flux_desired=None)
    assert np.all(kf == kfph)
    #Check that the mean flux of the output is roughly the same
    mfph = np.mean(np.exp(-rnspecph.get_tau("H",1,1215)))
    assert np.abs(mfph/mf -1 ) < 0.03
    print("Redshift %g. Mean flux from UVB %g, rescaled %g. UVB factor %g." % (spec.red, mfph, mf, photo_factor))
    return kf, pkf/pkfph

def mean_flux_effect_plot(base, collisional=True, name=""):
    """Plot the effect of doing mean flux rescaling vs a UVB."""
    (kf22, dpkf22) = get_mean_flux_effect(11, base, collisional=collisional)
    (kf24, dpkf24) = get_mean_flux_effect(10, base, collisional=collisional)
    (kf3, dpkf3) = get_mean_flux_effect(7, base, collisional=collisional)
    (kf35, dpkf35) = get_mean_flux_effect(5, base, collisional=collisional)

    plt.semilogx(kf22, dpkf22, ls="-", label=r"$z=2.2$", color="blue")
    plt.semilogx(kf24, dpkf24, ls="--", label=r"$z=2.4$", color="red")
    plt.semilogx(kf3, dpkf3, ls=":", label=r"$z=3$", color="black")
    plt.semilogx(kf35, dpkf35, ls="-.", label=r"$z=3.4$", color="brown")

    plt.xlim(1.e-3, 0.1)
    plt.axvspan(1.084e-3, 1.95e-2, facecolor='grey', alpha=0.2)
    plt.ylim(0.95,1.01)
    plt.xlabel(r'$k$ ($\mathrm{s}\,\mathrm{km}^{-1}$)')
    plt.ylabel(r'$P_\mathrm{F}(k)$ ratio')
    plt.tight_layout()
    plt.legend(loc="lower right")
    #plt.title(r"Flux power spectru")
    plt.savefig("plots/mean_flux_power"+name+".pdf")
    plt.clf()

def mean_flux_gamma_dep(base1, base2, collisional=True, name=""):
    """Plot the effect of doing mean flux rescaling vs a UVB."""
    (kf22, dpkf22) = get_mean_flux_effect(11, base1, collisional=collisional)
    (kf222, dpkf222) = get_mean_flux_effect(11, base2, collisional=collisional)
    (kf3, dpkf3) = get_mean_flux_effect(7, base1, collisional=collisional)
    (kf23, dpkf23) = get_mean_flux_effect(7, base2, collisional=collisional)

    plt.semilogx(kf22, dpkf22, ls="-", label=r"$z=2.2$, $\gamma = 1.6$", color="blue")
    plt.semilogx(kf222, dpkf222, ls="--", label=r"$z=2.2$, $\gamma = 1.3$", color="blue")
    plt.semilogx(kf3, dpkf3, ls="-.", label=r"$z=3$, $\gamma = 1.6$", color="black")
    plt.semilogx(kf23, dpkf23, ls=":", label=r"$z=3$, $\gamma = 1.3$", color="black")

    plt.xlim(1.e-3, 0.1)
    plt.axvspan(1.084e-3, 1.95e-2, facecolor='grey', alpha=0.2)
    plt.ylim(0.95,1.01)
    plt.xlabel(r'$k$ ($\mathrm{s}\,\mathrm{km}^{-1}$)')
    plt.ylabel(r'$P_\mathrm{F}(k)$ ratio')
    plt.tight_layout()
    plt.legend(loc="lower right")
    #plt.title(r"Flux power spectru")
    plt.savefig("plots/mean_flux_power_gamma"+name+".pdf")
    plt.clf()


if __name__ == "__main__":
    #filtering_effect_plot(10, "simulations/hires_s8_test/ns0.97As2.2e-09heat_slope0.083heat_amp0.92hub0.69/output")
    mean_flux_effect_plot("simulations/hires_s8_test/ns0.97As2.2e-09heat_slope0.083heat_amp0.92hub0.69/output")
    mean_flux_gamma_dep("simulations/hires_s8_test/ns0.97As2.2e-09heat_slope0.083heat_amp0.92hub0.69/output", "simulations/hires_s8_quadratic/ns1.1As2.1e-09heat_slope-0.5heat_amp1hub0.7/output")
    #mean_flux_effect_plot("simulations/hires_s8_test/ns0.97As2.2e-09heat_slope0.083heat_amp0.92hub0.69/output", collisional=False, name="_nocolis")
