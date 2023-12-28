"""Small script to compare flux power spectra around each axis"""
from os import path
import numpy as np
import scipy.interpolate
from fake_spectra import spectra
from fake_spectra import fluxstatistics as fstat
from lyaemu.lyman_data import BOSSData
import matplotlib.pyplot as plt

def rebin_power_to_kms(kfkms, kfmpc, flux_powers):
    """Rebins a power spectrum to constant km/s bins.
    Bins larger than the box are discarded. The return type is thus a list,
    with each redshift bin having potentially different lengths."""
    # interpolate simulation output for averaging
    rebinned = scipy.interpolate.interpolate.interp1d(kfmpc, flux_powers)
    new_flux_powers = rebinned(kfkms)
    # final flux power array
    return new_flux_powers

base = path.expanduser("~/shared/Lya_emu_spectra/emu_full/ns0.972Ap1.69e-09herei3.87heref2.65alphaq2.12hub0.722omegamh20.144hireionz7.53bhfeedback0.0507/output/")

for nn in range(4, 23):
    if nn == 17:
        continue
    spec = spectra.Spectra(nn, base, None, None, savefile="lya_forest_spectra_grid_480.hdf5", res=None)
    tau = spec.get_tau("H", 1, 1215)
    axis = spec.axis
    flux_power = []
    for ax in range(1,4):
        ii = np.where(axis == ax)[0]
        (kf, avg_flux_power) = fstat.flux_power(tau[ii,:], spec.vmax, spec_res=spec.spec_res, mean_flux_desired=None, window=False)
        flux_power.append(avg_flux_power)
    flux = np.array(flux_power)

    lyd=BOSSData(datafile="dr14")
    kfkms = lyd.get_kf()
    newflux1 = rebin_power_to_kms(kfkms=kfkms, kfmpc=kf[1:], flux_powers=flux[1,1:]/flux[0,1:])
    newflux2 = rebin_power_to_kms(kfkms=kfkms, kfmpc=kf[1:], flux_powers=flux[2,1:]/flux[0,1:])
    plt.semilogx(kfkms, (newflux1-1), label="y/x", ls="--", color="brown")
    plt.semilogx(kfkms, (newflux2-1), label="z/x", ls="-", color="grey")
    plt.xlabel(r"$k_F$ (s/km)")
    plt.ylabel("Axis ratio")
    plt.title("z=%.2g" % spec.red)
    plt.savefig("cv_axes-%.2g.pdf" % spec.red)
    plt.clf()
