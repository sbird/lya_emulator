"""Small script to compare flux power spectra around each axis"""
import h5py
import numpy as np
import os.path as path
from fake_spectra import spectra
from fake_spectra import fluxstatistics as fstat

base = path.expanduser("~/shared/Lya_emu_spectra/emu_full/ns0.972Ap1.69e-09herei3.87heref2.65alphaq2.12hub0.722omegamh20.144hireionz7.53bhfeedback0.0507/output/")
spec = spectra.Spectra(14, base, None, None, savefile="lya_forest_spectra_grid_480.hdf5", res=None)
tau = spec.get_tau("H", 1, 1215)
axis = spec.axis
flux_power = []
for ax in range(1,4):
    ii = np.where(axis == ax)[0]
    (kf, avg_flux_power) = fstat.flux_power(tau[ii,:], spec.vmax, spec_res=spec.spec_res, mean_flux_desired=None, window=False)
    flux_power.append(avg_flux_power)

flux_power = np.array(flux_power)
with h5py.File("partial_flux.hdf5", 'w') as ff:
    ff["fluxes"] = flux_power
    ff["kf"] = kf
