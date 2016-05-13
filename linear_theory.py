"""A module for doing linear theory computations of the Lyman-alpha forest.
This uses the 'biased tracer' formalism. It computes the 3D power spectrum and is intended only for testing!"""

import math
import numpy as np
import scipy.interpolate
import camb

def hubble(zz, omega_m, hub=0.7):
    """Hubble expansion at redshift zz"""
    return hub*100 * np.sqrt(omega_m * (1+zz)**3 + (1-omega_m))

def flux_power_3d(matpow, mu, bias_flux=-0.14, beta_flux=-0.2):
    """The 3D flux power from the matter power, assuming the forest is a biased tracer."""
    return (bias_flux + (1+mu**2)*beta_flux )**2 * matpow

def flux_power_1d(matpow, kvals, *, bias_flux=-0.14, beta_flux=-0.2):
    """The 1D flux power spectrum from the matter power, the integral of the 3D.
    Result has units of L."""
    P3D = flux_power_3d(matpow, 1, bias_flux=bias_flux, beta_flux=beta_flux)
    return kvals, np.array([np.trapz(P3D[i:]*kvals[i:], kvals[i:]) for i in range(np.size(kvals))])/math.pi/2

def matter_power(*, hub = 0.7, omega_b = 0.049, omega_c = 0.25, ns=0.965, As = 2.41e-9, zz=3.):
    """Get a matter power spectrum using CAMB's python interface."""
    #Set up a new set of parameters for CAMB
    pars = camb.CAMBparams()
    #This function sets up CosmoMC-like settings, with one massive neutrino and helium set using BBN consistency
    pars.set_cosmology(H0=hub*100, ombh2=omega_b*hub**2, omch2=omega_c*hub**2, mnu=0., omk=0, tau=0.06)
    pars.InitPower.set_params(As=As, ns=ns, r=0)
    if len(zz) == 1:
        zz = [zz,]
    pars.set_matter_power(redshifts=zz, kmax=10)
    assert pars.validate()
    pars.NonLinear = camb.model.NonLinear_none
    results = camb.get_results(pars)
    kh, _, pk = results.get_matter_power_spectrum(minkh=1e-2, maxkh=10, npoints = 200)
    return kh, pk

def get_flux_power(*, kf, zz, hub = 0.7, omega_b = 0.049, omega_c = 0.25, ns=0.965, As = 2.41e-9, bias_flux=-0.14, beta_flux=-0.2):
    """Get a flux power spectrum from cosmology."""
    (kh, pks) = matter_power(hub = hub, omega_b = omega_b, omega_c = omega_c, ns=ns, As = As, zz=zz)
    results = np.array([flux_power_1d(pk[::10], kh[::10], bias_flux=bias_flux, beta_flux=beta_flux) for pk in pks])
    zz = np.array(zz)
    kvals = results[0,0,:]
    pf = results[:,1,:]
    newpf = [scipy.interpolate.interp1d(kvals, ppf) for ppf in pf]
    #Convert units from comoving Mpc to km/s:
    convert = hubble(zz, hub=hub, omega_m = omega_b+omega_c) / (1+zz)
    #from 1/(km/s) to 1/ Mpc
    kmpc = [kf * cc for cc in convert]
    #Convert back to km/s
    pfbin = (np.array([nn(km) for nn,km in zip(newpf,kmpc)]).T * convert).T
    assert np.shape(pfbin) == (np.size(zz), np.size(kf))
    #Return binned onto observed data
    return pfbin
