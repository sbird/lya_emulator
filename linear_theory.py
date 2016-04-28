"""A module for doing linear theory computations of the Lyman-alpha forest.
This uses the 'biased tracer' formalism. It computes the 3D power spectrum and is intended only for testing!"""

import math
import numpy as np
import camb

def flux_power_3d(matter_power, mu, bias_flux=-0.14, beta_flux=-0.2):
    """The 3D flux power from the matter power, assuming the forest is a biased tracer."""
    return (bias_flux + (1+mu**2)*beta_flux )**2 * matter_power

def flux_power_1d(matter_power, kvals, *, bias_flux=-0.14, beta_flux=-0.2):
    """The 1D flux power spectrum from the matter power, the integral of the 3D.
    Result has units of L."""
    P3D = flux_power_3d(matter_power, 1, bias_flux=bias_flux, beta_flux=beta_flux)
    return kvals, np.array([np.trapz(P3D[i:]*kvals[i:], kvals[i:]) for i in range(np.size(kvals))])/math.pi/2

def matter_power(*, hub = 0.7, omega_b = 0.049, omega_c = 0.25, ns=0.965, As = 2.41e-9, zz=3.):
    """Get a matter power spectrum using CAMB's python interface."""
    #Set up a new set of parameters for CAMB
    pars = camb.CAMBparams()
    #This function sets up CosmoMC-like settings, with one massive neutrino and helium set using BBN consistency
    pars.set_cosmology(H0=hub*100, ombh2=omega_b*hub**2, omch2=omega_c*hub**2, mnu=0., omk=0, tau=0.06)
    pars.InitPower.set_params(As=As, ns=ns, r=0)
    pars.set_matter_power(redshifts=[zz,], kmax=10)
    assert pars.validate()
    pars.NonLinear = camb.model.NonLinear_none
    results = camb.get_results(pars)
    kh, _, pk = results.get_matter_power_spectrum(minkh=1e-2, maxkh=10, npoints = 200)
    return kh, pk[0]

def get_flux_power(*, hub = 0.7, omega_b = 0.049, omega_c = 0.25, ns=0.965, As = 2.41e-9, zz=3., bias_flux=-0.14, beta_flux=-0.2):
    """Get a flux power spectrum from cosmology."""
    (kh, pk) = matter_power(hub = hub, omega_b = omega_b, omega_c = omega_c, ns=ns, As = As, zz=zz)
    pf = flux_power_1d(pk, kh, bias_flux=bias_flux, beta_flux=beta_flux)
    return pf
