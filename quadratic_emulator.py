#!/usr/bin/env python
# vim: set fileencoding=UTF-8 :

"""
An emulator to interpolate between flux power spectra computed for the Lyman alpha forest.
Fits the change in the flux power spectrum with a quadratic function.
"""

import math
import numpy as np
from coarse_grid import Emulator
from gpemulator import SkLearnGP
import flux_power

def Hubble(zz, om, H0):
    """ Hubble parameter. Hubble(Redshift) """
    #Conversion factor between s/km and h/Mpc is (1+z)/H(z)
    return 100*H0*math.sqrt(om*(1+zz)**3+(1-om))

class QuadraticPoly(SkLearnGP):
    """
    Given a set of simulations with different parameters, produce the expected quantity interpolated to a new set of parameters.
    Takes as arguments:
        knots - a list of knots, each of which holds directories which vary every parameter in turn.
        bestfit - Best fit simulation. We use the input simulations to compute changes in the interpolated quantity,
                     and then multiply to get the best fit.
                     The best fit used in each knot to compute the different coefficients need not be this simulation!
    """
    def __init__(self, *args, **kwargs):
        self.bfnum = 0
        super().__init__(*args, **kwargs)
        self.intol = 1e-2

    def _get_interp(self, flux_vectors):
        """Do the actual interpolation. Called in parent's __init__"""
        self.tables = {}
        self.bestfv = flux_vectors[self.bfnum]
        self.bestpar = self.params[self.bfnum,:]
        for pp in range(np.shape(self.params)[1]):
            self.tables[pp] = self._calc_coeffs(flux_vectors,self.params[:,pp], pp)

    def predict(self, params):
        """Get the interpolated quantity by evaluating the quadratic fit"""
        #Interpolate onto desired bins
        #Do parameter correction
        newq = np.ones_like(self.bestfv)
        assert np.shape(params) == (1,np.shape(self.bestpar)[0])
        dpp = params[0] - self.bestpar
        for pp,dp in enumerate(dpp):
            newq += self.tables[pp][:,0]*dp**2 +self.tables[pp][:,1]*dp
        mean = newq * self.bestfv
        std = 0.001*np.ones_like(mean)
        return [mean,], [std,]

    def _flux_deriv(self, PFdif, pdif):
        """Calculate the flux-derivative for a single parameter change"""
        assert np.size(pdif) == np.size(PFdif)
        mat=np.vstack([pdif**2, pdif] ).T
        (derivs, _,_, _)=np.linalg.lstsq(mat, PFdif, rcond=None)
        return derivs

    def _get_changes(self, flux_vectors, params, pind):
        """Get the change in parameters, delta p and the corresponding change
        in the flux power spectrum, delta P_F, rebinned to match the desired output bins"""
        dfv = flux_vectors/self.bestfv - 1.
        #Compute change in parameters
        dparams  = params - self.bestpar[pind]
        #Find only those positions where this parameter changed.
        ind = np.where(np.abs(dparams) > 1e-3*np.abs(self.bestpar[pind]))
        assert (pind > 0 and len(ind[0]) == 4) or (pind == 0 and len(ind[0]) == 9)
        return (dfv[ind], dparams[ind])

    def _calc_coeffs(self, flux_vectors, params, pind):
        """
            Calculate the flux derivatives for a single redshift, for a single shifting parameter
            Input:
                sims - list of simulations. Must be of Simulation type above, ie, define a get_quantity, a get_params and a get_bins.
            Output: (kbins d2P...kbins dP (flat vector of length 2xkbins))
        """
        #Get the change in the interpoaltion value with parameter
        (dfv, dparams) = self._get_changes(flux_vectors, params,pind)
        #Pass each k value to flux_deriv in turn.
        # Format of returned data from flux_derivs is (a,b) where it fits to:
        # dto_interp = a params**2 + b params
        results = np.array([self._flux_deriv(dfv[:,k], dparams) for k in range(np.shape(dfv)[1])])
        #So results should have shape
        assert np.shape(results) == (np.size(self.bestfv), 2)
        return results

class QuadraticEmulator(Emulator):
    """Do emulation with a simple quadratic interpolation."""
    def build_params(self, nsamples,limits = None, use_existing=False):
        """Build a list of directories and parameters from a hypercube sample"""
        if use_existing:
            raise ValueError("Refinement not supported")
        #Find centroid.
        if limits is None:
            limits = self.param_limits
        centroid = (limits[:,0]+limits[:,1])/2.
        nparams = len(limits[:,0])
        #Change one parameter at a time.
        n1par = (nsamples-1)//nparams
        sims = [centroid,]
        for pp in range(nparams):
            dp = (limits[pp,1] - limits[pp, 0])/n1par
            pthis = np.zeros_like(centroid)
            pthis[pp] = 1
            for nn in range(n1par//2):
                up = centroid+dp*(nn+1)*pthis
                sims.append(up)
                down = centroid-dp*(nn+1)*pthis
                sims.append(down)
        return np.array(sims)

    def get_emulator(self, max_z=4.2):
        """ Build an emulator for the desired k_F and our simulations.
            kf gives the desired k bins in s/km.
            Mean flux rescaling is handled (if mean_flux=True) as follows:
            1. A set of flux power spectra are generated for every one of a list of possible mean flux values.
            2. Each flux power spectrum in the set is rescaled to the same mean flux.
            3.
        """
        gp = self._get_custom_emulator(emuobj=QuadraticPoly, max_z=max_z)
        return gp

    def get_flux_vectors(self, max_z=4.2, kfunits="kms"):
        """Get the desired flux vectors and their parameters.
        This is subclassed so that we only change the mean flux parameters around the best fit, central, model."""
        pvals = self.get_parameters()
        nparams = np.shape(pvals)[1]
        assert nparams == len(self.param_names)
        myspec = flux_power.MySpectra(max_z=max_z, max_k=self.maxk)
        mean_fluxes = self.mf.get_mean_flux(myspec.zout)
        #First get best fit
        mfind = np.shape(mean_fluxes)[0]//2
        medmf = mean_fluxes[mfind,:]
        #List of indices that aren't the best-fit mean flux
        nmfind = list(range(0,mfind))+list(range(mfind+1,np.shape(mean_fluxes)[0]))
        mean_fluxes = mean_fluxes[nmfind]
        aparams = pvals
        #Note this gets tau_0 as a linear scale factor from the observed power law
        dpvals = self.mf.get_params()
        if dpvals is not None:
            aparams = [np.concatenate([dpvals[mfind],pvals[0]]),]
            aparams += [np.concatenate([dp, pvals[0]]) for dp in dpvals[nmfind]]
            aparams += [np.concatenate([dpvals[mfind], pv]) for pv in pvals[1:]]
            aparams = np.array(aparams)
        try:
            kfmpc, kfkms, flux_vectors = self.load_flux_vectors(aparams, savefile="quadratic_flux_vectors.hdf5")
        except (AssertionError, OSError):
            powers = [self._get_fv(pp, myspec) for pp in pvals]
            #First we want the best-fit
            flux_vectors = [powers[0].get_power_native_binning(mean_fluxes = medmf),]
            #Now best fit parameters, with varying mean flux values.
            flux_vectors += [powers[0].get_power_native_binning(mean_fluxes = mef) for mef in mean_fluxes]
            #Now the rest
            flux_vectors += [ps.get_power_native_binning(mean_fluxes = medmf) for ps in powers[1:]]
            flux_vectors = np.array(flux_vectors)
            #'natively' binned k values in km/s units as a function of redshift
            kfkms = [powers[0].get_kf_kms(),]
            kfkms += [powers[0].get_kf_kms() for mef in mean_fluxes]
            kfkms += [ps.get_kf_kms() for ps in powers[1:]]
            #Same in all boxes
            kfmpc = powers[0].kf
            assert np.all(np.abs(powers[0].kf/ powers[-1].kf-1) < 1e-6)
            self.save_flux_vectors(aparams, kfmpc, kfkms, flux_vectors, savefile="quadratic_flux_vectors.hdf5")
        assert np.shape(flux_vectors)[0] == np.shape(aparams)[0]
        if kfunits == "kms":
            kf = kfkms
        else:
            kf = kfmpc
        return aparams, kf, flux_vectors
