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

    def _get_interp(self, params, flux_vectors, bfnum=0):
        """Do the actual interpolation. Called in parent's __init__"""
        self.tables = {}
        self.bestfv = flux_vectors[bfnum]
        self.bestpar = params[bfnum,:]
        for pp in range(np.shape(params)[1]):
            self.tables[pp] = self._calc_coeffs(flux_vectors,params[:,pp], pp)

    def predict(self, params,fSiIII=0.):
        """Get the interpolated quantity by evaluating the quadratic fit"""
        #Interpolate onto desired bins
        #Do parameter correction
        newq = np.ones_like(self.bestfv)
        assert np.shape(params) == (1,np.shape(self.bestpar)[0])
        dpp = params[0] - self.bestpar
        for pp,dp in enumerate(dpp):
            newq += self.tables[pp][:,0]*dp**2 +self.tables[pp][:,1]*dp
        mean = newq * self.bestfv
        std = np.zeros_like(mean)
        return [mean,], std

    def _flux_deriv(self, PFdif, pdif):
        """Calculate the flux-derivative for a single parameter change"""
        assert np.size(pdif) == np.size(PFdif)
        mat=np.vstack([pdif**2, pdif] ).T
        (derivs, _,_, _)=np.linalg.lstsq(mat, PFdif)
        return derivs

    def dump(self, savedir, dumpfile="quad_training.json"):
        """Dump training data to a textfile."""
        super().dump(savedir,dumpfile=dumpfile)

    def load(self,savedir, dumpfile="quad_training.json"):
        """Load parameters from a textfile."""
        super().load(savedir,dumpfile=dumpfile)

    def _get_changes(self, flux_vectors, params, pind):
        """Get the change in parameters, delta p and the corresponding change
        in the flux power spectrum, delta P_F, rebinned to match the desired output bins"""
        dfv = flux_vectors/self.bestfv - 1.
        #Compute change in parameters
        dparams  = params - self.bestpar[pind]
        #Find only those positions where this parameter changed.
        ind = np.where(np.abs(dparams) > 1e-3*self.bestpar[pind])
        assert (pind < 5 and len(ind[0]) == 4) or len(ind[0]) == 9
        assert ind[0]
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

    def get_emulator(self, mean_flux=False, max_z=4.2):
        """ Build an emulator for the desired k_F and our simulations.
            kf gives the desired k bins in s/km.
            Mean flux rescaling is handled (if mean_flux=True) as follows:
            1. A set of flux power spectra are generated for every one of a list of possible mean flux values.
            2. Each flux power spectrum in the set is rescaled to the same mean flux.
            3.
        """
        gp = self._get_custom_emulator(emuobj=QuadraticPoly, mean_flux=mean_flux, max_z=max_z,intol=1e-2)
        return gp

    def _get_fv(self, pp,myspec, mean_flux):
        """Helper function to get a single flux vector, and deal with the mean flux."""
        di = self.get_outdir(pp)
        print(di)
        tau0_factors = None
        if mean_flux:
            ti = self.dense_param_names['tau0']
            tlim = self.dense_param_limits[ti]
            tau0_factors = np.linspace(tlim[0], tlim[1], self.dense_samples)
            midpt = self.dense_samples//2
	    #First simulation needs extra entries with different mean fluxes.
            try:
                self.mf_done
		#Other simulatons just need a mean flux set.
                pvals_new = np.zeros((1, len(pp)+1))
                pvals_new[:,:len(pp)] = pp
                tau0_factors = np.array([tau0_factors[midpt]])
            except AttributeError:
                pvals_new = np.zeros((self.dense_samples, len(pp)+1))
                pvals_new[:,:len(pp)] = np.tile(pp, (self.dense_samples,1))
                #Mid value should be zeroth entry so it is central point of the whole emulator
                tmp = tau0_factors[midpt]
                tau0_factors[1:midpt+1] = tau0_factors[0:midpt]
                tau0_factors[0] = tmp
                assert tau0_factors[0] != tau0_factors[midpt]
                self.mf_done = True
            #Use the mean flux at z=3 as the index parameter.
            #best accuracy should be achieved if the derived parameter is linear in the input.
            pvals_new[:,-1] = np.exp(-1*tau0_factors*flux_power.obs_mean_tau(3.))
        else:
            pvals_new = pp.reshape((1,len(pp)))
        fv = myspec.get_flux_power(di,self.kf, tau0_factors = tau0_factors)
        assert np.shape(fv)[0] == np.shape(pvals_new)[0]
        return pvals_new, fv
