#!/usr/bin/env python
# vim: set fileencoding=UTF-8 :

"""
An emulator to interpolate between flux power spectra computed for the Lyman alpha forest.
Fits the change in the flux power spectrum with a quadratic function.
"""

import math
import numpy as np
from smooth import rebin
from coarse_grid import Emulator

def Hubble(zz, om, H0):
    """ Hubble parameter. Hubble(Redshift) """
    #Conversion factor between s/km and h/Mpc is (1+z)/H(z)
    return 100*H0*math.sqrt(om*(1+zz)**3+(1-om))

class OldQuadraticEmulator(object):
    """
    Given a set of simulations with different parameters, produce the expected quantity interpolated to a new set of parameters.
    Takes as arguments:
        knots - a list of knots, each of which holds directories which vary every parameter in turn.
        bestfit - Best fit simulation. We use the input simulations to compute changes in the interpolated quantity,
                     and then multiply to get the best fit.
                     The best fit used in each knot to compute the different coefficients need not be this simulation!
    """
    def __init__(self, knots, bestfit, sdsskbins=np.array([])):
        #SDSS kbins, in s/km units.
        if np.size(sdsskbins) == 0:
            sdsskbins=np.array([0.00141,0.00178,0.00224,0.00282,0.00355,0.00447,0.00562,0.00708,0.00891,0.01122,0.01413,0.01778])
        self.sdsskbins = sdsskbins
        self.tables = {}
        #Compute coefficients separately for each parameter.
        for knot in knots:
            self.tables[knot.name] = self._calc_coeffs(knot.bfnum, knot.sims,knot.name)
        self.bestfitsim = bestfit

    def get_interpolated(self,newparams):
        """Get the interpolated quantity by evaluating the quadratic fit"""
        #Interpolate onto desired bins
        newq = rebin(self.bestfitsim.get_quantity(), self.bestfitsim.get_bins(), self.sdsskbins)
        #Do parameter correction
        for (pp, nval) in newparams.iteritems():
            dp = nval-self.bestfitsim.get_param(pp)
            newq += self.tables[pp][:,0]*dp**2 +self.tables[pp][:,1]*dp
        return newq

    def _flux_deriv(self, PFdif, pdif):
        """Calculate the flux-derivative for a single parameter change"""
        assert np.size(pdif) == np.size(PFdif)
        mat=np.vstack([pdif**2, pdif] ).T
        (derivs, _,_, _)=np.linalg.lstsq(mat, PFdif)
        return derivs

    def _get_changes(self, bfnum, sims, paramname):
        """Get the change in parameters, delta p and the corresponding change
        in the flux power spectrum, delta P_F, rebinned to match the desired output bins"""
        #Get the quantity we want to interpolate: this will be an array of different kbins.
        #So to_interp is (nsims, nbins)
        assert np.size(sims) > bfnum
        to_interp = [ss.get_quantity() for ss in sims]
        kbins = sims[bfnum].get_bins()
        assert np.shape(to_interp) == (np.size(sims), np.size(kbins))
        #Rebin the flux power for the k values we want
        #Avoid rebinning outside the allowed range.
        #Happens with a box is smaller than 60 Mpc.
        lowest = np.where(self.sdsskbins > kbins[0])[0][0]
        assert lowest < np.size(self.sdsskbins)
        #Set all changes outside the interpolation range to zero.
        bto_interp = np.array([rebin(tt, kbins, self.sdsskbins[lowest:]) for tt in to_interp])
        dto_interp = np.zeros((np.size(sims), np.size(self.sdsskbins)))
        dto_interp[:,lowest:] = bto_interp/bto_interp[bfnum,:] - 1.0
        #Compute change in parameters
        params = np.array([ss.get_param(paramname) for ss in sims])
        #Compute parameter differences
        params -= params[bfnum]
        #Make sure this parameter does change for the passed simulations
        assert np.any(params != 0)
        return (dto_interp, params)

    def _calc_coeffs(self, bfnum, sims, paramname):
        """
            Calculate the flux derivatives for a single redshift, for a single shifting parameter
            Input:
                bfnum - which of the simulations to use as the one to expand around
                sims - list of simulations. Must be of Simulation type above, ie, define a get_quantity, a get_params and a get_bins.
            Output: (kbins d2P...kbins dP (flat vector of length 2xkbins))
        """
        #Get the change in the interpoaltion value with parameter
        (dto_interp, dparams) = self._get_changes(bfnum, sims, paramname)
        #Pass each k value to flux_deriv in turn.
        # Format of returned data from flux_derivs is (a,b) where it fits to:
        # dto_interp = a params**2 + b params
        results =np.array([self._flux_deriv(dto_interp[:,k], dparams) for k in range(np.shape(dto_interp)[1])])
        #So results should have shape
        assert np.shape(results) == (np.size(self.sdsskbins), 2)
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
            for nn in range(1,n1par//2):
                up = centroid+dp*nn*pthis
                sims.append(up)
                down = centroid-dp*nn*pthis
                sims.append(down)
        return np.array(sims)
