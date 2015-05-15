#!/usr/bin/env python
# vim: set fileencoding=UTF-8 :

"""
Various flux derivative stuff
"""

import numpy as np
import math
from smooth import rebin
import os.path
import glob
import re

def corr_table(table, dvecs,table_name):
    """Little function to adjust a table so it has a different central value"""
    new=np.array(table)
    new[12:,:] = table[12:,:]+2*table[0:12,:]*dvecs
    pkd="/home/spb41/cosmomc-src/cosmomc/data/lya-interp/"
    np.savetxt(pkd+table_name,new,("%1.3g","%1.3g","%1.3g","%1.3g","%1.3g","%1.3g","%1.3g","%1.3g","%1.3g","%1.3g","%1.3g"))
    return new

def MacDonaldPF(sdss, zz, Hubblez):
    """Load the SDSS power spectrum"""
    psdss=sdss[np.where(sdss[:,0] == zz)][:,1:3]
    fbar=math.exp(-0.0023*(1+zz)**3.65)
    #multiply by the hubble parameter to be in 1/(km/s)
    scale=Hubblez/(1.0+zz)
    PF=psdss[:,1]*fbar**2/scale
    k=psdss[:,0]*scale
    return (k, PF)

def Hubble(zz, om, H0):
    """ Hubble parameter. Hubble(Redshift) """
    #Conversion factor between s/km and h/Mpc is (1+z)/H(z)
    return 100*H0*math.sqrt(om*(1+zz)**3+(1-om))

class Simulation(object):
    """A class to store a single simulation snapshot and get the quantity from that simulation that needs to be interpolated.
        All the loading functions should be created by overloading this class.
        Should store parameter values."""
    def __init__(self, filename, params):
        self.filename = filename
        self.params = params

    def get_quantity(self):
        """Function to get a quantity that we want to interpolate"""
        raise NotImplementedError

    def get_bins(self):
        """Get the binned kvalues for this quantity"""
        raise NotImplementedError

    def get_param(self,paramname):
        """Get the parameter values for this simulation snapshot, over which we will do the interpolation"""
        return self.params[paramname]

    def get_paramnames(self):
        """Get possible parameter names"""
        return self.params.keys()

class FluxPowSimulation(Simulation):
    """Specialist class for loading a flux power spectrum. This should handle all file loading, rebinning, etc."""
    def __init__(self, filename, params, zz, om, H0=0.7, box=60., sdsskbins=np.array([])):
        self.H0 = H0
        self.box=box
        #SDSS kbins, in s/km units.
        if np.size(sdsskbins) == 0:
            sdsskbins=np.array([0.00141,0.00178,0.00224,0.00282,0.00355,0.00447,0.00562,0.00708,0.00891,0.01122,0.01413,0.01778])
        self.sdsskbins = sdsskbins
        super(FluxPowSimulation, self).__init__(filename, params)
        (self.k, self.PF) = self.loadpk(zz, H0, om)
        self.sdssPF = rebin(self.PF, self.k, self.sdsskbins)

    def loadpk(self, zz, H0, om):
        """Load a flux power spectrum in s/km units, from a text file."""
        #Get table from file in Fourier units
        flux_power=np.loadtxt(self.filename)
        #Convert to Mpc/h units
        scale=H0/self.box
        #k is now in h/Mpc
        k=(flux_power[1:,0]-0.5)*scale*2.0*math.pi
        #PF is now in Mpc/h (comoving)
        PF=flux_power[1:,1]/scale
        #Convert to s/km: H is in km/s/Mpc, so this gets us km/s
        scale2 = Hubble(zz,H0, om)/(1.0+zz)/H0
        PF*=scale2
        k/=scale2
        return (k, PF)

    def get_quantity(self):
        """Return a flux power spectrum, rebinned to be in SDSS units"""
        return self.sdssPF

    def get_bins(self):
        """Get the binned kvalues for this quantity"""
        return self.sdsskbins

class Knot(object):
    """A basic structure class to store a number of simulation runs, with parameter values."""
    def __init__(self, name, base, snapnum, zz, bestfit, bf_params, om=0.2669, box=60., H0=0.71):
        self.name = name
        #Find all directories with this name
        dirs = glob.glob( base+"/"+name+"*/" )
        dirs = [d for d in dirs if re.search(name+r"[0-9\.]*/", d)]
        #Find the parameter value by globbing the directory
        matches = [ re.search(name+r"([0-9\.]*)", dd) for dd in dirs ]
        params = [ float(mm.groups()[0]) for mm in matches ]
        #Build a list of simulations, loading the flux power spectrum each time
        self.sims = []
        filename = os.path.join("flux-power", "snapshot_"+str(snapnum).rjust(3,'0')+"_flux_power.txt")
        for (dd, pp) in zip(dirs, params):
            pdict = dict(bf_params)
            pdict[name] = pp
            simdir = os.path.join(dd, filename)
            self.sims.append(FluxPowSimulation(simdir, pdict, zz=zz, om=om, box=box, H0=H0) )
        bfbase = os.path.join(base, bestfit)
        bf = os.path.join(bfbase, filename)
        self.sims.append(FluxPowSimulation(bf, bf_params, zz=zz, om=om, box=box, H0=H0))
        self.bfnum=-1

class QuadraticEmulator(object):
    """
    Given a set of simulations with different parameters, produce the expected quantity interpolated to a new set of parameters.
    Takes as arguments:
        knots - a list of knots, each of which holds directories which vary every parameter in turn.
        bestfit - Best fit simulation. We use the input simulations to compute changes in the interpolated quantity,
                     and then multiply to get the best fit.
                     The best fit used in each knot to compute the different coefficients need not be this simulation!
    """
    def __init__(self, knots, bestfit):
        self.tables = {}
        #Compute coefficients separately for each parameter.
        for knot in knots:
            self.tables[knot.name] = self.calc_coeffs(knot.bfnum, knot.sims,knot.name)
        self.bestfitsim = bestfit

    def get_quantity(self,newparams):
        """Get the interpolated quantity by evaluating the quadratic fit"""
        newq = self.bestfitsim.get_quantity()
        for (pp, nval) in newparams.iteritems():
            dp = nval-self.bestfitsim.get_param(pp)
            newq += self.tables[pp][:,0]*dp**2 +self.tables[pp][:,1]*dp
        return newq

    def flux_deriv(self, PFdif, pdif):
        """Calculate the flux-derivative for a single parameter change"""
        assert np.size(pdif) == np.size(PFdif)
        mat=np.vstack([pdif**2, pdif] ).T
        (derivs, _,_, _)=np.linalg.lstsq(mat, PFdif)
        return derivs

    def calc_coeffs(self, bfnum, sims, paramname):
        """
            Calculate the flux derivatives for a single redshift, for a single shifting parameter
            Input:
                bfnum - which of the simulations to use as the one to expand around
                sims - list of simulations. Must be of Simulation type above, ie, define a get_quantity, a get_params and a get_bins.
            Output: (kbins d2P...kbins dP (flat vector of length 2xkbins))
        """
        #Get the quantity we want to interpolate: this will be an array of different scales.
        #So to_interp is (nsims, nbins)
        assert np.size(sims) > bfnum
        to_interp = np.array([ss.get_quantity() for ss in sims])
        kbins = sims[bfnum].get_bins()
        assert np.shape(to_interp) == (np.size(sims), np.size(kbins))
        params = np.array([ss.get_param(paramname) for ss in sims])
        #Compute parameter differences
        params -= params[bfnum]
        #Make sure this parameter does change for the passed simulations
        assert np.any(params != 0)
        #Compute differences in the interpolated quantity
        dto_interp = to_interp/to_interp[bfnum,:] - 1.0
        #So now we have an array of data values.
        #Pass each k value to flux_deriv in turn.
        # Format of returned data from flux_derivs is (a,b) where it fits to:
        # dto_interp = a params**2 + b params
        results =np.array([self.flux_deriv(dto_interp[:,k], params) for k in xrange(np.size(kbins))])
        #So results should have shape
        assert np.shape(results) == (np.size(kbins), 2)
        return results

if __name__=='__main__':
    #Parameters of the 2010 simulations
    #Note how every knot has a different best-fit simulation!
    #As I recall there was a bug with the B knots that made small scales wrong somehow, so I had to use bf2 for the C knot.
    #The other two are just to allow for different box sizes
    basedir = "/home/spb/codes/Lyman-alpha/MinParametricRecon/runs/"
    zzz = { n : 4.2-0.2*n for n in xrange(12) }
    omega0 = 0.2669
    hubble0 = 0.71
    #I checked that these are the same parameters for all 4 knots
    bestfit_params = {'AA':0.94434469,'B': 0.93149282,'C': 0.91868144,'D': 0.9060194}
    quads = {}
    for snap in zzz.keys():
        AA_knot = Knot("AA",basedir, bestfit="boxcorr400",snapnum=snap,zz=zzz[snap], om=omega0, H0=hubble0, bf_params=bestfit_params, box=120.)
        B_knot = Knot("B",basedir, bestfit="best-fit",snapnum=snap,zz=zzz[snap], om=omega0, H0=hubble0, bf_params=bestfit_params, box=60.)
        C_knot = Knot("C",basedir, bestfit="bf2",snapnum=snap,zz=zzz[snap], om=omega0, H0=hubble0, bf_params=bestfit_params, box=60.)
        D_knot = Knot("D",basedir, bestfit="bfD",snapnum=snap,zz=zzz[snap], om=omega0, H0=hubble0, bf_params=bestfit_params, box=48.)
        #Best-fit model to use when not computing differences.
        #There are also hr2a, hr3, hr4 and hr4a directories with flux power spectra in them.
        #I have no memory of what these files are - there are no simulation snapshots.
        hr2 = os.path.join("hr2/flux-power", "snapshot_"+str(snap).rjust(3,'0')+"_flux_power.txt")
        emubf = FluxPowSimulation(os.path.join(basedir, hr2), bestfit_params, zz=zzz[snap], om=omega0, box=60, H0=hubble0)
        quads[snap] = QuadraticEmulator((AA_knot, B_knot, C_knot, D_knot),emubf)
