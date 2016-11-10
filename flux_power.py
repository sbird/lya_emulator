"""Modules to generate the flux power spectrum from a simulation box."""
from __future__ import print_function
import os.path
import numpy as np
import scipy.interpolate
import spectra
import rescaledspectra

class MySpectra(object):
    """This class stores the randomly positioned sightlines once,
       so that they are the same for each emulator point."""
    def __init__(self, numlos = 32000):
        self.NumLos = numlos
        #Use the right values for SDSS or BOSS.
        self.spec_res = 200.
        self.axis = np.ones(self.NumLos)
        self.cofm = np.array([])
        #Re-seed for repeatability
        np.random.seed(23)
        #Want output every 0.2 from z=4.2 to z=2.0
        self.zout = np.arange(4.2,1.9,-0.2)

    def _get_spectra_snap(self, snap, base, box=60.,mean_flux_desired=None):
        """Get a snapshot with generated HI spectra"""
        #If savefile exists, reload. Otherwise do not.
        reload_file = False
        savefile = "lya_forest_spectra.hdf5"
        if not os.path.exists(os.path.join(os.path.join(base,"snapdir_"+str(snap).rjust(3,'0')),savefile)):
            reload_file = True
            #Get some spectra positions if we don't already have them
            if np.size(self.cofm) != 3*self.NumLos:
                self.cofm = box*np.random.random_sample((self.NumLos,3))
        #Get the snapshot
        ss = spectra.Spectra(snap, base, self.cofm, self.axis, res=self.spec_res/4, savefile=savefile,spec_res = self.spec_res, reload_file=reload_file)
        #Make sure we will use the same spectra positions for all future snapshots.
        self.cofm = ss.cofm
        self.axis = ss.axis
        #Now if the redshift is something we want, generate the flux power
        if np.min(np.abs(ss.red - self.zout)) < 0.05:
            kf, flux_power = ss.get_flux_power_1D("H",1,1215, mean_flux_desired=mean_flux_desired)
            if reload_file:
                ss.save_file()
            return kf,flux_power
        return np.array([]),np.array([])

    def get_flux_power(self, base, kf, mean_flux=None, flat=False):
        """Get the flux power spectrum in the format used by McDonald 2004
        for a snapshot set."""
        fluxlist = []
        for snap in range(1000):
            snapdir = os.path.join(base,"snapdir_"+str(snap).rjust(3,'0'))
            if not os.path.exists(snapdir):
                #We ran out of snapshots
                break
            try:
                kf_sim,flux_power_sim = self._get_spectra_snap(snap, base,mean_flux_desired=mean_flux)
            except IOError:
                raise IOError("Could not load snapshot: "+snapdir)
            #Now if the redshift is something we want, generate the flux power
            if np.size(flux_power_sim) > 2:
                #Rebin flux power to have desired k bins
                rebinned=scipy.interpolate.interpolate.interp1d(kf_sim,flux_power_sim)
                fluxlist.append(rebinned(kf))
        #Make sure we have enough outputs
        assert len(fluxlist) == np.size(self.zout)
        flux_power = np.array(fluxlist)
        if flat:
            flux_power = np.ravel(flux_power)
        return flux_power
