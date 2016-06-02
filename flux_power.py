"""Modules to generate the flux power spectrum from a simulation box."""
from __future__ import print_function
import numpy as np
import spectra

class MySpectra(object):
    """This class stores the randomly positioned sightlines once,
       so that they are the same for each emulator point."""
    def __init__(self, numlos = 32000, box=60.):
        self.NumLos = numlos
        self.axis = np.ones(self.NumLos)
        #Sightlines at random positions
        #Re-seed for repeatability
        np.random.seed(23)
        self.cofm = box*np.random.random_sample((self.NumLos,3))
        #Use the right values for SDSS or BOSS.
        self.spec_res = 200.

    def _get_spectra_snap(self, snap, base):
        """Get a snapshot with generated HI spectra"""
        try:
            #First try to reload a savefile
            ss = spectra.Spectra(snap, base, self.cofm, self.axis, res=self.spec_res/4, savefile="lya_forest_spectra.hdf5",spec_res = self.spec_res, reload_file=False)
        except IOError:
            #If we couldn't, regenerate the spectra
            ss = spectra.Spectra(snap, base, self.cofm, self.axis, res=self.spec_res/4, savefile="lya_forest_spectra.hdf5",spec_res = self.spec_res,reload_file=True)
            ss.get_tau("H",1,1215)
            ss.save_file()
        return ss

    def get_flux_power(self, base):
        """Get the flux power spectrum in the format used by McDonald 2004
        for a snapshot set."""
        #Want output every 0.2 from z=4.2 to z=2.0
        zout = np.arange(4.2,1.9,-0.2)
        fluxlist = []
        for snap in range(100):
            try:
                ss = self._get_spectra_snap(snap, base)
                #Now if the redshift is something we want, generate the flux power
                if np.min(np.abs(ss.red - zout)) < 0.05:
                    fluxlist.append(ss.get_flux_power_1D("H",1,1215))
            except IOError:
                #We ran out of snapshots
                break
            #Make sure we have enough outputs
            assert len(fluxlist) == np.size(zout)
            flux_power = np.array(fluxlist)
        return flux_power

