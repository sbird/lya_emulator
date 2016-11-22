"""Modules to generate the flux power spectrum from a simulation box."""
from __future__ import print_function
import os.path
import numpy as np
import scipy.interpolate
import spectra
import abstractsnapshot as absn
import rescaledspectra

class MySpectra(object):
    """This class stores the randomly positioned sightlines once,
       so that they are the same for each emulator point."""
    def __init__(self, numlos = 32000, max_z= 4.2):
        self.NumLos = numlos
        #Use the right values for SDSS or BOSS.
        self.spec_res = 200.
        self.NumLos = numlos
        #Want output every 0.2 from z=max to z=2.0
        self.zout = np.arange(max_z,1.9,-0.2)
        self.savefile = "lya_forest_spectra.hdf5"

    def _get_cofm(self, num, base):
        """Get an array of sightlines."""
        try:
            #Use saved sightlines if we have them.
            return (self.cofm, self.axis)
        except AttributeError:
            #Otherwise get sightlines at random positions
            #Re-seed for repeatability
            np.random.seed(23)
            box = _get_header_attr_from_snap("BoxSize", num, base)
            #All through y axis
            axis = np.ones(self.NumLos)
            cofm = box*np.random.random_sample((self.NumLos,3))
            return cofm, axis

    def _check_redshift(self, red):
        """Check the redshift of a snapshot set is what we want."""
        if np.min(np.abs(red - self.zout)) > 0.01:
            raise ValueError("Unwanted redshift")

    def _get_spectra_snap(self, snap, base,mean_flux_desired=None):
        """Get a snapshot with generated HI spectra"""
        #If savefile exists, reload. Otherwise do not.
        def mkspec(snap, base, cofm, axis, rf):
            """Helper function"""
            return spectra.Spectra(snap, base, cofm, axis, res=self.spec_res/4., savefile=self.savefile,spec_res = self.spec_res, reload_file=rf)
        #First try to get data from the savefile, and if we can't, try the snapshot.
        try:
            ss = mkspec(snap, base, None, None, rf=False)
            self._check_redshift(ss.red)
        except OSError:
            #Check the redshift is ok
            red = _get_header_attr_from_snap("Redshift", snap, base)
            self._check_redshift(red)
            #Make sure we have sightlines
            (cofm, axis) = self._get_cofm(snap, base)
            ss = mkspec(snap, base, cofm, axis, rf=True)
            #Get optical depths and save
            _ = ss.get_tau("H",1,1215)
            ss.save_file()
        #Check we have the same spectra
        try:
            assert np.all(ss.cofm == self.cofm)
        except AttributeError:
            #If this is the first load, we just want to use the snapshot values.
            (self.cofm, self.axis) = (ss.cofm, ss.axis)
        #Now generate the flux power
        kf, flux_power = ss.get_flux_power_1D("H",1,1215, mean_flux_desired=mean_flux_desired)
        return kf,flux_power

    def get_flux_power(self, base, kf, mean_flux=None, flat=False):
        """Get the flux power spectrum in the format used by McDonald 2004
        for a snapshot set."""
        fluxlist = []
        for snap in range(1000):
            snapdir = os.path.join(base,"snapdir_"+str(snap).rjust(3,'0'))
            #We ran out of snapshots
            if not os.path.exists(snapdir) or len(fluxlist) == np.size(self.zout):
                break
            try:
                kf_sim,flux_power_sim = self._get_spectra_snap(snap, base,mean_flux_desired=mean_flux)
            except IOError:
                raise IOError("Could not load snapshot: "+snapdir)
            except ValueError:
                #Signifies the redshift wasn't interesting.
                pass
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

def _get_header_attr_from_snap(attr, num, base):
    """Get a header attribute from a snapshot, if it exists."""
    with absn.AbstractSnapshotFactory(num, base) as f:
        value = f.get_header_attr(attr)
        return value
