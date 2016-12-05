"""Modules to generate the flux power spectrum from a simulation box."""
from __future__ import print_function
import os.path
import numpy as np
import scipy.interpolate
import spectra
import abstractsnapshot as absn

def obs_mean_tau(redshift):
    """The mean flux from 0711.1862: is (0.0023±0.0007) (1+z)^(3.65±0.21)
    Todo: check for updated values."""
    return 0.0023*(1.0+redshift)**3.65

def sim_mean_tau(redshift):
    """For the simulations, we use a different mean flux. Because they are highly under-resolved,
    they do not reproduce the mean flux from observations (by a factor of four or so). So we use
    a different value for now to avoid rescaling by a huge factor."""
    return 0.01 * (1.0 + redshift)**3.3

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
            return 0
        else:
            return 1

    def _get_spectra_snap(self, snap, base):
        """Get a snapshot with generated HI spectra"""
        #If savefile exists, reload. Otherwise do not.
        def mkspec(snap, base, cofm, axis, rf):
            """Helper function"""
            return spectra.Spectra(snap, base, cofm, axis, res=self.spec_res/4., savefile=self.savefile,spec_res = self.spec_res, reload_file=rf,sf_neutral=False,quiet=True)
        #First try to get data from the savefile, and if we can't, try the snapshot.
        try:
            ss = mkspec(snap, base, None, None, rf=False)
        except OSError:
            #Check the redshift is ok
            red = _get_header_attr_from_snap("Redshift", snap, base)
            if not self._check_redshift(red):
                return None
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
        return ss

    def get_flux_power(self, base, kf, tau0_factors=None):
        """Get the flux power spectrum in the format used by McDonald 2004
        for a snapshot set."""
        fluxlists = [list([]) for _ in tau0_factors]
        for snap in range(1000):
            snapdir = os.path.join(base,"snapdir_"+str(snap).rjust(3,'0'))
            #We ran out of snapshots
            if not os.path.exists(snapdir):
                break
            #We have all we need
            if len(fluxlists[0]) == np.size(self.zout):
                break
            try:
                ss = self._get_spectra_snap(snap, base)
                if self._check_redshift(ss.red):
                    fluxlists = self._gen_flux_pow_from_snap(kf, ss, fluxlists, tau0_factors)
            except IOError:
                #Happens when we haven't transferred the starting snapshots
                if len(fluxlists[0]) == 0:
                    continue
                raise IOError("Could not load snapshot: "+snapdir)
        #Make sure we have enough outputs
        for ff in fluxlists:
            assert len(ff) == np.size(self.zout)
        flux_arr = np.array([np.ravel(np.array(ff)) for ff in fluxlists])
        return flux_arr

    def _gen_flux_pow_from_snap(self,kf, ss, fluxlists, tau0_factors=None):
        """Generate the flux power for a list of optical depths from a snapshot.
        flux_powers is a list of lists of arrays, shape [tau0][redshift]
        If tau0_factors is None, fluxlists has one entry, fluxlists[0]."""
        mf = None
        for ii in range(np.size(tau0_factors)):
            if tau0_factors is not None:
                mf = np.exp(-sim_mean_tau(ss.red)*tau0_factors[ii])
            kf_sim, flux_power_sim = ss.get_flux_power_1D("H",1,1215, mean_flux_desired=mf)
            #Rebin flux power to have desired k bins
            rebinned=scipy.interpolate.interpolate.interp1d(kf_sim,flux_power_sim)
            fluxlists[ii].append(rebinned(kf))
        return fluxlists

def _get_header_attr_from_snap(attr, num, base):
    """Get a header attribute from a snapshot, if it exists."""
    f = absn.AbstractSnapshotFactory(num, base)
    value = f.get_header_attr(attr)
    del f
    return value
