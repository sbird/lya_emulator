"""Modules to generate the flux power spectrum from a simulation box."""
from __future__ import print_function
import argparse
import os.path
import scipy.interpolate
import numpy as np
from mpi4py import MPI
from fake_spectra import spectra
from fake_spectra import griddedspectra
from fake_spectra import abstractsnapshot as absn

def rebin_power_to_kms(kfkms, kfmpc, flux_powers, zbins, omega_m, omega_l=None):
    """Rebins a power spectrum to constant km/s bins.
    Bins larger than the box are discarded. The return type is thus a list,
    with each redshift bin having potentially different lengths."""
    if omega_l is None:
        omega_l = 1 - omega_m
    nz = np.size(zbins)
    assert np.size(flux_powers) == nz * np.size(kfmpc)
    # conversion factor from mpc to kms units
    velfac = 1./(1+zbins) * 100.0*np.sqrt(omega_m*(1 + zbins)**3 + omega_l)
    # original simulation output
    okmsbins = np.outer(kfmpc, 1./velfac).T
    flux_powers = flux_powers.reshape(okmsbins.shape)
    # interpolate simulation output for averaging
    rebinned = [scipy.interpolate.interpolate.interp1d(okmsbins[i], flux_powers[i]) for i in range(nz)]
    new_flux_powers = np.array([rebinned[i](kfkms) for i in range(nz)])
    # final flux power array
    assert np.min(new_flux_powers) > 0
    return np.repeat([kfkms], nz, axis=0), new_flux_powers

class FluxPower:
    """Class stores the flux power spectrum."""
    def __init__(self, maxk):
        self.spectrae = []
        self.snaps = []
        self.maxk = maxk
        self.kf = None

    def add_snapshot(self,snapshot, spec):
        """Add a power spectrum to the list."""
        self.snaps.append(snapshot)
        self.spectrae.append(spec)

    def len(self):
        """Get the number of snapshots in the list"""
        return len(self.spectrae)

    def get_power(self, kf, mean_fluxes, tau_thresh=None):
        """Generate a flux power spectrum rebinned to be like the flux power from BOSS.
        This can be used as an artificial data vector."""
        mf = None
        flux_arr = np.empty(shape=(self.len(),np.size(kf)))
        for (i,ss) in enumerate(self.spectrae):
            if mean_fluxes is not None:
                mf = mean_fluxes[i]
            kf_sim, flux_power_sim = ss.get_flux_power_1D("H",1,1215, mean_flux_desired=mf, tau_thresh=tau_thresh)
            #Rebin flux power to have desired k bins
            rebinned=scipy.interpolate.interpolate.interp1d(kf_sim,flux_power_sim)
            ii = np.where(kf > kf_sim[0])
            ff = flux_power_sim[0]*np.ones_like(kf)
            ff[ii] = rebinned(kf[ii])
            flux_arr[i] = ff
        flux_arr = np.ravel(flux_arr)
        assert np.shape(flux_arr) == (self.len()*np.size(kf),)
        self.drop_table()
        return flux_arr

    def get_power_native_binning(self, mean_fluxes, tau_thresh=None):
        """ Generate the flux power, with known optical depth, from a list of snapshots.
            maxk should be in comoving Mpc/h.
            kf is stored in comoving Mpc/h units.
            The P_F returned is in km/s units.
        """
        mf = None
        flux_arr = np.array([])
        for (i,ss) in enumerate(self.spectrae):
            if mean_fluxes is not None:
                mf = mean_fluxes[i]
            kf_sim, flux_power_sim = ss.get_flux_power_1D("H",1,1215, mean_flux_desired=mf, tau_thresh=tau_thresh)
            #Store k_F in comoving Mpc/h units, so that it is independent of redshift.
            vscale = ss.velfac * 3.085678e24/ss.units.UnitLength_in_cm
            kf_sim *= vscale
            ii = np.where(kf_sim <= self.maxk)
            flux_arr = np.append(flux_arr,flux_power_sim[ii])
            if self.kf is None:
                self.kf = kf_sim[ii]
            else:
                assert np.all(np.abs(kf_sim[ii]/self.kf - 1) < 1e-5)
        flux_arr = np.ravel(flux_arr)
        assert np.shape(flux_arr) == (self.len()*np.size(self.kf),)
        self.drop_table()
        return flux_arr

    def get_kf_kms(self):
        """Get a vector of kf in km/s units for all redshifts."""
        kfkms = np.array([ self.kf / (ss.velfac * 3.085678e24/ss.units.UnitLength_in_cm) for ss in self.spectrae])
        return kfkms

    def get_zout(self):
        """Get output redshifts"""
        return np.array([ss.red for ss in self.spectrae])

    def drop_table(self):
        """Reset the H1 tau array in all spectra, so it needs to be loaded from disc again."""
        for ss in self.spectrae:
            ss.tau[('H',1,1215)] = np.array([0])

class MySpectra:
    """This class stores the randomly positioned sightlines once,
       so that they are the same for each emulator point.
       max_k is in comoving h/Mpc."""
    def __init__(self, ngrid=480, max_z=5.4, min_z=2.0, max_k=5.):
        #For SDSS or BOSS the spectral resolution is
        #60 km/s at 5000 A and 80 km/s at 4300 A.
        #In principle we could generate smoothed spectra
        #and then correct the window function.
        #However, this non-linear smoothing will change the mean flux
        #and hence does not commute with mean flux rescaling.
        #I have checked that for k < 0.1 the corrected version
        #is identical to the unsmoothed version (without mean flux rescaling).
        self.spec_res = 0.
        #For BOSS the pixel resolution is actually 69 km/s.
        #We use a much smaller pixel resolution so that the window functions
        #are small for mean flux rescaling, and also so that HCDs are confined.
        self.pix_res = 10.
        #Want output every 0.2 from z=max to z=2.0, matching SDSS.
        nz = int(np.round((max_z-min_z)/0.2, 1)) + 1
        self.zout = np.linspace(max_z, min_z, nz)
        self.max_k = max_k
        self.ngrid = ngrid
        self.savefile = "lya_forest_spectra_grid_%d.hdf5" % ngrid

    def _check_redshift(self, red):
        """Check the redshift of a snapshot set is what we want."""
        if np.min(np.abs(red - self.zout)) > 0.01:
            return 0
        return 1

    def _get_spectra_snap(self, snap, base):
        """Get a snapshot with generated HI spectra"""
        #First try to get data from the savefile, and if we can't, try the snapshot.
        try:
            ss = spectra.Spectra(snap, base, None, None, res=self.pix_res, savefile=self.savefile,spec_res = self.spec_res, reload_file=False,sf_neutral=False,quiet=True, load_snapshot=False, MPI=MPI)
            if not self._check_redshift(ss.red):
                return None
        except OSError:
            #Check the redshift is ok
            red = 1./_get_header_attr_from_snap("Time", snap, base)-1.
            if not self._check_redshift(red):
                return None
            #Make sure we have sightlines
            ss = griddedspectra.GriddedSpectra(snap, base, nspec = self.ngrid, res=self.pix_res, axis=-1, savefile=self.savefile, spec_res = self.spec_res, reload_file=True,sf_neutral=False,quiet=False, MPI=MPI)
            #Get optical depths and save
            _ = ss.get_tau("H",1,1215)
            ss.save_file()
        return ss

    def get_snapshot_list(self, base, snappref="SPECTRA_"):
        """Get the flux power spectrum in the format used by McDonald 2004
        for a snapshot set."""
        #print('Looking for spectra in', base)
        powerspectra = FluxPower(maxk=self.max_k)
        for snap in range(30):
            snapdir = os.path.join(base,snappref+str(snap).rjust(3,'0'))
            #We ran out of snapshots
            if not os.path.exists(snapdir):
                snapdir = os.path.join(base,"PART_"+str(snap).rjust(3,'0'))
                if not os.path.exists(snapdir):
                    snapdir = os.path.join(base, "snap_"+str(snap).rjust(3,'0'))
                    if not os.path.exists(snapdir):
                        continue
            #We have all we need
            if powerspectra.len() == np.size(self.zout):
                break
            try:
                ss = self._get_spectra_snap(snap, base)
#                 print('Found spectra in', ss)
                if ss is not None:
                    powerspectra.add_snapshot(snap,ss)
            except IOError:
                print("Didn't find spectra in %s snap %d because of IOError: " % (base, snap), flush=True)
                continue
        #Make sure we have enough outputs, ignoring missing values at the start and the end
        if powerspectra.len() != np.size(self.zout):
            reds = [ss.red for ss in powerspectra.spectrae]
            nz = int(np.round((np.max(reds)-np.min(reds))/0.2, 1)) + 1
            self.zout = np.linspace(np.max(reds), np.min(reds), nz)
            print("Redshift now from z=%g to %g" % (self.zout[0], self.zout[-1]), flush=True)
        if powerspectra.len() != np.size(self.zout):
            raise ValueError("Found only",powerspectra.len(),"of",np.size(self.zout),"z=",self.zout[0], self.zout[-1],"from snaps:",powerspectra.snaps)
        return powerspectra

def _get_header_attr_from_snap(attr, num, base):
    """Get a header attribute from a snapshot, if it exists."""
    f = absn.AbstractSnapshotFactory(num, base)
    value = f.get_header_attr(attr)
    del f
    return value

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('base', type=str, help='Snapshot directory')
    parser.add_argument('--ngrid', type=int, help='Number of sightlines per side', default=480, required=False)
    args = parser.parse_args()
    myspec = MySpectra(ngrid=args.ngrid)
    myspec.get_snapshot_list(args.base)
