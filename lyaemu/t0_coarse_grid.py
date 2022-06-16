"""Generate a coarse grid for constructing a median temperature emulator"""
from __future__ import print_function
import os
import numpy as np
import h5py
from . import flux_power
from . import gpemulator
from . import tempdens
from .coarse_grid import Emulator

def get_latex(key):
    """Get a latex name if it exists, otherwise return the key."""
    #Names for pretty-printing some parameters in Latex
    print_names = { 'ns': r'n_\mathrm{s}', 'Ap': r'A_\mathrm{P}', 'herei': r'z_\mathrm{He i}', 'heref': r'z_\mathrm{He f}', 'hub':'h', 'tau0':r'\tau_0', 'dtau0':r'd\tau_0'}
    try:
        return print_names[key]
    except KeyError:
        return key

class T0Emulator(Emulator):
    """Subclass of coarse_grid.Emulator. Stores parameter names and limits, generates T0 file from particle snapshots, and gets an emulator.
    Parameters:
    - basedir: directory to load or create emulator
    - param_names: dictionary containing names of the parameters as well as a unique integer list of positions
    - param_limits: Nx2 array containing upper and lower limits of each parameter, in the order given by the integer stored in param_names
    - tau_thresh: threshold optical depth for spectra. Kept here only for consistency with json parameter file.
    - npart, box: particle number and box size of emulator simulations.
    """
    def __init__(self, basedir, param_names=None, param_limits=None, tau_thresh=None, npart=512, box=60, fullphysics=True):
        super().__init__(basedir=basedir, param_names=param_names, param_limits=param_limits, kf=None, mf=None, npart=npart, box=box, tau_thresh=tau_thresh, fullphysics=fullphysics)

    def get_emulator(self, max_z=4.2, min_z=2.0):
        """ Build an emulator for T0 from simulations."""
        aparams, meanT = self.get_meanT(max_z=max_z, min_z=min_z)
        plimits = self.get_param_limits(include_dense=False)
        gp = gpemulator.MultiBinGP(params=aparams, kf=kf, powers=flux_vectors, param_limits=plimits, singleGP=None)
        return gp

    # def do_loo_cross_validation(self, *, remove=None, max_z=4.2, subsample=None):
    #     """Do cross-validation by constructing an emulator missing
    #        a single simulation and checking accuracy.
    #        The remove parameter chooses which simulation to leave out. If None this is random."""
    #     aparams, kf, flux_vectors = self.get_flux_vectors(max_z=max_z, kfunits="mpc")
    #     rng = np.random.default_rng()
    #     if remove is None:
    #         nsims = np.shape(aparams)[0]
    #         rng = np.random.default_rng()
    #         remove = rng.integers(0,nsims)
    #     aparams_rem = np.delete(aparams, remove, axis=0)
    #     flux_vectors_rem = np.delete(flux_vectors, remove, axis=0)
    #     if subsample is not None:
    #         nsims = np.shape(aparams_rem)[0]
    #         reorder = rng.permutation(nsims)
    #         aparams_rem = aparams_rem[reorder[:subsample]]
    #         flux_vectors_rem = flux_vectors_rem[reorder[:subsample]]
    #     plimits = self.get_param_limits(include_dense=True)
    #     gp = gpemulator.MultiBinGP(params=aparams_rem, kf=kf, powers = flux_vectors_rem, param_limits = plimits)
    #     flux_predict, std_predict = gp.predict(aparams[remove, :].reshape(1, -1))
    #     err = (flux_vectors[remove,:] - flux_predict[0])/std_predict[0]
    #     return kf, flux_vectors[remove,:] / flux_predict[0] - 1, err

    def get_meanT(self, filename="emulator_meanT.hdf5"):
        """Get and save the T0 and parameters"""
        aparams = self.get_parameters()
        assert np.shape(aparams)[1] == len(self.param_names)
        try:
            meanT = self.load_meanT(aparams, savefile=filename)
        except (AssertionError, OSError):
            print("Could not load T0, regenerating from disc")
            new_inds, meanT = self.check_meanT(aparams, savefile=filename)
            for ind in new_inds:
                di = self.get_outdir(aparams[ind], strsz=3)
                snaps = self.get_snaps(di)
                new_meanT = np.array([tempdens.get_median_temp(snap, di) for snap in snaps])
                meanT[ind] = new_meanT
                self.save_meanT(aparams, meanT, savefile=filename)
        return aparams, meanT

    def check_meanT(self, aparams, savefile="emulator_meanT.hdf5"):
        """Cross-reference existing file samples with requested."""
        nsims, nz = np.shape(aparams)[0], self.myspec.zout.size
        meanT = np.zeros([nsims, nz])
        # check if file exists -- if not, return 'empty' meanT and all indices
        if not os.path.exists(os.path.join(self.basedir, savefile)):
            return np.arange(nsims), meanT
        load = h5py.File(os.path.join(self.basedir, savefile), 'r')
        inparams, old_meanT = np.array(load["params"]), np.array(load["meanT"])
        load.close()
        assert np.isin(inparams, aparams).all(axis=1).min() == 1, "Non-matching file '%s' exists on path. Move or delete to generate T0 file." % savefile
        # continue running if no new parameters -- return indices yet to be filled
        if np.all(inparams == aparams):
            return np.where(np.all(old_meanT == 0, axis=1))[0], old_meanT
        # otherwise, find new parameters and return indices for them
        subset = np.isin(aparams, inparams).all(axis=1)
        new_inds = np.where(subset == False)[0] # indices of aparams that are not in inparams
        meanT[subset] = old_meanT # fill meanT with already computed values
        return new_inds, meanT


    def save_meanT(self, aparams, meanT, savefile="emulator_meanT.hdf5"):
        """Save the mean temperatures and parameters to a file."""
        save = h5py.File(os.path.join(self.basedir, savefile), 'w')
        save.attrs["classname"] = str(self.__class__)
        save["zout"] = self.myspec.zout
        save["params"] = aparams
        save["meanT"] = meanT
        save.close()

    def load_meanT(self, aparams, savefile="emulator_meanT.hdf5"):
        """Load the mean temperatures from a file."""
        load = h5py.File(os.path.join(self.basedir, savefile), 'r')
        inparams = np.array(load["params"])
        meanT = np.array(load["meanT"])
        zout = np.array(load["zout"])
        self.myspec.zout = zout
        name = str(load.attrs["classname"])
        load.close()
        assert name.split(".")[-1] == str(self.__class__).split(".")[-1]
        assert np.shape(inparams) == np.shape(aparams)
        assert np.all(inparams - aparams < 1e-3)
        assert np.all(meanT != 0)
        return meanT

    def get_snaps(self, base):
        """Get list of snapshot numbers from simulation at base, that matches zout"""
        snapshots = []
        for snap in range(30):
            snapdir = os.path.join(base, 'PART_'+str(snap).rjust(3,'0'))
            if os.path.exists(snapdir):
                red = 1./flux_power._get_header_attr_from_snap("Time", snap, base) - 1
                if np.min(np.abs(red - self.myspec.zout)) < 0.01:
                    snapshots.append(snap)
        return snapshots
