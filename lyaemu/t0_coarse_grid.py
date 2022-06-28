"""Generate a coarse grid for constructing a median temperature emulator"""
# from __future__ import print_function
import os
import numpy as np
import h5py
from . import flux_power
from . import t0_gpemulator
from . import tempdens


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

class T0Emulator:
    """Stores parameter names and limits, generates T0 file from particle snapshots, and gets an emulator.

    Parameters:
    - basedir: directory to load or create emulator
    - param_names: dictionary containing names of the parameters as well as a unique integer list of positions
    - param_limits: Nx2 array containing upper and lower limits of each parameter, in the order given by the integer stored in param_names
    - tau_thresh: threshold optical depth for spectra. Kept here only for consistency with json parameter file.
    - npart, box: particle number and box size of emulator simulations.
    """
    def __init__(self, basedir, param_names=None, param_limits=None, tau_thresh=None, npart=512, box=60, fullphysics=True):
        if param_names is None:
            self.param_names = {'ns':0, 'Ap':1, 'herei':2, 'heref':3, 'alphaq':4, 'hub':5, 'omegamh2':6, 'hireionz':7, 'bhfeedback':8}
        else:
            self.param_names = param_names
        #Parameters:
        if param_limits is None:
            self.param_limits = np.array([[0.8, 0.995], # ns: 0.8 - 0.995. Notice that this is not ns at the CMB scale!
                                          [1.2e-09, 2.6e-09], #Ap: amplitude of power spectrum at 8/2pi Mpc scales (see 1812.04654)!
                                          [3.5, 4.1], #herei: redshift at which helium reionization starts.
                                                      # 4.0 is default, we use a linear history with 3.5-4.5
                                          [2.6, 3.2], # heref: redshift at which helium reionization finishes. 2.8 is default.
                                                      # Thermal history suggests late, HeII Lyman alpha suggests earlier.
                                          [1.6, 2.5], # alphaq: quasar spectral index. 1 - 2.5 Controls IGM temperature.
                                          [0.65, 0.75], # hub: hubble constant (also changes omega_M)
                                          [0.14, 0.146],# omegam h^2: We fix omega_m h^2 = 0.143+-0.001 (Planck 2018 best-fit) and vary omega_m and h^2 to match it.
                                                        # h^2 itself has little effect on the forest.
                                          [6.5,8],   #Mid-point of HI reionization
                                          [0.03, 0.07],  # BH feedback parameter
                                       #   [3.2, 4.2] # Wind speed
                                ])
        else:
            self.param_limits = param_limits

        # Remove the BH parameter for not full physics.
        self.fullphysics = fullphysics
        if not fullphysics:
            bhind = self.param_names.pop('bhfeedback')
            self.param_limits = np.delete(self.param_limits, bhind, 0)

        self.npart = npart
        self.box = box
        self.omegabh2 = 0.0224
        self.max_z = 5.4
        self.min_z = 2.0
        self.sample_params = []
        self.basedir = os.path.expanduser(basedir)
        self.tau_thresh = tau_thresh


    def load(self, dumpfile="emulator_params.json"):
        """Load parameters from a textfile."""
        tau_thresh = self.tau_thresh
        real_basedir = self.basedir
        with open(os.path.join(real_basedir, dumpfile), 'r') as jsin:
            indict = json.load(jsin)
        self.__dict__ = indict
        self._fromarray()
        self.tau_thresh = tau_thresh
        self.basedir = real_basedir
        self.myspec = flux_power.MySpectra(max_z=self.max_z, min_z=self.min_z, max_k=self.maxk)

    def _fromarray(self):
        """Convert the data stored as lists back to arrays."""
        for arr in self.really_arrays:
            self.__dict__[arr] = np.array(self.__dict__[arr])
        self.really_arrays = []

    def get_emulator(self, max_z=4.2, min_z=2.0):
        """ Build an emulator for T0 from simulations."""
        # ADD MAX_Z AND MIN_Z SO REDSHIFTS CAN BE SELECTED
        # ADD MAX_Z AND MIN_Z SO REDSHIFTS CAN BE SELECTED
        # ADD MAX_Z AND MIN_Z SO REDSHIFTS CAN BE SELECTED
        # ADD MAX_Z AND MIN_Z SO REDSHIFTS CAN BE SELECTED
        # ADD MAX_Z AND MIN_Z SO REDSHIFTS CAN BE SELECTED
        # ADD MAX_Z AND MIN_Z SO REDSHIFTS CAN BE SELECTED
        aparams, meanT = self.get_meanT()
        plimits = self.get_param_limits()
        gp = t0_gpemulator.T0MultiBinGP(params=aparams, temps=meanT, param_limits=plimits)
        return gp

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

    def get_parameters(self):
        """Get the list of parameter vectors in this emulator."""
        return self.sample_params

    def get_param_limits(self):
        """Get the reprocessed limits on the parameters for the likelihood."""
        return self.param_limits

    def get_outdir(self, pp, strsz=3):
        """Get the simulation output directory path for a parameter set."""
        return os.path.join(os.path.join(self.basedir, self.build_dirname(pp, strsz=strsz)),"output")

    def build_dirname(self,params, strsz=3):
        """Make a directory name for a given set of parameter values"""
        parts = ['',]*(len(self.param_names))
        # Transform the dictionary into a list of string parts,
        # sorted in the same way as the parameter array.
        fstr = "%."+str(strsz)+"g"
        for nn,val in self.param_names.items():
            parts[val] = nn+fstr % params[val]
        name = ''.join(str(elem) for elem in parts)
        return name
