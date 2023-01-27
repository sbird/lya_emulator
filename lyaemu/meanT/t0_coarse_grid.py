"""Generate a coarse grid for constructing a mean temperature emulator"""
import os
import numpy as np
import h5py
import json
from .. import flux_power
from . import t0_gpemulator
from .. import tempdens

def get_latex(key):
    """Get a latex name if it exists, otherwise return the key."""
    #Names for pretty-printing some parameters in Latex
    print_names = { 'ns': r'n_\mathrm{s}', 'Ap': r'A_\mathrm{P}', 'herei': r'z_\mathrm{He i}', 'heref': r'z_\mathrm{He f}', 'hub':'h', 'tau0':r'\tau_0', 'dtau0':r'd\tau_0', 'alphaq':r'\alpha_q', 'omegamh2':r'\Omega_M h^2', 'hireionz':'z_{Hi}', 'bhfeedback':'\epsilon_{AGN}'}
    try:
        return print_names[key]
    except KeyError:
        return key

class T0Emulator:
    """Stores parameter names and limits, generates T0 file from particle snapshots, and gets an emulator.
    Parameters:
        - basedir: directory to load or create emulator
        - param_names: dictionary containing names of the parameters as well as a unique integer list of positions
        - param_limits: Nx2 array containing upper and lower limits of each parameter, in the order given by the integer stored in param_names
        - npart, box: particle number and box size of emulator simulations.
    """
    def __init__(self, basedir, param_names=None, param_limits=None, npart=512, box=60, fullphysics=True, max_z=5.4, min_z=2.0):
        if param_names is None:
            self.param_names = {'ns':0, 'Ap':1, 'herei':2, 'heref':3, 'alphaq':4, 'hub':5, 'omegamh2':6, 'hireionz':7, 'bhfeedback':8}
        else:
            self.param_names = param_names
        if param_limits is None:
            # see parameter descriptions in coarse_grid.py
            self.param_limits = np.array([[0.8, 0.995], [1.2e-09, 2.6e-09], [3.5, 4.1], [2.6, 3.2], [1.3, 2.5], [0.65, 0.75], [0.14, 0.146], [6.5,8], [0.03, 0.07]])
        else:
            self.param_limits = param_limits
        # Remove the BH parameter for not full physics.
        self.fullphysics = fullphysics
        if not fullphysics:
            bhind = self.param_names.pop('bhfeedback')
            self.param_limits = np.delete(self.param_limits, bhind, 0)
        self.npart, self.box = npart, box
        self.omegabh2 = 0.0224
        self.max_z, self.min_z = max_z, min_z
        self.sample_params = []
        self.basedir = os.path.expanduser(basedir)

    def load(self, dumpfile="T0emulator_params.json"):
        """Load parameters from a json file."""
        real_basedir = self.basedir
        min_z = self.min_z
        max_z = self.max_z
        with open(os.path.join(real_basedir, dumpfile), 'r') as jsin:
            indict = json.load(jsin)
        self.__dict__ = indict
        self._fromarray()
        self.basedir = real_basedir
        self.min_z = min_z
        self.max_z = max_z
        self.myspec = flux_power.MySpectra(max_z=self.max_z, min_z=self.min_z)

    def _fromarray(self):
        """Convert the data stored as lists back to arrays."""
        for arr in self.really_arrays:
            self.__dict__[arr] = np.array(self.__dict__[arr])
        self.really_arrays = []

    def get_emulator(self, max_z=5.4, min_z=2.0, zinds=None):
        """ Build an emulator for T0 from simulations."""
        if zinds is not None: req_z = self.myspec.zout[zinds]
        else: req_z = None
        aparams, meanT = self.get_meanT(max_z=max_z, min_z=min_z, req_z=req_z)
        plimits = self.get_param_limits()
        gp = t0_gpemulator.T0MultiBinGP(params=aparams, temps=meanT, param_limits=plimits)
        return gp

    def get_MFemulator(self, HRbasedir, max_z=5.4, min_z=2.2, zinds=None):
        """ Build a multi-fidelity emulator for T0 from simulations."""
        if zinds is not None: req_z = self.myspec.zout[zinds]
        else: req_z = None
        # get lower resolution parameters & temperatures
        self.load()
        LRparams, LRmeanT = self.get_meanT(max_z=max_z, min_z=min_z, req_z=req_z)
        # get higher resolution parameters & temperatures
        HRemu = T0Emulator(HRbasedir, max_z=max_z, min_z=min_z)
        HRemu.load()
        HRparams, HRmeanT = HRemu.get_meanT(max_z=max_z, min_z=min_z, req_z=req_z)
        # check parameter limits and get/train the multi-fidelity GP
        assert np.all(self.get_param_limits() == HRemu.get_param_limits())
        gp = t0_gpemulator.T0MultiBinAR1(LRparams=LRparams, HRparams=HRparams, LRtemps=LRmeanT, HRtemps=HRmeanT, param_limits=self.get_param_limits())
        return gp

    def get_meanT(self, filename="emulator_meanT.hdf5", max_z=5.4, min_z=2.0, req_z=None):
        """Get and save T0 and parameters"""
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
        # get meanT for specified redshift range (z goes from high to low)
        if req_z is not None:
            zinds = np.intersect1d(np.round(req_z, 2), np.round(self.myspec.zout, 2), return_indices=True)[2][::-1]
            meanT = meanT[:, zinds]
            self.myspec.zout = self.myspec.zout[zinds]
        else:
            assert np.round(self.myspec.zout[-1], 1) <= min_z
            maxbin = np.where(np.round(self.myspec.zout, 1) >= min_z)[0].max() + 1
            assert np.round(self.myspec.zout[0], 1) >= max_z
            minbin = np.where(np.round(self.myspec.zout, 1) <= max_z)[0].min()
            meanT = meanT[:, minbin:maxbin]
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
        # note, if json and hdf5 parameter orders differ, this will sort by json params
        subset = np.isin(aparams, inparams).all(axis=1)
        new_inds = np.where(subset == False)[0] # indices of aparams that are not in inparams
        for pp in range(np.shape(inparams)[0]): # fill meanT with already computed values
            ii = np.where(np.all(aparams == inparams[pp], axis=1) == True)[0]
            meanT[ii] = old_meanT[pp]
        return new_inds, meanT

    def save_meanT(self, aparams, meanT, savefile="emulator_meanT.hdf5"):
        """Save the mean temperatures and parameters to a file."""
        save = h5py.File(os.path.join(self.basedir, savefile), 'w')
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
        load.close()
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
        fstr = "%."+str(strsz)+"g"
        for nn,val in self.param_names.items():
            parts[val] = nn+fstr % params[val]
        name = ''.join(str(elem) for elem in parts)
        return name

    def print_pnames(self):
        """Get parameter names for printing"""
        n_latex = []
        sort_names = sorted(list(self.param_names.items()), key=lambda k:(k[1],k[0]))
        for key, _ in sort_names:
            n_latex.append((key, get_latex(key)))
        return n_latex

    def generate_loo_errors(self, HRemu=None, min_z=2.2, max_z=3.8):
        """Calculate leave-one-out errors for all training simulations.
        HRemu should be a separate instance of the emulator class."""
        self.load()
        nsims, nparams = np.shape(self.sample_params)
        nz = flux_power.MySpectra(max_z=max_z, min_z=min_z).zout.size
        if HRemu is not None:
            HRemu.load()
            nsims, nparams = np.shape(HRemu.sample_params)
        predict, std, true = np.zeros([nsims, nz]), np.zeros([nsims, nz]), np.zeros([nsims, nz])
        params = np.zeros([nsims, nparams])
        for i in range(nsims):
            predict[i], std[i], true[i], params[i] = self.single_loo(i, HRemu=HRemu, min_z=min_z, max_z=max_z)
        return predict, std, true, params

    def single_loo(self, remove, HRemu=None, min_z=2.2, max_z=3.8):
        """Calculate the leave-one-out errors for the training sample at index 'remove'
        HRemu should be a separate instance of the emulator class."""
        # get the full set of parameters and temperatures (LF & HF if using multi-fidelity)
        aparams, meanT = self.get_meanT(min_z=min_z, max_z=max_z)
        plimits = self.get_param_limits()
        if HRemu is not None:
            HRparams, HRmeanT = HRemu.get_meanT(max_z=max_z, min_z=min_z)
            # remove indicated index from training set, then train and predict removed output
            aparams_train = np.delete(HRparams, remove, axis=0)
            meanT_train = np.delete(HRmeanT, remove, axis=0)
            gp = t0_gpemulator.T0MultiBinAR1(LRparams=aparams, HRparams=aparams_train, LRtemps=meanT, HRtemps=meanT_train, param_limits=plimits)
            meant_predict, std_predict = gp.predict(HRparams[remove].reshape(1, -1))
            # return predicted output, true output, prediction error, and parameters for predicted
            return meant_predict, std_predict, HRmeanT[remove], HRparams[remove]
        else:
            # or do the same, but for the LF only
            aparams_train = np.delete(aparams, remove, axis=0)
            meanT_train = np.delete(meanT, remove, axis=0)
            gp = t0_gpemulator.T0MultiBinGP(params=aparams_train, temps=meanT_train, param_limits=plimits)
            meant_predict, std_predict = gp.predict(aparams[remove].reshape(1, -1))
            return meant_predict, std_predict, meanT[remove], aparams[remove]
