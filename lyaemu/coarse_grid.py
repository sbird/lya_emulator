"""Generate a coarse grid for the emulator and test it"""
from __future__ import print_function
import os
import os.path
import shutil
import glob
import string
import json
import math
import numpy as np
import h5py
from .SimulationRunner.SimulationRunner import galaxysimulation
from .SimulationRunner.SimulationRunner import lyasimulation
from . import latin_hypercube
from . import flux_power
from . import lyman_data
from . import gpemulator
from .mean_flux import ConstMeanFlux

def get_latex(key):
    """Get a latex name if it exists, otherwise return the key."""
    #Names for pretty-printing some parameters in Latex
    print_names = { 'ns': r'n_\mathrm{P}', 'Ap': r'A_\mathrm{P}', 'herei': r'z_\mathrm{He i}', 'heref': r'z_\mathrm{He f}', 'hub':'h', 'tau0':r'\tau_0', 'dtau0':r'd\tau_0', 'alphaq':r'\alpha_q', 'omegamh2':r'\Omega_M h^2', 'hireionz':'z_{Hi}', 'bhfeedback':'\epsilon_{AGN}'}
    try:
        return print_names[key]
    except KeyError:
        return key

class Emulator:
    """Small wrapper class to store parameter names and limits, generate simulations and get an emulator.
    Parameters:
    - basedir: directory to load or create emulator
    - param_names: dictionary containing names of the parameters as well as a unique integer list of positions
    - param_limits: Nx2 array containing upper and lower limits of each parameter, in the order given
                    by the integer stored in param_names
    - kf: k bins to use when getting spectra
    - mf: mean flux object, which takes mean flux parameters and outputs the mean flux in each redshift bin
    - limitfac: factor to uniformly grow the parameter limits by.
    """
    def __init__(self, basedir, param_names=None, param_limits=None, kf=None, mf=None, limitfac=1, tau_thresh=None, npart=512, box=60, fullphysics=True):
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
                                          [1.3, 2.5], # alphaq: quasar spectral index. 1 - 2.5 Controls IGM temperature.
                                          [0.65, 0.75], # hub: hubble constant (also changes omega_M)
                                          [0.14, 0.146],# omegam h^2: We fix omega_m h^2 = 0.143+-0.001 (Planck 2018 best-fit) and vary omega_m and h^2 to match it.
                                                        # h^2 itself has little effect on the forest.
                                          [6.5,8],   #Mid-point of HI reionization
                                          [0.03, 0.07],  # BH feedback parameter
                                       #   [3.2, 4.2] # Wind speed
                                ])
        else:
            self.param_limits = param_limits

        #Remove the BH parameter for not full physics.
        self.fullphysics = fullphysics
        if not fullphysics:
            bhind = self.param_names.pop('bhfeedback')
            self.param_limits = np.delete(self.param_limits, bhind, 0)
        if limitfac != 1:
            param_cent = (self.param_limits[:,0] + self.param_limits[:,1])/2.
            param_width = (- self.param_limits[:,0] + self.param_limits[:,1])/2.
            self.param_limits[:,0] = param_cent - param_width * limitfac
            self.param_limits[:,1] = param_cent + param_width * limitfac
        if kf is None:
            self.kf = lyman_data.BOSSData().get_kf()
        else:
            self.kf = kf
        if mf is None:
            self.mf = ConstMeanFlux(None)
        else:
            self.mf = mf

        self.npart = npart
        self.box = box
        self.set_maxk()
        #This is the Planck best-fit value. We do not have to change it because
        #it is a) very well measured and b) mostly degenerate with the mean flux.
        self.omegabh2 = 0.0224
        self.max_z = 5.4
        self.min_z = 2.0
        self.sample_params = []
        self.basedir = os.path.expanduser(basedir)
        self.tau_thresh = tau_thresh
        if not os.path.exists(basedir):
            os.mkdir(basedir)

    def set_maxk(self):
        """Get the maximum k in Mpc/h that we will need."""
        #Maximal velfactor: the h dependence cancels but there is an omegam
        minhub = 0.65 #self.param_limits[self.param_names['hub'],0]
        omgah2 = 0.146 # self.param_limits[self.param_names['omegamh2'],1]
        velfac = lambda a: a * 100.0* np.sqrt(omgah2/minhub**2/a**3 + (1 - omgah2/minhub))
        #Maximum k value to use in comoving Mpc/h.
        #Comes out to k ~ 5, which is a bit larger than strictly necessary.
        self.maxk = np.max(self.kf) * velfac(1/(1+4.4)) * 2

    def build_dirname(self,params, include_dense=False, strsz=3):
        """Make a directory name for a given set of parameter values"""
        ndense = include_dense * len(self.mf.dense_param_names)
        parts = ['',]*(len(self.param_names) + ndense)
        #Transform the dictionary into a list of string parts,
        #sorted in the same way as the parameter array.
        fstr = "%."+str(strsz)+"g"
        for nn,val in self.mf.dense_param_names.items():
            parts[val] = nn+fstr % params[val]
        for nn,val in self.param_names.items():
            parts[ndense+val] = nn+fstr % params[ndense+val]
        name = ''.join(str(elem) for elem in parts)
        return name

    def print_pnames(self):
        """Get parameter names for printing"""
        n_latex = []
        sort_names = sorted(list(self.mf.dense_param_names.items()), key=lambda k:(k[1],k[0]))
        for key, _ in sort_names:
            n_latex.append((key, get_latex(key)))
        sort_names = sorted(list(self.param_names.items()), key=lambda k:(k[1],k[0]))
        for key, _ in sort_names:
            n_latex.append((key, get_latex(key)))
        return n_latex

    def _fromarray(self):
        """Convert the data stored as lists back to arrays."""
        for arr in self.really_arrays:
            self.__dict__[arr] = np.array(self.__dict__[arr])
        self.really_arrays = []

    def _recon_one(self, pdir):
        """Get the parameters of a simulation from the SimulationICs.json file"""
        with open(os.path.join(pdir, "SimulationICs.json"), 'r', encoding='UTF-8') as jsin:
            sics = json.load(jsin)
        ev = np.zeros_like(self.param_limits[:,0])
        pn = self.param_names
        ev[pn['heref']] = sics["here_f"]
        ev[pn['herei']] = sics["here_i"]
        ev[pn['alphaq']] = sics["alpha_q"]
        ev[pn['hub']] = sics["hubble"]
        ev[pn['ns']] = sics["ns"]
        ev[pn['omegamh2']] = sics["omega0"]*sics["hubble"]**2
        ev[pn['hireionz']] = sics["hireionz"]
        if self.fullphysics:
            ev[pn['bhfeedback']] = sics["bhfeedback"]
        assert abs(sics["redend"] - self.min_z) < 0.01
        wmap = sics["scalar_amp"]
        #Convert pivot of the scalar amplitude from amplitude
        #at 8 Mpc (k = 0.78) to pivot scale of 0.05
        conv = (0.05/(2*math.pi/8.))**(sics["ns"]-1.)
        ev[pn['Ap']] = wmap / conv
        return ev

    def reconstruct(self):
        """Reconstruct the parameters of an emulator by loading the parameters of each simulation in turn."""
        dirs = glob.glob(os.path.join(self.basedir, "*/"))
        self.sample_params = np.array([self._recon_one(pdir) for pdir in dirs])
        assert np.shape(self.sample_params) == (len(dirs), np.size(self.param_limits[:,0]))

    def dump(self, dumpfile="emulator_params.json"):
        """Dump parameters to a textfile."""
        #Backup existing parameter file
        fdump = os.path.join(self.basedir, dumpfile)
        if os.path.exists(fdump):
            backup = fdump + ".backup"
            r=1
            while os.path.exists(backup):
                backup = fdump + "_r"+str(r)+".backup"
                r+=1
            shutil.move(fdump, backup)
        try:
            myspec = self.myspec
            self.myspec = None
        except AttributeError:
            pass
        #Arrays can't be serialised so convert them back and forth to lists
        self.really_arrays = []
        mf = self.mf
        self.mf = []
        for nn, val in self.__dict__.items():
            if isinstance(val, np.ndarray):
                self.__dict__[nn] = val.tolist()
                self.really_arrays.append(nn)
        with open(fdump, 'w', encoding='UTF-8') as jsout:
            json.dump(self.__dict__, jsout)
        self._fromarray()
        self.mf = mf
        try:
            self.myspec = myspec
        except NameError:
            pass

    def load(self,dumpfile="emulator_params.json"):
        """Load parameters from a textfile."""
        kf = self.kf
        mf = self.mf
        tau_thresh = self.tau_thresh
        real_basedir = self.basedir
        with open(os.path.join(real_basedir, dumpfile), 'r', encoding='UTF-8') as jsin:
            indict = json.load(jsin)
        self.__dict__ = indict
        self._fromarray()
        self.kf = kf
        self.mf = mf
        self.tau_thresh = tau_thresh
        self.basedir = real_basedir
        self.set_maxk()
        self.myspec = flux_power.MySpectra(max_z=self.max_z, min_z=self.min_z, max_k=self.maxk)

    def get_outdir(self, pp, strsz=3):
        """Get the simulation output directory path for a parameter set."""
        return os.path.join(os.path.join(self.basedir, self.build_dirname(pp, strsz=strsz)),"output")

    def get_parameters(self):
        """Get the list of parameter vectors in this emulator."""
        return self.sample_params

    def build_params(self, nsamples,limits = None):
        """Build a list of directories and parameters from a hypercube sample"""
        if limits is None:
            limits = self.param_limits
        #Consider only prior points inside the limits
        prior_points = None
        if np.size(self.sample_params) != 0:
            ii = np.where(np.all(self.sample_params > limits[:,0],axis=1)*np.all(self.sample_params < limits[:,1],axis=1))
            prior_points = self.sample_params[ii]
        return latin_hypercube.get_hypercube_samples(limits, nsamples,prior_points=prior_points)

    def gen_simulations(self, nsamples,samples=None):
        """Initialise the emulator by generating simulations for various parameters.
        Box size is in Mpc/h, not Mpc so that the bins of the flux power spectrum in s/km are at fixed values
        no matter the cosmology."""
        self.sample_params = self.build_params(nsamples)
        if samples is None:
            samples = self.sample_params
        else:
            self.sample_params = np.vstack([self.sample_params, samples])
        #Generate ICs for each set of parameter inputs
        for ev in samples:
            self.do_ic_generation(ev)
        self.dump()

    def do_ic_generation(self,ev):
        """Do the actual IC generation."""
        outdir = os.path.join(self.basedir, self.build_dirname(ev))
        if os.path.exists(outdir):
            return
        pn = self.param_names
        href = ev[pn['heref']]
        hrei = ev[pn['herei']]
        aq = ev[pn['alphaq']]
        hub = ev[pn['hub']]
        #Convert pivot of the scalar amplitude from amplitude
        #at 8 Mpc (k = 0.78) to pivot scale of 0.05
        ns = ev[pn['ns']]
        hireionz = ev[pn['hireionz']]
        om0 = ev[pn['omegamh2']]/hub**2
        omb = self.omegabh2 / hub**2
        wmap = (0.05/(2*math.pi/8.))**(ns-1.) * ev[pn['Ap']]
        if self.fullphysics:
            bhfeedback = ev[pn['bhfeedback']]
            ss = galaxysimulation.GalaxySim(outdir=outdir, box=self.box, npart=self.npart, ns=ns, scalar_amp=wmap, redend=2.0,
                                         here_f = href, here_i = hrei, alpha_q = aq, hubble=hub, omega0=om0, omegab=omb,
                                         hireionz = hireionz, bhfeedback = bhfeedback,
                                         unitary=True, seed=422317, timelimit=6)
        else:
            ss = lyasimulation.LymanAlphaSim(outdir=outdir, box=self.box, npart=self.npart, ns=ns, scalar_amp=wmap, redend=2.0,
                                         here_f = href, here_i = hrei, alpha_q = aq, hubble=hub, omega0=om0, omegab=omb,
                                         hireionz = hireionz, unitary=True, seed=422317, timelimit=6)
        try:
            ss.make_simulation()
            fpfile = os.path.join(os.path.dirname(__file__),"flux_power.py")
            shutil.copy(fpfile, os.path.join(outdir, "flux_power.py"))
            ss._cluster.generate_spectra_submit(outdir)
        except RuntimeError as e:
            print(str(e), " while building: ",outdir)

    def get_param_limits(self, include_dense=True):
        """Get the reprocessed limits on the parameters for the likelihood."""
        if not include_dense:
            return self.param_limits
        dlim = self.mf.get_limits()
        if dlim is not None:
            #Dense parameters go first as they are 'slow'
            plimits = np.vstack([dlim, self.param_limits])
            assert np.shape(plimits)[1] == 2
            return plimits
        return self.param_limits

    def get_nsample_params(self):
        """Get the number of sparse parameters, those sampled by simulations."""
        return np.shape(self.param_limits)[0]

    def _get_fv(self, pp):
        """Helper function to get a single flux vector."""
        di = self.get_outdir(pp, strsz=3)
        if not os.path.exists(di):
            di = self.get_outdir(pp, strsz=2)
        powerspectra = self.myspec.get_snapshot_list(base=di)
        return powerspectra

    def get_emulator(self, max_z=4.2, min_z=2.0, traindir=None, savefile="emulator_flux_vectors.hdf5"):
        """ Build an emulator for the desired k_F and our simulations.
            kf gives the desired k bins in s/km.
            Mean flux rescaling is handled (if mean_flux=True) as follows:
            1. A set of flux power spectra are generated for every one of a list of possible mean flux values.
            2. Each flux power spectrum in the set is rescaled to the same mean flux.
            3.
        """
        aparams, kf, flux_vectors = self.get_flux_vectors(max_z=max_z, min_z=min_z, kfunits="mpc", savefile=savefile)
        plimits = self.get_param_limits(include_dense=True)
        nz = int(flux_vectors.shape[1]/kf.size)
        gp = gpemulator.MultiBinGP(params=aparams, kf=kf, powers=flux_vectors, param_limits=plimits, zout=np.linspace(max_z, min_z, nz), traindir=traindir)
        return gp

    def get_MFemulator(self, HRbasedir, max_z=4.6, min_z=2.2, traindir=None):
        """Build a multi-fidelity emulator for the flux power spectrum."""
        # get lower resolution parameters & temperatures
        self.load()
        LRparams, kf, LRfps = self.get_flux_vectors(max_z=max_z, min_z=min_z, kfunits="mpc")
        nz = int(LRfps.shape[1]/kf.size)
        # get higher resolution parameters & temperatures
        HRemu = Emulator(HRbasedir, mf=self.mf, kf=self.kf, tau_thresh=self.tau_thresh)
        HRemu.load()
        HRparams, HRkf, HRfps = HRemu.get_flux_vectors(max_z=max_z, min_z=min_z, kfunits="mpc")
        # check parameter limits, k-bins, number of redshifts, and get/train the multi-fidelity GP
        assert np.all(self.get_param_limits(include_dense=True) == HRemu.get_param_limits(include_dense=True))
        assert np.all(kf - HRkf < 1e-3)
        assert nz == int(HRfps.shape[1]/HRkf.size)
        gp = gpemulator.MultiBinGP(params=LRparams, HRdat=[HRparams, HRfps], powers=LRfps, param_limits=self.get_param_limits(include_dense=True), kf=kf, zout=np.linspace(max_z, min_z, nz), traindir=traindir)
        return gp

    def get_flux_vectors(self, max_z=4.2, min_z=2.0, kfunits="kms", savefile="emulator_flux_vectors.hdf5"):
        """Get the desired flux vectors and their parameters"""
        pvals = self.get_parameters()
        nparams = np.shape(pvals)[1]
        nsims = np.shape(pvals)[0]
        assert nparams == len(self.param_names)
        aparams = pvals
        #Note this gets tau_0 as a linear scale factor from the observed power law
        dpvals = self.mf.get_params()
        #Savefile prefix
        mfc = "cc"
        if dpvals is not None:
            newdp = dpvals[0] + (dpvals-dpvals[0]) / (np.size(dpvals)+1) * np.size(dpvals)
            #Make sure we don't overflow the parameter limits
            dpvals = newdp
            aparams = np.array([np.concatenate([dp, pvals[i]]) for dp in dpvals for i in range(nsims)])
            mfc = "mf"
        try:
            kfmpc, kfkms, flux_vectors = self.load_flux_vectors(aparams, mfc=mfc)
        except (AssertionError, OSError) as err:
            print(f"Unexpected err={err}, type(err)={type(err)}", flush=True)
            print("Could not load flux vectors, regenerating from disc, save to " + savefile, flush=True)
            powers = [self._get_fv(pp) for pp in pvals]
            mef = lambda pp: self.mf.get_mean_flux(self.myspec.zout, params=pp)[0]
            if dpvals is not None:
                flux_vectors = np.array([powers[i].get_power_native_binning(mean_fluxes=mef(dp), tau_thresh=self.tau_thresh) for dp in dpvals for i in range(nsims)])
                #'natively' binned k values in km/s units as a function of redshift
                kfkms = [ps.get_kf_kms() for _ in dpvals for ps in powers]
            else:
                flux_vectors = np.array([powers[i].get_power_native_binning(mean_fluxes=mef(dpvals), tau_thresh=self.tau_thresh) for i in range(nsims)])
                #'natively' binned k values in km/s units as a function of redshift
                kfkms = [ps.get_kf_kms() for ps in powers]
            #Same in all boxes
            kfmpc = powers[0].kf
            assert np.all(np.abs(powers[0].kf/ powers[-1].kf-1) < 1e-6)
            self.save_flux_vectors(aparams, kfmpc, kfkms, flux_vectors, mfc=mfc, savefile=savefile)
        assert np.shape(flux_vectors)[0] == np.shape(aparams)[0]
        if kfunits == "kms":
            kf = kfkms
        else:
            kf = kfmpc
        #Cut out redshifts that we don't want this time
        assert np.round(self.myspec.zout[-1], 1) <= min_z
        maxbin = np.where(np.round(self.myspec.zout, 1) >= min_z)[0].max() + 1
        assert np.round(self.myspec.zout[0], 1) >= max_z
        minbin = np.where(np.round(self.myspec.zout, 1) <= max_z)[0].min()
        kflen = np.shape(kf)[-1]
        newflux = flux_vectors[:, minbin*kflen:maxbin*kflen]
        return aparams, kf, newflux

    def save_flux_vectors(self, aparams, kfmpc, kfkms, flux_vectors, mfc="mf", savefile="emulator_flux_vectors.hdf5"):
        """Save the flux vectors and parameters to a file, which is the only thing read on reload."""
        if self.tau_thresh is not None:
            savefile = savefile[:-5]+'_tau'+str(int(self.tau_thresh))+savefile[-5:]
        save = h5py.File(os.path.join(self.basedir, mfc+"_"+savefile), 'w')
        save.attrs["classname"] = str(self.__class__)
        save["params"] = aparams
        save["zout"] = self.myspec.zout
        save["flux_vectors"] = flux_vectors
        #Save in both km/s and Mpc/h units.
        save["kfkms"] = kfkms
        save["kfmpc"] = kfmpc
        save.close()

    def load_flux_vectors(self, aparams, mfc="mf", savefile="emulator_flux_vectors.hdf5"):
        """Save the flux vectors and parameters to a file, which is the only thing read on reload."""
        if self.tau_thresh is not None:
            savefile = savefile[:-5]+'_tau'+str(int(self.tau_thresh))+savefile[-5:]
        finalpath = os.path.join(self.basedir, mfc+"_"+savefile)
        print("Loading flux powers from: ",finalpath)
        load = h5py.File(finalpath, 'r')
        inparams = np.array(load["params"])
        flux_vectors = np.array(load["flux_vectors"])
        kfkms = np.array(load["kfkms"])
        kfmpc = np.array(load["kfmpc"])
        zout = np.array(load["zout"])
        self.myspec.zout = zout
        name = str(load.attrs["classname"])
        load.close()
        assert name.rsplit(".", maxsplit=1)[-1] == str(self.__class__).rsplit(".", maxsplit=1)[-1]
        assert np.shape(inparams) == np.shape(aparams)
        assert np.all(inparams - aparams < 1e-3)
        return kfmpc, kfkms, flux_vectors

    def generate_loo_errors(self, HRemu=None, min_z=2.2, max_z=4.6):
        """Calculate leave-one-out errors for all training simulations.
        HRemu should be a separate instance of the emulator class."""
        self.load()
        aparams, _, _ = self.get_flux_vectors(min_z=min_z, max_z=max_z, kfunits="mpc")
        nsims, nparams = np.shape(aparams)
        nz = flux_power.MySpectra(max_z=max_z, min_z=min_z).zout.size
        if HRemu is not None:
            HRemu.load()
            aparams, _, _ = HRemu.get_flux_vectors(min_z=min_z, max_z=max_z, kfunits="mpc")
            nsims, nparams = np.shape(aparams)
        predict, std, true = np.zeros([nsims, nz, self.kf.size]), np.zeros([nsims, nz, self.kf.size]), np.zeros([nsims, nz, self.kf.size])
        params = np.zeros([nsims, nparams])
        if len(self.mf.dense_param_names) != 0:
            for i in range(nsims//10):
                print('Iteration Number', str(i+1)+'/'+str(nsims//10))
                predict[i*10:(i+1)*10], std[i*10:(i+1)*10], true[i*10:(i+1)*10], params[i*10:(i+1)*10] = self.single_loo(i, HRemu=HRemu, min_z=min_z, max_z=max_z)
        else:
            for i in range(nsims):
                print('Iteration Number', str(i+1)+'/'+str(nsims))
                predict[i], std[i], true[i], params[i] = self.single_loo(i, HRemu=HRemu, min_z=min_z, max_z=max_z)
        return predict, std, true, params

    def single_loo(self, remove, HRemu=None, min_z=2.2, max_z=4.6):
        """Calculate the leave-one-out errors for the training sample at index 'remove'
        HRemu should be a separate instance of the emulator class."""
        # get the full set of parameters and flux power spectra (LF & HF if using multi-fidelity)
        aparams, kf, flux_vectors = self.get_flux_vectors(min_z=min_z, max_z=max_z, kfunits="mpc")
        zout = np.linspace(max_z, min_z, int(flux_vectors.shape[1]/kf.size))
        plimits = self.get_param_limits(include_dense=True)
        ndense = len(self.mf.dense_param_names)
        hindex = ndense + self.param_names["hub"]
        omegamh2_index = ndense + self.param_names["omegamh2"]
        if HRemu is not None:
            HRparams, HRkf, HRflux = HRemu.get_flux_vectors(min_z=min_z, max_z=max_z, kfunits="mpc")
            # if mean flux rescaling, remove all training samples associated with the simulation
            if ndense != 0:
                remove = np.arange(remove, np.shape(HRparams)[0], np.shape(HRparams)[0]//10)
            # remove indicated index from training set, then train and predict removed output
            aparams_train = np.delete(HRparams, remove, axis=0)
            flux_train = np.delete(HRflux, remove, axis=0)
            gp = gpemulator.MultiBinGP(params=aparams, HRdat=[aparams_train, flux_train], powers=flux_vectors, param_limits=plimits, kf=kf, zout=zout)
            predict = np.zeros([remove.size, zout.size, self.kf.size])
            std = np.zeros([remove.size, zout.size, self.kf.size])
            true = np.zeros([remove.size, zout.size, self.kf.size])
            for i in range(remove.size):
                fp, sp = gp.predict(HRparams[remove[i]].reshape(1, -1))
                omega_m = HRparams[remove[i], omegamh2_index]/HRparams[remove[i], hindex]**2
                _, predict[i] = flux_power.rebin_power_to_kms(kfkms=self.kf, kfmpc=HRkf, flux_powers=fp[0], zbins=zout, omega_m=omega_m)
                _, std[i] = flux_power.rebin_power_to_kms(kfkms=self.kf, kfmpc=HRkf, flux_powers=sp[0], zbins=zout, omega_m=omega_m)
                _, true[i] = flux_power.rebin_power_to_kms(kfkms=self.kf, kfmpc=HRkf, flux_powers=HRflux[remove[i]], zbins=zout, omega_m=omega_m)
            # return predicted output, true output, prediction error, and parameters for predicted
            return predict, std, true, HRparams[remove]
        else:
            # or do the same, but for the LF only
            if ndense != 0:
                remove = np.arange(remove, np.shape(aparams)[0], np.shape(aparams)[0]//10)
            aparams_train = np.delete(aparams, remove, axis=0)
            flux_train = np.delete(flux_vectors, remove, axis=0)
            gp = gpemulator.MultiBinGP(params=aparams_train, kf=kf, powers=flux_train, param_limits=plimits, zout=zout)
            predict = np.zeros([remove.size, zout.size, self.kf.size])
            std = np.zeros([remove.size, zout.size, self.kf.size])
            true = np.zeros([remove.size, zout.size, self.kf.size])
            for i in range(remove.size):
                fp, sp = gp.predict(aparams[remove[i]].reshape(1, -1))
                omega_m = aparams[remove[i], omegamh2_index]/aparams[remove[i], hindex]**2
                _, predict[i] = flux_power.rebin_power_to_kms(kfkms=self.kf, kfmpc=kf, flux_powers=fp[0], zbins=zout, omega_m=omega_m)
                _, std[i] = flux_power.rebin_power_to_kms(kfkms=self.kf, kfmpc=kf, flux_powers=sp[0], zbins=zout, omega_m=omega_m)
                _, true[i] = flux_power.rebin_power_to_kms(kfkms=self.kf, kfmpc=kf, flux_powers=flux_vectors[remove[i]], zbins=zout, omega_m=omega_m)
            return predict, std, true, aparams[remove]

