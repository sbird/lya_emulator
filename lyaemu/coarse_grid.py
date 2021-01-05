"""Generate a coarse grid for the emulator and test it"""
from __future__ import print_function
import os
import os.path
import shutil
import glob
import string
import json
import numpy as np
import h5py
from .SimulationRunner.SimulationRunner import lyasimulation
from . import latin_hypercube
from . import flux_power
from . import lyman_data
from . import gpemulator
from .mean_flux import ConstMeanFlux

def get_latex(key):
    """Get a latex name if it exists, otherwise return the key."""
    #Names for pretty-printing some parameters in Latex
    print_names = { 'ns': r'n_\mathrm{s}', 'As': r'A_\mathrm{s}', 'herei': r'z_\mathrm{He i}', 'heref': r'z_\mathrm{He f}', 'hub':'h', 'tau0':r'\tau_0', 'dtau0':r'd\tau_0'}
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
    def __init__(self, basedir, param_names=None, param_limits=None, kf=None, mf=None, limitfac=1):
        if param_names is None:
            self.param_names = {'ns':0, 'As':1, 'herei':2, 'heref':3, 'alphaq':4, 'hub':5, 'omegamh2':6}
        else:
            self.param_names = param_names
        #Parameters: ns: 0.8 - 0.995. Notice that this is not ns at the CMB scale!
        # As: amplitude of power spectrum at 0.5 h/Mpc scales (note slightly larger than last paper!
        # herei: redshift at which helium reionization starts.
        # 4.0 is default, we use a linear history with 3.5-4.5
        # heref: redshift at which helium reionization finishes. 2.8 is default.
        # Thermal history suggests late, HeII Lyman alpha suggests earlier.
        # hub: hubble constant (also changes omega_M)
        # alphaq: quasar spectral index. 1 - 2.5 Controls IGM temperature.
        # omegam h^2: We fix omega_m h^2 = 0.143+-0.001 (Planck 2018 best-fit) and vary omega_m and h^2 to match it.
        # h^2 itself has little effect on the forest.
        if param_limits is None:
            self.param_limits = np.array([[0.8, 0.995], [1.2e-09, 2.6e-09], [4.0, 4.5], [2.6, 3.2], [1.4, 2.5], [0.65, 0.75], [0.14, 0.146]])
        else:
            self.param_limits = param_limits
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

        self.set_maxk()
        #This is the Planck best-fit value. We do not have to change it because
        #it is a) very well measured and b) mostly degenerate with the mean flux.
        self.omegabh2 = 0.0224
        self.sample_params = []
        self.basedir = os.path.expanduser(basedir)
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
        with open(os.path.join(pdir, "SimulationICs.json"), 'r') as jsin:
            sics = json.load(jsin)
        ev = np.zeros_like(self.param_limits[:,0])
        pn = self.param_names
        ev[pn['heref']] = sics["here_f"]
        ev[pn['herei']] = sics["here_i"]
        ev[pn['alphaq']] = sics["alpha_q"]
        ev[pn['hub']] = sics["hubble"]
        ev[pn['ns']] = sics["ns"]
        wmap = sics["scalar_amp"]
        #Convert pivot of the scalar amplitude from amplitude
        #at k = 0.5 (~13 Mpc) to pivot scale of 0.05
        conv = (0.05/(0.5))**(sics["ns"]-1.)
        ev[pn['As']] = wmap / conv
        return ev

    def reconstruct(self):
        """Reconstruct the parameters of an emulator by loading the parameters of each simulation in turn."""
        dirs = glob.glob(os.path.join(self.basedir, "*"))
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
        #Arrays can't be serialised so convert them back and forth to lists
        self.really_arrays = []
        mf = self.mf
        self.mf = []
        for nn, val in self.__dict__.items():
            if isinstance(val, np.ndarray):
                self.__dict__[nn] = val.tolist()
                self.really_arrays.append(nn)
        with open(fdump, 'w') as jsout:
            json.dump(self.__dict__, jsout)
        self._fromarray()
        self.mf = mf

    def load(self,dumpfile="emulator_params.json"):
        """Load parameters from a textfile."""
        kf = self.kf
        mf = self.mf
        real_basedir = self.basedir
        with open(os.path.join(real_basedir, dumpfile), 'r') as jsin:
            indict = json.load(jsin)
        self.__dict__ = indict
        self._fromarray()
        self.kf = kf
        self.mf = mf
        self.basedir = real_basedir
        self.set_maxk()

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

    def gen_simulations(self, nsamples, npart=256.,box=40,samples=None):
        """Initialise the emulator by generating simulations for various parameters."""
        self.sample_params = self.build_params(nsamples)
        if samples is None:
            samples = self.sample_params
        else:
            self.sample_params = np.vstack([self.sample_params, samples])
        #Generate ICs for each set of parameter inputs
        for ev in samples:
            self._do_ic_generation(ev, npart, box)
        self.dump()

    def _do_ic_generation(self,ev,npart,box):
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
        #at k = 0.5 h/Mpc (~13 Mpc) to pivot scale of 0.05
        ns = ev[pn['ns']]
        om0 = ev[pn['omegamh2']]/hub**2
        omb = self.omegabh2 / hub**2
        wmap = (0.05/0.5)**(ns-1.) * ev[pn['As']]
        ss = lyasimulation.LymanAlphaSim(outdir=outdir, box=box,npart=npart, ns=ns, scalar_amp=wmap, redend=2.2, here_f = href, here_i = hrei, alpha_q = aq, hubble=hub, omega0=om0, omegab=omb,unitary=True, seed=422317)
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

    def _get_fv(self, pp,myspec):
        """Helper function to get a single flux vector."""
        di = self.get_outdir(pp, strsz=3)
        if not os.path.exists(di):
            di = self.get_outdir(pp, strsz=2)
        powerspectra = myspec.get_snapshot_list(base=di)
        return powerspectra

    def get_emulator(self, max_z=4.2):
        """ Build an emulator for the desired k_F and our simulations.
            kf gives the desired k bins in s/km.
            Mean flux rescaling is handled (if mean_flux=True) as follows:
            1. A set of flux power spectra are generated for every one of a list of possible mean flux values.
            2. Each flux power spectrum in the set is rescaled to the same mean flux.
            3.
        """
        gp = self._get_custom_emulator(emuobj=None, max_z=max_z)
        return gp

    def get_flux_vectors(self, max_z=4.2, kfunits="kms"):
        """Get the desired flux vectors and their parameters"""
        pvals = self.get_parameters()
        nparams = np.shape(pvals)[1]
        nsims = np.shape(pvals)[0]
        assert nparams == len(self.param_names)
        myspec = flux_power.MySpectra(max_z=max_z, max_k=self.maxk)
        aparams = pvals
        #Note this gets tau_0 as a linear scale factor from the observed power law
        dpvals = self.mf.get_params()
        nuggets = np.zeros_like(pvals[:,0])
        #Savefile prefix
        mfc = "cc"
        if dpvals is not None:
            #Add a small offset to the mean flux in each simulation to improve support
            nuggets = np.arange(nsims)/nsims * (dpvals[-1] - dpvals[0])/(np.size(dpvals)+1)
            newdp = dpvals[0] + (dpvals-dpvals[0]) / (np.size(dpvals)+1) * np.size(dpvals)
            #Make sure we don't overflow the parameter limits
            assert (newdp[-1] + nuggets[-1] < dpvals[-1]) and (newdp[0] + nuggets[0] >= dpvals[0])
            dpvals = newdp
            aparams = np.array([np.concatenate([dp+nuggets[i],pvals[i]]) for dp in dpvals for i in range(nsims)])
            mfc = "mf"
        try:
            kfmpc, kfkms, flux_vectors = self.load_flux_vectors(aparams, mfc=mfc)
        except (AssertionError, OSError):
            print("Could not load flux vectors, regenerating from disc")
            powers = [self._get_fv(pp, myspec) for pp in pvals]
            mef = lambda pp: self.mf.get_mean_flux(myspec.zout, params=pp)[0]
            if dpvals is not None:
                flux_vectors = np.array([powers[i].get_power_native_binning(mean_fluxes = mef(dp+nuggets[i])) for dp in dpvals for i in range(nsims)])
                #'natively' binned k values in km/s units as a function of redshift
                kfkms = [ps.get_kf_kms() for _ in dpvals for ps in powers]
            else:
                flux_vectors = np.array([powers[i].get_power_native_binning(mean_fluxes = mef(dpvals)) for i in range(nsims)])
                #'natively' binned k values in km/s units as a function of redshift
                kfkms = [ps.get_kf_kms() for ps in powers]
            #Same in all boxes
            kfmpc = powers[0].kf
            assert np.all(np.abs(powers[0].kf/ powers[-1].kf-1) < 1e-6)
            self.save_flux_vectors(aparams, kfmpc, kfkms, flux_vectors, mfc=mfc)
        assert np.shape(flux_vectors)[0] == np.shape(aparams)[0]
        if kfunits == "kms":
            kf = kfkms
        else:
            kf = kfmpc
        return aparams, kf, flux_vectors

    def save_flux_vectors(self, aparams, kfmpc, kfkms, flux_vectors, mfc="mf", savefile="emulator_flux_vectors.hdf5"):
        """Save the flux vectors and parameters to a file, which is the only thing read on reload."""
        save = h5py.File(os.path.join(self.basedir, mfc+"_"+savefile), 'w')
        save.attrs["classname"] = str(self.__class__)
        save["params"] = aparams
        save["flux_vectors"] = flux_vectors
        #Save in both km/s and Mpc/h units.
        save["kfkms"] = kfkms
        save["kfmpc"] = kfmpc
        save.close()

    def load_flux_vectors(self, aparams, mfc="mf", savefile="emulator_flux_vectors.hdf5"):
        """Save the flux vectors and parameters to a file, which is the only thing read on reload."""
        load = h5py.File(os.path.join(self.basedir, mfc+"_"+savefile), 'r')
        inparams = np.array(load["params"])
        flux_vectors = np.array(load["flux_vectors"])
        kfkms = np.array(load["kfkms"])
        kfmpc = np.array(load["kfmpc"])
        name = str(load.attrs["classname"])
        load.close()
        assert name.split(".")[-1] == str(self.__class__).split(".")[-1]
        assert np.shape(inparams) == np.shape(aparams)
        assert np.all(inparams - aparams < 1e-3)
        return kfmpc, kfkms, flux_vectors

    def _get_custom_emulator(self, *, emuobj, max_z=4.2):
        """Helper to allow supporting different emulators."""
        aparams, kf, flux_vectors = self.get_flux_vectors(max_z=max_z, kfunits="mpc")
        plimits = self.get_param_limits(include_dense=True)
        gp = gpemulator.MultiBinGP(params=aparams, kf=kf, powers = flux_vectors, param_limits = plimits, singleGP=emuobj)
        return gp

    def do_loo_cross_validation(self, *, remove=None, max_z=4.2):
        """Do cross-validation by constructing an emulator missing
           a single simulation and checking accuracy.
           The remove parameter chooses which simulation to leave out. If None this is random."""
        aparams, kf, flux_vectors = self.get_flux_vectors(max_z=max_z, kfunits="mpc")
        if remove is None:
            nsims = np.shape(aparams)[0]
            rng = np.random.default_rng()
            remove = rng.integers(0,nsims)
        aparams_rem = np.delete(aparams, remove, axis=0)
        flux_vectors_rem = np.delete(flux_vectors, remove, axis=0)
        plimits = self.get_param_limits(include_dense=True)
        gp = gpemulator.MultiBinGP(params=aparams_rem, kf=kf, powers = flux_vectors_rem, param_limits = plimits)
        flux_predict, _ = gp.predict(aparams[remove, :].reshape(1, -1))
        return kf, flux_vectors[remove,:] / flux_predict - 1


class KnotEmulator(Emulator):
    """Specialise parameter class for an emulator using knots.
    Thermal parameters turned off."""
    def __init__(self, basedir, nknots=4, kf=None, mf=None):
        param_names = {'herei':nknots, 'heref':nknots+1, 'alphaq': nknots+2, 'hub':nknots+3, 'omegamh2':nknots+4}
        #Assign names like AA, BB, etc.
        for i in range(nknots):
            param_names[string.ascii_uppercase[i]*2] = i
        self.nknots = nknots
        param_limits = np.append(np.repeat(np.array([[0.6,1.5]]),nknots,axis=0),[[3.5,4.5],[2.5,3.2],[0.65,0.75],[0.14,0.146]],axis=0)
        super().__init__(basedir=basedir, param_names = param_names, param_limits = param_limits, kf=kf, mf=mf)
        #Linearly spaced knots in k space:
        #these do not quite hit the edges of the forest region, because we want some coverage over them.
        self.knot_pos = np.linspace(0.15, 1.5,nknots)
        #Used for early iterations.
        #self.knot_pos = [0.15,0.475,0.75,1.19]

    def _do_ic_generation(self,ev,npart,box):
        """Do the actual IC generation."""
        outdir = os.path.join(self.basedir, self.build_dirname(ev))
        if os.path.exists(outdir):
            return
        pn = self.param_names
        aq = ev[pn['alpha_q']]
        hei = ev[pn['herei']]
        hef = ev[pn['heref']]
        hub = ev[pn['hub']]
        om0 = ev[pn['omegamh2']]/hub**2
        omb = self.omegabh2 / hub**2
        ss = lyasimulation.LymanAlphaKnotICs(outdir=outdir, box=box,npart=npart, knot_pos = self.knot_pos, knot_val=ev[0:self.nknots],hubble=hub, redend=2.2, here_f = hef, here_i = hei, alpha_q = aq, omega0=om0, omegab=omb, unitary=True, seed=422317)
        try:
            ss.make_simulation()
        except RuntimeError as e:
            print(str(e), " while building: ",outdir)

def get_simulation_parameters_knots(base):
    """Get the parameters of a knot-based simulation from the SimulationICs JSON file."""
    jsin = open(os.path.join(base, "SimulationICs.json"), 'r')
    pp = json.load(jsin)
    knv = pp["knot_val"]
    #This will fail!
    parvec = [0., 1., *knv, pp['herei'], pp['heref'],pp['alphaq'], pp["hubble"]]
    return parvec

def get_simulation_parameters_s8(base, dt0=0, t0=1, pivot=0.05):
    """Get the parameters of a sigma8-ns-based simulation from the SimulationICs JSON file."""
    jsin = open(os.path.join(base, "SimulationICs.json"), 'r')
    pp = json.load(jsin)
    #Change the pivot value
    As = pp['scalar_amp'] / (pivot/0.5)**(pp['ns']-1.)
    parvec = [dt0, t0, pp['ns'], As, pp['herei'], pp['heref'],pp['alphaq'], pp["hubble"]]
    return parvec
