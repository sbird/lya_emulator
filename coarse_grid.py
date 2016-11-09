"""Generate a coarse grid for the emulator and test it"""
from __future__ import print_function
import os
import os.path
import json
import numpy as np
import matplotlib
import gpemulator
import latin_hypercube
import flux_power
from SimulationRunner import simulationics
from SimulationRunner import lyasimulation
matplotlib.use('PDF')
import matplotlib.pyplot as plt

class Params(object):
    """Small class to store parameter names and limits"""
    def __init__(self, basedir, param_names=None, param_limits=None):
        if param_names is None:
            self.param_names = ['ns', 'As', 'heat_slope', 'heat_amp', 'hub']
        else:
            self.param_names = param_names
        if param_limits is None:
            self.param_limits = np.array([[0.6, 1.5], [1.5e-9, 4.0e-9], [0., 0.5],[0.25,2],[0.65,0.75]])
        else:
            self.param_limits = param_limits
        self.dense_param_names = ['tau0',]
        #Limits on factors to multiply the thermal history by.
        self.dense_param_limits = np.array([[0.1,0.8],])
        self.dense_samples = 5
        self.sample_params = []
        self.basedir = basedir
        if not os.path.exists(basedir):
            os.mkdir(basedir)

    def build_dirname(self,params):
        """Make a directory name for a given set of parameter values"""
        assert len(params) == len(self.param_names)
        name = ""
        for nn,pp in zip(self.param_names, params):
            name += nn+'%.2g' % pp
        return name

    def dump(self, dumpfile="emulator_params.json"):
        """Dump parameters to a textfile."""
        #Arrays can't be serialised so convert them back and forth to lists
        self.param_limits = self.param_limits.tolist()
        self.sample_params = self.sample_params.tolist()
        with open(os.path.join(self.basedir, dumpfile), 'w') as jsout:
            json.dump(self.__dict__, jsout)
        self.param_limits = np.array(self.param_limits)
        self.sample_params = np.array(self.sample_params)

    def load(self,dumpfile="emulator_params.json"):
        """Load parameters from a textfile."""
        #No need to store the dense parameters
        dpns = (self.dense_param_names, self.dense_param_limits, self.dense_samples)
        with open(os.path.join(self.basedir, dumpfile), 'r') as jsin:
            self.__dict__ = json.load(jsin)
        self.param_limits = np.array(self.param_limits)
        self.sample_params = np.array(self.sample_params)
        (self.dense_param_names, self.dense_param_limits, self.dense_samples) = dpns

    def get_dirs(self):
        """Get the list of directories in this emulator."""
        return [self._get_path(dd) for dd in self.sample_dirs]

    def _get_path(self, dirname):
        """Convert the directory name for a simulation into a disc path."""
        return os.path.join(os.path.join(self.basedir, dirname),"output")

    def get_parameters(self):
        """Get the list of parameter vectors in this emulator."""
        return self.sample_params

    def build_params(self, nsamples):
        """Build a list of directories and parameters from a hyercube sample"""
        self.sample_params = latin_hypercube.get_hypercube_samples(self.param_limits, nsamples)
        self.sample_dirs = [self.build_dirname(ev) for ev in self.sample_params]

    def gen_simulations(self, nsamples, npart=256.,box=60,):
        """Initialise the emulator by generating simulations for various parameters."""
        if len(self.sample_params) != nsamples:
            self.build_params(nsamples)
        self.dump()
        #Generate ICs for each set of parameter inputs
        for ev,edir in zip(self.sample_params, self.sample_dirs):
            outdir = os.path.join(self.basedir, edir)
            assert self.param_names[0] == 'ns'
            assert self.param_names[1] == 'As'
            assert self.param_names[2] == 'heat_slope'
            assert self.param_names[3] == 'heat_amp'
            assert self.param_names[4] == 'hub'
            #Use Planck 2015 cosmology
            ss = simulationics.SimulationICs(outdir=outdir, box=box,npart=npart, ns=ev[0], scalar_amp=ev[1], code_args={'rescale_gamma': True, 'rescale_slope': ev[2], 'rescale_amp' :ev[3]}, code_class=lyasimulation.LymanAlphaSim, hubble=ev[4], omegac=0.25681, omegab=0.0483)
            try:
                ss.make_simulation()
            except RuntimeError as e:
                print(str(e), " while building: ",outdir)
        return

    def _add_dense_params(self, pvals):
        """From the matrix representing the 'sparse' (ie, corresponding to an N-body) simulation,
        add extra parameters corresponding to each dense parameter, which corresponds to some modification of the spectrum."""
        #Index of the first 'dense' parameter
        #The interpolator class doesn't distinguish, but the flux power loader needs to.
        dense = np.shape(pvals)[1]
        #Number of dense parameters
        ndense = len(self.dense_param_names)
        #Each emulated power spectrum is enforced to have the same mean flux.
        #First value of dense_param_vals should be the mean flux.
        assert self.dense_param_names[0] == 'tau0'
        #This grid will hold the expanded grid of parameters: dense parameters are on the end.
        #Initially make it NaN as a poisoning technique.
        pvals_new = np.nan*np.zeros((np.shape(pvals)[0]*self.dense_samples, np.shape(pvals)[1]+ndense))
        pvals_new[:,:dense] = np.tile(pvals,(self.dense_samples,1))
        for dd in range(dense, dense+ndense):
            #Build grid of mean fluxes
            dlim = self.dense_param_limits[dd-dense]
            #This is not right for ndense > 1.
            dense = np.repeat(np.linspace(dlim[0], dlim[1], self.dense_samples),np.shape(pvals)[0])
            pvals_new[:,dd] = dense
        assert not np.any(np.isnan(pvals_new))
        return pvals_new

    def get_emulator(self, kf, mean_flux=False):
        """ Build an emulator for the desired k_F and our simulations.
            kf gives the desired k bins in s/km.
            Mean flux rescaling is handled (if mean_flux=True) as follows:
            1. A set of flux power spectra are generated for every one of a list of possible mean flux values.
            2. Each flux power spectrum in the set is rescaled to the same mean flux.
            3.
        """
        myspec = flux_power.MySpectra()
        pvals = self.get_parameters()
        dense = len(self.param_names)
        assert np.shape(pvals)[1] == dense
        if mean_flux:
            pvals = self._add_dense_params(pvals)
        flux_vectors = np.array([myspec.get_flux_power(self._get_path(self.build_dirname(pp[:dense])),kf, dense_params = pp[dense:], flat=True) for pp in pvals])
        assert np.shape(flux_vectors) == (np.size(self.get_dirs())*self.dense_samples, np.size(myspec.zout)*np.size(kf))
        gp = gpemulator.SkLearnGP(params=pvals, kf=kf, flux_vectors=flux_vectors)
        return gp

class KnotParams(Params):
    """Specialise parameter class for an emulator using knots.
    Thermal parameters turned off."""
    def __init__(self, basedir):
        param_names = ['AA', 'BB', 'CC', 'DD', 'hub']
        param_limits = np.append(np.repeat(np.array([[0.6,1.5]]),4,axis=0),[[0.65,0.75]],axis=0)
        super().__init__(basedir=basedir, param_names = param_names, param_limits = param_limits)

    def gen_simulations(self, nsamples, npart=256.,box=60,):
        """Initialise the emulator by generating simulations for various parameters."""
        if len(self.sample_params) != nsamples:
            self.build_params(nsamples)
        self.dump()
        #Generate ICs for each set of parameter inputs
        for ev,edir in zip(self.sample_params, self.sample_dirs):
            outdir = os.path.join(self.basedir, edir)
            #Use Planck 2015 cosmology
            ss = lyasimulation.LymanAlphaKnotICs(outdir=outdir, box=box,npart=npart, knot_val=ev[0:4],hubble=ev[4], omegac=0.25681, omegab=0.0483)
            try:
                ss.make_simulation()
            except RuntimeError as e:
                print(str(e), " while building: ",outdir)
        return

def lnlike_linear(params, *, gp=None, data=None):
    """A simple emcee likelihood function for the Lyman-alpha forest."""
    assert gp is not None
    assert data is not None
    predicted,cov = gp.predict(params)
    diff = predicted-data.pf
    return -np.dot(diff,np.dot(data.invcovar + np.identity(np.size(diff))/cov,diff))/2.0

def init_lnlike(basedir, data=None):
    """Initialise the emulator by loading the flux power spectra from the simulations."""
    #Parameter names
    params = Params(basedir)
    params.load()
    data = gpemulator.SDSSData()
    gp = params.get_emulator(data.get_kf())
    return gp, data

def plot_test_interpolate(emulatordir,testdir):
    """Make a plot showing the interpolation error."""
    params = Params(emulatordir)
    params.load()
    data = gpemulator.SDSSData()
    gp = params.get_emulator(data.get_kf(), mean_flux=True)
    params_test = Params(testdir)
    params_test.load()
    myspec = flux_power.MySpectra()
    #Constant mean flux.
    mf = 0.3
    for pp,dd,nn in zip(params_test.get_parameters(),params_test.get_dirs(), params_test.sample_dirs):
        pp = np.append(pp, mf)
        predicted,_ = gp.predict(pp)
        exact = myspec.get_flux_power(dd,data.get_kf(),mean_flux_desired=mf,flat=True)
        ratio = predicted[0]/exact
        nred = len(myspec.zout)
        nk = len(data.get_kf())
        assert np.shape(ratio) == (nred*nk,)
        for i in range(nred):
            plt.loglog(data.get_kf(),ratio[i*nk:(i+1)*nk],label=myspec.zout[i])
        plt.xlabel(r"$k_F$ (s/km)")
        plt.ylabel(r"Predicted/Exact")
        plt.title(nn)
        plt.legend(loc=0)
        plt.show()
        plt.savefig(nn+"mf0.3.pdf")
        print(nn+".pdf")
        plt.clf()
    return gp
