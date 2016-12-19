"""Generate a coarse grid for the emulator and test it"""
from __future__ import print_function
import os
import os.path
import string
import math
import json
import numpy as np
import gpemulator
import latin_hypercube
import flux_power
import matter_power
from SimulationRunner import simulationics
from SimulationRunner import lyasimulation

class Emulator(object):
    """Small wrapper class to store parameter names and limits, generate simulations and get an emulator."""
    def __init__(self, basedir, param_names=None, param_limits=None, kf=None):
        if param_names is None:
            self.param_names = {'ns':0, 'As':1, 'heat_slope':2, 'heat_amp':3, 'hub':4}
        else:
            self.param_names = param_names
        if param_limits is None:
            self.param_limits = np.array([[0.6, 1.5], [1.5e-9, 4.0e-9], [0., 0.5],[0.25,2],[0.65,0.75]])
        else:
            self.param_limits = param_limits
        if kf is None:
            self.kf = gpemulator.SDSSData().get_kf()
        else:
            self.kf = kf
        self.dense_param_names = { 'tau0': 0 }
        #Limits on factors to multiply the thermal history by.
        #Mean flux is known to about 10% from SDSS, so we don't need a big range.
        self.dense_param_limits = np.array([[0.8,1.2],])
        self.dense_samples = 5
        self.sample_params = []
        self.basedir = os.path.expanduser(basedir)
        if not os.path.exists(basedir):
            os.mkdir(basedir)

    def build_dirname(self,params):
        """Make a directory name for a given set of parameter values"""
        parts = ['',]*len(self.param_names)
        #Transform the dictionary into a list of string parts,
        #sorted in the same way as the parameter array.
        for nn,val in self.param_names.items():
            parts[val] = nn+'%.2g' % params[val]
        name = ''.join(str(elem) for elem in parts)
        return name

    def _fromarray(self):
        """Convert the data stored as lists back to arrays."""
        for arr in self.really_arrays:
            self.__dict__[arr] = np.array(self.__dict__[arr])
        self.really_arrays = []

    def dump(self, dumpfile="emulator_params.json"):
        """Dump parameters to a textfile."""
        #Arrays can't be serialised so convert them back and forth to lists
        self.really_arrays = []
        for nn, val in self.__dict__.items():
            if isinstance(val, np.ndarray):
                self.__dict__[nn] = val.tolist()
                self.really_arrays.append(nn)
        with open(os.path.join(self.basedir, dumpfile), 'w') as jsout:
            json.dump(self.__dict__, jsout)
        self._fromarray()

    def load(self,dumpfile="emulator_params.json"):
        """Load parameters from a textfile."""
        kf = self.kf
        with open(os.path.join(self.basedir, dumpfile), 'r') as jsin:
            indict = json.load(jsin)
        #Make sure dense parameters are not over-written
        indict['dense_param_limits'] = self.dense_param_limits
        indict['dense_samples'] = self.dense_samples
        self.__dict__ = indict
        self._fromarray()
        self.kf = kf

    def get_outdir(self, pp):
        """Get the simulation output directory path for a parameter set."""
        return os.path.join(os.path.join(self.basedir, self.build_dirname(pp)),"output")

    def get_parameters(self):
        """Get the list of parameter vectors in this emulator."""
        return self.sample_params

    def build_params(self, nsamples,limits = None, use_existing=False):
        """Build a list of directories and parameters from a hyercube sample"""
        if limits is None:
            limits = self.param_limits
        #Consider only prior points inside the limits
        prior_points = None
        if use_existing:
            ii = np.where(np.all(self.sample_params > limits[:,0],axis=1)*np.all(self.sample_params < limits[:,1],axis=1))
            prior_points = self.sample_params[ii]
        return latin_hypercube.get_hypercube_samples(limits, nsamples,prior_points=prior_points)

    def gen_simulations(self, nsamples, npart=256.,box=60,samples=None):
        """Initialise the emulator by generating simulations for various parameters."""
        if samples is not None:
            self.sample_params = np.vstack([self.sample_params, samples])
        if len(self.sample_params) == 0:
            self.sample_params = self.build_params(nsamples)
        self.dump()
        #Generate ICs for each set of parameter inputs
        [self._do_ic_generation(ev, npart, box) for ev in self.sample_params]
        return

    def _do_ic_generation(self,ev,npart,box):
        """Do the actual IC generation."""
        outdir = os.path.join(self.basedir, self.build_dirname(ev))
        pn = self.param_names
        #Use Planck 2015 cosmology
        ca={'rescale_gamma': True, 'rescale_slope': ev[pn['heat_slope']], 'rescale_amp' :ev[pn['heat_amp']]}
        ss = simulationics.SimulationICs(outdir=outdir, box=box,npart=npart, ns=ev[pn['ns']], scalar_amp=ev[pn['As']], code_args = ca, code_class=lyasimulation.LymanAlphaMPSim, hubble=ev[pn['hub']], omegac=0.25681, omegab=0.0483)
        try:
            ss.make_simulation()
        except RuntimeError as e:
            print(str(e), " while building: ",outdir)

    def _add_dense_params(self, pvals):
        """From the matrix representing the 'sparse' (ie, corresponding to an N-body) simulation,
        add extra parameters corresponding to each dense parameter, which corresponds to some modification of the spectrum."""
        #Index of the first 'dense' parameter
        #The interpolator class doesn't distinguish, but the flux power loader needs to.
        dense = np.shape(pvals)[1]
        #Number of dense parameters
        ndense = len(self.dense_param_names)
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

    def get_param_limits(self, include_dense=True):
        """Get the reprocessed limits on the parameters for emcee."""
        if not include_dense:
            return self.param_limits
        comb = np.vstack([self.param_limits, self.dense_param_limits])
        comb[-1,:] = np.exp(-comb[-1,::-1]*flux_power.obs_mean_tau(3.))
        assert np.shape(comb)[1] == 2
        return comb

    def _get_fv(self, pp,myspec, mean_flux):
        """Helper function to get a single flux vector."""
        di = self.get_outdir(pp)
        print(di)
        tau0_factors = None
        if mean_flux:
            ti = self.dense_param_names['tau0']
            tlim = self.dense_param_limits[ti]
            tau0_factors = np.linspace(tlim[0], tlim[1], self.dense_samples)
            pvals_new = np.zeros((self.dense_samples, len(pp)+1))
            pvals_new[:,:len(pp)] = np.tile(pp, (self.dense_samples,1))
            #Use the mean flux at z=3 as the index parameter;
            #best accuracy should be achieved if the derived parameter is linear in the input.
            pvals_new[:,-1] = np.exp(-tau0_factors*flux_power.obs_mean_tau(3.))
        else:
            pvals_new = pp.reshape((1,len(pp)))
        fv = myspec.get_flux_power(di,self.kf, tau0_factors = tau0_factors)
        assert np.shape(fv)[0] == np.shape(pvals_new)[0]
        return pvals_new, fv

    def get_emulator(self, mean_flux=False, max_z=4.2):
        """ Build an emulator for the desired k_F and our simulations.
            kf gives the desired k bins in s/km.
            Mean flux rescaling is handled (if mean_flux=True) as follows:
            1. A set of flux power spectra are generated for every one of a list of possible mean flux values.
            2. Each flux power spectrum in the set is rescaled to the same mean flux.
            3.
        """
        pvals = self.get_parameters()
        nparams = np.shape(pvals)[1]
        assert np.shape(pvals)[1] == len(self.param_names)
        #Try to load the emulator from the savefile if we can.
        try:
            gp = gpemulator.SkLearnGP(params=nparams+mean_flux, flux_vectors=None, kf = self.kf, savedir=self.basedir)
            return gp
        except IOError:
            pass
        myspec = flux_power.MySpectra(max_z=max_z)
        pnew, fluxes = zip(*[self._get_fv(pp, myspec, mean_flux=mean_flux) for pp in pvals])
        pvals = np.array(pnew).reshape(-1,np.shape(pnew[0])[1])
        flux_vectors = np.array(fluxes).reshape(-1,np.shape(fluxes[0])[1])
        #Check shape is ok.
        nsamples = np.shape(self.get_parameters())[0]*np.max([1,mean_flux*self.dense_samples])
        assert np.shape(flux_vectors) == (nsamples, np.size(myspec.zout)*np.size(self.kf))
        gp = gpemulator.SkLearnGP(params=pvals, kf=self.kf, flux_vectors=flux_vectors, savedir=self.basedir)
        #Check we reproduce the input
        test,_ = gp.predict(pvals[0,:].reshape(1,-1))
        assert np.max(np.abs(test[0] / flux_vectors[0,:]-1)) < 1e-6
        return gp

class KnotEmulator(Emulator):
    """Specialise parameter class for an emulator using knots.
    Thermal parameters turned off."""
    def __init__(self, basedir, nknots=4, use_h0=True):
        param_names = {}
        if use_h0:
            param_names['hub'] = nknots
        #Assign names like AA, BB, etc.
        for i in range(nknots):
            param_names[string.ascii_uppercase[i]*2] = i
        self.nknots = nknots
        param_limits = np.append(np.repeat(np.array([[0.6,1.5]]),nknots,axis=0),[[0.65,0.75]],axis=0)
        super().__init__(basedir=basedir, param_names = param_names, param_limits = param_limits)
        #Linearly spaced knots in k space:
        #these do not quite hit the edges of the forest region, because we want some coverage over them.
        self.knot_pos = np.linspace(0.15, 1.5,nknots)
        #Used for early iterations.
        #self.knot_pos = [0.15,0.475,0.75,1.19]

    def _do_ic_generation(self,ev,npart,box):
        """Do the actual IC generation."""
        outdir = os.path.join(self.basedir, self.build_dirname(ev))
        try:
            hub = ev[self.param_names['hub']]
        except KeyError:
            #If not in emulator.
            hub = 0.69
        #Use Planck 2015 cosmology
        ss = lyasimulation.LymanAlphaKnotICs(outdir=outdir, box=box,npart=npart, knot_pos = self.knot_pos, knot_val=ev[0:self.nknots],hubble=hub, code_class=lyasimulation.LymanAlphaMPSim, omegac=0.25681, omegab=0.0483)
        try:
            ss.make_simulation()
        except RuntimeError as e:
            print(str(e), " while building: ",outdir)

class MatterPowerEmulator(Emulator):
    """Build an emulator based on the matter power spectrum instead of the flux power spectrum, for testing."""
    def load(self,dumpfile="emulator_params.json"):
        """Load parameters from a textfile. Reset the k values to something sensible for matter power."""
        super().load(dumpfile=dumpfile)
        self.kf = np.logspace(np.log10(3*math.pi/60.),np.log10(2*math.pi/60.*256),20)

    def _get_fv(self, pp,myspec, mean_flux):
        """Helper function to get a single matter power vector."""
        di = self.get_outdir(pp)
        (_,_) = myspec, mean_flux
        fv = matter_power.get_matter_power(di,kk=self.kf, redshift = 3.)
        return fv
