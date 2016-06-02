"""Generate a coarse grid for the emulator and test it"""
from __future__ import print_function
import os
import os.path
import json
import numpy as np
# import emcee

import gpemulator
import latin_hypercube
import flux_power
from SimulationRunner import lyasimulation,clusters

class Params(object):
    """Small class to store parameter names and limits"""
    def __init__(self, basedir):
        self.param_names = ['ns', 'As', 'heat_slope', 'heat_amp', 'hub']
        #Not sure what the limits on heat_slope should be.
        self.param_limits = np.array([[0.6, 1.5], [1.5e-9, 4.0e-9], [0., 0.5],[0.25,3],[0.65,0.75]])
        self.sample_params = []
        self.basedir = basedir
        if not os.path.exists(basedir):
            os.mkdir(basedir)

    def build_dirname(self,params):
        """Make a directory name for a given set of parameter values"""
        assert len(params) == len(self.param_names)
        name = ""
        for nn,pp in zip(self.param_names, params):
            name += nn+'%.1e' % pp
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
        with open(os.path.join(self.basedir, dumpfile), 'r') as jsin:
            self.__dict__ = json.load(jsin)
        self.param_limits = np.array(self.param_limits)
        self.sample_params = np.array(self.sample_params)

    def get_dirs(self):
        """Get the list of directories in this emulator."""
        return [os.path.join(os.path.join(self.basedir, dd),"output") for dd in self.sample_dirs]

    def get_parameters(self):
        """Get the list of parameter vectors in this emulator."""
        return self.sample_params

    def build_params(self, nsamples):
        """Build a list of directories and parameters from a hyercube sample"""
        self.sample_params = latin_hypercube.get_hypercube_samples(self.param_limits, nsamples)
        self.sample_dirs = [self.build_dirname(ev) for ev in self.sample_params]

    def gen_simulations(self, nsamples, npart=256.,box=60,):
        """Initialise the emulator by generating simulations for various parameters."""
        LymanAlphaSim = clusters.hypatia_mpi_decorate(lyasimulation.LymanAlphaSim)
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
            ss = LymanAlphaSim(outdir, box,npart, ns=ev[0], scalar_amp=ev[1],rescale_gamma=True, rescale_slope=ev[2], rescale_amp=ev[3], hubble=ev[4], omegac=0.25681, omegab=0.0483)
            try:
                ss.make_simulation()
            except RuntimeError as e:
                print(str(e), " while building: ",outdir)
        return

    def get_emulator(self):
        """Build an emulator from the desired parameter values and simulations"""
        myspec = flux_power.MySpectra()
        flux_vectors = np.array([myspec.get_flux_power(pp) for pp in self.get_dirs()])
        pvals = self.get_parameters()
        gp = gpemulator.SkLearnGP(tau_means = pvals[:,0], ns = pvals[:,1], As = pvals[:,2], kf=data.kf, flux_vectors=flux_vectors)
        return gp

def lnlike_linear(params, *, gp=None, data=None):
    """A simple emcee likelihood function for the Lyman-alpha forest using the
       simple linear model with only cosmological parameters.
       This neglects many important properties!"""
    assert gp is not None
    assert data is not None
    #TODO: Add logic to rescale derived power to the desired mean flux.
    predicted,cov = gp.predict(params)
    diff = predicted-data.pf
    return -np.dot(diff,np.dot(data.invcovar + np.identity(np.size(diff))/cov,diff))/2.0

def init_lnlike(basedir, data=None):
    """Initialise the emulator by loading the flux power spectra from the simulations."""
    #Parameter names
    params = Params(basedir)
    params.load()
    gp = params.get_emulator()
    data = gpemulator.SDSSData()
    return gp, data

