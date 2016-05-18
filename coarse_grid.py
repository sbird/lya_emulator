"""Generate a coarse grid for the emulator and test it"""
from __future__ import print_function
import os.path
import numpy as np
# import emcee

# import gpemulator
import latin_hypercube
from SimulationRunner import lyasimulation

class Params(object):
    """Small class to store parameter names and limits"""
    def __init__(self):
        self.param_names = ['ns', 'As', 'heat_slope', 'heat_amp', 'hub']
        #Not sure what the limits on heat_slope should be.
        self.param_limits = np.array([[0.6, 1.5], [1.5e-9, 4.0e-9], [0., 0.5],[0.25,3],[0.65,0.75]])

    def build_dirname(self,params):
        """Make a directory name for a given set of parameter values"""
        assert len(params) == len(self.param_names)
        name = ""
        for nn,pp in zip(self.param_names, params):
            name += nn+str(np.round(pp,2))
        return name

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

# def init_lnline(nsamples, data=None):
#     """Initialise the emulator by loading the flux power spectra from the simulations."""
#     #Parameter names
#     params = Params()
#     data = gpemulator.SDSSData()
    #Glob the directories? Or load from a text file?
    #Get unique values
#     flux_vectors = np.array([get_flux_power(pp) for pp in params])
#     gp = gpemulator.SkLearnGP(tau_means = params[:,0], ns = params[:,1], As = params[:,2], kf=data.kf, flux_vectors=flux_vectors)
#     return gp, data

def gen_simulations(nsamples,basedir, npart=256.,box=60,):
    """Initialise the emulator by generating simulations for various parameters."""
    params = Params()
    with open(os.path.join(basedir, "emulator_params.txt"),'w') as saved_params:
        saved_params.write(str(params.param_names))
        toeval = latin_hypercube.get_hypercube_samples(params.param_limits, nsamples)
        #Generate ICs for each set of parameter inputs
        for ev in toeval:
            outdir = os.path.join(basedir, params.build_dirname(ev))
            saved_params.write(str(ev)+"  :  "+outdir)
            assert params.param_names[0] == 'ns'
            assert params.param_names[1] == 'As'
            assert params.param_names[2] == 'heat_slope'
            assert params.param_names[3] == 'heat_amp'
            assert params.param_names[4] == 'hub'
            #Use Planck 2015 cosmology
            ss = lyasimulation.LymanAlphaSim(outdir, box,npart, ns=ev[0], scalar_amp=ev[1],rescale_gamma=True, rescale_slope=ev[2], rescale_amp=ev[3], hubble=ev[4], omegac=0.25681, omegab=0.0483)
            try:
                ss.make_simulation()
            except RuntimeError as e:
                print(str(e), " while building: ",outdir)
    return

