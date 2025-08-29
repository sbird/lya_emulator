"""Functions to use for running chains. Includes separate functions for Flux Power Spectrum only chains, Mean Temperature only chains, and chains using both.

Inputs include:
    - savefile = str, path and filename to save the chain outputs
    - basedir = str, path to the emulator files (json, hdf5), which should include both FPS and T0
    - HRbasedir = str, path to the HF emulator files (if None, use single-fidelity emulator)
    - traindir = str, path to the directory containing the saved GP (***fps only)
    - min_z, max_z = float, range of redshifts (inclusive)
    - loo_errors = bool, whether to use leave-one-out errors
    - burnin = int, number of burnin steps to take
    - nsamples = int, total number of steps to take before exiting (if not converged)
    - pscale = float, scale of initial proposal scale
    - tau_thresh = float, optical depth threshold (***fps only and fps+t0)
    - hprior = str ('planck' or 'shoes'), whether to include a prior on h
    - oprior, bhprior = bool, whether to include a prior on omegam h^2 or bhfeedback
    - dataset = str ('fps', 'wavelet', 'curvature', 'bpdf', or 'combined'), which data set from Gaikwad to use (***t0 only)

Usage notes:
    - Best to run with multiple chains using MPI. To do this, it is easiest to write a script with the desired inputs (above), import and call these functions using those inputs, then run that script using mpiexec or mpirun (-n 4 is a good number). Add the --bind-by core option to mpiexec/mpirun commands, to avoid using all available cores (which slows the run due to the excess overhead).
"""

from . import likelihood as lk
from meanT import t0_likelihood as t0lk


# chain using both flux power spectrum and mean temperature
def run_chain(savefile, basedir, HRbasedir=None, traindir=None, min_z=2.2, max_z=4.6, loo_errors=True, tau_thresh=1e6, burnin=3e4, nsamples=3e5, hprior='none', oprior=False, bhprior=False, pscale=100):
    # get the likelihood class object
    like = lk.LikelihoodClass(basedir, tau_thresh=tau_thresh, traindir=traindir, max_z=max_z, min_z=min_z, optimise_GP=False, HRbasedir=HRbasedir, loo_errors=loo_errors, use_meant=True)
    # then run the chain -- note that savefile = None means it will not save
    chain = like.do_sampling(savefile=savefile, burnin=burnin, nsamples=nsamples, hprior=hprior, oprior=oprior, bhprior=bhprior, pscale=pscale, use_meant=True)
    return chain

# chain using only the mean temperature
def run_t0chain(savefile, basedir, HRbasedir=None, min_z=2.2, max_z=3.8, loo_errors=True, dataset='fps', burnin=3e4, nsamples=3e5, hprior='none', oprior=False, bhprior=False, pscale=4):
    # get the likelihood class object
    like = t0lk.T0LikelihoodClass(basedir, max_z=max_z, min_z=min_z, optimise_GP=False, HRbasedir=HRbasedir, dataset=dataset, loo_errors=loo_errors)
    # then run the chain -- note that savefile = None means it will not save
    chain = like.do_sampling(savefile=savefile, burnin=burnin, nsamples=nsamples, hprior=hprior, oprior=oprior, bhprior=bhprior, dataset=dataset, pscale=pscale)
    return chain

# chain using only the flux power spectrum
def run_fpschain(savefile, basedir, HRbasedir=None, traindir=None, min_z=2.2, max_z=4.6, loo_errors=True, tau_thresh=1e6, burnin=3e4, nsamples=3e5, hprior='none', oprior=False, bhprior=False, pscale=100):
    # get the likelihood class object
    like = lk.LikelihoodClass(basedir, tau_thresh=tau_thresh, traindir=traindir, max_z=max_z, min_z=min_z, optimise_GP=False, HRbasedir=HRbasedir, loo_errors=loo_errors)
    # then run the chain -- note that savefile = None means it will not save
    chain = like.do_sampling(savefile=savefile, burnin=burnin, nsamples=nsamples, hprior=hprior, oprior=oprior, bhprior=bhprior, pscale=pscale)
    return chain
