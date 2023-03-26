"""Module for computing the likelihood function for the IGM mean temperature emulator."""
from datetime import datetime
import numpy as np
from . import t0_coarse_grid
from .. import flux_power
from cobaya.likelihood import Likelihood
from cobaya.run import run as cobaya_run
from cobaya.log import LoggedError
from mpi4py import MPI
import json
import h5py
import os

def load_data(datafile, index, max_z=3.8, min_z=2.0):
    """Load "fake data" mean temperatures"""
    zout = flux_power.MySpectra(max_z=max_z, min_z=min_z).zout
    meanT_file = h5py.File(datafile, 'r')
    # find indices in meanT_file for relevant redshifts
    zintersect = np.intersect1d(np.round(zout, 2), np.round(meanT_file['zout'][:], 2), return_indices=True)
    # from high to low redshift
    data_meanT =meanT_file['meanT'][index][zintersect[2]][::-1]
    return data_meanT

class T0LikelihoodClass:
    """Class to contain likelihood and MCMC sampling computations."""
    def __init__(self, basedir, max_z=3.8, min_z=2.2, optimise_GP=True, json_file='T0emulator_params.json', dataset='fps', HRbasedir=None, loo_errors=False):
        """Parameters:
        - basedir: directory to load emulator
        - min_z, max_z: minimum and maximum redshift to include in the likelihood
        - optimise_GP: whether to train the GPs (if sampling, set initial call to false to avoid training twice)
        - json_file: json file containing various emulator settings and parameters
        - dataset: which dataset from Gaikwad 2021, arXiv 2009.00016. See data_dict below for options.
        - HRbasedir: directory to load high resolution emulator, turns this into a multi-fidelity emulator, with predictions for the HR outputs.
        - loo_errors: whether to use leave-one-out errors in place of emulator errors. File with errors must be in same directory as emulator files, and be called 'loo_t0.hdf5'
        """
        # Needed for Cobaya dictionary construction
        self.basedir, self.json_file = basedir, json_file
        self.HRbasedir, self.loo_errors = HRbasedir, loo_errors
        self.max_z, self.min_z = max_z, min_z
        myspec = flux_power.MySpectra(max_z=max_z, min_z=min_z)
        self.zout = myspec.zout
        # Load data vector (wavelet, curvature, bpdf, fps, or combined)
        data_dict = {'wavelet': (0,1,2), 'curvature': (0,3,4), 'bpdf': (0,5,6), 'fps': (0,7,8), 'combined': (0,9,10)}
        meanT_file = os.path.join(os.path.dirname(__file__), '../data/Gaikwad/Gaikwad_2020b_T0_Evolution_All_Statistics.txt')
        self.obs_z, self.meanT, self.error = np.loadtxt(meanT_file, usecols=data_dict[dataset])[::-1].T
        zinds = np.intersect1d(np.round(self.obs_z, 2), np.round(self.zout, 2), return_indices=True)[2][::-1]
        self.zout = self.zout[zinds]
        self.obs_z, self.meanT, self.error = self.obs_z[zinds], self.meanT[zinds], self.error[zinds]

        if loo_errors: self.get_loo_errors()
        self.emulator = t0_coarse_grid.T0Emulator(basedir, max_z=max_z, min_z=min_z)
        # Load the parameters, etc. associated with this emulator (overwrite defaults)
        self.emulator.load(dumpfile=json_file)
        self.param_limits = self.emulator.get_param_limits()
        self.ndim = np.shape(self.param_limits)[0]
        assert np.shape(self.param_limits)[1] == 2

        # Generate emulator
        comm = MPI.COMM_WORLD
        rank = comm.Get_rank()
        if optimise_GP:
            gpemu = None
            if rank == 0:
                #Build the emulator only on rank 0 and broadcast
                print('Beginning to generate emulator at', str(datetime.now()))
                if HRbasedir is not None:
                    gpemu = self.emulator.get_MFemulator(HRbasedir=HRbasedir, max_z=max_z, min_z=min_z, zinds=zinds)
                else:
                    gpemu = self.emulator.get_emulator(max_z=max_z, min_z=min_z, zinds=zinds)
                print('Finished generating emulator at', str(datetime.now()))
            self.gpemu = comm.bcast(gpemu, root = 0)

    def get_loo_errors(self, savefile="loo_t0.hdf5"):
        if self.HRbasedir is not None:
            filepath = os.path.join(self.HRbasedir, savefile)
        else:
            filepath = os.path.join(self.basedir, savefile)
        ff = h5py.File(filepath, 'r')
        tpp, tpt, looz = ff['meanT_predict'][:], ff['meanT_true'][:], ff['zout'][:]
        ff.close()
        zinds = np.where([(looz <= self.max_z)*(looz >= self.min_z)])[1]
        # after loading the absolute difference, calculate errors including BOSS data
        self.loo = np.mean(np.abs(tpp - tpt)[:, zinds], axis=0)
        return self.loo

    def calculate_loo_errors(self, savefile='loo_t0.hdf5'):
        """Calculate leave-one-out errors: saves predicted temperature, true temperature,
        predicted error, and parameters for each simulation in the training set."""
        # call to loo function in coarse_grid with appropriate settings
        if self.HRbasedir is not None:
            HRemu = t0_coarse_grid.T0Emulator(self.HRbasedir, min_z=self.min_z, max_z=self.max_z)
            predict, std, true, params = self.emulator.generate_loo_errors(HRemu=HRemu, min_z=self.min_z, max_z=self.max_z)
            savepath = self.HRbasedir
        else:
            predict, std, true, params = self.emulator.generate_loo_errors(min_z=self.min_z, max_z=self.max_z)
            savepath = self.basedir
        # save into hdf5 file
        savefile = h5py.File(os.path.join(savepath, savefile), 'w')
        savefile['meanT_predict'] = predict
        savefile['std_predict'] = std
        savefile['meanT_true'] = true
        savefile['params'] = params
        savefile['zout'] = self.zout
        savefile.close()

    def get_predicted(self, params):
        """Helper function to get the predicted mean temperature and error."""
        predicted, std = self.gpemu.predict(np.array(params).reshape(1, -1))
        return predicted, std

    def likelihood(self, params, data_meanT=None, include_emu=True, hprior='none', oprior=False, bhprior=False):
        """A simple likelihood function for the mean temperature."""
        # Default to use is Gaikwad data
        if data_meanT is None:
            data_meanT, data_error = self.meanT, self.error
        else:
            # if simulation is used as data, assume ~8.5% 'measurement' error
            data_error = data_meanT*0.085
        # Set parameter limits as the hull of the original emulator.
        if np.any(params >= self.param_limits[:, 1]) or np.any(params <= self.param_limits[:, 0]):
            return -np.inf
        # get predicted and calculate chi^2
        predicted, std = self.get_predicted(params)
        diff = data_meanT - predicted
        error = data_error**2
        if include_emu:
            if self.loo_errors:
                error = data_error**2 + self.loo**2
            else:
                error = data_error**2 + std**2
        chi2 = -np.sum(diff**2/(2*error) + 0.5*np.log(error))
        chi2 += self.hubble_prior(params, source=hprior)
        if oprior: chi2 += self.omega_prior(params)
        if bhprior: chi2 += self.bhfeedback_prior(params)
        assert 0 > chi2 > -2**31
        assert not np.isnan(chi2)
        return chi2

    def hubble_prior(self, params, source='none'):
        """Return a prior on little h (either Planck or SH0ES)"""
        if source == 'none': return 0
        hh = self.emulator.param_names['hub']
        if source == 'shoes':
            shoes_mean, shoes_sigma = 0.7304, 0.0104 # SH0ES arxiv: 2112.04510
            return -((params[hh]-shoes_mean)/shoes_sigma)**2
        if source == 'planck':
            planck_mean, planck_sigma = 0.6741, 0.005 # Planck arxiv: 1807.06209
            return -((params[hh]-planck_mean)/planck_sigma)**2
        else: return 0

    def omega_prior(self, params):
        """Return a prior on Omega_m h^2 (Planck 2018)"""
        # values from Planck: arxiv 1807.06209
        oo = self.emulator.param_names['omegamh2']
        planck_mean, planck_sigma = 0.1424, 0.001
        return -((params[oo]-planck_mean)/planck_sigma)**2

    def bhfeedback_prior(self, params):
        """Return a prior on black hole feedback (marginalize out)"""
        # value range is [0.03, 0.07]
        bh = self.emulator.param_names['bhfeedback']
        if self.mf_slope:
            bh = bh + 2
        bh_mean, bh_sigma = 0.05, 0.01
        return -((params[bh]-bh_mean)/bh_sigma)**2

    def make_cobaya_dict(self, *, data_meanT, burnin, nsamples, dataset, hprior='none', oprior=False, bhprior=False, pscale=50):
        """Return a dictionary that can be used to run Cobaya MCMC sampling."""
        # Parameter names
        pnames = self.emulator.print_pnames()
        # Get parameter ranges for use as a rough estimate of proposal pdf width
        prange = (self.param_limits[:, 1]-self.param_limits[:, 0])
        # Build the dictionary
        info = {}
        info["likelihood"] = {__name__+".T0CobayaLikelihoodClass": {"basedir": self.basedir, "max_z": self.max_z, "min_z": self.min_z, "optimise_GP": True, "json_file": self.json_file, "HRbasedir": self.HRbasedir, "loo_errors": self.loo_errors, "hprior": hprior, "oprior": oprior, "bhprior": bhprior, "data_meanT": data_meanT, "dataset": dataset}}
        # Each of the parameters has a prior with limits and a proposal width (the proposal covariance matrix
        # is learned, so the value given needs to be small enough for the sampler to get started)
        info["params"] = {pnames[i][0]: {'prior': {'min': self.param_limits[i, 0], 'max': self.param_limits[i, 1]}, 'proposal': prange[i]/pscale, 'latex': pnames[i][1]} for i in range(self.ndim)}
        # Set up the mcmc sampler options (to do seed runs, add the option 'seed': integer between 0 and 2**32 - 1)
        info["sampler"] = {"mcmc": {"burn_in": burnin, "max_samples": nsamples, "Rminus1_stop": 0.01, "output_every": '60s', "learn_proposal": True, "learn_proposal_Rminus1_max": 20, "learn_proposal_Rminus1_max_early": 30}}
        return info

    def do_sampling(self, savefile=None, datadir=None, index=None, burnin=3e4, nsamples=3e5, pscale=4, hprior='none', oprior=False, bhprior=False, dataset='fps'):
        """Run MCMC using Cobaya. Cobaya supports MPI, with a separate chain for each process (for HPCC, 4-6 chains recommended).
        burnin and nsamples are per chain. If savefile is None, the chain will not be saved."""
        # If datadir is None, default is to use the flux power data from BOSS (dr14 or dr9)
        data_meanT = None
        if datadir is not None and index is not None:
            # Load the data directory (i.e. use a simulation flux power as data)
            data_meanT = load_data(datadir, index, max_z=self.max_z, min_z=self.min_z)

        # Construct the "info" dictionary used by Cobaya
        info = self.make_cobaya_dict(data_meanT=data_meanT, pscale=pscale, burnin=burnin, nsamples=nsamples, hprior=hprior, oprior=oprior, bhprior=bhprior, dataset=dataset)

        if savefile is not None:
            info["output"] = savefile

        # Set up MPI protections (as suggested in Cobaya documentation)
        comm = MPI.COMM_WORLD
        rank = comm.Get_rank()
        success = False
        try:
            # Run the sampler, Cobaya MCMC -- resume will only work if a savefile is given (so it can load previous chain)
            updated_info, sampler = cobaya_run(info, resume=True)
            success = True
        except LoggedError as err:
            pass
        success = all(comm.allgather(success))
        if not success and rank == 0:
            print("Sampling failed!")
        else:
            all_chains = comm.gather(sampler.products()["sample"], root=0)
            return sampler, all_chains


class T0CobayaLikelihoodClass(Likelihood, T0LikelihoodClass):
    """Class inheriting Cobaya functionality. Strictly for use with Cobaya sampling."""
    # Cobaya automatically recognizes (and sets as default) inputs with the following names
    basedir: str
    max_z: float = 3.8
    min_z: float = 2.0
    optimise_GP: bool = True
    json_file: str = 'T0emulator_params.json'
    dataset: str = 'fps'
    HRbasedir: str = None
    data_meanT: float = None
    loo_errors: bool = False
    hprior: str = 'none'
    oprior: bool = False
    bhprior: bool = False
    # Required for Cobaya to correctly parse which parameters are for input
    input_params_prefix: str = ""

    def initialize(self):
        """Initialization of Cobaya likelihood using LikelihoodClass init.
        Gets the emulator by loading the flux power spectra from the simulations."""
        T0LikelihoodClass.__init__(self, self.basedir, max_z=self.max_z, min_z=self.min_z,
                         optimise_GP=self.optimise_GP, json_file=self.json_file,
                         dataset=self.dataset, HRbasedir=self.HRbasedir, loo_errors=self.loo_errors)

    def logp(self, **params_values):
        """Cobaya-compatible call to the base class likelihood function.
        Must be called logp."""
        # self.input_params is specially recognized by Cobaya (will be the "params" section
        # of the Cobaya dictionary passed to it)
        params = np.array([params_values[p] for p in self.input_params])
        return self.likelihood(params, data_meanT=self.data_meanT, hprior=self.hprior, oprior=self.oprior, bhprior=self.bhprior)
