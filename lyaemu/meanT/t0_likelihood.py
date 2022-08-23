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

def load_data(datafile, index, max_z=3.8, min_z=2.0, tau_thresh=None):
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
    def __init__(self, basedir, max_z=3.8, min_z=2.0, optimise_GP=True, json_file='T0emulator_params.json', tau_thresh=None):
        # Needed for Cobaya dictionary construction
        self.basedir, self.json_file = basedir, json_file
        self.tau_thresh = tau_thresh
        self.max_z, self.min_z = max_z, min_z
        myspec = flux_power.MySpectra(max_z=max_z, min_z=min_z)
        self.zout = myspec.zout
        # Load data vector (Wavelet:1,2 Curvature:3,4 BPDF:5,6 FPS:7,8 Combined:9,10)
        meanT_file = os.path.join(os.path.dirname(__file__), '../data/Gaikwad/Gaikwad_2020b_T0_Evolution_All_Statistics.txt')
        ## ADD OPTION TO USE OTHER DATA
        self.obs_z, self.meanT, self.error = np.loadtxt(meanT_file, usecols=(0,7,8))[::-1].T
        if max_z > 3.8:
            boera = np.array([[5.0, 4.6, 4.2], [7.37e3, 7.31e3, 8.31e3], [1530., 1115., 1155.]])
            high_z_gaikwad = np.array([[5.4], [11e3], [1.6e3]])
            self.obs_z = np.append(high_z_gaikwad[0], np.append(boera[0], self.obs_z))
            self.meanT = np.append(high_z_gaikwad[1], np.append(boera[1], self.meanT))
            self.error = np.append(high_z_gaikwad[2], np.append(boera[2], self.error))
        zinds = np.intersect1d(np.round(self.obs_z, 2), np.round(self.zout, 2), return_indices=True)[2][::-1]

        self.emulator = t0_coarse_grid.T0Emulator(basedir, tau_thresh=tau_thresh)
        # Load the parameters, etc. associated with this emulator (overwrite defaults)
        self.emulator.load(dumpfile=json_file)
        self.param_limits = self.emulator.get_param_limits()
        self.ndim = np.shape(self.param_limits)[0]
        assert np.shape(self.param_limits)[1] == 2
        # Generate emulator
        if optimise_GP:
            print('Beginning to generate emulator at', str(datetime.now()))
            self.gpemu = self.emulator.get_emulator(max_z=max_z, min_z=min_z)
            self.gpemu.gps = list(np.array(self.gpemu.gps)[zinds])
            self.gpemu.temps = self.gpemu.temps[:, zinds]
            self.gpemu.nz = len(self.gpemu.gps)
            print('Finished generating emulator at', str(datetime.now()))

    def get_predicted(self, params):
        """Helper function to get the predicted mean temperature and error."""
        predicted, std = self.gpemu.predict(np.array(params).reshape(1, -1))
        return predicted, std

    def likelihood(self, params, data_meanT=None, cosmo_priors=False, include_emu=True):
        """A simple likelihood function for the mean temperature."""
        # Default to use is Gaikwad data
        if data_meanT is None:
            data_meanT, data_error = self.meanT, self.error
        else:
            # if simulation is used as data, assume ~10% 'measurement' error
            data_error = data_meanT*0.085
        # Set parameter limits as the hull of the original emulator.
        if np.any(params >= self.param_limits[:, 1]) or np.any(params <= self.param_limits[:, 0]):
            return -np.inf
        # get predicted and calculate chi^2
        predicted, std = self.get_predicted(params)
        diff = data_meanT - predicted
        error = data_error**2
        if include_emu:
            error = data_error**2 + std**2
        chi2 = -np.sum(diff**2/(2*error) + 0.5*np.log(error))
        if cosmo_priors:
            # add a prior on little h and omega_m h^2
            chi2 += self.hubble_prior(params, low_z=True)
            chi2 += self.omega_prior(params)
        assert 0 > chi2 > -2**31
        assert not np.isnan(chi2)
        return chi2

    def hubble_prior(self, params, low_z=True):
        """Return a prior on little h (either Planck or SH0ES)"""
        hh = self.emulator.param_names['hub']
        if low_z:
            shoes_mean, shoes_sigma = 0.7304, 0.0104 # SH0ES arxiv: 2112.04510
            h_prior = -((params[hh]-shoes_mean)/shoes_sigma)**2
        else:
            planck_mean, planck_sigma = 0.6741, 0.005 # Planck arxiv: 1807.06209
            h_prior = -((params[hh]-planck_mean)/planck_sigma)**2
        return h_prior

    def omega_prior(self, params):
        """Return a prior on Omega_m h^2 (Planck 2018)"""
        oo = self.emulator.param_names['omegamh2']
        planck_mean, planck_sigma = 0.1424, 0.001 # Planck arxiv: 1807.06209
        o_prior = -((params[oo]-planck_mean)/planck_sigma)**2
        return o_prior

    def make_cobaya_dict(self, *, data_meanT, burnin, nsamples, pscale=50):
        """Return a dictionary that can be used to run Cobaya MCMC sampling."""
        # Parameter names
        pnames = self.emulator.print_pnames()
        # Get parameter ranges for use as a rough estimate of proposal pdf width
        prange = (self.param_limits[:, 1]-self.param_limits[:, 0])
        # Build the dictionary
        info = {}
        info["likelihood"] = {__name__+".T0CobayaLikelihoodClass": {"basedir": self.basedir, "max_z": self.max_z, "min_z": self.min_z, "optimise_GP": True, "json_file": self.json_file, "tau_thresh": self.tau_thresh, "data_meanT": data_meanT}}
        # Each of the parameters has a prior with limits and a proposal width (the proposal covariance matrix
        # is learned, so the value given needs to be small enough for the sampler to get started)
        info["params"] = {pnames[i][0]: {'prior': {'min': self.param_limits[i, 0], 'max': self.param_limits[i, 1]}, 'proposal': prange[i]/pscale, 'latex': pnames[i][1]} for i in range(self.ndim)}
        # Set up the mcmc sampler options (to do seed runs, add the option 'seed': integer between 0 and 2**32 - 1)
        info["sampler"] = {"mcmc": {"burn_in": burnin, "max_samples": nsamples, "Rminus1_stop": 0.01, "output_every": '60s', "learn_proposal": True, "learn_proposal_Rminus1_max": 20, "learn_proposal_Rminus1_max_early": 30}}
        return info

    def do_acceptance_check(self, info, steps=100):
        """Run a short chain to check the initial acceptance rate."""
        print("-----------------------------------------------------")
        print("Test run to check acceptance rate")
        info_test = info.copy()
        # Don't do a burn-in, limit the sample to steps, and increase the max_tries to ensure it succeeds
        info_test.update({"sampler": {"mcmc": {"burn_in": 0, "max_samples": steps, "max_tries": '1000d'}}})
        updated_info, sampler = cobaya_run(info_test)
        print('Acceptance rate:', sampler.get_acceptance_rate())
        print("----------------------------------------------------- \n")
        assert sampler.get_acceptance_rate() > 0.01, "Acceptance rate very low. Consider decreasing the proposal width by increasing the pscale parameter"

    def do_sampling(self, savefile=None, datadir=None, index=None, burnin=3e4, nsamples=3e5, pscale=4, test_accept=True):
        """Run MCMC using Cobaya. Cobaya supports MPI, with a separate chain for each process (for HPCC, 4-6 chains recommended).
        burnin and nsamples are per chain. If savefile is None, the chain will not be saved."""
        # If datadir is None, default is to use the flux power data from BOSS (dr14 or dr9)
        data_meanT = None
        if datadir is not None and index is not None:
            # Load the data directory (i.e. use a simulation flux power as data)
            data_meanT = load_data(datadir, index, max_z=self.max_z, min_z=self.min_z, tau_thresh=self.tau_thresh)

        # Construct the "info" dictionary used by Cobaya
        info = self.make_cobaya_dict(data_meanT=data_meanT, pscale=pscale, burnin=burnin, nsamples=nsamples)
        # Test run a fraction of the full chain to check acceptance rate before running full chain
        if test_accept is True:
            self.do_acceptance_check(info, steps=100)
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
    tau_thresh: int = None
    data_meanT: float = None
    # Required for Cobaya to correctly parse which parameters are for input
    input_params_prefix: str = ""

    def initialize(self):
        """Initialization of Cobaya likelihood using LikelihoodClass init.
        Gets the emulator by loading the flux power spectra from the simulations."""
        T0LikelihoodClass.__init__(self, self.basedir, max_z=self.max_z, min_z=self.min_z,
                         optimise_GP=self.optimise_GP, json_file=self.json_file,
                         tau_thresh=self.tau_thresh)

    def logp(self, **params_values):
        """Cobaya-compatible call to the base class likelihood function.
        Must be called logp."""
        # self.input_params is specially recognized by Cobaya (will be the "params" section
        # of the Cobaya dictionary passed to it)
        params = np.array([params_values[p] for p in self.input_params])
        return self.likelihood(params, data_meanT=self.data_meanT)
